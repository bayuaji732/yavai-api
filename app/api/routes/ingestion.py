import os
import regex
import tempfile
import pandas as pd
from datetime import datetime
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from app.core.spark_config import create_spark_session
from app.services.spark_service import SparkService
from app.core import config

router = APIRouter()
spark_service = SparkService()

@router.post("/csv/import")
async def import_csv(
    csv_file: UploadFile = File(...),
    app_name: str = Form(...),
    table_name: str = Form(...),
    delimiter: str = Form(",")
):
    if not table_name:
        raise HTTPException(status_code=400, detail="Table name is required")
    
    try:
        # Clean table name
        table_name = table_name.lower()
        table_name = regex.sub(r'\s+', '_', table_name)
        table_name = regex.sub(r'[^A-Za-z0-9_]', '', table_name)
        
        spark_session = create_spark_session(app_name)
        
        # Save CSV to temp location
        temp_csv_path = os.path.join(tempfile.gettempdir(), 'temp_csv_file.csv')
        with open(temp_csv_path, 'wb') as f:
            content = await csv_file.read()
            f.write(content)
        
        # Read and process CSV
        df_pandas = pd.read_csv(temp_csv_path, sep=delimiter, dtype=str)
        df_spark = spark_service.pandas_to_spark(spark_session, df_pandas)
        
        # Clean column names
        for column_name in df_spark.columns:
            new_column_name = regex.sub(r'\s+', '_', column_name)
            new_column_name = regex.sub(r'[^A-Za-z0-9_]', '', new_column_name)
            new_column_name = new_column_name.lower()
            df_spark = df_spark.withColumnRenamed(column_name, new_column_name)
        
        df_spark = spark_service.add_timestamp_column(df_spark)
        
        # Save as external table
        hdfs_path = f"{config.HDFS_NAME_NODE}{config.HIVE_DIR_PATH}/{table_name}"
        df_spark.write.mode("overwrite").format('orc').option("path", hdfs_path).saveAsTable(table_name)
        
        spark_session.stop()
        
        return {
            "statusCode": 200,
            "status": "OK",
            "message": f"The external table {table_name} has been created in Hive"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/csv/append")
async def append_csv(
    csv_file: UploadFile = File(...),
    app_name: str = Form(...),
    table_name: str = Form(...),
    delimiter: str = Form(",")
):
    if not table_name:
        raise HTTPException(status_code=400, detail="Table name is required")
    
    try:
        # Clean table name
        table_name = table_name.lower()
        table_name = regex.sub(r'\s+', '_', table_name)
        table_name = regex.sub(r'[^A-Za-z0-9_]', '', table_name)
        
        spark_session = create_spark_session(app_name)
        
        # Determine existing table format
        table_info_df = spark_session.sql(f"DESCRIBE FORMATTED {table_name}")
        table_info_rows = table_info_df.collect()
        
        storage_format = None
        delimiter_write = delimiter
        
        for row in table_info_rows:
            if row.col_name and "InputFormat" in row.col_name:
                input_format = row.data_type.lower()
                if "orcinputformat" in input_format:
                    storage_format = "orc"
                    break
                elif "parquetinputformat" in input_format:
                    storage_format = "parquet"
                    break
                elif "textinputformat" in input_format:
                    storage_format = "csv"
                    for prop_row in table_info_rows:
                        if prop_row.col_name and "field.delim" in prop_row.col_name.lower():
                            delimiter_write = prop_row.data_type
                            break
                    break
        
        if not storage_format:
            raise HTTPException(
                status_code=500,
                detail=f"Could not determine storage format of table '{table_name}'"
            )
        
        # Save and read CSV
        temp_csv_path = os.path.join(tempfile.gettempdir(), 'temp_csv_file.csv')
        with open(temp_csv_path, 'wb') as f:
            content = await csv_file.read()
            f.write(content)
        
        df_pandas = pd.read_csv(temp_csv_path, sep=delimiter, dtype=str)
        
        # Add timestamp if needed
        existing_columns = spark_session.sql(f"DESCRIBE {table_name}").select("col_name").rdd.flatMap(lambda x: x).collect()
        if 'fg_date' in existing_columns and 'fg_date' not in df_pandas.columns:
            df_pandas['fg_date'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        
        df_spark = spark_service.pandas_to_spark(spark_session, df_pandas)
        
        # Clean column names
        for column_name in df_spark.columns:
            new_column_name = regex.sub(r'\s+', '_', column_name)
            new_column_name = regex.sub(r'[^A-Za-z0-9_]', '', new_column_name)
            new_column_name = new_column_name.lower()
            df_spark = df_spark.withColumnRenamed(column_name, new_column_name)
        
        # Write based on detected format
        if storage_format == "orc":
            df_spark.write.mode("append").format('orc').saveAsTable(table_name)
        elif storage_format == "parquet":
            df_spark.write.mode("append").format('parquet').saveAsTable(table_name)
        elif storage_format == "csv":
            df_spark.write.mode("append").format('csv').option("sep", delimiter_write).saveAsTable(table_name)
        else:
            raise HTTPException(status_code=500, detail=f"Unsupported storage format '{storage_format}'")
        
        spark_session.stop()
        
        return {
            "statusCode": 200,
            "status": "OK",
            "message": f"Data from CSV has been inserted into table {table_name}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sav/convert-to-csv")
async def convert_sav_to_csv(sav_file: UploadFile = File(...)):
    try:
        import pyreadstat
        
        temp_dir = tempfile.mkdtemp()
        temp_sav_path = os.path.join(temp_dir, sav_file.filename)
        
        with open(temp_sav_path, 'wb') as f:
            content = await sav_file.read()
            f.write(content)
        
        original_filename = os.path.splitext(sav_file.filename)[0]
        
        df, meta = pyreadstat.read_sav(
            temp_sav_path,
            apply_value_formats=True,
            formats_as_category=True,
            formats_as_ordered_category=False
        )
        
        csv_data = df.to_csv(sep='|', index=False)
        
        os.remove(temp_sav_path)
        os.rmdir(temp_dir)
        
        from fastapi.responses import Response
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={original_filename}.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))