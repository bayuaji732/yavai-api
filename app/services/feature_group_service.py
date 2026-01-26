import os
import regex
from typing import Any
from pyspark.sql import SparkSession

class FeatureGroupService:
    
    def save_feature_group_data(self, spark_session: SparkSession, feature_group_object: Any) -> dict:
        """Save feature group data from API call"""
        from app.services.spark_service import SparkService
        spark_service = SparkService()
        
        # Get file path
        file_path = self._get_file_path(feature_group_object.file_item_id)
        if not file_path:
            raise ValueError("Cannot write feature group data: file path is empty")
        
        # Read file based on data type
        if feature_group_object.data_type in ["xls", "xlsx"]:
            spark_dataframe = spark_service.read_excel(spark_session, file_path)
        elif feature_group_object.data_type in ["csv", "tsv"]:
            spark_dataframe = spark_service.read_csv(spark_session, file_path)
        else:
            raise ValueError("Data types not supported. Use xls/xlsx or csv/tsv format")
        
        # Clean column names
        for column_name in spark_dataframe.columns:
            new_column_name = regex.sub(r'\s+', '_', column_name)
            new_column_name = regex.sub(r'[^A-Za-z0-9_]', '', new_column_name)
            new_column_name = new_column_name.lower()
            spark_dataframe = spark_dataframe.withColumnRenamed(column_name, new_column_name)
        
        # Select and extract features
        spark_dataframe = self._select_columns(spark_dataframe, feature_group_object)
        spark_dataframe = self._extract_features(spark_session, spark_dataframe, feature_group_object)
        
        # Save to Hive
        spark_service.save_to_hive(spark_session, spark_dataframe, feature_group_object)
        
        return feature_group_object.to_dict() if hasattr(feature_group_object, 'to_dict') else {}
    
    def preview_feature_group_data(self, spark_session: SparkSession, feature_group_object: Any):
        """Preview feature group data"""
        if not feature_group_object.table_name or feature_group_object.status != "SUCCESS":
            raise ValueError("Feature group currently has no data")
        
        columns = self._get_columns_to_extract(feature_group_object)
        extracted_columns = [self._clean_column_name(col) for col in columns]
        
        df = spark_session.sql(f"SELECT * FROM {feature_group_object.table_name} LIMIT 20")
        
        if 'fg_date' in df.columns:
            columns_result = ', '.join(extracted_columns + ['fg_date'])
        else:
            columns_result = ', '.join(extracted_columns)
        
        query = f"SELECT {columns_result} FROM {feature_group_object.table_name}"
        return spark_session.sql(query).toPandas()
    
    def download_feature_group_data(self, spark_session: SparkSession, feature_group_object: Any, destination_path: str) -> str:
        """Download feature group data as CSV"""
        query = f"SELECT * FROM {feature_group_object.table_name}"
        spark_df = spark_session.sql(query)
        pandas_df = spark_df.toPandas()
        
        os.makedirs(destination_path, exist_ok=True)
        local_file_path = os.path.join(destination_path, f"{feature_group_object.feature_group_id}.csv")
        pandas_df.to_csv(local_file_path, index=False)
        
        return local_file_path
    
    def add_column_feature_group_data(self, spark_session: SparkSession, feature_group_object: Any):
        """Add columns to feature group"""
        columns = self._get_columns_to_extract(feature_group_object)
        extracted_columns = [self._clean_column_name(col) + ' string' for col in columns]
        columns_result = ', '.join(extracted_columns)
        
        spark_session.sql(f"ALTER TABLE {feature_group_object.table_name} ADD COLUMNS ({columns_result})")
    
    def delete_feature_group_data(self, spark_session: SparkSession, feature_group_object: Any):
        """Delete feature group data"""
        spark_session.sql(f"DROP TABLE IF EXISTS {feature_group_object.table_name}")
    
    def _get_file_path(self, file_item_id: str) -> str:
        """Get file path from API"""
        import requests
        from app.core import config
        
        url = f"{config.YAVAI_API_BASE_URL}/dataset-management/api/v1/lib/files/{file_item_id}/s3a-path"
        response = requests.get(url, headers={"Content-Type": "application/json"}, verify=False)
        
        if response.status_code != 200:
            raise ValueError(f"Failed to get file path: {response.status_code}")
        
        return response.json().get("data")
    
    def _select_columns(self, dataframe, feature_group_object):
        """Select specified columns from dataframe"""
        columns = self._get_columns_to_extract(feature_group_object)
        return dataframe.select(*columns)
    
    def _extract_features(self, spark_session, dataframe, feature_group_object):
        """Extract features using specified algorithms"""
        # Placeholder for feature extraction logic
        # Would implement TF-IDF, OneHot, TextCleansing based on feature.extraction_algorithm
        return dataframe
    
    def _get_columns_to_extract(self, feature_group_object) -> list:
        """Get list of columns to extract"""
        return [feature.name for feature in feature_group_object.features]
    
    def _clean_column_name(self, column_name: str) -> str:
        """Clean column name"""
        new_name = regex.sub(r'\s+', '_', column_name)
        new_name = regex.sub(r'[^A-Za-z0-9_]', '', new_name)
        return new_name.lower()