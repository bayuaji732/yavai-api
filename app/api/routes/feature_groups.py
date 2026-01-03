import os
import csv
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse
from models.requests import FeatureGroupRequest, DownloadRequest, AddColumnRequest, FeatureGroupSizeRequest
from services.feature_group_service import FeatureGroupService
from services.spark_service import parse_feature_group_json
from app.core.spark_config import create_spark_session
from db.redis import get_redis_client
from hdfs.ext.kerberos import KerberosClient
from core import config

router = APIRouter()
service = FeatureGroupService()

@router.post("/")
async def create_feature_group(request: FeatureGroupRequest):
    if not request.app_token:
        raise HTTPException(status_code=401, detail="Request does not contain application token")
    
    try:
        feature_group_obj = parse_feature_group_json(request.feature_group)
        spark_session = create_spark_session(request.app_name)
        
        saved_feature_group = service.save_feature_group_data(
            spark_session=spark_session,
            feature_group_object=feature_group_obj
        )
        
        spark_session.stop()
        
        return {
            "statusCode": 200,
            "status": "OK",
            "message": "Feature data saved",
            "data": saved_feature_group
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/preview")
async def preview_feature_group(request: FeatureGroupRequest):
    try:
        feature_group_obj = parse_feature_group_json(request.feature_group)
        spark_session = create_spark_session(request.app_name)
        
        dataframe = service.preview_feature_group_data(
            spark_session=spark_session,
            feature_group_object=feature_group_obj
        )
        
        csv_data = dataframe.to_csv(index=False)
        spark_session.stop()
        
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=feature_preview.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/download")
async def download_feature_group(request: DownloadRequest):
    try:
        feature_group_obj = parse_feature_group_json(request.feature_group)
        online_mode = feature_group_obj.feature_group_online
        
        if online_mode:
            redis_client = get_redis_client()
            redis_key = feature_group_obj.table_name
            redis_data = redis_client.smembers(redis_key)
            
            if not redis_data:
                raise HTTPException(status_code=404, detail=f"No data found in Redis for key {redis_key}")
            
            fullname = f"{request.destination_path}.csv"
            
            with open(fullname, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["key", "value"])
                for item in redis_data:
                    writer.writerow([item, item.split(":")[-1]])
        else:
            spark_session = create_spark_session(request.app_name)
            fullname = service.download_feature_group_data(
                spark_session=spark_session,
                feature_group_object=feature_group_obj,
                destination_path=request.destination_path
            )
            spark_session.stop()
        
        if not os.path.exists(fullname):
            raise HTTPException(status_code=404, detail=f"File {fullname} does not exist")
        
        return FileResponse(
            fullname,
            media_type="text/csv",
            filename=os.path.basename(fullname)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add-column")
async def add_column_feature_group(request: AddColumnRequest):
    try:
        feature_group_obj = parse_feature_group_json(request.feature_group)
        spark_session = create_spark_session(request.app_name)
        
        service.add_column_feature_group_data(
            spark_session=spark_session,
            feature_group_object=feature_group_obj
        )
        
        spark_session.stop()
        
        return {
            "statusCode": 200,
            "status": "OK",
            "message": "Added column successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/delete")
async def delete_feature_group(request: FeatureGroupRequest):
    if not request.app_token:
        raise HTTPException(status_code=401, detail="Request does not contain application token")
    
    try:
        feature_group_obj = parse_feature_group_json(request.feature_group)
        spark_session = create_spark_session(request.app_name)
        
        service.delete_feature_group_data(
            spark_session=spark_session,
            feature_group_object=feature_group_obj
        )
        
        spark_session.stop()
        
        return {
            "statusCode": 200,
            "status": "OK",
            "message": "Feature data deleted"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/size")
async def get_feature_group_size(request: FeatureGroupSizeRequest):
    try:
        total_size = 0
        client = KerberosClient(config.HDFS_HOST)
        
        for table in request.data:
            try:
                path_dir = os.path.join(config.HIVE_DIR_PATH, table['table_name'])
                dir_status = client.content(path_dir, strict=True)
                
                if dir_status and 'spaceConsumed' in dir_status:
                    total_size += dir_status['spaceConsumed']
            except Exception as e:
                print(f"Exception: {str(e)}")
                continue
        
        return {
            "StatusCode": 200,
            "Status": "OK",
            "Total Size": total_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))