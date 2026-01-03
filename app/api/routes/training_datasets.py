from fastapi import APIRouter, HTTPException, Response
from models.requests import TrainingDatasetRequest
from services.training_dataset_service import TrainingDatasetService
from services.spark_service import parse_training_dataset_json
from app.core.spark_config import create_spark_session

router = APIRouter()
service = TrainingDatasetService()

@router.post("/")
async def create_training_dataset(request: TrainingDatasetRequest):
    if not request.app_token:
        raise HTTPException(status_code=401, detail="Request does not contain application token")
    
    try:
        training_dataset_obj = parse_training_dataset_json(request.training_dataset)
        training_dataset_dto_obj = eval(request.data)
        
        spark_session = create_spark_session(request.app_name)
        
        result = service.save_training_dataset_data(
            spark_session=spark_session,
            training_dataset_object=training_dataset_obj,
            training_dataset_dto_object=training_dataset_dto_obj
        )
        
        spark_session.stop()
        
        return {
            "statusCode": 200,
            "status": "OK",
            "message": "Training dataset saved",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/preview")
async def preview_training_dataset(request: TrainingDatasetRequest):
    try:
        training_dataset_obj = parse_training_dataset_json(request.training_dataset)
        spark_session = create_spark_session(request.app_name)
        
        dataframe = service.preview_training_dataset_data(
            spark_session=spark_session,
            training_dataset_object=training_dataset_obj
        )
        
        csv_data = dataframe.to_csv(index=False)
        spark_session.stop()
        
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=training_dataset_preview.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/delete")
async def delete_training_dataset(request: TrainingDatasetRequest):
    if not request.app_token:
        raise HTTPException(status_code=401, detail="Request does not contain application token")
    
    try:
        training_dataset_obj = parse_training_dataset_json(request.training_dataset)
        spark_session = create_spark_session(request.app_name)
        
        service.delete_training_dataset_data(
            spark_session=spark_session,
            training_dataset_object=training_dataset_obj
        )
        
        spark_session.stop()
        
        return {
            "statusCode": 200,
            "status": "OK",
            "message": "Training dataset data deleted"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))