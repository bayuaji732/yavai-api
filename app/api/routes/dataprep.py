from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional

from models.dataprep import (
    DataPrepRequest, DataPrepResponse, BatchDataPrepRequest,
    FeatureGroupProfilingRequest, FeatureGroupProfilingResponse,
    TrainingDatasetProfilingRequest, TrainingDatasetProfilingResponse, TrainingDatasetBatchProfilingRequest
)
from services.data_profiling_service import DataProfilingService
from services.feature_profiling_service import FeatureProfilingService
from services.training_dataset_profiling_service import TrainingDatasetProfilingService
from services.dataprep_db_service import DataPrepDBService
from core.utils import setup_logger

router = APIRouter()
logger = setup_logger(__name__)


# ============================================
# DATASET ENDPOINTS
# ============================================

@router.post("/dataset/profiling", response_model=DataPrepResponse, tags=["Dataset"])
async def run_dataset_profiling(
    request: DataPrepRequest,
    background_tasks: BackgroundTasks
):
    """
    Process a single dataset file.
    
    Triggers data preparation for the specified file. The processing happens
    asynchronously in the background.
    
    - **table_name**: Database table name (e.g., "dataset")
    - **file_id**: Unique identifier for the file
    - **file_type**: Type of file (csv, tsv, xls, xlsx, sav)
    """
    try:
        logger.info(f"Received request to process file_id: {request.file_id}")
        
        dataset_service = DataProfilingService()
        if not dataset_service.is_valid_file_type(request.file_type):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {dataset_service.allowed_file_types}"
            )
        
        background_tasks.add_task(
            dataset_service.process_single_dataset,
            request.table_name,
            request.file_id,
            request.file_type
        )
        
        return DataPrepResponse(
            status="accepted",
            message="Dataset processing started",
            file_id=request.file_id
        )
        
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dataset/profiling/batch", response_model=DataPrepResponse, tags=["Dataset"])
async def run_dataset_profiling_batch(
    request: BatchDataPrepRequest,
    background_tasks: BackgroundTasks
):
    """
    Process multiple datasets based on filters.
    
    Batch processes all datasets matching the specified criteria.
    """
    try:
        logger.info(f"Received batch request for table: {request.table_name}")
        
        dataset_service = DataProfilingService()
        background_tasks.add_task(
            dataset_service.process_batch_datasets,
            request.table_name,
            request.filters
        )
        
        return DataPrepResponse(
            status="accepted",
            message="Batch processing started",
            file_id="batch"
        )
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dataset/profiling/status/{file_id}", tags=["Dataset"])
async def get_dataset_profiling_status(file_id: str, table_name: str = "dataset"):
    """
    Get the processing status of a dataset.
    
    Status codes:
    - 0: Successfully processed
    - 1: Pending processing
    - 2: Currently processing
    - 3: Processing failed
    - None: Not found or error
    """
    try:
        db_service = DataProfilingService()
        status = db_service.get_dataset_status(table_name, file_id)
        
        if status is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        return {
            "file_id": file_id,
            "status": status,
            "message": "Status retrieved successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dataset/profiling/list", tags=["Dataset"])
async def get_list_datasets(
    table_name: str = "dataset",
    status: Optional[int] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    List datasets with optional filtering.
    
    - **table_name**: Database table name
    - **status**: Filter by status (0, 1, 2, 3)
    - **limit**: Maximum number of results
    - **offset**: Number of results to skip
    """
    try:
        db_service = DataProfilingService()
        datasets = db_service.list_datasets(table_name, status, limit, offset)
        
        return {
            "total": len(datasets),
            "datasets": datasets,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# FEATURE GROUP ENDPOINTS
# ============================================

@router.post("/feature-groups/profiling", response_model=FeatureGroupProfilingResponse, tags=["Feature Group"])
async def run_feature_group_profiling(
    request: FeatureGroupProfilingRequest,
    background_tasks: BackgroundTasks
):
    """
    Process a single feature group.
    
    Triggers data preparation for the specified feature group table from either
    Hive (offline) or Redis (online) storage.
    
    - **table_name**: Name of the feature group table
    - **online**: True for Redis storage, False for Hive storage
    """
    try:
        logger.info(f"Received request to process feature group: {request.table_name}")
        
        feature_group_service = FeatureProfilingService()
        background_tasks.add_task(
            feature_group_service.process_single_feature_group,
            request.table_name,
            request.online
        )
        
        return FeatureGroupProfilingResponse(
            status="accepted",
            message="Feature group processing started",
            table_name=request.table_name
        )
        
    except Exception as e:
        logger.error(f"Error processing feature group: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feature-groups/profiling/batch", response_model=FeatureGroupProfilingResponse, tags=["Feature Group"])
async def run_feature_group_profiling_batch(background_tasks: BackgroundTasks) -> FeatureGroupProfilingResponse:
    """
    Process all pending feature groups.
    
    Batch processes all feature groups with dataprep_status = 1 (pending).
    """
    try:
        logger.info("Received batch request for feature groups")
        
        feature_group_service = FeatureProfilingService()
        background_tasks.add_task(
            feature_group_service.process_batch_feature_groups
        )
        
        return FeatureGroupProfilingResponse(
            status="accepted",
            message="Batch feature group processing started"
        )
        
    except Exception as e:
        logger.error(f"Error in batch feature group processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-groups/profiling/status/{table_name}", tags=["Feature Group"])
async def get_feature_group_profiling_status(table_name: str):
    """
    Get the processing status of a feature group.
    
    Status codes:
    - 0: Successfully processed
    - 1: Pending processing
    - 2: Currently processing
    - 3: Processing failed
    """
    try:
        db_service = DataPrepDBService()
        status = db_service.get_feature_group_status(table_name)
        
        if status is None:
            raise HTTPException(status_code=404, detail="Feature group not found")
        
        return {
            "table_name": table_name,
            "status": status,
            "message": "Status retrieved successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting feature group status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-groups/profiling/list", tags=["Feature Group"])
async def get_list_feature_groups(
    status: Optional[int] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    List feature groups with optional filtering.
    
    - **status**: Filter by status (0, 1, 2, 3)
    - **limit**: Maximum number of results
    - **offset**: Number of results to skip
    """
    try:
        db_service = DataPrepDBService()
        feature_groups = db_service.list_feature_groups(status, limit, offset)
        
        return {
            "total": len(feature_groups),
            "feature_groups": feature_groups,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error listing feature groups: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# TRAINING DATASET ENDPOINTS
# ============================================

@router.post("/training-datasets/profiling", response_model=TrainingDatasetProfilingResponse, tags=["Training Dataset"])
async def run_training_dataset_profiling(
    request: TrainingDatasetProfilingRequest,
    background_tasks: BackgroundTasks
):
    """
    Process a single training dataset.
    
    Downloads the dataset from HDFS and triggers data preparation.
    
    - **td_id**: Training dataset unique identifier
    - **hdfs_path**: Full HDFS path to the dataset
    - **dataset_format**: Format of the dataset (csv, tfrecord)
    """
    try:
        logger.info(f"Received request to process training dataset ID: {request.td_id}")
        
        training_dataset_service = TrainingDatasetProfilingService()
        background_tasks.add_task(
            training_dataset_service.process_single_training_dataset,
            request.td_id,
            request.hdfs_path,
            request.dataset_format
        )
        
        return TrainingDatasetProfilingResponse(
            status="accepted",
            message="Training dataset processing started",
            td_id=request.td_id
        )
        
    except Exception as e:
        logger.error(f"Error processing training dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/training-datasets/profiling/batch", response_model=TrainingDatasetProfilingResponse, tags=["Training Dataset"])
async def run_training_datasets_profiling_batch(
    request: TrainingDatasetBatchProfilingRequest,
    background_tasks: BackgroundTasks
):
    """
    Process all pending training datasets.
    
    Batch processes all training datasets with dataprep_status = 1 (pending).
    Optionally filter by dataset format.
    
    - **dataset_format**: Optional filter for dataset format (csv, tfrecord)
    """
    try:
        logger.info(f"Received batch request for training datasets (format: {request.dataset_format})")
        
        training_dataset_service = TrainingDatasetProfilingService()
        background_tasks.add_task(
            training_dataset_service.process_batch_training_datasets,
            request.dataset_format
        )
        
        return TrainingDatasetProfilingResponse(
            status="accepted",
            message="Batch training dataset processing started"
        )
        
    except Exception as e:
        logger.error(f"Error in batch training dataset processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-datasets/profiling/status/{td_id}", tags=["Training Dataset"])
async def get_training_dataset_profiling_status(td_id: str):
    """
    Get the processing status of a training dataset.
    
    Status codes:
    - 0: Successfully processed
    - 1: Pending processing
    - 2: Currently processing
    - 3: Processing failed
    """
    try:
        db_service = DataPrepDBService()
        status = db_service.get_training_dataset_status(td_id)
        
        if status is None:
            raise HTTPException(status_code=404, detail="Training dataset not found")
        
        return {
            "td_id": td_id,
            "status": status,
            "message": "Status retrieved successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training dataset status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-datasets/profiling/list", tags=["Training Dataset"])
async def get_list_training_datasets(
    status: Optional[int] = None,
    dataset_format: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    List training datasets with optional filtering.
    
    - **status**: Filter by status (0, 1, 2, 3)
    - **dataset_format**: Filter by format (csv, tfrecord)
    - **limit**: Maximum number of results
    - **offset**: Number of results to skip
    """
    try:
        db_service = DataPrepDBService()
        training_datasets = db_service.list_training_datasets(
            status, dataset_format, limit, offset
        )
        
        return {
            "total": len(training_datasets),
            "training_datasets": training_datasets,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error listing training datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))