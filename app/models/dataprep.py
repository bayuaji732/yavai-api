from pydantic import BaseModel, Field
from typing import Optional


# ============================================
# DATASET MODELS
# ============================================

class DataPrepRequest(BaseModel):
    """Request model for profiling a single dataset."""
    table_name: str = Field(..., description="Name of the table to process")
    file_id: str = Field(..., description="File ID to process")
    file_type: str = Field(..., description="File type (csv, tsv, xls, xlsx, sav)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "table_name": "dataset",
                "file_id": "12345",
                "file_type": "csv"
            }
        }


class DataPrepResponse(BaseModel):
    """Response model for data preparation operations."""
    status: str
    message: str
    file_id: Optional[str] = None
    task_id: Optional[str] = None


class BatchDataPrepRequest(BaseModel):
    """Request model for batch profiling datasets."""
    table_name: str = Field(..., description="Name of the table to process")
    filters: Optional[dict] = Field(None, description="Optional filters for batch processing")
    
    class Config:
        json_schema_extra = {
            "example": {
                "table_name": "dataset",
                "filters": {"status": 1}
            }
        }


# ============================================
# FEATURE GROUP PROFILING MODELS
# ============================================

class FeatureGroupProfilingRequest(BaseModel):
    """Request model for profiling a feature group."""
    table_name: str = Field(..., description="Name of the feature group table")
    online: bool = Field(..., description="Whether the data is stored in Redis (True) or Hive (False)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "table_name": "customer_features",
                "online": False
            }
        }


class FeatureGroupProfilingResponse(BaseModel):
    """Response model for feature group profiling operations."""
    status: str
    message: str
    table_name: Optional[str] = None


# ============================================
# TRAINING DATASET PROFILING MODELS
# ============================================

class TrainingDatasetProfilingRequest(BaseModel):
    """Request model for profiling a training dataset."""
    td_id: str = Field(..., description="Training dataset ID")
    hdfs_path: str = Field(..., description="HDFS path to the dataset")
    dataset_format: str = Field(..., description="Dataset format (csv, tfrecord)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "td_id": "12345",
                "hdfs_path": "hdfs://namenode:8020/datasets/training/data.csv",
                "dataset_format": "csv"
            }
        }


class TrainingDatasetBatchProfilingRequest(BaseModel):
    """Request model for batch profiling training datasets."""
    dataset_format: Optional[str] = Field(None, description="Filter by dataset format (csv, tfrecord)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "dataset_format": "csv"
            }
        }


class TrainingDatasetProfilingResponse(BaseModel):
    """Response model for training dataset profiling operations."""
    status: str
    message: str
    td_id: Optional[str] = None


# ============================================
# STATUS MODELS
# ============================================

class StatusResponse(BaseModel):
    """Response model for status queries."""
    id: str
    status: int
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "12345",
                "status": 0,
                "message": "Successfully processed"
            }
        }


class ListResponse(BaseModel):
    """Generic response model for list operations."""
    total: int
    items: list
    limit: int
    offset: int