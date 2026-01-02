from pydantic import BaseModel
from typing import List, Optional

class FeatureGroupRequest(BaseModel):
    app_token: Optional[str] = None
    app_name: str
    feature_group: str

class TrainingDatasetRequest(BaseModel):
    app_token: Optional[str] = None
    app_name: str
    training_dataset: str
    data: Optional[str] = None

class DownloadRequest(BaseModel):
    app_name: str
    destination_path: str
    feature_group: str

class AddColumnRequest(BaseModel):
    app_token: str
    app_name: str
    feature_group: str

class FeatureGroupSizeRequest(BaseModel):
    data: List[dict]