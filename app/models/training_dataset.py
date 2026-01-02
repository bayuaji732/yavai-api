from typing import Optional, List
from pydantic import BaseModel

class TrainingDataset(BaseModel):
    training_dataset_id: Optional[str] = None
    name: str
    description: str
    owner_username: Optional[str] = None
    dataset_format: str = "csv"
    permission: Optional[int] = None
    path: Optional[str] = None
    status: Optional[str] = None
    feature_group_ids: Optional[List[str]] = []
    
    class Config:
        from_attributes = True