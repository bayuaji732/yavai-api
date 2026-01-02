from typing import Optional, List
from pydantic import BaseModel
from models.feature import Feature

class FeatureGroup(BaseModel):
    feature_group_id: Optional[str] = None
    name: str
    description: str
    owner_username: str
    file_item_id: Optional[List[str]] = None
    features: Optional[List[Feature]] = []
    partition_keys: Optional[List[str]] = None
    table_name: Optional[str] = None
    database_name: Optional[str] = None
    status: str = "EMPTY"
    permission: Optional[int] = 1
    data_type: Optional[str] = None
    feature_group_online: bool = False
    
    class Config:
        from_attributes = True