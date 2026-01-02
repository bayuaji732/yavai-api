from typing import Optional, List
from pydantic import BaseModel

class Feature(BaseModel):
    feature_id: Optional[str] = None
    name: str
    description: Optional[str] = ""
    feature_type: str
    extraction_algorithm: Optional[str] = None
    text_cleaning: Optional[List[str]] = None
    language: Optional[str] = ""
    partition_key: bool = False
    feature_group_id: Optional[str] = None
    
    class Config:
        from_attributes = True