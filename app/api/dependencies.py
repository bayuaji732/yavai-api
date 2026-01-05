from fastapi import Depends
from app.core.security import verify_token
from app.core.spark_config import create_spark_session
from pyspark.sql import SparkSession

def get_current_user(token_data: dict = Depends(verify_token)) -> dict:
    """Dependency to get current authenticated user"""
    return token_data

def get_spark_session(app_name: str = "FeatureStore") -> SparkSession:
    """Dependency to get Spark session"""
    return create_spark_session(app_name)