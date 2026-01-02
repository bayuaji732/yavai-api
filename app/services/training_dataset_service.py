from pyspark.sql import SparkSession
from typing import Any

class TrainingDatasetService:
    
    def save_training_dataset_data(
        self,
        spark_session: SparkSession,
        training_dataset_object: Any,
        training_dataset_dto_object: Any
    ) -> dict:
        """Save training dataset data"""
        from services.spark_service import SparkService
        spark_service = SparkService()
        
        # Get feature groups and build query
        feature_group_obj = self._get_feature_group(training_dataset_dto_object.feature_group_id)
        query = self._build_query(feature_group_obj, training_dataset_dto_object)
        
        # Read data from query
        data = query.read(spark_session)
        
        # Create path and save
        training_dataset_object.path = self._create_path(training_dataset_object)
        spark_service.save_training_dataset(spark_session, training_dataset_object, data)
        
        return training_dataset_object.to_dict() if hasattr(training_dataset_object, 'to_dict') else {}
    
    def preview_training_dataset_data(self, spark_session: SparkSession, training_dataset_object: Any):
        """Preview training dataset"""
        from services.spark_service import SparkService
        spark_service = SparkService()
        
        dataframe = spark_service.read_training_dataset(spark_session, training_dataset_object)
        return dataframe.toPandas() if hasattr(dataframe, 'toPandas') else dataframe
    
    def delete_training_dataset_data(self, spark_session: SparkSession, training_dataset_object: Any):
        """Delete training dataset"""
        if not training_dataset_object.path or not training_dataset_object.path.startswith("hdfs://"):
            return
        
        sc = spark_session.sparkContext
        fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
        fs.delete(sc._jvm.org.apache.hadoop.fs.Path(training_dataset_object.path), True)
    
    def _get_feature_group(self, feature_group_id: str):
        """Get feature group from API"""
        import requests
        from core import config
        
        url = f"{config.YAVAI_API_BASE_URL}/dataset-management/api/v1/lib/feature-groups/{feature_group_id}"
        response = requests.get(url, headers={"Content-Type": "application/json"}, verify=False)
        
        if response.status_code != 200:
            raise ValueError(f"Failed to get feature group: {response.status_code}")
        
        return response.json().get("data")
    
    def _build_query(self, feature_group_obj, training_dataset_dto_object):
        """Build query for training dataset"""
        # Simplified query building - in real implementation would use fs_query module
        class SimpleQuery:
            def read(self, spark_session):
                # Read from feature group table
                return spark_session.sql(f"SELECT * FROM {feature_group_obj.get('tableName')}")
        
        return SimpleQuery()
    
    def _create_path(self, training_dataset_object) -> str:
        """Create HDFS path for training dataset"""
        from core import config
        import os
        
        path = os.path.join(config.HDFS_NAME_NODE, "user", "apps", "hive")
        path = os.path.join(path, f"training_dataset_{training_dataset_object.training_dataset_id}")
        return path