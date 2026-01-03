import logging
import os
import pandas as pd
import json
import requests
from ydata_profiling import ProfileReport

from services.dataprep_db_service import DataPrepDBService
from core.utils import cleanup_file
from core import config
from db.hive import get_hive_connection
from db.redis import get_redis_client

logger = logging.getLogger(__name__)


class FeatureProfilingService:
    """Service for feature group dataset processing operations."""
    
    def __init__(self):
        self.db_service = DataPrepDBService()
    
    def get_data_from_hive(self, table: str) -> pd.DataFrame:
        """Fetch data from Hive."""
        try:
            with get_hive_connection() as conn:
                cursor = conn.cursor()
                query = f"SELECT * FROM {table}"
                cursor.execute(query)
                result = cursor.fetchall()
                    
                if not result:
                    logger.warning(f"No data fetched from Hive for table: {table}")
                    return pd.DataFrame()
                    
                columns = [desc[0] for desc in cursor.description]
                df = pd.DataFrame(result, columns=columns)
                return df
        except Exception as e:
            logger.error(f"Error fetching data from Hive for table {table}: {e}")
            return pd.DataFrame()
    
    def get_data_from_redis(self, table: str) -> pd.DataFrame:
        """Fetch data from Redis."""
        try:
            r = get_redis_client()
            redis_type = r.type(table)
            
            if redis_type == "string":
                data = r.get(table)
                return pd.DataFrame(json.loads(data)) if data else pd.DataFrame()
            elif redis_type == "list":
                data = r.lrange(table, 0, -1)
                return pd.DataFrame([json.loads(row) for row in data])
            else:
                logger.warning(f"Unsupported Redis data type '{redis_type}' for table: {table}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching data from Redis for table {table}: {e}")
            return pd.DataFrame()
    
    def generate_report(self, table: str, online: bool) -> bool:
        """Generate profiling report for feature group."""
        try:
            logger.info(f"Generating report for table: {table}, online: {online}")
            
            # Fetch data based on storage type
            if online:
                df = self.get_data_from_redis(table)
            else:
                df = self.get_data_from_hive(table)
            
            if df.empty:
                logger.warning(f"No data available for table: {table}")
                return False
            
            # Clean and generate report
            df_cleaned = df.dropna()
            id_name = table.replace("_", "-")
            json_path = os.path.join(config.LOCAL_DIR, f"{id_name}.json")
            
            profile = ProfileReport(df_cleaned, minimal=True)
            profile.to_file(json_path)
            
            logger.info(f"Report saved: {json_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating report for table {table}: {e}")
            return False
    
    def upload_json_file(self, table: str) -> bool:
        """Upload JSON report to server."""
        try:
            id_name = table.replace("_", "-")
            json_path = os.path.join(config.LOCAL_DIR, f"{id_name}.json")
            
            if not os.path.exists(json_path):
                logger.error(f"JSON file not found: {json_path}")
                return False
            
            url = f"{config.URL2}{id_name}/dataprep"
            
            with open(json_path, 'rb') as json_file:
                files = {'file': ('upload json', json_file)}
                response = requests.post(url, files=files, verify=False, timeout=300)
                
                if response.status_code == 200:
                    logger.info(f"Successfully uploaded report for table: {table}")
                    cleanup_file(json_path)
                    return True
                else:
                    logger.error(f"Upload failed for table {table}: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error uploading JSON for table {table}: {e}")
            return False
    
    def process_single_feature_group(self, table: str, online: bool):
        """Process a single feature group."""
        try:
            logger.info(f"Starting processing for feature group: {table}")
            
            # Update status to processing (2)
            self.db_service.update_feature_group_status(table, 2)
            
            # Generate report
            if not self.generate_report(table, online):
                logger.error(f"Failed to generate report for table: {table}")
                self.db_service.update_feature_group_status(table, 3)
                return
            
            # Upload report
            if not self.upload_json_file(table):
                logger.error(f"Failed to upload report for table: {table}")
                self.db_service.update_feature_group_status(table, 3)
                return
            
            # Mark as processed (0)
            self.db_service.update_feature_group_status(table, 0)
            logger.info(f"Successfully processed feature group: {table}")
            
        except Exception as e:
            logger.error(f"Error processing feature group {table}: {e}")
            self.db_service.update_feature_group_status(table, 3)
    
    def process_batch_feature_groups(self):
        """Process all pending feature groups."""
        try:
            feature_groups = self.db_service.get_pending_feature_groups()
            logger.info(f"Found {len(feature_groups)} feature groups to process")
            
            for table, online in feature_groups:
                self.process_single_feature_group(table, online)
                
        except Exception as e:
            logger.error(f"Error in batch processing feature groups: {e}")
            raise