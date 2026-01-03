import logging
import os
import pandas as pd
import pandas_tfrecords as pdtf
import requests
from pyarrow import fs
from pyarrow.fs import FileSelector
from ydata_profiling import ProfileReport
from typing import Optional

from services.dataprep_db_service import DataPrepDBService
from core import config
from core.utils import cleanup_file

logger = logging.getLogger(__name__)


class TrainingDatasetProfilingService:
    """Service for training dataset processing operations."""
    
    def __init__(self):
        self.db_service = DataPrepDBService()
        self._setup_hadoop_env()
    
    def _setup_hadoop_env(self):
        """Setup Hadoop environment variables."""
        os.environ["HADOOP_HOME"] = config.HADOOP_HOME
        os.environ["HADOOP_CONF_DIR"] = "/etc/hadoop/conf"
        os.environ["ARROW_LIBHDFS_DIR"] = f"{config.HADOOP_HOME}/lib/native"
        
        class_path = ":".join([
            os.environ.get("CLASSPATH", ""),
            os.popen("hadoop classpath").read().strip(),
            f"{config.HADOOP_HOME}/client/*",
            "/usr/yava/3.1.0.0-0000/hive/lib/*",
            "/usr/yava/current/hadoop-client/*",
        ])
        os.environ["CLASSPATH"] = class_path
    
    def download_file_from_hdfs(self, td_id: str, hdfs_path: str, dataset_format: str) -> bool:
        """Download file from HDFS to local directory."""
        try:
            logger.info(f"Downloading from HDFS: {hdfs_path}")
            
            # Connect to HDFS
            hdfs_fs = fs.HadoopFileSystem(
                kerb_ticket=config.TICKET_CACHE_PATH,
                host=config.HDFS_NAMENODE,
                port=8020
            )
            
            # Extract path without HDFS URI
            if hdfs_path.startswith("hdfs://"):
                path_parts = hdfs_path.split("/", 3)
                hdfs_directory = "/" + path_parts[3]
            else:
                hdfs_directory = hdfs_path
            
            # Get files in directory
            file_selector = FileSelector(hdfs_directory, allow_not_found=False, recursive=False)
            files_in_directory = [file_info.path for file_info in hdfs_fs.get_file_info(file_selector)]
            
            if len(files_in_directory) != 1:
                logger.error(f"HDFS directory should contain only one file, found: {len(files_in_directory)}")
                return False
            
            # Download file
            local_destination = os.path.join(config.LOCAL_DIR, f"{td_id}.{dataset_format}")
            
            with hdfs_fs.open_input_stream(files_in_directory[0], compression='detect') as hdfs_file:
                with open(local_destination, 'wb') as local_file:
                    while chunk := hdfs_file.read(2 ** 20):  # 1 MB chunks
                        local_file.write(chunk)
            
            logger.info(f"File downloaded successfully to: {local_destination}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading file from HDFS for ID {td_id}: {e}")
            return False
    
    def generate_report(self, td_id: str, dataset_format: str) -> bool:
        """Generate profiling report for training dataset."""
        try:
            logger.info(f"Generating report for training dataset ID: {td_id}")
            
            # Load data based on format
            if dataset_format == 'tfrecord':
                tfrecord_path = os.path.join(config.LOCAL_DIR, f"{td_id}.tfrecord")
                df = pdtf.tfrecords_to_pandas(tfrecord_path)
            else:
                file_path = os.path.join(config.LOCAL_DIR, f"{td_id}.{dataset_format}")
                df = pd.read_csv(
                    file_path, 
                    engine='python', 
                    sep=None, 
                    parse_dates=True, 
                    encoding="utf-8"
                )
            
            # Generate report
            df_cleaned = df.dropna()
            json_path = os.path.join(config.LOCAL_DIR, f"{td_id}.json")
            
            profile = ProfileReport(df_cleaned, minimal=True)
            profile.to_file(json_path)
            
            logger.info(f"Report saved: {json_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating report for ID {td_id}: {e}")
            return False
    
    def upload_json_file(self, td_id: str) -> bool:
        """Upload JSON report to server."""
        try:
            json_path = os.path.join(config.LOCAL_DIR, f"{td_id}.json")
            
            if not os.path.exists(json_path):
                logger.error(f"JSON file not found: {json_path}")
                return False
            
            url = f"{config.URL3}{td_id}/dataprep"
            
            with open(json_path, 'rb') as json_file:
                files = {'file': ('upload json', json_file)}
                response = requests.post(url, files=files, verify=False, timeout=300)
                
                if response.status_code == 200:
                    logger.info(f"Successfully uploaded report for ID: {td_id}")
                    cleanup_file(json_path)
                    return True
                else:
                    logger.error(f"Upload failed for ID {td_id}: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error uploading JSON for ID {td_id}: {e}")
            return False
    
    def process_single_training_dataset(self, td_id: str, hdfs_path: str, dataset_format: str):
        """Process a single training dataset."""
        try:
            logger.info(f"Starting processing for training dataset ID: {td_id}")
            
            # Update status to processing (2)
            self.db_service.update_training_dataset_status(td_id, 2)
            
            # Download from HDFS
            if not self.download_file_from_hdfs(td_id, hdfs_path, dataset_format):
                logger.error(f"Failed to download file for ID: {td_id}")
                self.db_service.update_training_dataset_status(td_id, 3)
                return
            
            # Generate report
            if not self.generate_report(td_id, dataset_format):
                logger.error(f"Failed to generate report for ID: {td_id}")
                self.db_service.update_training_dataset_status(td_id, 3)
                # Cleanup downloaded file
                file_path = os.path.join(config.LOCAL_DIR, f"{td_id}.{dataset_format}")
                cleanup_file(file_path)
                return
            
            # Upload report
            if not self.upload_json_file(td_id):
                logger.error(f"Failed to upload report for ID: {td_id}")
                self.db_service.update_training_dataset_status(td_id, 3)
                return
            
            # Cleanup downloaded file
            file_path = os.path.join(config.LOCAL_DIR, f"{td_id}.{dataset_format}")
            cleanup_file(file_path)
            
            # Mark as processed (0)
            self.db_service.update_training_dataset_status(td_id, 0)
            logger.info(f"Successfully processed training dataset ID: {td_id}")
            
        except Exception as e:
            logger.error(f"Error processing training dataset ID {td_id}: {e}")
            self.db_service.update_training_dataset_status(td_id, 3)
    
    def process_batch_training_datasets(self, dataset_format: Optional[str] = None):
        """Process all pending training datasets."""
        try:
            training_datasets = self.db_service.get_pending_training_datasets(dataset_format)
            logger.info(f"Found {len(training_datasets)} training datasets to process")
            
            for td_id, hdfs_path, format_type in training_datasets:
                self.process_single_training_dataset(str(td_id), hdfs_path, format_type)
                
        except Exception as e:
            logger.error(f"Error in batch processing training datasets: {e}")
            raise