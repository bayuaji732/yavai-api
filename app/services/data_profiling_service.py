import logging
import os
import pandas as pd
import requests
from ydata_profiling import ProfileReport
from typing import Optional, Dict, Any
import warnings
from urllib3.exceptions import InsecureRequestWarning

from app.services.dataprep_db_service import DataPrepDBService
from app.core import config
from app.core.utils import ensure_local_dir, cleanup_file

warnings.filterwarnings("ignore", category=InsecureRequestWarning)
logger = logging.getLogger(__name__)

class DataProfilingService:
    """Service for dataset processing operations."""
    
    def __init__(self):
        self.db_service = DataPrepDBService()
        self.allowed_file_types = config.ALLOWED_FILE_TYPES
        ensure_local_dir()
    
    def is_valid_file_type(self, file_type: str) -> bool:
        """Check if file type is supported."""
        return file_type in self.allowed_file_types
    
    def process_single_dataset(
        self, 
        table_name: str, 
        file_id: str, 
        file_type: str
    ):
        """Process a single dataset file."""
        try:
            logger.info(f"Starting processing for file_id: {file_id}, type: {file_type}")

            # Check if already processed
            current_status = self.db_service.get_dataset_status(table_name, file_id)
            if current_status == 0:
                logger.info(f"File {file_id} already processed. Skipping.")
                return
            
            # Update status to processing (2)
            self.db_service.update_preprocessing_status(table_name, file_id, 2)
            
            # Download file
            if not self.download_file(file_id, file_type):
                logger.error(f"Failed to download file_id: {file_id}")
                self.db_service.update_preprocessing_status(table_name, file_id, 3)
                return
            
            # Generate report
            if not self.generate_report(file_id, file_type):
                logger.error(f"Failed to generate report for file_id: {file_id}")
                self.db_service.update_preprocessing_status(table_name, file_id, None)
                return
            
            # Upload JSON files
            if not self.upload_json_files(file_id, file_type):
                logger.error(f"Failed to upload JSON for file_id: {file_id}")
                self.db_service.update_preprocessing_status(table_name, file_id, None)
                return
            
            # Mark as processed
            self.db_service.update_preprocessing_status(table_name, file_id, 0)
            logger.info(f"Successfully processed file_id: {file_id}")
            
        except Exception as e:
            logger.error(f"Error processing file_id {file_id}: {e}")
            self.db_service.update_preprocessing_status(table_name, file_id, None)
    
    def process_batch_datasets(
        self, 
        table_name: str, 
        filters: Optional[Dict[str, Any]] = None
    ):
        """Process multiple datasets based on filters."""
        try:
            status = filters.get('status', 1) if filters else 1
            datasets = self.db_service.get_datasets_by_status(table_name, status)
            
            logger.info(f"Found {len(datasets)} datasets to process")
            
            for dataset in datasets:
                file_id = str(dataset['id'])
                file_type = str(dataset['file_type'])
                self.process_single_dataset(table_name, file_id, file_type)
                
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise
    
    def download_file(self, file_id: str, file_type: str) -> bool:
        """Download a file from the server."""
        url = f"{config.URL}{file_id}/download2"
        file_path = os.path.join(config.LOCAL_DIR, f"{file_id}.{file_type}")
        
        try:
            logger.info(f"Downloading file from: {url}")
            response = requests.get(url, verify=False, timeout=300)
            
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Successfully downloaded file_id: {file_id}")
                return True
            else:
                logger.error(f"Download failed with status: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading file_id {file_id}: {e}")
            return False
    
    def generate_report(self, file_id: str, file_type: str) -> bool:
        """Generate a profile report for the given file."""
        file_path = os.path.join(config.LOCAL_DIR, f"{file_id}.{file_type}")
        
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False
            
            logger.info(f"Generating report for file_id: {file_id}")
            
            if file_type in ['csv', 'tsv']:
                sep = '\t' if file_type == 'tsv' else None
                df = pd.read_csv(file_path, sep=sep, engine='python', on_bad_lines='skip')
                return self._generate_and_save_report(df, file_id, file_type)
                
            elif file_type == 'sav':
                df = pd.read_spss(file_path)
                return self._generate_and_save_report(df, file_id, file_type)
                
            elif file_type in ['xls', 'xlsx']:
                xls = pd.ExcelFile(file_path)
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(
                        xls, 
                        sheet_name, 
                        engine='xlrd' if file_type == 'xls' else 'openpyxl'
                    )
                    self._generate_and_save_report(df, file_id, sheet_name)
                return True
            else:
                logger.warning(f"Unsupported file type: {file_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error generating report for file_id {file_id}: {e}")
            return False
    
    def _generate_and_save_report(
        self, 
        df: pd.DataFrame, 
        file_id: str, 
        identifier: str
    ) -> bool:
        """Generate and save a profile report for a DataFrame."""
        try:
            df_cleaned = df.fillna(0)
            json_path = os.path.join(config.LOCAL_DIR, f"{file_id}_{identifier}.json")
            
            profile = ProfileReport(df_cleaned, minimal=True)
            profile.to_file(json_path)
            
            logger.info(f"Report saved: {json_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving report for {file_id}_{identifier}: {e}")
            return False
    
    def upload_json_files(self, file_id: str, file_type: str) -> bool:
        """Upload JSON files to the server."""
        try:
            json_files = [
                os.path.join(root, file)
                for root, dirs, files in os.walk(config.LOCAL_DIR)
                for file in files
                if file.startswith(f"{file_id}_") and file.endswith(".json")
            ]
            
            if not json_files:
                logger.error(f"No JSON files found for file_id: {file_id}")
                return False
            
            logger.info(f"Found {len(json_files)} JSON files to upload")
            
            for json_path in json_files:
                url = f"{config.URL}{file_id}/upload-dataprep"
                file_name = f'{file_id}_{file_type}.json'
                
                with open(json_path, 'rb') as json_file:
                    files = {'file': (file_name, json_file)}
                    response = requests.post(url, files=files, verify=False, timeout=300)
                    
                    if response.status_code == 200:
                        logger.info(f"Successfully uploaded: {json_path}")
                        cleanup_file(json_path)
                    else:
                        logger.error(f"Upload failed for {json_path}: {response.text}")
                        return False
            
            # Cleanup downloaded file
            file_path = os.path.join(config.LOCAL_DIR, f"{file_id}.{file_type}")
            cleanup_file(file_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error uploading JSON files for file_id {file_id}: {e}")
            return False