from psycopg2.extras import RealDictCursor
from typing import Optional, List, Dict, Any, Tuple

from db.postgres import get_db_connection
from core.utils import setup_logger

logger = setup_logger(__name__)

class DataPrepDBService:
    """Service for database operations."""

    def get_dataset_by_file_id(self, table_name: str, file_id: str) -> Optional[Dict[str, Any]]:
        """Get dataset information by file ID."""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                sql = f"""
                    SELECT fi.id, fi.file_type, d.dataset_preprocessed
                    FROM {table_name} d
                    INNER JOIN file_item_version fiv ON fiv.version_id = d.latest_version
                    INNER JOIN file_item fi ON fi.id = fiv.file_item_id
                    WHERE fi.id = %s AND fi.file_type IN ('csv', 'tsv', 'xls', 'xlsx', 'sav')
                """
                
                cursor.execute(sql, (file_id,))
                result = cursor.fetchone()
                
                return dict(result) if result else None
                
        except Exception as e:
            logger.error(f"Error getting dataset by file_id {file_id}: {e}")
            raise
    
    def get_datasets_by_status(self, table_name: str, status: int = 1) -> List[Dict[str, Any]]:
        """Get all datasets with specific preprocessing status."""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                sql = f"""
                    SELECT fi.id, fi.file_type
                    FROM {table_name} d
                    INNER JOIN file_item_version fiv ON fiv.version_id = d.latest_version
                    INNER JOIN file_item fi ON fi.id = fiv.file_item_id
                    WHERE d.dataset_preprocessed = %s 
                    AND fi.file_type IN ('csv', 'tsv', 'xls', 'xlsx', 'sav')
                """
                
                cursor.execute(sql, (status,))
                results = cursor.fetchall()
                
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Error getting datasets by status: {e}")
            raise
    
    def update_preprocessing_status(
        self, 
        table_name: str, 
        file_id: str, 
        status: Optional[int]
    ):
        """Update the preprocessing status for a dataset."""
        status_map = {
            0: 'dataset_preprocessed = 0',
            1: 'dataset_preprocessed = 1',
            None: 'dataset_preprocessed = NULL'
        }
        
        if status not in status_map:
            logger.warning(f"Invalid status: {status}")
            return
        
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                sql = f"""
                    UPDATE {table_name}
                    SET {status_map[status]}
                    WHERE datasetid = (
                        SELECT datasetid
                        FROM {table_name} d
                        INNER JOIN file_item_version fiv ON fiv.version_id = d.latest_version
                        INNER JOIN file_item fi ON fi.id = fiv.file_item_id
                        WHERE fiv.file_item_id = %s
                    )
                """
                
                cursor.execute(sql, (file_id,))
                logger.info(f"Updated status for file_id {file_id} to {status}")
                
        except Exception as e:
            logger.error(f"Error updating status for file_id {file_id}: {e}")
            raise
    
    def get_dataset_status(self, table_name: str, file_id: str) -> Optional[int]:
        """Get the preprocessing status of a dataset."""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                sql = f"""
                    SELECT d.dataset_preprocessed
                    FROM {table_name} d
                    INNER JOIN file_item_version fiv ON fiv.version_id = d.latest_version
                    INNER JOIN file_item fi ON fi.id = fiv.file_item_id
                    WHERE fi.id = %s
                """
                
                cursor.execute(sql, (file_id,))
                result = cursor.fetchone()
                
                return result[0] if result else None
                
        except Exception as e:
            logger.error(f"Error getting status for file_id {file_id}: {e}")
            raise
    
    def list_datasets(
        self, 
        table_name: str, 
        status: Optional[int] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List datasets with optional filtering."""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                sql = f"""
                    SELECT fi.id, fi.file_type, d.dataset_preprocessed
                    FROM {table_name} d
                    INNER JOIN file_item_version fiv ON fiv.version_id = d.latest_version
                    INNER JOIN file_item fi ON fi.id = fiv.file_item_id
                    WHERE fi.file_type IN ('csv', 'tsv', 'xls', 'xlsx', 'sav')
                """
                
                params = []
                if status is not None:
                    sql += " AND d.dataset_preprocessed = %s"
                    params.append(status)
                
                sql += " ORDER BY fi.id LIMIT %s OFFSET %s"
                params.extend([limit, offset])
                
                cursor.execute(sql, tuple(params))
                results = cursor.fetchall()
                
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Error listing datasets: {e}")
            raise

    # Feature Group methods
    def get_pending_feature_groups(self) -> List[Tuple[str, bool]]:
        """Get all pending feature groups (status = 1)."""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                sql = """
                    SELECT table_name, online
                    FROM feature_group
                    WHERE dataprep_status = 1 AND table_name IS NOT NULL
                """
                
                cursor.execute(sql)
                results = cursor.fetchall()
                
                return results
                
        except Exception as e:
            logger.error(f"Error getting pending feature groups: {e}")
            raise
    
    def update_feature_group_status(self, table_name: str, status: int):
        """Update the dataprep status for a feature group."""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                sql = """
                    UPDATE feature_group
                    SET dataprep_status = %s
                    WHERE table_name = %s
                """
                
                cursor.execute(sql, (status, table_name))
                logger.info(f"Updated feature group status for {table_name} to {status}")
                
        except Exception as e:
            logger.error(f"Error updating feature group status for {table_name}: {e}")
            raise
    
    def get_feature_group_status(self, table_name: str) -> Optional[int]:
        """Get the dataprep status of a feature group."""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                sql = """
                    SELECT dataprep_status
                    FROM feature_group
                    WHERE table_name = %s
                """
                
                cursor.execute(sql, (table_name,))
                result = cursor.fetchone()
                
                return result[0] if result else None
                
        except Exception as e:
            logger.error(f"Error getting feature group status for {table_name}: {e}")
            raise
    
    def list_feature_groups(
        self, 
        status: Optional[int] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List feature groups with optional filtering."""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                sql = "SELECT table_name, online, dataprep_status FROM feature_group WHERE 1=1"
                
                params = []
                if status is not None:
                    sql += " AND dataprep_status = %s"
                    params.append(status)
                
                sql += " ORDER BY table_name LIMIT %s OFFSET %s"
                params.extend([limit, offset])
                
                cursor.execute(sql, tuple(params))
                results = cursor.fetchall()
                
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Error listing feature groups: {e}")
            raise
    
    # Training Dataset methods
    def get_pending_training_datasets(
        self, 
        dataset_format: Optional[str] = None
    ) -> List[Tuple[int, str, str]]:
        """Get all pending training datasets (status = 1)."""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                sql = """
                    SELECT id, path, dataset_format
                    FROM training_dataset
                    WHERE dataprep_status = 1
                """
                
                params = []
                if dataset_format:
                    sql += " AND dataset_format = %s"
                    params.append(dataset_format)
                
                cursor.execute(sql, tuple(params))
                results = cursor.fetchall()
                
                return results
                
        except Exception as e:
            logger.error(f"Error getting pending training datasets: {e}")
            raise
    
    def update_training_dataset_status(self, td_id: str, status: int):
        """Update the dataprep status for a training dataset."""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                sql = """
                    UPDATE training_dataset
                    SET dataprep_status = %s
                    WHERE id = %s
                """
                
                cursor.execute(sql, (status, td_id))
                logger.info(f"Updated training dataset status for ID {td_id} to {status}")
                
        except Exception as e:
            logger.error(f"Error updating training dataset status for ID {td_id}: {e}")
            raise
    
    def get_training_dataset_status(self, td_id: str) -> Optional[int]:
        """Get the dataprep status of a training dataset."""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                sql = """
                    SELECT dataprep_status
                    FROM training_dataset
                    WHERE id = %s
                """
                
                cursor.execute(sql, (td_id,))
                result = cursor.fetchone()
                
                return result[0] if result else None
                
        except Exception as e:
            logger.error(f"Error getting training dataset status for ID {td_id}: {e}")
            raise
    
    def list_training_datasets(
        self, 
        status: Optional[int] = None,
        dataset_format: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List training datasets with optional filtering."""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                sql = """
                    SELECT id, path, dataset_format, dataprep_status 
                    FROM training_dataset 
                    WHERE 1=1
                """
                
                params = []
                if status is not None:
                    sql += " AND dataprep_status = %s"
                    params.append(status)
                
                if dataset_format:
                    sql += " AND dataset_format = %s"
                    params.append(dataset_format)
                
                sql += " ORDER BY id LIMIT %s OFFSET %s"
                params.extend([limit, offset])
                
                cursor.execute(sql, tuple(params))
                results = cursor.fetchall()
                
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Error listing training datasets: {e}")
            raise