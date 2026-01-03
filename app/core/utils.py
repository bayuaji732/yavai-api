import os
import logging
from logging.handlers import RotatingFileHandler
from core import config

def setup_logger(name: str) -> logging.Logger:
    """Set up and return a configured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(config.LOG_FILE_PATH)
    os.makedirs(log_dir, exist_ok=True)
    
    # Rotating file handler
    handler = RotatingFileHandler(
        config.LOG_FILE_PATH,
        maxBytes=config.MAX_LOG_SIZE,
        backupCount=config.BACKUP_COUNT
    )
    handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def ensure_local_dir():
    """Ensure the local directory exists."""
    os.makedirs(config.LOCAL_DIR, exist_ok=True)

def cleanup_file(file_path: str):
    """Remove a file if it exists."""
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            logging.error(f"Error removing file {file_path}: {e}")