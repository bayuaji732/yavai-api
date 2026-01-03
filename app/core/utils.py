import os
import logging
from logging.handlers import RotatingFileHandler
from core import config

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