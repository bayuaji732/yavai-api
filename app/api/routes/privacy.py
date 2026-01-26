import os
import json
import requests
import numpy as np
import logging
from fastapi import APIRouter, HTTPException
from app.db.postgres import get_db_connection
from app.models.requests import PrivacyDetectionRequest
from app.services.privacy_service import PrivacyService

# Setup Logger
logger = logging.getLogger(__name__)

router = APIRouter()
service = PrivacyService()

@router.post("/")
async def privacy_detection(request: PrivacyDetectionRequest):
    if not request.token:
        logger.warning("Privacy detection request missing authentication token")
        raise HTTPException(status_code=401, detail="Authentication token is required")
    
    logger.info(f"Received privacy detection request for file_id: {request.file_item_id}")
    temp_file_path = None
    
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute('''
                SELECT privacy_detection_result, privacy_detection_type
                FROM file_item
                WHERE id = %s
            ''', (request.file_item_id,))
            result = cur.fetchone()
            
            cached_result = result[0] if result else None
            cached_type = result[1] if result else None
            
            if cached_result and cached_result != "null":
                logger.info(f"Cache hit for file_id: {request.file_item_id}")
                detection_type = cached_type or "unknown"
                data = []
                status_message = "Unknown cached result"
                
                if cached_result in ["processing_error", "no_eyes_found", "no_pii_found", "unsupported_format"]:
                    message_map = {
                        "processing_error": "Previous processing resulted in an error (from cache)",
                        "no_eyes_found": "No faces or eye landmarks detected previously (from cache)",
                        "no_pii_found": "No PII columns detected previously (from cache)",
                        "unsupported_format": "Unsupported file format (from cache)"
                    }
                    status_message = message_map.get(cached_result, "Unknown cached result")
                else:
                    try:
                        data = json.loads(cached_result)
                        if detection_type == "eye_detection":
                            status_message = f"Retrieved {len(data)} eye region(s) from cache"
                        elif detection_type == "pii_detection":
                            status_message = f"Retrieved {len(data)} PII column(s) from cache"
                        else:
                            status_message = "Privacy detection results retrieved from cache"
                    except json.JSONDecodeError:
                        logger.error(f"Malformed cache for file_item_id {request.file_item_id}: {cached_result[:100]}")
                        status_message = "Error: Cached data is corrupt"
                
                return {
                    "statusCode": 200,
                    "status": "OK",
                    "message": status_message,
                    "detection_type": detection_type,
                    "data": data
                }
            
            # No cache - process file
            logger.info(f"No cache found for {request.file_item_id}. Initiating processing...")
            cur.execute('SELECT file_type FROM file_item WHERE id = %s', (request.file_item_id,))
            file_type_result = cur.fetchone()
            
            if not file_type_result:
                logger.warning(f"File ID {request.file_item_id} not found in database")
                raise HTTPException(status_code=404, detail="File not found")
            
            file_type = file_type_result[0].lower() if file_type_result[0] else 'unknown'
            
            processing_category, detection_type, status_message, data = service.process_file(
                file_item_id=request.file_item_id,
                file_type=file_type,
                token=request.token
            )
            
            data_serializable = convert_to_serializable(data)
            
            # Improved status logic for database storage
            if data_serializable:
                db_result_value = json.dumps(data_serializable)
            else:
                # Use a cleaner status code for DB if possible, otherwise fallback to message content
                if "No faces" in status_message:
                    db_result_value = "no_eyes_found"
                elif "No PII" in status_message:
                    db_result_value = "no_pii_found"
                elif "Unsupported" in status_message:
                    db_result_value = "unsupported_format"
                elif "Error" in status_message or "failed" in status_message.lower():
                    db_result_value = "processing_error"
                else:
                    db_result_value = status_message.split()[0].lower().replace(" ", "_")
            
            cur.execute('''
                INSERT INTO file_item (id, privacy_detection_result, privacy_detection_type)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    privacy_detection_result = EXCLUDED.privacy_detection_result,
                    privacy_detection_type = EXCLUDED.privacy_detection_type
            ''', (request.file_item_id, db_result_value, detection_type))
            
            logger.info(f"Processing complete for {request.file_item_id}. Status: {status_message}")
            
            return {
                "statusCode": 200,
                "status": "OK",
                "message": status_message,
                "detection_type": detection_type,
                "data": data
            }
            
    except requests.exceptions.RequestException as e:
        logger.error(f"External service request failed: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to retrieve file from external service: {str(e)}")
    except Exception as e:
        logger.exception(f"Unhandled exception in privacy endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass

def convert_to_serializable(obj):
    """Convert numpy types to native Python types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    return obj