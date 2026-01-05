import os
import json
import requests
from fastapi import APIRouter, HTTPException, Form
from app.db.postgres import get_db_connection
from app.services.privacy_service import PrivacyService

router = APIRouter()
service = PrivacyService()

@router.post("/{file_item_id}")
async def privacy_detection(file_item_id: str, token: str = Form(...)):
    if not token:
        raise HTTPException(status_code=401, detail="Authentication token is required in form data")
    
    temp_file_path = None
    
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute('''
                SELECT privacy_detection_result, privacy_detection_type
                FROM file_item
                WHERE id = %s
            ''', (file_item_id,))
            result = cur.fetchone()
            
            cached_result = result[0] if result else None
            cached_type = result[1] if result else None
            
            if cached_result and cached_result != "null":
                detection_type = cached_type or "unknown"
                
                if cached_result in ["processing_error", "no_eyes_found", "no_pii_found", "unsupported_format"]:
                    data = []
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
                        raise HTTPException(status_code=400, detail="Cached privacy detection data is malformed")
                
                return {
                    "statusCode": 200,
                    "status": "OK",
                    "message": status_message,
                    "detection_type": detection_type,
                    "data": data
                }
            
            # No cache - process file
            cur.execute('SELECT file_type FROM file_item WHERE id = %s', (file_item_id,))
            file_type_result = cur.fetchone()
            
            if not file_type_result:
                raise HTTPException(status_code=404, detail="File not found")
            
            file_type = file_type_result[0].lower() if file_type_result[0] else 'unknown'
            
            processing_category, detection_type, status_message, data = service.process_file(
                file_item_id=file_item_id,
                file_type=file_type,
                token=token
            )
            
            # Cache result
            db_result_value = json.dumps(data) if data else status_message.split()[0].lower().replace(" ", "_")
            
            cur.execute('''
                INSERT INTO file_item (id, privacy_detection_result, privacy_detection_type)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    privacy_detection_result = EXCLUDED.privacy_detection_result,
                    privacy_detection_type = EXCLUDED.privacy_detection_type
            ''', (file_item_id, db_result_value, detection_type))
            
            return {
                "statusCode": 200,
                "status": "OK",
                "message": status_message,
                "detection_type": detection_type,
                "data": data
            }
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to retrieve file from external service: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass