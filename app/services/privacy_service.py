import os
import tempfile
import requests
import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Optional
from PIL import Image
from pillow_heif import register_heif_opener
import cv2
import dlib
import spacy
from app.core import config

# Setup Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

IMAGE_TYPES = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'heic', 'heif', 'webp'}
DATA_TYPES = {'csv', 'tsv', 'txt', 'xls', 'xlsx', 'sav'}
DEFAULT_PII_THRESHOLD = 0.15
SAMPLE_ROW_LIMIT = 10

class PrivacyService:
    
    def __init__(self):
        self.nlp = self._load_spacy_model()
    
    def _load_spacy_model(self):
        """Load spaCy model for PII detection"""
        SPACY_MODEL = "en_core_web_sm"
        try:
            logger.info(f"Loading spaCy model: {SPACY_MODEL}")
            return spacy.load(SPACY_MODEL)
        except OSError:
            logger.info(f"Model '{SPACY_MODEL}' not found. Attempting download...")
            try:
                spacy.cli.download(SPACY_MODEL)
                return spacy.load(SPACY_MODEL)
            except Exception as e:
                logger.error(f"Failed to load spaCy model '{SPACY_MODEL}'. PII detection disabled. Error: {e}")
                return None
    
    def process_file(self, file_item_id: str, file_type: str, token: str) -> Tuple[str, str, str, List]:
        """Process file for privacy detection"""
        logger.info(f"Starting privacy processing for file_id: {file_item_id} (Type: {file_type})")
        
        # Determine processing category
        if file_type in IMAGE_TYPES:
            processing_category = "image"
            temp_suffix = f".{file_type}"
        elif file_type in DATA_TYPES:
            processing_category = "data"
            temp_suffix = f".{file_type}"
        else:
            logger.warning(f"Unsupported file type: {file_type}")
            return "unknown", "unknown", f"Unsupported file type: {file_type}", []
        
        # Download file
        try:
            temp_file_path = self._download_file(file_item_id, processing_category, token, temp_suffix)
        except Exception as e:
            logger.error(f"Download failed for {file_item_id}: {str(e)}")
            return processing_category, "unknown", f"Download failed: {str(e)}", []
        
        try:
            if processing_category == "image":
                return self._process_image(temp_file_path)
            elif processing_category == "data":
                return self._process_data(temp_file_path)
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logger.debug(f"Cleaned up temp file: {temp_file_path}")
    
    def _download_file(self, file_item_id: str, processing_category: str, token: str, suffix: str) -> str:
        """Download file from external service"""
        headers = {'Authorization': f'Bearer {token}'}
        
        if processing_category == "image":
            url = f"{config.YAVAI_API_BASE_URL}/dataset-management/api/v1/files/{file_item_id}/dataset-preview"
        else:
            url = f"{config.YAVAI_API_BASE_URL}/dataset-management/api/v1/files/{file_item_id}/download"
        
        logger.debug(f"Downloading from URL: {url}")
        response = requests.get(url, stream=True, verify=False, headers=headers)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            temp_file_path = tmp_file.name
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
        
        logger.info(f"File downloaded successfully to {temp_file_path}")
        return temp_file_path
    
    def _process_image(self, file_path: str) -> Tuple[str, str, str, List]:
        """Process image for eye detection"""
        try:
            register_heif_opener()
            with Image.open(file_path) as pil_img:
                pil_img_rgb = pil_img.convert("RGB")
                image_np = np.array(pil_img_rgb, dtype=np.uint8).copy()
                
                coordinates = self._detect_eyes(image_np)
                
                if coordinates is None:
                    msg = "Error: Internal failure during eye detection logic"
                    logger.error(msg)
                    return "image", "eye_detection", msg, []
                elif not coordinates:
                    msg = "Analysis complete: No faces or eye landmarks detected"
                    logger.info(msg)
                    return "image", "eye_detection", msg, []
                else:
                    msg = f"Success: Detected {len(coordinates)} eye region(s)"
                    logger.info(msg)
                    return "image", "eye_detection", msg, coordinates
                    
        except Exception as e:
            logger.exception(f"Critical error processing image file: {e}")
            return "image", "eye_detection", f"Processing Error: {str(e)}", []
    
    def _process_data(self, file_path: str) -> Tuple[str, str, str, List]:
        """Process data file for PII detection"""
        try:
            df = self._read_data_file(file_path)
            
            if df is None:
                msg = "Error: Could not read data file (invalid format or empty)"
                logger.error(msg)
                return "data", "pii_detection", msg, []
            
            pii_columns = self._detect_pii_columns(df)
            
            if not pii_columns:
                return "data", "pii_detection", "Analysis complete: No PII columns detected", []
            else:
                return "data", "pii_detection", f"Success: Detected {len(pii_columns)} PII column(s)", pii_columns
        except Exception as e:
            logger.exception(f"Critical error processing data file: {e}")
            return "data", "pii_detection", f"Processing Error: {str(e)}", []

    def _read_data_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Read data file into pandas DataFrame"""
        if not os.path.exists(file_path):
            logger.error(f"File path does not exist: {file_path}")
            return None
        
        file_extension = os.path.splitext(file_path)[1].lower()
        logger.debug(f"Reading data file with extension: {file_extension}")
        
        try:
            if file_extension == '.csv':
                return pd.read_csv(file_path)
            elif file_extension == '.tsv':
                return pd.read_csv(file_path, sep='\t')
            elif file_extension in ['.txt', '.data', '']:
                for sep in ['\t', ',', r'\s+']:
                    try:
                        return pd.read_csv(file_path, sep=sep)
                    except:
                        continue
                return None
            elif file_extension in ['.xls', '.xlsx']:
                return pd.read_excel(file_path)
            elif file_extension == '.sav':
                import pyreadstat
                df, _ = pyreadstat.read_sav(file_path)
                return df
            else:
                logger.warning(f"Unhandled data file extension: {file_extension}")
                return None
        except Exception as e:
            logger.error(f"Pandas read error: {e}")
            return None

    def _detect_pii_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect PII columns in DataFrame"""
        if df.empty:
            logger.warning("DataFrame is empty, skipping PII detection")
            return []
        if self.nlp is None:
            logger.warning("SpaCy model not loaded, skipping PII detection")
            return []
        
        pii_columns = []
        pii_labels = ["PERSON", "GPE", "LOC", "EMAIL", "PHONE_NUMBER", "SSN", "CREDIT_CARD", "IP_ADDRESS"]
        
        logger.info(f"Scanning {len(df.columns)} columns for PII...")
        
        for column_name in df.columns:
            non_null_series = df[column_name].dropna()
            if len(non_null_series) == 0:
                continue
            
            sample = non_null_series.sample(min(len(non_null_series), SAMPLE_ROW_LIMIT), random_state=42)
            
            pii_count = 0
            total_count = 0
            
            for cell_value in sample:
                text = str(cell_value).strip()
                if not text:
                    total_count += 1
                    continue
                
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ in pii_labels:
                        pii_count += 1
                total_count += 1
            
            if total_count > 0 and (pii_count / total_count) >= DEFAULT_PII_THRESHOLD:
                logger.info(f"PII Detected in column: {column_name}")
                pii_columns.append(column_name)
        
        return pii_columns

    def _detect_eyes(self, image_np: np.ndarray) -> Optional[List[Tuple[int, int, int, int]]]:
        """Detect eye regions in image"""
        try:
            # Handle color channels
            if len(image_np.shape) == 2:
                gray = image_np
            elif len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            else:
                logger.error(f"Invalid image shape for detection: {image_np.shape}")
                return None
            
            # Try dlib first
            logger.info("Attempting eye detection via Dlib...")
            coordinates = self._detect_with_dlib(gray, image_np.shape)
            if coordinates:
                logger.info(f"Dlib detection successful. Found {len(coordinates)} regions.")
                return coordinates
            
            # Fallback to OpenCV
            logger.info("Dlib returned no results or failed. Falling back to OpenCV Haar Cascades...")
            coordinates = self._detect_with_opencv(gray, image_np.shape)
            if coordinates:
                logger.info(f"OpenCV detection successful. Found {len(coordinates)} regions.")
            else:
                logger.info("OpenCV detection returned no results.")
            
            return coordinates
            
        except Exception as e:
            logger.exception(f"Unexpected error in _detect_eyes: {e}")
            return None

    def _detect_with_dlib(self, gray, image_shape) -> List[Tuple[int, int, int, int]]:
        """Detect eyes using dlib with explicit logging"""
        try:
            detector = dlib.get_frontal_face_detector()
            predictor_path = config.SHAPE_PREDICTOR_FULL_PATH
            
            # --- Added Logging for Troubleshooting ---
            logger.debug(f"Looking for Dlib Shape Predictor at: {predictor_path}")
            
            if not os.path.exists(predictor_path):
                logger.error(f"Dlib Shape Predictor file NOT FOUND at: {predictor_path}. Check config and file system.")
                return []
            
            try:
                predictor = dlib.shape_predictor(predictor_path)
                logger.debug("Dlib Shape Predictor loaded successfully.")
            except Exception as e:
                logger.error(f"File exists but Dlib failed to load predictor: {e}")
                return []
            # -----------------------------------------

            faces = detector(gray, 1)
            
            if not faces:
                logger.debug("Dlib: No faces detected in image.")
                return []
            
            coordinates = []
            for face in faces:
                landmarks = predictor(gray, face)
                left_eye = [landmarks.part(n) for n in range(36, 42)]
                right_eye = [landmarks.part(n) for n in range(42, 48)]
                
                all_eye_points = left_eye + right_eye
                if all_eye_points:
                    bbox = self._calculate_bbox(all_eye_points, image_shape)
                    if bbox:
                        coordinates.append(bbox)
            
            return coordinates
        except Exception as e:
            logger.exception(f"Critical error during Dlib detection process: {e}")
            return []

    def _detect_with_opencv(self, gray, image_shape) -> List[Tuple[int, int, int, int]]:
        """Detect eyes using OpenCV Haar Cascades"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                logger.error(f"OpenCV Cascade file missing: {cascade_path}")
                # Try to load without checking path (sometimes internal to cv2)
            
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if face_cascade.empty():
                logger.error("Failed to load OpenCV CascadeClassifier")
                return []
                
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 0:
                return []
            
            coordinates = []
            for (x, y, w, h) in faces:
                eye_coords = self._estimate_eye_region(x, y, w, h, image_shape)
                if eye_coords:
                    coordinates.append(eye_coords)
            
            return coordinates
        except Exception as e:
            logger.error(f"Error in OpenCV detection: {e}")
            return []

    def _calculate_bbox(self, points, image_shape, padding=0.15) -> Optional[Tuple[int, int, int, int]]:
        """Calculate bounding box from points"""
        if not points:
            return None
        
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        width = max_x - min_x
        height = max_y - min_y
        
        pad_x = int(width * padding)
        pad_y = int(height * padding)
        
        x1 = max(0, min_x - pad_x)
        y1 = max(0, min_y - pad_y)
        x2 = min(image_shape[1], max_x + pad_x)
        y2 = min(image_shape[0], max_y + pad_y)
        
        return (int(x1), int(y1), int(x2), int(y2))

    def _estimate_eye_region(self, x, y, w, h, image_shape) -> Tuple[int, int, int, int]:
        """Estimate eye region from face bounding box"""
        eye_y_start = y + int(h * 0.30)
        eye_y_end = y + int(h * 0.45)
        eye_x_start = x + int(w * 0.15)
        eye_x_end = x + int(w * 0.85)
        
        eye_width = eye_x_end - eye_x_start
        eye_height = eye_y_end - eye_y_start
        
        pad_x = int(eye_width * 0.15)
        pad_y = int(eye_height * 0.20)
        
        x1 = max(0, eye_x_start - pad_x)
        y1 = max(0, eye_y_start - pad_y)
        x2 = min(image_shape[1], eye_x_end + pad_x)
        y2 = min(image_shape[0], eye_y_end + pad_y)
        
        return (int(x1), int(y1), int(x2), int(y2))