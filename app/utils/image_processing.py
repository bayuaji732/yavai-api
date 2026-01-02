import cv2
import dlib
import numpy as np
from typing import List, Tuple, Optional

class ImageProcessor:
    
    @staticmethod
    def load_image(file_path: str) -> Optional[np.ndarray]:
        """Load image from file path"""
        try:
            img = cv2.imread(file_path)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return None
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    @staticmethod
    def detect_faces_dlib(image: np.ndarray) -> List:
        """Detect faces using dlib"""
        detector = dlib.get_frontal_face_detector()
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return detector(gray, 1)
    
    @staticmethod
    def detect_faces_opencv(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV"""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        return faces.tolist() if len(faces) > 0 else []
    
    @staticmethod
    def blur_region(image: np.ndarray, bbox: Tuple[int, int, int, int], blur_strength: int = 25) -> np.ndarray:
        """Blur a region of the image"""
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        blurred_roi = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
        image[y1:y2, x1:x2] = blurred_roi
        return image