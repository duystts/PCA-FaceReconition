import cv2
import numpy as np

def detect_and_crop_face(image, face_cascade=None):
    """Phát hiện và cắt khuôn mặt từ ảnh"""
    if face_cascade is None:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        # Lấy khuôn mặt lớn nhất
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        # Mở rộng vùng một chút
        margin = int(0.1 * min(w, h))
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        
        return gray[y:y+h, x:x+w]
    return gray

def normalize_lighting(image):
    """Chuẩn hóa ánh sáng"""
    # Histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

def preprocess_face_image(image_path, target_size=(100, 100)):
    """Tiền xử lý ảnh khuôn mặt hoàn chỉnh"""
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Phát hiện và cắt khuôn mặt
    face = detect_and_crop_face(image)
    if face is None:
        return None
    
    # Chuẩn hóa ánh sáng
    face = normalize_lighting(face)
    
    # Resize về kích thước chuẩn
    face = cv2.resize(face, target_size)
    
    return face