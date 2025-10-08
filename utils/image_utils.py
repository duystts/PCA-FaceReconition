import cv2
import numpy as np
import os
from PIL import Image

def load_images_from_folder(folder_path, target_size=(100, 100)):
    """
    Đọc tất cả ảnh từ thư mục và chuyển về kích thước chuẩn
    
    Args:
        folder_path: Đường dẫn thư mục chứa ảnh
        target_size: Kích thước mục tiêu (width, height)
    
    Returns:
        images: List các ảnh đã được resize
        labels: List nhãn tương ứng
        label_names: Dict mapping từ label số sang tên
    """
    images = []
    labels = []
    label_names = {}
    current_label = 0
    
    # Duyệt qua từng thư mục con (mỗi thư mục = 1 người)
    for person_name in os.listdir(folder_path):
        person_path = os.path.join(folder_path, person_name)
        
        if not os.path.isdir(person_path):
            continue
            
        label_names[current_label] = person_name
        
        # Đọc tất cả ảnh trong thư mục của người này
        for filename in os.listdir(person_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(person_path, filename)
                
                # Đọc và xử lý ảnh
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize về kích thước chuẩn
                    img_resized = cv2.resize(img, target_size)
                    images.append(img_resized)
                    labels.append(current_label)
        
        current_label += 1
    
    return np.array(images), np.array(labels), label_names

def preprocess_image(image, target_size=(100, 100)):
    """
    Tiền xử lý ảnh: chuyển grayscale, resize, normalize
    
    Args:
        image: Ảnh input (có thể là BGR hoặc RGB)
        target_size: Kích thước mục tiêu
    
    Returns:
        Ảnh đã được tiền xử lý
    """
    # Chuyển sang grayscale nếu cần
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize
    image = cv2.resize(image, target_size)
    
    # Normalize về [0, 1]
    image = image.astype(np.float32) / 255.0
    
    return image

def images_to_vectors(images):
    """
    Chuyển đổi mảng ảnh 2D thành vectors 1D
    
    Args:
        images: Mảng ảnh shape (n_samples, height, width)
    
    Returns:
        Mảng vectors shape (n_samples, height*width)
    """
    n_samples = images.shape[0]
    return images.reshape(n_samples, -1)

def vectors_to_images(vectors, image_shape):
    """
    Chuyển đổi vectors 1D thành ảnh 2D
    
    Args:
        vectors: Mảng vectors shape (n_samples, height*width)
        image_shape: Tuple (height, width)
    
    Returns:
        Mảng ảnh shape (n_samples, height, width)
    """
    return vectors.reshape(-1, image_shape[0], image_shape[1])

def normalize_image_for_display(image):
    """
    Chuẩn hóa ảnh để hiển thị (0-255)
    """
    image = image.copy()
    image = (image - image.min()) / (image.max() - image.min())
    return (image * 255).astype(np.uint8)