import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

def create_synthetic_face(size=(100, 100), person_id=0, variation=0):
    """
    Tạo khuôn mặt tổng hợp đơn giản để demo
    
    Args:
        size: Kích thước ảnh
        person_id: ID của người (ảnh hưởng đến đặc điểm khuôn mặt)
        variation: Biến thể của cùng một người
    
    Returns:
        Ảnh khuôn mặt tổng hợp
    """
    width, height = size
    
    # Tạo ảnh trắng
    img = Image.new('L', size, color=240)
    draw = ImageDraw.Draw(img)
    
    # Tham số cho từng người (khác nhau một chút)
    np.random.seed(person_id * 100 + variation)
    
    # Khuôn mặt oval
    face_margin = 10 + person_id * 2
    face_coords = [
        face_margin, face_margin + person_id * 3,
        width - face_margin, height - face_margin - person_id * 2
    ]
    draw.ellipse(face_coords, fill=200 - person_id * 10)
    
    # Mắt
    eye_y = height // 3 + person_id * 2
    eye_size = 8 + person_id
    
    # Mắt trái
    left_eye_x = width // 3 - person_id
    draw.ellipse([
        left_eye_x - eye_size//2, eye_y - eye_size//2,
        left_eye_x + eye_size//2, eye_y + eye_size//2
    ], fill=50)
    
    # Mắt phải
    right_eye_x = 2 * width // 3 + person_id
    draw.ellipse([
        right_eye_x - eye_size//2, eye_y - eye_size//2,
        right_eye_x + eye_size//2, eye_y + eye_size//2
    ], fill=50)
    
    # Mũi
    nose_x = width // 2 + np.random.randint(-3, 4)
    nose_y = height // 2 + person_id
    nose_size = 3 + person_id // 2
    draw.ellipse([
        nose_x - nose_size, nose_y - nose_size//2,
        nose_x + nose_size, nose_y + nose_size//2
    ], fill=150 - person_id * 5)
    
    # Miệng
    mouth_y = 2 * height // 3 + person_id * 2
    mouth_width = 15 + person_id * 2
    mouth_height = 4 + person_id // 2
    draw.ellipse([
        width//2 - mouth_width//2, mouth_y - mouth_height//2,
        width//2 + mouth_width//2, mouth_y + mouth_height//2
    ], fill=100)
    
    # Thêm một chút nhiễu
    img_array = np.array(img)
    noise = np.random.normal(0, 5, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    return img_array

def create_sample_dataset(data_folder="data", n_people=5, n_images_per_person=4):
    """
    Tạo dataset mẫu với khuôn mặt tổng hợp
    
    Args:
        data_folder: Thư mục lưu dữ liệu
        n_people: Số người
        n_images_per_person: Số ảnh mỗi người
    """
    print(f"Tạo dataset mẫu với {n_people} người, mỗi người {n_images_per_person} ảnh...")
    
    # Tạo thư mục data nếu chưa có
    os.makedirs(data_folder, exist_ok=True)
    
    for person_id in range(n_people):
        person_name = f"person_{person_id + 1}"
        person_folder = os.path.join(data_folder, person_name)
        os.makedirs(person_folder, exist_ok=True)
        
        print(f"Tạo ảnh cho {person_name}...")
        
        for img_id in range(n_images_per_person):
            # Tạo ảnh tổng hợp
            face_img = create_synthetic_face(
                size=(100, 100),
                person_id=person_id,
                variation=img_id
            )
            
            # Lưu ảnh
            img_path = os.path.join(person_folder, f"img_{img_id + 1}.png")
            cv2.imwrite(img_path, face_img)
    
    print(f"✅ Đã tạo xong dataset mẫu tại thư mục '{data_folder}'")
    print(f"Tổng cộng: {n_people * n_images_per_person} ảnh")
    print("\nBạn có thể:")
    print("1. Chạy 'python train_model.py' để huấn luyện model")
    print("2. Chạy 'streamlit run app_demo.py' để xem demo")
    print("\nLưu ý: Đây là dữ liệu tổng hợp để demo. Để có kết quả tốt hơn,")
    print("hãy thay thế bằng ảnh khuôn mặt thật trong thư mục data/")

def main():
    """
    Hàm main để tạo dữ liệu mẫu
    """
    print("🎭 Tạo dữ liệu mẫu cho demo PCA Face Recognition")
    print("=" * 50)
    
    # Kiểm tra xem đã có dữ liệu chưa
    data_folder = "data"
    if os.path.exists(data_folder) and os.listdir(data_folder):
        response = input(f"Thư mục '{data_folder}' đã có dữ liệu. Ghi đè? (y/N): ").strip().lower()
        if response != 'y':
            print("Hủy tạo dữ liệu mẫu.")
            return
    
    # Tạo dataset mẫu
    create_sample_dataset(
        data_folder=data_folder,
        n_people=5,
        n_images_per_person=4
    )

if __name__ == "__main__":
    main()