import numpy as np
import os
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utils.image_utils import load_images_from_folder, images_to_vectors, normalize_image_for_display

class FaceRecognitionPCA:
    def __init__(self, n_components=50, target_size=(100, 100)):
        """
        Khởi tạo model PCA cho nhận diện khuôn mặt
        
        Args:
            n_components: Số thành phần chính cần giữ lại
            target_size: Kích thước ảnh chuẩn
        """
        self.n_components = n_components
        self.target_size = target_size
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.mean_face = None
        self.eigenfaces = None
        self.training_data_pca = None
        self.training_labels = None
        self.label_names = None
        
    def train(self, data_folder):
        """
        Huấn luyện model PCA với dữ liệu từ thư mục
        
        Args:
            data_folder: Đường dẫn thư mục chứa dữ liệu training
        """
        print("Đang đọc dữ liệu training...")
        
        # Đọc dữ liệu
        images, labels, label_names = load_images_from_folder(data_folder, self.target_size)
        
        if len(images) == 0:
            raise ValueError("Không tìm thấy ảnh nào trong thư mục data!")
        
        print(f"Đã đọc {len(images)} ảnh từ {len(label_names)} người")
        
        # Chuyển ảnh thành vectors và normalize
        image_vectors = images_to_vectors(images.astype(np.float32) / 255.0)
        
        # Tính mean face
        self.mean_face = np.mean(image_vectors, axis=0)
        
        # Trừ mean face
        centered_data = image_vectors - self.mean_face
        
        # Chuẩn hóa dữ liệu
        centered_data = self.scaler.fit_transform(centered_data)
        
        print("Đang thực hiện PCA...")
        
        # Điều chỉnh n_components nếu cần
        max_components = min(len(images), image_vectors.shape[1])
        if self.n_components > max_components:
            self.n_components = max_components - 1
            print(f"Điều chỉnh n_components xuống {self.n_components}")
            self.pca = PCA(n_components=self.n_components)
        
        # Thực hiện PCA
        self.training_data_pca = self.pca.fit_transform(centered_data)
        
        # Lưu thông tin
        self.training_labels = labels
        self.label_names = label_names
        self.eigenfaces = self.pca.components_
        
        # In thông tin
        explained_variance_ratio = np.sum(self.pca.explained_variance_ratio_)
        print(f"PCA hoàn thành!")
        print(f"Số thành phần chính: {self.n_components}")
        print(f"Tỷ lệ phương sai được giữ lại: {explained_variance_ratio:.2%}")
        
        return self
    
    def save_model(self, model_path="models/pca_face_model.pkl"):
        """
        Lưu model đã train
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'pca': self.pca,
            'scaler': self.scaler,
            'mean_face': self.mean_face,
            'eigenfaces': self.eigenfaces,
            'training_data_pca': self.training_data_pca,
            'training_labels': self.training_labels,
            'label_names': self.label_names,
            'target_size': self.target_size,
            'n_components': self.n_components
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model đã được lưu tại: {model_path}")
    
    def visualize_eigenfaces(self, n_faces=9, save_path="models/eigenfaces.png"):
        """
        Hiển thị và lưu eigenfaces
        """
        if self.eigenfaces is None:
            print("Model chưa được train!")
            return
        
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        fig.suptitle('Top 9 Eigenfaces', fontsize=16)
        
        for i in range(min(n_faces, len(self.eigenfaces))):
            row, col = i // 3, i % 3
            
            # Chuyển eigenface về dạng ảnh
            eigenface = self.eigenfaces[i].reshape(self.target_size)
            eigenface_normalized = normalize_image_for_display(eigenface)
            
            axes[row, col].imshow(eigenface_normalized, cmap='gray')
            axes[row, col].set_title(f'Eigenface {i+1}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Eigenfaces đã được lưu tại: {save_path}")

def main():
    """
    Hàm main để chạy training
    """
    # Kiểm tra thư mục data
    data_folder = "data"
    if not os.path.exists(data_folder):
        print("Tạo thư mục data mẫu...")
        os.makedirs(data_folder)
        print("Vui lòng thêm ảnh vào thư mục data/ theo cấu trúc:")
        print("data/")
        print("  ├── person1/")
        print("  │   ├── img1.jpg")
        print("  │   └── img2.jpg")
        print("  └── person2/")
        print("      ├── img1.jpg")
        print("      └── img2.jpg")
        return
    
    # Kiểm tra có dữ liệu không
    if not any(os.path.isdir(os.path.join(data_folder, item)) 
               for item in os.listdir(data_folder)):
        print("Không tìm thấy thư mục con nào trong data/")
        print("Vui lòng tạo thư mục cho mỗi người và thêm ảnh vào!")
        return
    
    try:
        # Khởi tạo và train model với face detection
        print("🎯 Sử dụng Face Detection để cắt khuôn mặt...")
        model = FaceRecognitionPCA(n_components=15, target_size=(100, 100))
        model.train(data_folder)
        
        # Lưu model
        model.save_model()
        
        # Hiển thị eigenfaces
        model.visualize_eigenfaces()
        
        print("\n✅ Training hoàn thành với Face Detection!")
        print("Model đã được cải tiến để tập trung vào khuôn mặt, giảm ảnh hưởng của background")
        print("Bạn có thể chạy demo bằng lệnh: streamlit run app_demo.py")
        
    except Exception as e:
        print(f"Lỗi trong quá trình training: {e}")

if __name__ == "__main__":
    main()