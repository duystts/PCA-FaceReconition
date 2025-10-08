import numpy as np
import joblib
import cv2
from sklearn.metrics.pairwise import euclidean_distances
from utils.image_utils import preprocess_image, images_to_vectors

class FaceRecognizer:
    def __init__(self, model_path="models/pca_face_model.pkl"):
        """
        Khởi tạo face recognizer từ model đã train
        
        Args:
            model_path: Đường dẫn đến model đã lưu
        """
        self.model_path = model_path
        self.model_data = None
        self.load_model()
    
    def load_model(self):
        """
        Load model đã train từ file
        """
        try:
            self.model_data = joblib.load(self.model_path)
            print("Model đã được load thành công!")
        except FileNotFoundError:
            print(f"Không tìm thấy model tại {self.model_path}")
            print("Vui lòng chạy train_model.py trước!")
            raise
    
    def preprocess_for_recognition(self, image):
        """
        Tiền xử lý ảnh để nhận diện
        
        Args:
            image: Ảnh input (numpy array hoặc đường dẫn file)
        
        Returns:
            Ảnh đã được tiền xử lý
        """
        # Nếu input là đường dẫn file
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Không thể đọc ảnh từ {image}")
        
        # Tiền xử lý ảnh
        processed_image = preprocess_image(image, self.model_data['target_size'])
        
        return processed_image
    
    def recognize(self, image, top_k=3):
        """
        Nhận diện khuôn mặt trong ảnh
        
        Args:
            image: Ảnh input
            top_k: Số kết quả top gần nhất cần trả về
        
        Returns:
            Dict chứa kết quả nhận diện
        """
        if self.model_data is None:
            raise ValueError("Model chưa được load!")
        
        # Tiền xử lý ảnh
        processed_image = self.preprocess_for_recognition(image)
        
        # Chuyển thành vector
        image_vector = images_to_vectors(processed_image.reshape(1, *processed_image.shape))
        
        # Trừ mean face
        centered_vector = image_vector - self.model_data['mean_face']
        
        # Chuẩn hóa
        scaled_vector = self.model_data['scaler'].transform(centered_vector)
        
        # Chiếu lên không gian PCA
        pca_vector = self.model_data['pca'].transform(scaled_vector)
        
        # Tính khoảng cách với tất cả ảnh training
        distances = euclidean_distances(pca_vector, self.model_data['training_data_pca'])[0]
        
        # Tìm top k gần nhất
        nearest_indices = np.argsort(distances)[:top_k]
        nearest_distances = distances[nearest_indices]
        nearest_labels = self.model_data['training_labels'][nearest_indices]
        
        # Chuyển đổi khoảng cách thành độ tương đồng (0-100%)
        max_distance = np.max(distances)
        similarities = (1 - nearest_distances / max_distance) * 100
        
        # Tạo kết quả
        results = []
        for i, (idx, distance, label, similarity) in enumerate(zip(
            nearest_indices, nearest_distances, nearest_labels, similarities)):
            
            person_name = self.model_data['label_names'][label]
            results.append({
                'rank': i + 1,
                'person_name': person_name,
                'label': label,
                'distance': distance,
                'similarity': similarity,
                'training_index': idx
            })
        
        # Dự đoán cuối cùng (người gần nhất)
        predicted_person = results[0]['person_name']
        confidence = results[0]['similarity']
        
        return {
            'predicted_person': predicted_person,
            'confidence': confidence,
            'top_matches': results,
            'pca_representation': pca_vector[0],
            'processed_image': processed_image
        }
    
    def reconstruct_from_pca(self, pca_vector):
        """
        Tái tạo ảnh từ biểu diễn PCA
        
        Args:
            pca_vector: Vector trong không gian PCA
        
        Returns:
            Ảnh được tái tạo
        """
        if self.model_data is None:
            raise ValueError("Model chưa được load!")
        
        # Tái tạo từ PCA
        reconstructed_scaled = self.model_data['pca'].inverse_transform(pca_vector.reshape(1, -1))
        
        # Inverse scaling
        reconstructed_centered = self.model_data['scaler'].inverse_transform(reconstructed_scaled)
        
        # Thêm mean face
        reconstructed_vector = reconstructed_centered + self.model_data['mean_face']
        
        # Chuyển về dạng ảnh
        reconstructed_image = reconstructed_vector.reshape(self.model_data['target_size'])
        
        # Clamp về [0, 1]
        reconstructed_image = np.clip(reconstructed_image, 0, 1)
        
        return reconstructed_image
    
    def get_eigenfaces(self, n_faces=9):
        """
        Lấy eigenfaces để hiển thị
        
        Args:
            n_faces: Số eigenfaces cần lấy
        
        Returns:
            List các eigenfaces
        """
        if self.model_data is None:
            raise ValueError("Model chưa được load!")
        
        eigenfaces = []
        for i in range(min(n_faces, len(self.model_data['eigenfaces']))):
            eigenface = self.model_data['eigenfaces'][i].reshape(self.model_data['target_size'])
            eigenfaces.append(eigenface)
        
        return eigenfaces

def main():
    """
    Hàm main để test nhận diện
    """
    import os
    
    # Kiểm tra model có tồn tại không
    model_path = "models/pca_face_model.pkl"
    if not os.path.exists(model_path):
        print("Model chưa được train!")
        print("Vui lòng chạy: python train_model.py")
        return
    
    # Khởi tạo recognizer
    recognizer = FaceRecognizer(model_path)
    
    # Test với ảnh mẫu (nếu có)
    test_image_path = input("Nhập đường dẫn ảnh cần nhận diện (hoặc Enter để bỏ qua): ").strip()
    
    if test_image_path and os.path.exists(test_image_path):
        try:
            result = recognizer.recognize(test_image_path)
            
            print(f"\nKết quả nhận diện:")
            print(f"Người được dự đoán: {result['predicted_person']}")
            print(f"Độ tin cậy: {result['confidence']:.1f}%")
            print(f"\nTop 3 kết quả gần nhất:")
            
            for match in result['top_matches']:
                print(f"  {match['rank']}. {match['person_name']} - {match['similarity']:.1f}%")
                
        except Exception as e:
            print(f"Lỗi khi nhận diện: {e}")
    else:
        print("Không có ảnh test. Chạy demo Streamlit để test: streamlit run app_demo.py")

if __name__ == "__main__":
    main()