import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from recognize import FaceRecognizer
from utils.image_utils import normalize_image_for_display

# Cấu hình trang
st.set_page_config(
    page_title="Demo PCA Face Recognition",
    page_icon="👤",
    layout="wide"
)

# CSS để làm đẹp giao diện
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_recognizer():
    """Load model PCA (cached)"""
    try:
        return FaceRecognizer("models/pca_face_model.pkl")
    except:
        return None

def plot_similarity_chart(results):
    """Tạo biểu đồ thanh độ tương đồng"""
    names = [result['person_name'] for result in results]
    similarities = [result['similarity'] for result in results]
    
    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=similarities,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(names)],
            text=[f'{sim:.1f}%' for sim in similarities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Mức độ tương đồng với Top 3 kết quả",
        xaxis_title="Người",
        yaxis_title="Độ tương đồng (%)",
        yaxis=dict(range=[0, 100]),
        height=400
    )
    
    return fig

def display_eigenfaces(recognizer):
    """Hiển thị eigenfaces"""
    eigenfaces = recognizer.get_eigenfaces(9)
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('Top 9 Eigenfaces', fontsize=16)
    
    for i, eigenface in enumerate(eigenfaces):
        row, col = i // 3, i % 3
        eigenface_normalized = normalize_image_for_display(eigenface)
        
        axes[row, col].imshow(eigenface_normalized, cmap='gray')
        axes[row, col].set_title(f'Eigenface {i+1}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🎭 Demo PCA Face Recognition</h1>', unsafe_allow_html=True)
    
    # Sidebar thông tin
    with st.sidebar:
        st.markdown("### 📋 Thông tin dự án")
        st.markdown("""
        **Đồ án:** Ứng dụng PCA trong nhận diện khuôn mặt
        
        **Nguyên lý:**
        - Sử dụng PCA để giảm chiều dữ liệu ảnh
        - Tạo ra các Eigenfaces (khuôn mặt cơ sở)
        - Nhận diện bằng cách so sánh trong không gian PCA
        
        **Quy trình:**
        1. Ảnh → Vector
        2. Chuẩn hóa dữ liệu  
        3. Áp dụng PCA
        4. So sánh khoảng cách Euclidean
        """)
        
        st.markdown("### ⚙️ Cài đặt")
        show_eigenfaces = st.checkbox("Hiển thị Eigenfaces", value=False)
        show_reconstruction = st.checkbox("Hiển thị ảnh tái tạo", value=True)
    
    # Load model
    recognizer = load_recognizer()
    
    if recognizer is None:
        st.error("❌ Không thể load model PCA!")
        st.markdown("""
        **Hướng dẫn:**
        1. Tạo thư mục `data/` và thêm ảnh theo cấu trúc:
           ```
           data/
           ├── person1/
           │   ├── img1.jpg
           │   └── img2.jpg
           └── person2/
               ├── img1.jpg
               └── img2.jpg
           ```
        2. Chạy lệnh: `python train_model.py`
        3. Refresh trang này
        """)
        return
    
    st.success("✅ Model PCA đã được load thành công!")
    
    # Phần upload ảnh
    st.markdown('<h2 class="section-header">📤 Upload ảnh để nhận diện</h2>', unsafe_allow_html=True)
    
    # Tạo 2 cột cho upload options
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Chọn ảnh từ máy tính",
            type=['png', 'jpg', 'jpeg'],
            help="Upload ảnh khuôn mặt để nhận diện"
        )
    
    with col2:
        use_camera = st.button("📷 Chụp ảnh từ webcam", help="Sử dụng camera để chụp ảnh")
    
    # Xử lý ảnh
    input_image = None
    
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
        input_image = np.array(input_image)
    
    elif use_camera:
        # Placeholder cho camera (cần implement thêm)
        st.info("Tính năng camera đang được phát triển. Vui lòng sử dụng upload file.")
    
    # Nếu có ảnh input
    if input_image is not None:
        st.markdown('<h2 class="section-header">🔍 Kết quả nhận diện</h2>', unsafe_allow_html=True)
        
        # Hiển thị ảnh gốc
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("**Ảnh gốc**")
            st.image(input_image, use_column_width=True)
        
        # Thực hiện nhận diện
        try:
            with st.spinner("Đang nhận diện..."):
                result = recognizer.recognize(input_image, top_k=3)
            
            # Hiển thị ảnh đã xử lý
            with col2:
                st.markdown("**Ảnh đã xử lý**")
                processed_img = (result['processed_image'] * 255).astype(np.uint8)
                st.image(processed_img, use_column_width=True, clamp=True)
            
            # Hiển thị ảnh tái tạo từ PCA
            if show_reconstruction:
                with col3:
                    st.markdown("**Ảnh tái tạo từ PCA**")
                    reconstructed = recognizer.reconstruct_from_pca(result['pca_representation'])
                    reconstructed_display = (reconstructed * 255).astype(np.uint8)
                    st.image(reconstructed_display, use_column_width=True, clamp=True)
            
            # Kết quả nhận diện
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown(f"### 🎯 Kết quả: **{result['predicted_person']}**")
            st.markdown(f"**Độ tin cậy:** {result['confidence']:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Biểu đồ độ tương đồng
            st.markdown("### 📊 Biểu đồ độ tương đồng")
            similarity_chart = plot_similarity_chart(result['top_matches'])
            st.plotly_chart(similarity_chart, use_container_width=True)
            
            # Bảng chi tiết top 3
            st.markdown("### 📋 Chi tiết Top 3 kết quả")
            
            # Tạo DataFrame cho bảng
            import pandas as pd
            df_results = pd.DataFrame([
                {
                    'Thứ hạng': match['rank'],
                    'Tên người': match['person_name'],
                    'Độ tương đồng (%)': f"{match['similarity']:.1f}%",
                    'Khoảng cách': f"{match['distance']:.3f}"
                }
                for match in result['top_matches']
            ])
            
            st.dataframe(df_results, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Lỗi khi nhận diện: {str(e)}")
    
    # Hiển thị Eigenfaces
    if show_eigenfaces:
        st.markdown('<h2 class="section-header">👻 Eigenfaces</h2>', unsafe_allow_html=True)
        st.markdown("""
        **Eigenfaces** là các thành phần chính (principal components) của tập ảnh khuôn mặt.
        Mỗi eigenface đại diện cho một "hướng biến thiên" quan trọng trong dữ liệu khuôn mặt.
        """)
        
        try:
            eigenfaces_fig = display_eigenfaces(recognizer)
            st.pyplot(eigenfaces_fig)
        except Exception as e:
            st.error(f"Lỗi khi hiển thị eigenfaces: {str(e)}")
    
    # Phần giải thích thuật toán
    with st.expander("🧠 Giải thích thuật toán PCA"):
        st.markdown("""
        ### Nguyên lý hoạt động:
        
        1. **Thu thập dữ liệu:** Tập ảnh khuôn mặt được chuyển thành vectors
        2. **Tính mean face:** Khuôn mặt trung bình của tập dữ liệu
        3. **Tính ma trận hiệp phương sai:** Đo lường sự biến thiên giữa các pixel
        4. **Tìm eigenvectors:** Các hướng biến thiên chính (eigenfaces)
        5. **Giảm chiều:** Chỉ giữ lại k eigenfaces quan trọng nhất
        6. **Nhận diện:** So sánh khoảng cách trong không gian PCA
        
        ### Ưu điểm:
        - Giảm chiều dữ liệu hiệu quả
        - Loại bỏ nhiễu
        - Tính toán nhanh
        
        ### Nhược điểm:
        - Nhạy cảm với ánh sáng
        - Yêu cầu ảnh cùng kích thước
        - Không tốt với tư thế khác nhau
        """)

if __name__ == "__main__":
    main()