# Ứng dụng PCA trong Nhận diện Khuôn mặt

## Tổng quan dự án

Dự án này minh họa việc ứng dụng thuật toán **Principal Component Analysis (PCA)** trong bài toán nhận diện khuôn mặt, sử dụng khái niệm **Eigenfaces**.

### Nguyên lý PCA và Eigenfaces

**PCA (Principal Component Analysis)** là một kỹ thuật giảm chiều dữ liệu bằng cách:
- Tìm các thành phần chính (principal components) có phương sai lớn nhất
- Chiếu dữ liệu gốc lên không gian con có chiều thấp hơn
- Giữ lại thông tin quan trọng nhất, loại bỏ nhiễu

**Eigenfaces** là các eigenvector của ma trận hiệp phương sai của tập ảnh khuôn mặt:
- Mỗi eigenface đại diện cho một "khuôn mặt cơ sở"
- Bất kỳ khuôn mặt nào cũng có thể biểu diễn bằng tổ hợp tuyến tính của eigenfaces
- Chỉ cần lưu trữ một số eigenface quan trọng nhất

### Quy trình nhận diện

1. **Ảnh gốc** → Chuyển đổi thành vector (flatten)
2. **Vector** → Chuẩn hóa và trừ mean
3. **PCA** → Chiếu lên không gian eigenfaces
4. **Nhận diện** → So sánh khoảng cách Euclidean với tập train

## Cấu trúc dự án

```
Demo-do-an/
├── data/                   # Thư mục chứa ảnh training
├── models/                 # Thư mục lưu model đã train
├── utils/                  # Các hàm tiện ích
├── train_model.py         # Huấn luyện PCA model
├── recognize.py           # Nhận diện ảnh mới
├── app_demo.py           # Giao diện Streamlit demo
├── requirements.txt       # Các thư viện cần thiết
└── README.md             # Tài liệu dự án
```

## Cách sử dụng

1. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Chuẩn bị dữ liệu:**
   - Tạo thư mục con trong `data/` cho mỗi người (ví dụ: `data/person1/`, `data/person2/`)
   - Đặt ảnh khuôn mặt (định dạng .jpg, .png) vào từng thư mục

3. **Huấn luyện model:**
   ```bash
   python train_model.py
   ```

4. **Chạy demo:**
   ```bash
   streamlit run app_demo.py
   ```

## Ưu điểm và Nhược điểm

### Ưu điểm:
- Giảm chiều dữ liệu hiệu quả
- Tính toán nhanh
- Loại bỏ nhiễu tốt
- Dễ hiểu và triển khai

### Nhược điểm:
- Nhạy cảm với điều kiện ánh sáng
- Yêu cầu ảnh cùng kích thước
- Hiệu quả giảm với tư thế khuôn mặt khác nhau
- Không tốt bằng các phương pháp deep learning hiện đại

## Hướng phát triển

- Sử dụng **LDA (Linear Discriminant Analysis)** để cải thiện khả năng phân biệt
- Áp dụng **CNN (Convolutional Neural Networks)** cho độ chính xác cao hơn
- Kết hợp với các kỹ thuật tiền xử lý ảnh nâng cao