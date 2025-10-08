#!/usr/bin/env python3
"""
Script chạy demo nhanh cho dự án PCA Face Recognition
"""

import os
import sys
import subprocess

def check_requirements():
    """Kiểm tra các thư viện cần thiết"""
    import_checks = [
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('cv2', 'opencv-python'),
        ('matplotlib', 'matplotlib'),
        ('streamlit', 'streamlit'),
        ('joblib', 'joblib'),
        ('PIL', 'Pillow')
    ]
    
    missing_packages = []
    
    for import_name, package_name in import_checks:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Thiếu các thư viện sau:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nCài đặt bằng lệnh: pip install -r requirements.txt")
        return False
    
    return True

def setup_demo():
    """Thiết lập demo"""
    print("🎭 Thiết lập Demo PCA Face Recognition")
    print("=" * 50)
    
    # Kiểm tra thư viện
    if not check_requirements():
        return False
    
    print("✅ Tất cả thư viện đã được cài đặt")
    
    # Kiểm tra dữ liệu
    data_folder = "data"
    model_file = "models/pca_face_model.pkl"
    
    if not os.path.exists(data_folder) or not os.listdir(data_folder):
        print("📁 Không tìm thấy dữ liệu training...")
        response = input("Tạo dữ liệu mẫu? (Y/n): ").strip().lower()
        
        if response != 'n':
            print("Tạo dữ liệu mẫu...")
            from create_sample_data import create_sample_dataset
            create_sample_dataset()
        else:
            print("Vui lòng thêm dữ liệu vào thư mục data/ trước khi chạy demo")
            return False
    
    # Kiểm tra model
    if not os.path.exists(model_file):
        print("🤖 Không tìm thấy model đã train...")
        print("Đang huấn luyện model...")
        
        try:
            from train_model import main as train_main
            train_main()
        except Exception as e:
            print(f"❌ Lỗi khi huấn luyện model: {e}")
            return False
    
    print("✅ Thiết lập hoàn tất!")
    return True

def run_streamlit_demo():
    """Chạy demo Streamlit"""
    print("\n🚀 Khởi động demo Streamlit...")
    print("Demo sẽ mở trong trình duyệt web")
    print("Nhấn Ctrl+C để dừng demo")
    print("-" * 30)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app_demo.py"])
    except KeyboardInterrupt:
        print("\n👋 Demo đã dừng")
    except Exception as e:
        print(f"❌ Lỗi khi chạy Streamlit: {e}")

def main():
    """Hàm main"""
    print("🎓 Demo Đồ Án: Ứng dụng PCA trong Nhận diện Khuôn mặt")
    print("Sinh viên có thể sử dụng script này để chạy demo nhanh")
    print("=" * 60)
    
    # Thiết lập demo
    if not setup_demo():
        print("\n❌ Thiết lập thất bại. Vui lòng kiểm tra lại!")
        return
    
    # Chạy demo
    print("\n" + "="*60)
    response = input("Chạy demo Streamlit ngay? (Y/n): ").strip().lower()
    
    if response != 'n':
        run_streamlit_demo()
    else:
        print("\n📋 Hướng dẫn chạy thủ công:")
        print("1. Huấn luyện model: python train_model.py")
        print("2. Test nhận diện: python recognize.py")
        print("3. Chạy demo web: streamlit run app_demo.py")

if __name__ == "__main__":
    main()