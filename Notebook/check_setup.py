"""
Script kiểm tra cài đặt và cấu trúc thư mục cho Animal Classification Project
Chạy trước khi bắt đầu với notebook
"""

import os
import sys

def check_python_version():
    """Kiểm tra phiên bản Python"""
    print("=" * 70)
    print("1. Kiểm tra Python Version")
    print("=" * 70)
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ CẢNH BÁO: Python 3.8 trở lên được khuyến nghị!")
        return False
    else:
        print("✓ Python version phù hợp!")
        return True

def check_packages():
    """Kiểm tra các packages cần thiết"""
    print("\n" + "=" * 70)
    print("2. Kiểm tra Python Packages")
    print("=" * 70)
    
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'cv2': 'opencv-python',
        'sklearn': 'scikit-learn',
        'skimage': 'scikit-image',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'PIL': 'Pillow',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'joblib': 'joblib'
    }
    
    all_ok = True
    for module, package in required_packages.items():
        try:
            if module == 'cv2':
                import cv2
                print(f"✓ {package:20} - version {cv2.__version__}")
            elif module == 'sklearn':
                import sklearn
                print(f"✓ {package:20} - version {sklearn.__version__}")
            elif module == 'skimage':
                import skimage
                print(f"✓ {package:20} - version {skimage.__version__}")
            elif module == 'PIL':
                from PIL import Image
                import PIL
                print(f"✓ {package:20} - version {PIL.__version__}")
            else:
                mod = __import__(module)
                version = getattr(mod, '__version__', 'unknown')
                print(f"✓ {package:20} - version {version}")
        except ImportError:
            print(f"❌ {package:20} - CHƯA CÀI ĐẶT!")
            all_ok = False
    
    return all_ok

def check_pytorch_gpu():
    """Kiểm tra PyTorch và GPU"""
    print("\n" + "=" * 70)
    print("3. Kiểm tra PyTorch và GPU")
    print("=" * 70)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.get_device_name(0)}")
            print("✓ GPU sẵn sàng sử dụng!")
        else:
            print("⚠ GPU không khả dụng - sẽ sử dụng CPU")
            print("   (Training sẽ chậm hơn nhưng vẫn hoạt động)")
        
        return True
    except Exception as e:
        print(f"❌ Lỗi khi kiểm tra PyTorch: {e}")
        return False

def check_directory_structure():
    """Kiểm tra cấu trúc thư mục"""
    print("\n" + "=" * 70)
    print("4. Kiểm tra Cấu trúc Thư mục")
    print("=" * 70)
    
    # Get project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    print(f"Project root: {project_root}")
    
    # Check required directories
    required_dirs = {
        'Notebook': os.path.join(project_root, 'Notebook'),
        'data': os.path.join(project_root, 'data'),
        'data/afhq-raw': os.path.join(project_root, 'data', 'afhq-raw'),
        'backend': os.path.join(project_root, 'backend'),
        'frontend': os.path.join(project_root, 'frontend'),
    }
    
    all_ok = True
    for name, path in required_dirs.items():
        if os.path.exists(path):
            print(f"✓ {name:20} - Tồn tại")
        else:
            if name == 'data/afhq-raw':
                print(f"❌ {name:20} - KHÔNG TỒN TẠI! (Cần tải dataset)")
                all_ok = False
            else:
                print(f"⚠ {name:20} - Không tồn tại (sẽ được tạo tự động)")
    
    # Check dataset structure
    dataset_path = os.path.join(project_root, 'data', 'afhq-raw')
    if os.path.exists(dataset_path):
        print("\n   Kiểm tra dataset:")
        expected_classes = ['cat', 'dog', 'wild']
        for cls in expected_classes:
            cls_path = os.path.join(dataset_path, cls)
            if os.path.exists(cls_path):
                count = len([f for f in os.listdir(cls_path) 
                           if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                print(f"   ✓ {cls:10} - {count} images")
            else:
                print(f"   ❌ {cls:10} - Thư mục không tồn tại!")
                all_ok = False
    
    return all_ok

def check_disk_space():
    """Kiểm tra dung lượng ổ đĩa"""
    print("\n" + "=" * 70)
    print("5. Kiểm tra Dung lượng Ổ đĩa")
    print("=" * 70)
    
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        
        gb = 1024 ** 3
        print(f"Total: {total // gb} GB")
        print(f"Used:  {used // gb} GB")
        print(f"Free:  {free // gb} GB")
        
        if free < 5 * gb:
            print("⚠ CẢNH BÁO: Dung lượng trống < 5GB. Khuyến nghị giải phóng thêm không gian.")
            return False
        else:
            print("✓ Dung lượng đủ!")
            return True
    except Exception as e:
        print(f"⚠ Không thể kiểm tra dung lượng: {e}")
        return True

def main():
    """Main function"""
    print("\n" + "=" * 70)
    print("ANIMAL CLASSIFICATION - KIỂM TRA CÀI ĐẶT")
    print("=" * 70 + "\n")
    
    results = []
    
    # Run all checks
    results.append(("Python Version", check_python_version()))
    results.append(("Packages", check_packages()))
    results.append(("PyTorch & GPU", check_pytorch_gpu()))
    results.append(("Directory Structure", check_directory_structure()))
    results.append(("Disk Space", check_disk_space()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TÓM TẮT KẾT QUẢ")
    print("=" * 70)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{name:25} - {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ TẤT CẢ KIỂM TRA ĐỀU PASS!")
        print("Bạn có thể bắt đầu chạy notebook.")
    else:
        print("⚠ MỘT SỐ KIỂM TRA THẤT BẠI!")
        print("\nHướng dẫn khắc phục:")
        print("1. Cài đặt packages thiếu: pip install -r requirements.txt")
        print("2. Tải dataset và đặt vào: data/afhq-raw/")
        print("3. Đảm bảo Python >= 3.8")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
