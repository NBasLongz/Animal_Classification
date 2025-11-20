# Animal Classification - Notebook Requirements

## Cài đặt

### 1. Tạo môi trường ảo (khuyến nghị)

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/Mac:
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. Cài đặt PyTorch (tùy chọn GPU/CPU)

#### CPU only:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### CUDA 11.8 (GPU):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### CUDA 12.1 (GPU):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Cấu trúc thư mục cần thiết

```
Animal-Classification/
├── Notebook/
│   ├── Demo_Animal_Classification.ipynb
│   └── requirements.txt
├── data/
│   ├── afhq-raw/          # Dataset gốc (cần tải về)
│   │   ├── cat/
│   │   ├── dog/
│   │   └── wild/
│   ├── afhq_split_80_20/  # Sẽ được tạo tự động
│   │   ├── train/
│   │   └── test/
│   └── processed/         # Sẽ được tạo tự động
├── models/                # Sẽ được tạo tự động
```

## Tải Dataset

### Option 1: Kaggle
1. Truy cập: https://www.kaggle.com/datasets/andrewmvd/animal-faces
2. Download dataset
3. Giải nén vào: `data/afhq-raw/`

### Option 2: Manual
Đảm bảo cấu trúc thư mục:
```
data/afhq-raw/
├── cat/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── dog/
│   ├── image1.jpg
│   └── ...
└── wild/
    ├── image1.jpg
    └── ...
```

## Chạy Notebook

1. Khởi động Jupyter:
```bash
jupyter notebook
```

2. Mở file: `Demo_Animal_Classification.ipynb`

3. Chạy các cells theo thứ tự từ trên xuống

## Kiểm tra cài đặt

```python
import torch
import cv2
import sklearn
import skimage
from PIL import Image

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"OpenCV: {cv2.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"scikit-image: {skimage.__version__}")
```

## Packages chính

### Machine Learning
- **scikit-learn**: SVM, Random Forest, preprocessing
- **scikit-image**: HOG features extraction

### Deep Learning
- **PyTorch**: Deep learning framework
- **torchvision**: Pre-trained models (ResNet50)

### Image Processing
- **OpenCV (cv2)**: Image manipulation, color histograms
- **Pillow (PIL)**: Image loading and conversion

### Data & Visualization
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib**: Plotting
- **Seaborn**: Statistical visualization

## Xử lý lỗi thường gặp

### 1. Module not found
```bash
pip install <module_name>
```

### 2. CUDA out of memory
- Giảm batch size
- Sử dụng CPU: `DEVICE = torch.device("cpu")`

### 3. OpenCV import error
```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

### 4. Dataset not found
- Kiểm tra đường dẫn: `data/afhq-raw/`
- Đảm bảo có 3 thư mục con: cat, dog, wild

## Lưu ý

- Python version: 3.8 hoặc cao hơn
- RAM khuyến nghị: 8GB trở lên
- GPU: Không bắt buộc nhưng sẽ nhanh hơn cho ResNet50
- Disk space: ~5GB cho dataset và models

## Support

Nếu gặp vấn đề, kiểm tra:
1. Python version: `python --version`
2. Pip version: `pip --version`
3. Virtual environment đã được kích hoạt
4. Dataset đã được tải và đặt đúng vị trí
