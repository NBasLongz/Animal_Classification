# Animal Classification Project

Dự án phân loại động vật sử dụng Computer Vision với 2 phương pháp:
1. **Handcrafted Features** (HOG + Color Histogram)
2. **Deep Learning** (ResNet50)

## Cấu trúc Project

```
Animal-Classification/
├── backend/                    # Flask API Server
│   ├── app.py                 # Main API application
│   ├── requirements.txt       # Python dependencies
│   └── README.md             # Backend documentation
│
├── frontend/                  # Web Interface
│   ├── index.html            # Main HTML page
│   ├── css/
│   │   └── styles.css        # Styles
│   └── js/
│       └── main.js           # JavaScript logic
│
├── Notebook/                  # Jupyter Notebooks
│   └── Demo_Animal_Classification.ipynb
│
├── models/                    # Trained Models (generated)
│   ├── rf_handcrafted.pkl
│   ├── svm_handcrafted.pkl
│   ├── rf_deep.pkl
│   ├── svm_deep.pkl
│   ├── scaler_handcrafted.pkl
│   ├── pca_handcrafted.pkl
│   ├── scaler_deep.pkl
│   └── pca_deep.pkl
│
├── data/                      # Data directory
│   ├── afhq-raw/             # Raw dataset
│   └── processed/            # Processed data (generated)
│       ├── config.json
│       ├── labels.json
│       └── *.npy files
│
└── README.md                 # This file
```

## Hướng dẫn Setup

### Bước 1: Clone Repository
```bash
git clone <repository-url>
cd Animal-Classification
```

### Bước 2: Chuẩn bị Dataset
1. Download AFHQ dataset từ Kaggle
2. Giải nén vào thư mục `data/afhq-raw/`

### Bước 3: Chạy Notebook để Train Models
1. Mở `Notebook/Demo_Animal_Classification.ipynb`
2. Chạy tất cả các cells theo thứ tự
3. Notebook sẽ tự động:
   - Chia dữ liệu train/test
   - Trích xuất đặc trưng
   - Train 4 models (RF-Hand, SVM-Hand, RF-Deep, SVM-Deep)
   - Lưu models và preprocessors vào thư mục `models/`
   - Lưu config và data vào thư mục `data/processed/`

### Bước 4: Chạy Backend API
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
# hoặc: source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
python app.py
```

Backend sẽ chạy tại: `http://localhost:5000`

### Bước 5: Chạy Frontend
Mở file `frontend/index.html` trong trình duyệt hoặc sử dụng Live Server.

## Sử dụng

### Web Demo
1. Truy cập frontend
2. Upload ảnh động vật
3. Chọn phương pháp (Handcrafted hoặc Deep Learning)
4. Chọn model (SVM hoặc Random Forest)
5. Click "Thực hiện Dự đoán"
6. Xem kết quả phân loại

### API Usage

#### Health Check
```bash
curl http://localhost:5000/api/health
```

#### Predict với 1 model
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_encoded_image",
    "method": "handcrafted",
    "model": "svm"
  }'
```

#### Predict với tất cả models
```bash
curl -X POST http://localhost:5000/api/predict-all \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_encoded_image"
  }'
```

## Models

### Handcrafted Features
- **HOG (Histogram of Oriented Gradients)**: Trích xuất đặc trưng hình dạng
- **Color Histogram**: Trích xuất đặc trưng màu sắc
- **PCA**: Giảm chiều dữ liệu

### Deep Learning
- **ResNet50**: Pre-trained trên ImageNet
- **Transfer Learning**: Sử dụng features từ lớp áp chót (2048 chiều)

### Classifiers
- **SVM (Support Vector Machine)**: RBF kernel
- **Random Forest**: 500 trees

## Dataset
- **Name**: AFHQ (Animal Faces-HQ)
- **Classes**: Cat, Dog, Wild
- **Split**: 80% Train, 20% Test

## Requirements

### Python Packages
- Flask
- NumPy
- OpenCV
- scikit-learn
- scikit-image
- PyTorch
- torchvision
- Pillow
- joblib

### Browser
- Modern browser với hỗ trợ ES6+
- JavaScript enabled

## Tác giả
- Nguyễn Bá Long (MSSV: 20120000)
- Nguyễn Công Thiết (MSSV: 20120001)

## Giảng viên hướng dẫn
TS. Mai Tiến Dũng

## Môn học
Nhập môn Thị giác Máy tính - CS231.Q11

## License
MIT License
