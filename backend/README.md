# Animal Classification Backend API

Backend API server cho dự án phân loại động vật sử dụng Flask.

## Cài đặt

1. Tạo môi trường ảo:
```bash
python -m venv venv
```

2. Kích hoạt môi trường ảo:
- Windows:
```bash
venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

3. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

## Chạy Server

```bash
python app.py
```

Server sẽ chạy tại: `http://localhost:5000`

## API Endpoints

### 1. Health Check
```
GET /api/health
```
Kiểm tra trạng thái server.

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda/cpu",
  "models_loaded": true
}
```

### 2. Predict (Single Model)
```
POST /api/predict
```
Dự đoán với 1 model cụ thể.

**Request Body:**
```json
{
  "image": "base64_encoded_image",
  "method": "handcrafted",  // hoặc "deep"
  "model": "svm"            // hoặc "rf"
}
```

**Response:**
```json
{
  "prediction": "cat",
  "confidence": 0.95,
  "probabilities": {
    "cat": 0.95,
    "dog": 0.03,
    "wild": 0.02
  },
  "processing_time": 0.123,
  "method": "handcrafted",
  "model": "svm"
}
```

### 3. Predict All Models
```
POST /api/predict-all
```
Dự đoán với tất cả 4 models cùng lúc.

**Request Body:**
```json
{
  "image": "base64_encoded_image"
}
```

**Response:**
```json
{
  "rf_handcrafted": {
    "prediction": "cat",
    "confidence": 0.92,
    "probabilities": {...},
    "processing_time": 0.05
  },
  "svm_handcrafted": {
    "prediction": "cat",
    "confidence": 0.95,
    "probabilities": {...},
    "processing_time": 0.03
  },
  "rf_deep": {
    "prediction": "cat",
    "confidence": 0.97,
    "probabilities": {...},
    "processing_time": 0.15
  },
  "svm_deep": {
    "prediction": "cat",
    "confidence": 0.98,
    "probabilities": {...},
    "processing_time": 0.12
  }
}
```

### 4. Get Labels
```
GET /api/labels
```
Lấy danh sách các labels.

**Response:**
```json
{
  "labels": ["cat", "dog", "wild"]
}
```

## Cấu trúc thư mục

```
backend/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
└── README.md          # This file

models/                # Trained models (từ notebook)
├── rf_handcrafted.pkl
├── svm_handcrafted.pkl
├── rf_deep.pkl
├── svm_deep.pkl
├── scaler_handcrafted.pkl
├── pca_handcrafted.pkl
├── scaler_deep.pkl
└── pca_deep.pkl

data/processed/        # Processed data (từ notebook)
├── config.json
└── labels.json
```

## Lưu ý

- Đảm bảo đã chạy notebook và lưu tất cả models trước khi chạy backend
- Server cần GPU để chạy ResNet50 nhanh hơn (tùy chọn)
- CORS đã được bật để frontend có thể gọi API
