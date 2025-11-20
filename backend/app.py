from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import numpy as np
import cv2
import joblib
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
from skimage.feature import hog
from skimage import exposure
import io
import base64
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'Notebook', 'models')
DATA_DIR = os.path.join(BASE_DIR, '..', 'Notebook', 'data', 'processed')

# Load configuration
config_path = os.path.join(DATA_DIR, 'config.json')
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# Extract configuration
IMG_SIZE_HOG = tuple(config['IMG_SIZE_HOG'])
HOG_ORIENTATIONS = config['HOG_ORIENTATIONS']
HOG_PIX_PER_CELL = tuple(config['HOG_PIX_PER_CELL'])
HOG_CELL_PER_BLOCK = tuple(config['HOG_CELL_PER_BLOCK'])
RGB_BINS = config['RGB_BINS']
USE_PCA = config['USE_PCA']
labels = config['labels']

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Load models and preprocessors
print("Loading models and preprocessors...")

# Handcrafted models
rf_hand = joblib.load(os.path.join(MODELS_DIR, 'rf_handcrafted.pkl'))
svm_hand = joblib.load(os.path.join(MODELS_DIR, 'svm_handcrafted.pkl'))
scaler_hand = joblib.load(os.path.join(MODELS_DIR, 'scaler_handcrafted.pkl'))

pca_hand_path = os.path.join(MODELS_DIR, 'pca_handcrafted.pkl')
pca_hand = joblib.load(pca_hand_path) if os.path.exists(pca_hand_path) else None

# Deep learning models
rf_deep = joblib.load(os.path.join(MODELS_DIR, 'rf_deep.pkl'))
svm_deep = joblib.load(os.path.join(MODELS_DIR, 'svm_deep.pkl'))
scaler_deep = joblib.load(os.path.join(MODELS_DIR, 'scaler_deep.pkl'))

pca_deep_path = os.path.join(MODELS_DIR, 'pca_deep.pkl')
pca_deep = joblib.load(pca_deep_path) if os.path.exists(pca_deep_path) else None

# Load ResNet50 for deep features
print("Loading ResNet50...")
resnet_weights = models.ResNet50_Weights.IMAGENET1K_V1
resnet_model = models.resnet50(weights=resnet_weights)
resnet_model.fc = nn.Identity()  # Remove classifier
resnet_model.to(DEVICE)
resnet_model.eval()
resnet_transform = resnet_weights.transforms()

print("✓ All models loaded successfully!")

def extract_handcrafted_features(img_array):
    """Extract HOG + Color Histogram features from image array"""
    # Resize
    img_resized = cv2.resize(img_array, IMG_SIZE_HOG)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # HOG
    hog_feat = hog(gray, orientations=HOG_ORIENTATIONS,
                    pixels_per_cell=HOG_PIX_PER_CELL,
                    cells_per_block=HOG_CELL_PER_BLOCK,
                    block_norm='L2-Hys')
    
    # Color Histogram
    color_feat = []
    for channel in cv2.split(img_resized):
        hist = cv2.calcHist([channel], [0], None, [RGB_BINS], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        color_feat.extend(hist)
    
    # Combine & normalize
    features = np.concatenate([hog_feat, np.array(color_feat, dtype=np.float32)])
    features = features / (np.linalg.norm(features) + 1e-8)
    
    return features

def extract_deep_features(img_pil):
    """Extract ResNet50 features from PIL image"""
    img_tensor = resnet_transform(img_pil).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        features = resnet_model(img_tensor)
    
    return features.cpu().flatten().numpy()

def process_image_from_base64(base64_string):
    """Convert base64 string to image arrays"""
    # Remove header if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64
    img_data = base64.b64decode(base64_string)
    
    # Convert to PIL Image
    img_pil = Image.open(io.BytesIO(img_data)).convert('RGB')
    
    # Convert to numpy array for OpenCV
    img_array = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    return img_pil, img_bgr

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': str(DEVICE),
        'models_loaded': True
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        print("\n=== NEW PREDICTION REQUEST ===")
        data = request.json
        
        if 'image' not in data:
            print("ERROR: No image provided")
            return jsonify({'error': 'No image provided'}), 400
        
        method = data.get('method', 'handcrafted')  # 'handcrafted' or 'deep'
        model_type = data.get('model', 'svm')  # 'svm' or 'rf'
        print(f"Method: {method}, Model: {model_type}")
        
        # Process image
        img_pil, img_bgr = process_image_from_base64(data['image'])
        
        start_time = time.time()
        
        if method == 'handcrafted':
            # Extract handcrafted features
            features = extract_handcrafted_features(img_bgr)
            features_scaled = scaler_hand.transform([features])
            
            if pca_hand:
                features_final = pca_hand.transform(features_scaled)
            else:
                features_final = features_scaled
            
            # Select model
            model = rf_hand if model_type == 'rf' else svm_hand
            
        else:  # deep learning
            # Extract deep features
            features = extract_deep_features(img_pil)
            features_scaled = scaler_deep.transform([features])
            
            if pca_deep:
                features_final = pca_deep.transform(features_scaled)
            else:
                features_final = features_scaled
            
            # Select model
            model = rf_deep if model_type == 'rf' else svm_deep
        
        # Predict
        prediction = model.predict(features_final)[0]
        probabilities = model.predict_proba(features_final)[0]
        
        processing_time = time.time() - start_time
        
        print(f"Prediction: {labels[prediction]}, Confidence: {probabilities[prediction]:.4f}")
        print(f"Processing time: {processing_time:.3f}s")
        print("=== REQUEST COMPLETED ===\n")
        
        # Prepare response
        result = {
            'prediction': labels[prediction],
            'confidence': float(probabilities[prediction]),
            'probabilities': {
                label: float(prob) 
                for label, prob in zip(labels, probabilities)
            },
            'processing_time': round(processing_time, 3),
            'method': method,
            'model': model_type
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"ERROR in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-all', methods=['POST'])
def predict_all():
    """Predict with all 4 models"""
    try:
        data = request.json
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Process image
        img_pil, img_bgr = process_image_from_base64(data['image'])
        
        results = {}
        
        # Handcrafted features
        features_hand = extract_handcrafted_features(img_bgr)
        features_hand_scaled = scaler_hand.transform([features_hand])
        if pca_hand:
            features_hand_final = pca_hand.transform(features_hand_scaled)
        else:
            features_hand_final = features_hand_scaled
        
        # Deep features
        features_deep = extract_deep_features(img_pil)
        features_deep_scaled = scaler_deep.transform([features_deep])
        if pca_deep:
            features_deep_final = pca_deep.transform(features_deep_scaled)
        else:
            features_deep_final = features_deep_scaled
        
        # RF Handcrafted
        start_time = time.time()
        pred = rf_hand.predict(features_hand_final)[0]
        prob = rf_hand.predict_proba(features_hand_final)[0]
        results['rf_handcrafted'] = {
            'prediction': labels[pred],
            'confidence': float(prob[pred]),
            'probabilities': {label: float(p) for label, p in zip(labels, prob)},
            'processing_time': round(time.time() - start_time, 3)
        }
        
        # SVM Handcrafted
        start_time = time.time()
        pred = svm_hand.predict(features_hand_final)[0]
        prob = svm_hand.predict_proba(features_hand_final)[0]
        results['svm_handcrafted'] = {
            'prediction': labels[pred],
            'confidence': float(prob[pred]),
            'probabilities': {label: float(p) for label, p in zip(labels, prob)},
            'processing_time': round(time.time() - start_time, 3)
        }
        
        # RF Deep
        start_time = time.time()
        pred = rf_deep.predict(features_deep_final)[0]
        prob = rf_deep.predict_proba(features_deep_final)[0]
        results['rf_deep'] = {
            'prediction': labels[pred],
            'confidence': float(prob[pred]),
            'probabilities': {label: float(p) for label, p in zip(labels, prob)},
            'processing_time': round(time.time() - start_time, 3)
        }
        
        # SVM Deep
        start_time = time.time()
        pred = svm_deep.predict(features_deep_final)[0]
        prob = svm_deep.predict_proba(features_deep_final)[0]
        results['svm_deep'] = {
            'prediction': labels[pred],
            'confidence': float(prob[pred]),
            'probabilities': {label: float(p) for label, p in zip(labels, prob)},
            'processing_time': round(time.time() - start_time, 3)
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/labels', methods=['GET'])
def get_labels():
    """Get available labels"""
    return jsonify({'labels': labels})

@app.route('/api/training-results', methods=['GET'])
def get_training_results():
    """Get training results for all models"""
    try:
        results_path = os.path.join(DATA_DIR, 'training_results.json')
        with open(results_path, 'r', encoding='utf-8') as f:
            training_results = json.load(f)
        
        # Update confusion matrix URL to point to our API
        training_results['confusion_matrix_url'] = 'http://localhost:5000/api/confusion-matrix'
        
        return jsonify(training_results)
    except Exception as e:
        print(f"ERROR loading training results: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/confusion-matrix', methods=['GET'])
def get_confusion_matrix():
    """Generate and return confusion matrix image for all models"""
    try:
        # Load test data
        y_test_hand = np.load(os.path.join(DATA_DIR, 'y_test_hand.npy'))
        y_test_deep = np.load(os.path.join(DATA_DIR, 'y_test_deep.npy'))
        X_test_hand = np.load(os.path.join(DATA_DIR, 'X_test_hand.npy'))
        X_test_deep = np.load(os.path.join(DATA_DIR, 'X_test_deep.npy'))
        
        # Scale and transform
        X_test_hand_scaled = scaler_hand.transform(X_test_hand)
        if pca_hand:
            X_test_hand_final = pca_hand.transform(X_test_hand_scaled)
        else:
            X_test_hand_final = X_test_hand_scaled
            
        X_test_deep_scaled = scaler_deep.transform(X_test_deep)
        if pca_deep:
            X_test_deep_final = pca_deep.transform(X_test_deep_scaled)
        else:
            X_test_deep_final = X_test_deep_scaled
        
        # Get predictions
        y_pred_rf_hand = rf_hand.predict(X_test_hand_final)
        y_pred_svm_hand = svm_hand.predict(X_test_hand_final)
        y_pred_rf_deep = rf_deep.predict(X_test_deep_final)
        y_pred_svm_deep = svm_deep.predict(X_test_deep_final)
        
        # Calculate confusion matrices
        cm_rf_hand = confusion_matrix(y_test_hand, y_pred_rf_hand)
        cm_svm_hand = confusion_matrix(y_test_hand, y_pred_svm_hand)
        cm_rf_deep = confusion_matrix(y_test_deep, y_pred_rf_deep)
        cm_svm_deep = confusion_matrix(y_test_deep, y_pred_svm_deep)
        
        # Calculate accuracies
        acc_rf_hand = (y_pred_rf_hand == y_test_hand).mean()
        acc_svm_hand = (y_pred_svm_hand == y_test_hand).mean()
        acc_rf_deep = (y_pred_rf_deep == y_test_deep).mean()
        acc_svm_deep = (y_pred_svm_deep == y_test_deep).mean()
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        cms = [cm_rf_hand, cm_svm_hand, cm_rf_deep, cm_svm_deep]
        titles = ['RF-Handcrafted', 'SVM-Handcrafted', 'RF-Deep', 'SVM-Deep']
        accs = [acc_rf_hand, acc_svm_hand, acc_rf_deep, acc_svm_deep]
        
        for idx, (cm, title, acc) in enumerate(zip(cms, titles, accs)):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels, ax=axes[idx])
            axes[idx].set_title(f'{title}\nAcc: {acc:.3f}',
                               fontweight='bold', fontsize=12)
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('True')
        
        plt.suptitle('Confusion Matrices - So sánh 4 Models', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return send_file(buf, mimetype='image/png')
        
    except Exception as e:
        print(f"ERROR generating confusion matrix: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualize-pipeline', methods=['POST'])
def visualize_pipeline():
    """Generate pipeline visualizations for uploaded image"""
    try:
        data = request.json
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Process image
        img_pil, img_bgr = process_image_from_base64(data['image'])
        
        # Convert PIL to numpy array
        img_rgb = np.array(img_pil)
        
        # Prepare visualizations
        visualizations = {}
        
        # 1. Original image (resized to 128x128 for handcrafted)
        img_resized = cv2.resize(img_bgr, IMG_SIZE_HOG)
        img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Convert to base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(img_resized_rgb, cv2.COLOR_RGB2BGR))
        visualizations['original_128'] = 'data:image/png;base64,' + base64.b64encode(buffer).decode()
        
        # 2. Grayscale
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        _, buffer = cv2.imencode('.png', gray)
        visualizations['grayscale'] = 'data:image/png;base64,' + base64.b64encode(buffer).decode()
        
        # 3. HOG Visualization
        hog_feat, hog_img = hog(gray, 
                                orientations=HOG_ORIENTATIONS,
                                pixels_per_cell=HOG_PIX_PER_CELL,
                                cells_per_block=HOG_CELL_PER_BLOCK,
                                block_norm='L2-Hys', 
                                visualize=True)
        hog_img_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))
        
        # Convert HOG to 8-bit for PNG encoding
        hog_img_8bit = (hog_img_rescaled * 255).astype(np.uint8)
        _, buffer = cv2.imencode('.png', hog_img_8bit)
        visualizations['hog'] = 'data:image/png;base64,' + base64.b64encode(buffer).decode()
        
        # 4. Color Histogram
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = ('b', 'g', 'r')
        color_names = ('Blue', 'Green', 'Red')
        
        for i, (color, name) in enumerate(zip(colors, color_names)):
            hist = cv2.calcHist([img_resized], [i], None, [RGB_BINS], [0, 256])
            ax.plot(hist, color=color, label=name, linewidth=2)
        
        ax.set_xlabel('Bins', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('RGB Color Histogram', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        visualizations['color_histogram'] = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
        
        # 5. ResNet preprocessing visualization
        # Resize to 256 maintaining aspect ratio
        img_array = np.array(img_pil)
        h, w = img_array.shape[:2]
        scale = 256 / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized_256 = cv2.resize(img_array, (new_w, new_h))
        
        # Center crop 224x224
        start_h = (new_h - 224) // 2
        start_w = (new_w - 224) // 2
        img_cropped = img_resized_256[start_h:start_h+224, start_w:start_w+224]
        
        # Convert to base64
        img_cropped_bgr = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.png', img_cropped_bgr)
        visualizations['resnet_input'] = 'data:image/png;base64,' + base64.b64encode(buffer).decode()
        
        return jsonify(visualizations)
        
    except Exception as e:
        print(f"ERROR in visualize_pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Animal Classification API Server")
    print(f"Device: {DEVICE}")
    print(f"Labels: {labels}")
    print(f"Models loaded: RF-Hand, SVM-Hand, RF-Deep, SVM-Deep")
    
    app.run(host='0.0.0.0', port=5000, debug=True) 
