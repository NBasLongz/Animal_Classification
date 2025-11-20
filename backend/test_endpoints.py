"""
Test script for backend endpoints
Run this after restarting the backend server
"""
import requests
import json

BASE_URL = "http://localhost:5000/api"

def test_health():
    print("Testing /api/health...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"✓ Status: {response.status_code}")
        print(f"  Response: {response.json()}\n")
        return True
    except Exception as e:
        print(f"✗ Error: {e}\n")
        return False

def test_training_results():
    print("Testing /api/training-results...")
    try:
        response = requests.get(f"{BASE_URL}/training-results")
        print(f"✓ Status: {response.status_code}")
        data = response.json()
        print(f"  Confusion Matrix URL: {data.get('confusion_matrix_url')}")
        print(f"  Models:")
        for model in data.get('models', []):
            print(f"    - {model['name']}: {model['accuracy']}%")
        print()
        return True
    except Exception as e:
        print(f"✗ Error: {e}\n")
        return False

def test_confusion_matrix():
    print("Testing /api/confusion-matrix...")
    try:
        response = requests.get(f"{BASE_URL}/confusion-matrix")
        print(f"✓ Status: {response.status_code}")
        print(f"  Content-Type: {response.headers.get('Content-Type')}")
        print(f"  Image size: {len(response.content)} bytes")
        
        # Save image for verification
        with open('confusion_matrix_test.png', 'wb') as f:
            f.write(response.content)
        print(f"  ✓ Saved to confusion_matrix_test.png\n")
        return True
    except Exception as e:
        print(f"✗ Error: {e}\n")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Backend Endpoint Testing")
    print("=" * 50 + "\n")
    
    if test_health():
        test_training_results()
        test_confusion_matrix()
    else:
        print("Backend is not running. Please start the backend first:")
        print("  python app.py")
