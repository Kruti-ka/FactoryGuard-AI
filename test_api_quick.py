"""
Quick test script for the API
"""

import requests
import json

# Test data
test_request = {
    "timestamp": "2024-01-15 10:30:00",
    "machine_id": "M001",
    "temperature": 75.5,
    "vibration": 0.45,
    "pressure": 100.2
}

print("Testing FactoryGuard AI API...")
print("="*60)

# Test health endpoint
print("\n1. Testing /health endpoint...")
try:
    response = requests.get('http://localhost:5000/health', timeout=5)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")

# Test prediction endpoint
print("\n2. Testing /predict endpoint...")
try:
    response = requests.post(
        'http://localhost:5000/predict',
        json=test_request,
        headers={'Content-Type': 'application/json'},
        timeout=10
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\nPrediction Results:")
        print(f"  Failure Probability: {result['failure_probability']:.2%}")
        print(f"  Prediction: {'FAILURE' if result['prediction'] == 1 else 'NORMAL'}")
        print(f"  Risk Level: {result['risk_level'].upper()}")
        print(f"  Latency: {result['latency_ms']:.2f}ms")
        print(f"\nExplanation: {result['explanation']}")
        print(f"\nTop Contributing Features:")
        for i, feat in enumerate(result['top_features'][:3], 1):
            print(f"  {i}. {feat['feature']}: {feat['shap_value']:.4f} ({feat['contribution']})")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*60)
