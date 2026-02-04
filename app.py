"""
FactoryGuard AI - Model-as-a-Service API
Real-time prediction endpoint with SHAP explanations
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import shap
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent))
from utils.feature_processor import FeatureProcessor, SHAPExplainer

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model and explainer
model = None
feature_processor = None
shap_explainer = None
model_metadata = None

# Paths
MODELS_DIR = Path(__file__).parent / 'models'
MODEL_PATH = MODELS_DIR / 'xgboost_best.pkl'
FEATURE_NAMES_PATH = MODELS_DIR / 'feature_names.pkl'
METADATA_PATH = MODELS_DIR / 'model_metadata.json'


def load_model_and_explainer():
    """
    Load model, feature names, and initialize SHAP explainer at startup
    """
    global model, feature_processor, shap_explainer, model_metadata
    
    print("=" * 70)
    print("LOADING MODEL AND INITIALIZING SHAP EXPLAINER")
    print("=" * 70)
    
    try:
        # Load model
        print(f"\nLoading model from: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        print("✓ Model loaded successfully")
        
        # Load feature names
        print(f"Loading feature names from: {FEATURE_NAMES_PATH}")
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        print(f"✓ Feature names loaded ({len(feature_names)} features)")
        
        # Load metadata if available
        if METADATA_PATH.exists():
            with open(METADATA_PATH, 'r') as f:
                model_metadata = json.load(f)
            print(f"✓ Model metadata loaded (version: {model_metadata.get('version', 'unknown')})")
        
        # Initialize feature processor
        feature_processor = FeatureProcessor(feature_names)
        print("✓ Feature processor initialized")
        
        # Initialize SHAP explainer with optimization
        print("\nInitializing SHAP TreeExplainer...")
        print("(Using interventional mode for faster computation)")
        shap_explainer = shap.TreeExplainer(
            model,
            feature_perturbation='interventional'  # Faster than default
        )
        print("✓ SHAP explainer initialized")
        
        print("\n" + "=" * 70)
        print("MODEL SERVICE READY")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    if model is None:
        return jsonify({
            "status": "unhealthy",
            "message": "Model not loaded"
        }), 503
    
    return jsonify({
        "status": "healthy",
        "model_version": model_metadata.get('version', 'unknown') if model_metadata else 'unknown',
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/model-info', methods=['GET'])
def model_info():
    """
    Get model metadata and information
    """
    if model_metadata is None:
        return jsonify({
            "error": "Model metadata not available"
        }), 404
    
    return jsonify(model_metadata), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint with SHAP explanations
    
    Request body:
    {
        "timestamp": "2024-01-15 10:30:00",
        "machine_id": "M001",
        "temperature": 75.5,
        "vibration": 0.45,
        "pressure": 100.2
    }
    
    Response:
    {
        "failure_probability": 0.78,
        "prediction": 1,
        "top_features": [...],
        "explanation": "High failure risk...",
        "timestamp": "2024-01-15T10:30:00Z"
    }
    """
    start_time = datetime.now()
    
    try:
        # Get request data
        sensor_data = request.get_json()
        
        if not sensor_data:
            return jsonify({
                "error": "No JSON data provided"
            }), 400
        
        # Validate input
        is_valid, error_msg = feature_processor.validate_input(sensor_data)
        if not is_valid:
            return jsonify({
                "error": f"Invalid input: {error_msg}"
            }), 400
        
        # Process features
        features_df = feature_processor.process_single_request(sensor_data)
        
        # Make prediction
        prediction_proba = model.predict_proba(features_df)[0]
        failure_probability = float(prediction_proba[1])  # Probability of failure
        prediction = int(failure_probability >= 0.5)
        
        # Calculate SHAP values
        shap_values = shap_explainer.shap_values(features_df)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get positive class SHAP values
        
        shap_values = shap_values[0]  # Get first (and only) row
        
        # Format top features
        top_features = SHAPExplainer.format_explanation(
            shap_values,
            features_df.values[0],
            features_df.columns.tolist(),
            top_n=5
        )
        
        # Generate text explanation
        explanation = SHAPExplainer.generate_text_explanation(
            failure_probability,
            top_features
        )
        
        # Calculate latency
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Build response
        response = {
            "failure_probability": round(failure_probability, 4),
            "prediction": prediction,
            "risk_level": "high" if failure_probability > 0.7 else "moderate" if failure_probability > 0.4 else "low",
            "top_features": top_features,
            "explanation": explanation,
            "timestamp": datetime.now().isoformat(),
            "latency_ms": round(latency_ms, 2),
            "machine_id": sensor_data.get('machine_id', 'unknown')
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"ERROR in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint for multiple samples
    
    Request body:
    {
        "samples": [
            {"timestamp": "...", "machine_id": "M001", "temperature": 75.5, ...},
            {"timestamp": "...", "machine_id": "M002", "temperature": 68.2, ...}
        ]
    }
    """
    try:
        data = request.get_json()
        samples = data.get('samples', [])
        
        if not samples:
            return jsonify({
                "error": "No samples provided"
            }), 400
        
        results = []
        for sample in samples:
            # Validate input
            is_valid, error_msg = feature_processor.validate_input(sample)
            if not is_valid:
                results.append({
                    "machine_id": sample.get('machine_id', 'unknown'),
                    "error": error_msg
                })
                continue
            
            # Process features
            features_df = feature_processor.process_single_request(sample)
            
            # Make prediction
            prediction_proba = model.predict_proba(features_df)[0]
            failure_probability = float(prediction_proba[1])
            prediction = int(failure_probability >= 0.5)
            
            results.append({
                "machine_id": sample.get('machine_id', 'unknown'),
                "failure_probability": round(failure_probability, 4),
                "prediction": prediction,
                "risk_level": "high" if failure_probability > 0.7 else "moderate" if failure_probability > 0.4 else "low"
            })
        
        return jsonify({
            "results": results,
            "total_samples": len(samples),
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Batch prediction failed: {str(e)}"
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": ["/health", "/model-info", "/predict", "/batch-predict"]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error"
    }), 500


if __name__ == '__main__':
    # Load model at startup
    if not load_model_and_explainer():
        print("\n❌ Failed to load model. Exiting.")
        sys.exit(1)
    
    # Run Flask app
    print("\n" + "=" * 70)
    print("STARTING FLASK SERVER")
    print("=" * 70)
    print("\nEndpoints:")
    print("  GET  /health          - Health check")
    print("  GET  /model-info      - Model metadata")
    print("  POST /predict         - Single prediction with SHAP")
    print("  POST /batch-predict   - Batch predictions")
    print("\n" + "=" * 70)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False  # Set to False for production
    )
