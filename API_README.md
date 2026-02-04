# FactoryGuard AI - Model-as-a-Service API

## Overview

Real-time prediction service for predictive maintenance with SHAP-based explanations.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements_api.txt
```

### 2. Ensure Model is Trained

The API requires a trained XGBoost model. If not already trained:

```bash
python src/xgboost_tuning.py
```

This will create:
- `models/xgboost_best.pkl` - Trained model
- `models/feature_names.pkl` - Feature names
- `models/model_metadata.json` - Model metadata

### 3. Start the API Server

```bash
python app.py
```

Expected output:
```
======================================================================
LOADING MODEL AND INITIALIZING SHAP EXPLAINER
======================================================================

Loading model from: models/xgboost_best.pkl
✓ Model loaded successfully
✓ Feature names loaded (25 features)
✓ Model metadata loaded (version: 20260204_171500)
✓ Feature processor initialized

Initializing SHAP TreeExplainer...
(Using interventional mode for faster computation)
✓ SHAP explainer initialized

======================================================================
MODEL SERVICE READY
======================================================================

======================================================================
STARTING FLASK SERVER
======================================================================

Endpoints:
  GET  /health          - Health check
  GET  /model-info      - Model metadata
  POST /predict         - Single prediction with SHAP
  POST /batch-predict   - Batch predictions

======================================================================
 * Running on http://0.0.0.0:5000
```

## API Endpoints

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_version": "20260204_171500",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### GET /model-info

Get model metadata and performance metrics.

**Response:**
```json
{
  "version": "20260204_171500",
  "training_date": "2024-01-15T10:00:00Z",
  "f1_score": 0.8542,
  "recall": 0.8923,
  "feature_count": 25,
  "best_params": {...}
}
```

### POST /predict

Make a single prediction with SHAP explanations.

**Request:**
```json
{
  "timestamp": "2024-01-15 10:30:00",
  "machine_id": "M001",
  "temperature": 75.5,
  "vibration": 0.45,
  "pressure": 100.2
}
```

**Response:**
```json
{
  "failure_probability": 0.7842,
  "prediction": 1,
  "risk_level": "high",
  "top_features": [
    {
      "feature": "temperature",
      "shap_value": 0.42,
      "feature_value": 75.5,
      "contribution": "increases"
    },
    {
      "feature": "vibration_roll_mean_12",
      "shap_value": 0.18,
      "feature_value": 0.48,
      "contribution": "increases"
    },
    {
      "feature": "temperature_lag_1",
      "shap_value": 0.12,
      "feature_value": 74.2,
      "contribution": "increases"
    },
    {
      "feature": "pressure",
      "shap_value": -0.08,
      "feature_value": 100.2,
      "contribution": "decreases"
    },
    {
      "feature": "vibration",
      "shap_value": 0.06,
      "feature_value": 0.45,
      "contribution": "increases"
    }
  ],
  "explanation": "High failure risk (78%) primarily due to elevated temperature (75.5°C) and sustained sensor patterns.",
  "timestamp": "2024-01-15T10:30:00Z",
  "latency_ms": 45.23,
  "machine_id": "M001"
}
```

### POST /batch-predict

Make predictions for multiple samples (without SHAP explanations for speed).

**Request:**
```json
{
  "samples": [
    {
      "timestamp": "2024-01-15 10:30:00",
      "machine_id": "M001",
      "temperature": 75.5,
      "vibration": 0.45,
      "pressure": 100.2
    },
    {
      "timestamp": "2024-01-15 10:31:00",
      "machine_id": "M002",
      "temperature": 68.2,
      "vibration": 0.35,
      "pressure": 98.5
    }
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "machine_id": "M001",
      "failure_probability": 0.7842,
      "prediction": 1,
      "risk_level": "high"
    },
    {
      "machine_id": "M002",
      "failure_probability": 0.3215,
      "prediction": 0,
      "risk_level": "low"
    }
  ],
  "total_samples": 2,
  "timestamp": "2024-01-15T10:31:00Z"
}
```

## Testing

### Unit Tests

Run feature processor tests:

```bash
pytest tests/test_feature_processor.py -v
```

### Latency Testing

Run performance tests:

```bash
# Sequential test (100 requests)
python tests/latency_test.py --requests 100

# Concurrent test (100 requests, 10 workers)
python tests/latency_test.py --requests 100 --concurrent 10

# High-risk scenario test
python tests/latency_test.py --requests 50 --scenario high_risk

# Custom URL
python tests/latency_test.py --url http://production-server:5000 --requests 200
```

**Expected Output:**
```
======================================================================
PERFORMANCE REPORT - Sequential
======================================================================

📊 Request Statistics:
  Total Requests:    100
  Successful:        100
  Failed:            0
  Success Rate:      100.00%

⏱️  Latency Metrics:
  Average:           45.23 ms
  Median (P50):      42.15 ms
  P95:               78.34 ms
  P99:               95.67 ms
  Min:               38.12 ms
  Max:               102.45 ms
  Std Dev:           12.34 ms

🚀 Throughput:
  Requests/sec:      22.11

✅ Performance Assessment:
  ✓ Average latency: EXCELLENT (45.23ms < 50ms target)
  ✓ P95 latency: EXCELLENT (78.34ms < 100ms)

======================================================================
```

## Example Usage

### Using curl

```bash
# Health check
curl http://localhost:5000/health

# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2024-01-15 10:30:00",
    "machine_id": "M001",
    "temperature": 75.5,
    "vibration": 0.45,
    "pressure": 100.2
  }'
```

### Using Python

```python
import requests

# Make prediction
response = requests.post(
    'http://localhost:5000/predict',
    json={
        'timestamp': '2024-01-15 10:30:00',
        'machine_id': 'M001',
        'temperature': 75.5,
        'vibration': 0.45,
        'pressure': 100.2
    }
)

result = response.json()
print(f"Failure Probability: {result['failure_probability']:.2%}")
print(f"Explanation: {result['explanation']}")
```

## Performance Optimization

If latency exceeds 50ms target:

1. **SHAP Optimization** (already implemented):
   - Using `TreeExplainer` with `feature_perturbation='interventional'`
   - Faster than default 'tree_path_dependent' mode

2. **Model Optimization**:
   - Consider model quantization
   - Use `ntree_limit` for faster inference

3. **Caching**:
   - Cache SHAP explainer (already done at startup)
   - Add Redis for feature caching

4. **Infrastructure**:
   - Use gunicorn with multiple workers:
     ```bash
     gunicorn -w 4 -b 0.0.0.0:5000 app:app
     ```

## Known Limitations

1. **Historical Features**: Lag and rolling features use current values as defaults since historical data is not maintained. For production, consider:
   - Maintaining state in Redis/database
   - Accepting pre-computed features from client
   - Using time-series database

2. **Concurrency**: Single-threaded Flask server. For production:
   - Use gunicorn with multiple workers
   - Consider async framework (FastAPI)

3. **SHAP Latency**: Computing SHAP values adds ~20-30ms. For ultra-low latency:
   - Use `/batch-predict` endpoint (no SHAP)
   - Pre-compute SHAP for common scenarios

## Production Deployment

### Using Gunicorn

```bash
pip install gunicorn

# Run with 4 workers
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 app:app
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt -r requirements_api.txt

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## Troubleshooting

### Model not found
```
ERROR: models/xgboost_best.pkl not found
```
**Solution**: Run `python src/xgboost_tuning.py` to train the model.

### Import errors
```
ModuleNotFoundError: No module named 'flask'
```
**Solution**: Install API dependencies: `pip install -r requirements_api.txt`

### High latency
```
Average latency: 150ms
```
**Solution**: 
1. Check if SHAP calculation is the bottleneck
2. Use `/batch-predict` for bulk operations
3. Consider model optimization or infrastructure scaling

## Support

For issues or questions, please refer to the main project documentation.
