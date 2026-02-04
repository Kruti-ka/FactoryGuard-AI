# FactoryGuard AI - Manual Testing Guide

## Overview

This guide provides step-by-step instructions for manually testing the Week 3 (XAI) and Week 4 (Model-as-a-Service) implementations.

---

## Week 3: XAI (Explainable AI) - Manual Testing

### Step 1: Verify SHAP Analysis Files

**Location:** `outputs/`

**Files to Check:**
```bash
# Navigate to outputs folder
cd d:\Internship-project1\factoryguard-ai\outputs

# List files
dir
```

**Expected Files:**
1. ✅ `shap_values.pkl` - SHAP values data
2. ✅ `feature_importance_analysis.csv` - Feature rankings
3. ✅ `shap_methodology_documentation.md` - Akshada's documentation
4. ✅ `visual_interpretation_guide.md` - Harish's documentation
5. ✅ `physical_validation_report.md` - Krutika's validation report
6. ✅ `week3_xai_final_report.md` - Consolidated report
7. ✅ `week3_xai_presentation.md` - Presentation slides

### Step 2: Review Feature Importance Rankings

**Open:** `outputs/feature_importance_analysis.csv`

```bash
# View in Excel or text editor
notepad outputs\feature_importance_analysis.csv
```

**What to Check:**
- 25 features listed
- Mean absolute SHAP values present
- Features ranked by importance
- Top feature should be `temperature` (highest importance)

### Step 3: View SHAP Visualizations

**Location:** `outputs/plots/`

**Files to Check:**
```bash
cd outputs\plots
dir
```

**Expected Visualizations:**
1. `shap_summary.png` - Summary plot (all features)
2. `shap_beeswarm.png` - Beeswarm plot
3. `shap_force_plots/` folder - 10 force plots
4. `shap_dependence/` folder - 5 dependence plots
5. `shap_waterfall/` folder - 5 waterfall plots

**How to View:**
- Open each PNG file in image viewer
- Verify plots are clear and readable
- Check that feature names are visible

### Step 4: Review Validation Report

**Open:** `outputs/physical_validation_report.md`

**Key Sections to Review:**
1. **Executive Summary** - Overall findings
2. **Validation Results** - Feature consistency (52%)
3. **Anomalies Identified** - 7 anomalies detected
4. **Recommendations** - 5 prioritized recommendations

**What to Show Your Guide:**
- Consistency rate: 52% (moderate concerns)
- High-priority anomalies: 3 (temperature-related)
- Recommendations for improvement

### Step 5: Review Consolidated XAI Report

**Open:** `outputs/week3_xai_final_report.md`

**This is the MAIN REPORT to show your guide!**

**Key Highlights:**
- Integrates all team members' work (Akshada, Harish, Krutika)
- Top 10 feature importance rankings
- SHAP visualizations embedded
- Physical validation findings
- Business impact assessment
- Comprehensive recommendations

---

## Week 4: Model-as-a-Service - Manual Testing

### Step 1: Start the API Server

**Terminal 1:**
```bash
cd d:\Internship-project1\factoryguard-ai
python app.py
```

**Expected Output:**
```
======================================================================
LOADING MODEL AND INITIALIZING SHAP EXPLAINER
======================================================================

Loading model from: models/xgboost_best.pkl
✓ Model loaded successfully
✓ Feature names loaded (25 features)
✓ Model metadata loaded
✓ Feature processor initialized
✓ SHAP explainer initialized

======================================================================
MODEL SERVICE READY
======================================================================

 * Running on http://127.0.0.1:5000
```

**✅ Success Indicator:** Server running on port 5000

### Step 2: Test Health Check Endpoint

**Open a NEW Terminal (Terminal 2):**
```bash
# Using curl (if available)
curl http://localhost:5000/health

# OR using PowerShell
Invoke-WebRequest -Uri http://localhost:5000/health | Select-Object -ExpandProperty Content
```

**Expected Response:**
```json
{
  "status": "healthy",
  "model_version": "20260204_171949",
  "timestamp": "2026-02-04T17:25:04.513409"
}
```

**✅ Success Indicator:** Status code 200, status = "healthy"

### Step 3: Test Prediction Endpoint (Normal Risk)

**Create a test file:** `test_normal.json`
```json
{
  "timestamp": "2024-01-15 10:30:00",
  "machine_id": "M001",
  "temperature": 65.0,
  "vibration": 0.35,
  "pressure": 98.0
}
```

**Run Test:**
```bash
# Using curl
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d @test_normal.json

# OR using PowerShell
$body = Get-Content test_normal.json -Raw
Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -Body $body -ContentType "application/json"
```

**Expected Response:**
```json
{
  "failure_probability": 0.2-0.4,
  "prediction": 0,
  "risk_level": "low",
  "top_features": [...],
  "explanation": "Low failure risk...",
  "latency_ms": 30-50
}
```

**✅ Success Indicator:** 
- Prediction = 0 (no failure)
- Risk level = "low"
- Latency < 100ms

### Step 4: Test Prediction Endpoint (High Risk)

**Create a test file:** `test_high_risk.json`
```json
{
  "timestamp": "2024-01-15 10:30:00",
  "machine_id": "M002",
  "temperature": 90.0,
  "vibration": 0.85,
  "pressure": 115.0
}
```

**Run Test:**
```bash
# Using PowerShell
$body = Get-Content test_high_risk.json -Raw
Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -Body $body -ContentType "application/json"
```

**Expected Response:**
```json
{
  "failure_probability": 0.7-0.9,
  "prediction": 1,
  "risk_level": "high",
  "top_features": [
    {
      "feature": "temperature",
      "shap_value": 0.4-0.5,
      "contribution": "increases"
    },
    ...
  ],
  "explanation": "High failure risk (XX%) primarily due to elevated temperature...",
  "latency_ms": 30-50
}
```

**✅ Success Indicator:**
- Prediction = 1 (failure expected)
- Risk level = "high"
- Temperature is top contributing feature

### Step 5: Test Error Handling

**Create invalid test:** `test_invalid.json`
```json
{
  "timestamp": "2024-01-15 10:30:00",
  "machine_id": "M003",
  "temperature": 250.0,
  "vibration": 0.45,
  "pressure": 100.0
}
```

**Run Test:**
```bash
$body = Get-Content test_invalid.json -Raw
Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -Body $body -ContentType "application/json"
```

**Expected Response:**
```json
{
  "error": "Invalid input: Temperature out of range (0-200): 250.0"
}
```

**✅ Success Indicator:** Error message returned, not a crash

### Step 6: Test Batch Prediction

**Create batch test:** `test_batch.json`
```json
{
  "samples": [
    {
      "timestamp": "2024-01-15 10:30:00",
      "machine_id": "M001",
      "temperature": 65.0,
      "vibration": 0.35,
      "pressure": 98.0
    },
    {
      "timestamp": "2024-01-15 10:31:00",
      "machine_id": "M002",
      "temperature": 85.0,
      "vibration": 0.75,
      "pressure": 110.0
    }
  ]
}
```

**Run Test:**
```bash
$body = Get-Content test_batch.json -Raw
Invoke-RestMethod -Uri http://localhost:5000/batch-predict -Method Post -Body $body -ContentType "application/json"
```

**Expected Response:**
```json
{
  "results": [
    {
      "machine_id": "M001",
      "failure_probability": 0.2-0.4,
      "prediction": 0,
      "risk_level": "low"
    },
    {
      "machine_id": "M002",
      "failure_probability": 0.6-0.8,
      "prediction": 1,
      "risk_level": "high"
    }
  ],
  "total_samples": 2
}
```

**✅ Success Indicator:** Both predictions returned correctly

---

## Demonstration Checklist for Project Guide

### Week 3: XAI Analysis

**Files to Show:**
1. ✅ `outputs/week3_xai_final_report.md` - **MAIN REPORT**
2. ✅ `outputs/week3_xai_presentation.md` - Presentation slides
3. ✅ `outputs/physical_validation_report.md` - Validation findings
4. ✅ `outputs/plots/shap_summary.png` - Key visualization
5. ✅ `outputs/feature_importance_analysis.csv` - Feature rankings

**Key Points to Highlight:**
- ✅ SHAP analysis completed for all 25 features
- ✅ Top feature: Temperature (highest importance)
- ✅ 22 visualizations created (summary, force, dependence, waterfall)
- ✅ Physical validation performed (52% consistency)
- ✅ 7 anomalies identified with recommendations
- ✅ Comprehensive documentation for all team members

### Week 4: Model-as-a-Service

**Live Demonstration:**
1. ✅ Start API server (`python app.py`)
2. ✅ Show health check endpoint
3. ✅ Test normal risk prediction
4. ✅ Test high risk prediction
5. ✅ Show SHAP explanations in response
6. ✅ Demonstrate error handling

**Files to Show:**
1. ✅ `app.py` - Flask API implementation
2. ✅ `API_README.md` - Complete documentation
3. ✅ `PERFORMANCE_REPORT.md` - Test results
4. ✅ `utils/feature_processor.py` - Feature processing logic

**Key Points to Highlight:**
- ✅ Real-time predictions with <50ms latency
- ✅ SHAP explanations for every prediction
- ✅ 4 API endpoints (health, model-info, predict, batch-predict)
- ✅ Comprehensive input validation
- ✅ Production-ready with error handling
- ✅ Complete API documentation

---

## Quick Demo Script for Project Guide

### 5-Minute Demo

**1. Week 3 Overview (2 minutes)**
```
"For Week 3, we implemented Explainable AI using SHAP values."

[Open: outputs/week3_xai_final_report.md]

"This consolidated report shows:
- Top 10 most important features for predicting failures
- Temperature is the #1 predictor
- We created 22 visualizations to explain the model
- Physical validation identified 7 anomalies
- 52% consistency rate suggests model needs refinement"

[Show: outputs/plots/shap_summary.png]

"This visualization shows how each feature impacts predictions."
```

**2. Week 4 Demo (3 minutes)**
```
"For Week 4, we built a Model-as-a-Service API."

[Terminal 1: python app.py]

"The API loads the model and SHAP explainer at startup."

[Terminal 2: Test health check]

"Health check confirms the API is running."

[Terminal 2: Test normal risk prediction]

"For normal sensor readings, we get low failure probability."

[Terminal 2: Test high risk prediction]

"For abnormal readings, we get high failure probability with explanation.
Notice the SHAP values show temperature is the main contributor."

[Show: API_README.md]

"Complete documentation with examples and deployment guide."
```

---

## Troubleshooting

### API Won't Start
**Error:** `Model not found`
**Solution:** Ensure `models/xgboost_best.pkl` exists. Run `python src/xgboost_tuning.py` if needed.

### Curl Not Available
**Solution:** Use PowerShell `Invoke-RestMethod` or `Invoke-WebRequest` instead.

### Port Already in Use
**Error:** `Address already in use`
**Solution:** Stop other Flask instances or change port in `app.py`.

---

## Summary

**Week 3 Deliverables:** 7 documents + 22 visualizations  
**Week 4 Deliverables:** 1 API + 4 endpoints + complete documentation

**Total Implementation Time:** ~2 weeks  
**Status:** ✅ Production Ready
