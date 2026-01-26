# FactoryGuard AI - IoT Predictive Maintenance Engine

**Product Brand Name:** FactoryGuard AI  
**Project Type:** Industrial AI - Predictive Maintenance

## 🎯 Project Overview

FactoryGuard AI is a predictive maintenance solution designed for a manufacturing facility with 500 robotic arms. The system predicts equipment failure 24 hours in advance using real-time sensor data (vibration, temperature, pressure), preventing costly unplanned downtime of $10,000 per hour.

## 🏗️ Project Structure

```
factoryguard-ai/
├── data/
│   ├── raw/              # Raw CSV sensor logs
│   └── processed/        # Cleaned, feature-engineered data
├── src/
│   ├── data_ingestion.py      # Member A - Week 1
│   ├── data_cleaning.py       # Member A - Week 1
│   ├── feature_engineering.py # Member B - Week 1
│   └── utils.py               # Shared utilities
├── notebooks/
│   └── eda.ipynb             # Exploratory Data Analysis
├── tests/
│   └── test_features.py      # Unit tests
├── models/
│   └── xgboost_best.pkl      # Final trained model
└── requirements.txt
```

## 👥 Team Roles

- **Akshada (Data Specialist)**: Data ingestion, cleaning, baseline models
- **Harish (Feature Engineer)**: Temporal feature engineering, model training
- **Krutika (Leader)**: Integration, validation, hyperparameter tuning

## 📅 4-Week Implementation Plan

### Week 1: Data Engineering
- ✅ Data ingestion and EDA
- ✅ Missing value handling
- ✅ Temporal feature engineering (rolling windows, EMA, lag features)
- ✅ Data leakage validation

### Week 2: Modeling & Optimization
- [ ] Logistic Regression baseline
- [ ] Class imbalance handling (SMOTE)
- [ ] Random Forest model
- [ ] XGBoost optimization (RandomizedSearchCV)

### Week 3: Explainability (XAI)
- [ ] SHAP value calculation
- [ ] Force plots and summary plots
- [ ] Physical validation

### Week 4: Deployment
- [ ] Model serialization
- [ ] Flask API development
- [ ] SHAP integration in API
- [ ] Latency testing (< 50ms target)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Data Pipeline (Week 1)
```python
# Example: Process raw sensor data
from src.data_ingestion import load_sensor_data
from src.data_cleaning import handle_missing_values, create_target_variable
from src.feature_engineering import create_all_features

# Load data
df = load_sensor_data('data/raw/sensor_logs.csv')

# Clean data
df = handle_missing_values(df)
df = create_target_variable(df, failure_window_hours=24)

# Create features
df = create_all_features(df)

# Save
df.to_csv('data/processed/model_ready_data.csv', index=False)
```

### 3. Train Model (Week 2)
```python
# Coming in Week 2
```

## 📊 Key Features

- **Temporal Feature Engineering**: Rolling windows (1h, 4h, 8h), Exponential Moving Averages
- **Imbalance Handling**: SMOTE for rare failure events (<1% of data)
- **High-Performance Model**: XGBoost with hyperparameter optimization
- **Explainability**: SHAP for local interpretability
- **Production-Ready**: Flask API with <50ms latency

## 🎯 Success Criteria

- **Week 1**: 50+ engineered features, no data leakage
- **Week 2**: F1-Score > 0.70, Recall > 0.75
- **Week 3**: SHAP explanations validated against physics
- **Week 4**: API latency < 50ms

## 📚 Technology Stack

- **Core**: Python 3.9+
- **ML Libraries**: Scikit-learn, XGBoost, imbalanced-learn
- **Interpretability**: SHAP
- **API**: Flask
- **Visualization**: Matplotlib, Seaborn

## 📝 Current Status

**Week 1 - In Progress**  
✅ Project structure created  
✅ Core modules implemented  
⏳ Waiting for sensor data

---

**FactoryGuard AI** - Preventing failures before they happen 🛡️
