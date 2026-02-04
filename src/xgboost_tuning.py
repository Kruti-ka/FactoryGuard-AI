
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score
from scipy.stats import randint, uniform
import joblib
import os
from pathlib import Path

# Get project root directory (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'model_ready_data.csv'
MODELS_DIR = PROJECT_ROOT / 'models'

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Load modeling-ready data
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded successfully from: {DATA_PATH}")
except FileNotFoundError:
    print(f"Error: {DATA_PATH} not found.")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root: {PROJECT_ROOT}")
    exit(1)

# Prepare features and target
# Exclude non-numeric columns and target columns
exclude_cols = ['timestamp', 'machine_id', 'failure', 'failure_within_24h']
# Select only numeric columns for features, just in case there are other string columns
feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
X = df[feature_cols]
y = df['failure_within_24h']

# Train-test split (temporal)
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    train_cutoff = df['timestamp'].quantile(0.7)
    train_mask = df['timestamp'] <= train_cutoff
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Calculate scale_pos_weight for imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

# XGBoost baseline
print("Setting up XGBoost RandomizedSearchCV...")
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    tree_method='hist',
    eval_metric='aucpr' # Precision-Recall AUC
)

# Hyperparameter search space
param_distributions = {
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'n_estimators': randint(100, 500),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'min_child_weight': randint(1, 10),
    'gamma': uniform(0, 0.5)
}

# RandomizedSearchCV with F1 scoring
# Note: n_iter=50 might take a while. Reduce for testing if needed.
random_search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_distributions,
    n_iter=50,
    scoring='f1',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

print("Starting hyperparameter tuning...")
random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
print("Best F1-Score (CV):", random_search.best_score_)

# Evaluate best model
best_xgb = random_search.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)
y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]

print("\n=== XGBoost Results ===")
print(classification_report(y_test, y_pred_xgb))
print(f"F1-Score: {f1_score(y_test, y_pred_xgb):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_xgb):.4f}")

# Save the best model with versioning
from datetime import datetime
import json

model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
versioned_model_path = MODELS_DIR / f'xgboost_best_v{model_version}.pkl'
model_path = MODELS_DIR / 'xgboost_best.pkl'  # Keep for backward compatibility

# Save versioned model
joblib.dump(best_xgb, versioned_model_path)
print(f"Versioned model saved to {versioned_model_path}")

# Save current model (backward compatibility)
joblib.dump(best_xgb, model_path)
print(f"Model saved to {model_path}")

# Save feature names
feature_names_path = MODELS_DIR / 'feature_names.pkl'
joblib.dump(feature_cols, feature_names_path)
print(f"Feature names saved to {feature_names_path}")

# Save model metadata
metadata = {
    "version": model_version,
    "model_path": str(versioned_model_path),
    "training_date": datetime.now().isoformat(),
    "f1_score": float(f1_score(y_test, y_pred_xgb)),
    "recall": float(recall_score(y_test, y_pred_xgb)),
    "feature_count": len(feature_cols),
    "best_params": random_search.best_params_,
    "scale_pos_weight": float(scale_pos_weight)
}

metadata_path = MODELS_DIR / 'model_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Model metadata saved to {metadata_path}")
