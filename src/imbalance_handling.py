
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, recall_score
from sklearn.model_selection import train_test_split
import os

# Load modeling-ready data
try:
    df = pd.read_csv('data/processed/model_ready_data.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: data/processed/model_ready_data.csv not found.")
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

# Apply SMOTE to training data
print("Applying SMOTE...")
# Handle potential issue if minority class is too small for 0.3 ratio
# But assuming robust data for now.
try:
    smote = SMOTE(sampling_strategy=0.3, random_state=42) # 30% minority class
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"Original class distribution:\n{y_train.value_counts()}")
    print(f"Balanced class distribution:\n{y_train_balanced.value_counts()}")
except Exception as e:
    print(f"SMOTE failed: {e}")
    # Fallback to original if SMOTE fails (e.g., extremely imbalanced or not enough samples)
    X_train_balanced, y_train_balanced = X_train, y_train

# Train Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_balanced, y_train_balanced)

print("Evaluating Random Forest...")
y_pred_rf = rf.predict(X_test)
print("\n=== Random Forest Results ===")
print(classification_report(y_test, y_pred_rf))
print(f"F1-Score: {f1_score(y_test, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features:")
print(feature_importance.head(10))

# Save feature importance
os.makedirs('reports', exist_ok=True)
feature_importance.to_csv('reports/feature_importance_rf.csv', index=False)
print("Feature importance saved to reports/feature_importance_rf.csv")
