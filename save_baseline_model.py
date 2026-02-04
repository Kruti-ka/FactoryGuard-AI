"""
Save Baseline Model for SHAP Analysis
Run this BEFORE shap_calculation.py if you haven't saved your model yet
"""

import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def save_baseline_model():
    """
    Train and save baseline model for SHAP analysis
    """
    print("=" * 70)
    print("SAVING BASELINE MODEL FOR SHAP ANALYSIS")
    print("=" * 70)
    
    # Load data
    print("\nLoading model-ready data...")
    df = pd.read_csv('data/processed/model_ready_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Prepare features
    exclude_cols = ['timestamp', 'machine_id', 'failure', 'failure_within_24h']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['failure_within_24h']
    timestamps = df['timestamp']
    
    # Temporal split (70-30)
    print("Performing temporal train-test split...")
    sort_idx = timestamps.argsort()
    X = X.iloc[sort_idx]
    y = y.iloc[sort_idx]
    
    split_idx = int(len(X) * 0.7)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    
    print(f"Training samples: {len(X_train)}")
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    print("Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        solver='lbfgs'
    )
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    
    with open('models/logistic_regression_baseline.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("\n[OK] Model saved: models/logistic_regression_baseline.pkl")
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("[OK] Scaler saved: models/scaler.pkl")
    
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    print("[OK] Feature names saved: models/feature_names.pkl")
    
    print("\n" + "=" * 70)
    print("Model saved successfully! Ready for SHAP analysis.")
    print("=" * 70)


if __name__ == "__main__":
    save_baseline_model()