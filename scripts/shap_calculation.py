"""
FactoryGuard AI - SHAP Values Calculation & Analysis
Akshada - Week 3, Days 1-3

This script calculates and analyzes SHAP values for model interpretability.
SHAP (SHapley Additive exPlanations) explains individual predictions.
"""

import pandas as pd
import numpy as np
import shap
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def load_best_model():
    """
    Load the best performing model from Week 2
    
    Returns:
        model: Trained model
        scaler: Fitted StandardScaler
        feature_names: List of feature names
    """
    print("=" * 70)
    print("LOADING BEST MODEL FROM WEEK 2")
    print("=" * 70)
    
    # Check if saved model exists
    if os.path.exists('models/logistic_regression_baseline.pkl'):
        print("\nLoading saved model...")
        with open('models/logistic_regression_baseline.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("Model and scaler loaded successfully!")
    else:
        print("\nSaved model not found. Retraining baseline model...")
        model, scaler = train_baseline_model()
    
    return model, scaler


def train_baseline_model():
    """
    Retrain baseline model if saved version not found
    """
    print("Training Logistic Regression baseline model...")
    
    # Load data
    df = pd.read_csv('data/processed/model_ready_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Prepare features
    exclude_cols = ['timestamp', 'machine_id', 'failure', 'failure_within_24h']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['failure_within_24h']
    
    # Temporal split
    timestamps = df['timestamp']
    sort_idx = timestamps.argsort()
    X = X.iloc[sort_idx]
    y = y.iloc[sort_idx]
    
    split_idx = int(len(X) * 0.7)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    
    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        solver='lbfgs'
    )
    model.fit(X_train_scaled, y_train)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/logistic_regression_baseline.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Model trained and saved!")
    
    return model, scaler


def load_test_data():
    """
    Load test dataset for SHAP analysis
    
    Returns:
        X_test: Test features (scaled)
        X_test_raw: Test features (unscaled, for display)
        y_test: Test labels
        feature_names: List of feature names
    """
    print("\n" + "=" * 70)
    print("LOADING TEST DATASET")
    print("=" * 70)
    
    # Load full dataset
    df = pd.read_csv('data/processed/model_ready_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Prepare features
    exclude_cols = ['timestamp', 'machine_id', 'failure', 'failure_within_24h']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['failure_within_24h']
    timestamps = df['timestamp']
    
    # Temporal split (same as Week 2)
    sort_idx = timestamps.argsort()
    X = X.iloc[sort_idx]
    y = y.iloc[sort_idx]
    
    split_idx = int(len(X) * 0.7)
    X_test_raw = X.iloc[split_idx:].copy()
    y_test = y.iloc[split_idx:].copy()
    
    print(f"\nTest set size: {len(X_test_raw)} samples")
    print(f"Features: {len(feature_cols)}")
    print(f"Failures in test set: {y_test.sum()} ({(y_test.sum()/len(y_test)*100):.2f}%)")
    
    return X_test_raw, y_test, feature_cols


def calculate_shap_values(model, scaler, X_test_raw, sample_size=None):
    """
    Calculate SHAP values for test set
    
    Args:
        model: Trained model
        scaler: Fitted StandardScaler
        X_test_raw: Test features (unscaled)
        sample_size: If set, use only this many samples (for speed)
        
    Returns:
        shap_values: SHAP values array
        explainer: SHAP explainer object
        X_test_scaled: Scaled test features
    """
    print("\n" + "=" * 70)
    print("CALCULATING SHAP VALUES")
    print("=" * 70)
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test_raw)
    
    # Use sample if specified (SHAP can be slow on large datasets)
    if sample_size and sample_size < len(X_test_scaled):
        print(f"\nUsing random sample of {sample_size} instances for SHAP calculation...")
        np.random.seed(42)
        sample_idx = np.random.choice(len(X_test_scaled), sample_size, replace=False)
        X_test_sample = X_test_scaled[sample_idx]
        X_test_raw_sample = X_test_raw.iloc[sample_idx]
    else:
        X_test_sample = X_test_scaled
        X_test_raw_sample = X_test_raw
        print(f"\nCalculating SHAP values for all {len(X_test_scaled)} test instances...")
    
    # Create SHAP explainer
    # For Logistic Regression, we use LinearExplainer (faster and exact)
    print("Creating SHAP explainer (LinearExplainer for Logistic Regression)...")
    explainer = shap.LinearExplainer(model, X_test_sample)
    
    # Calculate SHAP values
    print("Computing SHAP values... (this may take a few minutes)")
    shap_values = explainer.shap_values(X_test_sample)
    
    # For binary classification, shap_values might be 2D array
    # We want SHAP values for positive class (failure prediction)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class
    
    print(f"\nSHAP values calculated!")
    print(f"SHAP values shape: {shap_values.shape}")
    
    return shap_values, explainer, X_test_sample, X_test_raw_sample


def analyze_feature_importance(shap_values, feature_names, X_test_raw_sample):
    """
    Perform statistical analysis on SHAP values
    
    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        X_test_raw_sample: Raw feature values
        
    Returns:
        importance_df: DataFrame with feature importance analysis
    """
    print("\n" + "=" * 70)
    print("ANALYZING FEATURE IMPORTANCE")
    print("=" * 70)
    
    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Calculate other statistics
    mean_shap = shap_values.mean(axis=0)
    std_shap = shap_values.std(axis=0)
    max_shap = np.abs(shap_values).max(axis=0)
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap,
        'mean_shap': mean_shap,
        'std_shap': std_shap,
        'max_abs_shap': max_shap
    })
    
    # Sort by mean absolute SHAP value (most important first)
    importance_df = importance_df.sort_values('mean_abs_shap', ascending=False)
    importance_df['rank'] = range(1, len(importance_df) + 1)
    
    print("\nTOP 10 MOST IMPORTANT FEATURES (by mean absolute SHAP value):")
    print("=" * 70)
    print(importance_df.head(10).to_string(index=False))
    
    print("\n\nFEATURE IMPORTANCE SUMMARY:")
    print("-" * 70)
    print(f"Total features analyzed: {len(feature_names)}")
    print(f"Mean SHAP value range: [{mean_shap.min():.4f}, {mean_shap.max():.4f}]")
    print(f"Most important feature: {importance_df.iloc[0]['feature']}")
    print(f"  - Mean |SHAP|: {importance_df.iloc[0]['mean_abs_shap']:.4f}")
    
    return importance_df


def save_shap_results(shap_values, explainer, importance_df, X_test_sample, feature_names):
    """
    Save SHAP values and analysis results for team sharing
    
    Args:
        shap_values: SHAP values array
        explainer: SHAP explainer object
        importance_df: Feature importance DataFrame
        X_test_sample: Test features used
        feature_names: List of feature names
    """
    print("\n" + "=" * 70)
    print("SAVING SHAP RESULTS")
    print("=" * 70)
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # Save SHAP values and related data
    shap_data = {
        'shap_values': shap_values,
        'explainer': explainer,
        'feature_names': feature_names,
        'X_test_sample': X_test_sample
    }
    
    with open('outputs/shap_values.pkl', 'wb') as f:
        pickle.dump(shap_data, f)
    print("SHAP values saved to: outputs/shap_values.pkl")
    
    # Save feature importance analysis as CSV
    importance_df.to_csv('outputs/feature_importance_analysis.csv', index=False)
    print("Feature importance saved to: outputs/feature_importance_analysis.csv")
    
    # Create detailed documentation
    create_documentation(importance_df, shap_values, feature_names)


def create_documentation(importance_df, shap_values, feature_names):
    """
    Create detailed documentation of SHAP methodology and findings
    """
    doc_path = 'outputs/shap_methodology_documentation.md'
    
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write("# SHAP Values Calculation & Analysis Documentation\n")
        f.write("**Week 3 - Akshada's Deliverable**\n\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("## What are SHAP Values?\n\n")
        f.write("SHAP (SHapley Additive exPlanations) values explain how much each feature\n")
        f.write("contributes to pushing a prediction away from the base value (average prediction).\n\n")
        
        f.write("**Key Properties:**\n")
        f.write("- Based on game theory (Shapley values)\n")
        f.write("- Each feature gets a fair share of credit for prediction\n")
        f.write("- Positive SHAP = pushes toward failure prediction\n")
        f.write("- Negative SHAP = pushes toward normal prediction\n\n")
        
        f.write("=" * 70 + "\n\n")
        
        f.write("## Methodology\n\n")
        f.write("**Model Used:** Logistic Regression (baseline model from Week 2)\n\n")
        f.write("**SHAP Explainer:** LinearExplainer (exact, fast for linear models)\n\n")
        f.write("**Dataset:**\n")
        f.write(f"- Test set size: {len(shap_values)} instances\n")
        f.write(f"- Number of features: {len(feature_names)}\n\n")
        
        f.write("**Calculation Steps:**\n")
        f.write("1. Loaded trained Logistic Regression model\n")
        f.write("2. Scaled test data using fitted StandardScaler\n")
        f.write("3. Created SHAP LinearExplainer\n")
        f.write("4. Calculated SHAP values for all test predictions\n")
        f.write("5. Analyzed feature importance using mean |SHAP|\n\n")
        
        f.write("=" * 70 + "\n\n")
        
        f.write("## Feature Importance Rankings\n\n")
        f.write("Features ranked by mean absolute SHAP value:\n\n")
        f.write("```\n")
        f.write(importance_df.head(15).to_string(index=False))
        f.write("\n```\n\n")
        
        f.write("=" * 70 + "\n\n")
        
        f.write("## Key Findings\n\n")
        
        top_5 = importance_df.head(5)
        f.write("**Top 5 Most Important Features:**\n\n")
        for i, row in top_5.iterrows():
            f.write(f"{row['rank']}. **{row['feature']}**\n")
            f.write(f"   - Mean |SHAP|: {row['mean_abs_shap']:.4f}\n")
            f.write(f"   - Mean SHAP: {row['mean_shap']:.4f}\n")
            direction = "increases" if row['mean_shap'] > 0 else "decreases"
            f.write(f"   - Impact: {direction} failure probability\n\n")
        
        f.write("=" * 70 + "\n\n")
        
        f.write("## Statistical Summary\n\n")
        f.write(f"- **Total features analyzed:** {len(feature_names)}\n")
        f.write(f"- **Mean SHAP value range:** [{shap_values.mean(axis=0).min():.4f}, {shap_values.mean(axis=0).max():.4f}]\n")
        f.write(f"- **Most variable feature:** {importance_df.sort_values('std_shap', ascending=False).iloc[0]['feature']}\n")
        f.write(f"- **Most consistent feature:** {importance_df.sort_values('std_shap').iloc[0]['feature']}\n\n")
        
        f.write("=" * 70 + "\n\n")
        
        f.write("## Next Steps (for Team)\n\n")
        f.write("**Harish's Tasks:**\n")
        f.write("- Create SHAP summary plots using shap_values.pkl\n")
        f.write("- Generate force plots for individual predictions\n")
        f.write("- Create dependence plots for top 5 features\n\n")
        
        f.write("**Krutika's Tasks:**\n")
        f.write("- Validate feature impacts against physical expectations\n")
        f.write("- Check if temperature/vibration impacts make sense\n")
        f.write("- Prepare consolidated XAI report\n\n")
        
        f.write("=" * 70 + "\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("- `outputs/shap_values.pkl` - SHAP values for team use\n")
        f.write("- `outputs/feature_importance_analysis.csv` - Detailed rankings\n")
        f.write("- `outputs/shap_methodology_documentation.md` - This file\n\n")
        
        f.write("**Prepared by:** Akshada\n")
        f.write("**Date:** Week 3, FactoryGuard AI Project\n")
    
    print(f"Documentation saved to: {doc_path}")


def main():
    """
    Main execution function for Akshada's SHAP calculation task
    """
    print("\n" + "=" * 70)
    print("AKSHADA'S SHAP VALUES CALCULATION & ANALYSIS")
    print("FactoryGuard AI - Week 3")
    print("=" * 70 + "\n")
    
    # Step 1: Load best model
    model, scaler = load_best_model()
    
    # Step 2: Load test data
    X_test_raw, y_test, feature_names = load_test_data()
    
    # Step 3: Calculate SHAP values
    # Note: Using sample_size=500 for speed. Remove this parameter to use all data.
    shap_values, explainer, X_test_sample, X_test_raw_sample = calculate_shap_values(
        model, scaler, X_test_raw, sample_size=500
    )
    
    # Step 4: Analyze feature importance
    importance_df = analyze_feature_importance(shap_values, feature_names, X_test_raw_sample)
    
    # Step 5: Save results
    save_shap_results(shap_values, explainer, importance_df, X_test_sample, feature_names)
    
    print("\n" + "=" * 70)
    print("SUCCESS: AKSHADA'S TASKS COMPLETE!")
    print("=" * 70)
    print("\nDeliverables:")
    print("  [OK] outputs/shap_values.pkl - SHAP values for team")
    print("  [OK] outputs/feature_importance_analysis.csv - Rankings")
    print("  [OK] outputs/shap_methodology_documentation.md - Documentation")
    print("\nNext Steps:")
    print("  -> Hand off shap_values.pkl to Harish for visualizations")
    print("  -> Share feature_importance_analysis.csv with Krutika for validation")
    print("  -> Team meeting to review findings")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()