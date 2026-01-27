"""
FactoryGuard AI - Baseline Model & Correlation Analysis
Akshada - Week 2, Days 1-2

This script establishes baseline performance with Logistic Regression
and analyzes feature correlations.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    f1_score, 
    recall_score, 
    precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_modeling_data(filepath='data/processed/model_ready_data.csv'):
    """Load the feature-engineered dataset"""
    print("=" * 70)
    print("LOADING MODELING DATA")
    print("=" * 70)
    
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"\n✓ Loaded {len(df):,} records")
    print(f"✓ Total features: {len(df.columns)}")
    
    return df


def prepare_features(df):
    """Prepare feature matrix and target variable"""
    print("\n" + "=" * 70)
    print("PREPARING FEATURES")
    print("=" * 70)
    
    # Exclude non-feature columns
    exclude_cols = ['timestamp', 'machine_id', 'failure', 'failure_within_24h']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['failure_within_24h']
    
    print(f"\n✓ Features: {len(feature_cols)}")
    print(f"✓ Target: failure_within_24h")
    print(f"✓ Samples: {len(X):,}")
    
    return X, y, feature_cols, df['timestamp']


def temporal_train_test_split(X, y, timestamps, train_ratio=0.7):
    """
    Split data temporally (no shuffling to prevent data leakage)
    """
    print("\n" + "=" * 70)
    print("TEMPORAL TRAIN-TEST SPLIT")
    print("=" * 70)
    
    # Sort by timestamp
    sort_idx = timestamps.argsort()
    X = X.iloc[sort_idx]
    y = y.iloc[sort_idx]
    timestamps = timestamps.iloc[sort_idx]
    
    # Split at 70% point
    split_idx = int(len(X) * train_ratio)
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"\n✓ Train size: {len(X_train):,} ({train_ratio*100:.0f}%)")
    print(f"✓ Test size: {len(X_test):,} ({(1-train_ratio)*100:.0f}%)")
    print(f"\n✓ Train period: {timestamps.iloc[0]} to {timestamps.iloc[split_idx-1]}")
    print(f"✓ Test period: {timestamps.iloc[split_idx]} to {timestamps.iloc[-1]}")
    
    # Check class distribution
    print(f"\nTrain class distribution:")
    print(f"  Normal (0): {(y_train==0).sum():,}")
    print(f"  Failure (1): {(y_train==1).sum():,}")
    print(f"  Imbalance ratio: {(y_train==0).sum() / (y_train==1).sum():.1f}:1")
    
    return X_train, X_test, y_train, y_test


def analyze_correlations(X_train, y_train, feature_cols):
    """
    Analyze and visualize feature correlations
    """
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)
    
    # Create correlation matrix
    corr_data = X_train.copy()
    corr_data['target'] = y_train
    corr_matrix = corr_data.corr()
    
    # Get correlations with target
    target_corr = corr_matrix['target'].drop('target').sort_values(ascending=False)
    
    print("\nTop 10 Features Correlated with Target:")
    print(target_corr.head(10))
    
    print("\nBottom 10 Features Correlated with Target:")
    print(target_corr.tail(10))
    
    # Plot full correlation heatmap
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                cbar_kws={'label': 'Correlation'})
    plt.title('Feature Correlation Matrix (Akshada - Baseline Analysis)', 
              fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs('reports', exist_ok=True)
    plt.savefig('reports/akshada_correlation_matrix.png', dpi=300)
    print("\n✓ Saved: reports/akshada_correlation_matrix.png")
    plt.close()
    
    # Plot top feature correlations with target
    plt.figure(figsize=(10, 8))
    top_features = target_corr.head(20)
    top_features.plot(kind='barh', color='steelblue')
    plt.title('Top 20 Features by Target Correlation', fontsize=14, fontweight='bold')
    plt.xlabel('Correlation with failure_within_24h')
    plt.tight_layout()
    plt.savefig('reports/akshada_top_features.png', dpi=300)
    print("✓ Saved: reports/akshada_top_features.png")
    plt.close()
    
    return target_corr


def train_baseline_model(X_train, X_test, y_train, y_test):
    """
    Train Logistic Regression baseline model
    """
    print("\n" + "=" * 70)
    print("TRAINING BASELINE MODEL - LOGISTIC REGRESSION")
    print("=" * 70)
    
    # Standardize features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression with class balancing
    print("Training model...")
    lr = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',  # Handle imbalance
        random_state=42,
        solver='lbfgs',
        n_jobs=-1
    )
    
    lr.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = lr.predict(X_test_scaled)
    y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate
    print("\n" + "=" * 70)
    print("BASELINE MODEL RESULTS")
    print("=" * 70)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Failure']))
    
    # Key metrics
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    
    print(f"\n{'='*70}")
    print("KEY METRICS (Baseline)")
    print(f"{'='*70}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC:   {auc:.4f}")
    except:
        auc = None
        print("ROC-AUC:   N/A")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Failure'],
                yticklabels=['Normal', 'Failure'])
    plt.title('Confusion Matrix - Baseline Logistic Regression', 
              fontsize=14, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('reports/akshada_baseline_confusion_matrix.png', dpi=300)
    print("\n✓ Saved: reports/akshada_baseline_confusion_matrix.png")
    plt.close()
    
    # ROC Curve
    if auc:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Baseline Model', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('reports/akshada_baseline_roc_curve.png', dpi=300)
        print("✓ Saved: reports/akshada_baseline_roc_curve.png")
        plt.close()
    
    return lr, scaler, {'f1': f1, 'recall': recall, 'precision': precision, 'auc': auc}


def generate_baseline_report(metrics, target_corr):
    """Generate comprehensive baseline report"""
    print("\n" + "=" * 70)
    print("GENERATING BASELINE REPORT")
    print("=" * 70)
    
    with open('reports/akshada_baseline_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("AKSHADA'S BASELINE MODEL REPORT\n")
        f.write("FactoryGuard AI - Week 2, Days 1-2\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("MODEL: LOGISTIC REGRESSION (Baseline)\n")
        f.write("-" * 70 + "\n\n")
        
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"F1-Score:  {metrics['f1']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        if metrics['auc']:
            f.write(f"ROC-AUC:   {metrics['auc']:.4f}\n")
        f.write("\n")
        
        f.write("TOP 10 MOST IMPORTANT FEATURES (by correlation)\n")
        f.write("-" * 70 + "\n")
        for i, (feat, corr) in enumerate(target_corr.head(10).items(), 1):
            f.write(f"{i:2d}. {feat:40s} {corr:+.4f}\n")
        f.write("\n")
        
        f.write("TASKS COMPLETED\n")
        f.write("-" * 70 + "\n")
        f.write("[OK] Baseline model trained (Logistic Regression)\n")
        f.write("[OK] Correlation analysis completed\n")
        f.write("[OK] Feature importance identified\n")
        f.write("[OK] Performance metrics established\n\n")
        
        f.write("NEXT STEPS\n")
        f.write("-" * 70 + "\n")
        f.write("-> Support Harish with Random Forest implementation\n")
        f.write("-> Prepare SMOTE for handling class imbalance\n")
        f.write("-> Compare baseline with advanced models\n\n")
        
        f.write("=" * 70 + "\n")
    
    print("✓ Saved: reports/akshada_baseline_report.txt")


def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print("AKSHADA'S BASELINE MODEL PIPELINE")
    print("FactoryGuard AI - Week 2")
    print("=" * 70 + "\n")
    
    # Load data
    df = load_modeling_data()
    
    # Prepare features
    X, y, feature_cols, timestamps = prepare_features(df)
    
    # Temporal split
    X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, timestamps)
    
    # Correlation analysis
    target_corr = analyze_correlations(X_train, y_train, feature_cols)
    
    # Train baseline model
    model, scaler, metrics = train_baseline_model(X_train, X_test, y_train, y_test)
    
    # Generate report
    generate_baseline_report(metrics, target_corr)
    
    print("\n" + "=" * 70)
    print("✅ AKSHADA'S BASELINE MODELING COMPLETE!")
    print("=" * 70)
    print("\n📦 Deliverables:")
    print("  ✓ Baseline F1-Score: {:.4f}".format(metrics['f1']))
    print("  ✓ Baseline Recall: {:.4f}".format(metrics['recall']))
    print("  ✓ Correlation matrix generated")
    print("  ✓ Feature importance analysis complete")


if __name__ == "__main__":
    main()