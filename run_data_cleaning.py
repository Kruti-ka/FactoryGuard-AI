"""
Akshada's Data Cleaning Pipeline Runner
Week 1, Days 3-4

This script runs the complete data cleaning pipeline for FactoryGuard AI.
"""

import sys
import os
sys.path.append('src')

from data_cleaning import clean_pipeline, handle_missing_values, remove_outliers, create_target_variable
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    """
    Main execution function for Akshada's data cleaning tasks
    """
    
    print("=" * 70)
    print("AKSHADA'S DATA CLEANING PIPELINE - FACTORYGUARD AI")
    print("=" * 70)
    
    # Step 1: Generate sample data (if not exists)
    if not os.path.exists('data/raw/sensor_logs.csv'):
        print("\n⚠️  No raw data found. Generating sample data...")
        sys.path.append('src')
        from generate_sample_data import generate_sample_sensor_data
        generate_sample_sensor_data(
            n_machines=10,
            n_days=30,
            hourly_samples=1,
            failure_rate=0.005
        )
    
    # Step 2: Run the complete cleaning pipeline
    print("\n" + "=" * 70)
    print("STARTING CLEANING PIPELINE")
    print("=" * 70)
    
    cleaned_df = clean_pipeline(
        input_path='data/raw/sensor_logs.csv',
        output_path='data/processed/clean_data.csv'
    )
    
    # Step 3: Generate validation report
    generate_cleaning_report(cleaned_df)
    
    print("\n" + "=" * 70)
    print("✅ AKSHADA'S TASKS COMPLETE!")
    print("=" * 70)
    print("\n📦 Deliverables:")
    print("  ✓ Clean dataset: data/processed/clean_data.csv")
    print("  ✓ Target variable: 'failure_within_24h' created")
    print("  ✓ Missing values handled")
    print("  ✓ Outliers removed")
    print("\n📊 Next Steps:")
    print("  → Hand off to Harish for feature engineering")
    print("  → Prepare for Week 2 baseline modeling")
    

def generate_cleaning_report(df):
    """
    Generate a comprehensive cleaning report with visualizations
    """
    print("\n" + "=" * 70)
    print("GENERATING CLEANING REPORT")
    print("=" * 70)
    
    # Create reports directory
    os.makedirs('reports', exist_ok=True)
    
    # 1. Class Distribution Plot
    plt.figure(figsize=(10, 6))
    df['failure_within_24h'].value_counts().plot(kind='bar', color=['green', 'red'])
    plt.title('Target Variable Distribution - failure_within_24h', fontsize=14, fontweight='bold')
    plt.xlabel('Class (0=Normal, 1=Failure within 24h)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    for i, v in enumerate(df['failure_within_24h'].value_counts()):
        plt.text(i, v + 100, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/akshada_class_distribution.png', dpi=300)
    print("✓ Saved: reports/akshada_class_distribution.png")
    plt.close()
    
    # 2. Sensor Statistics After Cleaning
    plt.figure(figsize=(15, 4))
    sensor_cols = ['vibration', 'temperature', 'pressure']
    
    for i, col in enumerate(sensor_cols, 1):
        plt.subplot(1, 3, i)
        plt.hist(df[col], bins=50, edgecolor='black', alpha=0.7)
        plt.title(f'{col.capitalize()} After Cleaning')
        plt.xlabel(col.capitalize())
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('reports/akshada_sensor_distributions.png', dpi=300)
    print("✓ Saved: reports/akshada_sensor_distributions.png")
    plt.close()
    
    # 3. Summary Statistics Report
    with open('reports/akshada_cleaning_summary.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("AKSHADA'S DATA CLEANING SUMMARY REPORT\n")
        f.write("FactoryGuard AI - Week 1\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Records: {len(df):,}\n")
        f.write(f"Unique Machines: {df['machine_id'].nunique()}\n")
        f.write(f"Time Range: {df['timestamp'].min()} to {df['timestamp'].max()}\n\n")
        
        f.write("TARGET VARIABLE (failure_within_24h)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Normal Records (0): {(df['failure_within_24h']==0).sum():,}\n")
        f.write(f"Failure Records (1): {(df['failure_within_24h']==1).sum():,}\n")
        f.write(f"Imbalance Ratio: {(df['failure_within_24h']==0).sum() / (df['failure_within_24h']==1).sum():.1f}:1\n\n")
        
        f.write("SENSOR STATISTICS AFTER CLEANING\n")
        f.write("-" * 70 + "\n")
        f.write(df[sensor_cols].describe().to_string())
        f.write("\n\n")
        
        f.write("MISSING VALUES\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Missing: {df.isnull().sum().sum()}\n")
        f.write(df.isnull().sum().to_string())
        f.write("\n\n")
        
        f.write("TASKS COMPLETED\n")
        f.write("-" * 70 + "\n")
        f.write("✓ Missing value imputation (interpolation method)\n")
        f.write("✓ Outlier removal (4 standard deviations)\n")
        f.write("✓ Target variable creation (24-hour failure window)\n")
        f.write("✓ Data validation checks\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("Ready for Feature Engineering (Harish)\n")
        f.write("=" * 70 + "\n")
    
    print("✓ Saved: reports/akshada_cleaning_summary.txt")
    

if __name__ == "__main__":
    main()