"""
FactoryGuard AI - Data Cleaning Module
Akshada - Week 1, Days 3-4

This module handles missing values, outliers, and target variable creation.
"""

import pandas as pd
import numpy as np
import os


def handle_missing_values(df, method='interpolate'):
    """
    Handle missing sensor values
    
    Args:
        df: Input DataFrame
        method: 'interpolate', 'forward_fill', or 'mean'
        
    Returns:
        DataFrame with imputed values
    """
    numeric_cols = ['vibration', 'temperature', 'pressure']
    
    print("\n=== Handling Missing Values ===")
    print(f"Missing values before: {df[numeric_cols].isnull().sum().sum()}")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    if method == 'interpolate':
        # Time-based interpolation per machine
        for machine_id in df['machine_id'].unique():
            mask = df['machine_id'] == machine_id
            machine_data = df.loc[mask, numeric_cols].copy()
            
            # Interpolate with limit of 12 consecutive hours
            interpolated = machine_data.interpolate(method='linear', limit=12, limit_direction='both')
            df.loc[mask, numeric_cols] = interpolated
            
    elif method == 'forward_fill':
        # Forward fill within each machine group
        df[numeric_cols] = df.groupby('machine_id')[numeric_cols].ffill(limit=6)
        
    elif method == 'mean':
        # Fill with machine-specific mean
        for col in numeric_cols:
            df[col].fillna(df.groupby('machine_id')[col].transform('mean'), inplace=True)
    
    # Drop rows with remaining NaNs (gaps > 12 hours)
    rows_before = len(df)
    df = df.dropna(subset=numeric_cols)
    rows_dropped = rows_before - len(df)
    
    print(f"Missing values after: {df[numeric_cols].isnull().sum().sum()}")
    print(f"Rows dropped: {rows_dropped}")
    print(f"Records remaining: {len(df)}")
    
    return df


def remove_outliers(df, columns, n_std=4):
    """
    Remove outliers beyond n standard deviations
    
    Args:
        df: Input DataFrame
        columns: List of columns to check for outliers
        n_std: Number of standard deviations threshold
        
    Returns:
        DataFrame with outliers removed
    """
    print("\n=== Removing Outliers ===")
    original_len = len(df)
    
    df = df.copy()
    
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        lower_bound = mean - n_std * std
        upper_bound = mean + n_std * std
        
        before = len(df)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        removed = before - len(df)
        
        print(f"{col}: removed {removed} outliers (range: {lower_bound:.2f} to {upper_bound:.2f})")
    
    total_removed = original_len - len(df)
    print(f"\nTotal removed: {total_removed} outlier records ({(total_removed/original_len)*100:.2f}%)")
    print(f"Records remaining: {len(df)}")
    
    return df


def create_target_variable(df, failure_window_hours=24):
    """
    Create binary target: 1 if failure occurs within next 24 hours
    
    Args:
        df: Input DataFrame
        failure_window_hours: Hours ahead to predict failure
        
    Returns:
        DataFrame with target variable 'failure_within_24h'
    """
    print("\n=== Creating Target Variable ===")
    print(f"Predicting failures within next {failure_window_hours} hours")
    
    df = df.copy()
    df = df.sort_values(['machine_id', 'timestamp'])
    
    # Initialize target variable
    df['failure_within_24h'] = 0
    
    for machine_id in df['machine_id'].unique():
        # Get all data for this machine
        machine_mask = df['machine_id'] == machine_id
        machine_data = df[machine_mask].copy()
        
        # Get failure times for this machine
        if 'failure' in df.columns:
            failure_times = machine_data[machine_data['failure'] == 1]['timestamp'].values
            
            if len(failure_times) > 0:
                print(f"Machine {machine_id}: {len(failure_times)} failure events")
            
            for failure_time in failure_times:
                # Mark all records within 24h before failure as positive
                time_window = pd.Timedelta(hours=failure_window_hours)
                
                mask = (
                    (df['machine_id'] == machine_id) & 
                    (df['timestamp'] < failure_time) & 
                    (df['timestamp'] >= failure_time - time_window)
                )
                df.loc[mask, 'failure_within_24h'] = 1
    
    # Print class distribution
    print("\n=== Target Variable Distribution ===")
    print(df['failure_within_24h'].value_counts())
    print(f"Positive class (failures): {df['failure_within_24h'].sum()}")
    print(f"Positive class percentage: {(df['failure_within_24h'].sum()/len(df))*100:.2f}%")
    
    # Check for class imbalance
    imbalance_ratio = (df['failure_within_24h'] == 0).sum() / (df['failure_within_24h'] == 1).sum()
    print(f"Imbalance ratio: {imbalance_ratio:.1f}:1 (negative:positive)")
    
    return df


def clean_pipeline(input_path, output_path):
    """
    Full data cleaning pipeline
    
    Args:
        input_path: Path to raw sensor data
        output_path: Path to save cleaned data
    """
    print("=" * 60)
    print("FactoryGuard AI - Data Cleaning Pipeline")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from: {input_path}")
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    
    # Step 1: Handle missing values
    df = handle_missing_values(df, method='interpolate')
    
    # Step 2: Remove outliers
    df = remove_outliers(df, columns=['vibration', 'temperature', 'pressure'], n_std=4)
    
    # Step 3: Create target variable
    df = create_target_variable(df, failure_window_hours=24)
    
    # Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print(f"✓ Data cleaning complete!")
    print(f"✓ Cleaned data saved to: {output_path}")
    print(f"✓ Final dataset size: {len(df)} records")
    print(f"✓ Ready for feature engineering!")
    print("=" * 60)
    
    return df


if __name__ == "__main__":
    # Run the cleaning pipeline
    clean_pipeline(
        input_path='data/raw/sensor_logs.csv',
        output_path='data/processed/clean_data.csv'
    )