"""
FactoryGuard AI - Data Cleaning Module
Member A - Week 1, Days 3-4

This module handles missing values, outliers, and target variable creation.
"""

import pandas as pd
import numpy as np


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
    
    print(f"Missing values before: {df[numeric_cols].isnull().sum().sum()}")
    
    if method == 'interpolate':
        # Time-based interpolation
        df[numeric_cols] = df.groupby('machine_id')[numeric_cols].transform(
            lambda x: x.interpolate(method='time', limit=12)  # Max 12 hours gap
        )
    elif method == 'forward_fill':
        df[numeric_cols] = df.groupby('machine_id')[numeric_cols].ffill(limit=6)
    elif method == 'mean':
        for col in numeric_cols:
            df[col].fillna(df.groupby('machine_id')[col].transform('mean'), inplace=True)
    
    # Drop rows with remaining NaNs (gaps > 12 hours)
    df = df.dropna(subset=numeric_cols)
    
    print(f"Missing values after: {df[numeric_cols].isnull().sum().sum()}")
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
    original_len = len(df)
    
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        df = df[(df[col] >= mean - n_std*std) & (df[col] <= mean + n_std*std)]
    
    print(f"Removed {original_len - len(df)} outlier records ({((original_len - len(df))/original_len)*100:.2f}%)")
    
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
    df = df.sort_values(['machine_id', 'timestamp'])
    
    # Initialize target variable
    df['failure_within_24h'] = 0
    
    for machine_id in df['machine_id'].unique():
        machine_data = df[df['machine_id'] == machine_id].copy()
        
        # Get failure times for this machine
        if 'failure' in df.columns:
            failure_times = machine_data[machine_data['failure'] == 1]['timestamp'].values
            
            for failure_time in failure_times:
                # Mark all records within 24h before failure
                mask = (
                    (df['machine_id'] == machine_id) & 
                    (df['timestamp'] < failure_time) & 
                    (df['timestamp'] >= failure_time - pd.Timedelta(hours=failure_window_hours))
                )
                df.loc[mask, 'failure_within_24h'] = 1
    
    # Print class distribution
    print("\n=== Target Variable Distribution ===")
    print(df['failure_within_24h'].value_counts())
    print(f"Positive class: {(df['failure_within_24h'].sum()/len(df))*100:.2f}%")
    
    return df


if __name__ == "__main__":
    print("Data Cleaning Module - FactoryGuard AI")
    print("Ready to clean and prepare data")
