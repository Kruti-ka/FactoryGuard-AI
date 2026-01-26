"""
FactoryGuard AI - Data Ingestion Module
Member A - Week 1, Days 1-2

This module handles loading and initial processing of raw sensor data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_sensor_data(filepath):
    """
    Load raw sensor CSV files
    
    Args:
        filepath: Path to the sensor CSV file
        
    Returns:
        DataFrame with sensor readings
    """
    df = pd.read_csv(filepath)
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    print(f"Loaded {len(df)} sensor records")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Columns: {list(df.columns)}")
    
    return df


def merge_datasets(sensor_df, failure_log_df):
    """
    Merge sensor data with failure logs
    
    Args:
        sensor_df: DataFrame with sensor readings
        failure_log_df: DataFrame with failure events
        
    Returns:
        Merged DataFrame
    """
    # Merge on machine_id and timestamp
    merged = pd.merge_asof(
        sensor_df.sort_values('timestamp'),
        failure_log_df.sort_values('timestamp'),
        on='timestamp',
        by='machine_id',
        direction='forward'
    )
    
    print(f"Merged dataset size: {len(merged)} records")
    
    return merged


def get_data_summary(df):
    """
    Print summary statistics of the dataset
    
    Args:
        df: Input DataFrame
    """
    print("\n=== Data Summary ===")
    print(f"Total records: {len(df)}")
    print(f"Unique machines: {df['machine_id'].nunique()}")
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"\nSensor statistics:")
    print(df[['vibration', 'temperature', 'pressure']].describe())
    

if __name__ == "__main__":
    # Example usage
    print("Data Ingestion Module - FactoryGuard AI")
    print("Ready to process sensor data")
