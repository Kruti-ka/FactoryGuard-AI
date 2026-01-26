"""
FactoryGuard AI - Feature Engineering Module
Member B - Week 1, Days 5-7

This module creates temporal features including rolling statistics, 
lag features, and exponential moving averages.
"""

import pandas as pd
import numpy as np


def create_lag_features(df, columns, lags=[1, 2, 3]):
    """
    Create lag features (t-1, t-2, t-3)
    
    Args:
        df: Input DataFrame
        columns: List of columns to create lags for
        lags: List of lag values
        
    Returns:
        DataFrame with lag features
    """
    df = df.sort_values(['machine_id', 'timestamp'])
    
    for col in columns:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df.groupby('machine_id')[col].shift(lag)
    
    print(f"Created {len(columns) * len(lags)} lag features")
    
    return df


def create_rolling_statistics(df, columns, windows=[1, 4, 8]):
    """
    Create rolling window statistics
    
    Args:
        df: Input DataFrame
        columns: List of columns to create rolling stats for
        windows: List of window sizes in hours
        
    Returns:
        DataFrame with rolling statistics
    """
    df = df.sort_values(['machine_id', 'timestamp'])
    
    feature_count = 0
    
    for col in columns:
        for window in windows:
            # Rolling mean
            df[f'{col}_rolling_mean_{window}h'] = df.groupby('machine_id')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # Rolling std (stability indicator)
            df[f'{col}_rolling_std_{window}h'] = df.groupby('machine_id')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            
            # Rolling max
            df[f'{col}_rolling_max_{window}h'] = df.groupby('machine_id')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
            
            # Rolling min
            df[f'{col}_rolling_min_{window}h'] = df.groupby('machine_id')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
            
            feature_count += 4
    
    print(f"Created {feature_count} rolling window features")
    
    return df


def create_ema_features(df, columns, spans=[2, 4, 8]):
    """
    Create Exponential Moving Average features
    
    Args:
        df: Input DataFrame
        columns: List of columns to create EMA for
        spans: List of span values in hours
        
    Returns:
        DataFrame with EMA features
    """
    df = df.sort_values(['machine_id', 'timestamp'])
    
    for col in columns:
        for span in spans:
            df[f'{col}_ema_{span}h'] = df.groupby('machine_id')[col].transform(
                lambda x: x.ewm(span=span, adjust=False).mean()
            )
    
    print(f"Created {len(columns) * len(spans)} EMA features")
    
    return df


def create_interaction_features(df):
    """
    Create domain-specific interaction features
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with interaction features
    """
    # Temperature-Pressure interaction (physical constraint)
    df['temp_pressure_interaction'] = df['temperature'] * df['pressure']
    
    # Vibration instability (vibration * std)
    if 'vibration_rolling_std_4h' in df.columns:
        df['vibration_instability'] = df['vibration'] * df['vibration_rolling_std_4h']
    
    # Temperature change rate
    if 'temperature_lag_1' in df.columns:
        df['temp_change_rate'] = df['temperature'] - df['temperature_lag_1']
    
    print("Created interaction features")
    
    return df


def create_all_features(df):
    """
    Create all temporal and interaction features
    
    Args:
        df: Input DataFrame with clean sensor data
        
    Returns:
        DataFrame with all engineered features
    """
    sensor_cols = ['vibration', 'temperature', 'pressure']
    
    print("\n=== Feature Engineering Pipeline ===")
    
    # Lag features
    df = create_lag_features(df, sensor_cols, lags=[1, 2, 3])
    
    # Rolling statistics
    df = create_rolling_statistics(df, sensor_cols, windows=[1, 4, 8])
    
    # Exponential Moving Averages
    df = create_ema_features(df, sensor_cols, spans=[2, 4, 8])
    
    # Interaction features
    df = create_interaction_features(df)
    
    # Drop rows with NaN values created by lag/rolling features
    initial_len = len(df)
    df = df.dropna()
    print(f"\nDropped {initial_len - len(df)} rows with NaN from feature creation")
    
    print(f"Final feature count: {len(df.columns)}")
    
    return df


if __name__ == "__main__":
    print("Feature Engineering Module - FactoryGuard AI")
    print("Ready to create temporal features")
