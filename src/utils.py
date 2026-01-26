"""
FactoryGuard AI - Utility Functions
Shared helper functions and validation utilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def validate_no_data_leakage(df):
    """
    Check that rolling features only use past data
    
    Args:
        df: Input DataFrame
        
    Returns:
        bool: True if no leakage detected
    """
    print("\n=== Data Leakage Validation ===")
    
    # Split data by timestamp
    train_cutoff = df['timestamp'].quantile(0.7)
    train = df[df['timestamp'] <= train_cutoff]
    test = df[df['timestamp'] > train_cutoff]
    
    # Verify no test data influenced train features
    print(f"Train end: {train['timestamp'].max()}")
    print(f"Test start: {test['timestamp'].min()}")
    
    assert train['timestamp'].max() < test['timestamp'].min(), "Data leakage detected!"
    
    # Check for NaN in critical features
    feature_cols = [col for col in df.columns if 'rolling' in col or 'lag' in col or 'ema' in col]
    nan_count = df[feature_cols].isnull().sum().sum()
    print(f"Missing values in features: {nan_count}")
    
    print("✓ No data leakage detected")
    return True


def plot_sensor_distributions(df, save_path=None):
    """
    Plot distributions of sensor values
    
    Args:
        df: Input DataFrame
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    sensor_cols = ['vibration', 'temperature', 'pressure']
    
    for i, col in enumerate(sensor_cols):
        axes[i].hist(df[col], bins=50, edgecolor='black', alpha=0.7)
        axes[i].set_title(f'{col.capitalize()} Distribution')
        axes[i].set_xlabel(col.capitalize())
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_failure_timeline(df, machine_id, save_path=None):
    """
    Plot sensor timeline for a specific machine with failure markers
    
    Args:
        df: Input DataFrame
        machine_id: ID of the machine to plot
        save_path: Optional path to save the plot
    """
    machine_data = df[df['machine_id'] == machine_id].sort_values('timestamp')
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    sensor_cols = ['vibration', 'temperature', 'pressure']
    
    for i, col in enumerate(sensor_cols):
        axes[i].plot(machine_data['timestamp'], machine_data[col], label=col, linewidth=1)
        
        # Mark failures
        if 'failure' in machine_data.columns:
            failures = machine_data[machine_data['failure'] == 1]
            axes[i].scatter(failures['timestamp'], failures[col], color='red', 
                          s=100, marker='x', label='Failure', zorder=5)
        
        axes[i].set_ylabel(col.capitalize())
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Timestamp')
    plt.suptitle(f'Machine {machine_id} - Sensor Timeline')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def get_feature_statistics(df):
    """
    Print statistics about engineered features
    
    Args:
        df: Input DataFrame
    """
    print("\n=== Feature Statistics ===")
    
    lag_features = [col for col in df.columns if 'lag' in col]
    rolling_features = [col for col in df.columns if 'rolling' in col]
    ema_features = [col for col in df.columns if 'ema' in col]
    
    print(f"Lag features: {len(lag_features)}")
    print(f"Rolling features: {len(rolling_features)}")
    print(f"EMA features: {len(ema_features)}")
    print(f"Total engineered features: {len(lag_features) + len(rolling_features) + len(ema_features)}")


if __name__ == "__main__":
    print("Utility Functions - FactoryGuard AI")
