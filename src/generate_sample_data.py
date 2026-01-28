"""
Sample Sensor Data Generator for Testing
This script generates synthetic sensor data for development and testing purposes.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


def generate_sample_sensor_data(
    n_machines=10,
    n_days=30,
    hourly_samples=1,
    failure_rate=0.005,
    output_path='data/raw/sensor_logs.csv'

):
    """
    Generate synthetic sensor data for testing
    
    Args:
        n_machines: Number of machines to simulate
        n_days: Number of days of data
        hourly_samples: Samples per hour
        failure_rate: Probability of failure (default 0.5%)
        output_path: Path to save the generated data
    """
    
    print(f"Generating sensor data for {n_machines} machines over {n_days} days...")
    
    # Time range
    start_time = datetime(2024, 1, 1)
    timestamps = []
    current_time = start_time
    
    for _ in range(n_days * 24 * hourly_samples):
        timestamps.append(current_time)
        current_time += timedelta(hours=1/hourly_samples)
    
    data = []
    
    for machine_id in range(1, n_machines + 1):
        # Base parameters for this machine (each machine has slightly different baseline)
        base_vibration = np.random.uniform(0.3, 0.5)
        base_temp = np.random.uniform(60, 70)
        base_pressure = np.random.uniform(100, 105)
        
        # Generate failure events for this machine
        failure_times = []
        for ts in timestamps:
            if np.random.random() < failure_rate:
                failure_times.append(ts)
        
        for ts in timestamps:
            # Normal operation with some noise
            vibration = base_vibration + np.random.normal(0, 0.05)
            temperature = base_temp + np.random.normal(0, 2)
            pressure = base_pressure + np.random.normal(0, 1)
            
            # If approaching failure (within 24-48 hours), increase degradation
            time_to_failure = None
            is_failure = 0
            
            for failure_time in failure_times:
                hours_to_failure = (failure_time - ts).total_seconds() / 3600
                
                if hours_to_failure == 0:
                    is_failure = 1
                
                if 0 < hours_to_failure <= 48:
                    # Gradual degradation before failure
                    degradation_factor = 1 - (hours_to_failure / 48)
                    
                    vibration += degradation_factor * 0.3  # Increase vibration
                    temperature += degradation_factor * 15  # Increase temperature
                    pressure -= degradation_factor * 5  # Decrease pressure
            
            # Add some correlations (realistic physics)
            temperature += vibration * 10  # High vibration causes heat
            
            data.append({
                'timestamp': ts,
                'machine_id': machine_id,
                'vibration': round(vibration, 3),
                'temperature': round(temperature, 2),
                'pressure': round(pressure, 2),
                'failure': is_failure
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # **FIX: Create output directory if it doesn't exist**
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create if there's a directory component
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n✓ Created directory: {output_dir}")
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Generated {len(df)} sensor records")
    print(f"✓ Machines: {n_machines}")
    print(f"✓ Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"✓ Total failures: {df['failure'].sum()}")
    print(f"✓ Failure rate: {(df['failure'].sum() / len(df)) * 100:.3f}%")
    print(f"✓ Saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    # Generate sample data
    df = generate_sample_sensor_data(
        n_machines=10,
        n_days=30,
        hourly_samples=1,
        failure_rate=0.005
    )
    
    print("\n=== Sample Records ===")
    print(df.head(10))