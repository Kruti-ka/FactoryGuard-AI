"""
FactoryGuard AI - Feature Processor for Real-time Predictions
Handles feature engineering for single prediction requests
"""

import pandas as pd
import numpy as np
from datetime import datetime


class FeatureProcessor:
    """
    Process raw sensor readings into model-ready features
    Handles lag and rolling features with default values for missing data
    """
    
    def __init__(self, feature_names):
        """
        Initialize feature processor
        
        Args:
            feature_names: List of feature names expected by the model
        """
        self.feature_names = feature_names
        self.sensor_cols = ['vibration', 'temperature', 'pressure']
        
    def create_time_features(self, timestamp):
        """
        Extract time-based features from timestamp
        
        Args:
            timestamp: datetime or string timestamp
            
        Returns:
            dict: Time features
        """
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        return {
            'hour': timestamp.hour,
            'day': timestamp.day,
            'month': timestamp.month,
            'day_of_week': timestamp.dayofweek
        }
    
    def process_single_request(self, sensor_data):
        """
        Process a single prediction request
        
        Args:
            sensor_data: dict with keys:
                - timestamp: str or datetime
                - machine_id: str
                - temperature: float
                - vibration: float
                - pressure: float
                
        Returns:
            pd.DataFrame: Single row with all required features
        """
        # Start with base sensor values
        features = {
            'temperature': sensor_data['temperature'],
            'vibration': sensor_data['vibration'],
            'pressure': sensor_data['pressure']
        }
        
        # Add time features
        time_features = self.create_time_features(sensor_data['timestamp'])
        features.update(time_features)
        
        # Add lag features (use current values as approximation)
        # In production, these would come from historical data
        for col in self.sensor_cols:
            for lag in [1, 2, 3]:
                # Use current value as default (limitation documented)
                features[f'{col}_lag_{lag}'] = sensor_data[col]
        
        # Add rolling mean features (use current values as approximation)
        for col in self.sensor_cols:
            for window in [3, 6, 12]:
                # Use current value as default
                features[f'{col}_roll_mean_{window}'] = sensor_data[col]
        
        # Create DataFrame with all features
        df = pd.DataFrame([features])
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0.0  # Default value for missing features
        
        # Return only the features in the correct order
        return df[self.feature_names]
    
    def validate_input(self, sensor_data):
        """
        Validate input sensor data
        
        Args:
            sensor_data: dict with sensor readings
            
        Returns:
            tuple: (is_valid, error_message)
        """
        required_fields = ['timestamp', 'machine_id', 'temperature', 'vibration', 'pressure']
        
        # Check required fields
        for field in required_fields:
            if field not in sensor_data:
                return False, f"Missing required field: {field}"
        
        # Validate numeric ranges
        try:
            temp = float(sensor_data['temperature'])
            vib = float(sensor_data['vibration'])
            press = float(sensor_data['pressure'])
            
            # Reasonable ranges for industrial sensors
            if not (0 <= temp <= 200):
                return False, f"Temperature out of range (0-200): {temp}"
            if not (0 <= vib <= 10):
                return False, f"Vibration out of range (0-10): {vib}"
            if not (0 <= press <= 200):
                return False, f"Pressure out of range (0-200): {press}"
                
        except (ValueError, TypeError) as e:
            return False, f"Invalid numeric value: {str(e)}"
        
        # Validate timestamp
        try:
            pd.to_datetime(sensor_data['timestamp'])
        except Exception as e:
            return False, f"Invalid timestamp format: {str(e)}"
        
        return True, None


class SHAPExplainer:
    """
    Generate human-readable explanations from SHAP values
    """
    
    @staticmethod
    def format_explanation(shap_values, feature_values, feature_names, top_n=5):
        """
        Format SHAP values into human-readable explanation
        
        Args:
            shap_values: numpy array of SHAP values
            feature_values: numpy array of feature values
            feature_names: list of feature names
            top_n: number of top features to include
            
        Returns:
            dict: Formatted explanation with top features
        """
        # Get absolute SHAP values for ranking
        abs_shap = np.abs(shap_values)
        
        # Get top N features by absolute SHAP value
        top_indices = np.argsort(abs_shap)[-top_n:][::-1]
        
        top_features = []
        for idx in top_indices:
            top_features.append({
                "feature": feature_names[idx],
                "shap_value": float(shap_values[idx]),
                "feature_value": float(feature_values[idx]),
                "contribution": "increases" if shap_values[idx] > 0 else "decreases"
            })
        
        return top_features
    
    @staticmethod
    def generate_text_explanation(failure_probability, top_features):
        """
        Generate human-readable text explanation
        
        Args:
            failure_probability: float (0-1)
            top_features: list of dicts with feature info
            
        Returns:
            str: Human-readable explanation
        """
        risk_level = "High" if failure_probability > 0.7 else "Moderate" if failure_probability > 0.4 else "Low"
        
        # Get main contributing factors
        positive_contributors = [f for f in top_features if f['contribution'] == 'increases']
        
        if not positive_contributors:
            return f"{risk_level} failure risk ({failure_probability:.0%}). All monitored parameters within normal ranges."
        
        # Build explanation
        main_factor = positive_contributors[0]
        feature_name = main_factor['feature'].replace('_', ' ')
        
        explanation = f"{risk_level} failure risk ({failure_probability:.0%}) primarily due to "
        
        # Simplify feature names for readability
        if 'temperature' in feature_name:
            explanation += f"elevated temperature ({main_factor['feature_value']:.1f}°C)"
        elif 'vibration' in feature_name:
            explanation += f"high vibration levels ({main_factor['feature_value']:.2f})"
        elif 'pressure' in feature_name:
            explanation += f"abnormal pressure ({main_factor['feature_value']:.1f})"
        else:
            explanation += f"{feature_name} ({main_factor['feature_value']:.2f})"
        
        # Add secondary factors if present
        if len(positive_contributors) > 1:
            explanation += " and sustained sensor patterns"
        
        explanation += "."
        
        return explanation
