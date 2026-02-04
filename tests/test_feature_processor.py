"""
Unit tests for Feature Processor
"""

import pytest
import pandas as pd
from datetime import datetime
from utils.feature_processor import FeatureProcessor, SHAPExplainer


class TestFeatureProcessor:
    """Test cases for FeatureProcessor class"""
    
    @pytest.fixture
    def feature_names(self):
        """Sample feature names for testing"""
        return [
            'temperature', 'vibration', 'pressure',
            'hour', 'day', 'month', 'day_of_week',
            'temperature_lag_1', 'temperature_lag_2', 'temperature_lag_3',
            'vibration_lag_1', 'vibration_lag_2', 'vibration_lag_3',
            'pressure_lag_1', 'pressure_lag_2', 'pressure_lag_3',
            'temperature_roll_mean_3', 'temperature_roll_mean_6', 'temperature_roll_mean_12',
            'vibration_roll_mean_3', 'vibration_roll_mean_6', 'vibration_roll_mean_12',
            'pressure_roll_mean_3', 'pressure_roll_mean_6', 'pressure_roll_mean_12'
        ]
    
    @pytest.fixture
    def processor(self, feature_names):
        """Create FeatureProcessor instance"""
        return FeatureProcessor(feature_names)
    
    @pytest.fixture
    def valid_sensor_data(self):
        """Valid sensor data for testing"""
        return {
            'timestamp': '2024-01-15 10:30:00',
            'machine_id': 'M001',
            'temperature': 75.5,
            'vibration': 0.45,
            'pressure': 100.2
        }
    
    def test_validate_input_valid(self, processor, valid_sensor_data):
        """Test validation with valid input"""
        is_valid, error = processor.validate_input(valid_sensor_data)
        assert is_valid == True
        assert error is None
    
    def test_validate_input_missing_field(self, processor):
        """Test validation with missing field"""
        data = {
            'timestamp': '2024-01-15 10:30:00',
            'machine_id': 'M001',
            'temperature': 75.5
            # Missing vibration and pressure
        }
        is_valid, error = processor.validate_input(data)
        assert is_valid == False
        assert 'Missing required field' in error
    
    def test_validate_input_invalid_temperature(self, processor, valid_sensor_data):
        """Test validation with out-of-range temperature"""
        data = valid_sensor_data.copy()
        data['temperature'] = 250.0  # Out of range
        is_valid, error = processor.validate_input(data)
        assert is_valid == False
        assert 'Temperature out of range' in error
    
    def test_validate_input_invalid_vibration(self, processor, valid_sensor_data):
        """Test validation with out-of-range vibration"""
        data = valid_sensor_data.copy()
        data['vibration'] = 15.0  # Out of range
        is_valid, error = processor.validate_input(data)
        assert is_valid == False
        assert 'Vibration out of range' in error
    
    def test_create_time_features(self, processor):
        """Test time feature extraction"""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        features = processor.create_time_features(timestamp)
        
        assert features['hour'] == 10
        assert features['day'] == 15
        assert features['month'] == 1
        assert features['day_of_week'] == 0  # Monday
    
    def test_process_single_request(self, processor, valid_sensor_data, feature_names):
        """Test processing single request"""
        result = processor.process_single_request(valid_sensor_data)
        
        # Check result is DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Check shape
        assert result.shape[0] == 1  # Single row
        assert result.shape[1] == len(feature_names)  # All features
        
        # Check feature names match
        assert list(result.columns) == feature_names
        
        # Check base sensor values
        assert result['temperature'].values[0] == 75.5
        assert result['vibration'].values[0] == 0.45
        assert result['pressure'].values[0] == 100.2


class TestSHAPExplainer:
    """Test cases for SHAPExplainer class"""
    
    def test_format_explanation(self):
        """Test SHAP explanation formatting"""
        import numpy as np
        
        shap_values = np.array([0.5, -0.3, 0.2, 0.1, -0.05])
        feature_values = np.array([75.5, 0.45, 100.2, 10, 15])
        feature_names = ['temperature', 'vibration', 'pressure', 'hour', 'day']
        
        top_features = SHAPExplainer.format_explanation(
            shap_values, feature_values, feature_names, top_n=3
        )
        
        # Check we got top 3 features
        assert len(top_features) == 3
        
        # Check first feature is temperature (highest absolute SHAP)
        assert top_features[0]['feature'] == 'temperature'
        assert top_features[0]['shap_value'] == 0.5
        assert top_features[0]['contribution'] == 'increases'
    
    def test_generate_text_explanation_high_risk(self):
        """Test text explanation for high risk"""
        top_features = [
            {'feature': 'temperature', 'shap_value': 0.5, 'feature_value': 85.0, 'contribution': 'increases'},
            {'feature': 'vibration', 'shap_value': 0.3, 'feature_value': 0.8, 'contribution': 'increases'}
        ]
        
        explanation = SHAPExplainer.generate_text_explanation(0.85, top_features)
        
        assert 'High failure risk' in explanation
        assert '85%' in explanation
        assert 'temperature' in explanation.lower()
    
    def test_generate_text_explanation_low_risk(self):
        """Test text explanation for low risk"""
        top_features = [
            {'feature': 'temperature', 'shap_value': -0.2, 'feature_value': 60.0, 'contribution': 'decreases'}
        ]
        
        explanation = SHAPExplainer.generate_text_explanation(0.15, top_features)
        
        assert 'Low failure risk' in explanation
        assert '15%' in explanation


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
