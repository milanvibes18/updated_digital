"""Test AI modules functionality"""
import pytest
import pandas as pd
import numpy as np

def test_health_score_calculator(sample_device_data):
    """Test health score calculator"""
    try:
        from AI_MODULES.health_score import HealthScoreCalculator
        
        calculator = HealthScoreCalculator()
        result = calculator.calculate_overall_health_score(sample_device_data, device_id="TEST_001")
        
        assert 'overall_score' in result
        assert 0 <= result['overall_score'] <= 1
        assert 'health_status' in result
        assert result['health_status'] in ['excellent', 'good', 'warning', 'critical', 'failure']
    except ImportError:
        pytest.skip("HealthScoreCalculator not available")

def test_predictive_analytics_engine(sample_device_data):
    """Test predictive analytics engine"""
    try:
        from AI_MODULES.predictive_analytics_engine import PredictiveAnalyticsEngine
        
        analytics = PredictiveAnalyticsEngine()
        # Test anomaly detection training
        anomaly_result = analytics.train_anomaly_detector(sample_device_data.drop(columns=['timestamp']))
        assert 'anomaly_ratio' in anomaly_result
        
        # Test anomaly detection prediction
        predictions = analytics.detect_anomalies(sample_device_data.drop(columns=['timestamp']))
        assert 'anomaly_count' in predictions
        
    except ImportError:
        pytest.skip("PredictiveAnalyticsEngine not available")

def test_alert_manager():
    """Test alert manager"""
    try:
        from AI_MODULES.alert_manager import AlertManager
        
        alert_manager = AlertManager()
        # Test a critical temperature alert
        test_data = {'temperature': 98, 'pressure': 1010, 'vibration': 0.2}
        
        alerts = alert_manager.evaluate_conditions(test_data, device_id="TEST_ALERT")
        
        assert isinstance(alerts, list)
        assert len(alerts) > 0
        assert alerts[0]['rule_name'] == 'temperature_critical'
        assert alerts[0]['severity'] == 'critical'
    except ImportError:
        pytest.skip("AlertManager not available")

def test_pattern_analyzer(sample_device_data):
    """Test pattern analyzer"""
    try:
        from AI_MODULES.pattern_analyzer import PatternAnalyzer
        
        analyzer = PatternAnalyzer()
        
        result = analyzer.analyze_temporal_patterns(sample_device_data, 'timestamp', ['temperature'])
        
        assert 'patterns_found' in result
        assert 'temperature' in result['patterns_found']
        assert 'cyclical' in result['patterns_found']['temperature']
    except ImportError:
        pytest.skip("PatternAnalyzer not available")