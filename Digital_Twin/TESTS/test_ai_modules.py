# Digital_Twin/TESTS/test_ai_modules.py
"""
Test AI modules functionality including Health Score, Predictive Analytics,
Alert Manager, Pattern Analyzer, and Recommendation Engine.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Fixture 'sample_device_data' provided by conftest.py

@pytest.mark.ai_module
def test_health_score_calculator(sample_device_data):
    """Test health score calculator basic functionality."""
    try:
        from Digital_Twin.AI_MODULES.health_score import HealthScoreCalculator
        calculator = HealthScoreCalculator()
    except ImportError:
        pytest.skip("HealthScoreCalculator not available")
    except Exception as e:
         pytest.skip(f"Skipping HealthScoreCalculator test due to init error: {e}") # Skip if config/env missing

    # Test with a single device's data (using last 50 points as 'history')
    device_id = "TEST_HEALTH_001"
    test_data_history = sample_device_data.tail(50).copy()
    test_data_latest = test_data_history.tail(1).copy() # Use the very last point as 'new' data

    # Prime history (optional, but good practice for testing stateful components)
    calculator.calculate_overall_health_score(test_data_history.iloc[:-1], device_id=device_id)

    # Calculate score with the latest point
    result = calculator.calculate_overall_health_score(test_data_latest, device_id=device_id)

    assert isinstance(result, dict)
    assert 'error' not in result, f"Health score calculation returned error: {result.get('message')}"
    assert 'overall_score' in result
    assert 0 <= result['overall_score'] <= 1
    assert 'health_status' in result
    assert result['health_status'] in ['excellent', 'good', 'warning', 'critical', 'failure', 'unknown'] # Allow 'unknown' if error occurs
    assert 'component_scores' in result
    assert 'performance' in result['component_scores']
    assert 'reliability' in result['component_scores']

@pytest.mark.ai_module
def test_predictive_analytics_engine_anomaly(sample_device_data):
    """Test predictive analytics engine anomaly detection."""
    try:
        from Digital_Twin.AI_MODULES.predictive_analytics_engine import PredictiveAnalyticsEngine
        analytics = PredictiveAnalyticsEngine()
    except ImportError:
        pytest.skip("PredictiveAnalyticsEngine not available")
    except Exception as e:
         pytest.skip(f"Skipping PredictiveAnalyticsEngine test due to init error: {e}")

    # Prepare data (remove non-numeric timestamp for default models)
    features = sample_device_data.drop(columns=['timestamp']).copy()

    # Test anomaly detection training (using Isolation Forest by default)
    # Use try-except as training might fail with insufficient data variation
    try:
        train_result = analytics.train_anomaly_detector(features, model_name="test_iforest")
        assert isinstance(train_result, dict)
        assert 'error' not in train_result
        assert 'anomaly_ratio' in train_result
        assert analytics.models.get("test_iforest") is not None
    except Exception as train_exc:
        pytest.fail(f"Anomaly detector training failed: {train_exc}")


    # Test anomaly detection prediction
    try:
        predictions = analytics.detect_anomalies(features, model_name="test_iforest")
        assert isinstance(predictions, dict)
        assert 'anomaly_count' in predictions
        assert predictions['anomaly_count'] >= 0
        assert 'anomaly_percentage' in predictions
        assert 'anomalies' in predictions
        assert isinstance(predictions['anomalies'], list)
    except Exception as pred_exc:
         pytest.fail(f"Anomaly detection prediction failed: {pred_exc}")

@pytest.mark.ai_module
def test_predictive_analytics_engine_failure(sample_device_data):
    """Test predictive analytics engine failure prediction."""
    try:
        from Digital_Twin.AI_MODULES.predictive_analytics_engine import PredictiveAnalyticsEngine
        analytics = PredictiveAnalyticsEngine()
    except ImportError:
        pytest.skip("PredictiveAnalyticsEngine not available")
    except Exception as e:
         pytest.skip(f"Skipping PredictiveAnalyticsEngine test due to init error: {e}")

    # Prepare data and add a dummy failure target
    data_with_target = sample_device_data.drop(columns=['timestamp']).copy()
    data_with_target['failure'] = np.random.choice([0, 1], size=len(data_with_target), p=[0.95, 0.05])

    # Test failure prediction training
    try:
        train_result = analytics.train_failure_predictor(data_with_target, target_column='failure', model_name="test_failure_rf")
        assert isinstance(train_result, dict)
        assert 'error' not in train_result
        assert 'test_score' in train_result # Accuracy score
        assert analytics.models.get("test_failure_rf") is not None
    except Exception as train_exc:
        pytest.fail(f"Failure predictor training failed: {train_exc}")

    # Test failure prediction
    try:
        predictions = analytics.predict_failure(data_with_target.drop(columns=['failure']), model_name="test_failure_rf")
        assert isinstance(predictions, dict)
        assert 'predictions' in predictions
        assert 'failure_probabilities' in predictions
        assert len(predictions['predictions']) == len(data_with_target)
        assert len(predictions['failure_probabilities']) == len(data_with_target)
    except Exception as pred_exc:
         pytest.fail(f"Failure prediction failed: {pred_exc}")

@pytest.mark.ai_module
def test_alert_manager_evaluation():
    """Test alert manager rule evaluation."""
    try:
        from Digital_Twin.AI_MODULES.alert_manager import AlertManager
        alert_manager = AlertManager() # Assumes default config file exists
    except ImportError:
        pytest.skip("AlertManager not available")
    except Exception as e:
         pytest.skip(f"Skipping AlertManager test due to init error: {e}") # Skip if config missing/invalid

    # Test data points
    critical_temp_data = {'temperature': 98, 'pressure': 1010, 'vibration': 0.2}
    warning_temp_data = {'temperature': 85, 'pressure': 1010, 'vibration': 0.2}
    normal_data = {'temperature': 70, 'pressure': 1010, 'vibration': 0.2}
    critical_health_data = {'health_score': 0.25}

    # Test critical temperature alert rule
    critical_alerts = alert_manager.evaluate_conditions(critical_temp_data, device_id="TEST_ALERT_CRIT")
    assert isinstance(critical_alerts, list)
    assert any(a['rule_name'] == 'temperature_critical' and a['severity'] == 'critical' for a in critical_alerts)

    # Test warning temperature alert rule
    warning_alerts = alert_manager.evaluate_conditions(warning_temp_data, device_id="TEST_ALERT_WARN")
    assert isinstance(warning_alerts, list)
    assert any(a['rule_name'] == 'temperature_high' and a['severity'] == 'warning' for a in warning_alerts)
    assert not any(a['rule_name'] == 'temperature_critical' for a in warning_alerts)

    # Test normal conditions (should trigger no temperature alerts)
    normal_alerts = alert_manager.evaluate_conditions(normal_data, device_id="TEST_ALERT_NORM")
    assert not any('temperature' in a['rule_name'] for a in normal_alerts)

    # Test health score alert
    health_alerts = alert_manager.evaluate_conditions(critical_health_data, device_id="TEST_ALERT_HEALTH")
    assert isinstance(health_alerts, list)
    assert any(a['rule_name'] == 'health_score_critical' and a['severity'] == 'critical' for a in health_alerts)

    # Cleanup alert processing thread if needed (depends on AlertManager implementation)
    if hasattr(alert_manager, 'stop'):
         alert_manager.stop()

@pytest.mark.ai_module
def test_pattern_analyzer_temporal(sample_device_data):
    """Test pattern analyzer temporal analysis."""
    try:
        from Digital_Twin.AI_MODULES.pattern_analyzer import PatternAnalyzer
        analyzer = PatternAnalyzer()
    except ImportError:
        pytest.skip("PatternAnalyzer not available")
    except Exception as e:
         pytest.skip(f"Skipping PatternAnalyzer test due to init error: {e}")

    # Use only temperature for simplicity
    result = analyzer.analyze_temporal_patterns(sample_device_data, timestamp_col='timestamp', value_cols=['temperature'])

    assert isinstance(result, dict)
    assert 'patterns_found' in result
    assert 'temperature' in result['patterns_found']
    # Check for specific analysis results
    assert 'cyclical' in result['patterns_found']['temperature']
    assert 'seasonal' in result['patterns_found']['temperature']
    assert 'trend' in result['patterns_found']['temperature']
    assert 'periodicity' in result['patterns_found']['temperature']
    # Check if cycles were detected (based on sample data sine wave)
    assert result['patterns_found']['temperature']['cyclical'].get('cycles_detected', 0) > 0

@pytest.mark.ai_module
def test_recommendation_engine():
    """Test basic recommendation engine generation."""
    try:
        from Digital_Twin.AI_MODULES.recommendation_engine import RecommendationEngine
        engine = RecommendationEngine() # Assumes default config file exists
    except ImportError:
        pytest.skip("RecommendationEngine not available")
    except Exception as e:
         pytest.skip(f"Skipping RecommendationEngine test due to init error: {e}") # Skip if config missing/invalid

    # Provide sample health data indicating issues
    sample_health_data = {
        'overall_score': 0.55, # Warning level
        'health_status': 'warning',
        'component_scores': {
            'performance': {'score': 0.7},
            'reliability': {'score': 0.8},
            'efficiency': {'score': 0.45}, # Low efficiency
            'maintenance': {'score': 0.6}
        },
        'trend_analysis': {'trend_direction': 'degrading', 'trend_strength': 0.06},
        'risk_assessment': {'overall_risk_level': 'medium'}
    }

    recommendations = engine.generate_recommendations(health_data=sample_health_data)

    assert isinstance(recommendations, dict)
    assert 'error' not in recommendations, f"Recommendation generation failed: {recommendations.get('message')}"
    assert 'summary' in recommendations
    assert recommendations['summary']['total_recommendations'] > 0
    # Check if specific types of recommendations were generated based on input
    assert len(recommendations.get('maintenance_recommendations', [])) > 0 or \
           len(recommendations.get('optimization_recommendations', [])) > 0 or \
           len(recommendations.get('operational_recommendations', [])) > 0

# Add more detailed tests for each AI module, covering edge cases, different inputs, etc.
# For example:
# - Test HealthScoreCalculator with missing data.
# - Test PredictiveAnalyticsEngine with different model types (if applicable).
# - Test AlertManager with alert durations and acknowledgements.
# - Test PatternAnalyzer with different pattern types and data shapes.
# - Test RecommendationEngine with different health scores and contexts.