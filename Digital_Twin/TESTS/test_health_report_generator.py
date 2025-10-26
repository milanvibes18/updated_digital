#!/usr/bin/env python3
"""
Unit tests for the HealthReportGenerator module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
from datetime import datetime

# Add project root to path to find modules
# Assumes this test file is in Digital_Twin/TESTS/
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    from REPORTS.health_report_generator import HealthReportGenerator
except ImportError:
    pytest.skip("HealthReportGenerator not found, skipping tests. Check PYTHONPATH.", allow_module_level=True)

# --- Fixtures ---

@pytest.fixture(scope="module")
def temp_report_dir(tmp_path_factory):
    """Create a temporary directory for generated reports for the module."""
    return tmp_path_factory.mktemp("reports_generated")

@pytest.fixture(scope="module")
def generator(temp_report_dir):
    """
    Fixture for a HealthReportGenerator instance.
    This instance will use the generator's built-in sample data functionality
    by ensuring its external components (db_manager, health_calculator) are None,
    which triggers the sample data fallback in _collect_report_data.
    """
    gen = HealthReportGenerator(output_dir=str(temp_report_dir))
    
    # Mock external components to None to force using _generate_sample_data
    gen.health_calculator = None
    gen.db_manager = None
    
    return gen

# --- Basic Helper Function Tests ---

def test_generator_instantiation(generator, temp_report_dir):
    """Test if the generator initializes correctly."""
    assert generator is not None
    assert generator.output_dir == temp_report_dir
    assert generator.logger is not None

def test_calculate_trend(generator):
    """Test the _calculate_trend helper function."""
    stable_series = pd.Series([1, 1, 1.1, 1, 0.9, 1.05, 1])
    improving_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    declining_series = pd.Series([10, 9, 8.5, 8, 7, 6, 5, 4, 3])
    short_series = pd.Series([1, 2])
    
    assert generator._calculate_trend(stable_series) == 'stable'
    assert generator._calculate_trend(improving_series) == 'improving'
    assert generator._calculate_trend(declining_series) == 'declining'
    assert generator._calculate_trend(short_series) == 'stable' # Not enough data
    assert generator._calculate_trend(pd.Series([])) == 'stable' # Empty

def test_determine_health_status(generator):
    """Test the _determine_health_status helper."""
    assert generator._determine_health_status(95) == 'excellent'
    assert generator._determine_health_status(90) == 'excellent'
    assert generator._determine_health_status(85) == 'good'
    assert generator._determine_health_status(75) == 'good'
    assert generator._determine_health_status(70) == 'fair'
    assert generator._determine_health_status(60) == 'fair'
    assert generator._determine_health_status(59.9) == 'poor'
    assert generator._determine_health_status(0) == 'poor'

# --- Analysis Function Tests (using sample data) ---

@pytest.fixture(scope="module")
def sample_data(generator):
    """Generate sample data using the generator's own method."""
    return generator._generate_sample_data(date_range_days=7)

def test_sample_data_generation(sample_data):
    """Test if sample data generation works and creates the expected structure."""
    assert sample_data is not None
    assert 'devices' in sample_data
    assert 'system_metrics' in sample_data
    assert 'energy_data' in sample_data
    assert len(sample_data['devices']) > 0
    assert len(sample_data['system_metrics']) > 0
    assert len(sample_data['energy_data']) > 0
    
    # Check for the specific trends built into the sample data
    df = pd.DataFrame(sample_data['devices'])
    assert 'Vibration Sensor 01' in df['device_name'].values
    assert 'Temperature Sensor 01' in df['device_name'].values

@pytest.fixture(scope="module")
def health_analysis(generator, sample_data):
    """Provide health analysis results from the sample data."""
    return generator._analyze_health_scores(sample_data)

def test_analyze_health_scores(health_analysis):
    """Test the health score analysis logic using the sample data."""
    assert health_analysis is not None
    assert 'overall_health' in health_analysis
    assert 'current' in health_analysis['overall_health']
    assert 'trend' in health_analysis['overall_health']
    assert 'distribution' in health_analysis['overall_health']
    assert 'device_analysis' in health_analysis
    assert len(health_analysis['device_analysis']) == 5 # 5 sample devices
    
    # Check that the 'Vibration Sensor' (designed to decline) is in critical_devices
    assert 'critical_devices' in health_analysis
    assert any('Vibration Sensor 01' in d['device_name'] for d in health_analysis['critical_devices'])
    assert health_analysis['critical_devices'][0]['trend'] == 'declining'
    
    # Check that the 'Temperature Sensor' (designed to improve) is in top_performers
    assert 'top_performers' in health_analysis
    assert any('Temperature Sensor 01' in d['device_name'] for d in health_analysis['top_performers'])
    assert health_analysis['top_performers'][0]['trend'] == 'improving'

def test_generate_predictions(generator, sample_data, health_analysis):
    """Test the prediction generation based on sample data."""
    predictions = generator._generate_predictions(sample_data, health_analysis)
    
    assert predictions is not None
    assert 'health_forecast' in predictions
    assert 'maintenance_predictions' in predictions
    
    # Check if it predicted maintenance for the declining vibration sensor
    assert len(predictions['maintenance_predictions']) > 0
    assert any('Vibration Sensor 01' in p['device_name'] for p in predictions['maintenance_predictions'])

def test_generate_recommendations(generator, sample_data, health_analysis):
    """Test the recommendation generation based on sample data."""
    predictions = generator._generate_predictions(sample_data, health_analysis)
    recommendations = generator._generate_recommendations(sample_data, health_analysis, predictions)
    
    assert recommendations is not None
    assert 'immediate_actions' in recommendations
    assert 'short_term_actions' in recommendations
    assert 'long_term_actions' in recommendations
    
    # Check for immediate action for the critical vibration sensor
    assert len(recommendations['immediate_actions']) > 0
    assert any('Vibration Sensor 01' in a['action'] for a in recommendations['immediate_actions'])
    
    # Check for disk usage recommendation (from sample system metrics)
    assert any('disk usage' in a['action'] for a in recommendations['short_term_actions'])

# --- Main Report Generation Test (Integration) ---

def test_generate_comprehensive_report(generator, temp_report_dir):
    """
    Test the main report generation function. 
    This acts as an integration test for the generator,
    relying on its internal fallback to sample data.
    """
    
    # We rely on the generator's fallback to sample data
    report_path_str = generator.generate_comprehensive_report(
        date_range_days=7,
        include_predictions=True,
        include_recommendations=True
    )
    
    assert report_path_str is not None
    report_path = Path(report_path_str)
    
    # 1. Check if file was created in the correct directory
    assert report_path.parent == temp_report_dir, "Report was not created in the specified output directory"
    assert report_path.exists(), "Report HTML file was not created"
    assert report_path.name.startswith('health_report_')
    assert report_path.name.endswith('.html')
    
    # 2. Check if file has content
    assert report_path.stat().st_size > 1000, "Report HTML file is empty or too small"
    
    # 3. Check HTML content for key sections
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    assert "Comprehensive Health Report" in content
    assert "AI-Powered Recommendations" in content
    assert "Predictive Analysis" in content
    assert "Vibration Sensor 01" in content # From sample data
    assert "Immediate inspection and maintenance required" in content # From critical device recommendation
    assert "Device Health Distribution" in content
    assert "System Performance Metrics" in content

def test_generate_quick_summary(generator):
    """Test the quick summary function."""
    summary = generator.generate_quick_summary()
    
    assert summary is not None
    assert 'timestamp' in summary
    assert 'summary' in summary
    assert 'overall_health' in summary['summary']
    assert 'critical_devices_count' in summary
    assert summary['critical_devices_count'] > 0 # Based on sample data
    assert 'total_devices' in summary
    assert summary['total_devices'] == 5 # Based on sample data