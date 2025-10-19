#!/usr/bin/env python3
"""
Shared pytest fixtures for Digital Twin System tests
"""

import pytest
import sys
import os
from pathlib import Path
import tempfile
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture
def test_database():
    """Create a temporary test database"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    # Create test database with sample data
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Create device_data table
        cursor.execute('''
            CREATE TABLE device_data (
                id INTEGER PRIMARY KEY,
                device_id TEXT,
                device_name TEXT,
                device_type TEXT,
                timestamp DATETIME,
                value REAL,
                health_score REAL,
                efficiency_score REAL,
                status TEXT,
                location TEXT
            )
        ''')
        
        # Insert test data
        test_data = []
        devices = ['DEVICE_001', 'DEVICE_002', 'DEVICE_003']
        for i, device_id in enumerate(devices):
            for j in range(10):
                test_data.append((
                    device_id,
                    f'Test Device {i+1}',
                    'temperature_sensor',
                    (datetime.now() - timedelta(hours=j)).isoformat(),
                    20 + i * 10 + np.random.normal(0, 2),
                    0.8 + np.random.normal(0, 0.1),
                    0.85 + np.random.normal(0, 0.05),
                    'normal',
                    f'Location {i+1}'
                ))
        
        cursor.executemany('''
            INSERT INTO device_data (device_id, device_name, device_type, timestamp, value, health_score, efficiency_score, status, location)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', test_data)
        
        conn.commit()
    
    yield db_path
    
    # Cleanup
    os.unlink(db_path)

@pytest.fixture
def sample_device_data():
    """Generate sample device data for testing"""
    np.random.seed(42)
    timestamps = pd.date_range('2024-01-01', periods=100, freq='H')
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'temperature': 20 + 5 * np.sin(np.arange(100) * 0.1) + np.random.normal(0, 1, 100),
        'pressure': 1013 + np.random.normal(0, 5, 100),
        'vibration': np.random.exponential(0.1, 100),
        'efficiency': np.random.beta(3, 2, 100) * 100,
        'health_score': np.random.uniform(0.6, 1.0, 100)
    })

@pytest.fixture
def flask_test_client():
    """Create a test client for Flask app"""
    try:
        from WEB_APPLICATION.enhanced_flask_app_v2 import create_app
        app_instance = create_app()
        app_instance.app.config['TESTING'] = True
        with app_instance.app.test_client() as client:
            yield client
    except ImportError:
        pytest.skip("Flask app not available")
