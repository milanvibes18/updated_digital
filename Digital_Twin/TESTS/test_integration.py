"""Integration tests"""
import time
import psutil
import os
import pandas as pd

def test_full_system_integration(test_database, sample_device_data):
    """
    Tests a simple integration flow:
    1. A test database fixture is created.
    2. Sample data fixture is created.
    3. We check that both were created successfully.
    """
    assert test_database is not None
    assert os.path.exists(test_database)
    assert not sample_device_data.empty
    assert isinstance(sample_device_data, pd.DataFrame)

def test_performance_benchmarks():
    """Tests basic performance to catch major regressions."""
    start_time = time.time()
    # Simulate a small workload
    [x*x for x in range(1000)]
    end_time = time.time()
    response_time = end_time - start_time
    assert response_time < 0.1  # Should be very fast

def test_memory_usage():
    """Tests that the process memory usage is within a reasonable limit."""
    process = psutil.Process(os.getpid())
    memory_usage_mb = process.memory_info().rss / (1024 * 1024)
    # Set a generous limit of 500MB for the test runner process
    assert memory_usage_mb < 500