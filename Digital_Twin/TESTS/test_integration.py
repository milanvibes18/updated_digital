# Digital_Twin/TESTS/test_integration.py
"""
Integration tests for the Digital Twin System.
Covers data flow, component interactions, and basic performance.
"""
import pytest
import time
import psutil
import os
import pandas as pd
from flask import Flask
from flask.testing import FlaskClient
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

# Assuming conftest.py provides flask_test_client and test_database fixtures
# Also assuming models are accessible, adjust imports if needed
try:
    from Digital_Twin.WEB_APPLICATION.enhanced_flask_app_v2 import DeviceData, Alert, get_session
except ImportError:
    # Define dummy classes if the app isn't directly importable in the test environment
    class DeviceData: pass
    class Alert: pass
    def get_session(): yield None; pytest.skip("Flask app models not available")

@pytest.mark.integration
def test_initial_setup_fixtures(flask_test_client: FlaskClient, test_database: str):
    """
    Tests that the basic test fixtures (app client, temporary DB) load correctly.
    """
    assert flask_test_client is not None
    assert isinstance(flask_test_client.application, Flask)
    assert test_database is not None
    assert os.path.exists(test_database)
    # Check if the /health endpoint works on the test client
    response = flask_test_client.get('/health')
    assert response.status_code == 200

@pytest.mark.integration
def test_data_ingestion_retrieval_flow(flask_test_client: FlaskClient, auth_token: str):
    """
    Tests the flow of inserting data (simulated) and retrieving it via API.
    Note: This assumes a way to inject data or relies on a running data generator/ingestor
          connected to the test database, which is complex.
          A simpler integration test checks API interactions based on initial fixture data.
    """
    if flask_test_client is None: pytest.skip("Flask app not available")

    headers = {'Authorization': f'Bearer {auth_token}'}

    # 1. Check initial device count (might be 0 or from fixtures)
    initial_response = flask_test_client.get('/api/devices', headers=headers)
    assert initial_response.status_code == 200
    initial_devices = initial_response.get_json()
    initial_count = len(initial_devices)
    print(f"Initial device count: {initial_count}")

    # 2. Simulate Data Ingestion (Difficult without direct DB access or specific endpoint)
    #    For a true integration test, you might:
    #    a) Have a test-specific API endpoint to insert data.
    #    b) Connect directly to the test DB (if flask_test_client uses it) and insert.
    #    c) Rely on data pre-populated by fixtures (like conftest.py).

    # Let's assume conftest.py pre-populated some data or we test the retrieval
    # based on the data available.

    # 3. Retrieve Devices via API
    response = flask_test_client.get('/api/devices', headers=headers)
    assert response.status_code == 200
    devices = response.get_json()
    assert isinstance(devices, list)

    # 4. Retrieve a specific device (if any exist)
    if devices:
        device_id = devices[0]['device_id']
        response_single = flask_test_client.get(f'/api/devices/{device_id}', headers=headers)
        assert response_single.status_code == 200
        device_data = response_single.get_json()
        assert device_data['device_id'] == device_id
    else:
        print("Skipping specific device retrieval test as no devices were found.")

@pytest.mark.integration
def test_alert_generation_and_retrieval(flask_test_client: FlaskClient, auth_token: str):
    """
    Tests if alerts generated (simulated) are retrievable via the API.
    Requires alert conditions to be met by test data or manual insertion.
    """
    if flask_test_client is None: pytest.skip("Flask app not available")
    headers = {'Authorization': f'Bearer {auth_token}'}

    # Simulate conditions that trigger an alert (e.g., insert high-value data)
    # This might require direct DB access within the test if no API exists for it.
    # OR: Assume an alert exists from fixture data.

    # Retrieve alerts
    response = flask_test_client.get('/api/alerts?limit=5', headers=headers)
    assert response.status_code == 200
    alerts = response.get_json()
    assert isinstance(alerts, list)

    # If alerts exist, try acknowledging one
    unacknowledged_alerts = [a for a in alerts if not a.get('acknowledged')]
    if unacknowledged_alerts:
        alert_to_ack = unacknowledged_alerts[0]
        alert_id = alert_to_ack['id']
        ack_response = flask_test_client.post(f'/api/alerts/acknowledge/{alert_id}', headers=headers)
        # Allow 404 if alert was somehow removed between GET and POST in concurrent tests
        assert ack_response.status_code in [200, 404]

        if ack_response.status_code == 200:
             # Verify it's now acknowledged
             response_after_ack = flask_test_client.get('/api/alerts?limit=10', headers=headers)
             alerts_after_ack = response_after_ack.get_json()
             acknowledged = False
             for alert in alerts_after_ack:
                 if alert['id'] == alert_id:
                     assert alert['acknowledged'] is True
                     acknowledged = True
                     break
             assert acknowledged, f"Alert {alert_id} was not marked as acknowledged after POST."
    else:
        print("Skipping alert acknowledgment test as no unacknowledged alerts were found.")


@pytest.mark.performance
def test_api_response_time(flask_test_client: FlaskClient, auth_token: str):
    """Tests that major API endpoints respond within a reasonable time."""
    if flask_test_client is None: pytest.skip("Flask app not available")
    headers = {'Authorization': f'Bearer {auth_token}'}
    endpoints_to_test = [
        '/health',
        '/api/dashboard',
        '/api/devices',
        '/api/alerts'
    ]
    max_response_time_seconds = 1.0 # Set a threshold (e.g., 1 second)

    for endpoint in endpoints_to_test:
        start_time = time.time()
        response = flask_test_client.get(endpoint, headers=headers)
        end_time = time.time()
        duration = end_time - start_time

        assert response.status_code == 200, f"Endpoint {endpoint} failed with status {response.status_code}"
        assert duration < max_response_time_seconds, f"Endpoint {endpoint} took {duration:.2f}s (limit {max_response_time_seconds}s)"

@pytest.mark.performance
def test_memory_usage_under_load(flask_test_client: FlaskClient, auth_token: str):
    """Approximates memory usage under simple simulated load."""
    if flask_test_client is None: pytest.skip("Flask app not available")
    headers = {'Authorization': f'Bearer {auth_token}'}
    process = psutil.Process(os.getpid())
    
    initial_memory_mb = process.memory_info().rss / (1024 * 1024)
    print(f"Initial memory usage: {initial_memory_mb:.2f} MB")

    # Simulate multiple API calls
    for _ in range(20):
        flask_test_client.get('/api/dashboard', headers=headers)
        flask_test_client.get('/api/devices', headers=headers)

    final_memory_mb = process.memory_info().rss / (1024 * 1024)
    memory_increase_mb = final_memory_mb - initial_memory_mb
    print(f"Final memory usage: {final_memory_mb:.2f} MB (Increase: {memory_increase_mb:.2f} MB)")

    # Set a generous limit for memory increase during the test run
    # This is highly dependent on the system and other running tests.
    max_allowed_increase_mb = 300
    assert memory_increase_mb < max_allowed_increase_mb, f"Memory usage increased by more than {max_allowed_increase_mb} MB during test"
    # Also check absolute usage
    assert final_memory_mb < 800, f"Final memory usage exceeded 800 MB"

# Add more integration tests as needed, e.g., testing Celery task integration via API calls
# def test_start_report_task_integration(flask_test_client, auth_token):
#     # ... (call POST /api/tasks/start_report) ...
#     # ... (poll GET /api/tasks/status/<task_id>) ...
#     # ... (assert task completes successfully) ...