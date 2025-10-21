# Digital_Twin/TESTS/test_celery_tasks.py
"""
Tests for Celery background tasks.
Uses Celery's testing utilities where possible.
"""

import pytest
from datetime import datetime, timedelta, timezone
import pandas as pd
from unittest.mock import patch, MagicMock
import os

# --- Attempt to import Celery app and tasks ---
try:
    # Important: Ensure the Flask app context is available for tasks
    # This might require initializing the Flask app within the test setup
    # or using Flask-Celery integration fixtures if available.
    # For simplicity, we'll try direct import and use mocking.
    from Digital_Twin.WEB_APPLICATION.enhanced_flask_app_v2 import (
        celery_app,
        generate_report_task,
        export_data_task,
        cleanup_inactive_clients_task,
        run_model_retraining_task,
        DigitalTwinApp # Import the app class to potentially mock its instance
    )
    # Use Celery testing features
    from celery.contrib.testing.worker import start_worker
    CELERY_AVAILABLE = True
except ImportError as e:
    print(f"Skipping Celery tests: Could not import Celery app or tasks - {e}")
    celery_app = None
    generate_report_task = None
    export_data_task = None
    cleanup_inactive_clients_task = None
    run_model_retraining_task = None
    DigitalTwinApp = None
    CELERY_AVAILABLE = False

# --- Fixtures ---

@pytest.fixture(scope="session", autouse=True)
def celery_config():
    """Configure Celery for testing (e.g., use in-memory broker)."""
    if not CELERY_AVAILABLE:
        pytest.skip("Celery not available")
        return {}
        
    # Use 'memory' broker/backend for faster, isolated tests
    # Ensure results are stored for checking status
    config = {
        'broker_url': 'memory://',
        'result_backend': 'cache+memory://',
        'task_always_eager': False,  # Set to False to test with a worker
        'task_eager_propagates': False,
    }
    celery_app.conf.update(config)
    return config

@pytest.fixture(scope="session")
def celery_worker_cls(celery_config):
     """Fixture to provide the worker class with test config."""
     if not CELERY_AVAILABLE:
         pytest.skip("Celery not available")
     # Return the worker class itself
     return celery_app.Worker

@pytest.fixture(scope="module")
def celery_worker_instance(celery_worker_cls):
     """Fixture to start and stop a test Celery worker for the module."""
     if not CELERY_AVAILABLE:
         pytest.skip("Celery not available")
         
     worker = celery_worker_cls(quiet=True, perform_ping_check=False)
     
     # Start worker in a separate thread
     worker_thread = worker.start()
     
     # Wait briefly for worker to potentially initialize
     time.sleep(1)

     # Ensure worker started (basic check)
     if not worker_thread or not worker_thread.is_alive():
         pytest.fail("Celery test worker failed to start.")
         
     print("\nCelery test worker started.")
     yield worker # Provide the running worker instance
     
     # Teardown: Stop the worker
     print("\nStopping Celery test worker...")
     worker.stop()
     worker_thread.join(timeout=5)
     print("Celery test worker stopped.")


# --- Mocks for Dependencies ---

@pytest.fixture
def mock_app_instance():
    """Provides a mock DigitalTwinApp instance for tasks."""
    mock = MagicMock(spec=DigitalTwinApp)
    # Mock methods used by tasks
    mock._cleanup_inactive_clients_logic = MagicMock(return_value=2) # Simulate cleaning 2 clients
    # Mock analytics engine within the app instance
    mock.analytics_engine = MagicMock()
    mock.analytics_engine.retrain_models = MagicMock(return_value={'status': 'success', 'models_updated': 3})
    return mock

@pytest.fixture
def mock_get_session():
    """Mocks the get_session context manager to simulate DB access."""
    mock_session = MagicMock()
    # Simulate finding some data for report/export
    mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [
        MagicMock(to_dict=lambda: {'id': 1, 'device_id': 'DEV_001', 'timestamp': datetime.now(timezone.utc)})
    ]
    
    @contextmanager
    def _mock_session_context():
        yield mock_session
        
    return _mock_session_context


# --- Task Tests ---

# @pytest.mark.usefixtures("celery_worker_instance") # Use this if task_always_eager=False
@pytest.mark.celery
# Mock dependencies needed by the task
@patch('Digital_Twin.WEB_APPLICATION.enhanced_flask_app_v2.get_session')
@patch('Digital_Twin.REPORTS.health_report_generator.HealthReportGenerator')
def test_generate_report_task(MockReportGenerator, mock_get_session_func, celery_app):
    """Test the generate_report_task."""
    if not CELERY_AVAILABLE: pytest.skip("Celery not available")
    
    # Configure mocks
    mock_report_instance = MockReportGenerator.return_value
    mock_report_instance.generate_comprehensive_report.return_value = "/mock/reports/report_123.html"
    mock_get_session_func.side_effect = mock_get_session().side_effect # Use the fixture's context manager

    # --- Execute Task ---
    # Using .apply() allows specifying options likecountdown
    # Using .get() waits for the result (requires result backend)
    # If task_always_eager=True, this runs synchronously
    result = generate_report_task.apply(args=["test_user"]).get(timeout=10) # Wait up to 10s

    # --- Assertions ---
    assert result['status'] == 'SUCCESS'
    assert 'report_path' in result
    assert result['report_path'] == "/mock/reports/report_123.html"
    mock_report_instance.generate_comprehensive_report.assert_called_once()
    mock_get_session_func.assert_called() # Check if DB session was requested


# @pytest.mark.usefixtures("celery_worker_instance")
@pytest.mark.celery
@patch('Digital_Twin.WEB_APPLICATION.enhanced_flask_app_v2.get_session')
@patch('pandas.DataFrame.to_csv') # Mock CSV writing
@patch('json.dump') # Mock JSON writing
def test_export_data_task(mock_json_dump, mock_to_csv, mock_get_session_func, celery_app):
    """Test the export_data_task for both JSON and CSV."""
    if not CELERY_AVAILABLE: pytest.skip("Celery not available")
    
    mock_get_session_func.side_effect = mock_get_session().side_effect

    # --- Test JSON Export ---
    result_json = export_data_task.apply(kwargs={'format_type': 'json', 'date_range_days': 1}).get(timeout=10)

    assert result_json['status'] == 'SUCCESS'
    assert 'export_path' in result_json
    assert result_json['export_path'].endswith('.json')
    assert 'filename' in result_json
    mock_json_dump.assert_called_once() # Check if json.dump was called
    mock_to_csv.assert_not_called() # Ensure CSV wasn't called
    mock_get_session_func.assert_called() # Check DB access

    # Reset mocks for next test
    mock_json_dump.reset_mock()
    mock_to_csv.reset_mock()
    mock_get_session_func.reset_mock()
    mock_get_session_func.side_effect = mock_get_session().side_effect # Re-apply side effect

    # --- Test CSV Export ---
    result_csv = export_data_task.apply(kwargs={'format_type': 'csv', 'date_range_days': 1}).get(timeout=10)

    assert result_csv['status'] == 'SUCCESS'
    assert 'export_path' in result_csv
    assert result_csv['export_path'].endswith('.csv')
    assert 'filename' in result_csv
    mock_to_csv.assert_called_once() # Check if to_csv was called
    mock_json_dump.assert_not_called() # Ensure JSON wasn't called
    mock_get_session_func.assert_called()


# @pytest.mark.usefixtures("celery_worker_instance")
@pytest.mark.celery
@patch('flask.current_app') # Mock current_app used inside the task
def test_cleanup_inactive_clients_task(mock_current_app, mock_app_instance, celery_app):
    """Test the cleanup_inactive_clients_task."""
    if not CELERY_AVAILABLE: pytest.skip("Celery not available")

    # Make current_app return our mock instance via extensions
    mock_current_app.extensions = {'digital_twin_instance': mock_app_instance}

    result = cleanup_inactive_clients_task.apply().get(timeout=10)

    assert result['status'] == 'SUCCESS'
    assert result['cleaned_count'] == 2 # Matches mock return value
    mock_app_instance._cleanup_inactive_clients_logic.assert_called_once()


# @pytest.mark.usefixtures("celery_worker_instance")
@pytest.mark.celery
@patch('flask.current_app')
def test_run_model_retraining_task(mock_current_app, mock_app_instance, celery_app):
    """Test the run_model_retraining_task."""
    if not CELERY_AVAILABLE: pytest.skip("Celery not available")

    mock_current_app.extensions = {'digital_twin_instance': mock_app_instance}

    result = run_model_retraining_task.apply().get(timeout=20) # Allow more time for mock retraining

    assert result['status'] == 'SUCCESS'
    assert 'retraining_result' in result
    assert result['retraining_result']['status'] == 'success'
    assert result['retraining_result']['models_updated'] == 3
    mock_app_instance.analytics_engine.retrain_models.assert_called_once()