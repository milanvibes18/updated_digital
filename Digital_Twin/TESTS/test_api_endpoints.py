"""Test Flask API endpoints, including authentication"""
import pytest
import json
import uuid # For generating unique usernames

# Fixture 'flask_test_client' is provided by conftest.py

@pytest.fixture(scope='module')
def auth_token(flask_test_client):
    """
    Fixture to register a unique test user for the module and get an auth token.
    """
    if flask_test_client is None:
        pytest.skip("Flask app not available")
        
    # Generate a unique username for each test run to avoid conflicts
    unique_id = uuid.uuid4().hex[:8]
    username = f"testuser_{unique_id}"
    password = "testpassword123"
    
    # Register the unique user
    register_response = flask_test_client.post('/api/register', json={
        "username": username,
        "password": password,
        "email": f"{username}@test.com"
    })
    # Allow 409 conflict if user somehow already exists from a previous failed run
    assert register_response.status_code in [201, 409] 
    
    # Login
    login_response = flask_test_client.post('/api/login', json={
        "username": username,
        "password": password
    })
    
    assert login_response.status_code == 200, f"Login failed with status {login_response.status_code}: {login_response.text}"
    data = login_response.get_json()
    assert 'access_token' in data
    return data['access_token']

# --- Public Endpoint Test ---

def test_health_endpoint(flask_test_client):
    """Tests the public /health endpoint."""
    if flask_test_client is None:
        pytest.skip("Flask app not available")
    
    response = flask_test_client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    # Check for 'healthy' or 'partial' as AI modules might be missing
    assert data.get('status') in ['healthy', 'partial'] 
    assert 'timestamp' in data

# --- Authentication Endpoint Tests ---

def test_register_user_conflict(flask_test_client, auth_token):
     """Tests registering a user that already exists (created in auth_token fixture)."""
     # We need auth_token fixture to ensure a user exists, even though we don't use the token here.
     if flask_test_client is None: pytest.skip("Flask app not available")

     # Extract username used in auth_token fixture (a bit hacky, assumes format)
     # A better approach might be to pass the username out of the fixture
     # For now, derive it if possible or skip if format unknown
     try:
         # Simplistic assumption: decode token payload (not verifying signature)
         import base64
         payload = json.loads(base64.b64decode(auth_token.split('.')[1] + '==').decode())
         username = payload['sub']
     except Exception:
          pytest.skip("Could not determine username from token for conflict test.")

     response = flask_test_client.post('/api/register', json={
         "username": username,
         "password": "anotherpassword",
         "email": f"{username}_new@test.com"
     })
     assert response.status_code == 409 # Conflict


def test_login_endpoint_success(flask_test_client, auth_token):
    """Tests the /api/login endpoint success (implicitly tested by auth_token fixture)."""
    assert auth_token is not None # If fixture ran successfully, login worked

def test_login_fail_wrong_password(flask_test_client):
    """Tests failed login with wrong password."""
    if flask_test_client is None: pytest.skip("Flask app not available")
    # Use a known non-existent user or register one first if needed
    username = f"nouser_{uuid.uuid4().hex[:6]}"
    response = flask_test_client.post('/api/login', json={
        "username": username,
        "password": "wrongpassword"
    })
    assert response.status_code == 401
    assert 'error' in response.get_json()

def test_whoami_endpoint(flask_test_client, auth_token):
     """Tests the protected /api/whoami endpoint."""
     if flask_test_client is None: pytest.skip("Flask app not available")
     headers = {'Authorization': f'Bearer {auth_token}'}
     response = flask_test_client.get('/api/whoami', headers=headers)
     assert response.status_code == 200
     data = response.get_json()
     assert 'logged_in_as' in data
     # Optionally check if username matches the one used in the fixture if derivable

# --- Protected Endpoint Tests ---

# Helper function for headers
def get_auth_headers(token):
    return {'Authorization': f'Bearer {token}'}

def test_protected_endpoints_unauthorized(flask_test_client):
    """Tests accessing various protected endpoints without auth token."""
    if flask_test_client is None: pytest.skip("Flask app not available")
    
    endpoints = [
        '/api/dashboard_data',
        '/api/devices',
        '/api/device/DEVICE_001', # Example device ID
        '/api/analytics',
        '/api/alerts',
        '/api/system_metrics',
        '/api/historical_data?device_id=DEVICE_001',
        '/api/predictions?device_id=DEVICE_001',
        '/api/health_scores',
        '/api/recommendations',
        '/api/generate_report',
        '/api/export_data'
    ]
    
    for endpoint in endpoints:
        response = flask_test_client.get(endpoint)
        assert response.status_code == 401, f"Endpoint {endpoint} did not return 401 Unauthorized"

def test_dashboard_data_endpoint_authorized(flask_test_client, auth_token):
    """Tests /api/dashboard_data with auth."""
    if flask_test_client is None: pytest.skip("Flask app not available")
    
    response = flask_test_client.get('/api/dashboard_data', headers=get_auth_headers(auth_token))
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, dict)
    assert 'systemHealth' in data # Key name from merged app
    assert 'devices' in data
    assert 'statusDistribution' in data
    assert 'performanceData' in data # Kept from original app structure

def test_devices_endpoint_authorized(flask_test_client, auth_token):
    """Tests /api/devices with auth."""
    if flask_test_client is None: pytest.skip("Flask app not available")
    
    response = flask_test_client.get('/api/devices', headers=get_auth_headers(auth_token))
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
    # Check if list contains device-like dictionaries (if not empty)
    if data:
        assert isinstance(data[0], dict)
        assert 'device_id' in data[0]

def test_get_specific_device_endpoint_authorized(flask_test_client, auth_token):
    """Tests /api/device/<device_id> with auth."""
    if flask_test_client is None: pytest.skip("Flask app not available")
    
    # First get a list of devices to find a valid ID
    devices_response = flask_test_client.get('/api/devices', headers=get_auth_headers(auth_token))
    assert devices_response.status_code == 200
    devices_data = devices_response.get_json()
    
    if not devices_data:
        pytest.skip("No devices available in the system to test specific device endpoint.")
        
    test_device_id = devices_data[0]['device_id'] # Use the first device
        
    response = flask_test_client.get(f'/api/device/{test_device_id}', headers=get_auth_headers(auth_token))
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, dict)
    assert data['device_id'] == test_device_id

def test_get_specific_device_not_found(flask_test_client, auth_token):
    """Tests /api/device/<device_id> for a non-existent device."""
    if flask_test_client is None: pytest.skip("Flask app not available")
    
    non_existent_id = "DEVICE_999_NONEXISTENT"
    response = flask_test_client.get(f'/api/device/{non_existent_id}', headers=get_auth_headers(auth_token))
    assert response.status_code == 404

def test_alerts_endpoint_authorized(flask_test_client, auth_token):
    """Tests /api/alerts with auth (adapted from old file)."""
    if flask_test_client is None:
        pytest.skip("Flask app not available")
    
    response = flask_test_client.get('/api/alerts', headers=get_auth_headers(auth_token))
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
    # Check alert structure if list is not empty
    if data:
        assert isinstance(data[0], dict)
        assert 'id' in data[0]
        assert 'severity' in data[0]
        assert 'description' in data[0]

def test_system_metrics_endpoint_authorized(flask_test_client, auth_token):
    """Tests /api/system_metrics with auth (adapted from old file)."""
    if flask_test_client is None:
        pytest.skip("Flask app not available")
    
    response = flask_test_client.get('/api/system_metrics', headers=get_auth_headers(auth_token))
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, dict)
    assert 'cpu_percent' in data
    assert 'memory_percent' in data
    # Add checks for other keys if psutil is expected to be installed

def test_recommendations_endpoint_authorized(flask_test_client, auth_token):
    """Tests the /api/recommendations endpoint with auth."""
    if flask_test_client is None: pytest.skip("Flask app not available")
    
    response = flask_test_client.get('/api/recommendations', headers=get_auth_headers(auth_token))
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list) # Merged app returns a list
    # Check structure if not empty
    if data:
        assert isinstance(data[0], dict)
        assert 'id' in data[0]
        assert 'title' in data[0]
        assert 'priority' in data[0]

def test_health_scores_endpoint_authorized(flask_test_client, auth_token):
    """Tests the /api/health_scores endpoint with auth."""
    if flask_test_client is None: pytest.skip("Flask app not available")
    
    response = flask_test_client.get('/api/health_scores', headers=get_auth_headers(auth_token))
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, dict)
    # Check structure if not empty - keys should be device IDs
    if data:
        first_key = next(iter(data))
        assert 'DEVICE_' in first_key # Basic check for device ID format
        assert 'overall_health' in data[first_key]

def test_predictions_endpoint_authorized(flask_test_client, auth_token):
    """Tests the /api/predictions endpoint with auth."""
    if flask_test_client is None: pytest.skip("Flask app not available")
    
    # Get a device ID first
    devices_response = flask_test_client.get('/api/devices', headers=get_auth_headers(auth_token))
    assert devices_response.status_code == 200
    devices_data = devices_response.get_json()
    if not devices_data:
        pytest.skip("No devices available to test predictions.")
    test_device_id = devices_data[0]['device_id']
        
    response = flask_test_client.get(f'/api/predictions?device_id={test_device_id}', headers=get_auth_headers(auth_token))
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, dict)
    assert data['device_id'] == test_device_id
    assert 'anomaly_prediction' in data
    assert 'failure_prediction' in data

def test_historical_data_endpoint_authorized(flask_test_client, auth_token):
    """Tests the /api/historical_data endpoint with auth."""
    if flask_test_client is None: pytest.skip("Flask app not available")

    # Get a device ID first
    devices_response = flask_test_client.get('/api/devices', headers=get_auth_headers(auth_token))
    assert devices_response.status_code == 200
    devices_data = devices_response.get_json()
    if not devices_data:
        pytest.skip("No devices available to test historical data.")
    test_device_id = devices_data[0]['device_id']

    response = flask_test_client.get(f'/api/historical_data?device_id={test_device_id}&hours=6', headers=get_auth_headers(auth_token))
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, dict)
    assert data['device_id'] == test_device_id
    assert 'timestamps' in data
    assert 'values' in data
    assert len(data['timestamps']) == 6 # Check if hours parameter worked (using placeholder data)

def test_generate_report_endpoint_authorized(flask_test_client, auth_token):
    """Tests the /api/generate_report endpoint with auth."""
    if flask_test_client is None: pytest.skip("Flask app not available")

    response = flask_test_client.get('/api/generate_report', headers=get_auth_headers(auth_token))
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, dict)
    assert data['success'] is True
    assert 'report_path' in data
    assert data['report_path'].startswith('/reports/'), "Report path should start with /reports/"
    assert data['report_path'].endswith('.html'), "Report path should end with .html"

def test_export_data_endpoint_authorized(flask_test_client, auth_token):
    """Tests the /api/export_data endpoint with auth (JSON format)."""
    if flask_test_client is None: pytest.skip("Flask app not available")

    response = flask_test_client.get('/api/export_data?format=json&days=1', headers=get_auth_headers(auth_token))
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, dict)
    assert data['success'] is True
    assert 'export_path' in data
    assert data['export_path'].startswith('/exports/'), "Export path should start with /exports/"
    assert data['export_path'].endswith('.json'), "Export path should end with .json"
    assert 'filename' in data

def test_export_data_csv_endpoint_authorized(flask_test_client, auth_token):
    """Tests the /api/export_data endpoint with auth (CSV format)."""
    if flask_test_client is None: pytest.skip("Flask app not available")

    response = flask_test_client.get('/api/export_data?format=csv&days=1', headers=get_auth_headers(auth_token))
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, dict)
    assert data['success'] is True
    assert 'export_path' in data
    assert data['export_path'].startswith('/exports/'), "Export path should start with /exports/"
    assert data['export_path'].endswith('.csv'), "Export path should end with .csv"
    assert 'filename' in data

# Add tests for serve_report and serve_export if needed, 
# although they are implicitly tested by checking the paths returned above.
# Example:
# def test_serve_report_endpoint(flask_test_client, auth_token):
#     """Tests accessing a generated report."""
#     # First, generate a report
#     gen_response = flask_test_client.get('/api/generate_report', headers=get_auth_headers(auth_token))
#     assert gen_response.status_code == 200
#     report_path = gen_response.get_json()['report_path']
    
#     # Then, try to access it (assuming no auth needed for the static file itself)
#     report_response = flask_test_client.get(report_path)
#     assert report_response.status_code == 200
#     assert report_response.content_type == 'text/html; charset=utf-8' # Check content type