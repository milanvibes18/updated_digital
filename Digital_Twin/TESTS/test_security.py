# Digital_Twin/TESTS/test_security.py
"""
Test security features including encryption, input validation, authentication,
authorization (RBAC), and rate limiting.
"""

import pytest
import json
import time
from flask.testing import FlaskClient
import base64
import uuid

# Fixtures 'flask_test_client', 'auth_token' are provided by conftest.py

# --- Encryption Tests ---

def test_database_encryption_roundtrip():
    """Tests the encrypt_data and decrypt_data methods."""
    try:
        from Digital_Twin.AI_MODULES.secure_database_manager import SecureDatabaseManager
        # Requires ENCRYPTION_KEY to be set in the environment for the test runner
        # This should be handled by the test setup (e.g., pytest-dotenv or CI secrets)
        db_manager = SecureDatabaseManager() # Assumes key is loaded on init
    except ImportError:
        pytest.skip("SecureDatabaseManager not available")
    except ValueError as e:
         pytest.skip(f"Skipping encryption test: {e}") # Skip if env var is missing

    test_data_dict = {"temperature": 25.5, "pressure": 1013.2, "notes": "Test data"}
    test_data_str = "Simple string test"

    # Test dictionary
    encrypted_dict = db_manager.encrypt_data(test_data_dict)
    assert isinstance(encrypted_dict, str)
    decrypted_dict_str = db_manager.decrypt_data(encrypted_dict)
    decrypted_dict = json.loads(decrypted_dict_str)
    assert decrypted_dict == test_data_dict

    # Test string
    encrypted_str = db_manager.encrypt_data(test_data_str)
    assert isinstance(encrypted_str, str)
    decrypted_str = db_manager.decrypt_data(encrypted_str)
    assert decrypted_str == test_data_str

def test_database_encryption_tampering():
    """Tests that tampered encrypted data raises an InvalidTag error."""
    try:
        from Digital_Twin.AI_MODULES.secure_database_manager import SecureDatabaseManager
        from cryptography.exceptions import InvalidTag
        db_manager = SecureDatabaseManager()
    except ImportError:
        pytest.skip("SecureDatabaseManager or InvalidTag not available")
    except ValueError as e:
         pytest.skip(f"Skipping encryption test: {e}")

    encrypted = db_manager.encrypt_data("Original data")
    encrypted_bytes = base64.b64decode(encrypted.encode('utf-8'))

    # Tamper with the last byte (part of the GCM tag)
    tampered_bytes = encrypted_bytes[:-1] + bytes([encrypted_bytes[-1] ^ 1])
    tampered_encrypted_str = base64.b64encode(tampered_bytes).decode('utf-8')

    with pytest.raises(InvalidTag):
        db_manager.decrypt_data(tampered_encrypted_str)

# --- Input Validation Tests ---

def test_register_input_validation_failures(flask_test_client: FlaskClient):
    """Tests input validation failures for the registration endpoint."""
    if flask_test_client is None: pytest.skip("Flask app not available")

    invalid_inputs = [
        ({"username": "us", "password": "password123", "email": "a@b.com"}, 400), # Username too short
        ({"username": "user1", "password": "pass", "email": "a@b.com"}, 400),     # Password too short
        ({"username": "user2", "password": "password123", "email": "invalid"}, 400), # Invalid email
        ({"username": "user3", "password": "password123"}, 400),                  # Missing email
        ({}, 400),                                                                 # Empty data
    ]

    for data, expected_status in invalid_inputs:
        response = flask_test_client.post('/api/auth/register', json=data)
        assert response.status_code == expected_status, f"Expected {expected_status} for input {data}, got {response.status_code}"
        assert 'error' in response.get_json()
        assert 'Validation failed' in response.get_json()['error']

def test_login_input_validation_failures(flask_test_client: FlaskClient):
    """Tests input validation failures for the login endpoint."""
    if flask_test_client is None: pytest.skip("Flask app not available")

    invalid_inputs = [
        ({"username": "user"}, 400), # Missing password
        ({"password": "password"}, 400), # Missing username
        ({}, 400),                    # Empty data
    ]

    for data, expected_status in invalid_inputs:
        response = flask_test_client.post('/api/auth/login', json=data)
        assert response.status_code == expected_status, f"Expected {expected_status} for input {data}, got {response.status_code}"
        assert 'error' in response.get_json()
        assert 'Validation failed' in response.get_json()['error']

# --- Rate Limiting Tests ---

@pytest.mark.slow # Mark as slow because it involves waiting
def test_rate_limiting_login(flask_test_client: FlaskClient):
    """Tests rate limiting on the login endpoint."""
    if flask_test_client is None: pytest.skip("Flask app not available")

    # Assuming login limit is "10 per minute" from the code
    limit = 10
    username = f"ratelimit_user_{uuid.uuid4().hex[:6]}"
    password = "password"

    # Make requests up to the limit - expect 401 (invalid credentials) or 400 (if validation fails on empty)
    for i in range(limit):
        response = flask_test_client.post('/api/auth/login', json={"username": username, "password": password})
        assert response.status_code in [401, 400], f"Request {i+1}/{limit} failed unexpectedly: {response.status_code}"

    # The next request should be rate-limited (429)
    response_over_limit = flask_test_client.post('/api/auth/login', json={"username": username, "password": password})
    assert response_over_limit.status_code == 429, f"Expected 429 after {limit} requests, got {response_over_limit.status_code}"
    assert "Rate limit exceeded" in response_over_limit.get_json()['error']

    # Optional: Wait for the rate limit window to reset and try again
    # print("Waiting for rate limit window to reset (approx 60s)...")
    # time.sleep(61)
    # response_after_wait = flask_test_client.post('/api/auth/login', json={"username": username, "password": password})
    # assert response_after_wait.status_code in [401, 400], "Request after waiting should not be rate limited"


# --- Authentication & Authorization Tests (Basic checks, more in test_api_endpoints.py) ---

def test_access_protected_endpoint_without_token(flask_test_client: FlaskClient):
    """Tests accessing a protected endpoint without any token."""
    if flask_test_client is None: pytest.skip("Flask app not available")
    response = flask_test_client.get('/api/auth/me')
    assert response.status_code == 401 # Should require auth

def test_access_protected_endpoint_with_invalid_token(flask_test_client: FlaskClient):
    """Tests accessing a protected endpoint with a fake/invalid token."""
    if flask_test_client is None: pytest.skip("Flask app not available")
    headers = {'Authorization': 'Bearer invalid.token.string'}
    response = flask_test_client.get('/api/auth/me', headers=headers)
    assert response.status_code == 422 # flask-jwt-extended returns 422 for malformed tokens

# Placeholder for Role-Based Access Control (RBAC) tests
@pytest.mark.skip(reason="RBAC test needs user setup with specific roles")
def test_rbac_admin_only_endpoint(flask_test_client: FlaskClient, auth_token: str):
    """Tests accessing an admin-only endpoint."""
    if flask_test_client is None: pytest.skip("Flask app not available")
    
    # 1. Access as a regular user (using auth_token from fixture, assuming it's 'user' role)
    headers_user = {'Authorization': f'Bearer {auth_token}'}
    response_user = flask_test_client.get('/api/admin/summary', headers=headers_user)
    assert response_user.status_code == 403 # Forbidden for regular user

    # 2. Access as an admin user (Requires creating an admin user and getting their token)
    #    This part needs modification based on how admin users are created/managed.
    #    admin_token = get_admin_token() # Helper function needed
    #    headers_admin = {'Authorization': f'Bearer {admin_token}'}
    #    response_admin = flask_test_client.get('/api/admin/summary', headers=headers_admin)
    #    assert response_admin.status_code == 200 # Allowed for admin
    pass