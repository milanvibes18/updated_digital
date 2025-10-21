# Digital_Twin/TESTS/test_websockets.py
"""
Test WebSocket functionality using a Socket.IO client.
Requires the Flask test client to be running with SocketIO enabled.
"""
import pytest
import socketio # Use the python-socketio client library
import time
from threading import Event

# Fixture 'flask_test_client' is provided by conftest.py
# Fixture 'auth_token' provides a JWT for authentication

@pytest.fixture(scope="module")
def socketio_client(flask_test_client, auth_token):
    """
    Fixture to create a connected Socket.IO test client.
    Connects to the test server managed by flask_test_client.
    Includes JWT authentication in the connection.
    """
    if flask_test_client is None:
        pytest.skip("Flask app/SocketIO server not available for testing.")

    # Base URL is tricky with test_client. We assume it runs locally.
    # The actual port might be dynamic; this needs careful handling in real CI.
    # For local testing, assuming default or a known test port.
    # A better approach might involve getting the server address from the test setup.
    base_url = "http://127.0.0.1:5000" # Adjust if your test server runs elsewhere

    sio = socketio.Client(reconnection_attempts=3, reconnection_delay=0.5)
    connected_event = Event()
    auth_failed_event = Event()
    connect_error_event = Event()
    connection_established_data = {}

    @sio.event
    def connect():
        print("Test client connected.")
        connected_event.set()

    @sio.event
    def disconnect():
        print("Test client disconnected.")
        connected_event.clear() # Reset event on disconnect

    @sio.event
    def connect_error(data):
        print(f"Test client connection error: {data}")
        connect_error_event.set()
        connected_event.set() # Signal completion even on error for timeout

    @sio.event
    def auth_failed(data):
        print(f"Test client authentication failed: {data}")
        auth_failed_event.set()
        connected_event.set() # Signal completion even on auth failure for timeout

    @sio.event
    def connection_established(data):
        print(f"Test client received connection_established: {data}")
        # Store data received upon successful connection/auth
        connection_established_data.update(data)


    try:
        # Pass auth token for connection
        print(f"Attempting connection to {base_url} with token...")
        sio.connect(base_url, auth={'token': auth_token}, transports=['websocket'])

        # Wait for connection, auth failure, or connect error, with a timeout
        connection_successful = connected_event.wait(timeout=5) # Wait up to 5 seconds

        if connect_error_event.is_set():
            sio.disconnect()
            pytest.fail("Socket.IO connection error during setup.")
        elif auth_failed_event.is_set():
            sio.disconnect()
            pytest.fail("Socket.IO authentication failed during setup.")
        elif not connection_successful:
            sio.disconnect()
            pytest.fail("Socket.IO connection timed out during setup.")
        else:
             # Check if we got the connection_established message
             assert 'status' in connection_established_data and connection_established_data['status'] == 'success'
             assert 'data' in connection_established_data and 'session_id' in connection_established_data['data']

        yield sio # Provide the connected client to tests

    finally:
        # Cleanup: Ensure disconnection after tests in this module run
        if sio.connected:
            sio.disconnect()

@pytest.mark.websocket
def test_websocket_connection_authentication(socketio_client):
    """Tests that the client successfully connected and authenticated."""
    assert socketio_client is not None
    assert socketio_client.connected

@pytest.mark.websocket
def test_websocket_ping_pong(socketio_client):
    """Tests the ping-pong mechanism for keepalive."""
    pong_received = Event()
    pong_data = {}

    @socketio_client.on('pong') # Assuming server sends 'pong'
    def handle_pong(data):
        print(f"Received pong: {data}")
        pong_data.update(data)
        pong_received.set()

    # Send ping
    print("Sending ping...")
    socketio_client.emit('ping') # Assuming client sends 'ping'

    # Wait for pong response
    success = pong_received.wait(timeout=3) # Wait up to 3 seconds
    assert success, "Did not receive pong response within timeout"
    assert 'status' in pong_data and pong_data['status'] == 'success'
    assert 'data' in pong_data and 'timestamp' in pong_data['data']

    # Clean up listener
    socketio_client.off('pong')

@pytest.mark.websocket
def test_websocket_join_leave_room(socketio_client):
    """Tests joining and leaving a room."""
    join_event = Event()
    leave_event = Event()
    test_room = "test_room_123"
    join_response_data = {}
    leave_response_data = {}

    @socketio_client.on('room_joined')
    def handle_join(data):
        print(f"Received room_joined: {data}")
        if data.get('data', {}).get('room') == test_room:
            join_response_data.update(data)
            join_event.set()

    @socketio_client.on('room_left')
    def handle_leave(data):
        print(f"Received room_left: {data}")
        if data.get('data', {}).get('room') == test_room:
             leave_response_data.update(data)
             leave_event.set()

    # Join room
    print(f"Emitting join_room for '{test_room}'...")
    socketio_client.emit('join_room', {'room': test_room})
    join_success = join_event.wait(timeout=3)
    assert join_success, f"Did not receive confirmation for joining room '{test_room}'"
    assert join_response_data.get('status') == 'success'

    # Leave room
    print(f"Emitting leave_room for '{test_room}'...")
    socketio_client.emit('leave_room', {'room': test_room})
    leave_success = leave_event.wait(timeout=3)
    assert leave_success, f"Did not receive confirmation for leaving room '{test_room}'"
    assert leave_response_data.get('status') == 'success'

    # Clean up listeners
    socketio_client.off('room_joined')
    socketio_client.off('room_left')

@pytest.mark.websocket
def test_websocket_receive_broadcast(socketio_client, flask_test_client, auth_token):
    """
    Tests if a client receives a broadcast message sent via the Flask app.
    This requires the Flask app and SocketIO server to share the same message queue (e.g., Redis).
    """
    message_received = Event()
    received_data = {}
    test_event = "test_broadcast_event"
    test_data = {"message": "hello broadcast", "value": 42}
    test_room = "broadcast_test_room"

    @socketio_client.on(test_event)
    def handle_broadcast(data):
        print(f"Received broadcast event '{test_event}': {data}")
        received_data.update(data)
        message_received.set()

    # Client joins the room first
    join_event = Event()
    @socketio_client.on('room_joined')
    def handle_join(data):
        if data.get('data', {}).get('room') == test_room: join_event.set()
    socketio_client.emit('join_room', {'room': test_room})
    assert join_event.wait(timeout=3), f"Failed to join room {test_room} for broadcast test"
    socketio_client.off('room_joined') # Remove listener


    # Trigger broadcast from Flask app (requires an endpoint or internal mechanism)
    # Simulate by directly calling the broadcast function if possible,
    # or by hitting a dedicated test endpoint.
    # We'll assume the `enhanced_flask_app_v2` instance is accessible via `flask_test_client.application.extensions`
    app_instance = flask_test_client.application.extensions.get("digital_twin_instance")
    if app_instance and app_instance.socketio:
        print(f"Sending broadcast message to room '{test_room}' via Flask app...")
        # Make sure the data structure matches your expected payload schema
        broadcast_payload = {'status': 'info', 'data': test_data}
        app_instance.socketio.emit(test_event, broadcast_payload, room=test_room)
    else:
        pytest.skip("Cannot access app instance or socketio object to trigger broadcast.")

    # Wait for the client to receive the broadcast
    receive_success = message_received.wait(timeout=5)
    assert receive_success, f"Client did not receive broadcast event '{test_event}' within timeout"
    # Check if the received data matches the 'data' part of the broadcast payload
    assert received_data.get('data') == test_data

    # Clean up listener
    socketio_client.off(test_event)
    # Clean up room
    socketio_client.emit('leave_room', {'room': test_room})


# Add more tests:
# - Test specific data update events ('dashboard_update', 'alert_update')
# - Test error handling (e.g., sending invalid data)
# - Test multiple clients interacting
# - Test reconnection scenarios (harder to automate reliably)