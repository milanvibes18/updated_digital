#!/usr/bin/env python3
"""
Advanced WebSocket manager for real-time data streaming, room management,
JWT authentication, and scalable communication in the Digital Twin system.
(File: Digital_Twin/AI_MODULES/realtime_websocket_manager.py)
"""

import json
import logging
import uuid
import time
from datetime import datetime
from typing import Dict, List, Set, Callable, Any, Optional
from flask import request, current_app
from flask_socketio import SocketIO, emit, join_room, leave_room
import threading
import queue
from collections import defaultdict, deque
import redis
import jwt # <-- Added for JWT validation
import os # <-- Added for path handling

class RealtimeWebSocketManager:
    """
    Advanced WebSocket manager for real-time data streaming, room management,
    JWT authentication, and scalable communication in the Digital Twin system.
    """

    def __init__(self, redis_url="redis://localhost:6379/0", max_connections=1000):
        self.app = None
        self.socketio: Optional[SocketIO] = None
        self.redis_client: Optional[redis.Redis] = None
        self.redis_url = redis_url # <-- Store Redis URL
        self.logger = self._setup_logging()
        self.max_connections = max_connections

        # Connection management
        self.active_connections: Dict[str, Dict] = {}
        self.room_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self.user_rooms: Dict[str, Set[str]] = defaultdict(set)

        # Data streaming
        self.data_streams: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.stream_subscribers: Dict[str, Set[str]] = defaultdict(set)

        # Rate limiting
        self.rate_limits: Dict[str, List[float]] = defaultdict(list)
        self.rate_limit_window = 60  # seconds
        self.rate_limit_max = 100   # requests per window (default)

        # Message queue for async processing
        self.message_queue = queue.Queue()
        self.processing_thread = None
        self.should_stop = threading.Event()

        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)

        # Initialize components
        self._initialize_redis() # <-- Uses self.redis_url

    def _setup_logging(self):
        """Setup logging for WebSocket manager."""
        logger = logging.getLogger('RealtimeWebSocketManager')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            try:
                # Use absolute path based on this file's location
                log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'LOGS')
                os.makedirs(log_dir, exist_ok=True)
                handler = logging.FileHandler(os.path.join(log_dir, 'digital_twin_websocket.log'))
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                # Add console handler
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
            except (AttributeError, FileNotFoundError, ImportError, NameError): # Added NameError for sys
                print("Warning: Could not configure file logging. Logging to console.")
                handler = logging.StreamHandler()
                logger.addHandler(handler)

        return logger

    def _initialize_redis(self):
        """Initialize Redis connection for scalability."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            self.redis_client.ping()
            self.logger.info(f"Redis connection established at {self.redis_url}")
        except Exception as e:
            self.logger.warning(f"Redis connection failed: {e}. Running without Redis.")
            self.redis_client = None

    def initialize_socketio(self, app):
        """Initialize SocketIO with the Flask app."""
        try:
            self.app = app

            # Configure SocketIO with Redis if available
            message_queue_url = self.redis_url if self.redis_client else None

            self.socketio = SocketIO(
                app,
                cors_allowed_origins=app.config.get('CORS_ALLOWED_ORIGINS', '*'),
                async_mode='eventlet', # Make sure eventlet is installed
                message_queue=message_queue_url,
                logger=app.config.get('DEBUG', False),
                engineio_logger=app.config.get('DEBUG', False),
                ping_timeout=60,
                ping_interval=25,
                # Consider adding channel if using multiple SocketIO instances with same Redis
                # channel=os.environ.get('SOCKETIO_CHANNEL', 'digital_twin')
            )

            # --- Rate Limit Config (moved here for app context access) ---
            self.rate_limit_window = app.config.get('WEBSOCKET_RATE_LIMIT_WINDOW', 60)
            self.rate_limit_max = app.config.get('WEBSOCKET_RATE_LIMIT_MAX', 100)
            self.logger.info(f"WebSocket rate limiting configured: {self.rate_limit_max} requests per {self.rate_limit_window} seconds.")
            # --- End Rate Limit Config ---

            self._register_event_handlers()
            self._start_processing_thread()

            self.logger.info("SocketIO initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize SocketIO: {e}", exc_info=True) # Added exc_info
            raise

    def _register_event_handlers(self):
        """Register default SocketIO event handlers."""

        @self.socketio.on('connect')
        def handle_connect(auth: Optional[Dict] = None):
            """
            Handle client connection and JWT authentication.
            Client must send {'token': '...'} in the auth payload or query string,
            or rely on HttpOnly cookie.
            """
            try:
                token = None
                # 1. Check auth payload
                if auth and 'token' in auth:
                    token = auth.get('token')
                # 2. Check query string
                elif request.args.get('token'):
                    token = request.args.get('token')
                # 3. Check Authorization header (less common for websockets but possible)
                else:
                    auth_header = request.headers.get('Authorization')
                    if auth_header and auth_header.startswith('Bearer '):
                        token = auth_header.split(' ')[1]
                # 4. Check for HttpOnly cookie (using flask_jwt_extended helper)
                # This requires verify_jwt_in_request to be flexible.
                # If using cookies, the check might happen implicitly later,
                # or we rely solely on token for WS. Let's assume token is primary for WS.

                if not token:
                    # If NO token methods worked, attempt implicit cookie verification IF configured
                    # Note: This usually requires @jwt_required from flask_jwt_extended on the handler,
                    # which isn't standard for socketio.on('connect'). We'll stick to explicit token for WS.
                    self.logger.warning("Connection attempt without explicit token.", sid=request.sid)
                    emit('auth_failed', {'status': 'error', 'message': 'Authentication token required via auth payload or query string.'})
                    return False # Reject connection

                try:
                    # Decode JWT using flask_jwt_extended's secret
                    secret = current_app.config['JWT_SECRET_KEY']
                    # Using decode_token handles expiry and signature verification
                    payload = decode_token(token)
                    user_id = payload.get('sub')
                    user_role = payload.get('role', 'user')

                    if not user_id:
                        self.logger.warning("Invalid token payload (missing 'sub').", sid=request.sid)
                        emit('auth_failed', {'status': 'error', 'message': 'Invalid token payload.'})
                        return False

                except ExpiredSignatureError:
                    self.logger.warning("Connection attempt with expired token.", sid=request.sid)
                    emit('auth_failed', {'status': 'error', 'message': 'Token has expired.'})
                    return False
                except (DecodeError, InvalidTokenError, Exception) as e: # Catch broader errors
                    self.logger.warning(f"Invalid token: {e}", sid=request.sid)
                    emit('auth_failed', {'status': 'error', 'message': 'Invalid authentication token.'})
                    return False

                # --- Authentication Successful ---
                session_id = request.sid

                if len(self.active_connections) >= self.max_connections:
                    self.logger.warning(f"Maximum connections reached. Rejecting user {user_id}.", sid=request.sid)
                    emit('error', {'status': 'error', 'message': 'Server is at capacity. Please try again later.'})
                    return False

                client_info = {
                    'session_id': session_id,
                    'user_id': user_id,
                    'role': user_role,
                    'connected_at': datetime.now().isoformat(),
                    'ip_address': self._get_client_ip(),
                    'user_agent': self._get_user_agent(),
                    'rooms': set(),
                    'last_activity': time.time()
                }

                self.active_connections[session_id] = client_info

                emit('connection_established', {
                    'status': 'success',
                    'data': {
                        'session_id': session_id,
                        'user_id': user_id,
                        'role': user_role,
                        'server_time': datetime.now().isoformat(),
                    }
                })

                self.logger.info(f"Client connected: {session_id} (User: {user_id}, Role: {user_role})")

                # Try sending initial data if available (e.g., from cache)
                # Ensure cache logic doesn't block the connection handler
                try:
                    # Safely access cache if available
                    cache = getattr(current_app.extensions.get("digital_twin_instance"), 'cache', None)
                    if cache:
                         initial_data = cache.get('dashboard_combined_data')
                         if initial_data:
                             emit('dashboard_update', {'data': initial_data}) # Use specific event
                             self.logger.debug("Sent initial dashboard data on connect.", sid=session_id)
                except Exception as e:
                    self.logger.warning(f"Could not send initial dashboard data: {e}", sid=session_id)

                return True # Confirm connection

            except Exception as e:
                self.logger.error(f"Connection error: {e}", exc_info=True)
                # Avoid emitting if socket might be invalid
                # emit('error', {'status': 'error', 'message': 'An internal server error occurred during connection.'})
                return False

        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            session_id = request.sid
            try:
                if session_id in self.active_connections:
                    client_rooms = self.active_connections[session_id].get('rooms', set()).copy() # Iterate copy
                    user_id = self.active_connections[session_id].get('user_id', 'unknown')

                    # Remove client info *before* potentially slow room cleanup
                    del self.active_connections[session_id]
                    self.user_rooms.pop(session_id, None) # Also remove user's room list

                    # Leave rooms
                    for room in client_rooms:
                        self._leave_room_internal(session_id, room) # Internal cleanup without socketio call

                    self.logger.info(f"Client disconnected: {session_id} (User: {user_id})")
                else:
                    self.logger.info(f"Client disconnected (untracked): {session_id}")
            except Exception as e:
                self.logger.error(f"Disconnection error: {e}", exc_info=True, sid=session_id)

        @self.socketio.on('join_room')
        @self.socketio.on('subscribe') # Alias for compatibility
        def handle_join_room(data):
            """Handle room join requests with rate limiting."""
            session_id = request.sid
            try:
                # --- Rate Limit Check ---
                if not self._check_rate_limit(session_id):
                    self.logger.warning("Rate limit exceeded for join_room", sid=session_id)
                    emit('error', {'status': 'error', 'message': 'Rate limit exceeded'})
                    return
                # --- End Check ---

                room_name = data.get('room') or data.get('room_name')
                if not room_name:
                    emit('error', {'status': 'error', 'message': 'Room name required'})
                    return

                self._join_room(session_id, room_name)

                emit('room_joined', {
                    'status': 'success',
                    'data': {
                        'room': room_name,
                        'members_count': len(self.room_subscriptions.get(room_name, set()))
                    }
                })
                emit('subscribed', {'data': {'room': room_name}}) # Compatibility

            except Exception as e:
                self.logger.error(f"Join room error: {e}", exc_info=True, sid=session_id, room=data.get('room'))
                emit('error', {'status': 'error', 'message': 'Failed to join room'})

        @self.socketio.on('leave_room')
        @self.socketio.on('unsubscribe') # Alias
        def handle_leave_room(data):
            """Handle room leave requests with rate limiting."""
            session_id = request.sid
            try:
                # --- Rate Limit Check ---
                if not self._check_rate_limit(session_id):
                    self.logger.warning("Rate limit exceeded for leave_room", sid=session_id)
                    emit('error', {'status': 'error', 'message': 'Rate limit exceeded'})
                    return
                # --- End Check ---

                room_name = data.get('room') or data.get('room_name')
                if not room_name:
                    emit('error', {'status': 'error', 'message': 'Room name required'})
                    return

                self._leave_room(session_id, room_name)

                emit('room_left', {
                    'status': 'success',
                    'data': {'room': room_name}
                })
                emit('unsubscribed', {'data': {'room': room_name}}) # Compatibility

            except Exception as e:
                self.logger.error(f"Leave room error: {e}", exc_info=True, sid=session_id, room=data.get('room'))
                emit('error', {'status': 'error', 'message': 'Failed to leave room'})

        @self.socketio.on('subscribe_stream')
        def handle_subscribe_stream(data):
            """Handle data stream subscription with rate limiting."""
            session_id = request.sid
            try:
                # --- Rate Limit Check ---
                if not self._check_rate_limit(session_id):
                    self.logger.warning("Rate limit exceeded for subscribe_stream", sid=session_id)
                    emit('error', {'status': 'error', 'message': 'Rate limit exceeded'})
                    return
                # --- End Check ---

                stream_id = data.get('stream_id')
                if not stream_id:
                    emit('error', {'status': 'error', 'message': 'Stream ID required'})
                    return

                self._subscribe_to_stream(session_id, stream_id)

                recent_data = list(self.data_streams.get(stream_id, []))[-10:] # Use get with default
                if recent_data:
                    emit('stream_data', {
                        'status': 'success',
                        'data': {
                            'stream_id': stream_id,
                            'payload': recent_data,
                            'type': 'historical'
                        }
                    })

                emit('stream_subscribed', {
                    'status': 'success',
                    'data': {'stream_id': stream_id}
                })

            except Exception as e:
                self.logger.error(f"Subscribe stream error: {e}", exc_info=True, sid=session_id, stream=data.get('stream_id'))
                emit('error', {'status': 'error', 'message': 'Failed to subscribe to stream'})

        @self.socketio.on('unsubscribe_stream')
        def handle_unsubscribe_stream(data):
            """Handle data stream unsubscription with rate limiting."""
            session_id = request.sid
            try:
                # --- Rate Limit Check ---
                if not self._check_rate_limit(session_id):
                    self.logger.warning("Rate limit exceeded for unsubscribe_stream", sid=session_id)
                    emit('error', {'status': 'error', 'message': 'Rate limit exceeded'})
                    return
                # --- End Check ---

                stream_id = data.get('stream_id')
                if not stream_id:
                    emit('error', {'status': 'error', 'message': 'Stream ID required'})
                    return

                self._unsubscribe_from_stream(session_id, stream_id)

                emit('stream_unsubscribed', {
                    'status': 'success',
                    'data': {'stream_id': stream_id}
                })

            except Exception as e:
                self.logger.error(f"Unsubscribe stream error: {e}", exc_info=True, sid=session_id, stream=data.get('stream_id'))
                emit('error', {'status': 'error', 'message': 'Failed to unsubscribe from stream'})

        @self.socketio.on('ping')
        @self.socketio.on('ping_from_client') # Alias
        def handle_ping(data=None): # Accept optional data
            """Handle ping requests for connection health."""
            session_id = request.sid
            try:
                if session_id in self.active_connections:
                    self.active_connections[session_id]['last_activity'] = time.time()

                # Include received timestamp if client sent one
                response_data = {'timestamp': datetime.now().isoformat()}
                if isinstance(data, dict) and 'timestamp' in data:
                    response_data['client_timestamp'] = data['timestamp']

                emit('pong', {
                    'status': 'success',
                    'data': response_data
                })
                emit('pong_from_server', {'data': response_data}) # Compatibility

            except Exception as e:
                self.logger.error(f"Ping/Pong error: {e}", exc_info=True, sid=session_id)

        @self.socketio.on('request_data')
        def handle_request_data(data):
            """Handle specific data requests with rate limiting."""
            session_id = request.sid
            try:
                # --- Rate Limit Check ---
                if not self._check_rate_limit(session_id):
                    self.logger.warning("Rate limit exceeded for request_data", sid=session_id)
                    emit('error', {'status': 'error', 'message': 'Rate limit exceeded'})
                    return
                # --- End Check ---

                data_type = data.get('type')
                filters = data.get('filters', {})

                if not data_type:
                     emit('error', {'status': 'error', 'message': 'Data type required'})
                     return

                # Queue the request for background processing
                self.message_queue.put({
                    'type': 'data_request',
                    'session_id': session_id,
                    'data_type': data_type,
                    'filters': filters,
                    'timestamp': time.time()
                })
                self.logger.debug("Queued data request", sid=session_id, data_type=data_type)

            except Exception as e:
                self.logger.error(f"Request data error: {e}", exc_info=True, sid=session_id)
                emit('error', {'status': 'error', 'message': 'Failed to process data request'})

        # Add a catch-all handler for unhandled events (optional)
        @self.socketio.on_error_default
        def default_error_handler(e):
             self.logger.error(f"Unhandled WebSocket error: {e}", sid=getattr(request, 'sid', 'N/A'))
             # Optionally emit a generic error back to the client
             if hasattr(request, 'sid'):
                 emit('error', {'status': 'error', 'message': 'An unexpected server error occurred.'}, room=request.sid)


    # --- Rate Limiting Method ---
    def _check_rate_limit(self, session_id):
        """Check if client is within rate limits."""
        if not session_id:
            self.logger.warning("Attempted rate limit check without session ID.")
            return False # Should not happen if called within event handler
        now = time.time()
        # Clean up old timestamps (more efficient than pop(0) repeatedly)
        self.rate_limits[session_id] = [t for t in self.rate_limits[session_id]
                                        if t >= now - self.rate_limit_window]
        # Check limit
        if len(self.rate_limits[session_id]) >= self.rate_limit_max:
            return False # Limit exceeded
        # Record current request time
        self.rate_limits[session_id].append(now)
        return True # Within limit

    # --- Room Management Methods ---
    def _join_room(self, session_id, room_name):
        """Add client to a room (both internal state and SocketIO)."""
        if session_id not in self.active_connections:
            self.logger.warning(f"Attempted to join room for inactive session: {session_id}")
            return
        try:
            # SocketIO join
            join_room(room_name, sid=session_id) # Explicitly pass sid
            # Internal state update
            self.room_subscriptions[room_name].add(session_id)
            self.active_connections[session_id]['rooms'].add(room_name)
            self.user_rooms[session_id] = self.active_connections[session_id]['rooms'] # Keep user_rooms in sync

            self.logger.info(f"Client joined room", sid=session_id, room=room_name)
        except Exception as e:
            # Handle cases where session might disconnect during join
            self.logger.error(f"Error joining room for client", sid=session_id, room=room_name, error=str(e))
            # Clean up internal state if join failed potentially
            self.room_subscriptions[room_name].discard(session_id)
            if session_id in self.active_connections:
                self.active_connections[session_id]['rooms'].discard(room_name)
            if session_id in self.user_rooms:
                 self.user_rooms[session_id].discard(room_name)


    def _leave_room(self, session_id, room_name):
        """Remove client from a room (both internal state and SocketIO)."""
        try:
            # SocketIO leave
            leave_room(room_name, sid=session_id) # Explicitly pass sid
        except Exception as e:
            # Log error but continue cleanup, session might already be gone
            self.logger.warning(f"SocketIO leave_room failed (session might be gone)", sid=session_id, room=room_name, error=str(e))

        # Internal state cleanup
        self._leave_room_internal(session_id, room_name)

    def _leave_room_internal(self, session_id, room_name):
        """Internal state cleanup for leaving a room."""
        self.room_subscriptions[room_name].discard(session_id)
        if session_id in self.user_rooms:
            self.user_rooms[session_id].discard(room_name)
        if session_id in self.active_connections:
             self.active_connections[session_id]['rooms'].discard(room_name)
        # Clean up empty room subscription sets
        if not self.room_subscriptions.get(room_name):
            self.room_subscriptions.pop(room_name, None)
        self.logger.info(f"Client left room (internal state)", sid=session_id, room=room_name)


    # --- Stream Management Methods (Unchanged logic, just added logging) ---
    def _subscribe_to_stream(self, session_id, stream_id):
        """Subscribe client to a data stream."""
        if session_id not in self.active_connections: return
        self.stream_subscribers[stream_id].add(session_id)
        stream_room = f'stream_{stream_id}'
        try:
            join_room(stream_room, sid=session_id)
            self.logger.info(f"Client subscribed to stream", sid=session_id, stream=stream_id)
        except Exception as e:
            self.logger.error(f"Error subscribing client to stream room", sid=session_id, stream=stream_id, error=str(e))
            self.stream_subscribers[stream_id].discard(session_id)


    def _unsubscribe_from_stream(self, session_id, stream_id):
        """Unsubscribe client from a data stream."""
        self.stream_subscribers[stream_id].discard(session_id)
        stream_room = f'stream_{stream_id}'
        try:
            leave_room(stream_room, sid=session_id)
        except Exception as e:
            self.logger.warning(f"SocketIO leave_room failed for stream (session might be gone)", sid=session_id, stream=stream_id, error=str(e))
        # Clean up empty stream subscription sets
        if not self.stream_subscribers.get(stream_id):
            self.stream_subscribers.pop(stream_id, None)
        self.logger.info(f"Client unsubscribed from stream", sid=session_id, stream=stream_id)

    # --- Background Processing (Unchanged logic, added logging) ---
    def _start_processing_thread(self):
        """Start background thread for message processing."""
        if self.processing_thread and self.processing_thread.is_alive():
            return
        self.processing_thread = threading.Thread(target=self._process_messages, daemon=True)
        self.processing_thread.start()
        self.logger.info("Message processing thread started")

    def _process_messages(self):
        """Process queued messages in background thread."""
        self.logger.info("Message processing loop started.")
        while not self.should_stop.is_set():
            try:
                message = self.message_queue.get(timeout=1)
                msg_type = message.get('type')
                self.logger.debug(f"Processing message", message_type=msg_type)
                if msg_type == 'data_request':
                    self._handle_data_request(message)
                elif msg_type == 'stream_data':
                    self._handle_stream_data(message)
                else:
                    handlers = self.event_handlers.get(msg_type, [])
                    for h in handlers:
                        try:
                            h(message)
                        except Exception as e:
                            self.logger.error(f"Custom message handler error", handler=h.__name__, error=str(e), exc_info=True)
                self.message_queue.task_done()
            except queue.Empty:
                continue # Normal timeout, continue loop
            except Exception as e:
                self.logger.error(f"Unexpected error in message processing loop", error=str(e), exc_info=True)
                # Avoid tight loop on persistent errors
                time.sleep(0.5)
        self.logger.info("Message processing loop stopped.")

    # --- Message Handlers (Data Request, Stream Data - unchanged logic) ---
    def _handle_data_request(self, message):
        # ... (implementation unchanged) ...
        # (Ensure proper error handling and logging inside)
         try:
            session_id = message.get('session_id')
            data_type = message.get('data_type')
            filters = message.get('filters', {})
            data = {}

            # --- Placeholder Data Retrieval ---
            # In a real app, this would query databases, call APIs, etc.
            # based on data_type and filters.
            if data_type == 'health_data':
                data = self._get_health_data(filters)
            elif data_type == 'device_status':
                data = self._get_device_status(filters)
            elif data_type == 'analytics':
                data = self._get_analytics_data(filters)
            else:
                data = {'error': 'Unknown data type requested', 'requested_type': data_type}
            # --- End Placeholder ---

            if self.socketio:
                try:
                    response_data = {
                        'status': 'success',
                        'data': {
                            'data_type': data_type,
                            'payload': data,
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                    if isinstance(data, dict) and 'error' in data:
                        response_data['status'] = 'error'
                        response_data['data']['message'] = data['error']

                    self.socketio.emit('data_response', response_data, room=session_id)
                    self.logger.debug("Sent data_response", sid=session_id, data_type=data_type)
                except Exception as e:
                    self.logger.error(f"Failed to emit data_response", sid=session_id, error=str(e))

         except Exception as e:
            self.logger.error(f"Data request handling error", error=str(e), exc_info=True)
            # Optionally emit an error back to the specific client
            if self.socketio and message.get('session_id'):
                try:
                    self.socketio.emit('error', {'status':'error', 'message': 'Failed to handle data request'}, room=message['session_id'])
                except Exception as emit_e:
                     self.logger.error(f"Failed to emit error response for data request", sid=message.get('session_id'), error=str(emit_e))


    def _handle_stream_data(self, message):
        # ... (implementation unchanged) ...
        # (Ensure proper error handling and logging inside)
        try:
            stream_id = message.get('stream_id')
            data = message.get('data')
            if stream_id is None:
                self.logger.warning("Stream data message missing stream_id", message_content=message)
                return

            timestamp_iso = datetime.now().isoformat()
            # Append to internal buffer
            self.data_streams[stream_id].append({
                'data': data,
                'timestamp': timestamp_iso
            })

            # Broadcast via SocketIO to the specific stream room
            if self.socketio:
                stream_room = f'stream_{stream_id}'
                payload = {
                    'status': 'success',
                    'data': {
                        'stream_id': stream_id,
                        'payload': data,
                        'timestamp': timestamp_iso,
                        'type': 'real_time'
                    }
                }
                try:
                    self.socketio.emit('stream_data', payload, room=stream_room)
                    self.logger.debug(f"Broadcasted stream data", stream=stream_id, room=stream_room)
                except Exception as e:
                    self.logger.error(f"Failed to emit stream_data", stream=stream_id, room=stream_room, error=str(e))

        except Exception as e:
            self.logger.error(f"Stream data handling error", error=str(e), exc_info=True)


    # --- Placeholder Data Getters (Unchanged) ---
    def _get_health_data(self, filters): return {'status': 'simulated', 'value': random.random()}
    def _get_device_status(self, filters): return {'device_count': len(self.active_connections), 'status': 'simulated'}
    def _get_analytics_data(self, filters): return {'analysis': 'simulated', 'trend': random.choice(['up', 'down', 'stable'])}

    # --- Public API Methods (broadcast, send, etc. - unchanged logic) ---
    # ... (Keep existing methods like broadcast_to_room, send_to_client, etc.) ...
    # (Ensure they use self.socketio correctly)

    def broadcast_to_room(self, room_name, event, data):
        """Broadcast message to all clients in a room."""
        if not self.socketio: return
        try:
            self.socketio.emit(event, data, room=room_name)
            self.logger.debug(f"Broadcasted event to room", event=event, room=room_name)
        except Exception as e:
            self.logger.error(f"Broadcast error", event=event, room=room_name, error=str(e), exc_info=True)

    def send_to_client(self, session_id, event, data):
        """Send message to a specific client by session id."""
        if not self.socketio: return
        try:
            self.socketio.emit(event, data, room=session_id)
            self.logger.debug(f"Sent event to client", event=event, sid=session_id)
        except Exception as e:
            self.logger.error(f"Send to client error", event=event, sid=session_id, error=str(e), exc_info=True)

    def broadcast_stream_data(self, stream_id, data):
        """Enqueue stream data for broadcasting."""
        try:
            self.message_queue.put({
                'type': 'stream_data',
                'stream_id': stream_id,
                'data': data,
                'timestamp': time.time()
            })
            self.logger.debug("Queued stream data", stream_id=stream_id)
        except Exception as e:
            self.logger.error(f"Failed to queue stream data", stream_id=stream_id, error=str(e))

    # --- Cleanup, Stats, Shutdown (Unchanged logic, added logging) ---
    def cleanup_inactive_connections(self, timeout_minutes=None):
        """Remove inactive connections based on last_activity."""
        if timeout_minutes is None:
            timeout_minutes = int(os.environ.get('CLIENT_TIMEOUT_MINUTES', 5))
        timeout_seconds = timeout_minutes * 60
        now = time.time()
        inactive_sessions = []
        cleaned_count = 0

        # Iterate over a copy of the keys to allow modification
        for session_id, info in list(self.active_connections.items()):
            last_activity = info.get('last_activity', 0)
            if (now - last_activity) > timeout_seconds:
                inactive_sessions.append(session_id)

        self.logger.info(f"Starting cleanup of inactive connections", timeout_minutes=timeout_minutes, found_inactive=len(inactive_sessions))

        for session_id in inactive_sessions:
            try:
                self.logger.info(f"Cleaning up inactive connection", sid=session_id)
                # Disconnect client via SocketIO FIRST
                if self.socketio:
                    self.socketio.disconnect(session_id, silent=True) # silent=True suppresses errors if already gone
                # Then cleanup internal state (handle_disconnect might do this too, ensure it's idempotent)
                if session_id in self.active_connections:
                     client_rooms = self.active_connections[session_id].get('rooms', set()).copy()
                     user_id = self.active_connections[session_id].get('user_id', 'unknown')
                     del self.active_connections[session_id]
                     self.user_rooms.pop(session_id, None)
                     for room in client_rooms:
                         self._leave_room_internal(session_id, room)
                     cleaned_count += 1
                     self.logger.info(f"Cleaned up session", sid=session_id, user_id=user_id)
            except Exception as e:
                self.logger.error(f"Error during cleanup for session", sid=session_id, error=str(e), exc_info=True)

        self.logger.info(f"Inactive connection cleanup finished", cleaned_count=cleaned_count)
        return cleaned_count

    def get_statistics(self):
        # ... (implementation unchanged) ...
        try:
            active_conns = len(self.active_connections)
            total_rooms = len(self.room_subscriptions)
            active_streams = len(self.stream_subscribers)
            q_size = self.message_queue.qsize()

            now = time.time()
            rate_limited_count = 0
            # Clean up old rate limit entries while counting
            for sid in list(self.rate_limits.keys()): # Iterate keys copy
                 self.rate_limits[sid] = [t for t in self.rate_limits[sid] if t >= now - self.rate_limit_window]
                 if len(self.rate_limits[sid]) >= self.rate_limit_max:
                      rate_limited_count += 1
                 elif not self.rate_limits[sid]: # Remove empty entries
                      del self.rate_limits[sid]


            uptime = time.time() - getattr(self, '_start_time', time.time())

            return {
                'active_connections': active_conns,
                'total_rooms': total_rooms,
                'active_streams': active_streams,
                'queued_messages': q_size,
                'rate_limited_clients_current': rate_limited_count,
                'uptime_seconds': round(uptime)
            }
        except Exception as e:
            self.logger.error(f"Statistics retrieval error: {e}", exc_info=True)
            return {}


    def shutdown(self):
        # ... (implementation unchanged, ensure logging) ...
        self.logger.info("Shutting down WebSocket manager...")
        self.should_stop.set()
        if self.processing_thread and self.processing_thread.is_alive():
            self.logger.info("Waiting for processing thread to finish...")
            self.processing_thread.join(timeout=5)
            if self.processing_thread.is_alive():
                self.logger.warning("Processing thread did not terminate cleanly.")

        if self.socketio:
            try:
                # Optionally notify connected clients before shutdown
                self.logger.info("Emitting server shutdown notice to clients...")
                self.socketio.emit('server_shutdown', {
                    'message': 'Server is shutting down.',
                    'timestamp': datetime.now().isoformat()
                })
                # Give a moment for message to send (though not guaranteed)
                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Failed to emit shutdown message", error=str(e))

        # Disconnect MQTT if managed here (though likely managed by generator)
        # self._disconnect_mqtt()

        # Clear internal state
        self.active_connections.clear()
        self.room_subscriptions.clear()
        self.user_rooms.clear()
        self.stream_subscribers.clear()
        self.data_streams.clear()
        self.rate_limits.clear()

        self.logger.info("WebSocket manager shutdown complete.")


    # --- Helper Methods (unchanged) ---
    def _get_client_ip(self):
        # ... (implementation unchanged) ...
        x_forwarded = request.headers.get('X-Forwarded-For', '')
        if x_forwarded:
            # Take the first IP if there's a list
            return x_forwarded.split(',')[0].strip()
        # Fallback using environment variables often set by proxies like Nginx
        return request.environ.get('HTTP_X_REAL_IP', request.remote_addr)

    def _get_user_agent(self):
        # ... (implementation unchanged) ...
        return request.headers.get('User-Agent', 'Unknown')

    # --- Custom Event Handling (unchanged) ---
    def add_event_handler(self, event_name: str, handler: Callable):
        # ... (implementation unchanged) ...
        # (Ensure logging inside)
        try:
            self.event_handlers[event_name].append(handler)
            dispatcher_attr = f'_dispatcher_registered_{event_name}'

            # Register with SocketIO only if initialized and not already done
            if self.socketio and not getattr(self.socketio, dispatcher_attr, False):

                def create_dispatcher(event):
                    @wraps(handler) # Preserve metadata of original handler if possible
                    def dispatcher(*args, **kwargs):
                        sid = getattr(request, 'sid', None)
                        if not sid:
                             self.logger.warning(f"Received custom event '{event}' without session ID.")
                             return # Cannot proceed without SID in most cases

                        # Apply rate limiting to custom events too
                        if not self._check_rate_limit(sid):
                            self.logger.warning(f"Rate limit exceeded for custom event '{event}'", sid=sid)
                            emit('error', {'status': 'error', 'message': 'Rate limit exceeded'}, room=sid)
                            return

                        self.logger.debug(f"Dispatching custom event", event=event, sid=sid)
                        try:
                            # Pass SID as the first argument convention
                            for h in self.event_handlers[event]:
                                h(sid, *args, **kwargs)
                        except Exception as e:
                            self.logger.error(f"Custom handler execution error", event=event, handler=h.__name__, error=str(e), exc_info=True)
                            # Optionally emit an error back to the client
                            emit('error', {'status': 'error', 'message': f'Error processing event {event}.'}, room=sid)

                    return dispatcher

                # Register the dispatcher function with SocketIO for the event
                self.socketio.on(event_name)(create_dispatcher(event_name))
                setattr(self.socketio, dispatcher_attr, True) # Mark as registered
                self.logger.info(f"Registered SocketIO dispatcher", event=event_name)

            self.logger.info(f"Appended custom handler", event=event_name, handler=handler.__name__)

        except Exception as e:
            self.logger.error(f"Failed to add custom event handler", event=event_name, error=str(e), exc_info=True)


# --- Example Usage (If run directly) ---
# Note: This manager is usually initialized and run via the main Flask app
if __name__ == "__main__":
    print("This script defines the RealtimeWebSocketManager.")
    print("It should be imported and initialized within your main Flask application (e.g., enhanced_flask_app_v2.py).")
    # Example minimal Flask app setup for demonstration
    from flask import Flask
    logging.basicConfig(level=logging.INFO) # Basic logging for demo

    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'test-secret!' # Needed for SocketIO
    app.config['JWT_SECRET_KEY'] = 'test-jwt-secret!' # Needed for auth
    app.config['WEBSOCKET_RATE_LIMIT_MAX'] = 5 # Lower limit for demo
    app.config['WEBSOCKET_RATE_LIMIT_WINDOW'] = 10

    print("Creating WebSocket Manager instance...")
    ws_manager = RealtimeWebSocketManager()
    ws_manager.initialize_socketio(app)
    print("WebSocket Manager initialized.")

    @app.route('/')
    def index():
        return "WebSocket Manager Demo Server Running"

    # Example: Broadcast time every 5 seconds
    def broadcast_time():
        while True:
            ws_manager.socketio.sleep(5) # Use socketio sleep for async compatibility
            now = datetime.now().isoformat()
            print(f"Broadcasting time: {now}")
            # Ensure broadcast happens within app context if needed by extensions
            with app.app_context():
                ws_manager.broadcast_to_room('time_updates', 'time_update', {'data': {'current_time': now}})

    ws_manager.socketio.start_background_task(broadcast_time)
    print("Background time broadcast task started.")

    print("Starting Flask-SocketIO server on http://127.0.0.1:5000...")
    print("Connect with a Socket.IO client (passing a valid JWT in auth) and join 'time_updates' room.")
    ws_manager.socketio.run(app, host='127.0.0.1', port=5000, debug=True)