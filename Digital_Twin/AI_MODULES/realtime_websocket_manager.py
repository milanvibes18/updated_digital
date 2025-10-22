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
        self.rate_limit_max = 100   # requests per window
        
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
            # Ensure LOGS directory exists or handle appropriately
            try:
                # Try to use the main app's log directory if configured, else default
                log_dir = current_app.config.get('LOG_DIR', 'LOGS')
                os.makedirs(log_dir, exist_ok=True)
                handler = logging.FileHandler(os.path.join(log_dir, 'digital_twin_websocket.log'))
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            except (AttributeError, FileNotFoundError, ImportError):
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
                async_mode='eventlet',
                message_queue=message_queue_url,
                logger=app.config.get('DEBUG', False),
                engineio_logger=app.config.get('DEBUG', False),
                ping_timeout=60,
                ping_interval=25,
                channel=os.environ.get('SOCKETIO_CHANNEL', 'digital_twin')
            )
            
            self._register_event_handlers()
            self._start_processing_thread()
            
            self.logger.info("SocketIO initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SocketIO: {e}")
            raise
    
    def _register_event_handlers(self):
        """Register default SocketIO event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect(auth: Optional[Dict] = None):
            """
            Handle client connection and JWT authentication.
            Client must send {'token': '...'} in the auth payload or query string.
            """
            try:
                token = None
                if auth and 'token' in auth:
                    token = auth.get('token')
                elif request.args.get('token'):
                    token = request.args.get('token')
                
                if not token:
                    auth_header = request.headers.get('Authorization')
                    if auth_header and auth_header.startswith('Bearer '):
                        token = auth_header.split(' ')[1]

                if not token:
                    self.logger.warning("Connection attempt without token.", sid=request.sid)
                    emit('auth_failed', {'status': 'error', 'message': 'Authentication token required.'})
                    return False  # Reject connection

                try:
                    # Decode JWT. Assumes secret key is in Flask app config
                    # --- FIX: Use JWT_SECRET_KEY, not SECRET_KEY ---
                    secret = current_app.config['JWT_SECRET_KEY']
                    payload = jwt.decode(token, secret, algorithms=["HS256"])
                    user_id = payload.get('sub') # 'sub' is standard for user ID
                    
                    # --- FIX: Get role from payload ---
                    user_role = payload.get('role', 'user') 
                    
                    if not user_id:
                        self.logger.warning("Invalid token payload (missing 'sub').", sid=request.sid)
                        emit('auth_failed', {'status': 'error', 'message': 'Invalid token payload.'})
                        return False
                        
                except jwt.ExpiredSignatureError:
                    self.logger.warning("Connection attempt with expired token.", sid=request.sid)
                    emit('auth_failed', {'status': 'error', 'message': 'Token has expired.'})
                    return False
                except jwt.InvalidTokenError as e:
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
                    'user_id': user_id, # <-- Store authenticated user
                    'role': user_role, # <-- FIX: Store user role
                    'connected_at': datetime.now().isoformat(),
                    'ip_address': self._get_client_ip(),
                    'user_agent': self._get_user_agent(),
                    'rooms': set(),
                    'last_activity': time.time()
                }
                
                self.active_connections[session_id] = client_info
                
                # Send welcome message
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
                
                # Try to send initial data
                try:
                    cache = getattr(current_app.extensions.get("digital_twin_instance"), 'cache', None)
                    if cache:
                        initial_data = cache.get('dashboard_combined_data')
                        if initial_data:
                            emit('dashboard_update', {'data': initial_data})
                except Exception as e:
                    self.logger.warning(f"Could not send initial dashboard data: {e}")
                    
                return True
                
            except Exception as e:
                self.logger.error(f"Connection error: {e}", exc_info=True)
                emit('error', {'status': 'error', 'message': 'An internal server error occurred during connection.'})
                return False
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            session_id = request.sid # <-- Get sid directly
            try:
                if session_id in self.active_connections:
                    # Leave all rooms
                    client_rooms = self.active_connections[session_id].get('rooms', set())
                    for room in client_rooms.copy():
                        self._leave_room(session_id, room, is_disconnecting=True)
                    
                    # Remove from active connections
                    user_id = self.active_connections[session_id].get('user_id', 'unknown')
                    del self.active_connections[session_id]
                    
                    self.logger.info(f"Client disconnected: {session_id} (User: {user_id})")
                else:
                    self.logger.info(f"Client disconnected (untracked): {session_id}")
            except Exception as e:
                self.logger.error(f"Disconnection error: {e}")
        
        @self.socketio.on('join_room') # Renamed from 'subscribe' in main app
        @self.socketio.on('subscribe') # Add alias for compatibility
        def handle_join_room(data):
            """Handle room join requests."""
            session_id = request.sid
            try:
                # Support both 'room' and 'room_name' keys
                room_name = data.get('room') or data.get('room_name')
                
                if not room_name:
                    emit('error', {'status': 'error', 'message': 'Room name required'})
                    return
                
                if not self._check_rate_limit(session_id):
                    emit('error', {'status': 'error', 'message': 'Rate limit exceeded'})
                    return
                
                self._join_room(session_id, room_name)
                
                emit('room_joined', { # Main app used 'subscribed'
                    'status': 'success',
                    'data': {
                        'room': room_name,
                        'members_count': len(self.room_subscriptions.get(room_name, set()))
                    }
                })
                emit('subscribed', {'data': {'room': room_name}}) # For compatibility
                
            except Exception as e:
                self.logger.error(f"Join room error: {e}")
                emit('error', {'status': 'error', 'message': 'Failed to join room'})
        
        @self.socketio.on('leave_room') # Renamed from 'unsubscribe' in main app
        @self.socketio.on('unsubscribe') # Add alias for compatibility
        def handle_leave_room(data):
            """Handle room leave requests."""
            session_id = request.sid
            try:
                room_name = data.get('room') or data.get('room_name')
                
                if not room_name:
                    emit('error', {'status': 'error', 'message': 'Room name required'})
                    return
                
                self._leave_room(session_id, room_name)
                
                emit('room_left', { # Main app used 'unsubscribed'
                    'status': 'success',
                    'data': {'room': room_name}
                })
                emit('unsubscribed', {'data': {'room': room_name}}) # For compatibility
                
            except Exception as e:
                self.logger.error(f"Leave room error: {e}")
                emit('error', {'status': 'error', 'message': 'Failed to leave room'})
        
        @self.socketio.on('subscribe_stream')
        def handle_subscribe_stream(data):
            """Handle data stream subscription."""
            session_id = request.sid
            try:
                stream_id = data.get('stream_id')
                
                if not stream_id:
                    emit('error', {'status': 'error', 'message': 'Stream ID required'})
                    return
                
                if not self._check_rate_limit(session_id):
                    emit('error', {'status': 'error', 'message': 'Rate limit exceeded'})
                    return
                
                self._subscribe_to_stream(session_id, stream_id)
                
                recent_data = list(self.data_streams[stream_id])[-10:]
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
                self.logger.error(f"Subscribe stream error: {e}")
                emit('error', {'status': 'error', 'message': 'Failed to subscribe to stream'})
        
        @self.socketio.on('unsubscribe_stream')
        def handle_unsubscribe_stream(data):
            """Handle data stream unsubscription."""
            session_id = request.sid
            try:
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
                self.logger.error(f"Unsubscribe stream error: {e}")
                emit('error', {'status': 'error', 'message': 'Failed to unsubscribe from stream'})
        
        @self.socketio.on('ping')
        @self.socketio.on('ping_from_client') # Alias for compatibility
        def handle_ping():
            """Handle ping requests for connection health."""
            session_id = request.sid
            try:
                if session_id in self.active_connections:
                    self.active_connections[session_id]['last_activity'] = time.time()
                
                emit('pong', {
                    'status': 'success',
                    'data': {'timestamp': datetime.now().isoformat()}
                })
                emit('pong_from_server', {'data': {'timestamp': datetime.now().isoformat()}}) # For compatibility
                
            except Exception as e:
                self.logger.error(f"Ping error: {e}")
        
        @self.socketio.on('request_data')
        def handle_request_data(data):
            """Handle specific data requests."""
            session_id = request.sid
            try:
                data_type = data.get('type')
                filters = data.get('filters', {})
                
                if not self._check_rate_limit(session_id):
                    emit('error', {'status': 'error', 'message': 'Rate limit exceeded'})
                    return
                
                self.message_queue.put({
                    'type': 'data_request',
                    'session_id': session_id,
                    'data_type': data_type,
                    'filters': filters,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                self.logger.error(f"Request data error: {e}")
                emit('error', {'status': 'error', 'message': 'Failed to process data request'})

    # -------------------------------
    # Helpers, processing & API
    # -------------------------------

    def _get_client_ip(self):
        """Get client IP address."""
        x_forwarded = request.headers.get('X-Forwarded-For', '')
        if x_forwarded:
            return x_forwarded.split(',')[0].strip()
        return request.environ.get('HTTP_X_REAL_IP', request.remote_addr)

    def _get_user_agent(self):
        """Get client user agent."""
        return request.headers.get('User-Agent', 'Unknown')

    def _check_rate_limit(self, session_id):
        """Check if client is within rate limits."""
        if not session_id:
            return False
        now = time.time()
        client_requests = self.rate_limits[session_id]
        while client_requests and client_requests[0] < now - self.rate_limit_window:
            client_requests.pop(0)
        if len(client_requests) >= self.rate_limit_max:
            return False
        client_requests.append(now)
        return True

    def _join_room(self, session_id, room_name):
        """Add client to a room (and make the Socket.IO 'join_room' call)."""
        try:
            join_room(room_name)
        except Exception:
            self.logger.warning(f"join_room({room_name}) failed for {session_id} (no request context)")
            pass

        self.room_subscriptions[room_name].add(session_id)
        if session_id in self.active_connections:
            self.user_rooms[session_id] = self.active_connections[session_id]['rooms']
        self.user_rooms[session_id].add(room_name)
        
        self.logger.info(f"Client {session_id} joined room {room_name}")

    def _leave_room(self, session_id, room_name, is_disconnecting=False):
        """Remove client from a room (and perform Socket.IO leave_room)."""
        if not is_disconnecting:
            try:
                leave_room(room_name)
            except Exception:
                self.logger.warning(f"leave_room({room_name}) failed for {session_id} (no request context)")
                pass

        self.room_subscriptions[room_name].discard(session_id)
        if session_id in self.user_rooms:
            self.user_rooms[session_id].discard(room_name)

        if session_id in self.active_connections:
             self.active_connections[session_id]['rooms'].discard(room_name)

        if not self.room_subscriptions.get(room_name):
            self.room_subscriptions.pop(room_name, None)

        self.logger.info(f"Client {session_id} left room {room_name}")

    def _subscribe_to_stream(self, session_id, stream_id):
        """Subscribe client to a data stream and create a 'stream room'."""
        self.stream_subscribers[stream_id].add(session_id)
        stream_room = f'stream_{stream_id}'
        try:
            join_room(stream_room)
        except Exception:
            pass
        self.logger.info(f"Client {session_id} subscribed to stream {stream_id}")

    def _unsubscribe_from_stream(self, session_id, stream_id):
        """Unsubscribe client from a data stream."""
        self.stream_subscribers[stream_id].discard(session_id)
        stream_room = f'stream_{stream_id}'
        try:
            leave_room(stream_room)
        except Exception:
            pass

        if not self.stream_subscribers.get(stream_id):
            self.stream_subscribers.pop(stream_id, None)

        self.logger.info(f"Client {session_id} unsubscribed from stream {stream_id}")

    def _start_processing_thread(self):
        """Start background thread for message processing if not already running."""
        if self.processing_thread and self.processing_thread.is_alive():
            return

        self._start_time = time.time()
        self.processing_thread = threading.Thread(target=self._process_messages, daemon=True)
        self.processing_thread.start()
        self.logger.info("Message processing thread started")

    def _process_messages(self):
        """Process queued messages in background thread."""
        while not self.should_stop.is_set():
            try:
                message = self.message_queue.get(timeout=1)
                msg_type = message.get('type')
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
                            self.logger.error(f"Custom message handler error: {e}")

                self.message_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Message processing error: {e}")

    def _handle_data_request(self, message):
        """Handle data request messages (called from background thread)."""
        try:
            session_id = message.get('session_id')
            data_type = message.get('data_type')
            filters = message.get('filters', {})
            data = {}

            if data_type == 'health_data':
                data = self._get_health_data(filters)
            elif data_type == 'device_status':
                data = self._get_device_status(filters)
            elif data_type == 'analytics':
                data = self._get_analytics_data(filters)
            else:
                data = {'error': 'Unknown data type', 'requested_type': data_type}

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
                except Exception as e:
                    self.logger.error(f"Emit data_response failed: {e}")

        except Exception as e:
            self.logger.error(f"Data request handling error: {e}")

    def _handle_stream_data(self, message):
        """Handle stream data and broadcast to subscribers."""
        try:
            stream_id = message.get('stream_id')
            data = message.get('data')
            if stream_id is None:
                self.logger.warning("Stream data message missing stream_id")
                return

            self.data_streams[stream_id].append({
                'data': data,
                'timestamp': datetime.now().isoformat()
            })

            if self.socketio:
                try:
                    self.socketio.emit('stream_data', {
                        'status': 'success',
                        'data': {
                            'stream_id': stream_id,
                            'payload': data,
                            'timestamp': datetime.now().isoformat(),
                            'type': 'real_time'
                        }
                    }, room=f'stream_{stream_id}')
                except Exception as e:
                    self.logger.error(f"Emit stream_data failed: {e}")

        except Exception as e:
            self.logger.error(f"Stream data handling error: {e}")

    # Placeholder retrieval methods
    def _get_health_data(self, filters):
        return {
            'error': 'Health data retrieval not implemented',
            'filters': filters
        }

    def _get_device_status(self, filters):
        return {
            'devices': [
                { 'id': 'sensor_001', 'status': 'online', 'last_seen': datetime.now().isoformat() }
            ]
        }

    def _get_analytics_data(self, filters):
        return {
            'error': 'Analytics data retrieval not implemented',
            'filters': filters
        }

    # -------------------------------
    # Public API
    # -------------------------------

    def broadcast_to_room(self, room_name, event, data):
        """
        Broadcast message to all clients in a room.
        Note: Assumes 'data' is already formatted with the correct schema.
        """
        try:
            if self.socketio:
                self.socketio.emit(event, data, room=room_name)
                self.logger.debug(f"Broadcasted {event} to room {room_name}")
        except Exception as e:
            self.logger.error(f"Broadcast error: {e}")

    def send_to_client(self, session_id, event, data):
        """
        Send message to a specific client by session id.
        Note: Assumes 'data' is already formatted with the correct schema.
        """
        try:
            if self.socketio:
                self.socketio.emit(event, data, room=session_id)
                self.logger.debug(f"Sent {event} to client {session_id}")
        except Exception as e:
            self.logger.error(f"Send to client error: {e}")

    def broadcast_stream_data(self, stream_id, data):
        """Enqueue stream data for broadcasting to subscribers."""
        try:
            self.message_queue.put({
                'type': 'stream_data',
                'stream_id': stream_id,
                'data': data,
                'timestamp': time.time()
            })
        except Exception as e:
            self.logger.error(f"Stream broadcast error: {e}")

    def create_room(self, room_name, description=None):
        """Create a new room (server-side) and store metadata in Redis."""
        try:
            if room_name not in self.room_subscriptions:
                self.room_subscriptions[room_name] = set()

                if self.redis_client:
                    room_info = {
                        'name': room_name,
                        'description': description,
                        'created_at': datetime.now().isoformat()
                    }
                    self.redis_client.hset('rooms', room_name, json.dumps(room_info))

                self.logger.info(f"Room created: {room_name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Room creation error: {e}")
            return False

    def delete_room(self, room_name):
        """Delete a room and remove all members (server-side)."""
        try:
            if room_name in self.room_subscriptions:
                clients = self.room_subscriptions[room_name].copy()
                for client_id in clients:
                    self._leave_room(client_id, room_name)

                if self.redis_client:
                    self.redis_client.hdel('rooms', room_name)

                self.logger.info(f"Room deleted: {room_name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Room deletion error: {e}")
            return False

    def get_room_info(self, room_name):
        """Get information about a room."""
        try:
            if room_name in self.room_subscriptions:
                return {
                    'name': room_name,
                    'members_count': len(self.room_subscriptions[room_name]),
                    'members': list(self.room_subscriptions[room_name])
                }
            return None
        except Exception as e:
            self.logger.error(f"Get room info error: {e}")
            return None

    def get_active_connections(self):
        """Return active connections summary."""
        try:
            return {
                'total': len(self.active_connections),
                'connections': [
                    {
                        'session_id': sid,
                        'user_id': info.get('user_id'),
                        'role': info.get('role'),
                        'connected_at': info.get('connected_at'),
                        'rooms': list(info.get('rooms', [])),
                        'ip_address': info.get('ip_address', 'unknown'),
                        'last_activity': info.get('last_activity')
                    }
                    for sid, info in self.active_connections.items()
                ]
            }
        except Exception as e:
            self.logger.error(f"Get connections error: {e}")
            return {'total': 0, 'connections': []}

    def cleanup_inactive_connections(self, timeout=300):
        """Remove inactive connections (server-side cleanup)."""
        try:
            now = time.time()
            inactive_sessions = []
            
            # Check for timeout (default 5 minutes)
            timeout = int(os.environ.get('CLIENT_TIMEOUT_MINUTES', 5)) * 60

            for session_id, info in list(self.active_connections.items()):
                if now - info.get('last_activity', 0) > timeout:
                    inactive_sessions.append(session_id)

            for session_id in inactive_sessions:
                client_rooms = self.active_connections[session_id].get('rooms', set()).copy()
                for room in client_rooms:
                    self._leave_room(session_id, room, is_disconnecting=True)
                
                if self.socketio:
                    try:
                        self.socketio.disconnect(session_id, silent=True)
                    except Exception as e:
                        self.logger.error(f"Error during socketio disconnect on cleanup: {e}", sid=session_id)

                if session_id in self.active_connections:
                    del self.active_connections[session_id]
                
                self.logger.info(f"Cleaned up inactive connection: {session_id}")

            return len(inactive_sessions)

        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
            return 0

    def add_event_handler(self, event_name: str, handler: Callable):
        """
        Add a custom event handler.
        """
        try:
            self.event_handlers[event_name].append(handler)
            dispatcher_attr = f'_dispatcher_registered_{event_name}'
            
            if self.socketio and not getattr(self.socketio, dispatcher_attr, False):
                
                def create_dispatcher(event):
                    def dispatcher(*args, **kwargs):
                        self.logger.debug(f"Dispatching custom event '{event}'")
                        try:
                            sid = request.sid
                            for h in self.event_handlers[event]:
                                h(sid, *args, **kwargs)
                        except TypeError:
                            try:
                                for h in self.event_handlers[event]:
                                    h(*args, **kwargs)
                            except Exception as e:
                                self.logger.error(f"Custom handler error for {event}: {e}")
                        except Exception as e:
                            self.logger.error(f"Custom handler error for {event}: {e}")
                    return dispatcher

                self.socketio.on(event_name, create_dispatcher(event_name))
                setattr(self.socketio, dispatcher_attr, True)
                self.logger.info(f"Registered SocketIO dispatcher for: {event_name}")
            
            self.logger.info(f"Appended handler for event: {event_name}")

        except Exception as e:
            self.logger.error(f"Add handler error: {e}")

    def get_statistics(self):
        """Return manager statistics."""
        try:
            return {
                'active_connections': len(self.active_connections),
                'total_rooms': len(self.room_subscriptions),
                'active_streams': len(self.stream_subscribers),
                'queued_messages': self.message_queue.qsize(),
                'rate_limited_clients': len([
                    sid for sid, requests in self.rate_limits.items()
                    if len(requests) >= self.rate_limit_max
                ]),
                'uptime_seconds': time.time() - getattr(self, '_start_time', time.time())
            }
        except Exception as e:
            self.logger.error(f"Statistics error: {e}")
            return {}

    def shutdown(self):
        """Gracefully shutdown the WebSocket manager."""
        try:
            self.logger.info("Shutting down WebSocket manager...")

            self.should_stop.set()
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5)

            if self.socketio:
                try:
                    self.socketio.emit('server_shutdown', {
                        'status': 'system',
                        'data': {
                            'message': 'Server is shutting down',
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                except Exception as e:
                    self.logger.error(f"Failed to emit shutdown: {e}")

            self.active_connections.clear()
            self.room_subscriptions.clear()
            self.user_rooms.clear()
            self.stream_subscribers.clear()
            self.data_streams.clear()

            self.logger.info("WebSocket manager shutdown complete")

        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")