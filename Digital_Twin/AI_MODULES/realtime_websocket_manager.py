import asyncio
import json
import logging
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Set, Callable, Any, Optional
import socketio
from flask_socketio import SocketIO, emit, join_room, leave_room, rooms
import threading
import queue
import weakref
from collections import defaultdict, deque
import redis
import pickle


class RealtimeWebSocketManager:
    """
    Advanced WebSocket manager for real-time data streaming, room management,
    and scalable communication in the Digital Twin system.
    """
    
    def __init__(self, app=None, redis_url="redis://localhost:6379/0", max_connections=1000):
        self.app = app
        self.socketio = None
        self.redis_client = None
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
        self._initialize_redis(redis_url)
        if app:
            self.initialize_socketio(app)
    
    def _setup_logging(self):
        """Setup logging for WebSocket manager."""
        logger = logging.getLogger('RealtimeWebSocketManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler('LOGS/digital_twin_websocket.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_redis(self, redis_url):
        """Initialize Redis connection for scalability."""
        try:
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            self.logger.info("Redis connection established")
        except Exception as e:
            self.logger.warning(f"Redis connection failed: {e}. Running without Redis.")
            self.redis_client = None
    def initialize_socketio(self, app):
        """Initialize SocketIO with the Flask app."""
        try:
            self.app = app
            
            # Configure SocketIO with Redis if available
            if self.redis_client:
                self.socketio = SocketIO(
                    app,
                    cors_allowed_origins="*",
                    async_mode='eventlet',
                    message_queue=self.redis_client.connection_pool.connection_kwargs['host'],
                    logger=True,
                    engineio_logger=True
                )
            else:
                self.socketio = SocketIO(
                    app,
                    cors_allowed_origins="*",
                    async_mode='eventlet',
                    logger=True,
                    engineio_logger=True
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
        def handle_connect():
            """Handle client connection."""
            try:
                session_id = self._generate_session_id()
                client_info = {
                    'session_id': session_id,
                    'connected_at': datetime.now().isoformat(),
                    'ip_address': self._get_client_ip(),
                    'user_agent': self._get_user_agent(),
                    'rooms': set(),
                    'last_activity': time.time()
                }
                
                self.active_connections[session_id] = client_info
                
                # Check connection limits
                if len(self.active_connections) > self.max_connections:
                    self.logger.warning("Maximum connections reached")
                    return False
                
                # Send welcome message
                emit('connection_established', {
                    'session_id': session_id,
                    'server_time': datetime.now().isoformat(),
                    'status': 'connected'
                })
                
                self.logger.info(f"Client connected: {session_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"Connection error: {e}")
                return False
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            try:
                session_id = self._get_session_id()
                if session_id in self.active_connections:
                    # Leave all rooms
                    client_rooms = self.active_connections[session_id].get('rooms', set())
                    for room in client_rooms.copy():
                        self._leave_room(session_id, room)
                    
                    # Remove from active connections
                    del self.active_connections[session_id]
                    
                    self.logger.info(f"Client disconnected: {session_id}")
                
            except Exception as e:
                self.logger.error(f"Disconnection error: {e}")
        
        @self.socketio.on('join_room')
        def handle_join_room(data):
            """Handle room join requests."""
            try:
                session_id = self._get_session_id()
                room_name = data.get('room')
                
                if not room_name:
                    emit('error', {'message': 'Room name required'})
                    return
                
                if not self._check_rate_limit(session_id):
                    emit('error', {'message': 'Rate limit exceeded'})
                    return
                
                self._join_room(session_id, room_name)
                
                emit('room_joined', {
                    'room': room_name,
                    'members_count': len(self.room_subscriptions[room_name])
                })
                
            except Exception as e:
                self.logger.error(f"Join room error: {e}")
                emit('error', {'message': 'Failed to join room'})
        
        @self.socketio.on('leave_room')
        def handle_leave_room(data):
            """Handle room leave requests."""
            try:
                session_id = self._get_session_id()
                room_name = data.get('room')
                
                if not room_name:
                    emit('error', {'message': 'Room name required'})
                    return
                
                self._leave_room(session_id, room_name)
                emit('room_left', {'room': room_name})
                
            except Exception as e:
                self.logger.error(f"Leave room error: {e}")
                emit('error', {'message': 'Failed to leave room'})
        
        @self.socketio.on('subscribe_stream')
        def handle_subscribe_stream(data):
            """Handle data stream subscription."""
            try:
                session_id = self._get_session_id()
                stream_id = data.get('stream_id')
                
                if not stream_id:
                    emit('error', {'message': 'Stream ID required'})
                    return
                
                if not self._check_rate_limit(session_id):
                    emit('error', {'message': 'Rate limit exceeded'})
                    return
                
                self._subscribe_to_stream(session_id, stream_id)
                
                # Send recent data if available
                recent_data = list(self.data_streams[stream_id])[-10:]
                if recent_data:
                    emit('stream_data', {
                        'stream_id': stream_id,
                        'data': recent_data,
                        'type': 'historical'
                    })
                
                emit('stream_subscribed', {'stream_id': stream_id})
                
            except Exception as e:
                self.logger.error(f"Subscribe stream error: {e}")
                emit('error', {'message': 'Failed to subscribe to stream'})
        
        @self.socketio.on('unsubscribe_stream')
        def handle_unsubscribe_stream(data):
            """Handle data stream unsubscription."""
            try:
                session_id = self._get_session_id()
                stream_id = data.get('stream_id')
                
                if not stream_id:
                    emit('error', {'message': 'Stream ID required'})
                    return
                
                self._unsubscribe_from_stream(session_id, stream_id)
                emit('stream_unsubscribed', {'stream_id': stream_id})
                
            except Exception as e:
                self.logger.error(f"Unsubscribe stream error: {e}")
                emit('error', {'message': 'Failed to unsubscribe from stream'})
        
        @self.socketio.on('ping')
        def handle_ping():
            """Handle ping requests for connection health."""
            try:
                session_id = self._get_session_id()
                if session_id in self.active_connections:
                    self.active_connections[session_id]['last_activity'] = time.time()
                
                emit('pong', {'timestamp': datetime.now().isoformat()})
                
            except Exception as e:
                self.logger.error(f"Ping error: {e}")
        
        @self.socketio.on('request_data')
        def handle_request_data(data):
            """Handle specific data requests."""
            try:
                session_id = self._get_session_id()
                data_type = data.get('type')
                filters = data.get('filters', {})
                
                if not self._check_rate_limit(session_id):
                    emit('error', {'message': 'Rate limit exceeded'})
                    return
                
                # Queue the data request for processing
                self.message_queue.put({
                    'type': 'data_request',
                    'session_id': session_id,
                    'data_type': data_type,
                    'filters': filters,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                self.logger.error(f"Request data error: {e}")
                emit('error', {'message': 'Failed to process data request'})
    # -------------------------------
    # Helpers, processing & API
    # -------------------------------

    def _generate_session_id(self):
        """Generate a unique session id (fallback; we prefer request.sid)."""
        return str(uuid.uuid4())

    def _get_session_id(self):
        """Get session ID from current request context — prefer request.sid for Socket.IO consistency."""
        from flask import request
        return getattr(request, 'sid', None) or self._generate_session_id()

    def _get_client_ip(self):
        """Get client IP address."""
        from flask import request
        # Respect common proxy headers if present
        x_forwarded = request.headers.get('X-Forwarded-For', '')
        if x_forwarded:
            # X-Forwarded-For may contain a list of IPs; take the first (original client)
            return x_forwarded.split(',')[0].strip()
        return request.environ.get('HTTP_X_REAL_IP', request.remote_addr)

    def _get_user_agent(self):
        """Get client user agent."""
        from flask import request
        return request.headers.get('User-Agent', 'Unknown')

    def _check_rate_limit(self, session_id):
        """Check if client is within rate limits."""
        if not session_id:
            return False

        now = time.time()
        client_requests = self.rate_limits[session_id]

        # Remove old requests outside the window (keep list small)
        while client_requests and client_requests[0] < now - self.rate_limit_window:
            client_requests.pop(0)

        # Check if under limit
        if len(client_requests) >= self.rate_limit_max:
            return False

        # Add current request
        client_requests.append(now)
        return True

    def _join_room(self, session_id, room_name):
        """Add client to a room (and make the Socket.IO 'join_room' call)."""
        try:
            # Make Socket.IO join the room for the current request context
            join_room(room_name)
        except Exception:
            # join_room requires proper request context; ignore if called outside handler
            pass

        self.room_subscriptions[room_name].add(session_id)
        self.user_rooms[session_id].add(room_name)

        if session_id in self.active_connections:
            # ensure 'rooms' is a set
            rooms_set = self.active_connections[session_id].get('rooms')
            if isinstance(rooms_set, set):
                rooms_set.add(room_name)
            else:
                self.active_connections[session_id]['rooms'] = {room_name}
        else:
            # create active connection placeholder (useful if called outside connect flow)
            self.active_connections[session_id] = {
                'session_id': session_id,
                'connected_at': datetime.now().isoformat(),
                'ip_address': 'unknown',
                'user_agent': 'unknown',
                'rooms': {room_name},
                'last_activity': time.time()
            }

        self.logger.info(f"Client {session_id} joined room {room_name}")

    def _leave_room(self, session_id, room_name):
        """Remove client from a room (and perform Socket.IO leave_room)."""
        try:
            leave_room(room_name)
        except Exception:
            # leave_room may fail if called outside a request context; we still update internal state
            pass

        self.room_subscriptions[room_name].discard(session_id)
        if session_id in self.user_rooms:
            self.user_rooms[session_id].discard(room_name)

        if session_id in self.active_connections:
            rooms_set = self.active_connections[session_id].get('rooms', set())
            if isinstance(rooms_set, set):
                rooms_set.discard(room_name)

        # Clean up empty rooms
        if not self.room_subscriptions.get(room_name):
            self.room_subscriptions.pop(room_name, None)

        self.logger.info(f"Client {session_id} left room {room_name}")

    def _subscribe_to_stream(self, session_id, stream_id):
        """Subscribe client to a data stream and create a 'stream room' for broadcasting."""
        self.stream_subscribers[stream_id].add(session_id)
        # Also add them to an internal room name we use for emitting stream messages
        try:
            join_room(f'stream_{stream_id}')
        except Exception:
            # as above, may fail outside handler but internal sets are still useful
            pass
        self.logger.info(f"Client {session_id} subscribed to stream {stream_id}")

    def _unsubscribe_from_stream(self, session_id, stream_id):
        """Unsubscribe client from a data stream."""
        self.stream_subscribers[stream_id].discard(session_id)
        try:
            leave_room(f'stream_{stream_id}')
        except Exception:
            pass

        if not self.stream_subscribers.get(stream_id):
            self.stream_subscribers.pop(stream_id, None)

        self.logger.info(f"Client {session_id} unsubscribed from stream {stream_id}")

    def _start_processing_thread(self):
        """Start background thread for message processing if not already running."""
        if self.processing_thread and self.processing_thread.is_alive():
            return

        # mark start time (useful for stats)
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
                    # custom message types can be handled by registered handlers
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

            # Process request synchronously here (or delegate to async workers)
            if data_type == 'health_data':
                data = self._get_health_data(filters)
            elif data_type == 'device_status':
                data = self._get_device_status(filters)
            elif data_type == 'analytics':
                data = self._get_analytics_data(filters)
            else:
                data = {'error': 'Unknown data type', 'requested_type': data_type}

            # Emit to the requesting client's room (Socket.IO session id room)
            if self.socketio:
                try:
                    self.socketio.emit('data_response', {
                        'data_type': data_type,
                        'data': data,
                        'timestamp': datetime.now().isoformat()
                    }, room=session_id)
                except Exception as e:
                    self.logger.error(f"Emit data_response failed: {e}")

        except Exception as e:
            self.logger.error(f"Data request handling error: {e}")

    def _handle_stream_data(self, message):
        """Handle stream data coming from your system and broadcast to subscribers."""
        try:
            stream_id = message.get('stream_id')
            data = message.get('data')
            if stream_id is None:
                self.logger.warning("Stream data message missing stream_id")
                return

            # Add to circular buffer for historical replay
            self.data_streams[stream_id].append({
                'data': data,
                'timestamp': datetime.now().isoformat()
            })

            # Broadcast to subscribers via the 'stream_{stream_id}' room
            if self.socketio:
                try:
                    self.socketio.emit('stream_data', {
                        'stream_id': stream_id,
                        'data': data,
                        'timestamp': datetime.now().isoformat(),
                        'type': 'real_time'
                    }, room=f'stream_{stream_id}')
                except Exception as e:
                    self.logger.error(f"Emit stream_data failed: {e}")

        except Exception as e:
            self.logger.error(f"Stream data handling error: {e}")

    # Placeholder retrieval methods (integrate with your DB/services)
    def _get_health_data(self, filters):
        """Get health data (placeholder - integrate with your database)."""
        return {
            'message': 'Health data retrieval not implemented',
            'filters': filters
        }

    def _get_device_status(self, filters):
        """Get device status data (simple stub)."""
        return {
            'devices': [
                {
                    'id': 'sensor_001',
                    'status': 'online',
                    'last_seen': datetime.now().isoformat()
                }
            ]
        }

    def _get_analytics_data(self, filters):
        """Get analytics data (placeholder)."""
        return {
            'message': 'Analytics data retrieval not implemented',
            'filters': filters
        }

    # -------------------------------
    # Public API
    # -------------------------------

    def broadcast_to_room(self, room_name, event, data):
        """Broadcast message to all clients in a room."""
        try:
            if self.socketio:
                self.socketio.emit(event, data, room=room_name)
                self.logger.info(f"Broadcasted {event} to room {room_name}")
        except Exception as e:
            self.logger.error(f"Broadcast error: {e}")

    def send_to_client(self, session_id, event, data):
        """Send message to a specific client by session id."""
        try:
            if self.socketio:
                self.socketio.emit(event, data, room=session_id)
                self.logger.info(f"Sent {event} to client {session_id}")
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
        """Create a new room (server-side) and optionally store metadata in Redis."""
        try:
            if room_name not in self.room_subscriptions:
                self.room_subscriptions[room_name] = set()

                if self.redis_client:
                    room_info = {
                        'name': room_name,
                        'description': description,
                        'created_at': datetime.now().isoformat()
                    }
                    # store as JSON string
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

            for session_id, info in list(self.active_connections.items()):
                if now - info.get('last_activity', 0) > timeout:
                    inactive_sessions.append(session_id)

            for session_id in inactive_sessions:
                client_rooms = self.active_connections[session_id].get('rooms', set()).copy()
                for room in client_rooms:
                    self._leave_room(session_id, room)
                del self.active_connections[session_id]
                self.logger.info(f"Cleaned up inactive connection: {session_id}")

            return len(inactive_sessions)

        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
            return 0

    def add_event_handler(self, event_name, handler):
        """Add a custom event handler both to registry and to Socket.IO if active."""
        try:
            self.event_handlers[event_name].append(handler)

            if self.socketio:
                # bind a wrapper that calls registered handlers
                @self.socketio.on(event_name)
                def custom_handler(*args, **kwargs):
                    try:
                        for h in self.event_handlers[event_name]:
                            h(*args, **kwargs)
                    except Exception as e:
                        self.logger.error(f"Custom handler error for {event_name}: {e}")

            self.logger.info(f"Event handler added for: {event_name}")

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

            # Stop processing thread
            self.should_stop.set()
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5)

            # Notify clients
            if self.socketio:
                try:
                    self.socketio.emit('server_shutdown', {
                        'message': 'Server is shutting down',
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    self.logger.error(f"Failed to emit shutdown: {e}")

            # Clear internal state
            self.active_connections.clear()
            self.room_subscriptions.clear()
            self.user_rooms.clear()
            self.stream_subscribers.clear()
            self.data_streams.clear()

            self.logger.info("WebSocket manager shutdown complete")

        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")

# -------------------------------
# Example usage (run guard)
# -------------------------------
if __name__ == "__main__":
    from flask import Flask, request

    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key'

    # Instantiate manager (this will call initialize_socketio inside __init__)
    ws_manager = RealtimeWebSocketManager(app)

    # Example custom event handler
    def handle_custom_event(data):
        print(f"Custom event received: {data}")

    ws_manager.add_event_handler('custom_event', handle_custom_event)

    # Create some rooms
    ws_manager.create_room('device_alerts', 'Device alert notifications')
    ws_manager.create_room('system_health', 'System health monitoring')

    # Start the Socket.IO server if present
    if ws_manager.socketio:
        print("WebSocket server starting on port 5000...")
        # Use eventlet or gevent if installed; this follows the SocketIO async_mode
        ws_manager.socketio.run(app, debug=True, port=5000)
