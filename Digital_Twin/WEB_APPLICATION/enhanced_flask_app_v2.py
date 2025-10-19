#!/usr/bin/env python3
"""
Enhanced Flask Application for Digital Twin System v3.0 (Production-Ready)
Main web application with Redis caching, PostgreSQL, optimized SocketIO,
and async task support.

Major Enhancements:
- Redis for distributed caching
- PostgreSQL via SQLAlchemy
- Optimized SocketIO with rooms and selective broadcasting
- Async task support with Celery
- Connection pooling and query optimization
- Improved error handling and monitoring
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import threading
import time
import uuid
from functools import wraps
import hashlib
import hmac
import secrets
import math
import random
import re
import pickle
from contextlib import contextmanager

# Flask imports
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
import eventlet

# Database imports
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError

# Redis imports
import redis
from redis.connection import ConnectionPool

# Security imports
from flask_jwt_extended import (
    JWTManager, jwt_required, create_access_token,
    get_jwt_identity, unset_jwt_cookies, decode_token
)
from jwt.exceptions import ExpiredSignatureError, DecodeError, InvalidTokenError
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.middleware.proxy_fix import ProxyFix

# Rate Limiting
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
except ImportError:
    logging.critical("Flask-Limiter not installed. Run 'pip install Flask-Limiter'. Exiting.")
    sys.exit(1)

# Background Scheduler
from apscheduler.schedulers.background import BackgroundScheduler

# Celery for async tasks
try:
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    logging.warning("Celery not installed. Async tasks will run synchronously.")
    CELERY_AVAILABLE = False

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
try:
    from CONFIG.app_config import config
    from AI_MODULES.secure_database_manager import SecureDatabaseManager
    from AI_MODULES.predictive_analytics_engine import PredictiveAnalyticsEngine
    from CONFIG.unified_data_generator import UnifiedDataGenerator
    from REPORTS.health_report_generator import HealthReportGenerator
    from AI_MODULES.health_score import HealthScoreCalculator
    from AI_MODULES.alert_manager import AlertManager
    from AI_MODULES.pattern_analyzer import PatternAnalyzer
    from AI_MODULES.recommendation_engine import RecommendationEngine
except ImportError as e:
    logging.critical(f"CRITICAL: Could not import essential modules: {e}")
    sys.exit(1)

# SQLAlchemy Base
Base = declarative_base()

# ==================== DATABASE MODELS ====================

class DeviceData(Base):
    """PostgreSQL model for device data"""
    __tablename__ = 'device_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(100), nullable=False, index=True)
    device_type = Column(String(100))
    device_name = Column(String(200))
    timestamp = Column(DateTime, nullable=False, index=True)
    value = Column(Float)
    status = Column(String(50), index=True)
    health_score = Column(Float)
    efficiency_score = Column(Float)
    location = Column(String(200))
    unit = Column(String(50))
    metadata = Column(Text)  # JSON string for additional data
    
    __table_args__ = (
        Index('idx_device_timestamp', 'device_id', 'timestamp'),
        Index('idx_status_timestamp', 'status', 'timestamp'),
    )

class Alert(Base):
    """PostgreSQL model for alerts"""
    __tablename__ = 'alerts'
    
    id = Column(String(100), primary_key=True)
    device_id = Column(String(100), nullable=False, index=True)
    alert_type = Column(String(100))
    severity = Column(String(50), index=True)
    description = Column(Text)
    timestamp = Column(DateTime, nullable=False, index=True)
    acknowledged = Column(Boolean, default=False)
    value = Column(Float)
    
    __table_args__ = (
        Index('idx_severity_timestamp', 'severity', 'timestamp'),
    )

# ==================== REDIS CACHE MANAGER ====================

class RedisCacheManager:
    """Manages Redis caching with connection pooling"""
    
    def __init__(self, redis_url: str = None, ttl: int = 300):
        self.redis_url = redis_url or os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        self.ttl = ttl  # Default TTL in seconds
        self.logger = logging.getLogger('RedisCacheManager')
        
        try:
            # Create connection pool
            self.pool = ConnectionPool.from_url(
                self.redis_url,
                max_connections=50,
                decode_responses=False  # We'll handle encoding
            )
            self.client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            self.client.ping()
            self.logger.info(f"Redis connected successfully: {self.redis_url}")
            self.available = True
            
        except Exception as e:
            self.logger.error(f"Redis connection failed: {e}. Falling back to in-memory cache.")
            self.available = False
            self.memory_cache = {}  # Fallback
    
    def get(self, key: str) -> Any:
        """Get value from cache"""
        try:
            if self.available:
                data = self.client.get(key)
                return pickle.loads(data) if data else None
            else:
                return self.memory_cache.get(key)
        except Exception as e:
            self.logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache with TTL"""
        try:
            ttl = ttl or self.ttl
            if self.available:
                serialized = pickle.dumps(value)
                return self.client.setex(key, ttl, serialized)
            else:
                self.memory_cache[key] = value
                return True
        except Exception as e:
            self.logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if self.available:
                return self.client.delete(key) > 0
            else:
                return self.memory_cache.pop(key, None) is not None
        except Exception as e:
            self.logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values"""
        try:
            if self.available:
                values = self.client.mget(keys)
                return {
                    k: pickle.loads(v) if v else None 
                    for k, v in zip(keys, values)
                }
            else:
                return {k: self.memory_cache.get(k) for k in keys}
        except Exception as e:
            self.logger.error(f"Cache get_many error: {e}")
            return {k: None for k in keys}
    
    def set_many(self, mapping: Dict[str, Any], ttl: int = None) -> bool:
        """Set multiple key-value pairs"""
        try:
            ttl = ttl or self.ttl
            if self.available:
                pipeline = self.client.pipeline()
                for key, value in mapping.items():
                    serialized = pickle.dumps(value)
                    pipeline.setex(key, ttl, serialized)
                pipeline.execute()
                return True
            else:
                self.memory_cache.update(mapping)
                return True
        except Exception as e:
            self.logger.error(f"Cache set_many error: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern"""
        try:
            if self.available:
                keys = self.client.keys(pattern)
                if keys:
                    return self.client.delete(*keys)
                return 0
            else:
                keys_to_delete = [k for k in self.memory_cache.keys() if pattern in k]
                for k in keys_to_delete:
                    del self.memory_cache[k]
                return len(keys_to_delete)
        except Exception as e:
            self.logger.error(f"Cache clear_pattern error: {e}")
            return 0
    
    def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter"""
        try:
            if self.available:
                return self.client.incr(key, amount)
            else:
                self.memory_cache[key] = self.memory_cache.get(key, 0) + amount
                return self.memory_cache[key]
        except Exception as e:
            self.logger.error(f"Cache increment error: {e}")
            return 0

# ==================== DATABASE MANAGER ====================

class PostgreSQLManager:
    """Manages PostgreSQL connections and operations"""
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or os.environ.get(
            'DATABASE_URL',
            'postgresql://postgres:password@localhost:5432/digital_twin'
        )
        self.logger = logging.getLogger('PostgreSQLManager')
        
        try:
            # Create engine with connection pooling
            self.engine = create_engine(
                self.db_url,
                poolclass=QueuePool,
                pool_size=20,
                max_overflow=40,
                pool_pre_ping=True,  # Verify connections before using
                pool_recycle=3600,   # Recycle connections after 1 hour
                echo=False
            )
            
            # Create tables
            Base.metadata.create_all(self.engine)
            
            # Session factory
            self.SessionLocal = scoped_session(sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            ))
            
            self.logger.info("PostgreSQL connected successfully")
            self.available = True
            
        except Exception as e:
            self.logger.error(f"PostgreSQL connection failed: {e}")
            self.available = False
    
    @contextmanager
    def get_session(self):
        """Context manager for database sessions"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Session error: {e}")
            raise
        finally:
            session.close()
    
    def bulk_insert_device_data(self, data_list: List[Dict]) -> bool:
        """Bulk insert device data efficiently"""
        try:
            with self.get_session() as session:
                objects = [
                    DeviceData(
                        device_id=d['device_id'],
                        device_type=d.get('device_type'),
                        device_name=d.get('device_name'),
                        timestamp=d['timestamp'],
                        value=d.get('value'),
                        status=d.get('status'),
                        health_score=d.get('health_score'),
                        efficiency_score=d.get('efficiency_score'),
                        location=d.get('location'),
                        unit=d.get('unit'),
                        metadata=json.dumps(d.get('metadata', {}))
                    )
                    for d in data_list
                ]
                session.bulk_save_objects(objects)
            return True
        except Exception as e:
            self.logger.error(f"Bulk insert error: {e}")
            return False
    
    def get_latest_device_data(self, limit: int = 100) -> pd.DataFrame:
        """Get latest device data efficiently"""
        try:
            with self.get_session() as session:
                # Subquery for latest timestamp per device
                from sqlalchemy import func
                subq = session.query(
                    DeviceData.device_id,
                    func.max(DeviceData.timestamp).label('max_ts')
                ).group_by(DeviceData.device_id).subquery()
                
                # Join to get full records
                query = session.query(DeviceData).join(
                    subq,
                    (DeviceData.device_id == subq.c.device_id) &
                    (DeviceData.timestamp == subq.c.max_ts)
                ).limit(limit)
                
                results = query.all()
                
                data = [{
                    'device_id': r.device_id,
                    'device_type': r.device_type,
                    'device_name': r.device_name,
                    'timestamp': r.timestamp,
                    'value': r.value,
                    'status': r.status,
                    'health_score': r.health_score,
                    'efficiency_score': r.efficiency_score,
                    'location': r.location,
                    'unit': r.unit
                } for r in results]
                
                return pd.DataFrame(data)
        except Exception as e:
            self.logger.error(f"Query error: {e}")
            return pd.DataFrame()
    
    def insert_alert(self, alert_data: Dict) -> bool:
        """Insert single alert"""
        try:
            with self.get_session() as session:
                alert = Alert(
                    id=alert_data['id'],
                    device_id=alert_data['device_id'],
                    alert_type=alert_data.get('type'),
                    severity=alert_data['severity'],
                    description=alert_data['description'],
                    timestamp=datetime.fromisoformat(alert_data['timestamp']),
                    value=alert_data.get('value')
                )
                session.add(alert)
            return True
        except Exception as e:
            self.logger.error(f"Alert insert error: {e}")
            return False
    
    def get_recent_alerts(self, limit: int = 10, severity: str = None) -> List[Dict]:
        """Get recent alerts"""
        try:
            with self.get_session() as session:
                query = session.query(Alert).filter(
                    Alert.acknowledged == False
                )
                
                if severity:
                    query = query.filter(Alert.severity == severity)
                
                query = query.order_by(Alert.timestamp.desc()).limit(limit)
                results = query.all()
                
                return [{
                    'id': r.id,
                    'device_id': r.device_id,
                    'type': r.alert_type,
                    'severity': r.severity,
                    'description': r.description,
                    'timestamp': r.timestamp.isoformat(),
                    'value': r.value
                } for r in results]
        except Exception as e:
            self.logger.error(f"Get alerts error: {e}")
            return []

# ==================== CELERY CONFIGURATION ====================

def make_celery(app):
    """Create Celery instance"""
    if not CELERY_AVAILABLE:
        return None
    
    celery = Celery(
        app.import_name,
        backend=os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1'),
        broker=os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/1')
    )
    celery.conf.update(app.config)
    
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
    
    celery.Task = ContextTask
    return celery

# ==================== MAIN APPLICATION CLASS ====================

class DigitalTwinApp:
    """Main Digital Twin Flask Application Class"""
    
    def __init__(self):
        self.app = None
        self.socketio = None
        self.cache = None
        self.db_pg = None
        self.db_manager = None
        self.analytics_engine = None
        self.health_calculator = None
        self.alert_manager = None
        self.pattern_analyzer = None
        self.recommendation_engine = None
        self.data_generator = None
        self.scheduler = None
        self.jwt = None
        self.limiter = None
        self.celery = None
        
        # Application state
        self.connected_clients = {}
        self.start_time = datetime.now()
        
        # SocketIO rooms for selective broadcasting
        self.rooms = {
            'device_updates': set(),
            'alerts': set(),
            'system_metrics': set()
        }
        
        # Initialize logging
        self.setup_logging()
        
        # Initialize Flask app
        self.create_app()
        
        # Initialize infrastructure
        self.initialize_infrastructure()
        
        # Initialize modules
        self.initialize_modules()
        
        # Setup routes
        self.setup_routes()
        
        # Setup WebSocket events
        self.setup_websocket_events()
        
        # Start background tasks
        self.start_background_tasks()
    
    def setup_logging(self):
        """Setup comprehensive logging system"""
        os.makedirs('LOGS', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('LOGS/digital_twin_app.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DigitalTwinApp')
        self.logger.info("Digital Twin Application v3.0 starting...")
    
    def create_app(self):
        """Create and configure Flask application"""
        self.app = Flask(__name__, static_folder='static', template_folder='templates')
        
        # ProxyFix for Nginx/HTTPS
        self.app.wsgi_app = ProxyFix(
            self.app.wsgi_app, 
            x_for=1, x_proto=1, x_host=1, x_prefix=1
        )
        
        # Configuration
        self.app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
        self.app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
        self.app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', secrets.token_hex(32))
        self.app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
        
        # JWT
        self.jwt = JWTManager(self.app)
        
        # Rate Limiter with Redis
        redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        self.limiter = Limiter(
            app=self.app,
            key_func=get_remote_address,
            default_limits=["1000 per hour", "100 per minute"],
            storage_uri=redis_url  # Use Redis for distributed rate limiting
        )
        self.logger.info("Flask-Limiter initialized with Redis.")
        
        # CORS
        allowed_origins = os.environ.get(
            'CORS_ALLOWED_ORIGINS',
            'http://localhost:3000,http://127.0.0.1:3000'
        ).split(',')
        CORS(self.app, origins=allowed_origins, supports_credentials=True,
             allow_headers=["Content-Type", "Authorization", "X-CSRF-Token"])
        
        # SocketIO with optimized settings
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins=allowed_origins,
            async_mode='eventlet',
            logger=False,
            engineio_logger=False,
            ping_timeout=60,
            ping_interval=25,
            message_queue=redis_url,  # Redis for message queue (multi-worker support)
            channel='digital_twin'
        )
        
        # Celery
        if CELERY_AVAILABLE:
            self.celery = make_celery(self.app)
            self.logger.info("Celery initialized for async tasks.")
        
        self.logger.info("Flask application created successfully")
    
    def initialize_infrastructure(self):
        """Initialize Redis and PostgreSQL"""
        # Redis Cache
        self.cache = RedisCacheManager(
            redis_url=os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
            ttl=300  # 5 minutes default TTL
        )
        
        # PostgreSQL
        self.db_pg = PostgreSQLManager(
            db_url=os.environ.get('DATABASE_URL')
        )
        
        self.logger.info("Infrastructure initialized (Redis & PostgreSQL)")
    
    def initialize_modules(self):
        """Initialize AI and analytics modules"""
        try:
            self.db_manager = SecureDatabaseManager()
            self.analytics_engine = PredictiveAnalyticsEngine()
            self.health_calculator = HealthScoreCalculator()
            self.alert_manager = AlertManager()
            self.pattern_analyzer = PatternAnalyzer()
            self.recommendation_engine = RecommendationEngine()
            self.data_generator = UnifiedDataGenerator()
            self.scheduler = BackgroundScheduler(daemon=True)
            
            self.logger.info("Modules initialized successfully")
        except Exception as e:
            self.logger.error(f"Module initialization error: {e}", exc_info=True)
    
    def setup_routes(self):
        """Setup all Flask routes"""
        
        # ==================== AUTH ENDPOINTS ====================
        
        @self.app.route('/api/register', methods=['POST'])
        @self.limiter.limit("5 per hour")
        def register():
            data = request.get_json()
            if not data:
                return jsonify({"error": "Invalid request"}), 400
            
            username = data.get('username')
            password = data.get('password')
            email = data.get('email')
            
            errors = {}
            if not username or len(username) < 3:
                errors['username'] = "Username must be at least 3 characters"
            if not password or len(password) < 8:
                errors['password'] = "Password must be at least 8 characters"
            if not email or not re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email):
                errors['email'] = "Valid email required"
            
            if errors:
                return jsonify({"error": "Validation failed", "details": errors}), 400
            
            try:
                if self.db_manager.create_user(username, password, email):
                    return jsonify({"message": "User created successfully"}), 201
                return jsonify({"error": "Username or email exists"}), 409
            except Exception as e:
                self.logger.error(f"Registration error: {e}")
                return jsonify({"error": "Registration failed"}), 500
        
        @self.app.route('/api/login', methods=['POST'])
        @self.limiter.limit("10 per minute")
        def login():
            data = request.get_json()
            if not data:
                return jsonify({"error": "Invalid request"}), 400
            
            username = data.get('username')
            password = data.get('password')
            
            if not username or not password:
                return jsonify({"error": "Username and password required"}), 400
            
            try:
                if not self.db_manager.authenticate_user(username, password):
                    return jsonify({"error": "Invalid credentials"}), 401
                
                access_token = create_access_token(identity=username)
                return jsonify(access_token=access_token), 200
            except Exception as e:
                self.logger.error(f"Login error: {e}")
                return jsonify({"error": "Login failed"}), 500
        
        @self.app.route('/api/logout', methods=['POST'])
        def logout():
            response = jsonify({"message": "Logout successful"})
            unset_jwt_cookies(response)
            return response, 200
        
        @self.app.route('/api/whoami')
        @jwt_required()
        @self.limiter.limit("120 per minute")
        def protected():
            return jsonify(logged_in_as=get_jwt_identity()), 200
        
        # ==================== PAGE ROUTES ====================
        
        @self.app.route('/')
        def index():
            try:
                return render_template('index.html')
            except:
                return "Digital Twin API v3.0 is running"
        
        @self.app.route('/dashboard')
        def dashboard():
            try:
                return render_template('enhanced_dashboard.html')
            except:
                return "Dashboard - Templates not found"
        
        @self.app.route('/analytics')
        def analytics():
            try:
                return render_template('analytics.html')
            except:
                return "Analytics - Templates not found"
        
        @self.app.route('/devices')
        def devices():
            try:
                return render_template('devices_view.html')
            except:
                return "Devices - Templates not found"
        
        # ==================== API ENDPOINTS ====================
        
        @self.app.route('/health')
        @self.limiter.exempt
        def health_check():
            """Health check endpoint"""
            try:
                status = {
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'version': '3.0.0',
                    'redis': self.cache.available,
                    'postgresql': self.db_pg.available,
                    'connected_clients': len(self.connected_clients),
                    'uptime': str(datetime.now() - self.start_time).split('.')[0]
                }
                return jsonify(status), 200
            except Exception as e:
                return jsonify({'status': 'unhealthy', 'error': str(e)}), 503
        
        @self.app.route('/api/dashboard_data')
        @jwt_required()
        @self.limiter.limit("120 per minute")
        def get_dashboard_data():
            """Get dashboard data with caching"""
            try:
                # Try cache first
                cached = self.cache.get('dashboard_data')
                if cached:
                    return jsonify(cached)
                
                # Fetch from database
                data = self._fetch_dashboard_data()
                
                # Cache for 30 seconds
                self.cache.set('dashboard_data', data, ttl=30)
                
                return jsonify(data)
            except Exception as e:
                self.logger.error(f"Error getting dashboard data: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/devices')
        @jwt_required()
        @self.limiter.limit("120 per minute")
        def get_devices():
            """Get all devices"""
            try:
                cached = self.cache.get('devices_list')
                if cached:
                    return jsonify(cached)
                
                if self.db_pg.available:
                    df = self.db_pg.get_latest_device_data(limit=1000)
                    devices = df.to_dict('records')
                else:
                    devices = []
                
                self.cache.set('devices_list', devices, ttl=30)
                return jsonify(devices)
            except Exception as e:
                self.logger.error(f"Error getting devices: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/device/<device_id>')
        @jwt_required()
        @self.limiter.limit("120 per minute")
        def get_device(device_id):
            """Get specific device"""
            try:
                cache_key = f'device:{device_id}'
                cached = self.cache.get(cache_key)
                if cached:
                    return jsonify(cached)
                
                # Fetch from database
                devices = self.cache.get('devices_list') or []
                device = next((d for d in devices if d.get('device_id') == device_id), None)
                
                if device:
                    self.cache.set(cache_key, device, ttl=30)
                    return jsonify(device)
                return jsonify({'error': 'Device not found'}), 404
            except Exception as e:
                self.logger.error(f"Error getting device {device_id}: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/alerts')
        @jwt_required()
        @self.limiter.limit("120 per minute")
        def get_alerts():
            """Get alerts from database"""
            try:
                limit = request.args.get('limit', 10, type=int)
                severity = request.args.get('severity', None)
                
                cache_key = f'alerts:{severity}:{limit}'
                cached = self.cache.get(cache_key)
                if cached:
                    return jsonify(cached)
                
                if self.db_pg.available:
                    alerts = self.db_pg.get_recent_alerts(limit=limit, severity=severity)
                else:
                    alerts = []
                
                self.cache.set(cache_key, alerts, ttl=10)
                return jsonify(alerts)
            except Exception as e:
                self.logger.error(f"Error getting alerts: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/predictions')
        @jwt_required()
        @self.limiter.limit("60 per minute")
        def get_predictions():
            """Get predictions"""
            try:
                device_id = request.args.get('device_id')
                cache_key = f'predictions:{device_id}'
                cached = self.cache.get(cache_key)
                if cached:
                    return jsonify(cached)
                
                # Get device data
                devices = self.cache.get('devices_list') or []
                device_data = next((d for d in devices if d.get('device_id') == device_id), None)
                
                if not device_data:
                    return jsonify({'error': 'Device not found'}), 404
                
                device_df = pd.DataFrame([device_data])
                
                # Run predictions
                anomalies = self.analytics_engine.detect_anomalies(device_df)
                try:
                    failure_pred = self.analytics_engine.predict_failure(device_df)
                except Exception as fe:
                    self.logger.warning(f"Failure prediction failed: {fe}")
                    failure_pred = {'error': str(fe), 'predictions': []}
                
                result = {
                    'device_id': device_id,
                    'anomaly_prediction': anomalies,
                    'failure_prediction': failure_pred
                }
                
                self.cache.set(cache_key, result, ttl=60)
                return jsonify(result)
            except Exception as e:
                self.logger.error(f"Error getting predictions: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/health_scores')
        @jwt_required()
        @self.limiter.limit("60 per minute")
        def get_health_scores():
            """Get health scores"""
            try:
                cache_key = 'health_scores'
                cached = self.cache.get(cache_key)
                if cached:
                    return jsonify(cached)
                
                devices = self.cache.get('devices_list') or []
                devices_df = pd.DataFrame(devices)
                
                if devices_df.empty:
                    return jsonify({}), 200
                
                health_scores = self.health_calculator.calculate_device_health_scores(devices_df)
                formatted_scores = {
                    dev_id: {'overall_health': score * 100}
                    for dev_id, score in health_scores.items()
                }
                
                self.cache.set(cache_key, formatted_scores, ttl=60)
                return jsonify(formatted_scores)
            except Exception as e:
                self.logger.error(f"Error getting health scores: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/recommendations')
        @jwt_required()
        @self.limiter.limit("30 per minute")
        def get_recommendations():
            """Get AI recommendations"""
            try:
                cache_key = 'recommendations'
                cached = self.cache.get(cache_key)
                if cached:
                    return jsonify(cached)
                
                devices = self.cache.get('devices_list') or []
                devices_df = pd.DataFrame(devices)
                
                if devices_df.empty:
                    return jsonify([]), 200
                
                # Prepare data
                health_data = {}
                if self.health_calculator:
                    health_scores = self.health_calculator.calculate_device_health_scores(devices_df)
                    health_data = {'device_scores': health_scores}
                
                pattern_data = {}
                if self.pattern_analyzer:
                    pattern_data = self.pattern_analyzer.analyze_temporal_patterns(
                        devices_df, 'timestamp', ['value']
                    )
                
                recommendations = self.recommendation_engine.generate_recommendations(
                    health_data=health_data,
                    pattern_analysis=pattern_data,
                    historical_data=devices_df
                )
                
                self.cache.set(cache_key, recommendations, ttl=120)
                return jsonify(recommendations)
            except Exception as e:
                self.logger.error(f"Error getting recommendations: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/system_metrics')
        @jwt_required()
        @self.limiter.limit("120 per minute")
        def get_system_metrics():
            """Get system metrics"""
            try:
                cache_key = 'system_metrics'
                cached = self.cache.get(cache_key)
                if cached:
                    return jsonify(cached)
                
                metrics = self._get_system_metrics()
                self.cache.set(cache_key, metrics, ttl=5)
                return jsonify(metrics)
            except Exception as e:
                self.logger.error(f"Error getting system metrics: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/generate_report')
        @jwt_required()
        @self.limiter.limit("10 per hour")
        def generate_report_endpoint():
            """Generate report asynchronously if Celery available"""
            try:
                if self.celery and CELERY_AVAILABLE:
                    task = generate_report_task.delay()
                    return jsonify({
                        'success': True,
                        'task_id': task.id,
                        'message': 'Report generation started'
                    })
                else:
                    # Synchronous fallback
                    report_path = self._generate_report_sync()
                    filename = os.path.basename(report_path)
                    return jsonify({
                        'success': True,
                        'report_path': f'/reports/{filename}',
                        'message': 'Report generated'
                    })
            except Exception as e:
                self.logger.error(f"Report generation error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/reports/<filename>')
        def serve_report(filename):
            """Serve report file"""
            try:
                reports_dir = os.path.join(os.path.dirname(__file__), '..', 'REPORTS', 'generated')
                return send_from_directory(reports_dir, filename)
            except Exception as e:
                return jsonify({'error': 'Report not found'}), 404
        
        @self.app.route('/api/export_data')
        @jwt_required()
        @self.limiter.limit("10 per hour")
        def export_data_endpoint():
            """Export data"""
            try:
                format_type = request.args.get('format', 'json')
                date_range = request.args.get('days', 7, type=int)
                
                if self.celery and CELERY_AVAILABLE:
                    task = export_data_task.delay(format_type, date_range)
                    return jsonify({
                        'success': True,
                        'task_id': task.id,
                        'message': 'Export started'
                    })
                else:
                    filepath = self._export_data_sync(format_type, date_range)
                    filename = os.path.basename(filepath)
                    return jsonify({
                        'success': True,
                        'export_path': f'/exports/{filename}',
                        'filename': filename
                    })
            except Exception as e:
                self.logger.error(f"Export error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/exports/<filename>')
        def serve_export(filename):
            """Serve export file"""
            try:
                exports_dir = os.path.join(os.path.dirname(__file__), '..', 'EXPORTS')
                return send_from_directory(exports_dir, filename)
            except Exception as e:
                return jsonify({'error': 'Export not found'}), 404
        
        self.logger.info("Routes setup completed")
    
    def setup_websocket_events(self):
        """Setup optimized WebSocket events"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle WebSocket connection with JWT auth"""
            token = request.args.get('token')
            if not token:
                self.logger.warning(f"Client {request.sid} connected without token")
                disconnect()
                return False
            
            try:
                jwt_data = decode_token(token)
                identity = jwt_data['sub']
                session['identity'] = identity
            except Exception as e:
                self.logger.warning(f"Client {request.sid} auth failed: {e}")
                disconnect()
                return False
            
            client_id = request.sid
            self.connected_clients[client_id] = {
                'connected_at': datetime.now(),
                'last_ping': datetime.now(),
                'identity': identity,
                'subscriptions': set()
            }
            
            self.logger.info(f"Client {client_id} (User: {identity}) connected. Total: {len(self.connected_clients)}")
            
            # Send initial data
            cached_data = self.cache.get('dashboard_data')
            if cached_data:
                emit('initial_data', cached_data)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle disconnect"""
            client_id = request.sid
            user_identity = session.get('identity', 'Unknown')
            
            if client_id in self.connected_clients:
                # Remove from all rooms
                subs = self.connected_clients[client_id].get('subscriptions', set())
                for sub in subs:
                    if sub in self.rooms:
                        self.rooms[sub].discard(client_id)
                
                del self.connected_clients[client_id]
                self.logger.info(f"Client {client_id} (User: {user_identity}) disconnected. Total: {len(self.connected_clients)}")
        
        @self.socketio.on('ping')
        def handle_ping():
            """Handle ping"""
            client_id = request.sid
            if client_id in self.connected_clients:
                self.connected_clients[client_id]['last_ping'] = datetime.now()
                emit('pong', {'timestamp': datetime.now().isoformat()})
        
        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            """Handle subscription to specific data streams"""
            try:
                client_id = request.sid
                sub_type = data.get('type')
                
                if sub_type in self.rooms:
                    # Add to room tracking
                    self.rooms[sub_type].add(client_id)
                    if client_id in self.connected_clients:
                        self.connected_clients[client_id]['subscriptions'].add(sub_type)
                    
                    # Join SocketIO room
                    join_room(sub_type)
                    emit('subscription_confirmed', {'type': sub_type})
                    self.logger.info(f"Client {client_id} subscribed to {sub_type}")
                else:
                    emit('error', {'message': 'Invalid subscription type'})
            except Exception as e:
                self.logger.error(f"Subscription error: {e}")
                emit('error', {'message': 'Subscription failed'})
        
        @self.socketio.on('unsubscribe')
        def handle_unsubscribe(data):
            """Handle unsubscription"""
            try:
                client_id = request.sid
                sub_type = data.get('type')
                
                if sub_type in self.rooms:
                    self.rooms[sub_type].discard(client_id)
                    if client_id in self.connected_clients:
                        self.connected_clients[client_id]['subscriptions'].discard(sub_type)
                    
                    leave_room(sub_type)
                    emit('unsubscription_confirmed', {'type': sub_type})
                    self.logger.info(f"Client {client_id} unsubscribed from {sub_type}")
            except Exception as e:
                self.logger.error(f"Unsubscription error: {e}")
        
        self.logger.info("WebSocket events setup completed")
    
    def start_background_tasks(self):
        """Start optimized background tasks"""
        
        def data_update_task():
            """Update data and broadcast to subscribed clients only"""
            self.logger.info("Starting data update task")
            update_count = 0
            
            while True:
                try:
                    # Generate new data
                    new_devices_df = self.data_generator.generate_device_data(
                        device_count=25,
                        days_of_data=0.003,
                        interval_minutes=1
                    )
                    
                    latest_devices_df = new_devices_df.loc[
                        new_devices_df.groupby('device_id')['timestamp'].idxmax()
                    ]
                    
                    # Store in PostgreSQL (bulk insert)
                    if self.db_pg.available:
                        devices_list = latest_devices_df.to_dict('records')
                        self.db_pg.bulk_insert_device_data(devices_list)
                    
                    # Update cache
                    dashboard_data = self._fetch_dashboard_data_from_df(latest_devices_df)
                    self.cache.set('dashboard_data', dashboard_data, ttl=30)
                    self.cache.set('devices_list', latest_devices_df.to_dict('records'), ttl=30)
                    
                    # Broadcast only to subscribed clients (optimized)
                    if self.rooms['device_updates']:
                        self.socketio.emit(
                            'data_update',
                            dashboard_data,
                            room='device_updates'
                        )
                    
                    # Check for alerts
                    self._check_and_send_alerts(latest_devices_df)
                    
                    update_count += 1
                    if update_count % 12 == 0:  # Log every minute
                        self.logger.info(f"Data update cycle {update_count} completed")
                    
                    eventlet.sleep(5)
                    
                except Exception as e:
                    self.logger.error(f"Data update task error: {e}", exc_info=True)
                    eventlet.sleep(60)
        
        def cleanup_task():
            """Cleanup inactive clients"""
            self.logger.info("Starting cleanup task")
            
            while True:
                try:
                    current_time = datetime.now()
                    timeout_threshold = current_time - timedelta(minutes=5)
                    
                    disconnected = [
                        cid for cid, info in self.connected_clients.items()
                        if info['last_ping'] < timeout_threshold
                    ]
                    
                    for client_id in disconnected:
                        # Remove from rooms
                        subs = self.connected_clients[client_id].get('subscriptions', set())
                        for sub in subs:
                            if sub in self.rooms:
                                self.rooms[sub].discard(client_id)
                        
                        del self.connected_clients[client_id]
                        self.logger.info(f"Cleaned up inactive client {client_id}")
                    
                    eventlet.sleep(300)
                    
                except Exception as e:
                    self.logger.error(f"Cleanup task error: {e}")
                    eventlet.sleep(600)
        
        def cache_warmup_task():
            """Periodically warm up cache"""
            self.logger.info("Starting cache warmup task")
            
            while True:
                try:
                    # Warm up dashboard data
                    if self.db_pg.available:
                        df = self.db_pg.get_latest_device_data(limit=100)
                        if not df.empty:
                            data = self._fetch_dashboard_data_from_df(df)
                            self.cache.set('dashboard_data', data, ttl=300)
                            self.cache.set('devices_list', df.to_dict('records'), ttl=300)
                    
                    eventlet.sleep(60)  # Every minute
                    
                except Exception as e:
                    self.logger.error(f"Cache warmup error: {e}")
                    eventlet.sleep(300)
        
        # Schedule model retraining
        if self.scheduler and self.analytics_engine:
            if not self.scheduler.get_job('model_retraining_task'):
                self.scheduler.add_job(
                    id='model_retraining_task',
                    func=self.analytics_engine.retrain_models,
                    trigger='interval',
                    hours=24
                )
                self.scheduler.start()
                self.logger.info("Model retraining scheduler started")
        
        # Start background tasks
        self.socketio.start_background_task(data_update_task)
        self.socketio.start_background_task(cleanup_task)
        self.socketio.start_background_task(cache_warmup_task)
        
        self.logger.info("All background tasks started")
    
    # ==================== HELPER METHODS ====================
    
    def _fetch_dashboard_data(self):
        """Fetch dashboard data from database"""
        try:
            if self.db_pg.available:
                df = self.db_pg.get_latest_device_data(limit=100)
                return self._fetch_dashboard_data_from_df(df)
            return {}
        except Exception as e:
            self.logger.error(f"Fetch dashboard data error: {e}")
            return {}
    
    def _fetch_dashboard_data_from_df(self, devices_df):
        """Build dashboard data from DataFrame"""
        try:
            if devices_df.empty:
                return {}
            
            devices_data = devices_df.to_dict('records')
            total_devices = len(devices_data)
            active_devices = devices_df[devices_df['status'] != 'offline'].shape[0]
            
            health_scores = devices_df['health_score'].dropna()
            avg_health = health_scores.mean() * 100 if not health_scores.empty else 0
            
            efficiency_scores = devices_df['efficiency_score'].dropna()
            avg_efficiency = efficiency_scores.mean() * 100 if not efficiency_scores.empty else 0
            
            return {
                'timestamp': datetime.now().isoformat(),
                'systemHealth': round(avg_health),
                'activeDevices': active_devices,
                'totalDevices': total_devices,
                'efficiency': round(avg_efficiency),
                'devices': devices_data
            }
        except Exception as e:
            self.logger.error(f"Build dashboard data error: {e}")
            return {}
    
    def _check_and_send_alerts(self, devices_df):
        """Check for alerts and broadcast to subscribed clients"""
        if not self.alert_manager:
            return
        
        try:
            all_alerts = []
            for _, device_row in devices_df.iterrows():
                device_data = device_row.to_dict()
                try:
                    triggered = self.alert_manager.evaluate_conditions(
                        data=device_data,
                        device_id=device_data.get('device_id')
                    )
                    all_alerts.extend(triggered)
                except Exception as e:
                    self.logger.error(f"Alert evaluation error: {e}")
            
            # Store in database
            if self.db_pg.available and all_alerts:
                for alert in all_alerts:
                    self.db_pg.insert_alert(alert)
            
            # Broadcast only to subscribed clients
            if self.rooms['alerts'] and all_alerts:
                for alert in all_alerts:
                    self.socketio.emit('alert_update', alert, room='alerts')
        
        except Exception as e:
            self.logger.error(f"Check alerts error: {e}")
    
    def _get_system_metrics(self):
        """Get system metrics"""
        try:
            import psutil
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'active_connections': len(self.connected_clients),
                'cache_available': self.cache.available,
                'database_available': self.db_pg.available
            }
        except ImportError:
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': random.uniform(10, 60),
                'memory_percent': random.uniform(30, 70),
                'disk_percent': random.uniform(40, 80),
                'active_connections': len(self.connected_clients),
                'cache_available': self.cache.available,
                'database_available': self.db_pg.available
            }
    
    def _generate_report_sync(self):
        """Generate report synchronously"""
        try:
            if HealthReportGenerator:
                report_gen = HealthReportGenerator()
                recent_data_df = self.db_pg.get_latest_device_data(limit=5000) if self.db_pg.available else pd.DataFrame()
                html_path = report_gen.generate_comprehensive_report(data_df=recent_data_df, date_range_days=7)
                return html_path
            return ""
        except Exception as e:
            self.logger.error(f"Report generation error: {e}")
            raise
    
    def _export_data_sync(self, format_type, date_range):
        """Export data synchronously"""
        try:
            exports_dir = os.path.join(os.path.dirname(__file__), '..', 'EXPORTS')
            os.makedirs(exports_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Get data from cache or database
            devices = self.cache.get('devices_list') or []
            
            if format_type.lower() == 'csv':
                filename = f'export_{timestamp}.csv'
                filepath = os.path.join(exports_dir, filename)
                pd.DataFrame(devices).to_csv(filepath, index=False)
            else:
                filename = f'export_{timestamp}.json'
                filepath = os.path.join(exports_dir, filename)
                with open(filepath, 'w') as f:
                    json.dump({'devices': devices}, f, indent=2, default=str)
            
            return filepath
        except Exception as e:
            self.logger.error(f"Export error: {e}")
            raise
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the application"""
        self.start_time = datetime.now()
        self.logger.info(f"Starting Digital Twin Application v3.0 on {host}:{port}")
        self.logger.info(f"Redis: {self.cache.available}, PostgreSQL: {self.db_pg.available}")
        
        try:
            self.socketio.run(
                self.app,
                host=host,
                port=port,
                debug=debug,
                use_reloader=False,
                log_output=True
            )
        except KeyboardInterrupt:
            self.logger.info("Application stopped by user")
            if self.scheduler and self.scheduler.running:
                self.scheduler.shutdown()
        except Exception as e:
            self.logger.critical(f"Application failed: {e}", exc_info=True)
            if self.scheduler and self.scheduler.running:
                self.scheduler.shutdown()
            raise

# ==================== CELERY TASKS ====================

if CELERY_AVAILABLE:
    @celery.task(bind=True)
    def generate_report_task(self):
        """Async report generation task"""
        try:
            app_instance = create_app()
            report_path = app_instance._generate_report_sync()
            return {'success': True, 'path': report_path}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @celery.task(bind=True)
    def export_data_task(self, format_type, date_range):
        """Async export task"""
        try:
            app_instance = create_app()
            filepath = app_instance._export_data_sync(format_type, date_range)
            return {'success': True, 'path': filepath}
        except Exception as e:
            return {'success': False, 'error': str(e)}

# ==================== ERROR HANDLERS ====================

def setup_error_handlers(app, limiter):
    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({'error': 'Not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
    
    @app.errorhandler(429)
    def ratelimit_handler(e):
        return jsonify(error="Rate limit exceeded", description=str(e.description)), 429
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        app.logger.error(f"Unhandled exception: {e}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred'}), 500

# ==================== APPLICATION FACTORY ====================

def create_app():
    """Application factory"""
    try:
        app_instance = DigitalTwinApp()
        setup_error_handlers(app_instance.app, app_instance.limiter)
        return app_instance
    except Exception as e:
        logging.critical(f"Failed to create application: {e}", exc_info=True)
        sys.exit(1)

# ==================== MAIN ====================

if __name__ == '__main__':
    os.environ.setdefault('FLASK_ENV', 'development')
    try:
        digital_twin_app = create_app()
        host = os.environ.get('HOST', '0.0.0.0')
        port = int(os.environ.get('PORT', 5000))
        debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
        digital_twin_app.run(host=host, port=port, debug=debug)
    except Exception as e:
        print(f"CRITICAL: Failed to start application: {e}")
        logging.critical(f"Failed to start application: {e}", exc_info=True)
        sys.exit(1)