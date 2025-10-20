#!/usr/bin/env python3
"""
Enhanced Flask Application for Digital Twin System v3.2 (Production-Ready - Hardened)
Main web application with Redis caching, PostgreSQL via SQLAlchemy ORM,
optimized SocketIO, async task support, and refactored database operations.

Major Enhancements in v3.2:
- (Correction) Removed corrupted, duplicated code from end of file.
- (Security) Implemented High-Priority Item #2:
  - Removed all hardcoded secrets and connection strings.
  - Application now requires environment variables for all secrets (e.g., SECRET_KEY, DATABASE_URL, REDIS_URL).
  - Added get_required_env() helper to ensure app fails fast if secrets are missing.
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
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, Index, func
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

# ==================== HELPER FUNCTIONS ====================

def setup_logging():
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
    logging.info("Logging configured.")
    return logging.getLogger('DigitalTwinApp')

# Setup logging early so helper can use it
logger = setup_logging()


def get_required_env(var_name: str) -> str:
    """Get a required environment variable or raise an error."""
    value = os.environ.get(var_name)
    if value is None:
        logger.critical(f"CRITICAL: Environment variable '{var_name}' is not set.")
        raise ValueError(f"Missing required environment variable: {var_name}")
    return value

# ==================== DATABASE MODELS ====================

class DeviceData(Base):
    """SQLAlchemy model for device data"""
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
    metadata = Column(Text)  # Store additional JSON data as text
    
    __table_args__ = (
        Index('idx_device_timestamp', 'device_id', 'timestamp'),
        Index('idx_status_timestamp', 'status', 'timestamp'),
    )
    
    def to_dict(self):
        """Helper method to convert model instance to dictionary"""
        data = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        if data.get('metadata'):
            try:
                data['metadata'] = json.loads(data['metadata'])
            except json.JSONDecodeError:
                data['metadata'] = {}
        return data


class Alert(Base):
    """SQLAlchemy model for alerts"""
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
        Index('idx_alert_device', 'device_id', 'timestamp'),
    )
    
    def to_dict(self):
        """Helper method to convert model instance to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class User(Base):
    """SQLAlchemy model for users"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    email = Column(String(200), unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Helper method to convert model instance to dictionary, excluding password"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at
        }


# ==================== DATABASE SETUP ====================

def create_database_engine(db_url: str = None):
    """Create and configure database engine"""
    # Use helper to get required env var. Remove hardcoded default.
    db_url = db_url or get_required_env('DATABASE_URL')
    
    engine = create_engine(
        db_url,
        poolclass=QueuePool,
        pool_size=20,
        max_overflow=40,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False
    )
    
    return engine


def create_tables(engine):
    """Creates all defined tables in the database if they don't exist"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables checked/created successfully.")
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to create database tables: {e}")
        raise


# Session factory (will be initialized in app)
SessionLocal = None


@contextmanager
def get_session():
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database session error: {e}", exc_info=True)
        raise
    finally:
        SessionLocal.remove()


# ==================== REDIS CACHE MANAGER ====================

class RedisCacheManager:
    """Manages Redis caching with connection pooling"""
    
    def __init__(self, redis_url: str = None, ttl: int = 300):
        # Use helper to get required env var. Remove hardcoded default.
        self.redis_url = redis_url or get_required_env('REDIS_URL')
        self.ttl = ttl
        self.logger = logging.getLogger('RedisCacheManager')
        
        try:
            self.pool = ConnectionPool.from_url(
                self.redis_url,
                max_connections=50,
                decode_responses=False
            )
            self.client = redis.Redis(connection_pool=self.pool)
            self.client.ping()
            self.logger.info(f"Redis connected successfully: {self.redis_url}")
            self.available = True
        except Exception as e:
            self.logger.error(f"Redis connection failed: {e}. Falling back to in-memory cache.")
            self.available = False
            self.memory_cache = {}
    
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


# ==================== CELERY CONFIGURATION ====================

def make_celery(app):
    """Create Celery instance"""
    if not CELERY_AVAILABLE:
        return None
    
    celery = Celery(
        app.import_name,
        # Use helper to get required env vars. Remove hardcoded defaults.
        backend=get_required_env('CELERY_RESULT_BACKEND'),
        broker=get_required_env('CELERY_BROKER_URL')
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
        self.engine = None
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
        
        # Use the logger configured at the start
        self.logger = logger
        self.logger.info("Digital Twin Application v3.2 (Hardened) starting...")
        
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
        """This function is now called at the module level before __init__."""
        pass
    
    def create_app(self):
        """Create and configure Flask application"""
        self.app = Flask(__name__, static_folder='static', template_folder='templates')
        
        # ProxyFix for Nginx/HTTPS
        self.app.wsgi_app = ProxyFix(
            self.app.wsgi_app, 
            x_for=1, x_proto=1, x_host=1, x_prefix=1
        )
        
        # Configuration
        # Use helper to get required env vars. Remove hardcoded defaults.
        self.app.config['SECRET_KEY'] = get_required_env('SECRET_KEY')
        self.app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
        self.app.config['JWT_SECRET_KEY'] = get_required_env('JWT_SECRET_KEY')
        self.app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
        
        # JWT
        self.jwt = JWTManager(self.app)
        
        # Rate Limiter with Redis
        # Use helper to get required env var.
        redis_url = get_required_env('REDIS_URL')
        self.limiter = Limiter(
            app=self.app,
            key_func=get_remote_address,
            default_limits=["1000 per hour", "100 per minute"],
            storage_uri=redis_url
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
            message_queue=redis_url, # Use the same redis_url
            channel='digital_twin'
        )
        
        # Celery
        if CELERY_AVAILABLE:
            self.celery = make_celery(self.app)
            self.logger.info("Celery initialized for async tasks.")
        
        self.logger.info("Flask application created successfully")
    
    def initialize_infrastructure(self):
        """Initialize Redis and PostgreSQL"""
        global SessionLocal
        
        # Redis Cache
        self.cache = RedisCacheManager(
            redis_url=get_required_env('REDIS_URL'), # Pass required var
            ttl=300
        )
        
        # PostgreSQL with SQLAlchemy
        try:
            self.engine = create_database_engine() # Will use get_required_env
            create_tables(self.engine)
            
            # Create scoped session factory
            SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            SessionLocal = scoped_session(SessionFactory)
            
            self.logger.info("PostgreSQL initialized successfully via SQLAlchemy ORM")
            self.db_available = True
        except Exception as e:
            self.logger.error(f"PostgreSQL initialization failed: {e}")
            self.db_available = False
        
        self.logger.info("Infrastructure initialized (Redis & PostgreSQL)")
    
    def initialize_modules(self):
        """Initialize AI and analytics modules"""
        try:
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
    
    # ==================== DATABASE OPERATIONS (REFACTORED) ====================
    
    def bulk_insert_device_data(self, data_list: List[Dict]) -> bool:
        """Bulk insert device data efficiently using SQLAlchemy"""
        try:
            with get_session() as session:
                objects = [
                    DeviceData(
                        device_id=d['device_id'],
                        device_type=d.get('device_type'),
                        device_name=d.get('device_name'),
                        timestamp=d['timestamp'] if isinstance(d['timestamp'], datetime) else datetime.fromisoformat(str(d['timestamp'])),
                        value=d.get('value'),
                        status=d.get('status'),
                        health_score=d.get('health_score'),
                        efficiency_score=d.get('efficiency_score'),
                        location=d.get('location'),
                        unit=d.get('unit'),
                        metadata=json.dumps(d.get('metadata', {})) if d.get('metadata') else None
                    )
                    for d in data_list
                ]
                session.bulk_save_objects(objects)
            self.logger.info(f"Bulk inserted {len(objects)} device data records.")
            return True
        except Exception as e:
            self.logger.error(f"Bulk insert error: {e}", exc_info=True)
            return False
    
    def get_latest_device_data(self, limit: int = 100) -> pd.DataFrame:
        """Get latest device data efficiently using SQLAlchemy"""
        try:
            with get_session() as session:
                # Subquery to find the latest timestamp for each device_id
                subq = session.query(
                    DeviceData.device_id,
                    func.max(DeviceData.timestamp).label('max_ts')
                ).group_by(DeviceData.device_id).subquery()
                
                # Join DeviceData with the subquery
                query = session.query(DeviceData).join(
                    subq,
                    (DeviceData.device_id == subq.c.device_id) &
                    (DeviceData.timestamp == subq.c.max_ts)
                ).order_by(DeviceData.timestamp.desc()).limit(limit)
                
                results = query.all()
                data = [r.to_dict() for r in results]
                
                self.logger.info(f"Fetched latest data for {len(results)} devices.")
                return pd.DataFrame(data)
        except Exception as e:
            self.logger.error(f"Get latest device data error: {e}", exc_info=True)
            return pd.DataFrame()
    
    def insert_alert(self, alert_data: Dict) -> bool:
        """Insert a single alert using SQLAlchemy"""
        try:
            with get_session() as session:
                ts = alert_data.get('timestamp')
                if isinstance(ts, str):
                    timestamp = datetime.fromisoformat(ts)
                elif isinstance(ts, datetime):
                    timestamp = ts
                else:
                    timestamp = datetime.utcnow()
                
                alert = Alert(
                    id=alert_data['id'],
                    device_id=alert_data['device_id'],
                    alert_type=alert_data.get('type') or alert_data.get('alert_type'),
                    severity=alert_data['severity'],
                    description=alert_data.get('description') or alert_data.get('message'),
                    timestamp=timestamp,
                    value=alert_data.get('value'),
                    acknowledged=alert_data.get('acknowledged', False)
                )
                session.add(alert)
            self.logger.info(f"Inserted alert: {alert_data['id']}")
            return True
        except Exception as e:
            self.logger.error(f"Alert insert error: {e}", exc_info=True)
            return False
    
    def get_recent_alerts(self, limit: int = 10, severity: str = None, acknowledged: bool = False) -> List[Dict]:
        """Get recent alerts using SQLAlchemy"""
        try:
            with get_session() as session:
                query = session.query(Alert).filter(
                    Alert.acknowledged == acknowledged
                )
                
                if severity:
                    query = query.filter(Alert.severity == severity)
                
                query = query.order_by(Alert.timestamp.desc()).limit(limit)
                results = query.all()
                
                self.logger.info(f"Fetched {len(results)} recent alerts.")
                return [r.to_dict() for r in results]
        except Exception as e:
            self.logger.error(f"Get alerts error: {e}", exc_info=True)
            return []
    
    def create_user_sqlalchemy(self, username: str, password: str, email: str = None) -> bool:
        """Creates a new user with SQLAlchemy"""
        password_hash = generate_password_hash(password)
        try:
            with get_session() as session:
                # Check if user already exists
                existing_user = session.query(User).filter(
                    (User.username == username) | (User.email == email if email else False)
                ).first()
                
                if existing_user:
                    self.logger.warning(f"User creation failed: Username '{username}' or email '{email}' already exists.")
                    return False
                
                new_user = User(username=username, password_hash=password_hash, email=email)
                session.add(new_user)
            
            self.logger.info(f"User '{username}' created successfully via SQLAlchemy.")
            return True
        except SQLAlchemyError as e:
            self.logger.error(f"Error creating user '{username}' via SQLAlchemy: {e}", exc_info=True)
            return False
    
    def authenticate_user_sqlalchemy(self, username: str, password: str) -> bool:
        """Authenticates a user using SQLAlchemy"""
        try:
            with get_session() as session:
                user = session.query(User).filter(User.username == username).first()
            
            if user and check_password_hash(user.password_hash, password):
                self.logger.info(f"User '{username}' authenticated successfully via SQLAlchemy.")
                return True
            
            self.logger.warning(f"Authentication failed for user '{username}' via SQLAlchemy.")
            return False
        except SQLAlchemyError as e:
            self.logger.error(f"Error authenticating user '{username}' via SQLAlchemy: {e}", exc_info=True)
            return False
    
    # ==================== ROUTES ====================
    
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
                if self.create_user_sqlalchemy(username, password, email):
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
                if not self.authenticate_user_sqlalchemy(username, password):
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
                return "Digital Twin API v3.2 is running"
        
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
                    'version': '3.2.0',
                    'redis': self.cache.available,
                    'postgresql': self.db_available,
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
                cached = self.cache.get('dashboard_data')
                if cached:
                    return jsonify(cached)
                
                data = self._fetch_dashboard_data()
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
                
                df = self.get_latest_device_data(limit=1000)
                devices = df.to_dict('records')
                
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
                acknowledged_param = request.args.get('acknowledged', 'false').lower()
                acknowledged_filter = acknowledged_param == 'true'
                
                cache_key = f'alerts:{severity}:{limit}:{acknowledged_filter}'
                cached = self.cache.get(cache_key)
                if cached:
                    return jsonify(cached)
                
                alerts = self.get_recent_alerts(limit=limit, severity=severity, acknowledged=acknowledged_filter)
                
                # Convert datetime objects for JSON serialization
                for alert in alerts:
                    if isinstance(alert.get('timestamp'), datetime):
                        alert['timestamp'] = alert['timestamp'].isoformat()
                
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
                
                devices = self.cache.get('devices_list') or []
                device_data = next((d for d in devices if d.get('device_id') == device_id), None)
                
                if not device_data:
                    return jsonify({'error': 'Device not found'}), 404
                
                device_df = pd.DataFrame([device_data])
                
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
    
    # ==================== WEBSOCKET EVENTS ====================
    
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
            
            cached_data = self.cache.get('dashboard_data')
            if cached_data:
                emit('initial_data', cached_data)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle disconnect"""
            client_id = request.sid
            user_identity = session.get('identity', 'Unknown')
            
            if client_id in self.connected_clients:
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
                    self.rooms[sub_type].add(client_id)
                    if client_id in self.connected_clients:
                        self.connected_clients[client_id]['subscriptions'].add(sub_type)
                    
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
    
    # ==================== BACKGROUND TASKS ====================
    
    def start_background_tasks(self):
        """Start optimized background tasks"""
        
        def data_update_task():
            """Update data and broadcast to subscribed clients only"""
            self.logger.info("Starting data update task")
            update_count = 0
            
            while True:
                try:
                    new_devices_df = self.data_generator.generate_device_data(
                        device_count=25,
                        days_of_data=0.003,
                        interval_minutes=1
                    )
                    
                    latest_devices_df = new_devices_df.loc[
                        new_devices_df.groupby('device_id')['timestamp'].idxmax()
                    ]
                    
                    # Store in PostgreSQL using refactored method
                    devices_list = latest_devices_df.to_dict('records')
                    # Convert pandas Timestamps to datetime objects
                    for d in devices_list:
                        if 'timestamp' in d and isinstance(d['timestamp'], pd.Timestamp):
                            d['timestamp'] = d['timestamp'].to_pydatetime()
                    
                    self.bulk_insert_device_data(devices_list)
                    
                    # Update cache
                    dashboard_data = self._fetch_dashboard_data_from_df(latest_devices_df)
                    self.cache.set('dashboard_data', dashboard_data, ttl=30)
                    self.cache.set('devices_list', latest_devices_df.to_dict('records'), ttl=30)
                    
                    # Broadcast only to subscribed clients
                    if self.rooms['device_updates']:
                        self.socketio.emit(
                            'data_update',
                            dashboard_data,
                            room='device_updates'
                        )
                    
                    # Check for alerts
                    self._check_and_send_alerts(latest_devices_df)
                    
                    update_count += 1
                    if update_count % 12 == 0:
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
                    if self.db_available:
                        df = self.get_latest_device_data(limit=100)
                        if not df.empty:
                            data = self._fetch_dashboard_data_from_df(df)
                            self.cache.set('dashboard_data', data, ttl=300)
                            self.cache.set('devices_list', df.to_dict('records'), ttl=300)
                    
                    eventlet.sleep(60)
                    
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
        """Fetch dashboard data from database using refactored method"""
        try:
            df = self.get_latest_device_data(limit=100)
            return self._fetch_dashboard_data_from_df(df)
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
            
            # Store in database using refactored method
            if self.db_available and all_alerts:
                for alert in all_alerts:
                    self.insert_alert(alert)
            
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
                'database_available': self.db_available
            }
        except ImportError:
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': random.uniform(10, 60),
                'memory_percent': random.uniform(30, 70),
                'disk_percent': random.uniform(40, 80),
                'active_connections': len(self.connected_clients),
                'cache_available': self.cache.available,
                'database_available': self.db_available
            }
    
    def _generate_report_sync(self):
        """Generate report synchronously"""
        try:
            if HealthReportGenerator:
                report_gen = HealthReportGenerator()
                recent_data_df = self.get_latest_device_data(limit=5000) if self.db_available else pd.DataFrame()
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
        self.logger.info(f"Starting Digital Twin Application v3.2 on {host}:{port}")
        self.logger.info(f"Redis: {self.cache.available}, PostgreSQL: {self.db_available}")
        
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
    # Need to get a celery instance. This is tricky without the app.
    # We will define them here, but they rely on the app context.
    # This assumes `celery = make_celery(create_app().app)` is called by the celery worker.
    # For now, let's create a placeholder celery app for defining tasks.
    _celery_app = Celery(
        'digital_twin_tasks',
        backend=os.environ.get('CELERY_RESULT_BACKEND'),
        broker=os.environ.get('CELERY_BROKER_URL')
    )

    @_celery_app.task(bind=True)
    def generate_report_task(self):
        """Async report generation task"""
        try:
            # Re-create app instance for task context
            app_instance = create_app()
            with app_instance.app.app_context():
                report_path = app_instance._generate_report_sync()
                return {'success': True, 'path': report_path}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @_celery_app.task(bind=True)
    def export_data_task(self, format_type, date_range):
        """Async export task"""
        try:
            # Re-create app instance for task context
            app_instance = create_app()
            with app_instance.app.app_context():
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
        
        # This is where Celery gets its app context
        if app_instance.celery:
            global generate_report_task, export_data_task
            
            # Re-bind tasks to the app's celery instance
            generate_report_task = app_instance.celery.task(bind=True)(generate_report_task.run)
            export_data_task = app_instance.celery.task(bind=True)(export_data_task.run)

        return app_instance
    except Exception as e:
        logger.critical(f"Failed to create application: {e}", exc_info=True)
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
        logger.critical(f"Failed to start application: {e}", exc_info=True)
        sys.exit(1)

# --- End of corrected file ---