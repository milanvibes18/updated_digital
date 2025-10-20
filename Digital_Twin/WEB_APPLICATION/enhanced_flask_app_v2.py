#!/usr/bin/env python3
"""
Enhanced Flask Application for Digital Twin System v3.2 (Production-Ready - Hardened)
Main web application with Redis caching, PostgreSQL via SQLAlchemy ORM,
optimized SocketIO, async task support, and refactored database operations.

Major Enhancements:
- Standardized on PostgreSQL via SQLAlchemy ORM (Partially implemented pattern below)
- Implemented High-Priority Item #2 (Secrets Management):
  - Removed all hardcoded secrets and connection strings.
  - Application now requires environment variables for all secrets.
  - Added get_required_env() helper.
- Added Flask-Limiter (Part of Item 6)
- Setup basic Celery integration (Part of Item 7)
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
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base # Updated import
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
    FLASK_LIMITER_AVAILABLE = True
except ImportError:
    logging.warning("Flask-Limiter not installed. Rate limiting disabled.")
    FLASK_LIMITER_AVAILABLE = False
    # Define dummy Limiter and decorators if not available
    class Limiter:
        def __init__(self, *args, **kwargs): pass
        def limit(self, *args, **kwargs): return lambda f: f
        def exempt(self, f): return f
        def init_app(self, app): pass
    def get_remote_address(): return "127.0.0.1"


# Background Scheduler (To be replaced by Celery Beat eventually)
from apscheduler.schedulers.background import BackgroundScheduler

# Celery for async tasks
try:
    from celery import Celery, Task # Added Task import
    CELERY_AVAILABLE = True
except ImportError:
    logging.warning("Celery not installed. Async tasks will run synchronously.")
    CELERY_AVAILABLE = False
    # Define dummy Celery if not available
    class Celery:
        def __init__(self, *args, **kwargs): pass
        def task(self, *args, **kwargs): return lambda f: f
    Task = object # Dummy Task base class

# Add project root to path
# Use relative pathing to be more robust
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import custom modules (Use try-except for robustness)
try:
    # from CONFIG.app_config import config # config.py is less flexible than env vars now
    from Digital_Twin.AI_MODULES.predictive_analytics_engine import PredictiveAnalyticsEngine
    from Digital_Twin.CONFIG.unified_data_generator import UnifiedDataGenerator
    from Digital_Twin.REPORTS.health_report_generator import HealthReportGenerator
    from Digital_Twin.AI_MODULES.health_score import HealthScoreCalculator
    from Digital_Twin.AI_MODULES.alert_manager import AlertManager
    from Digital_Twin.AI_MODULES.pattern_analyzer import PatternAnalyzer
    from Digital_Twin.AI_MODULES.recommendation_engine import RecommendationEngine
    # Assuming secure_database_manager might still be used for encryption/users if not fully replaced by SQLAlchemy models
    from Digital_Twin.AI_MODULES.secure_database_manager import SecureDatabaseManager
except ImportError as e:
    logging.warning(f"Could not import some modules: {e}. Functionality might be limited.")
    # Define dummy classes if needed
    PredictiveAnalyticsEngine = HealthScoreCalculator = AlertManager = PatternAnalyzer = RecommendationEngine = UnifiedDataGenerator = HealthReportGenerator = SecureDatabaseManager = object


# ==================== HELPER FUNCTIONS ====================

def setup_logging():
    """Setup comprehensive logging system"""
    log_dir = Path('LOGS')
    log_dir.mkdir(exist_ok=True)
    log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'digital_twin_app.log'),
            logging.StreamHandler(sys.stdout) # Log to console as well
        ]
    )
    # Silence overly verbose libraries
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    logging.getLogger('apscheduler').setLevel(logging.WARNING)
    logging.getLogger('socketio').setLevel(logging.WARNING)
    logging.getLogger('engineio').setLevel(logging.WARNING)

    logger_instance = logging.getLogger('DigitalTwinApp')
    logger_instance.info("Logging configured.")
    return logger_instance

logger = setup_logging()

def get_required_env(var_name: str) -> str:
    """Get a required environment variable or raise an error."""
    value = os.environ.get(var_name)
    if value is None:
        logger.critical(f"CRITICAL: Environment variable '{var_name}' is not set.")
        raise ValueError(f"Missing required environment variable: {var_name}")
    logger.debug(f"Loaded required env var: {var_name}")
    return value

def get_optional_env(var_name: str, default: Any = None) -> Any:
    """Get an optional environment variable."""
    value = os.environ.get(var_name, default)
    logger.debug(f"Loaded optional env var: {var_name} (default: {default})")
    return value

# ==================== DATABASE MODELS (SQLAlchemy) ====================
# Item 1: Standardize on PostgreSQL via SQLAlchemy ORM

Base = declarative_base()

class DeviceData(Base):
    """SQLAlchemy model for device data"""
    __tablename__ = 'device_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(100), nullable=False, index=True)
    device_type = Column(String(100))
    device_name = Column(String(200))
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
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
            except (json.JSONDecodeError, TypeError): # Added TypeError check
                data['metadata'] = {} # Default to empty dict if metadata is not valid JSON
        # Convert datetime to ISO string
        if isinstance(data.get('timestamp'), datetime):
            data['timestamp'] = data['timestamp'].isoformat()
        return data


class Alert(Base):
    """SQLAlchemy model for alerts"""
    __tablename__ = 'alerts'

    # Changed id to String to match AlertManager's UUID generation
    id = Column(String(100), primary_key=True, default=lambda: str(uuid.uuid4()))
    device_id = Column(String(100), nullable=False, index=True)
    rule_name = Column(String(100)) # Added rule_name
    severity = Column(String(50), index=True)
    message = Column(Text) # Renamed from description for clarity
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    acknowledged = Column(Boolean, default=False, index=True)
    resolved = Column(Boolean, default=False, index=True) # Added resolved status
    value = Column(Float) # Value that triggered the alert
    metadata = Column(Text) # Extra context

    __table_args__ = (
        Index('idx_severity_timestamp', 'severity', 'timestamp'),
        Index('idx_alert_device', 'device_id', 'timestamp'),
    )

    def to_dict(self):
        """Helper method to convert model instance to dictionary"""
        data = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        if isinstance(data.get('timestamp'), datetime):
            data['timestamp'] = data['timestamp'].isoformat()
        if data.get('metadata'):
            try:
                data['metadata'] = json.loads(data['metadata'])
            except (json.JSONDecodeError, TypeError):
                data['metadata'] = {}
        return data


class User(Base):
    """SQLAlchemy model for users"""
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    email = Column(String(200), unique=True, index=True) # Added index
    created_at = Column(DateTime, default=datetime.utcnow)
    # Add roles, permissions etc. later if needed

    def to_dict(self):
        """Helper method to convert model instance to dictionary, excluding password"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# ==================== DATABASE SETUP (SQLAlchemy) ====================

# Global engine and session factory
engine = None
SessionLocal = None

def init_database(db_url: str):
    """Initializes the database engine and session factory."""
    global engine, SessionLocal
    if engine is None:
        try:
            engine = create_engine(
                db_url,
                poolclass=QueuePool,
                pool_size=int(get_optional_env('DB_POOL_SIZE', 20)),
                max_overflow=int(get_optional_env('DB_MAX_OVERFLOW', 40)),
                pool_pre_ping=True,
                pool_recycle=3600, # Recycle connections hourly
                echo=get_optional_env('SQLALCHEMY_ECHO', 'False').lower() == 'true'
            )
            # Test connection
            with engine.connect() as connection:
                logger.info("Database connection successful.")

            # Create tables
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables checked/created successfully.")

            # Create scoped session factory
            SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            SessionLocal = scoped_session(SessionFactory)
            logger.info("SQLAlchemy SessionLocal created.")

        except Exception as e:
            logger.critical(f"CRITICAL: Failed to initialize database: {e}", exc_info=True)
            raise # Re-raise to prevent app from starting in broken state


@contextmanager
def get_session():
    """Provide a transactional scope around a series of operations."""
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    session = SessionLocal()
    logger.debug(f"Acquired DB session {id(session)}")
    try:
        yield session
        session.commit()
        logger.debug(f"Committed DB session {id(session)}")
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database session error (rolled back): {e}", exc_info=True)
        raise
    finally:
        SessionLocal.remove()
        logger.debug(f"Removed DB session {id(session)}")


# ==================== REDIS CACHE MANAGER ====================

class RedisCacheManager:
    """Manages Redis caching with connection pooling"""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RedisCacheManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, redis_url: str = None, ttl: int = 300):
        if self._initialized:
            return
        self.redis_url = redis_url or get_required_env('REDIS_URL')
        self.ttl = ttl
        self.logger = logging.getLogger('RedisCacheManager')
        self.client = None
        self.available = False
        self.memory_cache = {} # Fallback

        try:
            # Decode responses=True makes get return strings directly
            self.client = redis.from_url(
                self.redis_url,
                decode_responses=True, # Changed for easier handling
                socket_connect_timeout=5, # Added timeout
                socket_keepalive=True
            )
            self.client.ping()
            self.logger.info(f"Redis connected successfully: {self.redis_url}")
            self.available = True
        except Exception as e:
            self.logger.error(f"Redis connection failed: {e}. Falling back to in-memory cache.")
            self.available = False

        self._initialized = True

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, attempting JSON deserialization."""
        try:
            if self.available and self.client:
                data_str = self.client.get(key)
                if data_str:
                    try:
                        # Attempt to parse as JSON first
                        return json.loads(data_str)
                    except json.JSONDecodeError:
                        # Fallback to returning the raw string if not JSON
                        self.logger.debug(f"Cache GET for '{key}' returned non-JSON string.")
                        return data_str
                return None
            else:
                return self.memory_cache.get(key)
        except Exception as e:
            self.logger.error(f"Cache GET error for key '{key}': {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with TTL, serializing complex types to JSON."""
        try:
            effective_ttl = ttl if ttl is not None else self.ttl
            # Serialize non-primitive types to JSON
            if not isinstance(value, (str, int, float, bool)):
                value_str = json.dumps(value, default=str) # Use default=str for datetimes etc.
            else:
                value_str = str(value)

            if self.available and self.client:
                return self.client.setex(key, effective_ttl, value_str)
            else:
                self.memory_cache[key] = value # Store original value in memory cache
                # Simulate TTL for memory cache (basic cleanup, not perfect)
                if effective_ttl > 0:
                     threading.Timer(effective_ttl, self.memory_cache.pop, args=[key, None]).start()
                return True
        except Exception as e:
            self.logger.error(f"Cache SET error for key '{key}': {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if self.available and self.client:
                return self.client.delete(key) > 0
            else:
                return self.memory_cache.pop(key, None) is not None
        except Exception as e:
            self.logger.error(f"Cache DELETE error for key '{key}': {e}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern"""
        deleted_count = 0
        try:
            if self.available and self.client:
                # Use scan_iter for potentially large number of keys
                keys_to_delete = [key for key in self.client.scan_iter(match=pattern)]
                if keys_to_delete:
                    deleted_count = self.client.delete(*keys_to_delete)
            else:
                keys_to_delete = [k for k in self.memory_cache.keys() if pattern in k] # Simple pattern matching
                for k in keys_to_delete:
                    del self.memory_cache[k]
                deleted_count = len(keys_to_delete)
            self.logger.info(f"Cleared {deleted_count} cache keys matching pattern '{pattern}'")
            return deleted_count
        except Exception as e:
            self.logger.error(f"Cache clear_pattern error for '{pattern}': {e}")
            return 0


# ==================== CELERY CONFIGURATION ====================
# Item 7: Migrate to Celery + Celery Beat

celery_app = None

def make_celery(app_name=__name__):
    """Factory to create a Celery app instance."""
    if not CELERY_AVAILABLE:
        return None

    try:
        broker_url = get_required_env('CELERY_BROKER_URL')
        result_backend = get_required_env('CELERY_RESULT_BACKEND')

        return Celery(
            app_name,
            broker=broker_url,
            backend=result_backend,
            include=['Digital_Twin.WEB_APPLICATION.enhanced_flask_app_v2'] # Point to this module for tasks
        )
    except Exception as e:
        logger.error(f"Failed to create Celery app: {e}", exc_info=True)
        return None

# Initialize Celery app globally - the worker will use this
celery_app = make_celery()

def init_celery(app: Flask, celery: Optional[Celery]):
    """Bind Celery configuration to Flask app and add context."""
    if not celery:
        return

    celery.conf.update(
        result_expires=timedelta(hours=1), # Store results for 1 hour
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone=get_optional_env('TZ', 'UTC'),
        enable_utc=True,
        # Add Celery Beat schedule here if needed
        # beat_schedule = {
        #     'retrain-models-daily': {
        #         'task': 'Digital_Twin.WEB_APPLICATION.enhanced_flask_app_v2.retrain_models_task',
        #         'schedule': crontab(hour=0, minute=0), # Run daily at midnight
        #     },
        # }
    )

    class ContextTask(Task): # Use the imported Task
        def __call__(self, *args, **kwargs):
            with app.app_context():
                # Re-initialize DB engine within task context if needed,
                # especially if using multiprocessing worker pools.
                # init_database(app.config['DATABASE_URL']) # Example
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    app.extensions["celery"] = celery
    logger.info("Celery initialized and linked with Flask app context.")


# ==================== MAIN APPLICATION CLASS ====================

class DigitalTwinApp:
    """Main Digital Twin Flask Application Class"""

    def __init__(self):
        self.app: Optional[Flask] = None
        self.socketio: Optional[SocketIO] = None
        self.cache: Optional[RedisCacheManager] = None
        # engine and SessionLocal are global now
        self.analytics_engine = None
        self.health_calculator = None
        self.alert_manager = None
        self.pattern_analyzer = None
        self.recommendation_engine = None
        self.data_generator = None
        self.scheduler: Optional[BackgroundScheduler] = None # Keep for now, might migrate fully to Celery Beat
        self.jwt: Optional[JWTManager] = None
        self.limiter: Optional[Limiter] = None
        self.celery: Optional[Celery] = celery_app # Use the global instance

        # Application state
        self.connected_clients: Dict[str, Dict] = {}
        self.start_time = datetime.now()

        # SocketIO rooms for selective broadcasting
        self.rooms: Dict[str, Set[str]] = defaultdict(set)

        self.logger = logger # Use the globally configured logger
        self.logger.info("Digital Twin Application v3.2 (SQLAlchemy+Secrets) starting...")

        # Create Flask app
        self._create_app()
        if not self.app:
             raise RuntimeError("Flask app creation failed.")

        # Initialize infrastructure (DB, Redis, Celery)
        self._initialize_infrastructure()

        # Initialize business logic modules
        self._initialize_modules()

        # Setup routes, websockets, background tasks
        self._setup_middleware()
        self._setup_routes()
        self._setup_websocket_events()
        self._start_background_tasks()
        self._setup_error_handlers()

        self.logger.info("Application initialization complete.")

    def _create_app(self):
        """Create and configure Flask application"""
        self.app = Flask(
            __name__,
            static_folder='static',
            template_folder='templates',
            instance_relative_config=True # Good practice
        )

        # Load configuration from environment variables
        self.app.config.from_mapping(
            SECRET_KEY=get_required_env('SECRET_KEY'),
            DEBUG=get_optional_env('FLASK_DEBUG', 'False').lower() == 'true',
            JWT_SECRET_KEY=get_required_env('JWT_SECRET_KEY'),
            JWT_ACCESS_TOKEN_EXPIRES=timedelta(hours=int(get_optional_env('JWT_EXPIRY_HOURS', 24))),
            JWT_TOKEN_LOCATION=['headers', 'cookies'], # Allow cookies for browser sessions
            JWT_COOKIE_SECURE=get_optional_env('FLASK_ENV', 'development') == 'production', # Use secure cookies in prod
            JWT_COOKIE_CSRF_PROTECT=True, # Enable CSRF protection for cookies
            JWT_CSRF_CHECK_FORM=True, # Check CSRF in forms too

            # Database URL
            DATABASE_URL=get_required_env('DATABASE_URL'),

            # Redis URL
            REDIS_URL=get_required_env('REDIS_URL'),

            # Celery config (optional, can also be in celeryconfig.py)
            CELERY_BROKER_URL=get_optional_env('CELERY_BROKER_URL'),
            CELERY_RESULT_BACKEND=get_optional_env('CELERY_RESULT_BACKEND'),

            # Rate Limiter
            RATELIMIT_STORAGE_URI=get_required_env('REDIS_URL'), # Use Redis for limiter
            RATELIMIT_STRATEGY='fixed-window', # or 'moving-window'
            RATELIMIT_HEADERS_ENABLED=True,

            # CORS
            CORS_ALLOWED_ORIGINS=get_optional_env('CORS_ALLOWED_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000').split(',')
        )

        self.logger.info(f"Flask app created. Environment: {self.app.config['ENV']}, Debug: {self.app.config['DEBUG']}")

    def _initialize_infrastructure(self):
        """Initialize DB, Redis, Celery, JWT, Limiter"""
        global engine, SessionLocal # Allow modification

        # --- Database (Item 1) ---
        init_database(self.app.config['DATABASE_URL'])
        self.db_available = engine is not None and SessionLocal is not None

        # --- Redis Cache (Item 2 - uses REDIS_URL from env) ---
        self.cache = RedisCacheManager(redis_url=self.app.config['REDIS_URL'])
        if not self.cache.available:
            self.logger.warning("Running with in-memory cache fallback.")

        # --- Celery (Item 7) ---
        if self.celery:
            init_celery(self.app, self.celery)

        # --- JWT ---
        self.jwt = JWTManager(self.app)
        self.logger.info("JWTManager initialized.")

        # --- Rate Limiter (Item 6) ---
        if FLASK_LIMITER_AVAILABLE:
            self.limiter = Limiter(key_func=get_remote_address)
            self.limiter.init_app(self.app)
            self.logger.info("Flask-Limiter initialized with Redis.")
        else:
            self.limiter = Limiter() # Dummy limiter

        # --- SocketIO ---
        allowed_origins = self.app.config['CORS_ALLOWED_ORIGINS']
        try:
            # Use Redis message queue if available and configured
            message_queue_url = self.app.config['REDIS_URL'] if self.cache.available else None
            if message_queue_url:
                logger.info(f"Using Redis message queue for SocketIO: {message_queue_url}")

            self.socketio = SocketIO(
                self.app,
                cors_allowed_origins=allowed_origins,
                async_mode='eventlet',
                logger=self.app.config['DEBUG'],      # Enable logs only in debug
                engineio_logger=self.app.config['DEBUG'],
                ping_timeout=60,
                ping_interval=25,
                message_queue=message_queue_url, # Pass Redis URL
                channel=get_optional_env('SOCKETIO_CHANNEL', 'digital_twin') # Optional channel name
            )
            self.logger.info("Flask-SocketIO initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Flask-SocketIO: {e}", exc_info=True)
            # Potentially fallback or raise error


    def _initialize_modules(self):
        """Initialize AI and analytics modules"""
        self.logger.info("Initializing application modules...")
        try:
            # Pass necessary config/components if needed
            self.analytics_engine = PredictiveAnalyticsEngine() if PredictiveAnalyticsEngine else None
            self.health_calculator = HealthScoreCalculator() if HealthScoreCalculator else None
            self.alert_manager = AlertManager() if AlertManager else None
            self.pattern_analyzer = PatternAnalyzer() if PatternAnalyzer else None
            self.recommendation_engine = RecommendationEngine() if RecommendationEngine else None
            self.data_generator = UnifiedDataGenerator(db_path=self.app.config['DATABASE_URL']) if UnifiedDataGenerator else None # Pass DB path if it needs it

            # Scheduler for tasks not suitable for Celery Beat (e.g., in-memory cleanup)
            self.scheduler = BackgroundScheduler(daemon=True)

            self.logger.info("Application modules initialized successfully.")
        except Exception as e:
            self.logger.error(f"Module initialization error: {e}", exc_info=True)
            # Decide if this is critical or if app can run degraded

    def _setup_middleware(self):
        """Setup Flask middleware"""
        # ProxyFix for Nginx/HTTPS
        self.app.wsgi_app = ProxyFix(
            self.app.wsgi_app,
            x_for=int(get_optional_env('PROXY_X_FOR', 1)),
            x_proto=int(get_optional_env('PROXY_X_PROTO', 1)),
            x_host=int(get_optional_env('PROXY_X_HOST', 1)),
            x_prefix=int(get_optional_env('PROXY_X_PREFIX', 1))
        )
        self.logger.info("ProxyFix middleware applied.")

        # CORS
        CORS(self.app,
             origins=self.app.config['CORS_ALLOWED_ORIGINS'],
             supports_credentials=True,
             allow_headers=["Content-Type", "Authorization", "X-CSRF-TOKEN"])
        self.logger.info(f"CORS configured for origins: {self.app.config['CORS_ALLOWED_ORIGINS']}")

        # Add any other middleware here (e.g., request timing, custom headers)


    # ==================== DATABASE OPERATIONS (SQLAlchemy) ====================
    # Item 1: Implement DB operations using SQLAlchemy session

    def bulk_insert_device_data(self, data_list: List[Dict]) -> bool:
        """Bulk insert device data efficiently using SQLAlchemy"""
        inserted_count = 0
        try:
            with get_session() as session:
                objects_to_insert = []
                for d in data_list:
                    # Basic validation
                    if not d.get('device_id') or not d.get('timestamp'):
                        self.logger.warning(f"Skipping invalid device data record: {d}")
                        continue

                    # Ensure timestamp is datetime
                    ts = d['timestamp']
                    if isinstance(ts, str):
                        try:
                            timestamp = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        except ValueError:
                            self.logger.warning(f"Skipping record with invalid timestamp format: {ts}")
                            continue
                    elif isinstance(ts, (int, float)):
                        timestamp = datetime.fromtimestamp(ts)
                    elif not isinstance(ts, datetime):
                         self.logger.warning(f"Skipping record with invalid timestamp type: {type(ts)}")
                         continue
                    else:
                        timestamp = ts # Already datetime

                    objects_to_insert.append(DeviceData(
                        device_id=d['device_id'],
                        device_type=d.get('device_type'),
                        device_name=d.get('device_name'),
                        timestamp=timestamp,
                        value=float(d['value']) if d.get('value') is not None else None,
                        status=d.get('status'),
                        health_score=float(d['health_score']) if d.get('health_score') is not None else None,
                        efficiency_score=float(d['efficiency_score']) if d.get('efficiency_score') is not None else None,
                        location=d.get('location'),
                        unit=d.get('unit'),
                        metadata=json.dumps(d.get('metadata', {})) if d.get('metadata') else None
                    ))

                if objects_to_insert:
                    session.bulk_save_objects(objects_to_insert)
                    inserted_count = len(objects_to_insert)
            if inserted_count > 0:
                self.logger.info(f"Bulk inserted {inserted_count} device data records.")
            return True
        except Exception as e:
            self.logger.error(f"Bulk insert error: {e}", exc_info=True)
            return False

    def get_latest_device_data(self, limit: int = 100) -> pd.DataFrame:
        """Get latest device data efficiently using SQLAlchemy"""
        cache_key = f"latest_devices:{limit}"
        cached = self.cache.get(cache_key)
        if cached is not None and isinstance(cached, list): # Check type
            self.logger.debug(f"Cache hit for '{cache_key}'")
            try:
                # Recreate DataFrame from cached dict list
                return pd.DataFrame(cached)
            except Exception as e:
                self.logger.warning(f"Failed to create DataFrame from cache for '{cache_key}': {e}")
                self.cache.delete(cache_key) # Invalidate cache

        self.logger.debug(f"Cache miss for '{cache_key}'")
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
                data_list = [r.to_dict() for r in results]

            self.logger.info(f"Fetched latest data for {len(results)} devices from DB.")
            if data_list:
                self.cache.set(cache_key, data_list, ttl=60) # Cache for 1 minute
                return pd.DataFrame(data_list)
            else:
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Get latest device data error: {e}", exc_info=True)
            return pd.DataFrame()

    def insert_alert(self, alert_data: Dict) -> bool:
        """Insert a single alert using SQLAlchemy"""
        try:
            with get_session() as session:
                ts = alert_data.get('timestamp')
                if isinstance(ts, str):
                    try:
                        timestamp = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    except ValueError:
                        timestamp = datetime.utcnow()
                elif isinstance(ts, datetime):
                    timestamp = ts
                else:
                    timestamp = datetime.utcnow()

                # Use AlertManager generated ID if available, else generate new
                alert_id = alert_data.get('id', str(uuid.uuid4()))

                alert = Alert(
                    id=alert_id,
                    device_id=alert_data['device_id'],
                    rule_name=alert_data.get('rule_name'),
                    severity=alert_data['severity'],
                    message=alert_data.get('description') or alert_data.get('message'),
                    timestamp=timestamp,
                    value=float(alert_data['value']) if alert_data.get('value') is not None else None,
                    acknowledged=alert_data.get('acknowledged', False),
                    resolved=alert_data.get('resolved', False),
                    metadata=json.dumps(alert_data.get('metadata', {})) if alert_data.get('metadata') else None
                )
                session.add(alert)
            self.logger.info(f"Inserted alert: {alert_id}")
            # Invalidate relevant alert cache
            self.cache.clear_pattern("alerts:*")
            return True
        except Exception as e:
            self.logger.error(f"Alert insert error: {e}", exc_info=True)
            return False

    def get_recent_alerts(self, limit: int = 10, severity: Optional[str] = None, acknowledged: Optional[bool] = None) -> List[Dict]:
        """Get recent alerts using SQLAlchemy"""
        cache_key = f"alerts:{severity}:{limit}:{acknowledged}"
        cached = self.cache.get(cache_key)
        if cached is not None and isinstance(cached, list):
            self.logger.debug(f"Cache hit for '{cache_key}'")
            return cached

        self.logger.debug(f"Cache miss for '{cache_key}'")
        try:
            with get_session() as session:
                query = session.query(Alert)

                if acknowledged is not None:
                    query = query.filter(Alert.acknowledged == acknowledged)

                if severity:
                    query = query.filter(Alert.severity == severity)

                # Optionally filter by resolved status if needed in future
                # query = query.filter(Alert.resolved == False)

                query = query.order_by(Alert.timestamp.desc()).limit(limit)
                results = query.all()

            data_list = [r.to_dict() for r in results]
            self.logger.info(f"Fetched {len(results)} recent alerts from DB.")
            self.cache.set(cache_key, data_list, ttl=30) # Cache for 30 seconds
            return data_list
        except Exception as e:
            self.logger.error(f"Get alerts error: {e}", exc_info=True)
            return []

    def acknowledge_alert_db(self, alert_id: str, user_id: str = "system") -> bool:
        """Acknowledge an alert in the database."""
        try:
            with get_session() as session:
                alert = session.query(Alert).filter(Alert.id == alert_id).first()
                if alert:
                    if not alert.acknowledged:
                        alert.acknowledged = True
                        alert.metadata = json.dumps({ # Add ack info to metadata
                            **(json.loads(alert.metadata) if alert.metadata else {}),
                            "acknowledged_by": user_id,
                            "acknowledged_at": datetime.utcnow().isoformat()
                        })
                        self.logger.info(f"Alert {alert_id} acknowledged by {user_id} in DB.")
                        # Invalidate cache
                        self.cache.clear_pattern("alerts:*")
                        return True
                    else:
                        self.logger.warning(f"Alert {alert_id} was already acknowledged.")
                        return False # Or True, depending on desired idempotency behavior
                else:
                    self.logger.warning(f"Alert {alert_id} not found for acknowledgment.")
                    return False
        except Exception as e:
            self.logger.error(f"DB Acknowledge error for {alert_id}: {e}", exc_info=True)
            return False

    def create_user_sqlalchemy(self, username: str, password: str, email: Optional[str] = None) -> bool:
        """Creates a new user with SQLAlchemy"""
        if not username or len(username) < 3:
            self.logger.warning("User creation failed: Invalid username.")
            return False
        if not password or len(password) < 8:
            self.logger.warning("User creation failed: Invalid password length.")
            return False
        if email and not re.match(r"[^@]+@[^@]+\.[^@]+", email):
             self.logger.warning("User creation failed: Invalid email format.")
             return False

        password_hash = generate_password_hash(password)
        try:
            with get_session() as session:
                # Check if user already exists (case-insensitive check might be better)
                existing_user = session.query(User).filter(
                    (func.lower(User.username) == func.lower(username)) |
                    (func.lower(User.email) == func.lower(email) if email else False)
                ).first()

                if existing_user:
                    self.logger.warning(f"User creation failed: Username '{username}' or email '{email}' already exists.")
                    return False

                new_user = User(username=username, password_hash=password_hash, email=email)
                session.add(new_user)
                # Commit happens automatically by context manager

            self.logger.info(f"User '{username}' created successfully via SQLAlchemy.")
            # Consider adding audit log here if SecureDatabaseManager isn't used for users
            return True
        except SQLAlchemyError as e:
            self.logger.error(f"Error creating user '{username}' via SQLAlchemy: {e}", exc_info=True)
            return False

    def authenticate_user_sqlalchemy(self, username: str, password: str) -> Optional[User]:
        """Authenticates a user using SQLAlchemy, returns User object on success."""
        try:
            with get_session() as session:
                # Case-insensitive username lookup might be desirable
                user = session.query(User).filter(func.lower(User.username) == func.lower(username)).first()

            if user and check_password_hash(user.password_hash, password):
                self.logger.info(f"User '{username}' authenticated successfully via SQLAlchemy.")
                # Consider adding audit log here
                return user # Return the user object

            self.logger.warning(f"Authentication failed for user '{username}' via SQLAlchemy.")
            # Consider adding audit log here
            return None
        except SQLAlchemyError as e:
            self.logger.error(f"Error authenticating user '{username}' via SQLAlchemy: {e}", exc_info=True)
            return None


    # ==================== ROUTES ====================

    def _setup_routes(self):
        """Setup all Flask routes"""
        self.logger.info("Setting up Flask routes...")

        # Apply limiter globally or per route
        limiter = self.limiter

        # ==================== AUTH ENDPOINTS ====================

        @self.app.route('/api/auth/register', methods=['POST'])
        @limiter.limit("5 per hour") # Stricter limit for registration
        def register():
            data = request.get_json()
            if not data:
                return jsonify({"error": "Invalid JSON request"}), 400

            username = data.get('username')
            password = data.get('password')
            email = data.get('email')

            # Basic Validation (more robust validation needed - Item 9)
            if not username or not password or not email:
                return jsonify({"error": "Username, password, and email required"}), 400
            # Add more specific validation here (length, format etc.)

            if self.create_user_sqlalchemy(username, password, email):
                return jsonify({"message": "User registered successfully"}), 201
            else:
                # Error logged in create_user_sqlalchemy
                return jsonify({"error": "Registration failed. Username or email may already exist."}), 409


        @self.app.route('/api/auth/login', methods=['POST'])
        @limiter.limit("10 per minute")
        def login():
            data = request.get_json()
            if not data:
                return jsonify({"error": "Invalid JSON request"}), 400

            username = data.get('username')
            password = data.get('password')

            if not username or not password:
                return jsonify({"error": "Username and password required"}), 400

            user = self.authenticate_user_sqlalchemy(username, password)

            if not user:
                return jsonify({"error": "Invalid credentials"}), 401

            # Identity can be username or user ID
            access_token = create_access_token(identity=user.username) # Or user.id
            response = jsonify(access_token=access_token, user=user.to_dict())

            # Set HttpOnly cookie (optional, depends on frontend needs)
            # set_access_cookies(response, access_token) # Needs import

            return response, 200

        @self.app.route('/api/auth/logout', methods=['POST'])
        # @jwt_required() # Optional: Require token to logout for session invalidation
        def logout():
            response = jsonify({"message": "Logout successful"})
            unset_jwt_cookies(response) # Clear cookies if used
            # Add token to blocklist if using JWT blocklisting
            return response, 200

        @self.app.route('/api/auth/me')
        @jwt_required()
        @limiter.limit("120 per minute")
        def protected():
            # Get the identity from the token
            current_user_identity = get_jwt_identity()
            # Optionally fetch full user details from DB based on identity
            # user = User.query.filter_by(username=current_user_identity).first()
            # return jsonify(user.to_dict()) if user else jsonify(logged_in_as=current_user_identity)
            return jsonify(logged_in_as=current_user_identity)

        # ==================== PAGE ROUTES ====================
        # Serve static frontend files (if not using Nginx)
        # These might be unnecessary if Nginx serves the React build directly

        @self.app.route('/', defaults={'path': ''})
        @self.app.route('/<path:path>')
        def serve_react_app(path):
             static_folder = self.app.static_folder
             if static_folder is None:
                  logger.error("Static folder not configured.")
                  return "Static folder not found", 404

             if path != "" and os.path.exists(os.path.join(static_folder, path)):
                  return send_from_directory(static_folder, path)
             else:
                  index_path = os.path.join(static_folder, 'index.html')
                  if not os.path.exists(index_path):
                       logger.error("index.html not found in static folder.")
                       return "Frontend entry point not found", 404
                  return send_from_directory(static_folder, 'index.html')

        # ==================== API ENDPOINTS ====================

        @self.app.route('/health')
        @limiter.exempt # No rate limit for health check
        def health_check():
            """Basic health check endpoint"""
            db_ok = False
            redis_ok = self.cache.available if self.cache else False
            try:
                # Check DB connection
                with engine.connect() as connection:
                    db_ok = True
            except Exception as e:
                logger.error(f"Health check DB connection failed: {e}")
                db_ok = False

            status = {
                'status': 'healthy' if db_ok and redis_ok else 'partial',
                'timestamp': datetime.now().isoformat(),
                'version': '3.2.0', # Hardcoded for now
                'checks': {
                    'redis': 'ok' if redis_ok else 'unavailable',
                    'postgresql': 'ok' if db_ok else 'unavailable',
                    'celery': 'ok' if CELERY_AVAILABLE else 'unavailable', # Basic check
                },
                'uptime': str(datetime.now() - self.start_time).split('.')[0]
            }
            status_code = 200 if status['status'] == 'healthy' else 503
            return jsonify(status), status_code

        @self.app.route('/metrics')
        @limiter.exempt # Expose metrics for Prometheus
        def prometheus_metrics():
            """Expose metrics for Prometheus scraping."""
            # Implementation depends on the chosen Prometheus library (Item 4)
            # Example using prometheus_client:
            # from prometheus_client import generate_latest, REGISTRY, Counter
            # c = Counter('my_failures', 'Description of counter')
            # c.inc() # Increment counter
            # return Response(generate_latest(REGISTRY), mimetype='text/plain')
            return jsonify({"message": "Prometheus metrics endpoint (not fully implemented)"}), 200


        @self.app.route('/api/dashboard') # Changed endpoint name for clarity
        #@jwt_required() # Protect endpoint
        @limiter.limit("60 per minute")
        def get_dashboard_data():
            """Get combined dashboard data"""
            # This endpoint combines data needed for the main dashboard view
            try:
                # Use cache for the combined dashboard data
                cache_key = 'dashboard_combined_data'
                cached = self.cache.get(cache_key)
                if cached:
                    return jsonify(cached)

                # Fetch components
                latest_devices_df = self.get_latest_device_data(limit=500) # Get more devices for overview
                recent_alerts = self.get_recent_alerts(limit=20, acknowledged=False) # Get unacknowledged alerts

                # --- Calculate Overview Metrics ---
                overview_data = {}
                if not latest_devices_df.empty:
                     total_devices = latest_devices_df['device_id'].nunique()
                     active_devices = latest_devices_df[latest_devices_df['status'] != 'offline'].shape[0] # Assuming 'offline' status exists
                     # Handle potential NaN scores before calculating mean
                     avg_health = latest_devices_df['health_score'].dropna().mean() * 100 if not latest_devices_df['health_score'].dropna().empty else 0
                     avg_efficiency = latest_devices_df['efficiency_score'].dropna().mean() * 100 if not latest_devices_df['efficiency_score'].dropna().empty else 0

                     # Simplified status distribution
                     status_counts = latest_devices_df['status'].value_counts().to_dict()
                     status_distribution = {
                         'normal': status_counts.get('normal', 0),
                         'warning': status_counts.get('warning', 0),
                         'critical': status_counts.get('critical', 0),
                         'offline': status_counts.get('offline', 0)
                     }
                     # Energy usage (example, needs proper calculation based on data)
                     energy_usage = latest_devices_df[latest_devices_df['device_type']=='power_meter']['value'].sum() / 1000 # Example kW

                     overview_data = {
                         'systemHealth': round(avg_health),
                         'activeDevices': active_devices,
                         'totalDevices': total_devices,
                         'efficiency': round(avg_efficiency),
                         'energyUsage': round(energy_usage, 1),
                         'energyCost': round(energy_usage * 24 * 0.12), # Example daily cost
                         'statusDistribution': status_distribution,
                         'timestamp': datetime.utcnow().isoformat() # Use UTC
                     }

                # --- Combine Data ---
                dashboard_payload = {
                    **overview_data,
                    'devices': latest_devices_df.to_dict('records')[:100], # Limit devices sent initially
                    'alerts': recent_alerts,
                    # Add performance data if generated/fetched separately
                    'performanceData': self._generate_dummy_performance_data(overview_data.get('systemHealth', 80), overview_data.get('efficiency', 85)) # Placeholder
                }

                self.cache.set(cache_key, dashboard_payload, ttl=30) # Cache for 30 seconds
                return jsonify(dashboard_payload)

            except Exception as e:
                self.logger.error(f"Error fetching dashboard data: {e}", exc_info=True)
                return jsonify({"error": "Failed to retrieve dashboard data"}), 500

        # Helper for dummy performance data
        def _generate_dummy_performance_data(base_health, base_efficiency, points=24):
            data = []
            now = datetime.utcnow()
            for i in range(points):
                ts = now - timedelta(hours=(points - 1 - i))
                health = max(0, min(100, base_health + np.random.normal(0, 3)))
                efficiency = max(0, min(100, base_efficiency + np.random.normal(0, 2)))
                data.append({
                    'timestamp': ts.strftime('%H:00'), # Hour format
                    'systemHealth': round(health),
                    'efficiency': round(efficiency)
                })
            return data
        self.app._generate_dummy_performance_data = _generate_dummy_performance_data


        @self.app.route('/api/devices')
        #@jwt_required()
        @limiter.limit("120 per minute")
        def get_devices():
            """Get latest status for all devices"""
            try:
                limit = request.args.get('limit', default=1000, type=int)
                # Fetching from DB via SQLAlchemy method (which includes caching)
                devices_df = self.get_latest_device_data(limit=limit)
                return jsonify(devices_df.to_dict('records'))
            except Exception as e:
                self.logger.error(f"Error getting devices: {e}", exc_info=True)
                return jsonify({"error": "Failed to retrieve devices"}), 500


        @self.app.route('/api/devices/<string:device_id>') # Use string type hint
        #@jwt_required()
        @limiter.limit("120 per minute")
        def get_device(device_id):
            """Get latest status for a specific device"""
            cache_key = f'device_latest:{device_id}'
            cached = self.cache.get(cache_key)
            if cached: return jsonify(cached)

            try:
                with get_session() as session:
                    device_data = session.query(DeviceData)\
                        .filter(DeviceData.device_id == device_id)\
                        .order_by(DeviceData.timestamp.desc())\
                        .first()

                if device_data:
                    data_dict = device_data.to_dict()
                    self.cache.set(cache_key, data_dict, ttl=60)
                    return jsonify(data_dict)
                else:
                    return jsonify({"error": "Device not found"}), 404
            except Exception as e:
                self.logger.error(f"Error getting device {device_id}: {e}", exc_info=True)
                return jsonify({"error": "Failed to retrieve device data"}), 500


        @self.app.route('/api/alerts')
        #@jwt_required()
        @limiter.limit("120 per minute")
        def get_alerts():
            """Get recent alerts"""
            try:
                limit = request.args.get('limit', default=50, type=int)
                severity = request.args.get('severity', default=None, type=str)
                acknowledged_str = request.args.get('acknowledged', default=None, type=str)

                acknowledged_filter: Optional[bool] = None
                if acknowledged_str is not None:
                    acknowledged_filter = acknowledged_str.lower() == 'true'

                # Use the DB method (which includes caching)
                alerts = self.get_recent_alerts(limit=limit, severity=severity, acknowledged=acknowledged_filter)
                return jsonify(alerts)
            except Exception as e:
                self.logger.error(f"Error getting alerts: {e}", exc_info=True)
                return jsonify({"error": "Failed to retrieve alerts"}), 500


        @self.app.route('/api/alerts/acknowledge/<string:alert_id>', methods=['POST'])
        #@jwt_required()
        @limiter.limit("30 per minute")
        def acknowledge_alert_api(alert_id):
             #user_id = get_jwt_identity() # Get user from JWT
             user_id = "temp_user" # Placeholder until JWT is fully enforced
             success = self.acknowledge_alert_db(alert_id, user_id)
             if success:
                 # Optionally fetch the updated alert to return
                 with get_session() as session:
                      alert = session.query(Alert).filter(Alert.id == alert_id).first()
                 updated_alert_data = alert.to_dict() if alert else None

                 # Notify WebSocket clients about acknowledgment
                 if self.socketio:
                     self.socketio.emit('alert_acknowledged', {
                         'alertId': alert_id,
                         'acknowledgedBy': user_id,
                         'timestamp': datetime.utcnow().isoformat()
                     }, room='alerts') # Assuming an 'alerts' room exists

                 return jsonify({"message": "Alert acknowledged", "alert": updated_alert_data}), 200
             else:
                 # Error logged in acknowledge_alert_db
                 return jsonify({"error": "Failed to acknowledge alert or alert not found"}), 404 # Or 500 if internal error

        # ... (Add other endpoints: /predictions, /health_scores, /recommendations etc.)
        # ... (Remember to apply @jwt_required() and @limiter decorators)
        # ... (Refactor data fetching to use SQLAlchemy sessions via get_session())

        # Placeholder for other required endpoints from the Action Plan
        @self.app.route('/api/predictions')
        #@jwt_required()
        @limiter.limit("60 per minute")
        def get_predictions():
            # TODO: Implement using self.analytics_engine and SQLAlchemy data
            device_id = request.args.get('device_id')
            if not device_id: return jsonify({"error": "device_id query parameter is required"}), 400
            # Fetch data for device, run prediction, cache results
            return jsonify({"message": f"Predictions for {device_id} (not implemented)", "device_id": device_id}), 501

        @self.app.route('/api/health_scores')
        #@jwt_required()
        @limiter.limit("60 per minute")
        def get_health_scores():
            # TODO: Implement using self.health_calculator and SQLAlchemy data
            # Fetch latest data for all devices, calculate scores, cache results
             return jsonify({"message": "Health scores (not implemented)"}), 501

        @self.app.route('/api/recommendations')
        #@jwt_required()
        @limiter.limit("30 per minute")
        def get_recommendations():
             # TODO: Implement using self.recommendation_engine, health data, patterns
             return jsonify({"message": "Recommendations (not implemented)"}), 501

        @self.app.route('/api/system_metrics')
        #@jwt_required()
        @limiter.limit("120 per minute")
        def get_system_metrics_api():
             # TODO: Fetch real system metrics (e.g., using psutil) or from DB if stored
             return jsonify(self._get_system_metrics()) # Use internal helper

        # Example Celery Task Trigger Route
        @self.app.route('/api/tasks/start_report')
        #@jwt_required()
        @limiter.limit("5 per hour")
        def start_report_task():
            if not self.celery:
                return jsonify({"error": "Async task runner not available"}), 503
            try:
                # Assuming generate_report_task is defined below or imported
                task = generate_report_task.delay()
                logger.info(f"Dispatched generate_report_task with ID: {task.id}")
                return jsonify({"task_id": task.id, "status": "PENDING"}), 202
            except Exception as e:
                logger.error(f"Failed to dispatch Celery task: {e}", exc_info=True)
                return jsonify({"error": "Failed to start report generation task"}), 500

        @self.app.route('/api/tasks/status/<string:task_id>')
        #@jwt_required()
        @limiter.limit("60 per minute")
        def get_task_status(task_id):
            if not self.celery:
                return jsonify({"error": "Async task runner not available"}), 503
            try:
                # task_result = self.celery.AsyncResult(task_id) # Older Celery versions
                task_result = generate_report_task.AsyncResult(task_id) # Access via task object
                response = {
                    'task_id': task_id,
                    'status': task_result.status,
                    'result': task_result.result if task_result.ready() else None
                }
                if task_result.failed():
                    # Be careful about exposing raw tracebacks
                    response['error'] = str(task_result.result)
                    logger.warning(f"Task {task_id} failed: {task_result.traceback}")

                return jsonify(response)
            except Exception as e:
                logger.error(f"Failed to get task status for {task_id}: {e}", exc_info=True)
                return jsonify({"error": "Failed to retrieve task status"}), 500


        # Serve generated reports/exports (adjust paths as needed)
        @self.app.route('/reports/<path:filename>')
        # @jwt_required() # Optional: Protect access to reports
        def serve_report(filename):
            reports_dir = Path(get_optional_env('REPORTS_DIR', 'REPORTS/generated'))
            logger.debug(f"Attempting to serve report: {reports_dir / filename}")
            if not reports_dir.is_dir():
                 logger.error(f"Reports directory not found: {reports_dir}")
                 return "Reports directory configuration error", 500
            try:
                return send_from_directory(reports_dir, filename, as_attachment=False)
            except FileNotFoundError:
                 logger.warning(f"Report file not found: {filename}")
                 return "Report not found", 404

        @self.app.route('/exports/<path:filename>')
        # @jwt_required() # Optional: Protect access to exports
        def serve_export(filename):
            exports_dir = Path(get_optional_env('EXPORTS_DIR', 'EXPORTS'))
            logger.debug(f"Attempting to serve export: {exports_dir / filename}")
            if not exports_dir.is_dir():
                 logger.error(f"Exports directory not found: {exports_dir}")
                 return "Exports directory configuration error", 500
            try:
                return send_from_directory(exports_dir, filename, as_attachment=True) # Force download
            except FileNotFoundError:
                 logger.warning(f"Export file not found: {filename}")
                 return "Export not found", 404

        self.logger.info("Flask routes setup completed.")

    # ==================== WEBSOCKET EVENTS ====================

    def _setup_websocket_events(self):
        """Setup optimized WebSocket events"""
        if not self.socketio:
            self.logger.warning("SocketIO not initialized, skipping WebSocket event setup.")
            return

        @self.socketio.on('connect')
        def handle_connect():
            # Authentication should ideally happen here using token from connect query/headers
            client_id = request.sid
            self.connected_clients[client_id] = {
                'connected_at': datetime.utcnow(),
                'last_ping': datetime.utcnow(),
                'identity': 'anonymous', # TODO: Replace with authenticated identity
                'subscriptions': set()
            }
            logger.info(f"Client connected: {client_id}. Total: {len(self.connected_clients)}")
            # Send initial data if available
            initial_data = self.cache.get('dashboard_combined_data')
            if initial_data:
                emit('dashboard_update', initial_data) # Send full initial state


        @self.socketio.on('disconnect')
        def handle_disconnect():
            client_id = request.sid
            if client_id in self.connected_clients:
                identity = self.connected_clients[client_id].get('identity', 'anonymous')
                subs = self.connected_clients[client_id].get('subscriptions', set())
                # Clean up room memberships
                for room in list(subs): # Iterate over a copy
                     self._leave_room_internal(client_id, room)

                del self.connected_clients[client_id]
                logger.info(f"Client {client_id} (User: {identity}) disconnected. Total: {len(self.connected_clients)}")


        @self.socketio.on('ping_from_client') # Use a custom event to avoid conflicts
        def handle_ping():
            client_id = request.sid
            if client_id in self.connected_clients:
                self.connected_clients[client_id]['last_ping'] = datetime.utcnow()
                emit('pong_from_server', {'timestamp': datetime.utcnow().isoformat()})


        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            client_id = request.sid
            if client_id not in self.connected_clients: return # Ignore unsubscribed clients
            try:
                room_name = data.get('room')
                if room_name:
                    self._join_room_internal(client_id, room_name)
                    emit('subscribed', {'room': room_name})
                    logger.info(f"Client {client_id} subscribed to room '{room_name}'")
                else:
                    emit('error', {'message': 'Room name missing'})
            except Exception as e:
                logger.error(f"Subscription error for {client_id}: {e}", exc_info=True)
                emit('error', {'message': 'Subscription failed'})


        @self.socketio.on('unsubscribe')
        def handle_unsubscribe(data):
            client_id = request.sid
            if client_id not in self.connected_clients: return
            try:
                room_name = data.get('room')
                if room_name:
                    self._leave_room_internal(client_id, room_name)
                    emit('unsubscribed', {'room': room_name})
                    logger.info(f"Client {client_id} unsubscribed from room '{room_name}'")
                else:
                    emit('error', {'message': 'Room name missing'})
            except Exception as e:
                logger.error(f"Unsubscription error for {client_id}: {e}", exc_info=True)
                emit('error', {'message': 'Unsubscription failed'})

        self.logger.info("WebSocket event handlers setup.")

    # Internal helpers for room management
    def _join_room_internal(self, client_id, room_name):
         join_room(room_name, sid=client_id)
         self.rooms[room_name].add(client_id)
         if client_id in self.connected_clients:
             self.connected_clients[client_id]['subscriptions'].add(room_name)

    def _leave_room_internal(self, client_id, room_name):
         leave_room(room_name, sid=client_id)
         if room_name in self.rooms:
              self.rooms[room_name].discard(client_id)
              if not self.rooms[room_name]: # Clean up empty room set
                   del self.rooms[room_name]
         if client_id in self.connected_clients:
              if 'subscriptions' in self.connected_clients[client_id]:
                  self.connected_clients[client_id]['subscriptions'].discard(room_name)


    # ==================== BACKGROUND TASKS ====================

    def _start_background_tasks(self):
        """Start background tasks (Data generation, cleanup, etc.)"""
        self.logger.info("Starting background tasks...")

        # --- Data Generation/Update Task ---
        # Runs in a separate thread via SocketIO's background task runner
        def data_update_task():
            logger.info("Background data update task started.")
            last_cache_update = time.monotonic()
            while True:
                try:
                    # Generate or fetch new data (using data_generator or db reads)
                    if self.data_generator:
                         # Generate a small batch simulating new readings
                         new_data_list = []
                         # Assume self.data_generator needs setup_simulation called first
                         if not getattr(self.data_generator, 'devices', None):
                             self.data_generator.setup_simulation(device_count=25)

                         # Generate one reading per simulated device
                         sim_time = datetime.utcnow() # Use current time
                         for device_meta in self.data_generator.devices:
                             # The step index isn't critical here, just use 0
                             reading = self.data_generator._generate_single_reading(device_meta, sim_time, 0)
                             new_data_list.append(reading)

                         if new_data_list:
                              # Insert into DB
                              self.bulk_insert_device_data(new_data_list)
                              latest_df = pd.DataFrame(new_data_list) # Use the newly generated data
                         else:
                              latest_df = pd.DataFrame()

                    else:
                         # Fallback if no generator: fetch latest from DB
                         latest_df = self.get_latest_device_data(limit=100)

                    if not latest_df.empty:
                        # --- Check for Alerts ---
                        if self.alert_manager:
                             self._check_and_send_alerts(latest_df)

                        # --- Update Dashboard Cache and Broadcast (less frequently) ---
                        current_time = time.monotonic()
                        if current_time - last_cache_update > 5: # Update cache/broadcast every 5s
                             logger.debug("Updating dashboard cache and broadcasting...")
                             dashboard_payload = self._create_dashboard_payload(latest_df) # Use helper
                             self.cache.set('dashboard_combined_data', dashboard_payload, ttl=30)
                             self.cache.set('latest_devices:500', latest_df.to_dict('records'), ttl=60) # Update device cache too

                             # Broadcast the full update
                             if self.socketio:
                                  self.socketio.emit('dashboard_update', dashboard_payload, room='dashboard_updates') # Use a specific room
                             last_cache_update = current_time
                        else:
                             logger.debug("Skipping cache update/broadcast (too soon).")

                    # Control loop speed
                    eventlet.sleep(float(get_optional_env('DATA_UPDATE_INTERVAL', 1))) # Default 1 second

                except Exception as e:
                    logger.error(f"Error in data update task: {e}", exc_info=True)
                    eventlet.sleep(10) # Wait longer after an error

        if self.socketio:
            self.socketio.start_background_task(data_update_task)
            # Add other background tasks (like cleanup) here if needed using socketio.start_background_task
            self._setup_periodic_tasks() # Setup APScheduler or Celery Beat tasks
        else:
            logger.warning("SocketIO not available, cannot start background tasks.")


    def _setup_periodic_tasks(self):
         """Setup tasks managed by APScheduler or Celery Beat."""
         # For now, using APScheduler as Celery Beat requires separate process/config
         if self.scheduler:
              if not self.scheduler.get_job('cleanup_clients'):
                   self.scheduler.add_job(
                        id='cleanup_clients',
                        func=self._cleanup_inactive_clients,
                        trigger='interval',
                        minutes=5
                   )
                   logger.info("Scheduled inactive client cleanup task.")

              # Add model retraining task if analytics engine is available
              if self.analytics_engine and hasattr(self.analytics_engine, 'retrain_models'):
                   # Check if job exists before adding
                   if not self.scheduler.get_job('model_retraining'):
                       self.scheduler.add_job(
                           id='model_retraining',
                           func=self._run_model_retraining, # Wrapper to run in app context
                           trigger='interval',
                           hours=int(get_optional_env('RETRAIN_INTERVAL_HOURS', 24)) # Default daily
                       )
                       logger.info("Scheduled model retraining task.")

              if not self.scheduler.running:
                   self.scheduler.start()
                   logger.info("APScheduler started.")
         else:
              logger.warning("APScheduler not available. Periodic tasks disabled.")


    def _run_model_retraining(self):
        """Wrapper to run model retraining within the Flask app context."""
        if not self.analytics_engine or not hasattr(self.analytics_engine, 'retrain_models'):
             logger.warning("Analytics engine or retrain_models method not available.")
             return
        with self.app.app_context():
             logger.info("Running scheduled model retraining...")
             try:
                  # The retrain_models method in analytics engine should handle data fetching
                  result = self.analytics_engine.retrain_models()
                  logger.info(f"Model retraining completed. Status: {result.get('status')}")
             except Exception as e:
                  logger.error(f"Scheduled model retraining failed: {e}", exc_info=True)

    def _cleanup_inactive_clients(self):
        """Cleanup inactive WebSocket clients."""
        with self.app.app_context(): # Ensure context for logging etc.
             now = datetime.utcnow()
             timeout = timedelta(minutes=int(get_optional_env('CLIENT_TIMEOUT_MINUTES', 5)))
             disconnected_count = 0
             inactive_sids = []

             # Identify inactive clients
             for sid, client_info in self.connected_clients.items():
                  if now - client_info.get('last_ping', now) > timeout:
                       inactive_sids.append(sid)

             # Disconnect them
             for sid in inactive_sids:
                  logger.warning(f"Disconnecting inactive client {sid}...")
                  if self.socketio:
                       self.socketio.disconnect(sid, silent=True) # silent=True suppresses disconnect event
                  # The handle_disconnect event should clean up rooms and self.connected_clients
                  disconnected_count += 1

             if disconnected_count > 0:
                  logger.info(f"Cleaned up {disconnected_count} inactive client(s).")


    # ==================== HELPER METHODS ====================

    def _create_dashboard_payload(self, latest_devices_df: pd.DataFrame) -> Dict:
        """Helper to create the consistent dashboard data payload."""
        overview_data = {}
        if not latest_devices_df.empty:
            total_devices = latest_devices_df['device_id'].nunique()
            active_devices = latest_devices_df[latest_devices_df['status'] != 'offline'].shape[0]
            avg_health = latest_devices_df['health_score'].dropna().mean() * 100 if not latest_devices_df['health_score'].dropna().empty else 0
            avg_efficiency = latest_devices_df['efficiency_score'].dropna().mean() * 100 if not latest_devices_df['efficiency_score'].dropna().empty else 0
            status_counts = latest_devices_df['status'].value_counts().to_dict()
            status_distribution = { 'normal': status_counts.get('normal', 0), 'warning': status_counts.get('warning', 0), 'critical': status_counts.get('critical', 0), 'offline': status_counts.get('offline', 0) }
            energy_usage = latest_devices_df[latest_devices_df['device_type']=='power_meter']['value'].sum() / 1000

            overview_data = {
                'systemHealth': round(avg_health),
                'activeDevices': active_devices,
                'totalDevices': total_devices,
                'efficiency': round(avg_efficiency),
                'energyUsage': round(energy_usage, 1),
                'energyCost': round(energy_usage * 24 * 0.12),
                'statusDistribution': status_distribution,
                'timestamp': datetime.utcnow().isoformat()
            }

        # Fetch recent alerts (cached)
        recent_alerts = self.get_recent_alerts(limit=20, acknowledged=False)

        return {
            **overview_data,
            'devices': latest_devices_df.to_dict('records')[:100],
            'alerts': recent_alerts,
            'performanceData': self.app._generate_dummy_performance_data(overview_data.get('systemHealth', 80), overview_data.get('efficiency', 85))
        }


    def _check_and_send_alerts(self, devices_df: pd.DataFrame):
        """Check for alerts and broadcast via WebSocket"""
        if not self.alert_manager or not self.socketio:
            return

        all_triggered_alerts = []
        try:
            # Iterate through the latest data for each device
            for _, device_row in devices_df.iterrows():
                device_data = device_row.to_dict()
                # Ensure timestamp is string for AlertManager if it expects it
                if isinstance(device_data.get('timestamp'), datetime):
                    device_data['timestamp'] = device_data['timestamp'].isoformat()

                triggered = self.alert_manager.evaluate_conditions(
                    data=device_data,
                    device_id=device_data.get('device_id')
                )
                if triggered:
                    all_triggered_alerts.extend(triggered)

            if all_triggered_alerts:
                # Insert alerts into DB (bulk insert might be better here)
                alerts_inserted = 0
                for alert in all_triggered_alerts:
                    if self.insert_alert(alert):
                         alerts_inserted += 1
                if alerts_inserted > 0:
                     logger.info(f"Inserted {alerts_inserted} new alerts into DB.")

                # Broadcast only new, unacknowledged alerts via WebSocket
                logger.info(f"Broadcasting {len(all_triggered_alerts)} triggered alerts...")
                for alert in all_triggered_alerts:
                    # Make sure timestamp is ISO format string for JSON
                    if isinstance(alert.get('timestamp'), datetime):
                        alert['timestamp'] = alert['timestamp'].isoformat()
                    self.socketio.emit('alert_update', alert, room='alerts') # Use 'alerts' room

        except Exception as e:
            logger.error(f"Error checking/sending alerts: {e}", exc_info=True)


    def _get_system_metrics(self) -> Dict:
        """Get system metrics using psutil if available"""
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'cpu_percent': None,
            'memory_percent': None,
            'disk_percent': None,
            'active_connections': len(self.connected_clients),
            'cache_available': self.cache.available if self.cache else False,
            'database_available': self.db_available,
            'celery_available': CELERY_AVAILABLE
        }
        try:
            import psutil
            metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            metrics['memory_percent'] = psutil.virtual_memory().percent
            metrics['disk_percent'] = psutil.disk_usage('/').percent
        except ImportError:
            logger.warning("psutil not installed, cannot get detailed system metrics.")
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
        return metrics

    # --- Error Handlers ---
    def _setup_error_handlers(self):
        """Centralized error handler setup."""
        app = self.app
        limiter = self.limiter

        @app.errorhandler(404)
        def not_found_error(error):
            logger.warning(f"404 Not Found: {request.url}")
            if request.path.startswith('/api/'):
                return jsonify(error="Resource not found"), 404
            # For non-API routes, potentially serve index.html for SPA routing
            static_folder = app.static_folder
            if static_folder and os.path.exists(os.path.join(static_folder, 'index.html')):
                 return send_from_directory(static_folder, 'index.html')
            return "Not Found", 404

        @app.errorhandler(500)
        def internal_error(error):
            logger.error(f"500 Internal Server Error: {error}", exc_info=True)
            return jsonify(error="Internal server error"), 500

        @app.errorhandler(429)
        def ratelimit_handler(e):
            logger.warning(f"Rate limit exceeded for {request.remote_addr}: {e.description}")
            return jsonify(error="Rate limit exceeded", description=str(e.description)), 429

        @app.errorhandler(SQLAlchemyError)
        def handle_db_exception(e):
            logger.error(f"Database error: {e}", exc_info=True)
            # Potentially rollback session if not handled by @get_session
            # session = SessionLocal()
            # session.rollback()
            # SessionLocal.remove()
            return jsonify(error="Database operation failed"), 500

        @app.errorhandler(Exception)
        def handle_exception(e):
            # Log general exceptions
            logger.error(f"Unhandled exception: {e}", exc_info=True)
            return jsonify(error="An unexpected error occurred"), 500

        logger.info("Error handlers registered.")


    def run(self, host='0.0.0.0', port=5000):
        """Run the application"""
        effective_port = int(get_optional_env('PORT', port))
        effective_host = get_optional_env('HOST', host)
        debug_mode = self.app.config.get('DEBUG', False)

        self.start_time = datetime.now()
        logger.info(f"Starting Digital Twin App v3.2 on {effective_host}:{effective_port} [Debug={debug_mode}]")
        logger.info(f"DB: {'OK' if self.db_available else 'FAIL'}, Redis: {'OK' if self.cache.available else 'FAIL'}, Celery: {'OK' if CELERY_AVAILABLE else 'FAIL'}")

        if not self.socketio:
            logger.critical("SocketIO failed to initialize. Cannot run.")
            return

        try:
            # Use eventlet or gevent runner based on async_mode
            self.socketio.run(
                self.app,
                host=effective_host,
                port=effective_port,
                debug=debug_mode,
                use_reloader=debug_mode, # Enable reloader only in debug
                log_output=debug_mode   # Show Flask logs only in debug
            )
        except KeyboardInterrupt:
            logger.info("Application interrupted by user (Ctrl+C). Shutting down...")
        except Exception as e:
            logger.critical(f"Application failed to start or run: {e}", exc_info=True)
        finally:
            self._shutdown()

    def _shutdown(self):
         """Graceful shutdown procedures."""
         logger.info("Initiating graceful shutdown...")
         if self.scheduler and self.scheduler.running:
              try:
                   self.scheduler.shutdown()
                   logger.info("APScheduler shut down.")
              except Exception as e:
                   logger.error(f"Error shutting down APScheduler: {e}")
         # Add cleanup for other resources if needed (e.g., closing file handles)
         logger.info("Shutdown complete.")


# ==================== CELERY TASKS ====================
# Item 7: Define Celery tasks

@celery_app.task(bind=True, name='digital_twin.generate_report')
def generate_report_task(self):
    """Async report generation task using Celery."""
    logger.info(f"Celery task started: generate_report_task (ID: {self.request.id})")
    # Need app context to access DB and other components
    # The ContextTask base class should handle this, but explicit check is good
    flask_app = current_app._get_current_object()
    if not flask_app:
        logger.error("Flask app context not available in Celery task.")
        return {'status': 'FAILURE', 'error': 'App context missing'}

    try:
        # Create components within the task context if they depend on app context
        # or ensure they are properly shared/initialized.
        # For simplicity, assume report_generator doesn't need explicit app context here.
        report_gen = HealthReportGenerator() # Assuming this is safe to instantiate here
        if not report_gen.db_manager:
             # If db_manager wasn't available at startup, try again in task context
             report_gen.db_manager = SecureDatabaseManager() # Re-check if this needs app context

        # Fetch data within the task using SQLAlchemy session
        with get_session() as session:
             # Example query: Fetch data needed for the report
             # Adjust query as needed by the report generator
             recent_data = session.query(DeviceData)\
                 .filter(DeviceData.timestamp >= datetime.utcnow() - timedelta(days=7))\
                 .all()
             report_data_df = pd.DataFrame([d.to_dict() for d in recent_data])

        if report_data_df.empty:
            logger.warning("No data found for report generation.")
            # Still generate a report, but it might be empty
            # return {'status': 'SUCCESS', 'result': 'No data for report'} # Or fail

        # Generate the report
        report_path = report_gen.generate_comprehensive_report(data_df=report_data_df, date_range_days=7)
        logger.info(f"Report generated by Celery task: {report_path}")
        # Return results that are JSON serializable
        return {'status': 'SUCCESS', 'report_path': str(report_path)}
    except Exception as e:
        logger.error(f"Celery task generate_report_task failed: {e}", exc_info=True)
        # Record failure; Celery stores traceback
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        # Optional: raise Ignore() to not retry, or raise Retry(exc=e, countdown=60)
        raise # Re-raise exception for Celery to handle


@celery_app.task(bind=True, name='digital_twin.export_data')
def export_data_task(self, format_type='json', date_range_days=7):
    """Async data export task using Celery."""
    logger.info(f"Celery task started: export_data_task (ID: {self.request.id}) Format: {format_type}, Days: {date_range_days}")
    flask_app = current_app._get_current_object()
    if not flask_app:
        logger.error("Flask app context not available in Celery task.")
        return {'status': 'FAILURE', 'error': 'App context missing'}

    try:
        export_dir = Path(get_optional_env('EXPORTS_DIR', 'EXPORTS'))
        export_dir.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f'export_{timestamp_str}.{format_type}'
        filepath = export_dir / filename

        # Fetch data
        with get_session() as session:
             start_date = datetime.utcnow() - timedelta(days=int(date_range_days))
             data_query = session.query(DeviceData)\
                 .filter(DeviceData.timestamp >= start_date)\
                 .order_by(DeviceData.timestamp.asc())
             # Stream results for large exports? Or limit?
             results = data_query.limit(50000).all() # Limit export size for now
             data_list = [d.to_dict() for d in results]

        if not data_list:
            logger.warning("No data found for export.")
            return {'status': 'SUCCESS', 'result': 'No data to export'}

        # Export
        if format_type == 'csv':
            df = pd.DataFrame(data_list)
            # Handle nested metadata if necessary
            if 'metadata' in df.columns:
                 df['metadata'] = df['metadata'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
            df.to_csv(filepath, index=False, date_format='%Y-%m-%dT%H:%M:%S.%fZ')
        else: # Default to JSON
            with open(filepath, 'w') as f:
                json.dump(data_list, f, indent=2, default=str)

        logger.info(f"Data exported by Celery task: {filepath}")
        return {'status': 'SUCCESS', 'export_path': str(filepath), 'filename': filename}
    except Exception as e:
        logger.error(f"Celery task export_data_task failed: {e}", exc_info=True)
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        raise


# ==================== APPLICATION FACTORY & MAIN ====================

def create_app_instance() -> DigitalTwinApp:
    """Application factory function"""
    try:
        # Load .env file if present (useful for local development)
        from dotenv import load_dotenv
        load_dotenv(verbose=True)
        logger.info(".env file loaded if present.")
    except ImportError:
        logger.info(".env file not loaded (dotenv not installed). Relying on system environment variables.")

    try:
        app_instance = DigitalTwinApp()
        return app_instance
    except Exception as e:
        logger.critical(f"FATAL: Failed to create application instance: {e}", exc_info=True)
        sys.exit(1) # Exit if app creation fails critically

# Create the app instance when the module is loaded
digital_twin_app_instance = create_app_instance()
flask_app = digital_twin_app_instance.app # Expose Flask app for Gunicorn/WSGI

if __name__ == '__main__':
    # This block runs only when the script is executed directly (e.g., python enhanced_flask_app_v2.py)
    # It uses the SocketIO development server.
    # For production, use Gunicorn as specified in the Dockerfile.
    digital_twin_app_instance.run()