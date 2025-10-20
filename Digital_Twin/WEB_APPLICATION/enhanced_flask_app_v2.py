#!/usr/bin/env python3
"""
Enhanced Flask Application for Digital Twin System v3.4 (Production-Ready - Hardened)
Main web application with Redis caching, PostgreSQL via SQLAlchemy ORM,
optimized SocketIO, async task support via Celery, structured logging,
input validation, Prometheus metrics, and enhanced security.

Major Enhancements from v3.3:
- Added Role-Based Access Control (RBAC) mechanisms (Item 4)
  - User model now has a 'role'
  - JWTs now contain a 'role' claim
  - Added @role_required decorator
"""

import os
import sys
import json
import logging # Still needed for basic setup before structlog takes over
import structlog # For structured logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set # Added Set
from collections import defaultdict, deque
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
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_from_directory, Response, current_app
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
import eventlet

# Database imports
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, Index, func
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError

# Redis imports
import redis
from redis.connection import ConnectionPool

# Security imports
from flask_jwt_extended import (
    JWTManager, jwt_required, create_access_token,
    get_jwt_identity, unset_jwt_cookies, decode_token,
    set_access_cookies, verify_jwt_in_request, get_jwt # Added verify_jwt_in_request, get_jwt
)
from jwt.exceptions import ExpiredSignatureError, DecodeError, InvalidTokenError
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_wtf.csrf import CSRFProtect # For form CSRF

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

# Background Scheduler (REMOVED - Replaced by Celery Beat)
# from apscheduler.schedulers.background import BackgroundScheduler

# Celery for async tasks and scheduled tasks (Beat)
try:
    from celery import Celery, Task
    from celery.schedules import crontab # For Celery Beat scheduling
    CELERY_AVAILABLE = True
except ImportError:
    logging.warning("Celery not installed. Async tasks and scheduled tasks will run synchronously or not at all.")
    CELERY_AVAILABLE = False
    # Define dummy Celery if not available
    class Celery:
        def __init__(self, *args, **kwargs): pass
        def task(self, *args, **kwargs): return lambda f: f
        conf = type('obj', (object,), {'beat_schedule': {}})() # Dummy conf with beat_schedule
    Task = object # Dummy Task base class
    crontab = object # Dummy crontab

# --- NEW: Prometheus Metrics ---
try:
    from prometheus_flask_exporter import PrometheusMetrics
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logging.warning("prometheus-flask-exporter not installed. Prometheus metrics disabled.")
    PROMETHEUS_AVAILABLE = False
    # Dummy class
    class PrometheusMetrics:
        def __init__(self, *args, **kwargs): pass
        def init_app(self, app): pass
        def counter(self, *args, **kwargs): return lambda f: f # Dummy decorator

# --- NEW: Input Validation ---
try:
    from marshmallow import Schema, fields, ValidationError
    MARSHMALLOW_AVAILABLE = True
except ImportError:
    logging.warning("Marshmallow not installed. Input validation disabled.")
    MARSHMALLOW_AVAILABLE = False
    # Dummy classes
    class Schema: pass
    class fields:
        Str = staticmethod(lambda **kwargs: None)
        Email = staticmethod(lambda **kwargs: None)
        Int = staticmethod(lambda **kwargs: None)
        Float = staticmethod(lambda **kwargs: None)
        Bool = staticmethod(lambda **kwargs: None)
        DateTime = staticmethod(lambda **kwargs: None)
        List = staticmethod(lambda **kwargs: None)
        Nested = staticmethod(lambda **kwargs: None)
    class ValidationError(Exception): pass


# Add project root to path
# Use relative pathing to be more robust
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import custom modules (Use try-except for robustness)
try:
    from Digital_Twin.AI_MODULES.predictive_analytics_engine import PredictiveAnalyticsEngine
    from Digital_Twin.CONFIG.unified_data_generator import UnifiedDataGenerator
    from Digital_Twin.REPORTS.health_report_generator import HealthReportGenerator
    from Digital_Twin.AI_MODULES.health_score import HealthScoreCalculator
    from Digital_Twin.AI_MODULES.alert_manager import AlertManager
    from Digital_Twin.AI_MODULES.pattern_analyzer import PatternAnalyzer
    from Digital_Twin.AI_MODULES.recommendation_engine import RecommendationEngine
    from Digital_Twin.AI_MODULES.secure_database_manager import SecureDatabaseManager
except ImportError as e:
    logging.warning(f"Could not import some modules: {e}. Functionality might be limited.")
    PredictiveAnalyticsEngine = HealthScoreCalculator = AlertManager = PatternAnalyzer = RecommendationEngine = UnifiedDataGenerator = HealthReportGenerator = SecureDatabaseManager = object


# ==================== LOGGING SETUP (Structlog) ====================
# Item 2: Integrate structured logging

def setup_logging():
    """Setup structured logging using structlog"""
    log_dir = Path('LOGS')
    log_dir.mkdir(exist_ok=True)
    log_level_str = os.environ.get('LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Configure standard logging first (for libraries not using structlog)
    logging.basicConfig(
        level=log_level,
        format='%(message)s', # structlog will handle formatting
        stream=sys.stdout,
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars, # Merge context vars
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            # Choose JSON or Console renderer based on environment
            structlog.processors.JSONRenderer() if os.environ.get('LOG_FORMAT') == 'json' else structlog.dev.ConsoleRenderer()
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Get the root logger configured by structlog
    log = structlog.get_logger('DigitalTwinApp')

    # Silence overly verbose libraries (using standard logging)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    # logging.getLogger('apscheduler').setLevel(logging.WARNING) # APScheduler removed
    logging.getLogger('socketio').setLevel(logging.WARNING)
    logging.getLogger('engineio').setLevel(logging.WARNING)
    logging.getLogger('celery').setLevel(logging.INFO) # Keep celery logs visible

    log.info("Structured logging configured.", log_level=log_level_str, log_format=os.environ.get('LOG_FORMAT', 'console'))
    return log

logger = setup_logging() # Use structlog logger globally

# ==================== HELPER FUNCTIONS ====================

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
    logger.debug(f"Loaded optional env var: {var_name}", default_value=default)
    return value

# ==================== INPUT VALIDATION SCHEMAS (Marshmallow) ====================
# Item 3: Implement input validation

class RegisterSchema(Schema):
    username = fields.Str(required=True, validate=lambda s: len(s) >= 3)
    password = fields.Str(required=True, validate=lambda s: len(s) >= 8)
    email = fields.Email(required=True)

class LoginSchema(Schema):
    username = fields.Str(required=True)
    password = fields.Str(required=True)

# Decorator for input validation
def validate_json(schema: Schema):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not MARSHMALLOW_AVAILABLE:
                logger.warning("Marshmallow not installed, skipping validation.")
                return f(*args, **kwargs)

            json_data = request.get_json()
            if not json_data:
                logger.warning("Validation failed: No JSON data received.")
                return jsonify({"error": "Invalid JSON request"}), 400
            try:
                validated_data = schema.load(json_data)
                # Pass validated data to the route function if needed (e.g., via g object)
                # from flask import g
                # g.validated_data = validated_data
                return f(*args, **kwargs)
            except ValidationError as err:
                logger.warning("Validation failed", errors=err.messages, received_data=json_data)
                return jsonify({"error": "Validation failed", "messages": err.messages}), 400
            except Exception as e:
                logger.error("Unexpected error during validation", error=str(e), exc_info=True)
                return jsonify({"error": "An internal error occurred during validation"}), 500
        return wrapper
    return decorator

# --- NEW: Decorator for Role-Based Access Control ---
def role_required(required_role: str):
    """Decorator to require a specific role from the JWT."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                # First, verify JWT is present and valid
                verify_jwt_in_request()
                # Get the full token claims
                claims = get_jwt()
                # Get role, default to 'user' if claim is somehow missing
                user_role = claims.get("role", "user")

                # Check for 'admin' role, or if a specific role matches
                # This logic grants 'admin' access to everything
                if user_role == 'admin':
                    return fn(*args, **kwargs)

                if user_role != required_role:
                    logger.warning(
                        "Role access denied",
                        required_role=required_role,
                        user_role=user_role,
                        identity=claims.get('sub')
                    )
                    return jsonify({"error": "Access forbidden: Insufficient permissions"}), 403

                return fn(*args, **kwargs)
            except (ExpiredSignatureError, DecodeError, InvalidTokenError) as e:
                 logger.warning("Role verification failed: Invalid JWT", error=str(e))
                 return jsonify({"error": f"Token is invalid: {str(e)}"}), 401
            except Exception as e:
                logger.error("Error during role verification", error=str(e), exc_info=True)
                # This could be triggered if verify_jwt_in_request() fails (e.g., no token)
                return jsonify({"error": "Authentication required"}), 401
        return wrapper
    return decorator


# ==================== DATABASE MODELS (SQLAlchemy) ====================
# (No changes needed for models)
Base = declarative_base()
# ... (DeviceData, Alert models remain the same) ...
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
    # --- NEW: Add role field ---
    role = Column(String(50), nullable=False, default='user', index=True) # e.g., 'user', 'admin'

    def to_dict(self):
        """Helper method to convert model instance to dictionary, excluding password"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'role': self.role, # --- NEW: Include role ---
        }

# ==================== DATABASE SETUP (SQLAlchemy) ====================
# (No changes needed for setup)
engine = None
SessionLocal = None
# ... (init_database and get_session remain the same) ...
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
    logger.debug(f"Acquired DB session", session_id=id(session))
    try:
        yield session
        session.commit()
        logger.debug(f"Committed DB session", session_id=id(session))
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database session error (rolled back)", error=str(e), exc_info=True)
        raise
    finally:
        SessionLocal.remove()
        logger.debug(f"Removed DB session", session_id=id(session))

# ==================== REDIS CACHE MANAGER ====================
# (No changes needed for Redis)
# ... (RedisCacheManager class remains the same) ...
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
        self.logger = structlog.get_logger('RedisCacheManager') # Use structlog
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
            self.logger.info(f"Redis connected successfully", redis_url=self.redis_url)
            self.available = True
        except Exception as e:
            self.logger.error(f"Redis connection failed. Falling back to in-memory cache.", error=str(e))
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
                        self.logger.debug(f"Cache GET returned non-JSON string.", cache_key=key)
                        return data_str
                return None
            else:
                return self.memory_cache.get(key)
        except Exception as e:
            self.logger.error(f"Cache GET error", cache_key=key, error=str(e))
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
            self.logger.error(f"Cache SET error", cache_key=key, error=str(e))
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if self.available and self.client:
                return self.client.delete(key) > 0
            else:
                return self.memory_cache.pop(key, None) is not None
        except Exception as e:
            self.logger.error(f"Cache DELETE error", cache_key=key, error=str(e))
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
            self.logger.info(f"Cleared cache keys", count=deleted_count, pattern=pattern)
            return deleted_count
        except Exception as e:
            self.logger.error(f"Cache clear_pattern error", pattern=pattern, error=str(e))
            return 0

# ==================== CELERY CONFIGURATION ====================
# Item 5 & 7: Migrate to Celery + Celery Beat

celery_app = None

def make_celery(app_name=__name__):
    """Factory to create a Celery app instance."""
    # (Remains the same)
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
        logger.error(f"Failed to create Celery app", error=str(e), exc_info=True)
        return None

celery_app = make_celery()

def init_celery(app: Flask, celery: Optional[Celery]):
    """Bind Celery configuration to Flask app, add context, and add Beat schedule."""
    if not celery:
        return

    # --- NEW: Celery Beat Schedule ---
    # Replaces APScheduler tasks
    celery.conf.beat_schedule = {
        'cleanup-inactive-clients-every-5-minutes': {
            'task': 'Digital_Twin.WEB_APPLICATION.enhanced_flask_app_v2.cleanup_inactive_clients_task',
            'schedule': timedelta(minutes=5),
        },
        'retrain-models-daily': {
            'task': 'Digital_Twin.WEB_APPLICATION.enhanced_flask_app_v2.run_model_retraining_task',
            # Run daily at 1 AM UTC by default, configurable via env var
            'schedule': crontab(
                hour=int(get_optional_env('RETRAIN_HOUR_UTC', 1)),
                minute=int(get_optional_env('RETRAIN_MINUTE_UTC', 0))
            ),
        },
    }

    celery.conf.update(
        result_expires=timedelta(hours=1),
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone=get_optional_env('TZ', 'UTC'),
        enable_utc=True,
    )

    class ContextTask(Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                # Bind structlog context for the duration of the task
                with structlog.contextvars.bound_contextvars(task_id=self.request.id, task_name=self.name):
                    logger.info("Executing Celery task within Flask context.")
                    try:
                        result = self.run(*args, **kwargs)
                        logger.info("Celery task completed successfully.")
                        return result
                    except Exception as e:
                            logger.error("Celery task failed", error=str(e), exc_info=True)
                            raise # Re-raise for Celery error handling
    celery.Task = ContextTask
    app.extensions["celery"] = celery
    logger.info("Celery initialized and linked with Flask app context.", beat_schedule_keys=list(celery.conf.beat_schedule.keys()))


# ==================== MAIN APPLICATION CLASS ====================

class DigitalTwinApp:
    """Main Digital Twin Flask Application Class"""

    def __init__(self):
        self.app: Optional[Flask] = None
        self.socketio: Optional[SocketIO] = None
        self.cache: Optional[RedisCacheManager] = None
        self.analytics_engine = None
        self.health_calculator = None
        self.alert_manager = None
        self.pattern_analyzer = None
        self.recommendation_engine = None
        self.data_generator = None
        # self.scheduler: Optional[BackgroundScheduler] = None # REMOVED
        self.jwt: Optional[JWTManager] = None
        self.limiter: Optional[Limiter] = None
        self.csrf: Optional[CSRFProtect] = None # NEW: CSRF Protection
        self.metrics: Optional[PrometheusMetrics] = None # NEW: Prometheus Metrics
        self.celery: Optional[Celery] = celery_app

        # Application state
        self.connected_clients: Dict[str, Dict] = {}
        self.start_time = datetime.now()
        self.rooms: Dict[str, Set[str]] = defaultdict(set)

        self.logger = logger # Use structlog logger
        self.logger.info("Digital Twin Application v3.4 starting...")

        self._create_app()
        if not self.app:
            raise RuntimeError("Flask app creation failed.")

        self._initialize_infrastructure()
        self._initialize_modules()
        self._setup_middleware()
        self._setup_routes()
        self._setup_websocket_events()
        self._start_background_tasks() # This will mainly start Celery Beat via config now
        self._setup_error_handlers()

        self.logger.info("Application initialization complete.")

    def _create_app(self):
        """Create and configure Flask application"""
        self.app = Flask(
            __name__,
            static_folder='static',
            template_folder='templates',
            instance_relative_config=True
        )

        self.app.config.from_mapping(
            SECRET_KEY=get_required_env('SECRET_KEY'), # Used for Flask session, CSRF
            DEBUG=get_optional_env('FLASK_DEBUG', 'False').lower() == 'true',
            JWT_SECRET_KEY=get_required_env('JWT_SECRET_KEY'), # Separate key for JWT
            JWT_ACCESS_TOKEN_EXPIRES=timedelta(hours=int(get_optional_env('JWT_EXPIRY_HOURS', 24))),
            JWT_TOKEN_LOCATION=['headers', 'cookies'],
            JWT_COOKIE_SECURE=get_optional_env('FLASK_ENV', 'development') == 'production',
            JWT_COOKIE_CSRF_PROTECT=True,
            JWT_CSRF_CHECK_FORM=True,
            # NEW: Needed for Flask-WTF CSRF
            WTF_CSRF_ENABLED=True,
            WTF_CSRF_SECRET_KEY=get_required_env('SECRET_KEY'), # Can reuse Flask secret key

            DATABASE_URL=get_required_env('DATABASE_URL'),
            REDIS_URL=get_required_env('REDIS_URL'),
            CELERY_BROKER_URL=get_optional_env('CELERY_BROKER_URL'),
            CELERY_RESULT_BACKEND=get_optional_env('CELERY_RESULT_BACKEND'),
            RATELIMIT_STORAGE_URI=get_required_env('REDIS_URL'),
            RATELIMIT_STRATEGY='fixed-window',
            RATELIMIT_HEADERS_ENABLED=True,
            CORS_ALLOWED_ORIGINS=get_optional_env('CORS_ALLOWED_ORIGINS', '*').split(',') # Default to '*' if not set
        )

        self.logger.info(f"Flask app created.", environment=self.app.config['ENV'], debug=self.app.config['DEBUG'])

    def _initialize_infrastructure(self):
        """Initialize DB, Redis, Celery, JWT, Limiter, CSRF, Prometheus"""
        global engine, SessionLocal

        init_database(self.app.config['DATABASE_URL'])
        self.db_available = engine is not None and SessionLocal is not None

        self.cache = RedisCacheManager(redis_url=self.app.config['REDIS_URL'])
        if not self.cache.available:
            self.logger.warning("Running with in-memory cache fallback.")

        if self.celery:
            init_celery(self.app, self.celery)

        self.jwt = JWTManager(self.app)
        self.logger.info("JWTManager initialized.")

        if FLASK_LIMITER_AVAILABLE:
            self.limiter = Limiter(key_func=get_remote_address)
            self.limiter.init_app(self.app)
            self.logger.info("Flask-Limiter initialized.")
        else:
            self.limiter = Limiter()

        # --- NEW: CSRF Protection ---
        self.csrf = CSRFProtect(self.app)
        self.logger.info("Flask-WTF CSRF protection initialized.")

        # --- NEW: Prometheus Metrics ---
        if PROMETHEUS_AVAILABLE:
            # Basic metrics: requests by endpoint, latency by endpoint
            self.metrics = PrometheusMetrics(self.app)
            self.logger.info("Prometheus metrics initialized.")
        else:
            self.metrics = PrometheusMetrics() # Dummy

        allowed_origins = self.app.config['CORS_ALLOWED_ORIGINS']
        try:
            message_queue_url = self.app.config['REDIS_URL'] if self.cache.available else None
            if message_queue_url:
                logger.info(f"Using Redis message queue for SocketIO", url=message_queue_url)

            self.socketio = SocketIO(
                self.app,
                cors_allowed_origins=allowed_origins,
                async_mode='eventlet',
                logger=self.app.config['DEBUG'],
                engineio_logger=self.app.config['DEBUG'],
                ping_timeout=60,
                ping_interval=25,
                message_queue=message_queue_url,
                channel=get_optional_env('SOCKETIO_CHANNEL', 'digital_twin')
            )
            self.logger.info("Flask-SocketIO initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Flask-SocketIO", error=str(e), exc_info=True)


    def _initialize_modules(self):
        """Initialize AI and analytics modules"""
        # (Remains largely the same, removed scheduler init)
        self.logger.info("Initializing application modules...")
        try:
            self.analytics_engine = PredictiveAnalyticsEngine() if PredictiveAnalyticsEngine else None
            self.health_calculator = HealthScoreCalculator() if HealthScoreCalculator else None
            self.alert_manager = AlertManager() if AlertManager else None
            self.pattern_analyzer = PatternAnalyzer() if PatternAnalyzer else None
            self.recommendation_engine = RecommendationEngine() if RecommendationEngine else None
            self.data_generator = UnifiedDataGenerator(db_path=self.app.config['DATABASE_URL']) if UnifiedDataGenerator else None

            # self.scheduler = BackgroundScheduler(daemon=True) # REMOVED

            self.logger.info("Application modules initialized successfully.")
        except Exception as e:
            self.logger.error(f"Module initialization error", error=str(e), exc_info=True)


    def _setup_middleware(self):
        """Setup Flask middleware"""
        # (Remains the same)
        self.app.wsgi_app = ProxyFix(
            self.app.wsgi_app,
            x_for=int(get_optional_env('PROXY_X_FOR', 1)),
            x_proto=int(get_optional_env('PROXY_X_PROTO', 1)),
            x_host=int(get_optional_env('PROXY_X_HOST', 1)),
            x_prefix=int(get_optional_env('PROXY_X_PREFIX', 1))
        )
        self.logger.info("ProxyFix middleware applied.")

        CORS(self.app,
             origins=self.app.config['CORS_ALLOWED_ORIGINS'],
             supports_credentials=True,
             allow_headers=["Content-Type", "Authorization", "X-CSRF-TOKEN"])
        self.logger.info(f"CORS configured", origins=self.app.config['CORS_ALLOWED_ORIGINS'])

        # --- NEW: Add request logging middleware ---
        @self.app.before_request
        def log_request_info():
            # Bind request info to structlog context
            structlog.contextvars.bind_contextvars(
                remote_addr=request.remote_addr,
                method=request.method,
                path=request.path,
                endpoint=request.endpoint,
                # request_id=str(uuid.uuid4()) # Optional: add request ID
            )
            logger.info("Request received")

        @self.app.after_request
        def log_response_info(response):
            logger.info("Request completed", status_code=response.status_code)
            # Clear context vars for this request
            structlog.contextvars.clear_contextvars()
            return response

    # ==================== DATABASE OPERATIONS (SQLAlchemy) ====================
    # (No changes needed for DB operations)
    # ... (bulk_insert_device_data, get_latest_device_data, etc. remain the same) ...
    def bulk_insert_device_data(self, data_list: List[Dict]) -> bool:
        """Bulk insert device data efficiently using SQLAlchemy"""
        inserted_count = 0
        try:
            with get_session() as session:
                objects_to_insert = []
                for d in data_list:
                    # Basic validation
                    if not d.get('device_id') or not d.get('timestamp'):
                        self.logger.warning(f"Skipping invalid device data record", record=d)
                        continue

                    # Ensure timestamp is datetime
                    ts = d['timestamp']
                    if isinstance(ts, str):
                        try:
                            timestamp = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        except ValueError:
                            self.logger.warning(f"Skipping record with invalid timestamp format", timestamp=ts)
                            continue
                    elif isinstance(ts, (int, float)):
                        timestamp = datetime.fromtimestamp(ts)
                    elif not isinstance(ts, datetime):
                            self.logger.warning(f"Skipping record with invalid timestamp type", type=str(type(ts)))
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
                self.logger.info(f"Bulk inserted device data records", count=inserted_count)
            return True
        except Exception as e:
            self.logger.error(f"Bulk insert error", error=str(e), exc_info=True)
            return False

    def get_latest_device_data(self, limit: int = 100) -> pd.DataFrame:
        """Get latest device data efficiently using SQLAlchemy"""
        cache_key = f"latest_devices:{limit}"
        cached = self.cache.get(cache_key)
        if cached is not None and isinstance(cached, list): # Check type
            self.logger.debug(f"Cache hit", cache_key=cache_key)
            try:
                # Recreate DataFrame from cached dict list
                return pd.DataFrame(cached)
            except Exception as e:
                self.logger.warning(f"Failed to create DataFrame from cache", cache_key=cache_key, error=str(e))
                self.cache.delete(cache_key) # Invalidate cache

        self.logger.debug(f"Cache miss", cache_key=cache_key)
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

            self.logger.info(f"Fetched latest device data from DB", count=len(results))
            if data_list:
                self.cache.set(cache_key, data_list, ttl=60) # Cache for 1 minute
                return pd.DataFrame(data_list)
            else:
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Get latest device data error", error=str(e), exc_info=True)
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
            self.logger.info(f"Inserted alert", alert_id=alert_id)
            # Invalidate relevant alert cache
            self.cache.clear_pattern("alerts:*")
            return True
        except Exception as e:
            self.logger.error(f"Alert insert error", error=str(e), exc_info=True)
            return False

    def get_recent_alerts(self, limit: int = 10, severity: Optional[str] = None, acknowledged: Optional[bool] = None) -> List[Dict]:
        """Get recent alerts using SQLAlchemy"""
        cache_key = f"alerts:{severity}:{limit}:{acknowledged}"
        cached = self.cache.get(cache_key)
        if cached is not None and isinstance(cached, list):
            self.logger.debug(f"Cache hit", cache_key=cache_key)
            return cached

        self.logger.debug(f"Cache miss", cache_key=cache_key)
        try:
            with get_session() as session:
                query = session.query(Alert)

                if acknowledged is not None:
                    query = query.filter(Alert.acknowledged == acknowledged)

                if severity:
                    query = query.filter(Alert.severity == severity)

                query = query.order_by(Alert.timestamp.desc()).limit(limit)
                results = query.all()

            data_list = [r.to_dict() for r in results]
            self.logger.info(f"Fetched recent alerts from DB", count=len(results))
            self.cache.set(cache_key, data_list, ttl=30) # Cache for 30 seconds
            return data_list
        except Exception as e:
            self.logger.error(f"Get alerts error", error=str(e), exc_info=True)
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
                        self.logger.info(f"Alert acknowledged in DB", alert_id=alert_id, user_id=user_id)
                        # Invalidate cache
                        self.cache.clear_pattern("alerts:*")
                        return True
                    else:
                        self.logger.warning(f"Alert already acknowledged", alert_id=alert_id)
                        return False # Or True, depending on desired idempotency behavior
                else:
                    self.logger.warning(f"Alert not found for acknowledgment", alert_id=alert_id)
                    return False
        except Exception as e:
            self.logger.error(f"DB Acknowledge error", alert_id=alert_id, error=str(e), exc_info=True)
            return False

    def create_user_sqlalchemy(self, username: str, password: str, email: Optional[str] = None) -> bool:
        """Creates a new user with SQLAlchemy"""
        # Validation moved to the route using Marshmallow
        password_hash = generate_password_hash(password)
        try:
            with get_session() as session:
                existing_user = session.query(User).filter(
                    (func.lower(User.username) == func.lower(username)) |
                    (func.lower(User.email) == func.lower(email) if email else False)
                ).first()

                if existing_user:
                    self.logger.warning(f"User creation failed: Username or email already exists.", username=username, email=email)
                    return False
                
                # Role will be set to 'user' by default from the model
                new_user = User(username=username, password_hash=password_hash, email=email)
                session.add(new_user)

            self.logger.info(f"User created successfully via SQLAlchemy.", username=username)
            return True
        except SQLAlchemyError as e:
            self.logger.error(f"Error creating user via SQLAlchemy", username=username, error=str(e), exc_info=True)
            return False

    def authenticate_user_sqlalchemy(self, username: str, password: str) -> Optional[User]:
        """Authenticates a user using SQLAlchemy, returns User object on success."""
        try:
            with get_session() as session:
                user = session.query(User).filter(func.lower(User.username) == func.lower(username)).first()

            if user and check_password_hash(user.password_hash, password):
                self.logger.info(f"User authenticated successfully via SQLAlchemy.", username=username)
                return user

            self.logger.warning(f"Authentication failed via SQLAlchemy.", username=username)
            return None
        except SQLAlchemyError as e:
            self.logger.error(f"Error authenticating user via SQLAlchemy", username=username, error=str(e), exc_info=True)
            return None

    # ==================== ROUTES ====================

    def _setup_routes(self):
        """Setup all Flask routes"""
        self.logger.info("Setting up Flask routes...")
        limiter = self.limiter
        metrics = self.metrics # Get Prometheus metrics instance

        # Define Prometheus metrics (if available)
        if PROMETHEUS_AVAILABLE:
            login_counter = metrics.counter(
                'logins_total', 'Total number of logins',
                labels={'status': lambda r: 'success' if r.status_code == 200 else 'failure'}
            )
            registration_counter = metrics.counter(
                'registrations_total', 'Total number of registrations',
                 labels={'status': lambda r: 'success' if r.status_code == 201 else 'failure'}
            )
            # You can add more metrics like histograms for request latency, etc.

        # ==================== AUTH ENDPOINTS (with Validation) ====================

        @self.app.route('/api/auth/register', methods=['POST'])
        @limiter.limit("5 per hour")
        @validate_json(RegisterSchema()) # Apply validation
        # @(registration_counter if PROMETHEUS_AVAILABLE else lambda f: f) # Apply counter conditionally
        def register():
            data = request.get_json() # Already validated
            username = data['username']
            password = data['password']
            email = data['email']

            if self.create_user_sqlalchemy(username, password, email):
                logger.info("User registration successful", username=username)
                # Apply counter manually for success
                if PROMETHEUS_AVAILABLE: registration_counter(lambda: Response(status=201)) # Dummy response for label
                return jsonify({"message": "User registered successfully"}), 201
            else:
                logger.warning("User registration failed in DB layer", username=username)
                # Apply counter manually for failure
                if PROMETHEUS_AVAILABLE: registration_counter(lambda: Response(status=409)) # Dummy response for label
                return jsonify({"error": "Registration failed. Username or email may already exist."}), 409

        @self.app.route('/api/auth/login', methods=['POST'])
        @limiter.limit("10 per minute")
        @validate_json(LoginSchema()) # Apply validation
        # @(login_counter if PROMETHEUS_AVAILABLE else lambda f: f) # Apply counter conditionally
        def login():
            data = request.get_json() # Already validated
            username = data['username']
            password = data['password']

            user = self.authenticate_user_sqlalchemy(username, password)

            if not user:
                logger.warning("Login failed: Invalid credentials", username=username)
                # Apply counter manually for failure
                if PROMETHEUS_AVAILABLE: login_counter(lambda: Response(status=401)) # Dummy response for label
                return jsonify({"error": "Invalid credentials"}), 401
            
            # --- NEW: Add role to JWT claims ---
            additional_claims = {"role": user.role}
            access_token = create_access_token(identity=user.username, additional_claims=additional_claims)

            response = jsonify(access_token=access_token, user=user.to_dict())

            # Set HttpOnly cookie
            set_access_cookies(response, access_token)
            logger.info("Login successful, JWT cookie set", username=username, role=user.role)

            # Apply counter manually for success
            if PROMETHEUS_AVAILABLE: login_counter(lambda: Response(status=200)) # Dummy response for label
            return response, 200

        @self.app.route('/api/auth/logout', methods=['POST'])
        def logout():
            # No validation needed
            current_user = get_jwt_identity() # Get identity before clearing
            response = jsonify({"message": "Logout successful"})
            unset_jwt_cookies(response) # Clear cookies
            logger.info("User logged out", username=current_user or "Unknown")
            # TODO: Add token to blocklist if using JWT blocklisting
            return response, 200

        @self.app.route('/api/auth/me')
        @jwt_required() # Ensures JWT is valid (either header or cookie)
        @limiter.limit("120 per minute")
        def protected():
            current_user_identity = get_jwt_identity()
            # --- NEW: Get role from claims ---
            claims = get_jwt()
            user_role = claims.get('role', 'user')

            logger.debug("Accessed protected /me endpoint", username=current_user_identity, role=user_role)
            # Optionally fetch full user from DB if needed
            # For now, just return identity and role
            return jsonify(logged_in_as=current_user_identity, role=user_role)

        # ==================== PAGE ROUTES ====================
        # (Serve React App - No changes)
        @self.app.route('/', defaults={'path': ''})
        @self.app.route('/<path:path>')
        def serve_react_app(path):
            static_folder = self.app.static_folder
            if static_folder is None:
                logger.error("Static folder not configured.")
                return "Static folder not found", 404

            # Security: Basic path validation to prevent directory traversal
            safe_path = os.path.normpath(os.path.join(static_folder, path)).lstrip(os.path.sep)
            full_path = os.path.join(static_folder, safe_path)

            if path != "" and os.path.exists(full_path) and os.path.isfile(full_path):
                # Ensure the resolved path is still within the static folder
                if os.path.commonprefix((os.path.realpath(full_path), os.path.realpath(static_folder))) == os.path.realpath(static_folder):
                    return send_from_directory(static_folder, safe_path)
                else:
                    logger.warning("Attempted directory traversal", requested_path=path)
                    return "Not Found", 404
            else:
                index_path = os.path.join(static_folder, 'index.html')
                if not os.path.exists(index_path):
                    logger.error("index.html not found in static folder.", path=index_path)
                    return "Frontend entry point not found", 404
                return send_from_directory(static_folder, 'index.html')


        # ==================== API ENDPOINTS ====================

        @self.app.route('/health')
        @limiter.exempt
        def health_check():
            # (Remains the same)
            db_ok = False
            redis_ok = self.cache.available if self.cache else False
            celery_ok = False
            try:
                with engine.connect() as connection: db_ok = True
            except Exception as e: logger.error(f"Health check DB failed", error=str(e))
            if self.celery:
                try:
                    # Basic check: ping a worker (might be slow/unreliable)
                    # Or check connection to broker/backend
                    self.celery.broker_connection().ensure_connection(max_retries=1)
                    celery_ok = True
                except Exception as e:
                    logger.warning("Health check Celery failed", error=str(e))

            status = {
                'status': 'healthy' if db_ok and redis_ok and celery_ok else 'partial',
                'timestamp': datetime.now().isoformat(),
                'version': '3.4.0', # Updated version
                'checks': {
                    'redis': 'ok' if redis_ok else 'unavailable',
                    'postgresql': 'ok' if db_ok else 'unavailable',
                    'celery': 'ok' if celery_ok else 'unavailable',
                },
                'uptime': str(datetime.now() - self.start_time).split('.')[0]
            }
            status_code = 200 if status['status'] == 'healthy' else 503
            return jsonify(status), status_code

        # --- UPDATED: /metrics endpoint for Prometheus ---
        @self.app.route('/metrics')
        @limiter.exempt
        def prometheus_metrics():
            """Expose metrics for Prometheus scraping."""
            if not PROMETHEUS_AVAILABLE or not self.metrics:
                return jsonify({"message": "Prometheus exporter not available."}), 503
            # The exporter handles generating the response automatically
            # We don't need to call generate_latest() explicitly if using PrometheusMetrics
            # This endpoint is automatically registered by the extension.
            # However, if we need custom metrics not auto-instrumented:
            # from prometheus_client import generate_latest, REGISTRY
            # Add custom metric collection logic here if needed
            # return Response(generate_latest(REGISTRY), mimetype='text/plain')
            logger.debug("Serving Prometheus metrics")
            return Response("", mimetype='text/plain') # Let the middleware handle it


        # ... (Other API endpoints like /api/dashboard, /api/devices, etc. remain largely the same) ...
        # Ensure @jwt_required() is uncommented for production where needed.
        @self.app.route('/api/dashboard')
        @jwt_required()
        @limiter.limit("60 per minute")
        def get_dashboard_data():
            """Get combined dashboard data"""
            try:
                cache_key = 'dashboard_combined_data'
                cached = self.cache.get(cache_key)
                if cached: return jsonify(cached)

                latest_devices_df = self.get_latest_device_data(limit=500)
                recent_alerts = self.get_recent_alerts(limit=20, acknowledged=False)
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

                dashboard_payload = {
                    **overview_data,
                    'devices': latest_devices_df.to_dict('records')[:100],
                    'alerts': recent_alerts,
                    'performanceData': self.app._generate_dummy_performance_data(overview_data.get('systemHealth', 80), overview_data.get('efficiency', 85))
                }
                self.cache.set(cache_key, dashboard_payload, ttl=30)
                return jsonify(dashboard_payload)
            except Exception as e:
                self.logger.error(f"Error fetching dashboard data", error=str(e), exc_info=True)
                return jsonify({"error": "Failed to retrieve dashboard data"}), 500

        @self.app.route('/api/devices')
        @jwt_required()
        @limiter.limit("120 per minute")
        def get_devices():
            try:
                limit = request.args.get('limit', default=1000, type=int)
                devices_df = self.get_latest_device_data(limit=limit)
                return jsonify(devices_df.to_dict('records'))
            except Exception as e:
                self.logger.error(f"Error getting devices", error=str(e), exc_info=True)
                return jsonify({"error": "Failed to retrieve devices"}), 500

        @self.app.route('/api/devices/<string:device_id>')
        @jwt_required()
        @limiter.limit("120 per minute")
        def get_device(device_id):
            cache_key = f'device_latest:{device_id}'
            cached = self.cache.get(cache_key)
            if cached: return jsonify(cached)
            try:
                with get_session() as session:
                    device_data = session.query(DeviceData).filter(DeviceData.device_id == device_id).order_by(DeviceData.timestamp.desc()).first()
                if device_data:
                    data_dict = device_data.to_dict()
                    self.cache.set(cache_key, data_dict, ttl=60)
                    return jsonify(data_dict)
                else:
                    return jsonify({"error": "Device not found"}), 404
            except Exception as e:
                self.logger.error(f"Error getting device", device_id=device_id, error=str(e), exc_info=True)
                return jsonify({"error": "Failed to retrieve device data"}), 500

        @self.app.route('/api/alerts')
        @jwt_required()
        @limiter.limit("120 per minute")
        def get_alerts():
            try:
                limit = request.args.get('limit', default=50, type=int)
                severity = request.args.get('severity', default=None, type=str)
                acknowledged_str = request.args.get('acknowledged', default=None, type=str)
                acknowledged_filter: Optional[bool] = None
                if acknowledged_str is not None: acknowledged_filter = acknowledged_str.lower() == 'true'
                alerts = self.get_recent_alerts(limit=limit, severity=severity, acknowledged=acknowledged_filter)
                return jsonify(alerts)
            except Exception as e:
                self.logger.error(f"Error getting alerts", error=str(e), exc_info=True)
                return jsonify({"error": "Failed to retrieve alerts"}), 500

        @self.app.route('/api/alerts/acknowledge/<string:alert_id>', methods=['POST'])
        @jwt_required()
        @limiter.limit("30 per minute")
        @self.csrf.exempt # Exempt if using header-based JWT primarily, or ensure CSRF token is sent
        def acknowledge_alert_api(alert_id):
            user_id = get_jwt_identity() # Get user from JWT
            success = self.acknowledge_alert_db(alert_id, user_id)
            if success:
                with get_session() as session:
                    alert = session.query(Alert).filter(Alert.id == alert_id).first()
                updated_alert_data = alert.to_dict() if alert else None
                if self.socketio:
                    self.socketio.emit('alert_acknowledged', {'alertId': alert_id, 'acknowledgedBy': user_id, 'timestamp': datetime.utcnow().isoformat()}, room='alerts')
                return jsonify({"message": "Alert acknowledged", "alert": updated_alert_data}), 200
            else:
                return jsonify({"error": "Failed to acknowledge alert or alert not found"}), 404
        
        # --- Example Admin-Only Route ---
        @self.app.route('/api/admin/summary')
        @jwt_required()
        @role_required('admin') # <-- NEW RBAC DECORATOR IN USE
        @limiter.limit("30 per minute")
        def get_admin_summary():
            """An example route only accessible to users with the 'admin' role."""
            logger.info("Admin summary endpoint accessed", user=get_jwt_identity())
            # In a real app, you'd fetch sensitive summary data here
            return jsonify({
                "message": "Welcome, Admin!",
                "total_users": session.query(User).count(),
                "total_devices": session.query(DeviceData.device_id).distinct().count(),
                "connected_clients": len(self.connected_clients)
            })

        # --- Placeholders ---
        @self.app.route('/api/predictions')
        @jwt_required()
        @limiter.limit("60 per minute")
        def get_predictions():
            device_id = request.args.get('device_id')
            if not device_id: return jsonify({"error": "device_id query parameter is required"}), 400
            # TODO: Implement
            logger.info("Serving predictions (placeholder)", device_id=device_id)
            return jsonify({"message": f"Predictions for {device_id} (not implemented)", "device_id": device_id}), 501

        @self.app.route('/api/health_scores')
        @jwt_required()
        @limiter.limit("60 per minute")
        def get_health_scores():
            # TODO: Implement
            logger.info("Serving health scores (placeholder)")
            return jsonify({"message": "Health scores (not implemented)"}), 501

        @self.app.route('/api/recommendations')
        @jwt_required()
        @limiter.limit("30 per minute")
        def get_recommendations():
            # TODO: Implement
            logger.info("Serving recommendations (placeholder)")
            return jsonify({"message": "Recommendations (not implemented)"}), 501

        @self.app.route('/api/system_metrics')
        @jwt_required()
        @limiter.limit("120 per minute")
        def get_system_metrics_api():
            logger.info("Serving system metrics")
            return jsonify(self._get_system_metrics())

        # --- Celery Task Routes ---
        @self.app.route('/api/tasks/start_report', methods=['POST'])
        @jwt_required()
        @role_required('admin') # Example: Only admins can run reports
        @limiter.limit("5 per hour")
        @self.csrf.exempt # Typically exempt for API calls if using header auth
        def start_report_task():
            if not self.celery: return jsonify({"error": "Async task runner not available"}), 503
            try:
                user_id = get_jwt_identity()
                task = generate_report_task.delay(user_id=user_id) # Pass context if needed
                logger.info(f"Dispatched generate_report_task", task_id=task.id, user_id=user_id)
                return jsonify({"task_id": task.id, "status": "PENDING"}), 202
            except Exception as e:
                logger.error(f"Failed to dispatch Celery task", error=str(e), exc_info=True)
                return jsonify({"error": "Failed to start report generation task"}), 500

        @self.app.route('/api/tasks/status/<string:task_id>')
        @jwt_required()
        @limiter.limit("60 per minute")
        def get_task_status(task_id):
            if not self.celery: return jsonify({"error": "Async task runner not available"}), 503
            try:
                task_result = generate_report_task.AsyncResult(task_id)
                response = {'task_id': task_id, 'status': task_result.status, 'result': task_result.result if task_result.ready() else None}
                if task_result.failed():
                    response['error'] = str(task_result.result)
                    logger.warning(f"Task failed", task_id=task_id, error=str(task_result.result), traceback=task_result.traceback)
                logger.debug("Checked task status", task_id=task_id, status=response['status'])
                return jsonify(response)
            except Exception as e:
                logger.error(f"Failed to get task status", task_id=task_id, error=str(e), exc_info=True)
                return jsonify({"error": "Failed to retrieve task status"}), 500

        # --- Static File Serving ---
        @self.app.route('/reports/<path:filename>')
        # @jwt_required() # Optional protection
        def serve_report(filename):
            reports_dir = Path(get_optional_env('REPORTS_DIR', 'REPORTS/generated'))
            logger.debug(f"Attempting to serve report", filename=filename, directory=str(reports_dir))
            if not reports_dir.is_dir():
                logger.error(f"Reports directory not found", directory=str(reports_dir))
                return "Reports directory configuration error", 500
            try:
                return send_from_directory(reports_dir, filename, as_attachment=False)
            except FileNotFoundError:
                logger.warning(f"Report file not found", filename=filename)
                return "Report not found", 404

        @self.app.route('/exports/<path:filename>')
        # @jwt_required() # Optional protection
        def serve_export(filename):
            exports_dir = Path(get_optional_env('EXPORTS_DIR', 'EXPORTS'))
            logger.debug(f"Attempting to serve export", filename=filename, directory=str(exports_dir))
            if not exports_dir.is_dir():
                logger.error(f"Exports directory not found", directory=str(exports_dir))
                return "Exports directory configuration error", 500
            try:
                return send_from_directory(exports_dir, filename, as_attachment=True)
            except FileNotFoundError:
                logger.warning(f"Export file not found", filename=filename)
                return "Export not found", 404

        self.logger.info("Flask routes setup completed.")


    # ==================== WEBSOCKET EVENTS ====================
    # (No major changes needed, added JWT verification example)

    def _setup_websocket_events(self):
        """Setup optimized WebSocket events with authentication"""
        if not self.socketio:
            self.logger.warning("SocketIO not initialized, skipping WebSocket event setup.")
            return

        # --- NEW: SocketIO Middleware for JWT Auth ---
        @self.socketio.on('connect')
        def handle_connect_auth():
            auth = request.args.get('token') # Or read from request.headers if preferred
            if not auth:
                logger.warning("WS connection attempt without token")
                return False # Reject connection

            try:
                # Use flask_jwt_extended to verify the token
                # --- NEW: Extract claims and pass to handler ---
                payload = decode_token(auth)
                user_id = payload.get('sub')
                user_role = payload.get('role', 'user') # Get role, default to 'user'
                logger.info("WS connection authenticated", sid=request.sid, user_id=user_id, role=user_role)
                handle_connect(identity=user_id, role=user_role) # Pass identity and role
                return True
            except (ExpiredSignatureError, DecodeError, InvalidTokenError) as e:
                logger.warning("WS authentication failed", error=str(e), sid=request.sid)
                return False # Reject connection
            except Exception as e:
                logger.error("Unexpected error during WS auth", error=str(e), exc_info=True)
                return False

        # Original connect handler (now called after auth success)
        # @self.socketio.on('connect') - This decorator is removed, handled by handle_connect_auth
        def handle_connect(identity: str = "anonymous", role: str = "user"): # NEW: Accept args
            client_id = request.sid
            self.connected_clients[client_id] = {
                'connected_at': datetime.utcnow(),
                'last_ping': datetime.utcnow(),
                'identity': identity, # NEW: Store identity
                'role': role,       # NEW: Store role
                'subscriptions': set()
            }
            logger.info(f"Client connected details stored", sid=client_id, identity=identity, role=role, total_clients=len(self.connected_clients))
            initial_data = self.cache.get('dashboard_combined_data')
            if initial_data:
                emit('dashboard_update', initial_data)


        @self.socketio.on('disconnect')
        def handle_disconnect():
            client_id = request.sid
            if client_id in self.connected_clients:
                identity = self.connected_clients[client_id].get('identity', 'anonymous')
                subs = self.connected_clients[client_id].get('subscriptions', set())
                for room in list(subs): self._leave_room_internal(client_id, room)
                del self.connected_clients[client_id]
                logger.info(f"Client disconnected", sid=client_id, identity=identity, total_clients=len(self.connected_clients))
            else:
                logger.info("Untracked client disconnected", sid=client_id)

        # ... (ping, subscribe, unsubscribe handlers remain the same) ...
        @self.socketio.on('ping_from_client')
        def handle_ping():
            client_id = request.sid
            if client_id in self.connected_clients:
                self.connected_clients[client_id]['last_ping'] = datetime.utcnow()
                emit('pong_from_server', {'timestamp': datetime.utcnow().isoformat()})

        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            client_id = request.sid
            if client_id not in self.connected_clients: return
            try:
                room_name = data.get('room')
                if room_name:
                    self._join_room_internal(client_id, room_name)
                    emit('subscribed', {'room': room_name})
                    logger.info(f"Client subscribed to room", sid=client_id, room=room_name)
                else: emit('error', {'message': 'Room name missing'})
            except Exception as e:
                logger.error(f"Subscription error", sid=client_id, error=str(e), exc_info=True)
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
                    logger.info(f"Client unsubscribed from room", sid=client_id, room=room_name)
                else: emit('error', {'message': 'Room name missing'})
            except Exception as e:
                logger.error(f"Unsubscription error", sid=client_id, error=str(e), exc_info=True)
                emit('error', {'message': 'Unsubscription failed'})

        self.logger.info("WebSocket event handlers setup.")

    # ... (Internal room helpers remain the same) ...
    def _join_room_internal(self, client_id, room_name):
        join_room(room_name, sid=client_id)
        self.rooms[room_name].add(client_id)
        if client_id in self.connected_clients:
            self.connected_clients[client_id]['subscriptions'].add(room_name)

    def _leave_room_internal(self, client_id, room_name):
        leave_room(room_name, sid=client_id)
        if room_name in self.rooms:
            self.rooms[room_name].discard(client_id)
            if not self.rooms[room_name]: del self.rooms[room_name]
        if client_id in self.connected_clients and 'subscriptions' in self.connected_clients[client_id]:
            self.connected_clients[client_id]['subscriptions'].discard(room_name)

    # ==================== BACKGROUND TASKS ====================

    def _start_background_tasks(self):
        """Start background tasks (Data generation). Celery Beat handles scheduled tasks."""
        self.logger.info("Starting background data generation task...")

        # Data Generation/Update Task (still useful for simulation/dev)
        def data_update_task():
            # (Remains the same as before) ...
            logger.info("Background data update task started.")
            last_cache_update = time.monotonic()
            while True:
                try:
                    if self.data_generator:
                        new_data_list = []
                        if not getattr(self.data_generator, 'devices', None):
                            self.data_generator.setup_simulation(device_count=25)
                        sim_time = datetime.utcnow()
                        for device_meta in self.data_generator.devices:
                            reading = self.data_generator._generate_single_reading(device_meta, sim_time, 0)
                            new_data_list.append(reading)
                        if new_data_list:
                                self.bulk_insert_device_data(new_data_list)
                                latest_df = pd.DataFrame(new_data_list)
                        else: latest_df = pd.DataFrame()
                    else: latest_df = self.get_latest_device_data(limit=100)

                    if not latest_df.empty:
                        if self.alert_manager: self._check_and_send_alerts(latest_df)
                        current_time = time.monotonic()
                        if current_time - last_cache_update > 5:
                            logger.debug("Updating dashboard cache and broadcasting...")
                            dashboard_payload = self._create_dashboard_payload(latest_df)
                            self.cache.set('dashboard_combined_data', dashboard_payload, ttl=30)
                            self.cache.set('latest_devices:500', latest_df.to_dict('records'), ttl=60)
                            if self.socketio:
                                    self.socketio.emit('dashboard_update', dashboard_payload, room='dashboard_updates')
                            last_cache_update = current_time
                        else: logger.debug("Skipping cache update/broadcast (too soon).")

                    eventlet.sleep(float(get_optional_env('DATA_UPDATE_INTERVAL', 1)))

                except Exception as e:
                    logger.error(f"Error in data update task", error=str(e), exc_info=True)
                    eventlet.sleep(10)

        if self.socketio and get_optional_env('ENABLE_DATA_GENERATOR', 'False').lower() == 'true':
            self.socketio.start_background_task(data_update_task)
            self.logger.info("Data generator background task started.")
        elif not self.socketio:
            logger.warning("SocketIO not available, cannot start background data task.")
        else:
            logger.info("Data generator background task is disabled (ENABLE_DATA_GENERATOR not true).")

        # APScheduler setup removed
        # self._setup_periodic_tasks() # REMOVED

    # ==================== CELERY TASKS (moved from bottom) ====================
    # Item 5: Define Celery tasks for scheduled jobs

    @staticmethod
    @celery_app.task(bind=True, name='digital_twin.cleanup_inactive_clients')
    def cleanup_inactive_clients_task(self):
        """Celery task to cleanup inactive WebSocket clients."""
        # Need app context - ContextTask handles this
        app_instance = current_app.extensions.get("digital_twin_instance")
        if not app_instance:
            logger.error("Could not get DigitalTwinApp instance in Celery task")
            return {'status': 'FAILURE', 'error': 'App instance missing'}

        logger.info("Running scheduled cleanup of inactive clients...")
        try:
            cleaned_count = app_instance._cleanup_inactive_clients_logic() # Call internal logic
            logger.info("Inactive client cleanup task finished", cleaned_count=cleaned_count)
            return {'status': 'SUCCESS', 'cleaned_count': cleaned_count}
        except Exception as e:
            logger.error(f"Celery task cleanup_inactive_clients failed", error=str(e), exc_info=True)
            self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
            raise

    @staticmethod
    @celery_app.task(bind=True, name='digital_twin.run_model_retraining')
    def run_model_retraining_task(self):
        """Celery task to run model retraining."""
        app_instance = current_app.extensions.get("digital_twin_instance")
        if not app_instance:
            logger.error("Could not get DigitalTwinApp instance in Celery task")
            return {'status': 'FAILURE', 'error': 'App instance missing'}

        if not app_instance.analytics_engine or not hasattr(app_instance.analytics_engine, 'retrain_models'):
            logger.warning("Analytics engine or retrain_models method not available.")
            return {'status': 'SKIPPED', 'message': 'Analytics engine not available'}

        logger.info("Running scheduled model retraining via Celery...")
        try:
            # The retrain_models method handles data fetching internally now
            result = app_instance.analytics_engine.retrain_models()
            logger.info(f"Model retraining task completed.", result_status=result.get('status'))
            return {'status': 'SUCCESS', 'retraining_result': result}
        except Exception as e:
            logger.error(f"Celery task run_model_retraining failed", error=str(e), exc_info=True)
            self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
            raise

    # Other Celery tasks (generate_report_task, export_data_task) remain the same

    # ==================== HELPER METHODS ====================
    # (No changes needed for helpers like _create_dashboard_payload, etc.)
    # ... (_check_and_send_alerts, _get_system_metrics remain the same) ...
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
        recent_alerts = self.get_recent_alerts(limit=20, acknowledged=False)
        return {
            **overview_data,
            'devices': latest_devices_df.to_dict('records')[:100],
            'alerts': recent_alerts,
            'performanceData': self.app._generate_dummy_performance_data(overview_data.get('systemHealth', 80), overview_data.get('efficiency', 85))
        }

    def _check_and_send_alerts(self, devices_df: pd.DataFrame):
        """Check for alerts and broadcast via WebSocket"""
        if not self.alert_manager or not self.socketio: return
        all_triggered_alerts = []
        try:
            for _, device_row in devices_df.iterrows():
                device_data = device_row.to_dict()
                if isinstance(device_data.get('timestamp'), datetime): device_data['timestamp'] = device_data['timestamp'].isoformat()
                triggered = self.alert_manager.evaluate_conditions(data=device_data, device_id=device_data.get('device_id'))
                if triggered: all_triggered_alerts.extend(triggered)
            if all_triggered_alerts:
                alerts_inserted = sum(1 for alert in all_triggered_alerts if self.insert_alert(alert))
                if alerts_inserted > 0: logger.info(f"Inserted new alerts into DB.", count=alerts_inserted)
                logger.info(f"Broadcasting triggered alerts...", count=len(all_triggered_alerts))
                for alert in all_triggered_alerts:
                    if isinstance(alert.get('timestamp'), datetime): alert['timestamp'] = alert['timestamp'].isoformat()
                    self.socketio.emit('alert_update', alert, room='alerts')
        except Exception as e:
            logger.error(f"Error checking/sending alerts", error=str(e), exc_info=True)

    def _get_system_metrics(self) -> Dict:
        # (Remains the same)
        metrics = {'timestamp': datetime.utcnow().isoformat(), 'cpu_percent': None, 'memory_percent': None, 'disk_percent': None, 'active_connections': len(self.connected_clients), 'cache_available': self.cache.available if self.cache else False, 'database_available': self.db_available, 'celery_available': CELERY_AVAILABLE}
        try:
            import psutil
            metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            metrics['memory_percent'] = psutil.virtual_memory().percent
            metrics['disk_percent'] = psutil.disk_usage('/').percent
        except ImportError: logger.warning("psutil not installed, cannot get detailed system metrics.")
        except Exception as e: logger.error(f"Error getting system metrics", error=str(e))
        return metrics

    # --- REMOVED: _cleanup_inactive_clients (logic moved to Celery task) ---
    def _cleanup_inactive_clients_logic(self) -> int:
        """Internal logic for cleaning up inactive WebSocket clients."""
        # This logic is now called by the Celery task
        now = datetime.utcnow()
        timeout = timedelta(minutes=int(get_optional_env('CLIENT_TIMEOUT_MINUTES', 5)))
        disconnected_count = 0
        inactive_sids = [
            sid for sid, client_info in self.connected_clients.items()
            if now - client_info.get('last_ping', now) > timeout
        ]

        for sid in inactive_sids:
            identity = self.connected_clients.get(sid, {}).get('identity', 'unknown')
            logger.warning(f"Disconnecting inactive client", sid=sid, identity=identity)
            if self.socketio:
                # Ensure disconnect runs in the correct context if needed, or handle potential errors
                try:
                    self.socketio.disconnect(sid, silent=True) # silent=True prevents disconnect event loop
                except Exception as e:
                    logger.error("Error during socketio disconnect", sid=sid, error=str(e))
            # Clean up internal state immediately, don't rely on disconnect handler which might not run
            if sid in self.connected_clients:
                subs = self.connected_clients[sid].get('subscriptions', set())
                for room in list(subs): self._leave_room_internal(sid, room)
                del self.connected_clients[sid]
            disconnected_count += 1

        if disconnected_count > 0:
            logger.info(f"Cleaned up inactive client(s).", count=disconnected_count)
        return disconnected_count


    # --- Error Handlers (No changes needed) ---
    def _setup_error_handlers(self):
        # (Remains the same)
        app = self.app
        @app.errorhandler(404)
        def not_found_error(error):
            logger.warning(f"404 Not Found", url=request.url)
            if request.path.startswith('/api/'): return jsonify(error="Resource not found"), 404
            static_folder = app.static_folder
            if static_folder and os.path.exists(os.path.join(static_folder, 'index.html')): return send_from_directory(static_folder, 'index.html')
            return "Not Found", 404
        @app.errorhandler(500)
        def internal_error(error):
            logger.error(f"500 Internal Server Error", error=str(error), exc_info=True)
            return jsonify(error="Internal server error"), 500
        @app.errorhandler(429)
        def ratelimit_handler(e):
            logger.warning(f"Rate limit exceeded", remote_addr=request.remote_addr, description=e.description)
            return jsonify(error="Rate limit exceeded", description=str(e.description)), 429
        @app.errorhandler(SQLAlchemyError)
        def handle_db_exception(e):
            logger.error(f"Database error occurred", error=str(e), exc_info=True)
            return jsonify(error="Database operation failed"), 500
        @app.errorhandler(Exception)
        def handle_exception(e):
            logger.error(f"Unhandled exception", error=str(e), exc_info=True)
            return jsonify(error="An unexpected error occurred"), 500
        logger.info("Error handlers registered.")

    def run(self, host='0.0.0.0', port=5000):
        # (Remains the same)
        effective_port = int(get_optional_env('PORT', port))
        effective_host = get_optional_env('HOST', host)
        debug_mode = self.app.config.get('DEBUG', False)
        self.start_time = datetime.now()
        logger.info(f"Starting Digital Twin App v3.4", host=effective_host, port=effective_port, debug=debug_mode)
        logger.info(f"Component Status", db_ok=self.db_available, redis_ok=self.cache.available, celery_ok=CELERY_AVAILABLE)
        if not self.socketio: logger.critical("SocketIO failed to initialize. Cannot run."); return
        try:
            self.socketio.run(self.app, host=effective_host, port=effective_port, debug=debug_mode, use_reloader=debug_mode, log_output=debug_mode)
        except KeyboardInterrupt: logger.info("Application interrupted. Shutting down...")
        except Exception as e: logger.critical(f"Application failed to run", error=str(e), exc_info=True)
        finally: self._shutdown()

    def _shutdown(self):
        # (Remains the same, removed scheduler shutdown)
        logger.info("Initiating graceful shutdown...")
        # if self.scheduler and self.scheduler.running: # REMOVED
        #     try: self.scheduler.shutdown(); logger.info("APScheduler shut down.")
        #     except Exception as e: logger.error(f"Error shutting down APScheduler", error=str(e))
        logger.info("Shutdown complete.")


# ==================== APPLICATION FACTORY & MAIN ====================

def create_app_instance() -> DigitalTwinApp:
    """Application factory function"""
    try:
        from dotenv import load_dotenv
        load_dotenv(verbose=True)
        logger.info(".env file loaded if present.")
    except ImportError:
        logger.info(".env file not loaded (dotenv not installed). Relying on system env vars.")
    try:
        app_instance = DigitalTwinApp()
        # --- NEW: Store instance in app context for Celery tasks ---
        app_instance.app.extensions["digital_twin_instance"] = app_instance
        return app_instance
    except Exception as e:
        logger.critical(f"FATAL: Failed to create application instance", error=str(e), exc_info=True)
        sys.exit(1)

digital_twin_app_instance = create_app_instance()
flask_app = digital_twin_app_instance.app

# --- Define Celery tasks AFTER flask_app is created ---
@celery_app.task(bind=True, name='digital_twin.generate_report')
def generate_report_task(self, user_id="system"): # Added user_id
    """Async report generation task using Celery."""
    logger.info(f"Celery task started: generate_report_task", task_id=self.request.id, user_id=user_id)
    # ContextTask handles app context
    try:
        report_gen = HealthReportGenerator()
        if not report_gen.db_manager:
            report_gen.db_manager = SecureDatabaseManager()

        with get_session() as session:
            recent_data = session.query(DeviceData).filter(DeviceData.timestamp >= datetime.utcnow() - timedelta(days=7)).all()
            report_data_df = pd.DataFrame([d.to_dict() for d in recent_data])

        if report_data_df.empty:
            logger.warning("No data found for report generation.", task_id=self.request.id)

        report_path = report_gen.generate_comprehensive_report(data_df=report_data_df, date_range_days=7)
        logger.info(f"Report generated by Celery task", task_id=self.request.id, report_path=report_path)
        return {'status': 'SUCCESS', 'report_path': str(report_path)}
    except Exception as e:
        logger.error(f"Celery task generate_report_task failed", task_id=self.request.id, error=str(e), exc_info=True)
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        raise

@celery_app.task(bind=True, name='digital_twin.export_data')
def export_data_task(self, format_type='json', date_range_days=7):
    """Async data export task using Celery."""
    logger.info(f"Celery task started: export_data_task", task_id=self.request.id, format=format_type, days=date_range_days)
    # ContextTask handles app context
    try:
        export_dir = Path(get_optional_env('EXPORTS_DIR', 'EXPORTS'))
        export_dir.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f'export_{timestamp_str}.{format_type}'
        filepath = export_dir / filename

        with get_session() as session:
            start_date = datetime.utcnow() - timedelta(days=int(date_range_days))
            data_query = session.query(DeviceData).filter(DeviceData.timestamp >= start_date).order_by(DeviceData.timestamp.asc())
            results = data_query.limit(50000).all() # Limit export size
            data_list = [d.to_dict() for d in results]

        if not data_list:
            logger.warning("No data found for export.", task_id=self.request.id)
            return {'status': 'SUCCESS', 'result': 'No data to export'}

        if format_type == 'csv':
            df = pd.DataFrame(data_list)
            if 'metadata' in df.columns: df['metadata'] = df['metadata'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
            df.to_csv(filepath, index=False, date_format='%Y-%m-%dT%H:%M:%S.%fZ')
        else: # JSON
            with open(filepath, 'w') as f: json.dump(data_list, f, indent=2, default=str)

        logger.info(f"Data exported by Celery task", task_id=self.request.id, export_path=str(filepath))
        return {'status': 'SUCCESS', 'export_path': str(filepath), 'filename': filename}
    except Exception as e:
        logger.error(f"Celery task export_data_task failed", task_id=self.request.id, error=str(e), exc_info=True)
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        raise

if __name__ == '__main__':
    digital_twin_app_instance.run()