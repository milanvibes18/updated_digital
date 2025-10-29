#!/usr/bin/env python3
"""
Enhanced Flask Application for Digital Twin System v3.6 (Production-Ready - Merged)
Main web application with Flask Blueprints, Redis caching, PostgreSQL via SQLAlchemy ORM,
optimized SocketIO (via Manager), async tasks via Celery, structured logging,
input validation, Prometheus metrics, RBAC, and enhanced security.

Refactored to delegate WebSocket logic to RealtimeWebSocketManager.
"""

import os
import sys
import json
import logging
import structlog
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
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
from flask import (
    Flask, render_template, request, jsonify, session, redirect, url_for,
    flash, send_from_directory, Response, current_app, Blueprint, g
)
from flask_cors import CORS
# -- SocketIO imports removed, will be handled by manager --
import eventlet

# Database imports
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, Index, func
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

# Redis imports
import redis
from redis.connection import ConnectionPool

# Security imports
from flask_jwt_extended import (
    JWTManager, jwt_required, create_access_token,
    get_jwt_identity, unset_jwt_cookies, decode_token,
    set_access_cookies, verify_jwt_in_request, get_jwt,
    create_refresh_token, set_refresh_cookies
)
from jwt.exceptions import ExpiredSignatureError, DecodeError, InvalidTokenError
import bcrypt
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_wtf.csrf import CSRFProtect

# Rate Limiting
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    FLASK_LIMITER_AVAILABLE = True
except ImportError:
    logging.warning("Flask-Limiter not installed. Rate limiting disabled.")
    FLASK_LIMITTER_AVAILABLE = False
    class Limiter:
        def __init__(self, *args, **kwargs): pass
        def limit(self, *args, **kwargs): return lambda f: f
        def exempt(self, f): return f
        def init_app(self, app): pass
    def get_remote_address(): return "127.0.0.1"

# Celery
try:
    from celery import Celery, Task
    from celery.schedules import crontab
    CELERY_AVAILABLE = True
except ImportError:
    logging.warning("Celery not installed. Async tasks will run synchronously.")
    CELERY_AVAILABLE = False
    class Celery:
        def __init__(self, *args, **kwargs): pass
        def task(self, *args, **kwargs): return lambda f: f
        conf = type('obj', (object,), {'beat_schedule': {}})()
    Task = object
    crontab = object

# Prometheus Metrics
try:
    from prometheus_flask_exporter import PrometheusMetrics
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logging.warning("prometheus-flask-exporter not installed. Prometheus metrics disabled.")
    PROMETHEUS_AVAILABLE = False
    class PrometheusMetrics:
        def __init__(self, *args, **kwargs): pass
        def init_app(self, app): pass
        def counter(self, *args, **kwargs): return lambda f: f

# Input Validation
try:
    from marshmallow import Schema, fields, ValidationError
    MARSHMALLOW_AVAILABLE = True
except ImportError:
    logging.warning("Marshmallow not installed. Input validation disabled.")
    MARSHMALLOW_AVAILABLE = False
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
        Dict = staticmethod(lambda **kwargs: None)
    class ValidationError(Exception): pass


# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import custom modules
try:
    from Digital_Twin.AI_MODULES.predictive_analytics_engine import PredictiveAnalyticsEngine
    from Digital_Twin.CONFIG.unified_data_generator import UnifiedDataGenerator
    from Digital_Twin.REPORTS.health_report_generator import HealthReportGenerator
    from Digital_Twin.AI_MODULES.health_score import HealthScoreCalculator
    from Digital_Twin.AI_MODULES.alert_manager import AlertManager
    from Digital_Twin.AI_MODULES.pattern_analyzer import PatternAnalyzer
    from Digital_Twin.AI_MODULES.recommendation_engine import RecommendationEngine
    from Digital_Twin.AI_MODULES.secure_database_manager import SecureDatabaseManager
    # --- Import WebSocket Manager ---
    from Digital_Twin.AI_MODULES.realtime_websocket_manager import RealtimeWebSocketManager
    # --- FIX 2: Import cleanup task ---
    from Digital_Twin.DATA_MANAGEMENT.auto_cleanup import perform_cleanup as perform_db_cleanup
except ImportError as e:
    logging.warning(f"Could not import some modules: {e}. Functionality might be limited.")
    PredictiveAnalyticsEngine = HealthScoreCalculator = AlertManager = PatternAnalyzer = RecommendationEngine = UnifiedDataGenerator = HealthReportGenerator = SecureDatabaseManager = RealtimeWebSocketManager = object
    perform_db_cleanup = None


# ==================== LOGGING SETUP (Structlog) ====================
def setup_logging():
    """Setup structured logging using structlog"""
    log_dir = Path('LOGS')
    log_dir.mkdir(exist_ok=True)
    log_level_str = os.environ.get('LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format='%(message)s',
        stream=sys.stdout,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if os.environ.get('LOG_FORMAT') == 'json' else structlog.dev.ConsoleRenderer()
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    log = structlog.get_logger('DigitalTwinApp')
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    logging.getLogger('socketio').setLevel(logging.WARNING)
    logging.getLogger('engineio').setLevel(logging.WARNING)
    logging.getLogger('celery').setLevel(logging.INFO)

    log.info("Structured logging configured.", log_level=log_level_str, log_format=os.environ.get('LOG_FORMAT', 'console'))
    return log

logger = setup_logging()

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

# ==================== STANDARDIZED API RESPONSE HELPER ====================
def create_standard_response(data: Any = None, error: str = None, status_code: int = 200) -> tuple:
    """Creates a standardized JSON response."""
    if error:
        response = {"error": error}
        if status_code == 200:
            if "not found" in error.lower():
                status_code = 404
            elif "validation failed" in error.lower():
                status_code = 400
            elif "unauthorized" in error.lower():
                status_code = 401
            elif "forbidden" in error.lower():
                status_code = 403
            else:
                status_code = 500
    else:
        response = {"data": data}

    return jsonify(response), status_code

# ==================== INPUT VALIDATION SCHEMAS (Marshmallow) ====================
class RegisterSchema(Schema):
    username = fields.Str(required=True, validate=lambda s: len(s) >= 3)
    password = fields.Str(required=True, validate=lambda s: len(s) >= 8)
    email = fields.Email(required=True)

class LoginSchema(Schema):
    username = fields.Str(required=True)
    password = fields.Str(required=True)

class SubscribeSchema(Schema):
    room = fields.Str(required=True, error_messages={"required": "Room name is required."})

class UnsubscribeSchema(Schema):
    room = fields.Str(required=True, error_messages={"required": "Room name is required."})

class StreamSubscribeSchema(Schema):
    stream_id = fields.Str(required=True, error_messages={"required": "Stream ID is required."})

class DataRequestSchema(Schema):
    type = fields.Str(required=True, error_messages={"required": "Data type is required."})
    filters = fields.Dict(required=False, default={})

# Decorator for HTTP JSON validation
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
                return create_standard_response(error="Invalid JSON request", status_code=400)
            try:
                g.validated_data = schema.load(json_data)
                return f(*args, **kwargs)
            except ValidationError as err:
                logger.warning("Validation failed", errors=err.messages, received_data=json_data)
                return create_standard_response(error="Validation failed", status_code=400)
            except Exception as e:
                logger.error("Unexpected error during validation", error=str(e), exc_info=True)
                return create_standard_response(error="An internal error occurred during validation", status_code=500)
        return wrapper
    return decorator

# --- WebSocket validation decorator removed, as logic is now in the manager ---

# Decorator for RBAC
def role_required(required_role: str):
    """Decorator to require a specific role from the JWT."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                verify_jwt_in_request()
                claims = get_jwt()
                user_role = claims.get("role", "user")

                if user_role == 'admin':
                    return fn(*args, **kwargs)

                if user_role != required_role:
                    logger.warning(
                        "Role access denied",
                        required_role=required_role,
                        user_role=user_role,
                        identity=claims.get('sub')
                    )
                    return create_standard_response(error="Access forbidden: Insufficient permissions", status_code=403)

                return fn(*args, **kwargs)
            except (ExpiredSignatureError, DecodeError, InvalidTokenError) as e:
                logger.warning("Role verification failed: Invalid JWT", error=str(e))
                return create_standard_response(error=f"Token is invalid: {str(e)}", status_code=401)
            except Exception as e:
                logger.error("Error during role verification", error=str(e), exc_info=True)
                return create_standard_response(error="Authentication required", status_code=401)
        return wrapper
    return decorator

# ==================== DATABASE MODELS (SQLAlchemy) ====================
Base = declarative_base()
UTCDateTime = DateTime(timezone=True)

class DeviceData(Base):
    """SQLAlchemy model for device data"""
    __tablename__ = 'device_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(100), nullable=False, index=True)
    device_type = Column(String(100))
    device_name = Column(String(200))
    timestamp = Column(UTCDateTime, nullable=False, index=True, default=lambda: datetime.now(timezone.utc))
    value = Column(Float)
    status = Column(String(50), index=True)
    health_score = Column(Float)
    efficiency_score = Column(Float)
    location = Column(String(200))
    unit = Column(String(50))
    metadata = Column(Text)

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
            except (json.JSONDecodeError, TypeError):
                data['metadata'] = {}
        if isinstance(data.get('timestamp'), datetime):
            data['timestamp'] = data['timestamp'].isoformat()
        return data

class Alert(Base):
    """SQLAlchemy model for alerts"""
    __tablename__ = 'alerts'

    id = Column(String(100), primary_key=True, default=lambda: str(uuid.uuid4()))
    device_id = Column(String(100), nullable=False, index=True)
    rule_name = Column(String(100))
    severity = Column(String(50), index=True)
    message = Column(Text)
    timestamp = Column(UTCDateTime, nullable=False, index=True, default=lambda: datetime.now(timezone.utc))
    acknowledged = Column(Boolean, default=False, index=True)
    resolved = Column(Boolean, default=False, index=True)
    value = Column(Float)
    metadata = Column(Text)

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
    email = Column(String(200), unique=True, index=True)
    created_at = Column(UTCDateTime, default=lambda: datetime.now(timezone.utc))
    role = Column(String(50), nullable=False, default='user', index=True)

    def to_dict(self):
        """Helper method to convert model instance to dictionary, excluding password"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'role': self.role,
        }

# ==================== DATABASE SETUP (SQLAlchemy) ====================
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
                pool_recycle=3600,
                echo=get_optional_env('SQLALCHEMY_ECHO', 'False').lower() == 'true'
            )
            with engine.connect() as connection:
                logger.info("Database connection successful.")

            Base.metadata.create_all(bind=engine)
            logger.info("Database tables checked/created successfully.")

            SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            SessionLocal = scoped_session(SessionFactory)
            logger.info("SQLAlchemy SessionLocal created.")

        except Exception as e:
            logger.critical(f"CRITICAL: Failed to initialize database: {e}", exc_info=True)
            raise

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
        self.logger = structlog.get_logger('RedisCacheManager')
        self.client = None
        self.available = False
        self.memory_cache = {}

        try:
            self.client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
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
                        return json.loads(data_str)
                    except json.JSONDecodeError:
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
            if not isinstance(value, (str, int, float, bool)):
                value_str = json.dumps(value, default=str)
            else:
                value_str = str(value)

            if self.available and self.client:
                return self.client.setex(key, effective_ttl, value_str)
            else:
                self.memory_cache[key] = value
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
                keys_to_delete = [key for key in self.client.scan_iter(match=pattern)]
                if keys_to_delete:
                    deleted_count = self.client.delete(*keys_to_delete)
            else:
                keys_to_delete = [k for k in self.memory_cache.keys() if pattern in k]
                for k in keys_to_delete:
                    del self.memory_cache[k]
                deleted_count = len(keys_to_delete)
            self.logger.info(f"Cleared cache keys", count=deleted_count, pattern=pattern)
            return deleted_count
        except Exception as e:
            self.logger.error(f"Cache clear_pattern error", pattern=pattern, error=str(e))
            return 0

# ==================== CELERY CONFIGURATION ====================
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
            # --- FIX: Ensure this matches the module path of *this* file ---
            include=['Digital_Twin.WEB_APPLICATION.enhanced_flask_app_v2']
        )
    except Exception as e:
        logger.error(f"Failed to create Celery app", error=str(e), exc_info=True)
        return None

celery_app = make_celery()

def init_celery(app: Flask, celery: Optional[Celery]):
    """Bind Celery configuration to Flask app, add context, and add Beat schedule."""
    if not celery:
        return

    celery.conf.beat_schedule = {
        'cleanup-inactive-clients-every-5-minutes': {
            # --- FIX: Use task name from decorator, not file path ---
            'task': 'digital_twin.cleanup_inactive_clients',
            'schedule': timedelta(minutes=5),
        },
        'retrain-models-daily': {
            # --- FIX: Use task name from decorator, not file path ---
            'task': 'digital_twin.run_model_retraining',
            'schedule': crontab(
                hour=int(get_optional_env('RETRAIN_HOUR_UTC', 1)),
                minute=int(get_optional_env('RETRAIN_MINUTE_UTC', 0))
            ),
        },
        # --- FIX 2: Add periodic data cleanup task ---
        'periodic-db-cleanup-daily': {
            'task': 'digital_twin.perform_db_cleanup',
            'schedule': crontab(
                hour=int(get_optional_env('DB_CLEANUP_HOUR_UTC', 2)),
                minute=int(get_optional_env('DB_CLEANUP_MINUTE_UTC', 30))
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
                with structlog.contextvars.bound_contextvars(task_id=self.request.id, task_name=self.name):
                    logger.info("Executing Celery task within Flask context.")
                    try:
                        result = self.run(*args, **kwargs)
                        logger.info("Celery task completed successfully.")
                        return result
                    except Exception as e:
                        logger.error("Celery task failed", error=str(e), exc_info=True)
                        raise
    celery.Task = ContextTask
    app.extensions["celery"] = celery
    logger.info("Celery initialized and linked with Flask app context.", beat_schedule_keys=list(celery.conf.beat_schedule.keys()))

# ==================== FLASK BLUEPRINTS ====================
auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')
api_bp = Blueprint('api', __name__, url_prefix='/api')
tasks_bp = Blueprint('tasks', __name__, url_prefix='/api/tasks')

# ==================== MAIN APPLICATION CLASS ====================
class DigitalTwinApp:
    """Main Digital Twin Flask Application Class"""

    def __init__(self):
        self.app: Optional[Flask] = None
        # --- Use the WebSocket Manager ---
        self.socketio_manager: Optional[RealtimeWebSocketManager] = None
        self.cache: Optional[RedisCacheManager] = None
        self.analytics_engine = None
        self.health_calculator = None
        self.alert_manager = None
        self.pattern_analyzer = None
        self.recommendation_engine = None
        self.data_generator = None
        self.jwt: Optional[JWTManager] = None
        self.limiter: Optional[Limiter] = None
        self.csrf: Optional[CSRFProtect] = None
        self.metrics: Optional[PrometheusMetrics] = None
        self.celery: Optional[Celery] = celery_app

        # --- Client/Room state removed, now handled by socketio_manager ---
        self.start_time = datetime.now()
        self.db_available = False

        self.logger = logger
        self.logger.info("Digital Twin Application v3.6 (Refactored) starting...")

        self._create_app()
        if not self.app:
            raise RuntimeError("Flask app creation failed.")

        self._initialize_infrastructure()
        self._initialize_modules()
        self._setup_middleware()
        self._register_blueprints()
        # --- _setup_websocket_events removed ---
        self._start_background_tasks()
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
        
        is_production = get_optional_env('FLASK_ENV', 'development') == 'production'

        self.app.config.from_mapping(
            SECRET_KEY=get_required_env('SECRET_KEY'),
            DEBUG=get_optional_env('FLASK_DEBUG', 'False').lower() == 'true',
            JWT_SECRET_KEY=get_required_env('JWT_SECRET_KEY'),
            
            JWT_ACCESS_TOKEN_EXPIRES=timedelta(hours=int(get_optional_env('JWT_EXPIRY_HOURS', 1))),
            JWT_REFRESH_TOKEN_EXPIRES=timedelta(days=int(get_optional_env('JWT_REFRESH_DAYS', 30))),

            JWT_TOKEN_LOCATION=['headers', 'cookies'],
            JWT_COOKIE_SECURE=is_production,
            JWT_COOKIE_SAMESITE='Lax',
            JWT_COOKIE_CSRF_PROTECT=True,
            JWT_CSRF_CHECK_FORM=True,
            
            JWT_REFRESH_COOKIE_PATH='/api/auth/refresh',
            JWT_REFRESH_COOKIE_SECURE=is_production,
            JWT_REFRESH_COOKIE_SAMESITE='Lax',
            JWT_REFRESH_COOKIE_CSRF_PROTECT=True,

            WTF_CSRF_ENABLED=True,
            WTF_CSRF_SECRET_KEY=get_required_env('SECRET_KEY'),

            DATABASE_URL=get_required_env('DATABASE_URL'),
            REDIS_URL=get_required_env('REDIS_URL'),
            CELERY_BROKER_URL=get_optional_env('CELERY_BROKER_URL'),
            CELERY_RESULT_BACKEND=get_optional_env('CELERY_RESULT_BACKEND'),
            RATELIMIT_STORAGE_URI=get_required_env('REDIS_URL'),
            RATELIMIT_STRATEGY='fixed-window',
            RATELIMIT_HEADERS_ENABLED=True,
            CORS_ALLOWED_ORIGINS=get_optional_env('CORS_ALLOWED_ORIGINS', '*').split(',')
        )

        self.logger.info(f"Flask app created.", environment=get_optional_env('FLASK_ENV', 'development'), debug=self.app.config['DEBUG'])

    def _initialize_infrastructure(self):
        """Initialize DB, Redis, Celery, JWT, Limiter, CSRF, Prometheus, and WebSocket Manager"""
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

        self.csrf = CSRFProtect(self.app)
        self.logger.info("Flask-WTF CSRF protection initialized.")

        if PROMETHEUS_AVAILABLE:
            self.metrics = PrometheusMetrics(self.app)
            self.logger.info("Prometheus metrics initialized.")
        else:
            self.metrics = PrometheusMetrics()

        # --- Initialize the WebSocket Manager ---
        try:
            if RealtimeWebSocketManager:
                self.socketio_manager = RealtimeWebSocketManager(redis_url=self.app.config['REDIS_URL'])
                self.socketio_manager.initialize_socketio(self.app)
                self.logger.info("RealtimeWebSocketManager initialized.")
            else:
                self.logger.error("RealtimeWebSocketManager module not loaded. WebSockets disabled.")
        except Exception as e:
            self.logger.error(f"Failed to initialize RealtimeWebSocketManager", error=str(e), exc_info=True)


    def _initialize_modules(self):
        """Initialize AI and analytics modules"""
        self.logger.info("Initializing application modules...")
        try:
            self.analytics_engine = PredictiveAnalyticsEngine() if PredictiveAnalyticsEngine else None
            self.health_calculator = HealthScoreCalculator() if HealthScoreCalculator else None
            self.alert_manager = AlertManager() if AlertManager else None
            self.pattern_analyzer = PatternAnalyzer() if PatternAnalyzer else None
            self.recommendation_engine = RecommendationEngine() if RecommendationEngine else None
            self.data_generator = UnifiedDataGenerator(db_path=self.app.config['DATABASE_URL']) if UnifiedDataGenerator else None

            self.logger.info("Application modules initialized successfully.")
        except Exception as e:
            self.logger.error(f"Module initialization error", error=str(e), exc_info=True)

    def _setup_middleware(self):
        """Setup Flask middleware"""
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

        @self.app.before_request
        def log_request_info():
            structlog.contextvars.bind_contextvars(
                remote_addr=request.remote_addr,
                method=request.method,
                path=request.path,
                endpoint=request.endpoint,
            )
            logger.info("Request received")

        @self.app.after_request
        def log_response_info(response):
            logger.info("Request completed", status_code=response.status_code)
            structlog.contextvars.clear_contextvars()
            return response

    def _register_blueprints(self):
        """Register Flask Blueprints."""
        self.app.register_blueprint(auth_bp)
        self.app.register_blueprint(api_bp)
        self.app.register_blueprint(tasks_bp)
        self.logger.info("Flask Blueprints registered.")

    # ==================== DATABASE OPERATIONS ====================
    def bulk_insert_device_data(self, data_list: List[Dict]) -> bool:
        """Bulk insert device data efficiently using SQLAlchemy"""
        inserted_count = 0
        try:
            with get_session() as session:
                objects_to_insert = []
                for d in data_list:
                    if not d.get('device_id') or not d.get('timestamp'):
                        self.logger.warning(f"Skipping invalid device data record", record=d)
                        continue

                    ts = d['timestamp']
                    timestamp = None
                    if isinstance(ts, str):
                        try:
                            timestamp = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                            if timestamp.tzinfo is None:
                                timestamp = timestamp.replace(tzinfo=timezone.utc)
                        except ValueError:
                            self.logger.warning(f"Skipping record with invalid timestamp format", timestamp=ts)
                            continue
                    elif isinstance(ts, (int, float)):
                        timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
                    elif isinstance(ts, datetime):
                        timestamp = ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
                    else:
                        self.logger.warning(f"Skipping record with invalid timestamp type", type=str(type(ts)))
                        continue

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
        if cached is not None and isinstance(cached, list):
            self.logger.debug(f"Cache hit", cache_key=cache_key)
            try:
                df = pd.DataFrame(cached)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            except Exception as e:
                self.logger.warning(f"Failed to create DataFrame from cache", cache_key=cache_key, error=str(e))
                self.cache.delete(cache_key)

        self.logger.debug(f"Cache miss", cache_key=cache_key)
        try:
            with get_session() as session:
                subq = session.query(
                    DeviceData.device_id,
                    func.max(DeviceData.timestamp).label('max_ts')
                ).group_by(DeviceData.device_id).subquery()

                query = session.query(DeviceData).join(
                    subq,
                    (DeviceData.device_id == subq.c.device_id) &
                    (DeviceData.timestamp == subq.c.max_ts)
                ).order_by(DeviceData.timestamp.desc()).limit(limit)

                results = query.all()
                data_list = [r.to_dict() for r in results]

            self.logger.info(f"Fetched latest device data from DB", count=len(results))
            if data_list:
                self.cache.set(cache_key, data_list, ttl=60)
                df = pd.DataFrame(data_list)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
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
                timestamp = None
                if isinstance(ts, str):
                    try:
                        timestamp = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        if timestamp.tzinfo is None:
                            timestamp = timestamp.replace(tzinfo=timezone.utc)
                    except ValueError:
                        timestamp = datetime.now(timezone.utc)
                elif isinstance(ts, datetime):
                    timestamp = ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
                else:
                    timestamp = datetime.now(timezone.utc)

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
            self.cache.set(cache_key, data_list, ttl=30)
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
                        alert.metadata = json.dumps({
                            **(json.loads(alert.metadata) if alert.metadata else {}),
                            "acknowledged_by": user_id,
                            "acknowledged_at": datetime.now(timezone.utc).isoformat()
                        })
                        self.logger.info(f"Alert acknowledged in DB", alert_id=alert_id, user_id=user_id)
                        self.cache.clear_pattern("alerts:*")
                        return True
                    else:
                        self.logger.warning(f"Alert already acknowledged", alert_id=alert_id)
                        return False
                else:
                    self.logger.warning(f"Alert not found for acknowledgment", alert_id=alert_id)
                    return False
        except Exception as e:
            self.logger.error(f"DB Acknowledge error", alert_id=alert_id, error=str(e), exc_info=True)
            return False

    def create_user_sqlalchemy(self, username: str, password: str, email: Optional[str] = None) -> bool:
        """Creates a new user with bcrypt hashed password"""
        try:
            password_bytes = password.encode('utf-8')
            salt = bcrypt.gensalt()
            password_hash = bcrypt.hashpw(password_bytes, salt).decode('utf-8')

            with get_session() as session:
                existing_user = session.query(User).filter(
                    (func.lower(User.username) == func.lower(username)) |
                    (func.lower(User.email) == func.lower(email) if email else False)
                ).first()

                if existing_user:
                    self.logger.warning(f"User creation failed: Username or email already exists.", username=username, email=email)
                    return False

                new_user = User(username=username, password_hash=password_hash, email=email)
                session.add(new_user)

            self.logger.info(f"User created successfully via SQLAlchemy (bcrypt).", username=username)
            return True
        except SQLAlchemyError as e:
            self.logger.error(f"Error creating user via SQLAlchemy", username=username, error=str(e), exc_info=True)
            return False
        except Exception as e:
            self.logger.error(f"Error hashing password during user creation", username=username, error=str(e), exc_info=True)
            return False

    def authenticate_user_sqlalchemy(self, username: str, password: str) -> Optional[User]:
        """Authenticates a user using bcrypt"""
        try:
            with get_session() as session:
                user = session.query(User).filter(func.lower(User.username) == func.lower(username)).first()

            if user and user.password_hash:
                password_bytes = password.encode('utf-8')
                stored_hash_bytes = user.password_hash.encode('utf-8')
                if bcrypt.checkpw(password_bytes, stored_hash_bytes):
                    self.logger.info(f"User authenticated successfully via SQLAlchemy (bcrypt).", username=username)
                    return user

            self.logger.warning(f"Authentication failed via SQLAlchemy (bcrypt).", username=username)
            return None
        except SQLAlchemyError as e:
            self.logger.error(f"Error authenticating user via SQLAlchemy", username=username, error=str(e), exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"Error checking password during authentication", username=username, error=str(e), exc_info=True)
            return None

    # ==================== WEBSOCKET EVENTS ====================
    # --- All WebSocket event logic removed, now handled by RealtimeWebSocketManager ---

    # ==================== BACKGROUND TASKS ====================
    def _start_background_tasks(self):
        """Start background tasks (Data generation). Celery Beat handles scheduled tasks."""
        self.logger.info("Setting up background data generation task...")

        def data_update_task():
            logger.info("Background data update task started.")
            last_cache_update = time.monotonic()
            while True:
                try:
                    current_app_instance = current_app.extensions.get("digital_twin_instance")
                    if not current_app_instance:
                        logger.error("Could not get DigitalTwinApp instance in data_update_task")
                        eventlet.sleep(60)
                        continue

                    if current_app_instance.data_generator:
                        new_data_list = []
                        if not getattr(current_app_instance.data_generator, 'devices', None):
                            current_app_instance.data_generator.setup_simulation(device_count=25)
                        sim_time = datetime.now(timezone.utc)
                        for device_meta in current_app_instance.data_generator.devices:
                            reading = current_app_instance.data_generator._generate_single_reading(device_meta, pd.Timestamp(sim_time), 0)
                            new_data_list.append(reading)
                        if new_data_list:
                            current_app_instance.bulk_insert_device_data(new_data_list)
                            latest_df = pd.DataFrame(new_data_list)
                        else:
                            latest_df = pd.DataFrame()
                    else:
                        latest_df = current_app_instance.get_latest_device_data(limit=100)

                    if not latest_df.empty:
                        if current_app_instance.alert_manager:
                            current_app_instance._check_and_send_alerts(latest_df)
                        current_time = time.monotonic()
                        if current_time - last_cache_update > 5:
                            logger.debug("Updating dashboard cache and broadcasting...")
                            dashboard_payload = current_app_instance._create_dashboard_payload(latest_df)
                            current_app_instance.cache.set('dashboard_combined_data', dashboard_payload, ttl=30)
                            current_app_instance.cache.set('latest_devices:500', latest_df.to_dict('records'), ttl=60)
                            
                            # --- Use WebSocket Manager to broadcast ---
                            if current_app_instance.socketio_manager:
                                current_app_instance.socketio_manager.broadcast_to_room(
                                    'dashboard_updates',
                                    'dashboard_update',
                                    {'data': dashboard_payload}
                                )
                            last_cache_update = current_time
                        else:
                            logger.debug("Skipping cache update/broadcast (too soon).")

                    eventlet.sleep(float(get_optional_env('DATA_UPDATE_INTERVAL', 1)))

                except Exception as e:
                    logger.error(f"Error in data update task", error=str(e), exc_info=True)
                    eventlet.sleep(10)

        # --- Check for socketio_manager.socketio ---
        if self.socketio_manager and self.socketio_manager.socketio and get_optional_env('ENABLE_DATA_GENERATOR', 'False').lower() == 'true':
            self.app.extensions["digital_twin_instance"] = self
            self.socketio_manager.socketio.start_background_task(data_update_task)
            self.logger.info("Data generator background task started.")
        elif not self.socketio_manager or not self.socketio_manager.socketio:
            logger.warning("SocketIO Manager not available, cannot start background data task.")
        else:
            logger.info("Data generator background task is disabled (ENABLE_DATA_GENERATOR not true).")

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
            status_distribution = {
                'normal': status_counts.get('normal', 0),
                'warning': status_counts.get('warning', 0),
                'critical': status_counts.get('critical', 0),
                'offline': status_counts.get('offline', 0)
            }
            energy_usage_df = latest_devices_df[latest_devices_df['device_type'] == 'power_meter']['value']
            energy_usage = energy_usage_df.sum() / 1000.0 if not energy_usage_df.empty else 0.0

            overview_data = {
                'systemHealth': round(avg_health),
                'activeDevices': active_devices,
                'totalDevices': total_devices,
                'efficiency': round(avg_efficiency),
                'energyUsage': round(energy_usage, 1),
                'energyCost': round(energy_usage * 24 * 0.12),
                'statusDistribution': status_distribution,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        recent_alerts = self.get_recent_alerts(limit=20, acknowledged=False)
        
        perf_data = []
        if 'systemHealth' in overview_data and 'efficiency' in overview_data:
            base_health = overview_data['systemHealth']
            base_eff = overview_data['efficiency']
            now = datetime.now(timezone.utc)
            for i in range(24):
                ts = now - timedelta(hours=i)
                perf_data.append({
                    'timestamp': ts.isoformat(),
                    'systemHealth': max(0, min(100, base_health + random.uniform(-2, 2) - i * 0.1)),
                    'efficiency': max(0, min(100, base_eff + random.uniform(-1, 1) - i * 0.05)),
                    'energyUsage': max(0, overview_data.get('energyUsage', 0) * (1 + random.uniform(-0.1, 0.1) - i * 0.01))
                })
            perf_data.reverse()

        return {
            **overview_data,
            'devices': latest_devices_df.to_dict('records')[:100],
            'alerts': recent_alerts,
            'performanceData': perf_data
        }

    def _check_and_send_alerts(self, devices_df: pd.DataFrame):
        """Check for alerts and broadcast via WebSocket"""
        # --- Check for socketio_manager ---
        if not self.alert_manager or not self.socketio_manager:
            return
        all_triggered_alerts = []
        try:
            for _, device_row in devices_df.iterrows():
                device_data = device_row.to_dict()
                if isinstance(device_data.get('timestamp'), datetime):
                    device_data['timestamp'] = device_data['timestamp'].isoformat()
                triggered = self.alert_manager.evaluate_conditions(data=device_data, device_id=device_data.get('device_id'))
                if triggered:
                    all_triggered_alerts.extend(triggered)

            if all_triggered_alerts:
                alerts_inserted = sum(1 for alert in all_triggered_alerts if self.insert_alert(alert))
                if alerts_inserted > 0:
                    logger.info(f"Inserted new alerts into DB.", count=alerts_inserted)

                logger.info(f"Broadcasting triggered alerts...", count=len(all_triggered_alerts))
                for alert in all_triggered_alerts:
                    if isinstance(alert.get('timestamp'), datetime):
                        alert['timestamp'] = alert['timestamp'].isoformat()
                    
                    # --- Use WebSocket Manager to broadcast ---
                    self.socketio_manager.broadcast_to_room(
                        'alerts',
                        'alert_update',
                        {'data': alert}
                    )

        except Exception as e:
            logger.error(f"Error checking/sending alerts", error=str(e), exc_info=True)

    def _get_system_metrics(self) -> Dict:
        # --- Get active connections from socketio_manager ---
        active_connections = 0
        if self.socketio_manager:
            active_connections = len(self.socketio_manager.active_connections)

        metrics = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'cpu_percent': None,
            'memory_percent': None,
            'disk_percent': None,
            'active_connections': active_connections,
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
            logger.error(f"Error getting system metrics", error=str(e))
        return metrics

    def _cleanup_inactive_clients_logic(self) -> int:
        """
        Internal logic for cleaning up inactive WebSocket clients.
        This is now delegated to the RealtimeWebSocketManager.
        """
        if self.socketio_manager:
            return self.socketio_manager.cleanup_inactive_connections()
        return 0

    # ==================== ERROR HANDLERS ====================
    def _setup_error_handlers(self):
        app = self.app

        @app.errorhandler(404)
        def not_found_error(error):
            logger.warning(f"404 Not Found", url=request.url)
            if request.path.startswith('/api/'):
                return create_standard_response(error="Resource not found", status_code=404)
            static_folder = app.static_folder
            if static_folder and os.path.exists(os.path.join(static_folder, 'index.html')):
                return send_from_directory(static_folder, 'index.html')
            return "Not Found", 404

        @app.errorhandler(500)
        def internal_error(error):
            logger.error(f"500 Internal Server Error", error=str(error), exc_info=True)
            return create_standard_response(error="Internal server error", status_code=500)

        @app.errorhandler(429)
        def ratelimit_handler(e):
            logger.warning(f"Rate limit exceeded", remote_addr=request.remote_addr, description=e.description)
            return create_standard_response(error="Rate limit exceeded", status_code=429)

        @app.errorhandler(SQLAlchemyError)
        def handle_db_exception(e):
            logger.error(f"Database error occurred", error=str(e), exc_info=True)
            return create_standard_response(error="Database operation failed", status_code=500)

        @app.errorhandler(ValidationError)
        def handle_validation_error(error):
            logger.warning("API Input Validation Error", errors=error.messages)
            return create_standard_response(error="Validation failed", status_code=400)

        @app.errorhandler(Exception)
        def handle_exception(e):
            logger.error(f"Unhandled exception", error=str(e), exc_info=True)
            error_message = "An unexpected error occurred"
            if current_app.config.get('DEBUG'):
                error_message = str(e)
            return create_standard_response(error=error_message, status_code=500)

        logger.info("Error handlers registered.")

    def run(self, host='0.0.0.0', port=5000):
        effective_port = int(get_optional_env('PORT', port))
        effective_host = get_optional_env('HOST', host)
        debug_mode = self.app.config.get('DEBUG', False)
        self.start_time = datetime.now()
        logger.info(f"Starting Digital Twin App v3.6", host=effective_host, port=effective_port, debug=debug_mode)
        logger.info(f"Component Status", db_ok=self.db_available, redis_ok=self.cache.available, celery_ok=CELERY_AVAILABLE)
        
        # --- Check for socketio_manager.socketio ---
        if not self.socketio_manager or not self.socketio_manager.socketio:
            logger.critical("SocketIO Manager failed to initialize. Cannot run.")
            return
        try:
            # --- Run using the manager's socketio instance ---
            self.socketio_manager.socketio.run(
                self.app, 
                host=effective_host, 
                port=effective_port, 
                debug=debug_mode, 
                use_reloader=debug_mode, 
                log_output=debug_mode
            )
        except KeyboardInterrupt:
            logger.info("Application interrupted. Shutting down...")
        except Exception as e:
            logger.critical(f"Application failed to run", error=str(e), exc_info=True)
        finally:
            self._shutdown()

    def _shutdown(self):
        logger.info("Initiating graceful shutdown...")
        if self.socketio_manager:
            self.socketio_manager.shutdown()
        logger.info("Shutdown complete.")


# ==================== WEBSOCKET NAMESPACE ====================
# --- DigitalTwinNamespace class removed ---
# --- All WebSocket logic is now in RealtimeWebSocketManager ---


# ==================== BLUEPRINT ROUTE DEFINITIONS ====================

# --- Auth Blueprint Routes ---
@auth_bp.route('/register', methods=['POST'])
@validate_json(RegisterSchema())
def register():
    data = g.validated_data
    app_instance = current_app.extensions["digital_twin_instance"]
    success = app_instance.create_user_sqlalchemy(data['username'], data['password'], data['email'])
    if success:
        return create_standard_response(data={"message": "User registered successfully"}, status_code=201)
    else:
        return create_standard_response(error="Registration failed. Username or email may already exist.", status_code=409)

@auth_bp.route('/login', methods=['POST'])
@validate_json(LoginSchema())
def login():
    data = g.validated_data
    app_instance = current_app.extensions["digital_twin_instance"]
    user = app_instance.authenticate_user_sqlalchemy(data['username'], data['password'])
    if not user:
        return create_standard_response(error="Invalid credentials", status_code=401)

    additional_claims = {"role": user.role}
    access_token = create_access_token(identity=user.username, additional_claims=additional_claims)
    refresh_token = create_refresh_token(identity=user.username)

    response_data = {"access_token": access_token, "user": user.to_dict()}
    response, status_code = create_standard_response(data=response_data, status_code=200)

    set_access_cookies(response, access_token)
    set_refresh_cookies(response, refresh_token)
    logger.info("Login successful, JWT cookies set", username=user.username, role=user.role)
    return response, status_code

@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    current_user = get_jwt_identity()
    user = None
    try:
        with get_session() as session:
            user = session.query(User).filter(func.lower(User.username) == func.lower(current_user)).first()
    except Exception as e:
        logger.error("Could not fetch user during refresh", username=current_user, error=str(e))
        return create_standard_response(error="Error verifying user", status_code=500)
    if not user:
        return create_standard_response(error="User not found", status_code=401)

    new_access_token = create_access_token(identity=current_user, additional_claims={"role": user.role})
    response, status_code = create_standard_response(data={"access_token": new_access_token})
    set_access_cookies(response, new_access_token)
    logger.info("Access token refreshed", username=current_user, role=user.role)
    return response, status_code

@auth_bp.route('/logout', methods=['POST'])
def logout():
    response, status_code = create_standard_response(data={"message": "Logout successful"})
    unset_jwt_cookies(response)
    return response, status_code

@auth_bp.route('/me')
@jwt_required()
def me():
    current_user = get_jwt_identity()
    claims = get_jwt()
    user_role = claims.get('role', 'user')
    return create_standard_response(data={'logged_in_as': current_user, 'role': user_role})

# --- API Blueprint Routes ---
@api_bp.route('/dashboard')
@jwt_required()
def get_dashboard_data():
    app_instance = current_app.extensions["digital_twin_instance"]
    cache_key = 'dashboard_combined_data'
    cached = app_instance.cache.get(cache_key)
    if cached:
        return create_standard_response(data=cached)
    try:
        latest_devices_df = app_instance.get_latest_device_data(limit=500)
        dashboard_payload = app_instance._create_dashboard_payload(latest_devices_df)
        app_instance.cache.set(cache_key, dashboard_payload, ttl=30)
        return create_standard_response(data=dashboard_payload)
    except Exception as e:
        logger.error(f"Error fetching dashboard data", error=str(e), exc_info=True)
        return create_standard_response(error="Failed to retrieve dashboard data", status_code=500)

@api_bp.route('/devices')
@jwt_required()
def get_devices():
    app_instance = current_app.extensions["digital_twin_instance"]
    try:
        limit = request.args.get('limit', default=1000, type=int)
        devices_df = app_instance.get_latest_device_data(limit=limit)
        return create_standard_response(data=devices_df.to_dict('records'))
    except Exception as e:
        logger.error(f"Error getting devices", error=str(e), exc_info=True)
        return create_standard_response(error="Failed to retrieve devices", status_code=500)

@api_bp.route('/devices/<string:device_id>')
@jwt_required()
def get_device(device_id):
    app_instance = current_app.extensions["digital_twin_instance"]
    cache_key = f'device_latest:{device_id}'
    cached = app_instance.cache.get(cache_key)
    if cached:
        return create_standard_response(data=cached)
    try:
        with get_session() as session:
            device_data = session.query(DeviceData).filter(DeviceData.device_id == device_id).order_by(DeviceData.timestamp.desc()).first()
        if device_data:
            data_dict = device_data.to_dict()
            app_instance.cache.set(cache_key, data_dict, ttl=60)
            return create_standard_response(data=data_dict)
        else:
            return create_standard_response(error="Device not found", status_code=404)
    except Exception as e:
        logger.error(f"Error getting device", device_id=device_id, error=str(e), exc_info=True)
        return create_standard_response(error="Failed to retrieve device data", status_code=500)

# --- NEW ENDPOINT (Fix 1) ---
@api_bp.route('/historical_data')
@jwt_required()
def get_historical_data():
    """
    Fetches historical data for a device based on query parameters.
    """
    device_id = request.args.get('device_id')
    if not device_id:
        return create_standard_response(error="device_id query parameter is required", status_code=400)
    
    try:
        # Parse start_time
        start_time_str = request.args.get('start_time')
        if start_time_str:
            start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00')).astimezone(timezone.utc)
        else:
            start_time = datetime.now(timezone.utc) - timedelta(days=7)
        
        # Parse end_time
        end_time_str = request.args.get('end_time')
        if end_time_str:
            end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00')).astimezone(timezone.utc)
        else:
            end_time = datetime.now(timezone.utc)
        
        limit = request.args.get('limit', default=1000, type=int)

        logger.info("Fetching historical data", device_id=device_id, start=start_time, end=end_time, limit=limit)
        
        with get_session() as session:
            query = session.query(DeviceData).filter(
                DeviceData.device_id == device_id,
                DeviceData.timestamp >= start_time,
                DeviceData.timestamp <= end_time
            ).order_by(DeviceData.timestamp.desc()).limit(limit)
            
            results = query.all()
            data_list = [r.to_dict() for r in results]
        
        return create_standard_response(data=data_list)
    
    except ValueError as e:
        logger.warning("Invalid timestamp format for historical data", error=str(e))
        return create_standard_response(error=f"Invalid date format: {e}", status_code=400)
    except Exception as e:
        logger.error(f"Error getting historical data", device_id=device_id, error=str(e), exc_info=True)
        return create_standard_response(error="Failed to retrieve historical data", status_code=500)
# --- END NEW ENDPOINT ---

@api_bp.route('/alerts')
@jwt_required()
def get_alerts():
    app_instance = current_app.extensions["digital_twin_instance"]
    try:
        limit = request.args.get('limit', default=50, type=int)
        severity = request.args.get('severity', type=str)
        acknowledged_str = request.args.get('acknowledged', type=str)
        acknowledged = acknowledged_str.lower() == 'true' if acknowledged_str else None
        alerts = app_instance.get_recent_alerts(limit, severity, acknowledged)
        return create_standard_response(data=alerts)
    except Exception as e:
        logger.error(f"Error getting alerts", error=str(e), exc_info=True)
        return create_standard_response(error="Failed to retrieve alerts", status_code=500)

@api_bp.route('/alerts/acknowledge/<string:alert_id>', methods=['POST'])
@jwt_required()
def acknowledge_alert_api(alert_id):
    app_instance = current_app.extensions["digital_twin_instance"]
    user_id = get_jwt_identity()
    success = app_instance.acknowledge_alert_db(alert_id, user_id)
    if success:
        updated_alert_data = None
        with get_session() as session:
            alert = session.query(Alert).filter(Alert.id == alert_id).first()
            if alert:
                updated_alert_data = alert.to_dict()

        # --- Use WebSocket Manager to broadcast ---
        if app_instance.socketio_manager:
            app_instance.socketio_manager.broadcast_to_room(
                'alerts',
                'alert_acknowledged',
                {
                    'data': {
                        'alertId': alert_id,
                        'acknowledgedBy': user_id,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                }
            )

        app_instance.cache.clear_pattern("alerts:*")
        return create_standard_response(data={"message": "Alert acknowledged", "alert": updated_alert_data})
    else:
        return create_standard_response(error="Failed to acknowledge alert or alert not found", status_code=404)

@api_bp.route('/predictions')
@jwt_required()
def get_predictions():
    device_id = request.args.get('device_id')
    if not device_id:
        return create_standard_response(error="device_id query parameter is required", status_code=400)
    
    app_instance = current_app.extensions["digital_twin_instance"]
    logger.info("Serving predictions", device_id=device_id)
    
    if app_instance.analytics_engine:
        try:
            with get_session() as session:
                recent_data_db = session.query(DeviceData).filter(DeviceData.device_id == device_id).order_by(DeviceData.timestamp.desc()).limit(50).all()
            if not recent_data_db:
                return create_standard_response(data={"message": f"No recent data for {device_id}", "device_id": device_id}, status_code=404)
            
            recent_data_df = pd.DataFrame([d.to_dict() for d in recent_data_db])
            anomaly_pred = app_instance.analytics_engine.detect_anomalies(recent_data_df)
            failure_pred = app_instance.analytics_engine.predict_failure(recent_data_df)

            return create_standard_response(data={
                "device_id": device_id,
                "anomaly_prediction": anomaly_pred,
                "failure_prediction": failure_pred
            })
        except Exception as e:
            logger.error("Error generating predictions", device_id=device_id, error=str(e), exc_info=True)
            return create_standard_response(error="Prediction generation failed", status_code=500)
    else:
        return create_standard_response(data={"message": f"Predictions for {device_id} (engine unavailable)", "device_id": device_id}, status_code=501)

@api_bp.route('/health_scores')
@jwt_required()
def get_health_scores():
    app_instance = current_app.extensions["digital_twin_instance"]
    logger.info("Serving health scores")
    if app_instance.health_calculator:
        try:
            latest_devices_df = app_instance.get_latest_device_data(limit=500)
            if latest_devices_df.empty:
                return create_standard_response(data={})

            fleet_data = {
                device_id: group_df
                for device_id, group_df in latest_devices_df.groupby('device_id')
            }
            fleet_health = app_instance.health_calculator.calculate_fleet_health_score(fleet_data)
            scores_only = {
                dev_id: result.get('overall_score')
                for dev_id, result in fleet_health.get('device_results', {}).items()
                if not result.get('error')
            }
            return create_standard_response(data=scores_only)
        except Exception as e:
            logger.error("Error calculating health scores", error=str(e), exc_info=True)
            return create_standard_response(error="Health score calculation failed", status_code=500)
    else:
        return create_standard_response(data={"message": "Health scores (engine unavailable)"}, status_code=501)

@api_bp.route('/recommendations')
@jwt_required()
def get_recommendations():
    app_instance = current_app.extensions["digital_twin_instance"]
    logger.info("Serving recommendations")
    if app_instance.recommendation_engine:
        try:
            health_data_mock = {'overall_score': 0.8}
            recs = app_instance.recommendation_engine.generate_recommendations(health_data=health_data_mock)
            flat_recs = []
            for cat in ['emergency', 'maintenance', 'optimization', 'operational', 'preventive']:
                flat_recs.extend(recs.get(f'{cat}_recommendations', []))
            flat_recs.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
            return create_standard_response(data=flat_recs[:10])
        except Exception as e:
            logger.error("Error generating recommendations", error=str(e), exc_info=True)
            return create_standard_response(error="Recommendation generation failed", status_code=500)
    else:
        return create_standard_response(data={"message": "Recommendations (engine unavailable)"}, status_code=501)

@api_bp.route('/system_metrics')
@jwt_required()
def get_system_metrics_api():
    app_instance = current_app.extensions["digital_twin_instance"]
    logger.info("Serving system metrics")
    return create_standard_response(data=app_instance._get_system_metrics())

@api_bp.route('/admin/summary')
@jwt_required()
@role_required('admin')
def get_admin_summary():
    """Admin-only route example."""
    logger.info("Admin summary endpoint accessed", user=get_jwt_identity())
    try:
        with get_session() as session:
            user_count = session.query(User).count()
            device_count = session.query(DeviceData.device_id).distinct().count()
        app_instance = current_app.extensions["digital_twin_instance"]
        
        # --- Get connected clients from manager ---
        connected_clients = 0
        if app_instance.socketio_manager:
            connected_clients = len(app_instance.socketio_manager.active_connections)
            
        return create_standard_response(data={
            "message": "Welcome, Admin!",
            "total_users": user_count,
            "total_devices": device_count,
            "connected_clients": connected_clients
        })
    except Exception as e:
        logger.error("Error fetching admin summary", error=str(e), exc_info=True)
        return create_standard_response(error="Failed to fetch admin summary", status_code=500)

# --- Tasks Blueprint Routes ---
@tasks_bp.route('/start_report', methods=['POST'])
@jwt_required()
@role_required('admin')
def start_report_task():
    if not celery_app:
        return create_standard_response(error="Async task runner not available", status_code=503)
    try:
        user_id = get_jwt_identity()
        task = celery_app.send_task('digital_twin.generate_report', args=[user_id])
        logger.info(f"Dispatched generate_report_task", task_id=task.id, user_id=user_id)
        return create_standard_response(data={"task_id": task.id, "status": "PENDING"}, status_code=202)
    except Exception as e:
        logger.error(f"Failed to dispatch Celery task", error=str(e), exc_info=True)
        return create_standard_response(error="Failed to start report generation task", status_code=500)

@tasks_bp.route('/status/<string:task_id>')
@jwt_required()
def get_task_status(task_id):
    if not celery_app:
        return create_standard_response(error="Async task runner not available", status_code=503)
    try:
        task_result = celery_app.AsyncResult(task_id)
        response_data = {'task_id': task_id, 'status': task_result.status}
        if task_result.ready():
            if task_result.successful():
                response_data['result'] = task_result.result
            else:
                response_data['error'] = str(task_result.result)
                logger.warning(f"Task failed", task_id=task_id, error=str(task_result.result))
        logger.debug("Checked task status", task_id=task_id, status=response_data['status'])
        return create_standard_response(data=response_data)
    except Exception as e:
        logger.error(f"Failed to get task status", task_id=task_id, error=str(e), exc_info=True)
        return create_standard_response(error="Failed to retrieve task status", status_code=500)

# --- NEW ENDPOINT (Fix 1) ---
@tasks_bp.route('/export_data', methods=['POST'])
@jwt_required()
def start_export_task():
    """
    Triggers an asynchronous data export task.
    """
    if not celery_app:
        return create_standard_response(error="Async task runner not available", status_code=503)
    
    data = request.get_json() or {}
    format_type = data.get('format', 'json')
    days = data.get('days', 7)

    if format_type not in ['json', 'csv']:
        return create_standard_response(error="Validation failed: 'format' must be 'json' or 'csv'", status_code=400)
    
    try:
        days = int(days)
        if days <= 0:
            raise ValueError("Days must be positive")
    except ValueError:
        return create_standard_response(error="Validation failed: 'days' must be a positive integer", status_code=400)
    
    try:
        user_id = get_jwt_identity()
        task = celery_app.send_task(
            'digital_twin.export_data', # This task already exists
            args=[format_type, days]
        )
        logger.info(f"Dispatched export_data_task", task_id=task.id, user_id=user_id, format=format_type, days=days)
        return create_standard_response(data={"task_id": task.id, "status": "PENDING"}, status_code=202)
    except Exception as e:
        logger.error(f"Failed to dispatch export task", error=str(e), exc_info=True)
        return create_standard_response(error="Failed to start export task", status_code=500)
# --- END NEW ENDPOINT ---


# ==================== CELERY TASK DEFINITIONS ====================
@celery_app.task(bind=True, name='digital_twin.cleanup_inactive_clients')
def cleanup_inactive_clients_task(self):
    """Celery task to cleanup inactive WebSocket clients."""
    app_instance = current_app.extensions.get("digital_twin_instance")
    if not app_instance:
        logger.error("Could not get DigitalTwinApp instance in Celery task")
        return {'status': 'FAILURE', 'error': 'App instance missing'}
    
    if not app_instance.socketio_manager:
        logger.error("SocketIO Manager not found in Celery task")
        return {'status': 'FAILURE', 'error': 'SocketIO Manager missing'}

    logger.info("Running scheduled cleanup of inactive clients...")
    try:
        # --- Call manager's cleanup method ---
        cleaned_count = app_instance.socketio_manager.cleanup_inactive_connections()
        logger.info("Inactive client cleanup task finished", cleaned_count=cleaned_count)
        return {'status': 'SUCCESS', 'cleaned_count': cleaned_count}
    except Exception as e:
        logger.error(f"Celery task cleanup_inactive_clients failed", error=str(e), exc_info=True)
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        raise

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
        result = app_instance.analytics_engine.retrain_models()
        logger.info(f"Model retraining task completed.", result_status=result.get('status'))
        return {'status': 'SUCCESS', 'retraining_result': result}
    except Exception as e:
        logger.error(f"Celery task run_model_retraining failed", error=str(e), exc_info=True)
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        raise

@celery_app.task(bind=True, name='digital_twin.generate_report')
def generate_report_task(self, user_id="system"):
    """Async report generation task using Celery."""
    logger.info(f"Celery task started: generate_report_task", task_id=self.request.id, user_id=user_id)
    try:
        report_gen = HealthReportGenerator()
        with get_session() as session:
            start_date = datetime.now(timezone.utc) - timedelta(days=7)
            recent_data = session.query(DeviceData).filter(DeviceData.timestamp >= start_date).all()
            report_data_list = [d.to_dict() for d in recent_data]

        report_data_df = pd.DataFrame(report_data_list)
        if not report_data_df.empty and 'timestamp' in report_data_df.columns:
            report_data_df['timestamp'] = pd.to_datetime(report_data_df['timestamp'])

        if report_data_df.empty:
            logger.warning("No data found for report generation.", task_id=self.request.id)
            return {'status': 'SUCCESS', 'result': 'No data for report'}

        report_path = report_gen.generate_comprehensive_report(
            report_data={'devices': report_data_list},
            date_range_days=7
        )
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
    try:
        export_dir = Path(get_optional_env('EXPORTS_DIR', 'EXPORTS'))
        export_dir.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        filename = f'export_{timestamp_str}.{format_type}'
        filepath = export_dir / filename

        with get_session() as session:
            start_date = datetime.now(timezone.utc) - timedelta(days=int(date_range_days))
            data_query = session.query(DeviceData).filter(DeviceData.timestamp >= start_date).order_by(DeviceData.timestamp.asc())
            results = data_query.limit(50000).all()
            data_list = [d.to_dict() for d in results]

        if not data_list:
            logger.warning("No data found for export.", task_id=self.request.id)
            return {'status': 'SUCCESS', 'result': 'No data to export'}

        if format_type == 'csv':
            df = pd.DataFrame(data_list)
            if 'metadata' in df.columns:
                df['metadata'] = df['metadata'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
            df.to_csv(filepath, index=False)
        else:
            with open(filepath, 'w') as f:
                json.dump(data_list, f, indent=2)

        logger.info(f"Data exported by Celery task", task_id=self.request.id, export_path=str(filepath))
        return {'status': 'SUCCESS', 'export_path': str(filepath), 'filename': filename}
    except Exception as e:
        logger.error(f"Celery task export_data_task failed", task_id=self.request.id, error=str(e), exc_info=True)
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        raise

# --- NEW TASK (Fix 2) ---
@celery_app.task(bind=True, name='digital_twin.perform_db_cleanup')
def perform_db_cleanup_task(self):
    """
    Celery wrapper task for performing database cleanup.
    """
    app_instance = current_app.extensions.get("digital_twin_instance")
    if not app_instance:
        logger.error("Could not get DigitalTwinApp instance in Celery task")
        return {'status': 'FAILURE', 'error': 'App instance missing'}

    if not perform_db_cleanup:
        logger.error("perform_db_cleanup function not imported/available.")
        return {'status': 'FAILURE', 'error': 'Cleanup function not loaded'}
    
    logger.info("Running scheduled database cleanup...")
    try:
        # Get retention days from env, with defaults
        data_days = int(get_optional_env('DATA_RETENTION_DAYS', 30))
        alert_days = int(get_optional_env('ALERT_RETENTION_DAYS', 90))
        
        result = perform_db_cleanup(
            data_retention_days=data_days,
            alert_retention_days=alert_days
        )
        logger.info("Database cleanup task finished", **result)
        return {'status': 'SUCCESS', **result}
    except Exception as e:
        logger.error(f"Celery task perform_db_cleanup failed", error=str(e), exc_info=True)
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        raise
# --- END NEW TASK ---


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
        app_instance.app.extensions["digital_twin_instance"] = app_instance
        return app_instance
    except Exception as e:
        logger.critical(f"FATAL: Failed to create application instance", error=str(e), exc_info=True)
        sys.exit(1)

digital_twin_app_instance = create_app_instance()
flask_app = digital_twin_app_instance.app


# ==================== SPECIAL ROUTES (Health, Metrics, Static Files) ====================
@flask_app.route('/health')
def health_check():
    """Health check endpoint"""
    app_instance = flask_app.extensions["digital_twin_instance"]
    db_ok = False
    redis_ok = app_instance.cache.available if app_instance.cache else False
    celery_ok = False
    
    try:
        with engine.connect() as connection:
            db_ok = True
    except Exception as e:
        logger.error(f"Health check DB failed", error=str(e))
    
    if app_instance.celery:
        try:
            app_instance.celery.broker_connection().ensure_connection(max_retries=1)
            celery_ok = True
        except Exception as e:
            logger.warning("Health check Celery failed", error=str(e))

    status = {
        'status': 'healthy' if db_ok and redis_ok and celery_ok else 'partial',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'version': '3.6.0',
        'checks': {
            'redis': 'ok' if redis_ok else 'unavailable',
            'postgresql': 'ok' if db_ok else 'unavailable',
            'celery': 'ok' if celery_ok else 'unavailable',
        },
        'uptime': str(datetime.now() - app_instance.start_time).split('.')[0]
    }
    status_code = 200 if status['status'] == 'healthy' else 503
    return jsonify(status), status_code

@flask_app.route('/metrics')
def prometheus_metrics():
    """Expose metrics for Prometheus scraping."""
    app_instance = flask_app.extensions["digital_twin_instance"]
    if not PROMETHEUS_AVAILABLE or not app_instance.metrics:
        return jsonify({"message": "Prometheus exporter not available."}), 503
    logger.debug("Serving Prometheus metrics")
    return Response("", mimetype='text/plain')

@flask_app.route('/reports/<path:filename>')
def serve_report(filename):
    """Serve generated report files"""
    reports_dir = Path(get_optional_env('REPORTS_DIR', 'REPORTS/generated'))
    logger.debug(f"Attempting to serve report", filename=filename, directory=str(reports_dir))
    if not reports_dir.is_dir():
        logger.error(f"Reports directory not found", directory=str(reports_dir))
        return "Reports directory configuration error", 500
    try:
        safe_filename = Path(filename).name
        if safe_filename != filename:
            logger.warning("Potential path traversal detected in report request", filename=filename)
            return "Invalid filename", 400
        return send_from_directory(reports_dir, safe_filename, as_attachment=False)
    except FileNotFoundError:
        logger.warning(f"Report file not found", filename=filename)
        return "Report not found", 404

@flask_app.route('/exports/<path:filename>')
def serve_export(filename):
    """Serve export files"""
    exports_dir = Path(get_optional_env('EXPORTS_DIR', 'EXPORTS'))
    logger.debug(f"Attempting to serve export", filename=filename, directory=str(exports_dir))
    if not exports_dir.is_dir():
        logger.error(f"Exports directory not found", directory=str(exports_dir))
        return "Exports directory configuration error", 500
    try:
        safe_filename = Path(filename).name
        if safe_filename != filename:
            logger.warning("Potential path traversal detected in export request", filename=filename)
            return "Invalid filename", 400
        return send_from_directory(exports_dir, safe_filename, as_attachment=True)
    except FileNotFoundError:
        logger.warning(f"Export file not found", filename=filename)
        return "Export not found", 404

@flask_app.route('/', defaults={'path': ''})
@flask_app.route('/<path:path>')
def serve_react_app(path):
    """Serve React frontend application"""
    static_folder = flask_app.static_folder
    if static_folder is None:
        logger.error("Static folder not configured.")
        return "Static folder not found", 404

    safe_path = os.path.normpath(os.path.join(static_folder, path)).lstrip(os.path.sep)
    full_path = os.path.join(static_folder, safe_path)

    if path != "" and os.path.exists(full_path) and os.path.isfile(full_path):
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


# ==================== MAIN EXECUTION ====================
if __name__ == '__main__':
    digital_twin_app_instance.run()