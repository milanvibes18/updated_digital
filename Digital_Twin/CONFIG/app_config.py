#!/usr/bin/env python3
"""
Application Configuration for Digital Twin System v3.0
Centralized configuration hub loading settings primarily via environment variables,
with fallback defaults. Uses python-dotenv to load a .env file if present.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from distutils.util import strtobool # For parsing boolean env vars
import json

# --- Load .env File ---
# Load environment variables from .env file located in the Digital_Twin directory
# This allows easy management for local development. In production, env vars
# should be set directly in the deployment environment.
env_path = Path(__file__).parent.parent / '.env' # Assumes .env is in Digital_Twin/
if env_path.is_file():
    load_dotenv(dotenv_path=env_path, verbose=True, override=True)
    print(f"Loaded environment variables from: {env_path}")
else:
    print(f"Warning: .env file not found at {env_path}. Relying on system environment variables.")

log = logging.getLogger(__name__)

# --- Helper Functions ---

def get_env_var(var_name: str, default: Any = None) -> Any:
    """Gets an environment variable, returning a default if not set."""
    return os.environ.get(var_name, default)

def get_bool_env_var(var_name: str, default: bool = False) -> bool:
    """Gets a boolean environment variable, handling various truthy/falsy strings."""
    value = os.environ.get(var_name)
    if value is None:
        return default
    try:
        return bool(strtobool(value))
    except ValueError:
        return default

def get_int_env_var(var_name: str, default: int) -> int:
    """Gets an integer environment variable."""
    try:
        return int(os.environ.get(var_name, default))
    except (ValueError, TypeError):
        return default

def get_float_env_var(var_name: str, default: float) -> float:
    """Gets a float environment variable."""
    try:
        return float(os.environ.get(var_name, default))
    except (ValueError, TypeError):
        return default

def get_list_env_var(var_name: str, default: List = None, separator: str = ',') -> List:
    """Gets a list environment variable (comma-separated)."""
    value = os.environ.get(var_name)
    if value:
        return [item.strip() for item in value.split(separator)]
    return default if default is not None else []

def get_json_env_var(var_name: str, default: Any = None) -> Any:
    """Gets an environment variable expected to contain JSON."""
    value = os.environ.get(var_name)
    if value:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            log.warning(f"Failed to parse JSON from environment variable '{var_name}'. Using default.")
            return default
    return default

# --- Configuration Class ---

class Config:
    """
    Central configuration class loading settings from environment variables.
    Provides structured access to application settings.
    """
    # General Application Settings
    FLASK_ENV: str = get_env_var('FLASK_ENV', 'development')
    DEBUG: bool = get_bool_env_var('FLASK_DEBUG', FLASK_ENV == 'development')
    SECRET_KEY: str = get_env_var('SECRET_KEY', 'default-super-secret-key-change-me') # Fallback only for dev
    JWT_SECRET_KEY: str = get_env_var('JWT_SECRET_KEY', 'default-jwt-secret-key-change-me') # Fallback only for dev
    TIMEZONE: str = get_env_var('TIMEZONE', 'UTC')
    LOG_LEVEL: str = get_env_var('LOG_LEVEL', 'INFO').upper()
    LOG_FORMAT: str = get_env_var('LOG_FORMAT', 'console') # 'console' or 'json'

    # Database Configuration (PostgreSQL example from .env)
    DATABASE_URL: str = get_env_var('DATABASE_URL', 'sqlite:///../DATABASE/default_dev.db') # Example fallback for dev
    DB_POOL_SIZE: int = get_int_env_var('DB_POOL_SIZE', 20)
    DB_MAX_OVERFLOW: int = get_int_env_var('DB_MAX_OVERFLOW', 40)
    SQLALCHEMY_ECHO: bool = get_bool_env_var('SQLALCHEMY_ECHO', False)

    # Redis Configuration
    REDIS_URL: str = get_env_var('REDIS_URL', 'redis://localhost:6379/0')
    CACHE_TTL_SECONDS: int = get_int_env_var('CACHE_TTL_SECONDS', 300) # 5 minutes

    # Celery Configuration
    CELERY_BROKER_URL: str = get_env_var('CELERY_BROKER_URL', 'redis://localhost:6379/1')
    CELERY_RESULT_BACKEND: str = get_env_var('CELERY_RESULT_BACKEND', 'redis://localhost:6379/2')

    # MQTT Configuration
    MQTT_BROKER_HOST: str = get_env_var('MQTT_BROKER_HOST', 'localhost')
    MQTT_BROKER_PORT: int = get_int_env_var('MQTT_BROKER_PORT', 1883)
    MQTT_INGEST_TOPIC: str = get_env_var('MQTT_INGEST_TOPIC', 'iot/device/+/data')
    MQTT_QOS: int = get_int_env_var('MQTT_QOS', 1)

    # Security & JWT Configuration
    # ENCRYPTION_KEY is critical and MUST be set in the environment
    ENCRYPTION_KEY: str = get_env_var('ENCRYPTION_KEY') # No default for critical key
    JWT_ACCESS_TOKEN_EXPIRES_HOURS: int = get_int_env_var('JWT_ACCESS_TOKEN_EXPIRES_HOURS', 1)
    JWT_REFRESH_TOKEN_EXPIRES_DAYS: int = get_int_env_var('JWT_REFRESH_TOKEN_EXPIRES_DAYS', 30)
    # JWT Cookie settings derived from FLASK_ENV
    JWT_COOKIE_SECURE: bool = FLASK_ENV == 'production'
    JWT_COOKIE_SAMESITE: str = 'Lax'
    JWT_COOKIE_CSRF_PROTECT: bool = True
    JWT_REFRESH_COOKIE_PATH: str = '/api/auth/refresh'

    # CSRF Protection (Flask-WTF)
    WTF_CSRF_ENABLED: bool = True
    WTF_CSRF_SECRET_KEY: str = get_env_var('SECRET_KEY', 'fallback-csrf-key') # Use Flask secret key

    # Rate Limiting (Flask-Limiter)
    RATELIMIT_STORAGE_URI: str = get_env_var('REDIS_URL', 'memory://') # Use Redis if available
    RATELIMIT_STRATEGY: str = 'fixed-window'
    RATELIMIT_HEADERS_ENABLED: bool = True
    # Default limits (can be overridden in specific routes)
    RATELIMIT_DEFAULT: str = get_env_var('RATELIMIT_DEFAULT', '200 per day;50 per hour')

    # CORS Configuration
    CORS_ALLOWED_ORIGINS: List[str] = get_list_env_var('CORS_ALLOWED_ORIGINS', ['*']) # Default to allow all for dev

    # Alerting & Notifications (Secrets loaded from env)
    SMTP_USERNAME: Optional[str] = get_env_var('SMTP_USERNAME')
    SMTP_PASSWORD: Optional[str] = get_env_var('SMTP_PASSWORD')
    SMTP_SERVER: str = get_env_var('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT: int = get_int_env_var('SMTP_PORT', 587)
    ALERT_EMAIL_RECIPIENTS: List[str] = get_list_env_var('ALERT_EMAIL_RECIPIENTS')
    SLACK_WEBHOOK_URL: Optional[str] = get_env_var('SLACK_WEBHOOK_URL')
    ALERT_WEBHOOK_URL: Optional[str] = get_env_var('WEBHOOK_URL')

    # AI Module Settings (Loading simple values, complex structures might need JSON loading)
    HEALTH_CRITICAL_THRESHOLD: float = get_float_env_var('HEALTH_CRITICAL_THRESHOLD', 0.3)
    HEALTH_WARNING_THRESHOLD: float = get_float_env_var('HEALTH_WARNING_THRESHOLD', 0.6)
    ANOMALY_CONTAMINATION: float = get_float_env_var('ANOMALY_CONTAMINATION', 0.1)
    FORECAST_HORIZON: int = get_int_env_var('FORECAST_HORIZON', 12)

    # Paths (relative to the Digital_Twin directory assumed)
    MODEL_PATH: str = get_env_var('MODEL_PATH', '../ANALYTICS/models/')
    CACHE_PATH: str = get_env_var('CACHE_PATH', '../ANALYTICS/analysis_cache/')
    REPORTS_PATH: str = get_env_var('REPORTS_PATH', '../REPORTS/generated/')
    EXPORTS_PATH: str = get_env_var('EXPORTS_PATH', '../EXPORTS/')
    LOG_DIR: str = get_env_var('LOG_DIR', '../LOGS/')

    # Feature Flags
    ENABLE_DATA_GENERATOR: bool = get_bool_env_var('ENABLE_DATA_GENERATOR', False)
    ENABLE_MODEL_RETRAINING: bool = get_bool_env_var('ENABLE_MODEL_RETRAINING', True)

    def __init__(self):
        # --- Validation for Critical Settings ---
        if not self.SECRET_KEY or 'default' in self.SECRET_KEY:
            log.warning("SECRET_KEY is using a default or is missing. Set a strong, unique key in production.")
        if not self.JWT_SECRET_KEY or 'default' in self.JWT_SECRET_KEY:
            log.warning("JWT_SECRET_KEY is using a default or is missing. Set a strong, unique key in production.")
        if not self.ENCRYPTION_KEY:
            log.critical("CRITICAL: ENCRYPTION_KEY is not set. Database encryption/decryption will fail.")
            # Depending on strictness, you might want to raise an error here
            # raise ValueError("Missing required environment variable: ENCRYPTION_KEY")
        elif len(self.ENCRYPTION_KEY) < 40: # Basic check for base64 length
             log.warning("ENCRYPTION_KEY seems short. Ensure it's a correctly base64-encoded 32-byte key.")

        log.info(f"Configuration loaded for environment: {self.FLASK_ENV}")
        log.info(f"Database URL: {self.DATABASE_URL.split('@')[0] if '@' in self.DATABASE_URL else self.DATABASE_URL}") # Avoid logging password
        log.info(f"Redis URL: {self.REDIS_URL}")
        log.info(f"MQTT Broker: {self.MQTT_BROKER_HOST}:{self.MQTT_BROKER_PORT}")

# Global instance
config = Config()

# --- Example Usage ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # Basic logging for demo

    print("\n--- Configuration Values ---")
    print(f"Environment: {config.FLASK_ENV}")
    print(f"Debug Mode: {config.DEBUG}")
    print(f"Database URL (masked): {config.DATABASE_URL.split('@')[0] if '@' in config.DATABASE_URL else config.DATABASE_URL}")
    print(f"Redis URL: {config.REDIS_URL}")
    print(f"MQTT Broker: {config.MQTT_BROKER_HOST}:{config.MQTT_BROKER_PORT}")
    print(f"Encryption Key Set: {'Yes' if config.ENCRYPTION_KEY else 'NO (CRITICAL!)'}")
    print(f"Allowed CORS Origins: {config.CORS_ALLOWED_ORIGINS}")
    print(f"Report Path: {config.REPORTS_PATH}")
    print(f"Email Recipients: {config.ALERT_EMAIL_RECIPIENTS}")
    print(f"Health Critical Threshold: {config.HEALTH_CRITICAL_THRESHOLD}")
    print(f"Data Generator Enabled: {config.ENABLE_DATA_GENERATOR}")

    # You can access config values directly:
    # from CONFIG.app_config import config
    # db_url = config.DATABASE_URL