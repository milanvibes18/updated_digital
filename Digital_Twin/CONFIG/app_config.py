#!/usr/bin/env python3
"""
Application Configuration for Digital Twin System
Centralized configuration management with environment-specific settings.
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    primary_path: str = "DATABASE/health_data.db"
    secure_path: str = "DATABASE/secure_database.db"
    backup_path: str = "SECURITY/data_backups/"
    pool_size: int = 10
    timeout: int = 30
    enable_wal: bool = True
    backup_interval_hours: int = 6
    
@dataclass
class SecurityConfig:
    """Security configuration settings"""
    encryption_key_path: str = "CONFIG/encryption.key"
    salt_key_path: str = "CONFIG/salt.key"
    jwt_secret: Optional[str] = None
    jwt_expiry_hours: int = 24
    session_timeout_minutes: int = 60
    max_login_attempts: int = 5
    enable_audit_logging: bool = True
    audit_retention_days: int = 365
    
@dataclass
class WebSocketConfig:
    """WebSocket configuration settings"""
    enabled: bool = True
    max_connections: int = 1000
    heartbeat_interval: int = 30
    message_queue_size: int = 10000
    compression: bool = True
    redis_url: str = "redis://localhost:6379/0"
    
@dataclass
class AnalyticsConfig:
    """Analytics and AI configuration settings"""
    enabled: bool = True
    real_time_processing: bool = True
    batch_size: int = 1000
    model_path: str = "ANALYTICS/models/"
    cache_path: str = "ANALYTICS/analysis_cache/"
    prediction_intervals: Dict[str, int] = None
    
    def __post_init__(self):
        if self.prediction_intervals is None:
            self.prediction_intervals = {
                "short_term_minutes": 60,
                "medium_term_hours": 24,
                "long_term_days": 7
            }
            
@dataclass
class AlertConfig:
    """Alert management configuration settings"""
    enabled: bool = True
    channels: List[str] = None
    thresholds: Dict[str, Dict[str, float]] = None
    escalation_settings: Dict[str, int] = None
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = ["websocket", "email", "sms"]
            
        if self.thresholds is None:
            self.thresholds = {
                "temperature": {"warning": 80, "critical": 95},
                "pressure": {"warning": 950, "critical": 1000},
                "vibration": {"warning": 0.5, "critical": 0.8},
                "efficiency": {"warning_low": 70, "critical_low": 50}
            }
            
        if self.escalation_settings is None:
            self.escalation_settings = {
                "warning_delay_minutes": 5,
                "critical_delay_minutes": 1,
                "max_retries": 3
            }

@dataclass
class MonitoringConfig:
    """System monitoring configuration settings"""
    system_metrics_enabled: bool = True
    interval_seconds: int = 60
    performance_thresholds: Dict[str, int] = None
    health_check_endpoints: List[str] = None
    
    def __post_init__(self):
        if self.performance_thresholds is None:
            self.performance_thresholds = {
                "cpu_percent": 80,
                "memory_percent": 85,
                "disk_percent": 90,
                "response_time_ms": 1000
            }
            
        if self.health_check_endpoints is None:
            self.health_check_endpoints = ["/health", "/api/health", "/metrics"]

@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    level: str = "INFO"
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    rotation_enabled: bool = True
    max_size_mb: int = 100
    backup_count: int = 5
    log_to_file: bool = True
    log_to_console: bool = False
    log_to_remote: bool = False

class ApplicationConfig:
    """Main application configuration class"""
    
    def __init__(self, env: str = None):
        self.env = env or os.getenv('FLASK_ENV', 'development')
        self.debug = self.env == 'development'
        
        # Initialize configuration sections
        self.database = DatabaseConfig()
        self.security = SecurityConfig()
        self.websocket = WebSocketConfig()
        self.analytics = AnalyticsConfig()
        self.alerts = AlertConfig()
        self.monitoring = MonitoringConfig()
        self.logging = LoggingConfig()
        
        # Application settings
        self.app_name = "Digital Twin System"
        self.app_version = "2.0.0"
        self.timezone = "UTC"
        
        # Load environment-specific overrides
        self._load_environment_config()
        
        # Validate configuration
        self._validate_config()
    
    def _load_environment_config(self):
        """Load environment-specific configuration overrides"""
        config_file = Path("CONFIG") / f"{self.env}_config.json"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    env_config = json.load(f)
                    self._apply_config_overrides(env_config)
            except Exception as e:
                print(f"Warning: Failed to load {self.env} config: {e}")
    
    def _apply_config_overrides(self, overrides: Dict):
        """Apply configuration overrides from environment config"""
        for section, settings in overrides.items():
            if hasattr(self, section):
                config_section = getattr(self, section)
                for key, value in settings.items():
                    if hasattr(config_section, key):
                        setattr(config_section, key, value)
    
    def _validate_config(self):
        """Validate configuration settings"""
        # Ensure required directories exist
        required_dirs = [
            Path("DATABASE"),
            Path("SECURITY/data_backups"),
            Path("ANALYTICS/models"),
            Path("ANALYTICS/analysis_cache"),
            Path("LOGS"),
            Path("CONFIG")
        ]
        
        for directory in required_dirs:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Validate critical settings
        if not self.security.jwt_secret:
            self.security.jwt_secret = os.getenv('JWT_SECRET', self._generate_secret_key())
    
    def _generate_secret_key(self) -> str:
        """Generate a secret key if not provided"""
        import secrets
        return secrets.token_hex(32)
    
    def get_database_url(self) -> str:
        """Get database URL for SQLAlchemy"""
        return f"sqlite:///{self.database.primary_path}"
    
    def get_redis_url(self) -> str:
        """Get Redis URL for caching and WebSocket"""
        return self.websocket.redis_url
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.env == 'production'
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.env == 'development'
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            'app_name': self.app_name,
            'app_version': self.app_version,
            'environment': self.env,
            'debug': self.debug,
            'timezone': self.timezone,
            'database': self.database.__dict__,
            'security': {k: v for k, v in self.security.__dict__.items() if 'secret' not in k.lower()},
            'websocket': self.websocket.__dict__,
            'analytics': self.analytics.__dict__,
            'alerts': self.alerts.__dict__,
            'monitoring': self.monitoring.__dict__,
            'logging': self.logging.__dict__
        }
    
    def export_config(self, filepath: str = None):
        """Export configuration to JSON file"""
        if not filepath:
            filepath = f"CONFIG/exported_config_{self.env}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

# Global configuration instance
config = ApplicationConfig()

# Environment-specific configurations
class DevelopmentConfig(ApplicationConfig):
    def __init__(self):
        super().__init__('development')
        self.debug = True
        self.logging.level = "DEBUG"
        self.logging.log_to_console = True

class ProductionConfig(ApplicationConfig):
    def __init__(self):
        super().__init__('production')
        self.debug = False
        self.logging.level = "INFO"
        self.logging.log_to_console = False
        self.security.enable_audit_logging = True
        self.monitoring.system_metrics_enabled = True

class TestingConfig(ApplicationConfig):
    def __init__(self):
        super().__init__('testing')
        self.debug = True
        self.database.primary_path = ":memory:"
        self.database.secure_path = ":memory:"
        self.websocket.enabled = False

# Configuration factory
def get_config(env: str = None) -> ApplicationConfig:
    """Get configuration based on environment"""
    env = env or os.getenv('FLASK_ENV', 'development')
    
    config_map = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig
    }
    
    config_class = config_map.get(env, ApplicationConfig)
    return config_class()

# Usage example:
# from CONFIG.app_config import config, get_config
# 
# # Use default config
# print(config.database.primary_path)
# 
# # Or get environment-specific config
# prod_config = get_config('production')