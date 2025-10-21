#!/usr/bin/env python3
"""
Application Configuration for Digital Twin System v2.1
Centralized configuration management loading from multiple JSON files.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# --- Configuration Loading ---

CONFIG_DIR = Path(__file__).parent.resolve()
ENV = os.getenv('FLASK_ENV', 'development')

log = logging.getLogger(__name__)

def load_json_config(file_path: Path) -> Dict[str, Any]:
    """Loads a JSON configuration file."""
    if file_path.exists():
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            log.error(f"Error decoding JSON from {file_path}: {e}")
            return {}
        except Exception as e:
            log.error(f"Error loading configuration file {file_path}: {e}")
            return {}
    else:
        log.warning(f"Configuration file not found: {file_path}")
        return {}

def merge_configs(base: Dict[str, Any], *overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Deeply merges dictionaries, with later dictionaries overriding earlier ones."""
    merged = base.copy()
    for override in overrides:
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = value
    return merged

class ApplicationConfig:
    """
    Main application configuration class. Loads and merges configurations.
    Serves as the central hub for accessing configuration settings.
    """

    def __init__(self, env: Optional[str] = None):
        self.env = env or ENV
        self.debug = self.env == 'development'
        self.config_data: Dict[str, Any] = {}

        self._load_all_configs()
        self._apply_env_overrides() # Allow environment variables to override JSON
        self._validate_config()

        log.info(f"Configuration loaded for environment: {self.env}")

    def _load_all_configs(self):
        """Loads base and environment-specific JSON configuration files."""
        # Define the base configuration files to load
        base_configs_to_load = {
            "system": CONFIG_DIR / "system_config.json",
            "alerts": CONFIG_DIR / "alert_config.json",
            "health_score": CONFIG_DIR / "health_score_config.json",
            "recommendation": CONFIG_DIR / "recommendation_config.json",
            # Add other base config files here if necessary
        }

        # Load all base configurations
        loaded_base_configs = {}
        for key, path in base_configs_to_load.items():
            config_content = load_json_config(path)
            # Use the 'configuration' key if present, otherwise use the whole file content
            loaded_base_configs[key] = config_content.get('configuration', config_content)

        # Merge base configs (e.g., system config is the base)
        # Order matters if there are overlapping keys; here, specific configs override 'system'
        self.config_data = merge_configs(
            loaded_base_configs.get("system", {}),
            loaded_base_configs.get("alerts", {}),
            loaded_base_configs.get("health_score", {}),
            loaded_base_configs.get("recommendation", {})
        )

        # Load and merge environment-specific config if it exists
        # Example: 'development_config.json' or 'production_config.json'
        # These files would contain only the keys that need overriding for that environment.
        env_specific_file = CONFIG_DIR / f"{self.env}_config.json"
        env_specific_config = load_json_config(env_specific_file)
        if env_specific_config:
            log.info(f"Applying environment overrides from {env_specific_file}")
            self.config_data = merge_configs(self.config_data, env_specific_config)

    def _apply_env_overrides(self):
        """Override config values with environment variables if they exist."""
        # Example: DATABASE_URL, SECRET_KEY, JWT_SECRET_KEY
        # These are critical and often set via environment
        db_url = os.getenv('DATABASE_URL')
        if db_url:
            # Assuming DB config is under a 'database' key from system_config.json
            if 'database' not in self.config_data: self.config_data['database'] = {}
            # Update the specific URL/path. Adjust key based on your structure.
            self.config_data['database']['url'] = db_url # Example key
            log.info("Overrode database URL from environment variable.")

        secret_key = os.getenv('SECRET_KEY')
        if secret_key:
            if 'security' not in self.config_data: self.config_data['security'] = {}
            self.config_data['security']['flask_secret_key'] = secret_key # Example key
            log.info("Overrode Flask SECRET_KEY from environment variable.")

        jwt_secret = os.getenv('JWT_SECRET_KEY')
        if jwt_secret:
             if 'security' not in self.config_data: self.config_data['security'] = {}
             self.config_data['security']['jwt_secret'] = jwt_secret # Example key
             log.info("Overrode JWT_SECRET_KEY from environment variable.")

        # Override notification secrets (example)
        smtp_user = os.getenv('SMTP_USERNAME')
        if smtp_user and 'notification_channels' in self.config_data and 'email' in self.config_data['notification_channels']:
             self.config_data['notification_channels']['email']['username'] = smtp_user
             log.info("Overrode SMTP username from environment variable.")
             
        smtp_pass = os.getenv('SMTP_PASSWORD')
        if smtp_pass and 'notification_channels' in self.config_data and 'email' in self.config_data['notification_channels']:
             self.config_data['notification_channels']['email']['password'] = smtp_pass
             log.info("Overrode SMTP password from environment variable.")
             
        # Add more environment variable overrides as needed...

    def _validate_config(self):
        """Basic validation for critical configuration settings."""
        if not self.get('security.jwt_secret'):
            log.warning("JWT_SECRET_KEY is not set. Security risk!")
        if not self.get('database.url'): # Check for the effective DB URL key
             log.warning("Database URL is not set.")
        # Add more validation checks...

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Access nested configuration values using dot notation.
        Example: config.get('database.primary.pool_size', 5)
        """
        keys = key_path.split('.')
        value = self.config_data
        try:
            for key in keys:
                if isinstance(value, dict):
                    value = value[key]
                else:
                    return default
            return value
        except (KeyError, TypeError):
            return default

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access config['database']."""
        if key in self.config_data:
            return self.config_data[key]
        else:
            raise KeyError(f"Configuration key '{key}' not found.")

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access config.database."""
        if name in self.config_data:
            # Recursively wrap dictionaries to allow nested attribute access
            value = self.config_data[name]
            if isinstance(value, dict):
                 # Simple wrapper for nested access
                 class AttrDict(dict):
                      def __getattr__(self, key):
                           try:
                                return self[key]
                           except KeyError:
                                raise AttributeError(f"'{name}' object has no attribute '{key}'")
                 return AttrDict(value)
            return value
        else:
            raise AttributeError(f"'ApplicationConfig' object has no attribute '{name}'")

    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.env == 'production'

    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.env == 'development'

    def to_dict(self) -> Dict[str, Any]:
        """Return the entire configuration dictionary."""
        # Be careful about exposing secrets if this is used widely
        return self.config_data

    def export_config(self, filepath: Optional[str] = None):
        """Export effective configuration to JSON file (excluding sensitive keys)."""
        if not filepath:
            filepath = CONFIG_DIR / f"effective_config_{self.env}.json"
        else:
             filepath = Path(filepath)

        # Create a copy and remove potential secrets before exporting
        exportable_config = json.loads(json.dumps(self.config_data, default=str)) # Deep copy
        
        # Define keys to scrub (adjust as needed)
        keys_to_scrub = ['password', 'secret_key', 'jwt_secret', 'encryption_key']
        
        def scrub_dict(d):
            if not isinstance(d, dict):
                return d
            scrubbed = {}
            for k, v in d.items():
                if any(secret in k.lower() for secret in keys_to_scrub):
                    scrubbed[k] = "***REDACTED***"
                elif isinstance(v, dict):
                    scrubbed[k] = scrub_dict(v)
                elif isinstance(v, list):
                     scrubbed[k] = [scrub_dict(item) for item in v]
                else:
                    scrubbed[k] = v
            return scrubbed

        scrubbed_data = scrub_dict(exportable_config)

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(scrubbed_data, f, indent=2)
            log.info(f"Effective (scrubbed) configuration exported to {filepath}")
        except Exception as e:
            log.error(f"Failed to export configuration: {e}")


# Global configuration instance (loads config based on FLASK_ENV)
config = ApplicationConfig()

# Example Usage (can be placed in your main app file):
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # Basic logging for demo

    print(f"--- Loading Configuration for ENV: {config.env} ---")

    # Accessing config values
    print(f"App Name: {config.get('application.name', 'Default App')}") # Using get()
    print(f"Database URL: {config.get('database.url', 'Not Set')}")
    print(f"Alerts Enabled: {config.alerts.get('enabled')}") # Using attribute access

    # Nested access
    try:
         print(f"Temp Warning Threshold: {config.thresholds['temperature']['warning']}") # Dict access
         print(f"JWT Expiry (Hours): {config.security.jwt_expiry_hours}") # Attribute access
    except (KeyError, AttributeError, TypeError) as e:
         print(f"Could not access nested key: {e}")

    # Print the structure (be mindful of secrets)
    # print("\n--- Full Effective Config (potentially includes secrets) ---")
    # import pprint
    # pprint.pprint(config.to_dict())

    # Export scrubbed config
    config.export_config()