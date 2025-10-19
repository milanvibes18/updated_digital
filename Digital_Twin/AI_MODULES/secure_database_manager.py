#!/usr/bin/env python3
"""
Secure Database Manager
Handles all database interactions, including data encryption/decryption,
user authentication, data integrity, and audit logging.
"""

import sqlite3
import json
import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from cryptography.fernet import Fernet
import base64
import os
import sys

# --- Added from File 2 ---
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash

# --- Merged Feature (Config from File 2) ---
# Add project root to path to find CONFIG
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Use a relative path to the config file
    from CONFIG.app_config import config
    DEFAULT_PRIMARY_DB = config.database.primary_path
    DEFAULT_USERS_DB = config.database.users_db_path
except ImportError:
    # Fallback for script execution (combines defaults from both files)
    print("Warning: CONFIG.app_config not found. Using default paths.")
    DEFAULT_PRIMARY_DB = 'DATABASE/secure_database.db'
    DEFAULT_USERS_DB = 'DATABASE/users.db'

class SecureDatabaseManager:
    """
    Manages secure database operations, including user auth, data encryption,
    integrity verification, and audit logging.
    """

    # --- Merged Feature (__init__) ---
    def __init__(self, 
                 db_path=DEFAULT_PRIMARY_DB, 
                 users_db_path=DEFAULT_USERS_DB, 
                 encryption_key_path="CONFIG/encryption.key", 
                 salt_key_path="CONFIG/salt.key"):
        
        self.db_path = db_path
        self.users_db_path = users_db_path  # From File 2
        self.encryption_key_path = encryption_key_path
        self.salt_key_path = salt_key_path
        
        # Setup logging (From File 1)
        self.logger = self._setup_logging()
        
        # Ensure directories exist (Merged)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.users_db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.encryption_key_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.salt_key_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize encryption (From File 1 - Superior)
        self.fernet = self._initialize_encryption()
        
        # Initialize databases (Merged)
        self._initialize_database()  # Initializes data/audit/device tables
        self.init_users_db()      # Initializes user table

    # --- Kept from File 1 (Superior Logging) ---
    def _setup_logging(self):
        """Setup security-focused logging."""
        logger = logging.getLogger('SecureDatabaseManager')
        
        # Prevent duplicate handlers if already configured
        if logger.hasHandlers():
            return logger
            
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        Path("LOGS").mkdir(exist_ok=True)
        
        handler = logging.FileHandler('LOGS/digital_twin_security.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

    # --- Kept from File 1 (Superior Key/Salt Generation) ---
    def _generate_key(self):
        """Generate a new encryption key."""
        return Fernet.generate_key()

    # --- Kept from File 1 ---
    def _generate_salt(self):
        """Generate a new salt for key derivation."""
        return secrets.token_bytes(32)

    # --- Kept from File 1 (Superior Initialization) ---
    def _initialize_encryption(self):
        """Initialize encryption with key management."""
        try:
            # Load or generate encryption key
            if os.path.exists(self.encryption_key_path):
                with open(self.encryption_key_path, 'rb') as key_file:
                    key = key_file.read()
            else:
                key = self._generate_key()
                with open(self.encryption_key_path, 'wb') as key_file:
                    key_file.write(key)
                os.chmod(self.encryption_key_path, 0o600)  # Restrict permissions
                
            # Load or generate salt
            if os.path.exists(self.salt_key_path):
                with open(self.salt_key_path, 'rb') as salt_file:
                    salt = salt_file.read()
            else:
                salt = self._generate_salt()
                with open(self.salt_key_path, 'wb') as salt_file:
                    salt_file.write(salt)
                os.chmod(self.salt_key_path, 0o600)
                
            # Create Fernet instance
            fernet = Fernet(key)
            self.logger.info("Encryption initialized successfully")
            return fernet
            
        except Exception as e:
            self.logger.error(f"Failed to initialize encryption: {e}")
            raise

    # --- Kept from File 1 (Core Secure Tables) ---
    def _initialize_database(self):
        """Initialize the secure database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create health data table (with data_hash)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS health_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        device_id TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        encrypted_data TEXT NOT NULL,
                        data_hash TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create audit log table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        action TEXT NOT NULL,
                        user_id TEXT,
                        table_name TEXT,
                        record_id TEXT,
                        old_values TEXT,
                        new_values TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        ip_address TEXT,
                        user_agent TEXT
                    )
                ''')
                
                # Create device registry table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS device_registry (
                        device_id TEXT PRIMARY KEY,
                        device_name TEXT NOT NULL,
                        device_type TEXT NOT NULL,
                        location TEXT,
                        status TEXT DEFAULT 'active',
                        encrypted_config TEXT,
                        last_seen DATETIME,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create user sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        encrypted_session_data TEXT,
                        expires_at DATETIME NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                self.logger.info("Secure data database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize secure data database: {e}")
            raise

    # --- Added from File 2 (User DB Initialization) ---
    def init_users_db(self):
        """Initializes the user database schema if it doesn't exist."""
        try:
            with sqlite3.connect(self.users_db_path) as conn:
                conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """)
                self.logger.info("User database initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize user database: {e}")
            raise

    # --- Kept from File 1 (Superior Encryption) ---
    def encrypt_data(self, data):
        """Encrypt data using Fernet encryption."""
        try:
            if isinstance(data, dict):
                data = json.dumps(data)
            elif not isinstance(data, (str, bytes)):
                data = str(data)
                
            if isinstance(data, str):
                data = data.encode()
                
            encrypted = self.fernet.encrypt(data)
            return base64.b64encode(encrypted).decode()
            
        except Exception as e:
            self.logger.error(f"Failed to encrypt data: {e}")
            raise

    # --- Kept from File 1 (Superior Decryption) ---
    def decrypt_data(self, encrypted_data):
        """Decrypt data using Fernet encryption."""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt data: {e}")
            raise

    # --- Kept from File 1 (Critical for Integrity) ---
    def _calculate_hash(self, data):
        """Calculate SHA-256 hash of data for integrity verification."""
        if isinstance(data, dict):
            data = json.dumps(data, sort_keys=True)
        elif not isinstance(data, str):
            data = str(data)
            
        return hashlib.sha256(data.encode()).hexdigest()

    # --- Kept from File 1 (Critical for Auditing) ---
    def log_audit_event(self, action, user_id=None, table_name=None, record_id=None, 
                        old_values=None, new_values=None, ip_address=None, user_agent=None):
        """Log security audit events."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO audit_log (action, user_id, table_name, record_id, 
                                         old_values, new_values, ip_address, user_agent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (action, user_id, table_name, record_id, 
                      json.dumps(old_values) if old_values else None,
                      json.dumps(new_values) if new_values else None,
                      ip_address, user_agent))
                
                conn.commit()
                self.logger.info(f"Audit event logged: {action}")
                
        except Exception as e:
            self.logger.error(f"Failed to log audit event: {e}")

    # --- Merged Feature (User Creation from File 2 + Audit from File 1) ---
    def create_user(self, username: str, password: str, email: str = None) -> bool:
        """Creates a new user with a hashed password."""
        password_hash = generate_password_hash(password)
        try:
            with sqlite3.connect(self.users_db_path) as conn:
                conn.execute(
                    "INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
                    (username, password_hash, email)
                )
                conn.commit()
            self.logger.info(f"User '{username}' created successfully.")
            # Add audit logging
            self.log_audit_event(
                action="USER_CREATED",
                user_id=username,
                table_name="users",
                new_values={"username": username, "email": email}
            )
            return True
        except sqlite3.IntegrityError:
            self.logger.warning(f"Failed to create user. Username '{username}' or email '{email}' already exists.")
            self.log_audit_event(action="USER_CREATE_FAILED", new_values={"username": username, "reason": "IntegrityError"})
            return False
        except sqlite3.Error as e:
            self.logger.error(f"Error creating user '{username}': {e}")
            self.log_audit_event(action="USER_CREATE_FAILED", new_values={"username": username, "reason": str(e)})
            return False

    # --- Merged Feature (User Auth from File 2 + Audit from File 1) ---
    def authenticate_user(self, username: str, password: str) -> bool:
        """Authenticates a user against the stored hash."""
        try:
            with sqlite3.connect(self.users_db_path) as conn:
                conn.row_factory = sqlite3.Row  # Use Row factory for dict-like access
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT password_hash FROM users WHERE username = ?",
                    (username,)
                )
                user = cursor.fetchone()
            
            if user and check_password_hash(user['password_hash'], password):
                self.logger.info(f"User '{username}' authenticated successfully.")
                self.log_audit_event(action="AUTH_SUCCESS", user_id=username)
                return True
            
            self.logger.warning(f"Authentication failed for user '{username}'.")
            self.log_audit_event(action="AUTH_FAILED", user_id=username)
            return False
        except sqlite3.Error as e:
            self.logger.error(f"Error authenticating user '{username}': {e}")
            self.log_audit_event(action="AUTH_ERROR", user_id=username, new_values={"error": str(e)})
            return False

    # --- Kept from File 1 (Superior Secure Insert) ---
    def insert_health_data(self, device_id, data, user_id=None):
        """Insert encrypted health data with integrity verification."""
        try:
            # Calculate hash before encryption
            data_hash = self._calculate_hash(data)
            
            # Encrypt the data
            encrypted_data = self.encrypt_data(data)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO health_data (device_id, timestamp, encrypted_data, data_hash)
                    VALUES (?, ?, ?, ?)
                ''', (device_id, datetime.now(), encrypted_data, data_hash))
                
                record_id = cursor.lastrowid
                conn.commit()
                
                # Log audit event
                self.log_audit_event(
                    action="INSERT",
                    user_id=user_id,
                    table_name="health_data",
                    record_id=str(record_id),
                    new_values={"device_id": device_id, "data_hash": data_hash}
                )
                
                self.logger.info(f"Health data inserted for device {device_id}")
                return record_id
                
        except Exception as e:
            self.logger.error(f"Failed to insert health data: {e}")
            raise

    # --- Kept from File 1 (Superior Secure Get with Integrity Check) ---
    def get_health_data(self, device_id=None, start_date=None, end_date=None, limit=None):
        """Retrieve and decrypt health data with integrity verification."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row # Use Row factory
                cursor = conn.cursor()
                
                query = "SELECT id, device_id, timestamp, encrypted_data, data_hash FROM health_data WHERE 1=1"
                params = []
                
                if device_id:
                    query += " AND device_id = ?"
                    params.append(device_id)
                    
                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date)
                    
                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date)
                    
                query += " ORDER BY timestamp DESC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                results = []
                for row in rows:
                    try:
                        # Decrypt data
                        decrypted_data = self.decrypt_data(row['encrypted_data'])
                        
                        # Parse JSON if applicable
                        try:
                            data = json.loads(decrypted_data)
                        except json.JSONDecodeError:
                            data = decrypted_data
                        
                        # Verify integrity
                        calculated_hash = self._calculate_hash(data)
                        if calculated_hash != row['data_hash']:
                            self.logger.warning(f"DATA TAMPERING DETECTED for record {row['id']}")
                            self.log_audit_event(action="INTEGRITY_FAIL", record_id=row['id'], table_name="health_data")
                            continue
                        
                        results.append({
                            'id': row['id'],
                            'device_id': row['device_id'],
                            'timestamp': row['timestamp'],
                            'data': data
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Failed to process record {row['id']}: {e}")
                        continue
                
                return results
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve health data: {e}")
            raise
    
    # --- Merged Feature (Pandas from File 2, using Secure Get from File 1) ---
    def get_health_data_as_dataframe(self, device_id=None, start_date=None, end_date=None, limit=None) -> pd.DataFrame:
        """
        Retrieve and decrypt health data, verify integrity, and return as a pandas DataFrame.
        """
        try:
            # Use the existing, secure method to get the data
            secure_data = self.get_health_data(device_id, start_date, end_date, limit)
            
            if not secure_data:
                self.logger.info("No health data found for DataFrame.")
                return pd.DataFrame()
            
            # Convert the list of dicts to a DataFrame
            df = pd.DataFrame(secure_data)
            
            # Unpack the 'data' dictionary (which holds metrics) into separate columns
            if 'data' in df.columns and not df['data'].isnull().all():
                # Ensure all 'data' entries are dicts, fill non-dicts with None or {}
                valid_data = df['data'].apply(lambda x: x if isinstance(x, dict) else {})
                data_df = pd.DataFrame(valid_data.tolist(), index=df.index)
                df = pd.concat([df.drop('data', axis=1), data_df], axis=1)
            
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            self.logger.info(f"Successfully converted {len(df)} secure records to DataFrame.")
            return df
        
        except Exception as e:
            self.logger.error(f"Failed to convert health data to DataFrame: {e}")
            return pd.DataFrame()

    # --- All Device/Session/Audit methods below are Kept from File 1 ---

    def register_device(self, device_id, device_name, device_type, location=None, config=None):
        """Register a new device with encrypted configuration."""
        try:
            encrypted_config = None
            if config:
                encrypted_config = self.encrypt_data(config)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO device_registry 
                    (device_id, device_name, device_type, location, encrypted_config, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (device_id, device_name, device_type, location, encrypted_config, datetime.now()))
                
                conn.commit()
                
                self.log_audit_event(
                    action="DEVICE_REGISTERED",
                    table_name="device_registry",
                    record_id=device_id,
                    new_values={"device_name": device_name, "device_type": device_type}
                )
                
                self.logger.info(f"Device registered: {device_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to register device: {e}")
            raise
    
    def get_devices(self):
        """Get list of registered devices."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT device_id, device_name, device_type, location, status, last_seen, created_at
                    FROM device_registry
                    ORDER BY device_name
                ''')
                
                rows = cursor.fetchall()
                
                devices = [dict(row) for row in rows]
                return devices
                
        except Exception as e:
            self.logger.error(f"Failed to get devices: {e}")
            raise
    
    def update_device_status(self, device_id, status):
        """Update device status and last seen timestamp."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE device_registry 
                    SET status = ?, last_seen = ?
                    WHERE device_id = ?
                ''', (status, datetime.now(), device_id))
                
                conn.commit()
                
                self.log_audit_event(
                    action="DEVICE_STATUS_UPDATED",
                    table_name="device_registry",
                    record_id=device_id,
                    new_values={"status": status}
                )
                
        except Exception as e:
            self.logger.error(f"Failed to update device status: {e}")
            raise
    
    def create_session(self, user_id, session_data, expires_in_minutes=60):
        """Create encrypted user session."""
        try:
            session_id = secrets.token_urlsafe(32)
            encrypted_session_data = self.encrypt_data(session_data)
            expires_at = datetime.now() + timedelta(minutes=expires_in_minutes)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO user_sessions (session_id, user_id, encrypted_session_data, expires_at)
                    VALUES (?, ?, ?, ?)
                ''', (session_id, user_id, encrypted_session_data, expires_at))
                
                conn.commit()
                
                self.log_audit_event(
                    action="SESSION_CREATED",
                    user_id=user_id,
                    table_name="user_sessions",
                    record_id=session_id
                )
                
                return session_id
                
        except Exception as e:
            self.logger.error(f"Failed to create session: {e}")
            raise
    
    def get_session(self, session_id):
        """Retrieve and decrypt user session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT user_id, encrypted_session_data, expires_at
                    FROM user_sessions
                    WHERE session_id = ? AND expires_at > ?
                ''', (session_id, datetime.now()))
                
                row = cursor.fetchone()
                
                if row:
                    session_data = json.loads(self.decrypt_data(row['encrypted_session_data']))
                    
                    return {
                        'user_id': row['user_id'],
                        'session_data': session_data,
                        'expires_at': row['expires_at']
                    }
                
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get session: {e}")
            raise
    
    def delete_session(self, session_id):
        """Delete user session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('DELETE FROM user_sessions WHERE session_id = ?', (session_id,))
                conn.commit()
                
                self.log_audit_event(
                    action="SESSION_DELETED",
                    table_name="user_sessions",
                    record_id=session_id
                )
                
        except Exception as e:
            self.logger.error(f"Failed to delete session: {e}")
            raise
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('DELETE FROM user_sessions WHERE expires_at < ?', (datetime.now(),))
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    self.logger.info(f"Cleaned up {deleted_count} expired sessions")
                    self.log_audit_event(action="SESSION_CLEANUP", new_values={"deleted_count": deleted_count})
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired sessions: {e}")
    
    def get_audit_logs(self, limit=100, action_filter=None, start_date=None, end_date=None):
        """Retrieve audit logs with filtering."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = '''
                    SELECT action, user_id, table_name, record_id, old_values, 
                           new_values, timestamp, ip_address, user_agent
                    FROM audit_log WHERE 1=1
                '''
                params = []
                
                if action_filter:
                    query += " AND action = ?"
                    params.append(action_filter)
                    
                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date)
                    
                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date)
                    
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                logs = []
                for row in rows:
                    log_entry = dict(row)
                    log_entry['old_values'] = json.loads(row['old_values']) if row['old_values'] else None
                    log_entry['new_values'] = json.loads(row['new_values']) if row['new_values'] else None
                    logs.append(log_entry)
                
                return logs
                
        except Exception as e:
            self.logger.error(f"Failed to get audit logs: {e}")
            raise
    
    def backup_database(self, backup_path=None):
        """Create a backup of the main database."""
        try:
            if not backup_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_dir = "SECURITY/data_backups"
                Path(backup_dir).mkdir(parents=True, exist_ok=True)
                backup_path = f"{backup_dir}/backup_main_{timestamp}.db"
            
            with sqlite3.connect(self.db_path) as source:
                with sqlite3.connect(backup_path) as backup:
                    source.backup(backup)
            
            self.logger.info(f"Main database backed up to {backup_path}")
            self.log_audit_event(
                action="DATABASE_BACKUP",
                table_name="main",
                new_values={"backup_path": backup_path}
            )
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Failed to backup main database: {e}")
            raise

    def close(self):
        """Close database connections and cleanup."""
        try:
            self.logger.info("Database manager closed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

# --- Merged Feature (Combined __main__ for Testing) ---
if __name__ == "__main__":
    # Setup basic logging to console for testing
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize secure database manager
    db_manager = SecureDatabaseManager()
    
    print("\n--- Testing User Management (from File 2) ---")
    db_manager.create_user("admin", "securepassword123", "admin@digitaltwin.com")
    db_manager.create_user("admin", "anotherpassword", "admin@digitaltwin.com") # Should fail
    
    print("\n--- Testing Authentication (from File 2) ---")
    print(f"Auth 'admin'/'securepassword123': {db_manager.authenticate_user('admin', 'securepassword123')}")
    print(f"Auth 'admin'/'wrongpassword': {db_manager.authenticate_user('admin', 'wrongpassword')}")
    
    print("\n--- Testing Device & Data (from File 1) ---")
    # Register a test device
    db_manager.register_device(
        device_id="sensor_001",
        device_name="Temperature Sensor 1",
        device_type="temperature",
        location="Factory Floor A",
        config={"sampling_rate": 60, "threshold": 75.0}
    )
    
    # Insert test data
    test_data = {
        "temperature": 72.5,
        "humidity": 45.2,
        "pressure": 1013.25,
        "vibration": 0.1
    }
    
    db_manager.insert_health_data("sensor_001", test_data, user_id="admin")
    
    # Retrieve data (original method)
    data = db_manager.get_health_data("sensor_001", limit=10)
    print(f"\nRetrieved {len(data)} records (raw list):")
    print(json.dumps(data, indent=2))
    
    print("\n--- Testing Pandas DataFrame (Merged Feature) ---")
    df = db_manager.get_health_data_as_dataframe(device_id="sensor_001", limit=10)
    print(f"Retrieved DataFrame with {len(df)} rows and columns: {list(df.columns)}")
    print(df.head())
    
    # Get devices
    devices = db_manager.get_devices()
    print(f"\nRegistered devices: {len(devices)}")
    
    # Get audit logs
    logs = db_manager.get_audit_logs(limit=5)
    print(f"\nRecent audit events: {len(logs)}")
    print(json.dumps(logs, indent=2))
    
    # Create backup
    backup_path = db_manager.backup_database()
    print(f"\nBackup created: {backup_path}")
    
    # Cleanup sessions
    db_manager.cleanup_expired_sessions()
    
    # Close
    db_manager.close()