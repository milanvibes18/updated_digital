#!/usr/bin/env python3
"""
Secure Database Manager
Handles all database interactions, including data encryption/decryption,
user authentication, data integrity, and audit logging.

Uses AES-256-GCM for authenticated encryption.
Requires ENCRYPTION_KEY environment variable to be set with a 32-byte,
base64-encoded key.
"""

import sqlite3
import json
import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from pathlib import Path
import base64
import os
import sys
import unittest

# --- NEW Imports for AES-256-GCM ---
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

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

# --- NEW HELPER FUNCTION (from Update) ---
def get_required_env(var_name: str) -> str:
    """
    Gets a required environment variable, raising a critical error if not found.
    """
    value = os.environ.get(var_name)
    if value is None:
        # Get logger instance if available, otherwise print
        logger_instance = logging.getLogger('SecureDatabaseManager')
        if logger_instance.hasHandlers():
            logger_instance.critical(f"CRITICAL: Environment variable '{var_name}' is not set.")
        else:
            print(f"CRITICAL: Environment variable '{var_name}' is not set.")
        raise ValueError(f"Missing required environment variable: {var_name}")
    return value

class SecureDatabaseManager:
    """
    Manages secure database operations, including user auth, data encryption,
    integrity verification, and audit logging.
    """

    # --- Merged Feature (__init__) ---
    # REFACTORED: Removed key/salt paths, now loaded from env.
    def __init__(self,
                 db_path=DEFAULT_PRIMARY_DB,
                 users_db_path=DEFAULT_USERS_DB):
        
        self.db_path = db_path
        self.users_db_path = users_db_path  # From File 2
        
        # Setup logging (From File 1)
        self.logger = self._setup_logging()
        
        # Ensure directories exist (Merged)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.users_db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize encryption (REFACTORED)
        self.aes_gcm_key = self._initialize_encryption()
        
        # Initialize databases (Merged)
        self._initialize_database()  # Initializes data/audit/device tables
        self.init_users_db()       # Initializes user table

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

    # --- NEW: Static method for key generation (from Update) ---
    @staticmethod
    def generate_key_for_env() -> str:
        """
        Generates a 32-byte (256-bit) key, base64-encoded for env vars.
        """
        key = secrets.token_bytes(32)  # 32 bytes = 256 bits
        return base64.b64encode(key).decode('utf-8')

    # --- REFACTORED: Load key from Environment Variable (from Update) ---
    def _initialize_encryption(self):
        """
        Initialize encryption by loading the 32-byte key from
        the ENCRYPTION_KEY environment variable.
        """
        try:
            # Use the helper function to get required env var
            key_b64 = get_required_env("ENCRYPTION_KEY")

            key = base64.b64decode(key_b64)

            if len(key) != 32:
                self.logger.error(
                    f"ENCRYPTION_KEY must be 32 bytes (256-bit), but got {len(key)} bytes."
                )
                raise ValueError("Invalid key length")

            self.logger.info("AES-256-GCM Encryption key loaded successfully from environment")
            return key # Store the raw bytes

        except Exception as e:
            self.logger.critical(f"CRITICAL: Failed to initialize encryption: {e}", exc_info=True)
            raise

    # --- REFACTORED: Removed data_hash column ---
    def _initialize_database(self):
        """Initialize the secure database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create health data table (REMOVED data_hash)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS health_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        device_id TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        encrypted_data TEXT NOT NULL,
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

    # --- REFACTORED: Use AES-256-GCM (from Update) ---
    def encrypt_data(self, data):
        """Encrypt data using AES-256-GCM."""
        if not self.aes_gcm_key: # Check if key was loaded
             raise RuntimeError("Encryption key not initialized.")
        try:
            if isinstance(data, dict):
                data_str = json.dumps(data)
            elif not isinstance(data, (str, bytes)):
                data_str = str(data)
            else:
                 data_str = data # Already string or bytes

            if isinstance(data_str, str):
                data_bytes = data_str.encode('utf-8')
            else:
                 data_bytes = data_str # Assume already bytes

            aesgcm = AESGCM(self.aes_gcm_key)
            nonce = secrets.token_bytes(12)  # GCM standard 96-bit nonce

            # encrypt() returns ciphertext + 16-byte authentication tag
            ciphertext_with_tag = aesgcm.encrypt(nonce, data_bytes, None)

            # Store as nonce + (ciphertext + tag), all base64 encoded
            combined = nonce + ciphertext_with_tag
            return base64.b64encode(combined).decode('utf-8')

        except Exception as e:
            self.logger.error(f"Failed to encrypt data: {e}", exc_info=True)
            raise

    # --- REFACTORED: Use AES-256-GCM (Merged from Original + Update) ---
    def decrypt_data(self, encrypted_data):
        """Decrypt data using AES-256-GCM and verify integrity."""
        if not self.aes_gcm_key: # From Update
             raise RuntimeError("Encryption key not initialized.")
        try:
            combined_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            
            # Extract the 12-byte nonce
            nonce = combined_bytes[:12]
            # Extract the remaining (ciphertext + tag)
            ciphertext_with_tag = combined_bytes[12:]
            
            if len(nonce) != 12:
                    raise ValueError("Invalid encrypted data format: nonce length incorrect")

            aesgcm = AESGCM(self.aes_gcm_key)
            
            # decrypt() will raise InvalidTag exception if auth fails
            decrypted = aesgcm.decrypt(nonce, ciphertext_with_tag, None)
            
            # Return raw string, as in original. Callers handle JSON parsing.
            return decrypted.decode('utf-8')
            
        except (base64.binascii.Error, ValueError) as e:
            self.logger.error(f"Failed to decrypt data (format/padding error): {e}")
            raise
        except InvalidTag:
            self.logger.warning("Failed to decrypt data: DATA TAMPERING DETECTED (InvalidTag)")
            raise
        except Exception as e:
            self.logger.error(f"Failed to decrypt data: {e}", exc_info=True) # exc_info from Update
            raise

    # --- REMOVED: _calculate_hash (redundant with AES-GCM) ---

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

    # --- REFACTORED: Removed data_hash ---
    def insert_health_data(self, device_id, data, user_id=None):
        """Insert encrypted health data."""
        try:
            # Hash calculation removed (now handled by GCM)
            
            # Encrypt the data
            encrypted_data = self.encrypt_data(data)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO health_data (device_id, timestamp, encrypted_data)
                    VALUES (?, ?, ?)
                ''', (device_id, datetime.now(), encrypted_data))
                
                record_id = cursor.lastrowid
                conn.commit()
                
                # Log audit event
                self.log_audit_event(
                    action="INSERT",
                    user_id=user_id,
                    table_name="health_data",
                    record_id=str(record_id),
                    new_values={"device_id": device_id} # Removed data_hash
                )
                
                self.logger.info(f"Health data inserted for device {device_id}")
                return record_id
                
        except Exception as e:
            self.logger.error(f"Failed to insert health data: {e}")
            raise

    # --- REFACTORED: Removed manual integrity check ---
    def get_health_data(self, device_id=None, start_date=None, end_date=None, limit=None):
        """Retrieve and decrypt health data (integrity verified by AES-GCM)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row # Use Row factory
                cursor = conn.cursor()
                
                # Removed data_hash from query
                query = "SELECT id, device_id, timestamp, encrypted_data FROM health_data WHERE 1=1"
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
                        # Decrypt data (this will raise InvalidTag if tampered)
                        # decrypt_data returns a string
                        decrypted_data = self.decrypt_data(row['encrypted_data'])
                        
                        # Parse JSON if applicable (as in original)
                        try:
                            data = json.loads(decrypted_data)
                        except json.JSONDecodeError:
                            data = decrypted_data
                        
                        # Manual hash verification removed
                        
                        results.append({
                            'id': row['id'],
                            'device_id': row['device_id'],
                            'timestamp': row['timestamp'],
                            'data': data
                        })
                        
                    except InvalidTag:
                        # This catches tampering
                        self.logger.warning(f"DATA TAMPERING DETECTED for record {row['id']}. Skipping.")
                        self.log_audit_event(action="INTEGRITY_FAIL", record_id=row['id'], table_name="health_data")
                        continue
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
    # (These are unchanged as their core logic was sound)

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
                    # decrypt_data returns a string, which json.loads parses
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

# --- NEW: Unit Test Class for Encryption ---
class TestEncryption(unittest.TestCase):
    """Tests the new AES-GCM encryption/decryption."""
    
    def setUp(self):
        """Set up a dummy key and a manager instance for testing."""
        # Generate a temporary key for this test run
        self.test_key = SecureDatabaseManager.generate_key_for_env()
        os.environ["ENCRYPTION_KEY"] = self.test_key
        
        # Use in-memory databases for testing
        self.db_manager = SecureDatabaseManager(
            db_path=":memory:", 
            users_db_path=":memory:"
        )
    
    def test_encrypt_decrypt_roundtrip_str(self):
        """Test encrypting and decrypting a simple string."""
        original_str = "This is a secret message!"
        encrypted = self.db_manager.encrypt_data(original_str)
        decrypted = self.db_manager.decrypt_data(encrypted)
        self.assertEqual(original_str, decrypted)
        self.assertNotEqual(original_str, encrypted)

    def test_encrypt_decrypt_roundtrip_dict(self):
        """Test encrypting and decrypting a dictionary."""
        original_dict = {"patient_id": 123, "data": {"bp": "120/80"}}
        encrypted = self.db_manager.encrypt_data(original_dict)
        # decrypt_data returns a string, as expected
        decrypted_str = self.db_manager.decrypt_data(encrypted)
        # The test correctly handles JSON loading
        decrypted_dict = json.loads(decrypted_str)
        self.assertEqual(original_dict, decrypted_dict)

    def test_tampered_data_fails(self):
        """Test that tampered ciphertext fails decryption."""
        original_str = "Sensitive info"
        encrypted = self.db_manager.encrypt_data(original_str)
        
        # Tamper the data
        encrypted_bytes = base64.b64decode(encrypted.encode('utf-8'))
        # Flip a bit in the ciphertext (last byte, which is part of the tag)
        tampered_bytes = encrypted_bytes[:-1] + bytes([encrypted_bytes[-1] ^ 1])
        tampered_encrypted_str = base64.b64encode(tampered_bytes).decode('utf-8')

        with self.assertRaises(InvalidTag):
            self.db_manager.decrypt_data(tampered_encrypted_str)

    def test_tampered_nonce_fails(self):
        """Test that tampered nonce fails decryption."""
        original_str = "Sensitive info"
        encrypted = self.db_manager.encrypt_data(original_str)
        
        # Tamper the nonce
        encrypted_bytes = base64.b64decode(encrypted.encode('utf-8'))
        # Flip a bit in the nonce (first byte)
        tampered_bytes = bytes([encrypted_bytes[0] ^ 1]) + encrypted_bytes[1:]
        tampered_encrypted_str = base64.b64encode(tampered_bytes).decode('utf-8')

        with self.assertRaises(InvalidTag):
            self.db_manager.decrypt_data(tampered_encrypted_str)

    def test_different_data_different_ciphertext(self):
        """Test that encrypting the same data twice yields different ciphertext."""
        original_str = "Data to be encrypted"
        encrypted1 = self.db_manager.encrypt_data(original_str)
        encrypted2 = self.db_manager.encrypt_data(original_str)
        self.assertNotEqual(encrypted1, encrypted2, "Nonces should be different, producing different ciphertext")

    def tearDown(self):
        """Clean up environment variable."""
        if "ENCRYPTION_KEY" in os.environ:
            del os.environ["ENCRYPTION_KEY"]

def run_demo():
    """Runs the original demo, assuming ENCRYPTION_KEY is set."""
    
    print("\n--- Running SecureDatabaseManager Demo ---")
    
    if not os.environ.get("ENCRYPTION_KEY"):
        print("\nFATAL: ENCRYPTION_KEY environment variable not set.")
        print("Please run 'python secure_database_manager.py generate_key' first.")
        print("Then, set the environment variable and run this demo again.")
        sys.exit(1)

    # Initialize secure database manager
    # Use test paths for the demo to avoid clobbering real dbs
    db_manager = SecureDatabaseManager(
        db_path="DATABASE/demo_secure_database.db",
        users_db_path="DATABASE/demo_users.db"
    )
    
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
    
    print("\n--- Demo Complete ---")
    print(f"Demo databases created at {db_manager.db_path} and {db_manager.users_db_path}")

# --- REFACTORED: Main execution block for testing and key generation ---
if __name__ == "__main__":
    # Setup basic logging to console
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if len(sys.argv) > 1:
        if sys.argv[1] == 'generate_key':
            print("--- New Base64-Encoded 256-bit (32-byte) AES Key ---")
            print("Set this as your ENCRYPTION_KEY environment variable:")
            print(SecureDatabaseManager.generate_key_for_env())
        
        elif sys.argv[1] == 'test':
            print("--- Running Encryption Unit Tests ---")
            # Run tests
            suite = unittest.TestSuite()
            suite.addTest(unittest.makeSuite(TestEncryption))
            runner = unittest.TextTestRunner()
            runner.run(suite)
        
        elif sys.argv[1] == 'run_demo':
            run_demo()
        
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage: python secure_database_manager.py [generate_key|test|run_demo]")
    
    else:
        print("Usage: python secure_database_manager.py [generate_key|test|run_demo]")
        print(" - generate_key: Generate a new encryption key for your .env file")
        print(" - test:          Run internal unit tests for encryption")
        print(" - run_demo:      Run the full demo (requires ENCRYPTION_KEY to be set)")