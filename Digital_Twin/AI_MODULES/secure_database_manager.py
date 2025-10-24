#!/usr/bin/env python3
"""
Secure Database Manager (PostgreSQL + SQLAlchemy ORM)
Handles database interactions using SQLAlchemy for PostgreSQL,
including data encryption/decryption, user authentication,
data integrity via AES-GCM, and audit logging.

Requires DATABASE_URL and ENCRYPTION_KEY environment variables.
"""

import json
import logging
import secrets
from datetime import datetime, timedelta, timezone # Added timezone
from pathlib import Path
import base64
import os
import sys
import unittest
from contextlib import contextmanager

# --- SQLAlchemy Imports ---
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Index, MetaData, func, select, update, delete
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base
from sqlalchemy.exc import SQLAlchemyError, IntegrityError # Added IntegrityError
from sqlalchemy.types import TypeDecorator # For handling encrypted text

# --- Cryptography Imports ---
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

# --- Other Imports ---
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash

# Add project root to path (Unchanged)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Helper Function for Environment Variables (MODIFIED) ---
def get_required_env(var_name: str) -> str:
    """
    Gets a required environment variable, raising a critical error if not found or empty.
    """
    value = os.environ.get(var_name)
    
    # Get logger instance
    logger_instance = logging.getLogger('SecureDatabaseManager')
    
    # Configure a basic handler if none exists (e.g., if called before class init)
    if not logger_instance.hasHandlers():
        print(f"Warning: Logger 'SecureDatabaseManager' not configured. Adding basic console handler for env check.")
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger_instance.addHandler(handler)
        logger_instance.setLevel(logging.INFO)

    # FIX: Strengthened validation to check for None or empty string
    if value is None or value.strip() == "":
        log_message = f"CRITICAL: Environment variable '{var_name}' is not set or is empty."
        logger_instance.critical(log_message)
        raise ValueError(f"Missing or empty required environment variable: {var_name}")
        
    return value.strip()

# --- SQLAlchemy Setup ---
Base = declarative_base()
engine = None
SessionLocal = None

# --- ORM Models ---
# Using UTC timezone for all datetimes
UTCDateTime = DateTime(timezone=True)

class HealthData(Base):
    __tablename__ = 'health_data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(100), nullable=False, index=True)
    timestamp = Column(UTCDateTime, nullable=False, index=True)
    encrypted_data = Column(Text, nullable=False)
    created_at = Column(UTCDateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index('ix_health_data_device_id_timestamp', 'device_id', 'timestamp'),
    )

class AuditLog(Base):
    __tablename__ = 'audit_log'
    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String(100), nullable=False)
    user_id = Column(String(100), index=True)
    table_name = Column(String(100))
    record_id = Column(String(100))
    old_values = Column(Text) # Store as JSON string
    new_values = Column(Text) # Store as JSON string
    timestamp = Column(UTCDateTime, default=lambda: datetime.now(timezone.utc), index=True)
    ip_address = Column(String(50))
    user_agent = Column(Text)

class DeviceRegistry(Base):
    __tablename__ = 'device_registry'
    device_id = Column(String(100), primary_key=True)
    device_name = Column(String(200), nullable=False)
    device_type = Column(String(100), nullable=False)
    location = Column(String(200))
    status = Column(String(50), default='active', index=True)
    encrypted_config = Column(Text) # Store encrypted JSON string
    last_seen = Column(UTCDateTime)
    created_at = Column(UTCDateTime, default=lambda: datetime.now(timezone.utc))

class UserSession(Base):
    __tablename__ = 'user_sessions'
    session_id = Column(String(100), primary_key=True)
    user_id = Column(String(100), nullable=False, index=True)
    encrypted_session_data = Column(Text) # Store encrypted JSON string
    expires_at = Column(UTCDateTime, nullable=False, index=True)
    created_at = Column(UTCDateTime, default=lambda: datetime.now(timezone.utc))

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    email = Column(String(200), unique=True, index=True)
    created_at = Column(UTCDateTime, default=lambda: datetime.now(timezone.utc))


# --- Database Session Management ---
@contextmanager
def get_db_session():
    """Provide a transactional scope around a series of operations."""
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call SecureDatabaseManager._initialize_database_connection()")

    session = SessionLocal()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        # FIX: Standardized logging
        logging.getLogger('SecureDatabaseManager').error(f"Database session error: {e}", exc_info=True)
        raise # Re-raise after logging and rollback
    finally:
        SessionLocal.remove()


class SecureDatabaseManager:
    """
    Manages secure database operations using SQLAlchemy for PostgreSQL.
    """

    def __init__(self):
        # FIX: Setup logging *first* so get_required_env can use it
        self.logger = self._setup_logging()
        self._initialize_database_connection() # Initialize SQLAlchemy engine and session
        self._initialize_encryption()
        self._initialize_schema() # Create tables if they don't exist

    def _setup_logging(self):
        """Setup security-focused logging."""
        logger = logging.getLogger('SecureDatabaseManager')
        
        # Prevent duplicate handlers if called multiple times (e.g., in tests)
        if logger.hasHandlers() and len(logger.handlers) > 0:
            return logger
            
        logger.setLevel(logging.INFO)
        Path("LOGS").mkdir(exist_ok=True)
        
        # File handler
        handler = logging.FileHandler('LOGS/digital_twin_security.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Add a console handler as well for visibility during demo/main
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

    def _initialize_database_connection(self):
        """Initialize SQLAlchemy engine and session factory."""
        global engine, SessionLocal
        if engine is None:
            try:
                db_url = get_required_env("DATABASE_URL")
                if not db_url.startswith("postgresql"):
                     raise ValueError("DATABASE_URL does not point to a PostgreSQL database.")

                # CONFIRM: Connection pooling is confirmed here
                engine = create_engine(
                    db_url,
                    pool_pre_ping=True,
                    pool_size=10, # Configure pool size
                    max_overflow=20,
                    pool_timeout=30, # Added timeout
                    pool_recycle=1800 # Recycle connections every 30 mins
                )
                # Test connection
                with engine.connect() as connection:
                    self.logger.info("SQLAlchemy PostgreSQL connection successful.")

                SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
                SessionLocal = scoped_session(SessionFactory)
                self.logger.info("SQLAlchemy SessionLocal created for PostgreSQL.")

            except Exception as e:
                self.logger.critical(f"CRITICAL: Failed to initialize PostgreSQL connection: {e}", exc_info=True)
                raise

    def _initialize_schema(self):
        """Create database tables if they don't exist."""
        global engine
        if engine is None:
            raise RuntimeError("Database engine not initialized.")
        try:
            Base.metadata.create_all(bind=engine)
            self.logger.info("Database schema checked/created successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize database schema: {e}", exc_info=True)
            raise

    # --- Encryption Methods (Largely Unchanged) ---
    @staticmethod
    def generate_key_for_env() -> str:
        key = secrets.token_bytes(32)
        return base64.b64encode(key).decode('utf-8')

    def _initialize_encryption(self):
        """Load AES-GCM key from environment variable."""
        try:
            key_b64 = get_required_env("ENCRYPTION_KEY")
            key = base64.b64decode(key_b64)
            if len(key) != 32:
                raise ValueError(f"ENCRYPTION_KEY must be 32 bytes, got {len(key)}")
            self.aes_gcm_key = key
            self.logger.info("AES-256-GCM Encryption key loaded successfully.")
        except Exception as e:
            self.logger.critical(f"CRITICAL: Failed to initialize encryption: {e}", exc_info=True)
            self.aes_gcm_key = None # Ensure it's None if loading fails
            raise

    def encrypt_data(self, data):
        """Encrypt data using AES-256-GCM."""
        if not self.aes_gcm_key:
             raise RuntimeError("Encryption key not initialized.")
        try:
            if isinstance(data, dict): data_str = json.dumps(data)
            elif not isinstance(data, (str, bytes)): data_str = str(data)
            else: data_str = data

            data_bytes = data_str.encode('utf-8') if isinstance(data_str, str) else data_str

            aesgcm = AESGCM(self.aes_gcm_key)
            nonce = secrets.token_bytes(12)
            ciphertext_with_tag = aesgcm.encrypt(nonce, data_bytes, None)
            combined = nonce + ciphertext_with_tag
            return base64.b64encode(combined).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Failed to encrypt data: {e}", exc_info=True)
            raise

    def decrypt_data(self, encrypted_data):
        """Decrypt data using AES-256-GCM and verify integrity."""
        if not self.aes_gcm_key:
             raise RuntimeError("Encryption key not initialized.")
        try:
            combined_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            nonce = combined_bytes[:12]
            ciphertext_with_tag = combined_bytes[12:]
            if len(nonce) != 12: raise ValueError("Invalid nonce length")

            aesgcm = AESGCM(self.aes_gcm_key)
            decrypted = aesgcm.decrypt(nonce, ciphertext_with_tag, None)
            return decrypted.decode('utf-8')
        except (base64.binascii.Error, ValueError) as e:
            self.logger.error(f"Failed to decrypt data (format error): {e}")
            raise # Re-raise as a generic exception or a custom crypto error
        except InvalidTag:
            self.logger.warning("DATA TAMPERING DETECTED (InvalidTag)")
            raise
        except Exception as e:
            self.logger.error(f"Failed to decrypt data: {e}", exc_info=True)
            raise

    # --- Database Operations (Unchanged) ---

    def log_audit_event(self, action, user_id=None, table_name=None, record_id=None,
                        old_values=None, new_values=None, ip_address=None, user_agent=None):
        """Log security audit events using SQLAlchemy."""
        try:
            with get_db_session() as session:
                audit_log = AuditLog(
                    action=action,
                    user_id=user_id,
                    table_name=table_name,
                    record_id=str(record_id) if record_id is not None else None,
                    old_values=json.dumps(old_values) if old_values else None,
                    new_values=json.dumps(new_values) if new_values else None,
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                session.add(audit_log)
            # FIX: Standardized logging
            self.logger.info(f"Audit event logged: {action}")
        except Exception as e:
            self.logger.error(f"Failed to log audit event: {e}")

    def create_user(self, username: str, password: str, email: str = None) -> bool:
        """Creates a new user with a hashed password using SQLAlchemy."""
        password_hash = generate_password_hash(password)
        try:
            with get_db_session() as session:
                new_user = User(username=username, password_hash=password_hash, email=email)
                session.add(new_user)

            self.logger.info(f"User '{username}' created successfully.")
            self.log_audit_event(
                action="USER_CREATED", user_id=username, table_name="users",
                new_values={"username": username, "email": email}
            )
            return True
        except IntegrityError: # Catch unique constraint violations
            self.logger.warning(f"Failed to create user. Username '{username}' or email '{email}' already exists.")
            self.log_audit_event(action="USER_CREATE_FAILED", new_values={"username": username, "reason": "IntegrityError"})
            return False
        except SQLAlchemyError as e:
            self.logger.error(f"Error creating user '{username}': {e}")
            self.log_audit_event(action="USER_CREATE_FAILED", new_values={"username": username, "reason": str(e)})
            return False

    def authenticate_user(self, username: str, password: str) -> bool:
        """Authenticates a user against the stored hash using SQLAlchemy."""
        try:
            with get_db_session() as session:
                user = session.execute(select(User).filter_by(username=username)).scalar_one_or_none()

            if user and check_password_hash(user.password_hash, password):
                self.logger.info(f"User '{username}' authenticated successfully.")
                self.log_audit_event(action="AUTH_SUCCESS", user_id=username)
                return True

            self.logger.warning(f"Authentication failed for user '{username}'.")
            self.log_audit_event(action="AUTH_FAILED", user_id=username)
            return False
        except SQLAlchemyError as e:
            self.logger.error(f"Error authenticating user '{username}': {e}")
            self.log_audit_event(action="AUTH_ERROR", user_id=username, new_values={"error": str(e)})
            return False

    def insert_health_data(self, device_id, data, user_id=None):
        """Insert encrypted health data using SQLAlchemy."""
        try:
            encrypted_data_str = self.encrypt_data(data)
            record_id = None
            with get_db_session() as session:
                health_record = HealthData(
                    device_id=device_id,
                    timestamp=datetime.now(timezone.utc),
                    encrypted_data=encrypted_data_str
                )
                session.add(health_record)
                session.flush() # Flush to get the ID before commit
                record_id = health_record.id

            if record_id is not None:
                self.log_audit_event(
                    action="INSERT", user_id=user_id, table_name="health_data",
                    record_id=record_id, new_values={"device_id": device_id}
                )
                self.logger.info(f"Health data inserted for device {device_id} (ID: {record_id})")
                return record_id
            else:
                 self.logger.error("Failed to retrieve record ID after insertion.")
                 return None

        except Exception as e:
            self.logger.error(f"Failed to insert health data: {e}", exc_info=True)
            raise

    def get_health_data(self, device_id=None, start_date=None, end_date=None, limit=None):
        """Retrieve and decrypt health data using SQLAlchemy."""
        try:
            results = []
            with get_db_session() as session:
                query = select(HealthData)
                if device_id:
                    query = query.filter(HealthData.device_id == device_id)
                if start_date:
                    start_date_aware = start_date.replace(tzinfo=timezone.utc) if start_date.tzinfo is None else start_date
                    query = query.filter(HealthData.timestamp >= start_date_aware)
                if end_date:
                    end_date_aware = end_date.replace(tzinfo=timezone.utc) if end_date.tzinfo is None else end_date
                    query = query.filter(HealthData.timestamp <= end_date_aware)

                query = query.order_by(HealthData.timestamp.desc())

                if limit:
                    query = query.limit(limit)

                records = session.execute(query).scalars().all()

            for row in records:
                try:
                    decrypted_data_str = self.decrypt_data(row.encrypted_data)
                    try: data = json.loads(decrypted_data_str)
                    except json.JSONDecodeError: data = decrypted_data_str

                    results.append({
                        'id': row.id,
                        'device_id': row.device_id,
                        'timestamp': row.timestamp.isoformat(),
                        'data': data
                    })
                except InvalidTag:
                    self.logger.warning(f"DATA TAMPERING DETECTED for record {row.id}. Skipping.")
                    self.log_audit_event(action="INTEGRITY_FAIL", record_id=row.id, table_name="health_data")
                    continue
                except Exception as e:
                    self.logger.error(f"Failed to process record {row.id}: {e}")
                    continue
            return results
        except Exception as e:
            self.logger.error(f"Failed to retrieve health data: {e}")
            raise

    def get_health_data_as_dataframe(self, device_id=None, start_date=None, end_date=None, limit=None) -> pd.DataFrame:
        """Retrieve health data as a pandas DataFrame using SQLAlchemy."""
        try:
            secure_data = self.get_health_data(device_id, start_date, end_date, limit)
            if not secure_data: return pd.DataFrame()

            df = pd.DataFrame(secure_data)

            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            if 'data' in df.columns and not df['data'].isnull().all():
                valid_data = df['data'].apply(lambda x: x if isinstance(x, dict) else {})
                data_df = pd.json_normalize(valid_data).set_index(df.index)
                df = pd.concat([df.drop('data', axis=1), data_df], axis=1)

            self.logger.info(f"Successfully converted {len(df)} secure records to DataFrame.")
            return df
        except Exception as e:
            self.logger.error(f"Failed to convert health data to DataFrame: {e}")
            return pd.DataFrame()

    def register_device(self, device_id, device_name, device_type, location=None, config=None):
        """Register or update a device using SQLAlchemy."""
        try:
            encrypted_config_str = self.encrypt_data(config) if config else None
            with get_db_session() as session:
                device = DeviceRegistry(
                    device_id=device_id,
                    device_name=device_name,
                    device_type=device_type,
                    location=location,
                    encrypted_config=encrypted_config_str,
                    last_seen=datetime.now(timezone.utc)
                )
                session.merge(device)

            self.log_audit_event(
                action="DEVICE_REGISTERED_OR_UPDATED", table_name="device_registry", record_id=device_id,
                new_values={"device_name": device_name, "device_type": device_type}
            )
            self.logger.info(f"Device registered/updated: {device_id}")
        except Exception as e:
            self.logger.error(f"Failed to register device: {e}")
            raise

    def get_devices(self):
        """Get list of registered devices using SQLAlchemy."""
        try:
            with get_db_session() as session:
                stmt = select(
                    DeviceRegistry.device_id, DeviceRegistry.device_name,
                    DeviceRegistry.device_type, DeviceRegistry.location,
                    DeviceRegistry.status, DeviceRegistry.last_seen,
                    DeviceRegistry.created_at
                ).order_by(DeviceRegistry.device_name)
                devices = session.execute(stmt).mappings().all()

            for device in devices:
                for key in ['last_seen', 'created_at']:
                    if isinstance(device.get(key), datetime):
                        device[key] = device[key].isoformat()

            return devices
        except Exception as e:
            self.logger.error(f"Failed to get devices: {e}")
            raise

    def update_device_status(self, device_id, status):
        """Update device status using SQLAlchemy."""
        try:
            with get_db_session() as session:
                stmt = update(DeviceRegistry).\
                    where(DeviceRegistry.device_id == device_id).\
                    values(status=status, last_seen=datetime.now(timezone.utc))
                result = session.execute(stmt)

                if result.rowcount == 0:
                    self.logger.warning(f"Device not found for status update: {device_id}")
                else:
                    self.log_audit_event(
                        action="DEVICE_STATUS_UPDATED", table_name="device_registry",
                        record_id=device_id, new_values={"status": status}
                    )
        except Exception as e:
            self.logger.error(f"Failed to update device status: {e}")
            raise

    # --- Session Management Methods (Unchanged) ---
    def create_session(self, user_id, session_data, expires_in_minutes=60):
        try:
            session_id = secrets.token_urlsafe(32)
            encrypted_session_data_str = self.encrypt_data(session_data)
            expires_at_dt = datetime.now(timezone.utc) + timedelta(minutes=expires_in_minutes)

            with get_db_session() as session:
                user_session = UserSession(
                    session_id=session_id,
                    user_id=user_id,
                    encrypted_session_data=encrypted_session_data_str,
                    expires_at=expires_at_dt
                )
                session.add(user_session)

            self.log_audit_event(action="SESSION_CREATED", user_id=user_id, table_name="user_sessions", record_id=session_id)
            return session_id
        except Exception as e:
            self.logger.error(f"Failed to create session: {e}")
            raise

    def get_session(self, session_id):
        try:
            with get_db_session() as session:
                now = datetime.now(timezone.utc)
                stmt = select(UserSession).filter(
                    UserSession.session_id == session_id,
                    UserSession.expires_at > now
                )
                row = session.execute(stmt).scalar_one_or_none()

            if row:
                session_data_str = self.decrypt_data(row.encrypted_session_data)
                session_data = json.loads(session_data_str)
                return {
                    'user_id': row.user_id,
                    'session_data': session_data,
                    'expires_at': row.expires_at.isoformat()
                }
            return None
        except InvalidTag:
             self.logger.warning(f"Session data tampering detected for session: {session_id}")
             return None
        except Exception as e:
            self.logger.error(f"Failed to get session: {e}")
            raise

    def delete_session(self, session_id):
        try:
            with get_db_session() as session:
                stmt = delete(UserSession).where(UserSession.session_id == session_id)
                session.execute(stmt)
            self.log_audit_event(action="SESSION_DELETED", table_name="user_sessions", record_id=session_id)
        except Exception as e:
            self.logger.error(f"Failed to delete session: {e}")
            raise

    def cleanup_expired_sessions(self):
        try:
            deleted_count = 0
            with get_db_session() as session:
                now = datetime.now(timezone.utc)
                stmt = delete(UserSession).where(UserSession.expires_at < now)
                result = session.execute(stmt)
                deleted_count = result.rowcount

            if deleted_count > 0:
                self.logger.info(f"Cleaned up {deleted_count} expired sessions")
                self.log_audit_event(action="SESSION_CLEANUP", new_values={"deleted_count": deleted_count})
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired sessions: {e}")

    # --- Audit Log Retrieval (Unchanged) ---
    def get_audit_logs(self, limit=100, action_filter=None, start_date=None, end_date=None):
        try:
            logs = []
            with get_db_session() as session:
                query = select(AuditLog)
                if action_filter: query = query.filter(AuditLog.action == action_filter)
                if start_date: query = query.filter(AuditLog.timestamp >= start_date)
                if end_date: query = query.filter(AuditLog.timestamp <= end_date)
                query = query.order_by(AuditLog.timestamp.desc()).limit(limit)

                records = session.execute(query).scalars().all()

            for row in records:
                log_entry = {c.name: getattr(row, c.name) for c in AuditLog.__table__.columns}
                log_entry['old_values'] = json.loads(row.old_values) if row.old_values else None
                log_entry['new_values'] = json.loads(row.new_values) if row.new_values else None
                log_entry['timestamp'] = row.timestamp.isoformat()
                logs.append(log_entry)
            return logs
        except Exception as e:
            self.logger.error(f"Failed to get audit logs: {e}")
            raise

    # --- REMOVED backup_database method ---

    def close(self):
        """Close database connections and cleanup (handled by scoped_session)."""
        global engine
        if engine:
             self.logger.info("SQLAlchemy engine connections managed.")
        self.logger.info("Database manager state closed/cleaned.")


# --- Unit Tests (FIX: Added new test) ---
class TestEncryptionWithSQLAlchemy(unittest.TestCase):
    """Tests encryption/decryption with SQLAlchemy ORM (using in-memory SQLite)."""
    engine = None
    SessionLocal = None

    @classmethod
    def setUpClass(cls):
        """Set up in-memory SQLite DB for tests."""
        cls.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=cls.engine)
        SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=cls.engine)
        cls.SessionLocal = scoped_session(SessionFactory)

        global SessionLocal
        cls._original_session_local = SessionLocal
        SessionLocal = cls.SessionLocal
        
        # FIX: Configure a basic logger for tests to avoid setup warnings
        logger = logging.getLogger('SecureDatabaseManager')
        if not logger.hasHandlers():
            logger.addHandler(logging.StreamHandler(sys.stdout))
            logger.setLevel(logging.DEBUG)


    @classmethod
    def tearDownClass(cls):
        """Clean up DB and restore original SessionLocal."""
        global SessionLocal
        SessionLocal = cls._original_session_local
        Base.metadata.drop_all(bind=cls.engine)
        if cls.engine:
            cls.engine.dispose()

    def setUp(self):
        """Set up a dummy key and a manager instance for testing."""
        self.test_key_b64 = SecureDatabaseManager.generate_key_for_env()
        os.environ["ENCRYPTION_KEY"] = self.test_key_b64
        # Set a dummy DB URL for the env check, even though we use in-memory
        os.environ["DATABASE_URL"] = "postgresql://user:pass@host/db" 

        self.db_manager = SecureDatabaseManager()
        self.db_manager.SessionLocal = self.SessionLocal

    def test_encrypt_decrypt_roundtrip_str(self):
        original_str = "Secret message!"
        encrypted = self.db_manager.encrypt_data(original_str)
        decrypted = self.db_manager.decrypt_data(encrypted)
        self.assertEqual(original_str, decrypted)
        self.assertNotEqual(original_str, encrypted)

    def test_encrypt_decrypt_roundtrip_dict(self):
        original_dict = {"id": 456, "value": 99.9}
        encrypted = self.db_manager.encrypt_data(original_dict)
        decrypted_str = self.db_manager.decrypt_data(encrypted)
        decrypted_dict = json.loads(decrypted_str)
        self.assertEqual(original_dict, decrypted_dict)

    def test_tampered_data_fails(self):
        encrypted = self.db_manager.encrypt_data("Sensitive")
        encrypted_bytes = base64.b64decode(encrypted.encode('utf-8'))
        tampered_bytes = encrypted_bytes[:-1] + bytes([encrypted_bytes[-1] ^ 1])
        tampered_encrypted_str = base64.b64encode(tampered_bytes).decode('utf-8')
        with self.assertRaises(InvalidTag):
            self.db_manager.decrypt_data(tampered_encrypted_str)
            
    # --- NEW TEST ---
    def test_decrypt_invalid_format(self):
        """Tests decryption with invalid base64 or structure."""
        # Test 1: Invalid Base64
        with self.assertRaises(Exception) as context_invalid_b64:
            self.db_manager.decrypt_data("not base64")
        self.assertIn("Invalid", str(context_invalid_b64.exception), 
                      "Should fail on invalid base64")

        # Test 2: Data too short (e.g., missing nonce)
        encrypted_too_short = base64.b64encode(b"short").decode('utf-8')
        with self.assertRaises(Exception) as context_too_short:
            self.db_manager.decrypt_data(encrypted_too_short)
        self.assertIn("Invalid", str(context_too_short.exception), 
                      "Should fail on data shorter than nonce")


    def tearDown(self):
        if "ENCRYPTION_KEY" in os.environ: del os.environ["ENCRYPTION_KEY"]
        if "DATABASE_URL" in os.environ: del os.environ["DATABASE_URL"]
        
        with self.SessionLocal() as session:
            session.query(HealthData).delete()
            session.query(User).delete()
            session.commit()
        self.SessionLocal.remove()


# --- Demo Function (Unchanged) ---
def run_demo():
    """Runs a demo using the PostgreSQL + SQLAlchemy manager."""
    print("\n--- Running SecureDatabaseManager Demo (PostgreSQL + SQLAlchemy) ---")
    
    # Env vars should be pre-checked by __main__ or loaded via dotenv
    try:
        db_manager = SecureDatabaseManager() # Initializes connection and schema

        print("\n--- Testing User Management ---")
        db_manager.create_user("orm_admin", "securepass987", "orm@digitaltwin.com")
        db_manager.create_user("orm_admin", "anotherpass", "orm@digitaltwin.com") # Should fail

        print("\n--- Testing Authentication ---")
        print(f"Auth 'orm_admin'/'securepass987': {db_manager.authenticate_user('orm_admin', 'securepass987')}")
        print(f"Auth 'orm_admin'/'wrongpass': {db_manager.authenticate_user('orm_admin', 'wrongpass')}")

        print("\n--- Testing Device & Data ---")
        db_manager.register_device(
            device_id="sensor_pg_001", device_name="PG Temp Sensor", device_type="temperature",
            location="Server Room", config={"rate": 30, "unit": "C"}
        )

        test_data = {"temperature": 22.5, "humidity": 55.1}
        rec_id = db_manager.insert_health_data("sensor_pg_001", test_data, user_id="orm_admin")
        print(f"Inserted health data, record ID: {rec_id}")

        data = db_manager.get_health_data("sensor_pg_001", limit=10)
        print(f"\nRetrieved {len(data)} records (raw list):")
        print(json.dumps(data, indent=2))

        print("\n--- Testing Pandas DataFrame ---")
        df = db_manager.get_health_data_as_dataframe(device_id="sensor_pg_001", limit=10)
        print(f"Retrieved DataFrame with {len(df)} rows and columns: {list(df.columns)}")
        if not df.empty: print(df.head())

        devices = db_manager.get_devices()
        print(f"\nRegistered devices: {len(devices)}")

        logs = db_manager.get_audit_logs(limit=5)
        print(f"\nRecent audit events: {len(logs)}")
        print(json.dumps(logs, indent=2))

        print("\n--- Database Backup ---")
        print("Note: Use PostgreSQL tools like 'pg_dump' for backups.")

        db_manager.cleanup_expired_sessions()
        db_manager.close()

        print("\n--- Demo Complete ---")

    except Exception as e:
        print(f"\n--- DEMO FAILED ---")
        # Use the configured logger if available
        logging.getLogger('SecureDatabaseManager').error("Demo failed", exc_info=True)


# --- Main Execution Block (FIX: Standardized logging setup) ---
if __name__ == "__main__":
    # FIX: Configure a basic root logger *only* for the script execution
    # This avoids interfering with the class logger if it's already set up
    # but provides output if the script is run directly.
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    if len(sys.argv) > 1:
        if sys.argv[1] == 'generate_key':
            print("--- New Base64-Encoded 256-bit AES Key ---")
            print("Set this as your ENCRYPTION_KEY environment variable:")
            print(SecureDatabaseManager.generate_key_for_env())

        elif sys.argv[1] == 'test':
            print("--- Running Encryption Unit Tests (using in-memory SQLite via SQLAlchemy) ---")
            # Env vars are set within the test setUp/tearDown methods
            suite = unittest.TestSuite()
            suite.addTest(unittest.makeSuite(TestEncryptionWithSQLAlchemy))
            runner = unittest.TextTestRunner()
            runner.run(suite)

        elif sys.argv[1] == 'run_demo':
            # Load .env for demo if available
            try:
                from dotenv import load_dotenv
                dotenv_path = Path(__file__).parent.parent / '.env'
                if dotenv_path.exists():
                    load_dotenv(dotenv_path=dotenv_path, override=True)
                    print(f"Loaded environment variables from: {dotenv_path}")
                else:
                    print(f"Warning: .env file not found at {dotenv_path}")
            except ImportError:
                print("Warning: python-dotenv not installed. Cannot load .env file.")
            
            # Check required vars *before* running demo
            if "ENCRYPTION_KEY" not in os.environ or \
               "DATABASE_URL" not in os.environ or \
               not os.environ["DATABASE_URL"].startswith("postgresql"):
                print("\nFATAL: DATABASE_URL (must be postgresql) and/or ENCRYPTION_KEY missing.")
                print("Please set them in your environment or .env file.")
                sys.exit(1)
                
            run_demo()

        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage: python secure_database_manager.py [generate_key|test|run_demo]")

    else:
        print("Usage: python secure_database_manager.py [generate_key|test|run_demo]")
        print(" - generate_key: Generate ENCRYPTION_KEY")
        print(" - test:          Run internal unit tests")
        print(" - run_demo:      Run demo (requires DATABASE_URL and ENCRYPTION_KEY env vars)")