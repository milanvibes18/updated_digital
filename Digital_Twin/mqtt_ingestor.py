#!/usr/bin/env python3
"""
MQTT Ingestor for Digital Twin System v1.2

Subscribes to MQTT topics for IoT device data, parses JSON payloads,
stores the data in PostgreSQL, handles QoS and reconnections robustly,
verifies DB pooling, and logs ingestion metrics for Prometheus.

v1.2 Updates:
- Improved MQTT reconnection logic using Paho's built-in mechanism.
- Enhanced error handling around database connections and operations.
- Added explicit DB connection test on startup.
- Added graceful shutdown handling.
- Ensured timezone awareness for timestamps.
- Verified pooling configuration.
"""

import paho.mqtt.client as mqtt
import json
import os
import sys
import logging
import time
from datetime import datetime, timezone # Use timezone-aware datetime
from pathlib import Path
from dotenv import load_dotenv
import signal # Added for graceful shutdown
import threading # Added for graceful shutdown

# --- Database Import ---
# Assuming SQLAlchemy ORM is preferred
try:
    from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, MetaData, Table, insert, Index
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.exc import SQLAlchemyError, OperationalError as SQLAlchemyOperationalError # Added OperationalError
    SQLALCHEMY_AVAILABLE = True
    psycopg2 = None # Ensure psycopg2 is None if SQLAlchemy is used
    print("Using SQLAlchemy for database connection.")
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logging.warning("SQLAlchemy not found, attempting psycopg2 fallback for database connection.")
    # Fallback to psycopg2
    try:
        import psycopg2
        from psycopg2 import pool, OperationalError as Psycopg2OperationalError # Added OperationalError
        from psycopg2.extras import Json # Use Json adapter for JSONB
        print("Using psycopg2 for database connection.")
    except ImportError:
        psycopg2 = None
        logging.critical("Neither SQLAlchemy nor psycopg2 found. Database functionality disabled.")

# --- Prometheus Import ---
try:
    from prometheus_client import start_http_server, Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("prometheus_client not installed. Metrics logging disabled.")
    # Define dummy classes/functions if Prometheus is unavailable
    class PrometheusDummy:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
    Counter = Gauge = Histogram = PrometheusDummy
    def start_http_server(*args, **kwargs): pass

# --- Add project root to path ---
# Use relative pathing to be more robust
project_root = Path(__file__).resolve().parent.parent.parent # Adjusted path
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
    print(f"Added project root to sys.path: {project_root}")

# --- Load Environment Variables ---
# Assumes .env is in the Digital_Twin directory, relative to project_root
env_path = project_root / 'Digital_Twin' / '.env'
if env_path.is_file():
    load_dotenv(dotenv_path=env_path, verbose=True)
    print(f"Loaded environment variables from: {env_path}")
else:
    print(f"Warning: .env file not found at {env_path}. Relying on system environment variables.")

# --- Configuration ---
MQTT_BROKER_HOST = os.getenv('MQTT_BROKER_HOST', 'localhost')
try:
    MQTT_BROKER_PORT = int(os.getenv('MQTT_BROKER_PORT', 1883))
except ValueError:
    MQTT_BROKER_PORT = 1883
    logging.warning(f"Invalid MQTT_BROKER_PORT. Using default {MQTT_BROKER_PORT}.")

MQTT_TOPIC = os.getenv('MQTT_INGEST_TOPIC', 'iot/device/+/data') # Wildcard topic
MQTT_CLIENT_ID = f'mqtt-ingestor-{os.getpid()}'
MQTT_QOS = int(os.getenv('MQTT_QOS', 1)) # Quality of Service Level 1 (At least once)
# **Verification**: Using Paho's built-in reconnect delay. Default is 1 sec, max 128 sec.
MQTT_RECONNECT_DELAY_MIN = int(os.getenv('MQTT_RECONNECT_DELAY_MIN', 1))
MQTT_RECONNECT_DELAY_MAX = int(os.getenv('MQTT_RECONNECT_DELAY_MAX', 120))


DATABASE_URL = os.getenv('DATABASE_URL')
DB_TABLE_NAME = os.getenv('DB_INGEST_TABLE', 'device_readings_raw')

PROMETHEUS_PORT = int(os.getenv('PROMETHEUS_INGESTOR_PORT', 8001))

LOG_LEVEL = os.getenv('LOG_LEVEL_INGESTOR', 'INFO').upper() # Specific log level
LOG_DIR = project_root / 'Digital_Twin' / 'LOGS' # Corrected log path
LOG_FILE = LOG_DIR / 'mqtt_ingestor.log'

# Removed manual reconnect settings, relying on paho defaults or client.reconnect_delay_set

# --- Logging Setup ---
LOG_DIR.mkdir(parents=True, exist_ok=True) # Ensure log directory exists
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout) # Also log to console
    ]
)
logger = logging.getLogger('MQTTIngestor')

# --- Prometheus Metrics ---
messages_received = Counter('mqtt_messages_received_total', 'Total number of MQTT messages received')
messages_parsed = Counter('mqtt_messages_parsed_total', 'Total number of MQTT messages successfully parsed')
messages_stored = Counter('mqtt_messages_stored_total', 'Total number of messages successfully stored in DB')
parse_errors = Counter('mqtt_parse_errors_total', 'Total number of JSON parsing errors')
db_errors = Counter('mqtt_db_errors_total', 'Total number of database insertion errors')
connection_status = Gauge('mqtt_connection_status', 'MQTT connection status (1=connected, 0=disconnected)')
db_connection_status = Gauge('db_connection_status', 'Database connection status (1=connected, 0=disconnected)') # Added DB status

# --- Database Connection & Setup ---
db_engine = None
db_pool = None
# Define table structure explicitly
metadata_obj = MetaData()
readings_table = Table(DB_TABLE_NAME, metadata_obj,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('device_id', String(100), nullable=False, index=True),
    Column('timestamp', DateTime(timezone=True), nullable=False, index=True),
    Column('raw_payload', Text), # Store raw JSON payload
    Column('temperature', Float, nullable=True),
    Column('humidity', Float, nullable=True),
    Column('pressure', Float, nullable=True),
    Column('vibration', Float, nullable=True),
    Column('battery', Float, nullable=True),
    Column('latitude', Float, nullable=True),
    Column('longitude', Float, nullable=True),
    Index(f'idx_{DB_TABLE_NAME}_device_timestamp', 'device_id', 'timestamp') # Composite index
)

def init_database_connection():
    global db_engine, db_pool
    if not DATABASE_URL:
        logger.critical("DATABASE_URL environment variable is not set. Cannot connect to database.")
        db_connection_status.set(0) # Update metric
        return False # Indicate failure

    if SQLALCHEMY_AVAILABLE:
        try:
            db_engine = create_engine(
                DATABASE_URL,
                pool_pre_ping=True, # **Verification**: Enables check on checkout
                pool_size=5,
                max_overflow=10,
                pool_timeout=30, # Time to wait for a connection
                pool_recycle=1800 # Recycle connections every 30 mins
            )
            # **Verification**: Explicit connection test
            with db_engine.connect() as connection:
                connection.execute(text("SELECT 1")) # Simple query to test
            logger.info(f"SQLAlchemy connected and tested successfully.")
            metadata_obj.create_all(db_engine)
            logger.info(f"Table '{DB_TABLE_NAME}' checked/created.")
            db_connection_status.set(1) # Update metric
            return True
        except SQLAlchemyOperationalError as op_err: # Catch specific connection errors
             logger.critical(f"SQLAlchemy OperationalError during connection/setup: {op_err}", exc_info=False)
             db_engine = None
             db_connection_status.set(0)
             return False
        except Exception as e:
            logger.critical(f"Failed to connect/setup database using SQLAlchemy: {e}", exc_info=True)
            db_engine = None
            db_connection_status.set(0)
            return False
    elif psycopg2:
        try:
            # **Verification**: SimpleConnectionPool implicitly handles basic pooling.
            # Reconnection logic is handled by retrying operations.
            db_pool = psycopg2.pool.SimpleConnectionPool(1, 5, DATABASE_URL)
            # **Verification**: Explicit connection test
            conn = db_pool.getconn()
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            logger.info("psycopg2 connected and tested successfully.")
            # Table creation logic (same as before)
            cur = conn.cursor()
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {DB_TABLE_NAME} (
                    id SERIAL PRIMARY KEY, device_id VARCHAR(100) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL, raw_payload JSONB,
                    temperature REAL, humidity REAL, pressure REAL, vibration REAL,
                    battery REAL, latitude REAL, longitude REAL
                );
            """)
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{DB_TABLE_NAME}_device_timestamp ON {DB_TABLE_NAME} (device_id, timestamp);")
            conn.commit()
            cur.close()
            db_pool.putconn(conn)
            logger.info(f"Table '{DB_TABLE_NAME}' checked/created.")
            db_connection_status.set(1)
            return True
        except Psycopg2OperationalError as op_err: # Catch specific connection errors
            logger.critical(f"psycopg2 OperationalError during connection/setup: {op_err}", exc_info=False)
            db_pool = None
            db_connection_status.set(0)
            return False
        except Exception as e:
            logger.critical(f"Failed to connect/setup database using psycopg2: {e}", exc_info=True)
            db_pool = None
            db_connection_status.set(0)
            return False
    else:
        logger.critical("No database library (SQLAlchemy or psycopg2) available.")
        db_connection_status.set(0)
        return False

# Function to attempt database reconnection (used in store functions)
def attempt_db_reconnect():
    logger.warning("Attempting to re-initialize database connection...")
    time.sleep(5) # Wait before retrying
    return init_database_connection()

def store_data_sqlalchemy(data_to_store: dict):
    """Stores data using SQLAlchemy engine with retry logic."""
    global db_engine
    if not db_engine:
        logger.error("SQLAlchemy engine not initialized. Attempting reconnect.")
        db_errors.inc()
        if not attempt_db_reconnect():
            return False
        # Re-assign engine after potential successful reconnect
        global db_engine
        if not db_engine: return False # Still failed

    retries = 3
    for attempt in range(retries):
        try:
            stmt = insert(readings_table).values(data_to_store)
            with db_engine.connect() as connection:
                connection.execute(stmt)
                connection.commit()
            db_connection_status.set(1) # Connection likely OK if insert worked
            return True
        except SQLAlchemyOperationalError as op_err: # Handle connection issues
            logger.error(f"SQLAlchemy OperationalError on insert (attempt {attempt+1}/{retries}): {op_err}", exc_info=False)
            db_errors.inc()
            db_connection_status.set(0)
            if attempt < retries - 1:
                logger.warning("Retrying DB operation after delay...")
                time.sleep(2 ** attempt) # Exponential backoff
                # Attempt to reconnect the engine explicitly if needed, or rely on pool_pre_ping
                if not attempt_db_reconnect(): # Try re-init if connection failed
                    logger.error("DB reconnect failed during retry.")
                    return False
            else:
                 logger.error("Max DB retries reached. Failed to store data.")
                 return False
        except SQLAlchemyError as e: # Handle other SQLAlchemy errors
            logger.error(f"SQLAlchemy DB insert error: {e}", exc_info=False)
            db_errors.inc()
            return False # Don't retry non-operational errors
        except Exception as e:
            logger.error(f"Unexpected error during SQLAlchemy insert: {e}", exc_info=True)
            db_errors.inc()
            return False # Don't retry unknown errors
    return False # Should not be reached if retries exhausted

def store_data_psycopg2(data_to_store: dict):
    """Stores data using psycopg2 connection pool with retry logic."""
    global db_pool
    if not db_pool:
        logger.error("psycopg2 pool not initialized. Attempting reconnect.")
        db_errors.inc()
        if not attempt_db_reconnect():
            return False
        global db_pool
        if not db_pool: return False

    retries = 3
    for attempt in range(retries):
        conn = None
        try:
            conn = db_pool.getconn()
            cur = conn.cursor()
            # Filter data_to_store (same as before)
            valid_columns = [ 'device_id', 'timestamp', 'raw_payload', 'temperature', 'humidity', 'pressure', 'vibration', 'battery', 'latitude', 'longitude']
            filtered_data = {k: v for k, v in data_to_store.items() if k in valid_columns}
            columns = ', '.join(filtered_data.keys())
            placeholders = ', '.join(['%s'] * len(filtered_data))
            values = list(filtered_data.values())
            if 'raw_payload' in filtered_data:
                payload_index = list(filtered_data.keys()).index('raw_payload')
                values[payload_index] = Json(values[payload_index])

            sql = f"INSERT INTO {DB_TABLE_NAME} ({columns}) VALUES ({placeholders})"
            cur.execute(sql, values)
            conn.commit()
            cur.close()
            db_pool.putconn(conn) # Return connection on success
            db_connection_status.set(1)
            return True
        except Psycopg2OperationalError as op_err: # Handle connection issues
            logger.error(f"psycopg2 OperationalError on insert (attempt {attempt+1}/{retries}): {op_err}", exc_info=False)
            db_errors.inc()
            db_connection_status.set(0)
            if conn:
                try:
                     db_pool.putconn(conn, close=True) # Close potentially broken connection
                except Exception: pass # Ignore errors during closing a bad connection
            if attempt < retries - 1:
                logger.warning("Retrying DB operation after delay...")
                time.sleep(2 ** attempt)
                if not attempt_db_reconnect():
                     logger.error("DB reconnect failed during retry.")
                     return False
            else:
                logger.error("Max DB retries reached. Failed to store data.")
                return False
        except Exception as e: # Handle other errors
            logger.error(f"psycopg2 DB insert error: {e}", exc_info=False)
            if conn: conn.rollback()
            db_errors.inc()
            if conn:
                try:
                    db_pool.putconn(conn) # Return connection even on other errors if acquired
                except Exception: pass
            return False # Don't retry non-operational errors
    return False

# --- MQTT Callbacks ---
def on_connect(client, userdata, flags, rc, properties=None):
    """Callback when MQTT client connects."""
    if rc == 0:
        logger.info(f"Connected to MQTT Broker: {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}")
        connection_status.set(1)
        try:
            client.subscribe(MQTT_TOPIC, qos=MQTT_QOS)
            logger.info(f"Subscribed to topic '{MQTT_TOPIC}' with QoS {MQTT_QOS}")
        except Exception as e:
            logger.error(f"Failed to subscribe to topic '{MQTT_TOPIC}': {e}")
    else:
        # **Improvement**: Log specific Paho error messages for connect failures
        logger.error(f"Failed to connect to MQTT Broker, return code {rc}: {mqtt.connack_string(rc)}")
        connection_status.set(0)
        # Paho handles reconnection attempts automatically based on reconnect_delay_set

def on_disconnect(client, userdata, rc, properties=None):
    """Callback when MQTT client disconnects."""
    connection_status.set(0)
    if rc != 0:
        # **Improvement**: Log unexpected disconnects
        logger.warning(f"Unexpectedly disconnected from MQTT Broker (rc={rc}). Paho client will attempt auto-reconnect.")
    else:
        logger.info("Cleanly disconnected from MQTT Broker.")
    # **Verification**: No manual schedule_reconnect needed; Paho handles it.

def on_message(client, userdata, msg):
    """Callback when a message is received."""
    messages_received.inc()
    logger.debug(f"Received message on topic '{msg.topic}': {msg.payload[:100].decode()}...")

    try:
        topic_parts = msg.topic.split('/')
        device_id = topic_parts[2] if len(topic_parts) >= 3 and topic_parts[1] == 'device' else "unknown_device"

        try:
            payload_str = msg.payload.decode('utf-8')
            payload = json.loads(payload_str)
            messages_parsed.inc()
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parse Error from {device_id} on {msg.topic}: {e}. Payload: {msg.payload[:200]}")
            parse_errors.inc()
            return
        except UnicodeDecodeError as e:
            logger.error(f"UTF-8 Decode Error from {device_id} on {msg.topic}: {e}")
            parse_errors.inc()
            return

        # Timestamp Handling (Ensure timezone awareness - UTC)
        timestamp_str = payload.get('timestamp')
        ts_datetime = datetime.now(timezone.utc) # Default to now(UTC)
        if timestamp_str:
            try:
                if isinstance(timestamp_str, str):
                    timestamp_str = timestamp_str.replace('Z', '+00:00')
                    ts_datetime = datetime.fromisoformat(timestamp_str)
                elif isinstance(timestamp_str, (int, float)):
                    timestamp_val = float(timestamp_str)
                    ts_datetime = datetime.fromtimestamp(timestamp_val / 1000.0, tz=timezone.utc) if timestamp_val > 1e11 else datetime.fromtimestamp(timestamp_val, tz=timezone.utc)
                else: raise ValueError("Unsupported timestamp type")

                if ts_datetime.tzinfo is None:
                    ts_datetime = ts_datetime.replace(tzinfo=timezone.utc) # Assume UTC if naive
                else:
                    ts_datetime = ts_datetime.astimezone(timezone.utc) # Convert to UTC

            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid timestamp format '{timestamp_str}' from {device_id}. Using current time. Error: {e}")
                # ts_datetime remains datetime.now(timezone.utc)
        else:
            logger.debug(f"Timestamp missing in payload from {device_id}. Using current time.")

        data_to_store = {
            'device_id': device_id,
            'timestamp': ts_datetime,
            'raw_payload': payload_str if SQLALCHEMY_AVAILABLE else payload,
            'temperature': payload.get('temperature'),
            'humidity': payload.get('humidity'),
            'pressure': payload.get('pressure'),
            'vibration': payload.get('vibration'),
            'battery': payload.get('battery'),
            'latitude': payload.get('location', {}).get('latitude') if isinstance(payload.get('location'), dict) else None,
            'longitude': payload.get('location', {}).get('longitude') if isinstance(payload.get('location'), dict) else None,
        }
        data_to_store = {k: v for k, v in data_to_store.items() if v is not None}

        # Store data using the appropriate function with retry
        stored_successfully = False
        if SQLALCHEMY_AVAILABLE:
            stored_successfully = store_data_sqlalchemy(data_to_store)
        elif psycopg2:
            stored_successfully = store_data_psycopg2(data_to_store)

        if stored_successfully:
            messages_stored.inc()
            logger.debug(f"Stored data for {device_id} at {ts_datetime.isoformat()}")

    except Exception as e:
        logger.error(f"Error processing message on {msg.topic}: {e}", exc_info=True)

# --- Graceful Shutdown ---
shutdown_flag = threading.Event()

def handle_signal(signum, frame):
    logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
    shutdown_flag.set()

# --- Main Execution ---
def main():
    logger.info("Starting MQTT Ingestor...")

    if not init_database_connection():
        logger.critical("Database connection failed on startup. Exiting.")
        return # Exit if DB connection fails

    if PROMETHEUS_AVAILABLE:
        try:
            start_http_server(PROMETHEUS_PORT)
            logger.info(f"Prometheus metrics server started on port {PROMETHEUS_PORT}")
        except Exception as e:
            logger.warning(f"Failed to start Prometheus server: {e}. Metrics may not be exposed.")

    client_userdata = {} # Userdata can store state if needed, but not used currently

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=MQTT_CLIENT_ID, userdata=client_userdata)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    # **Verification**: Configure Paho's reconnect delay
    client.reconnect_delay_set(min_delay=MQTT_RECONNECT_DELAY_MIN, max_delay=MQTT_RECONNECT_DELAY_MAX)
    logger.info(f"MQTT auto-reconnect delay set: min={MQTT_RECONNECT_DELAY_MIN}s, max={MQTT_RECONNECT_DELAY_MAX}s")


    connection_status.set(0)
    db_connection_status.set(0 if not (db_engine or db_pool) else 1) # Initial DB status

    logger.info(f"Attempting connection to MQTT broker {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}...")
    client.connect_async(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
    client.loop_start()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info("MQTT Ingestor running. Press Ctrl+C to stop.")

    while not shutdown_flag.is_set():
        try:
            # Check DB connection periodically (optional)
            # if time.monotonic() % 60 < 1: # Check roughly every minute
            #    test_db_connection()
            time.sleep(1)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Shutting down...")
            shutdown_flag.set()

    # --- Shutdown Sequence ---
    logger.info("Stopping MQTT loop...")
    client.loop_stop()
    logger.info("Disconnecting MQTT client...")
    # Add timeout to disconnect to prevent hanging
    try:
        client.disconnect()
        logger.info("MQTT client disconnected.")
    except Exception as e:
        logger.warning(f"Error during MQTT disconnect: {e}")

    if db_pool:
        db_pool.closeall()
        logger.info("psycopg2 Database connection pool closed.")
    elif db_engine:
        db_engine.dispose() # Dispose SQLAlchemy engine pool
        logger.info("SQLAlchemy engine pool disposed.")

    logger.info("MQTT Ingestor stopped.")

if __name__ == "__main__":
    from sqlalchemy import text # Import text for raw SQL test in init_db
    main()