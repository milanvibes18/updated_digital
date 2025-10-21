#!/usr/bin/env python3
"""
MQTT Ingestor for Digital Twin System v1.1

Subscribes to MQTT topics for IoT device data, parses JSON payloads,
stores the data in PostgreSQL, handles QoS and reconnections,
and logs ingestion metrics for Prometheus.

v1.1 Updates: Added explicit table creation, improved logging,
             graceful shutdown, fixed potential DB connection issues.
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

# --- Database Import ---
# Assuming SQLAlchemy ORM is preferred based on enhanced_flask_app_v2.py
try:
    from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, MetaData, Table, insert, Index
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.exc import SQLAlchemyError
    SQLALCHEMY_AVAILABLE = True
    psycopg2 = None # Ensure psycopg2 is None if SQLAlchemy is used
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logging.warning("SQLAlchemy not found, attempting psycopg2 fallback for database connection.")
    # Fallback to psycopg2
    try:
        import psycopg2
        from psycopg2 import pool
        from psycopg2.extras import Json # Use Json adapter for JSONB
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
project_root = Path(__file__).resolve().parent.parent.parent # Adjusted path based on file location
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

DATABASE_URL = os.getenv('DATABASE_URL')
# Use a distinct table name for raw readings if desired
DB_TABLE_NAME = os.getenv('DB_INGEST_TABLE', 'device_readings_raw') # Changed default

PROMETHEUS_PORT = int(os.getenv('PROMETHEUS_INGESTOR_PORT', 8001))

LOG_LEVEL = os.getenv('LOG_LEVEL_INGESTOR', 'INFO').upper() # Specific log level
LOG_DIR = project_root / 'Digital_Twin' / 'LOGS' # Corrected log path
LOG_FILE = LOG_DIR / 'mqtt_ingestor.log'

RECONNECT_DELAY_SECS = 5
MAX_RECONNECT_ATTEMPTS = 10

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

# --- Database Connection & Setup ---
db_engine = None
db_pool = None
# Define table structure explicitly for clarity and creation
metadata_obj = MetaData()
readings_table = Table(DB_TABLE_NAME, metadata_obj,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('device_id', String(100), nullable=False, index=True),
    Column('timestamp', DateTime(timezone=True), nullable=False, index=True),
    Column('raw_payload', Text), # Store raw JSON payload
    # Add specific columns based on expected JSON payload for easier querying
    Column('temperature', Float, nullable=True),
    Column('humidity', Float, nullable=True),
    Column('pressure', Float, nullable=True),
    Column('vibration', Float, nullable=True),
    Column('battery', Float, nullable=True),
    Column('latitude', Float, nullable=True),
    Column('longitude', Float, nullable=True),
    # Add more specific fields as needed
    Index(f'idx_{DB_TABLE_NAME}_device_timestamp', 'device_id', 'timestamp') # Composite index
)

def init_database_connection():
    global db_engine, db_pool
    if not DATABASE_URL:
        logger.critical("DATABASE_URL environment variable is not set. Cannot connect to database.")
        return False # Indicate failure

    if SQLALCHEMY_AVAILABLE:
        try:
            db_engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
            with db_engine.connect() as connection:
                logger.info(f"SQLAlchemy connected to database.")
            # Create table if it doesn't exist
            metadata_obj.create_all(db_engine)
            logger.info(f"Table '{DB_TABLE_NAME}' checked/created.")
            return True
        except Exception as e:
            logger.critical(f"Failed to connect/setup database using SQLAlchemy: {e}", exc_info=True)
            db_engine = None
            return False
    elif psycopg2:
        try:
            db_pool = psycopg2.pool.SimpleConnectionPool(1, 5, DATABASE_URL)
            conn = db_pool.getconn()
            logger.info("psycopg2 connected to database via pool.")
            cur = conn.cursor()
            # Create table using psycopg2 syntax
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {DB_TABLE_NAME} (
                    id SERIAL PRIMARY KEY,
                    device_id VARCHAR(100) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    raw_payload JSONB,
                    temperature REAL,
                    humidity REAL,
                    pressure REAL,
                    vibration REAL,
                    battery REAL,
                    latitude REAL,
                    longitude REAL
                );
            """)
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{DB_TABLE_NAME}_device_timestamp ON {DB_TABLE_NAME} (device_id, timestamp);")
            conn.commit()
            cur.close()
            db_pool.putconn(conn)
            logger.info(f"Table '{DB_TABLE_NAME}' checked/created.")
            return True
        except Exception as e:
            logger.critical(f"Failed to connect/setup database using psycopg2: {e}", exc_info=True)
            db_pool = None
            return False
    else:
        logger.critical("No database library (SQLAlchemy or psycopg2) available.")
        return False

def store_data_sqlalchemy(data_to_store: dict):
    """Stores data using SQLAlchemy engine."""
    global db_engine
    if not db_engine:
        logger.error("SQLAlchemy engine not initialized. Cannot store data.")
        db_errors.inc()
        return False

    try:
        # Use the defined table object
        stmt = insert(readings_table).values(data_to_store)
        with db_engine.connect() as connection:
            connection.execute(stmt)
            connection.commit()
        return True
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemy DB insert error: {e}", exc_info=False) # Reduce log noise slightly
        db_errors.inc()
        return False
    except Exception as e:
        logger.error(f"Unexpected error during SQLAlchemy insert: {e}", exc_info=True)
        db_errors.inc()
        return False

def store_data_psycopg2(data_to_store: dict):
    """Stores data using psycopg2 connection pool."""
    global db_pool
    if not db_pool:
        logger.error("psycopg2 pool not initialized. Cannot store data.")
        db_errors.inc()
        return False

    conn = None
    try:
        conn = db_pool.getconn()
        cur = conn.cursor()
        # Filter data_to_store to only include columns defined in the table
        # (This assumes psycopg2 table definition matches the SQLAlchemy one)
        valid_columns = [
            'device_id', 'timestamp', 'raw_payload', 'temperature', 'humidity',
            'pressure', 'vibration', 'battery', 'latitude', 'longitude'
        ]
        filtered_data = {k: v for k, v in data_to_store.items() if k in valid_columns}

        columns = ', '.join(filtered_data.keys())
        placeholders = ', '.join(['%s'] * len(filtered_data))
        values = list(filtered_data.values())

        # Special handling for JSONB raw_payload
        if 'raw_payload' in filtered_data:
            payload_index = list(filtered_data.keys()).index('raw_payload')
            values[payload_index] = Json(values[payload_index]) # Use Json adapter

        sql = f"INSERT INTO {DB_TABLE_NAME} ({columns}) VALUES ({placeholders})"
        cur.execute(sql, values)
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        logger.error(f"psycopg2 DB insert error: {e}", exc_info=False)
        if conn: conn.rollback()
        db_errors.inc()
        return False
    finally:
        if conn:
            db_pool.putconn(conn)

# --- MQTT Callbacks ---
def on_connect(client, userdata, flags, rc, properties=None):
    """Callback when MQTT client connects."""
    if rc == 0:
        logger.info(f"Connected to MQTT Broker: {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}")
        connection_status.set(1)
        try:
            # Subscribe on connect/reconnect
            client.subscribe(MQTT_TOPIC, qos=MQTT_QOS)
            logger.info(f"Subscribed to topic '{MQTT_TOPIC}' with QoS {MQTT_QOS}")
            userdata['reconnect_attempts'] = 0 # Reset attempts on success
        except Exception as e:
            logger.error(f"Failed to subscribe to topic '{MQTT_TOPIC}': {e}")
            # Consider disconnecting or retrying subscription
    else:
        logger.error(f"Failed to connect to MQTT Broker, return code {rc}")
        connection_status.set(0)
        # Reconnect logic is handled by paho-mqtt automatically or via on_disconnect
        # schedule_reconnect(client, userdata) # Manual reconnect might be needed for specific errors

def on_disconnect(client, userdata, rc, properties=None):
    """Callback when MQTT client disconnects."""
    connection_status.set(0)
    if rc != 0:
        logger.warning(f"Unexpectedly disconnected from MQTT Broker (rc={rc}). Will attempt auto-reconnect.")
        # Paho-mqtt usually handles reconnections automatically unless explicitly disabled.
        # schedule_reconnect(client, userdata) # Add manual backoff if needed
    else:
        logger.info("Cleanly disconnected from MQTT Broker.")

# --- REMOVED schedule_reconnect: Rely on Paho's built-in reconnect ---

def on_message(client, userdata, msg):
    """Callback when a message is received."""
    messages_received.inc()
    logger.debug(f"Received message on topic '{msg.topic}': {msg.payload[:100].decode()}...") # Log snippet

    try:
        # Extract device_id from topic (e.g., iot/device/DEVICE_ID/data)
        topic_parts = msg.topic.split('/')
        if len(topic_parts) >= 3 and topic_parts[0] == 'iot' and topic_parts[1] == 'device':
            device_id = topic_parts[2]
        else:
            logger.warning(f"Could not extract device_id from topic: {msg.topic}")
            device_id = "unknown_device"

        # Parse JSON payload
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

        # Prepare data for storage - Timestamp Handling
        timestamp_str = payload.get('timestamp')
        ts_datetime = None
        if timestamp_str:
            try:
                # Try ISO format (handle 'Z' or offset)
                if isinstance(timestamp_str, str):
                    timestamp_str = timestamp_str.replace('Z', '+00:00')
                    ts_datetime = datetime.fromisoformat(timestamp_str)
                # Try epoch seconds/milliseconds
                elif isinstance(timestamp_str, (int, float)):
                    # Assume milliseconds if large number, else seconds
                    timestamp_val = float(timestamp_str)
                    ts_datetime = datetime.fromtimestamp(timestamp_val / 1000.0, tz=timezone.utc) if timestamp_val > 1e11 else datetime.fromtimestamp(timestamp_val, tz=timezone.utc)
                else:
                    raise ValueError("Unsupported timestamp type")

                # Ensure timezone is UTC
                if ts_datetime.tzinfo is None:
                    ts_datetime = ts_datetime.replace(tzinfo=timezone.utc)
                else:
                    ts_datetime = ts_datetime.astimezone(timezone.utc)

            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid timestamp format '{timestamp_str}' from {device_id}. Using current time. Error: {e}")
                ts_datetime = datetime.now(timezone.utc)
        else:
            logger.debug(f"Timestamp missing in payload from {device_id}. Using current time.")
            ts_datetime = datetime.now(timezone.utc)

        # Prepare data dict based on table schema
        data_to_store = {
            'device_id': device_id,
            'timestamp': ts_datetime,
            'raw_payload': payload_str if SQLALCHEMY_AVAILABLE else payload, # Store raw JSON string or dict
            # Extract specific fields safely using .get()
            'temperature': payload.get('temperature'),
            'humidity': payload.get('humidity'),
            'pressure': payload.get('pressure'),
            'vibration': payload.get('vibration'),
            'battery': payload.get('battery'),
            'latitude': payload.get('location', {}).get('latitude') if isinstance(payload.get('location'), dict) else None,
            'longitude': payload.get('location', {}).get('longitude') if isinstance(payload.get('location'), dict) else None,
        }
        # Remove keys with None values if DB columns don't allow NULLs easily
        data_to_store = {k: v for k, v in data_to_store.items() if v is not None}


        # Store data using the appropriate function
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

    # Initialize Database Connection
    if not init_database_connection():
        logger.critical("Database connection failed. Exiting.")
        return # Exit if DB connection fails

    # Start Prometheus metrics server
    if PROMETHEUS_AVAILABLE:
        try:
            start_http_server(PROMETHEUS_PORT)
            logger.info(f"Prometheus metrics server started on port {PROMETHEUS_PORT}")
        except Exception as e:
            logger.warning(f"Failed to start Prometheus server: {e}. Metrics may not be exposed.")

    # User data to store state
    client_userdata = {'reconnect_attempts': 0}

    # Initialize MQTT Client
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=MQTT_CLIENT_ID, userdata=client_userdata)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    # Optional: Add username/password authentication if needed
    # mqtt_user = os.getenv('MQTT_USERNAME')
    # mqtt_password = os.getenv('MQTT_PASSWORD')
    # if mqtt_user and mqtt_password:
    #     client.username_pw_set(mqtt_user, mqtt_password)
    #     logger.info("MQTT username/password authentication enabled.")

    # Set initial connection status
    connection_status.set(0)

    # Attempt to connect (non-blocking)
    logger.info(f"Attempting connection to MQTT broker {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}...")
    client.connect_async(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)

    # Start the MQTT loop in a background thread
    client.loop_start()

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info("MQTT Ingestor running. Press Ctrl+C to stop.")

    # Keep the main thread alive, checking for shutdown signal
    while not shutdown_flag.is_set():
        try:
            # You can add periodic health checks or other tasks here if needed
            time.sleep(1)
        except KeyboardInterrupt: # Handle Ctrl+C in the sleep
            logger.info("KeyboardInterrupt received. Shutting down...")
            shutdown_flag.set()

    # --- Shutdown Sequence ---
    logger.info("Stopping MQTT loop...")
    client.loop_stop()
    logger.info("Disconnecting MQTT client...")
    client.disconnect()
    logger.info("MQTT client disconnected.")

    if db_pool:
        db_pool.closeall()
        logger.info("psycopg2 Database connection pool closed.")
    # SQLAlchemy engine pool manages itself generally

    logger.info("MQTT Ingestor stopped.")

if __name__ == "__main__":
    main()