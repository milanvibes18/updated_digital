#!/usr/bin/env python3
"""
MQTT Ingestor for Digital Twin System

Subscribes to MQTT topics for IoT device data, parses JSON payloads,
stores the data in PostgreSQL, handles QoS and reconnections,
and logs ingestion metrics for Prometheus.
"""

import paho.mqtt.client as mqtt
import json
import os
import sys
import logging
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# --- Database Import ---
# Assuming you'll use SQLAlchemy ORM similar to secure_database_manager
# Adjust imports based on your actual ORM setup if different
try:
    from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, MetaData, Table, insert
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.exc import SQLAlchemyError
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    # Fallback to psycopg2 if SQLAlchemy is not used directly here
    try:
        import psycopg2
        from psycopg2 import pool
    except ImportError:
        psycopg2 = None
    SQLALCHEMY_AVAILABLE = False
    logging.warning("SQLAlchemy not found, attempting psycopg2 fallback for database connection.")


# --- Prometheus Import ---
try:
    from prometheus_client import start_http_server, Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("prometheus_client not installed. Metrics logging disabled.")
    # Define dummy classes/functions if Prometheus is unavailable
    class Counter:
        def __init__(self, *args, **kwargs): pass; 
        def inc(self, *args, **kwargs): pass
        def start_http_server(*args, **kwargs): pass

# --- Add project root to path ---
project_root = Path(__file__).resolve().parent.parent # Adjust based on actual file location
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- Load Environment Variables ---
env_path = project_root / '.env'
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
DB_TABLE_NAME = os.getenv('DB_INGEST_TABLE', 'device_readings')

PROMETHEUS_PORT = int(os.getenv('PROMETHEUS_INGESTOR_PORT', 8001))

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
LOG_FILE = Path('LOGS') / 'mqtt_ingestor.log'

RECONNECT_DELAY_SECS = 5
MAX_RECONNECT_ATTEMPTS = 10

# --- Logging Setup ---
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
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
if PROMETHEUS_AVAILABLE:
    messages_received = Counter('mqtt_messages_received_total', 'Total number of MQTT messages received')
    messages_parsed = Counter('mqtt_messages_parsed_total', 'Total number of MQTT messages successfully parsed')
    messages_stored = Counter('mqtt_messages_stored_total', 'Total number of messages successfully stored in DB')
    parse_errors = Counter('mqtt_parse_errors_total', 'Total number of JSON parsing errors')
    db_errors = Counter('mqtt_db_errors_total', 'Total number of database insertion errors')
    connection_status = Gauge('mqtt_connection_status', 'MQTT connection status (1=connected, 0=disconnected)')
else:
    # Assign dummy counters if Prometheus is unavailable
    messages_received = messages_parsed = messages_stored = parse_errors = db_errors = Counter('dummy', 'dummy')
    connection_status = Gauge('dummy', 'dummy') if 'Gauge' in locals() else Counter('dummy', 'dummy')

# --- Database Connection ---
db_engine = None
db_pool = None

def init_database_connection():
    global db_engine, db_pool
    if not DATABASE_URL:
        logger.critical("DATABASE_URL environment variable is not set. Cannot connect to database.")
        sys.exit(1)

    if SQLALCHEMY_AVAILABLE:
        try:
            db_engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
            # Test connection
            with db_engine.connect():
                logger.info(f"SQLAlchemy connected to database via engine.")
            # Optional: Create table if it doesn't exist using metadata reflection
            metadata = MetaData()
            try:
                # Attempt to reflect table, if it fails, create it
                readings_table = Table(DB_TABLE_NAME, metadata, autoload_with=db_engine)
                logger.info(f"Table '{DB_TABLE_NAME}' already exists.")
            except Exception:
                logger.info(f"Table '{DB_TABLE_NAME}' not found, creating...")
                readings_table = Table(DB_TABLE_NAME, metadata,
                    Column('id', Integer, primary_key=True, autoincrement=True),
                    Column('device_id', String(100), nullable=False, index=True),
                    Column('timestamp', DateTime(timezone=True), nullable=False, index=True),
                    Column('data', Text), # Store raw JSON or specific fields
                    # --- Add specific columns based on expected JSON payload ---
                    Column('temperature', Float, nullable=True),
                    Column('humidity', Float, nullable=True),
                    Column('pressure', Float, nullable=True),
                    Column('vibration', Float, nullable=True),
                    Column('battery', Float, nullable=True),
                    Column('latitude', Float, nullable=True),
                    Column('longitude', Float, nullable=True)
                    # --- End specific columns ---
                )
                metadata.create_all(db_engine)
                logger.info(f"Table '{DB_TABLE_NAME}' created.")
        except Exception as e:
            logger.critical(f"Failed to connect to database using SQLAlchemy: {e}", exc_info=True)
            db_engine = None # Ensure it's None on failure
    elif psycopg2:
        try:
            # Basic connection pooling with psycopg2
            db_pool = psycopg2.pool.SimpleConnectionPool(1, 5, DATABASE_URL)
            # Test connection
            conn = db_pool.getconn()
            logger.info("psycopg2 connected to database via pool.")
            # Check/create table (example)
            cur = conn.cursor()
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {DB_TABLE_NAME} (
                    id SERIAL PRIMARY KEY,
                    device_id VARCHAR(100) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    data JSONB,
                    -- Add specific columns if needed
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
        except Exception as e:
            logger.critical(f"Failed to connect to database using psycopg2: {e}", exc_info=True)
            db_pool = None
    else:
        logger.critical("No database library (SQLAlchemy or psycopg2) available.")
        sys.exit(1)

def store_data_sqlalchemy(data_to_store: dict):
    """Stores data using SQLAlchemy engine."""
    global db_engine
    if not db_engine:
        logger.error("SQLAlchemy engine not initialized. Cannot store data.")
        return False

    metadata = MetaData()
    try:
        readings_table = Table(DB_TABLE_NAME, metadata, autoload_with=db_engine)
        stmt = insert(readings_table).values(data_to_store)
        with db_engine.connect() as connection:
            connection.execute(stmt)
            connection.commit() # Commit transaction
        return True
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemy DB insert error: {e}", exc_info=True)
        db_errors.inc()
        return False
    except Exception as e: # Catch reflection errors etc.
        logger.error(f"Error autoloading/inserting with SQLAlchemy: {e}", exc_info=True)
        db_errors.inc()
        return False


def store_data_psycopg2(data_to_store: dict):
    """Stores data using psycopg2 connection pool."""
    global db_pool
    if not db_pool:
        logger.error("psycopg2 pool not initialized. Cannot store data.")
        return False

    conn = None
    try:
        conn = db_pool.getconn()
        cur = conn.cursor()
        # Ensure keys match table columns exactly
        columns = ', '.join(data_to_store.keys())
        placeholders = ', '.join(['%s'] * len(data_to_store))
        values = [data_to_store[key] for key in data_to_store.keys()] # Keep order

        sql = f"INSERT INTO {DB_TABLE_NAME} ({columns}) VALUES ({placeholders})"
        cur.execute(sql, values)
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        logger.error(f"psycopg2 DB insert error: {e}", exc_info=True)
        if conn: conn.rollback() # Rollback on error
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
        # Subscribe to the topic
        try:
            client.subscribe(MQTT_TOPIC, qos=MQTT_QOS)
            logger.info(f"Subscribed to topic '{MQTT_TOPIC}' with QoS {MQTT_QOS}")
            # Reset reconnect attempts on successful connection
            userdata['reconnect_attempts'] = 0
        except Exception as e:
            logger.error(f"Failed to subscribe to topic '{MQTT_TOPIC}': {e}")
            # Consider attempting reconnect or exiting if subscription fails critically
    else:
        logger.error(f"Failed to connect to MQTT Broker, return code {rc}")
        connection_status.set(0)
        # Schedule reconnect attempt
        schedule_reconnect(client, userdata)

def on_disconnect(client, userdata, rc, properties=None):
    """Callback when MQTT client disconnects."""
    logger.warning(f"Disconnected from MQTT Broker with result code {rc}. Attempting reconnection...")
    connection_status.set(0)
    # Don't schedule reconnect here if rc=0 (clean disconnect),
    # but schedule if it's unexpected (rc != 0)
    if rc != 0:
        schedule_reconnect(client, userdata)

def schedule_reconnect(client, userdata):
    """Handles reconnection logic with exponential backoff."""
    if userdata['reconnect_attempts'] < MAX_RECONNECT_ATTEMPTS:
        userdata['reconnect_attempts'] += 1
        delay = RECONNECT_DELAY_SECS * (2 ** (userdata['reconnect_attempts'] - 1))
        logger.info(f"Scheduling reconnection attempt {userdata['reconnect_attempts']} in {delay} seconds...")
        time.sleep(delay) # Blocking sleep is acceptable in this simple script context
        try:
            client.reconnect()
        except Exception as e:
            logger.error(f"Reconnect attempt failed: {e}")
            # Schedule the next attempt even if this one failed
            schedule_reconnect(client, userdata)
    else:
        logger.critical(f"Maximum reconnect attempts ({MAX_RECONNECT_ATTEMPTS}) reached. Exiting.")
        # Optionally, you could stop trying or reset attempts after a longer pause
        sys.exit(1) # Exit if cannot reconnect

def on_message(client, userdata, msg):
    """Callback when a message is received."""
    messages_received.inc()
    logger.debug(f"Received message on topic '{msg.topic}': {msg.payload.decode()}")

    try:
        # Extract device_id from topic (assuming format like iot/device/DEVICE_ID/data)
        topic_parts = msg.topic.split('/')
        if len(topic_parts) >= 3:
            device_id = topic_parts[2]
        else:
            logger.warning(f"Could not extract device_id from topic: {msg.topic}")
            device_id = "unknown_device"

        # Parse JSON payload
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            messages_parsed.inc()
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON payload from {device_id} on topic {msg.topic}: {e}")
            parse_errors.inc()
            return
        except UnicodeDecodeError as e:
            logger.error(f"Failed to decode payload (not UTF-8?) from {device_id} on topic {msg.topic}: {e}")
            parse_errors.inc()
            return

        # Prepare data for storage
        timestamp = payload.get('timestamp')
        if timestamp:
            try:
                # Attempt to parse ISO format or epoch seconds/milliseconds
                if isinstance(timestamp, (int, float)):
                    # Assume milliseconds if large number, else seconds
                    ts_datetime = datetime.fromtimestamp(timestamp / 1000.0) if timestamp > 1e11 else datetime.fromtimestamp(timestamp)
                else:
                    # Handle potential 'Z' for UTC
                    timestamp = timestamp.replace('Z', '+00:00')
                    ts_datetime = datetime.fromisoformat(timestamp)
            except (ValueError, TypeError) as e:
                 logger.warning(f"Invalid timestamp format '{timestamp}' from {device_id}. Using current time. Error: {e}")
                 ts_datetime = datetime.now()
        else:
            ts_datetime = datetime.now() # Use current time if timestamp is missing


        # --- Prepare columns for DB ---
        data_to_store = {
            'device_id': device_id,
            'timestamp': ts_datetime,
            'data': json.dumps(payload), # Store the full JSON payload
            # --- Extract specific fields if they exist ---
            'temperature': payload.get('temperature'),
            'humidity': payload.get('humidity'),
            'pressure': payload.get('pressure'),
            'vibration': payload.get('vibration'),
            'battery': payload.get('battery'),
            'latitude': payload.get('location', {}).get('latitude') if isinstance(payload.get('location'), dict) else None,
            'longitude': payload.get('location', {}).get('longitude') if isinstance(payload.get('location'), dict) else None,
        }
        # Remove keys with None values to avoid DB type issues if columns are strictly typed
        data_to_store = {k: v for k, v in data_to_store.items() if v is not None}

        # Store data in PostgreSQL
        stored_successfully = False
        if SQLALCHEMY_AVAILABLE:
            stored_successfully = store_data_sqlalchemy(data_to_store)
        elif psycopg2:
            stored_successfully = store_data_psycopg2(data_to_store)

        if stored_successfully:
            messages_stored.inc()
            logger.debug(f"Stored data for device {device_id} at {ts_datetime}")
        # Error logging is handled within the store functions

    except Exception as e:
        logger.error(f"Error processing message on topic {msg.topic}: {e}", exc_info=True)
        # Increment general error counter?

# --- Main Execution ---
def main():
    logger.info("Starting MQTT Ingestor...")

    # Initialize Database Connection
    init_database_connection()
    if not (db_engine or db_pool):
        logger.critical("Database connection failed. Exiting.")
        return # Exit if DB connection fails

    # Start Prometheus metrics server
    if PROMETHEUS_AVAILABLE:
        try:
            start_http_server(PROMETHEUS_PORT)
            logger.info(f"Prometheus metrics server started on port {PROMETHEUS_PORT}")
        except Exception as e:
            logger.warning(f"Failed to start Prometheus server: {e}. Metrics may not be exposed.")

    # User data to store state like reconnect attempts
    client_userdata = {'reconnect_attempts': 0}

    # Initialize MQTT Client
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=MQTT_CLIENT_ID, userdata=client_userdata)

    # Assign callbacks
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    # Set initial connection status
    connection_status.set(0)

    # Attempt to connect
    try:
        logger.info(f"Attempting connection to MQTT broker {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}...")
        client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
    except Exception as e:
        logger.error(f"Initial connection failed: {e}. Will attempt reconnection.")
        # Reconnection logic will be triggered by on_disconnect or scheduled attempt
        schedule_reconnect(client, client_userdata) # Start reconnect attempts


    # Start the MQTT loop (blocking)
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down...")
    except Exception as e:
        logger.critical(f"Unhandled exception in MQTT loop: {e}", exc_info=True)
    finally:
        logger.info("Disconnecting MQTT client...")
        client.disconnect()
        logger.info("MQTT client disconnected.")
        if db_pool:
            db_pool.closeall()
            logger.info("Database connection pool closed.")
        # SQLAlchemy engine pool is managed automatically

if __name__ == "__main__":
    main()