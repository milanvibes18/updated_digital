import numpy as np
import pandas as pd
import sqlite3
import json
import logging
import random
import uuid
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import math
import warnings
# --- NEW: Import dotenv ---
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# --- NEW: Load .env file ---
# Load environment variables from .env file, if it exists
# This should be called early in the script
env_path = Path('.') / '.env'
if env_path.is_file():
    load_dotenv(dotenv_path=env_path, verbose=True)
    print(f"Loaded environment variables from: {env_path}")
else:
    print("Warning: .env file not found. Relying on system environment variables.")
# --- End Load ---

# --- Optional MQTT Import ---
try:
    import paho.mqtt.client as mqtt
    PAHO_MQTT_AVAILABLE = True
except ImportError:
    PAHO_MQTT_AVAILABLE = False
    print("Warning: 'paho-mqtt' library not found. MQTT functionality will be disabled.")
    # Define a dummy class if import fails to avoid NameErrors
    class mqtt:
        Client = object

# --- Merged Feature (Config Import from File 2) ---
# Add project root to path
# Use relative pathing to be more robust
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)


try:
    # --- UPDATED Import Path ---
    # Assuming app_config is directly inside CONFIG now, relative to project root
    from CONFIG.app_config import config
except ImportError:
    print("Warning: CONFIG.app_config not found. Using default paths.")
    class Config:
        class Database:
            primary_path = 'DATABASE/health_data.db'
        database = Database()
    config = Config()
# --- End Merged Feature ---

class UnifiedDataGenerator:
    """
    Comprehensive data generator for Digital Twin applications.
    Generates realistic industrial IoT data with patterns, anomalies, trends,
    correlations, and maintenance events.

    Can generate historical batch datasets or run real-time simulations with
    optional MQTT publishing.
    """

    # --- UPDATED __init__ with MQTT support AND .env loading ---
    def __init__(self,
                 db_path: str = None,
                 seed: int = 42,
                 mqtt_broker: Optional[str] = None,
                 mqtt_port: Optional[int] = None): # Allow None for port

        self.db_path = db_path or config.database.primary_path
        self.seed = seed
        self.logger = self._setup_logging()

        # Set random seeds for reproducibility
        np.random.seed(seed)
        random.seed(seed)

        # Device configurations (Includes configurable noise_factor per type)
        self.device_types = {
            'temperature_sensor': {
                'normal_range': (15, 35), 'critical_range': (0, 60), 'noise_factor': 0.5,
                'seasonal_amplitude': 5, 'daily_amplitude': 3
            },
            'pressure_sensor': {
                'normal_range': (900, 1100), 'critical_range': (800, 1200), 'noise_factor': 2.0,
                'seasonal_amplitude': 10, 'daily_amplitude': 5
            },
            'vibration_sensor': {
                'normal_range': (0.1, 0.3), 'critical_range': (0, 1.0), 'noise_factor': 0.02,
                'seasonal_amplitude': 0.05, 'daily_amplitude': 0.1
            },
            'humidity_sensor': {
                'normal_range': (40, 70), 'critical_range': (10, 90), 'noise_factor': 1.0,
                'seasonal_amplitude': 15, 'daily_amplitude': 8
            },
            'power_meter': {
                'normal_range': (1000, 5000), 'critical_range': (0, 10000), 'noise_factor': 50,
                'seasonal_amplitude': 500, 'daily_amplitude': 800
            }
        }

        # Location data
        self.locations = [
            {'name': 'Factory Floor A', 'x': 10, 'y': 20, 'zone': 'production'},
            {'name': 'Factory Floor B', 'x': 50, 'y': 20, 'zone': 'production'},
            {'name': 'Warehouse Section 1', 'x': 10, 'y': 80, 'zone': 'storage'},
            {'name': 'Warehouse Section 2', 'x': 50, 'y': 80, 'zone': 'storage'},
            {'name': 'Quality Control Lab', 'x': 80, 'y': 50, 'zone': 'quality'},
            {'name': 'Maintenance Workshop', 'x': 90, 'y': 10, 'zone': 'maintenance'},
            {'name': 'Office Building', 'x': 20, 'y': 90, 'zone': 'administrative'},
            {'name': 'Server Room', 'x': 85, 'y': 85, 'zone': 'it_infrastructure'}
        ]

        # Simulation parameters
        self.anomaly_probability = 0.02
        self.trend_probability = 0.1
        self.correlation_map = {
            'vibration_sensor': {'target': 'temperature_sensor', 'effect': 0.05}
        }
        self.maintenance_probability = 0.02

        # --- State variables for simulation ---
        self.devices = []
        self.device_drifts = {}
        self.sim_cumulative_energy = 0

        # --- MQTT Setup (Reading from Env Vars) ---
        # Prioritize arguments, then env vars, then defaults
        effective_mqtt_broker = mqtt_broker or os.getenv('MQTT_BROKER_HOST')
        try:
            # Handle potential ValueError if env var is not an int
            effective_mqtt_port = mqtt_port if mqtt_port is not None else int(os.getenv('MQTT_BROKER_PORT', 1883))
        except (ValueError, TypeError):
             self.logger.warning(f"Invalid MQTT_BROKER_PORT environment variable. Using default 1883.")
             effective_mqtt_port = 1883 # Fallback to default if conversion fails


        self.mqtt_client: Optional[mqtt.Client] = None
        self.mqtt_connected = False
        if effective_mqtt_broker and PAHO_MQTT_AVAILABLE:
            self._setup_mqtt(effective_mqtt_broker, effective_mqtt_port)
        elif mqtt_broker and not PAHO_MQTT_AVAILABLE: # Log error only if explicitly passed
            self.logger.error("MQTT broker provided, but 'paho-mqtt' library is not installed.")
        elif effective_mqtt_broker and not PAHO_MQTT_AVAILABLE: # Log warning if from env
             self.logger.warning("MQTT broker configured via environment, but 'paho-mqtt' not installed.")
        # --- End MQTT Setup ---

    def __del__(self):
        """Clean up resources, especially MQTT connection."""
        if self.mqtt_client and self.mqtt_connected:
            self.logger.info("Disconnecting from MQTT broker.")
            self._disconnect_mqtt()

    def _setup_logging(self):
        """Setup logging for data generator."""
        logger = logging.getLogger('UnifiedDataGenerator')
        logger.setLevel(logging.INFO)

        log_dir = Path('LOGS')
        log_dir.mkdir(parents=True, exist_ok=True) # Ensure LOGS dir exists

        if not logger.handlers:
            handler = logging.FileHandler(log_dir / 'digital_twin_generator.log') # Changed log file name
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            # Optional: Add console handler for immediate feedback
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    # --- New MQTT Methods ---

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.logger.info(f"Successfully connected to MQTT broker.")
            self.mqtt_connected = True
        else:
            self.logger.error(f"Failed to connect to MQTT broker, return code {rc}")

    def _on_mqtt_disconnect(self, client, userdata, rc):
        self.logger.info("Disconnected from MQTT broker.")
        self.mqtt_connected = False

    def _setup_mqtt(self, broker: str, port: int):
        """Initialize and connect the MQTT client."""
        if not PAHO_MQTT_AVAILABLE:
            return

        try:
            client_id = f'digital-twin-generator-{uuid.uuid4()}'
            self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id) # Specify API version
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect

            self.logger.info(f"Attempting MQTT connection to {broker}:{port}...")
            # Added timeout to connect_async
            self.mqtt_client.connect_async(broker, port, 60)
            self.mqtt_client.loop_start()

            # Add a small delay to allow connection attempt
            # time.sleep(2) # Can uncomment if connection status is needed immediately

        except Exception as e:
            self.logger.error(f"Error setting up MQTT client: {e}", exc_info=True)
            self.mqtt_client = None

    def _disconnect_mqtt(self):
        """Disconnect the MQTT client."""
        if self.mqtt_client:
            try:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
            except Exception as e:
                self.logger.error(f"Error during MQTT disconnect: {e}", exc_info=True)
            finally:
                self.mqtt_connected = False
                self.mqtt_client = None

    def _publish_mqtt(self, topic: str, payload: Dict):
        """Publish a data payload to an MQTT topic."""
        if self.mqtt_client and self.mqtt_connected:
            try:
                # Convert datetime to ISO format string for JSON serialization
                def dt_converter(o):
                    if isinstance(o, (datetime, pd.Timestamp)):
                        return o.isoformat()
                    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

                json_payload = json.dumps(payload, default=dt_converter)
                result, mid = self.mqtt_client.publish(topic, json_payload)
                if result != mqtt.MQTT_ERR_SUCCESS:
                     self.logger.warning(f"MQTT publish failed for topic {topic} with code {result}")
                # else:
                #     self.logger.debug(f"Published MQTT message mid={mid} to {topic}")

            except TypeError as e:
                 self.logger.error(f"Failed to serialize payload for MQTT: {e}. Payload: {payload}")
            except Exception as e:
                self.logger.warning(f"Failed to publish MQTT message to {topic}: {e}", exc_info=True)
        elif self.mqtt_client and not self.mqtt_connected:
             self.logger.debug(f"Skipping MQTT publish: Client not connected.")
        # else: No client configured

    # --- Batch Generation Methods (Original Logic - No changes needed here) ---
    # ... (generate_device_data, _generate_device_metadata, _generate_device_time_series) ...
    def generate_device_data(self,
                             device_count: int = 20,
                             days_of_data: int = 30,
                             interval_minutes: int = 5) -> pd.DataFrame:
        """
        Generate comprehensive historical device sensor data.
        """
        try:
            self.logger.info(f"Generating historical device data for {device_count} devices over {days_of_data} days")

            start_time = datetime.now() - timedelta(days=days_of_data)
            end_time = datetime.now()
            time_points = pd.date_range(start_time, end_time, freq=f'{interval_minutes}min')

            # Setup devices and drifts for this batch generation
            self.setup_simulation(device_count, start_time)

            all_data = []

            for device in self.devices:
                # Generate the full time series for this device
                device_data = self._generate_device_time_series(device, time_points)
                all_data.extend(device_data)

            df = pd.DataFrame(all_data)
            df = self._add_calculated_fields(df)
            df = self._add_maintenance_events(df)

            anomaly_count = len(df[df['status'] == 'anomaly'])
            self.logger.info(f"Added {anomaly_count} anomalies to historical dataset")

            self.logger.info(f"Generated {len(df):,} historical data records")
            return df

        except Exception as e:
            self.logger.error(f"Device data generation error: {e}", exc_info=True)
            raise

    def _generate_device_metadata(self, count: int, start_time: datetime) -> List[Dict]:
        """Generate metadata for devices."""
        devices = []
        for i in range(count):
            device_type = np.random.choice(list(self.device_types.keys()))
            location = random.choice(self.locations)

            device = {
                'device_id': f"DEVICE_{i+1:03d}",
                'device_name': f"{device_type.replace('_', ' ').title()} {i+1:03d}",
                'device_type': device_type,
                'location': location['name'],
                'location_x': location['x'],
                'location_y': location['y'],
                'zone': location['zone'],
                'installation_date': (start_time + timedelta(days=random.randint(0, 30))).isoformat(), # Store as ISO string
                'manufacturer': random.choice(['Siemens', 'ABB', 'Schneider', 'Rockwell', 'Honeywell']),
                'model': f"Model-{random.randint(1000, 9999)}",
                'firmware_version': f"{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
                'status': 'active',
                'last_maintenance': (start_time + timedelta(days=random.randint(0, 60))).isoformat(), # Store as ISO string
                'config': self.device_types[device_type] # Config might contain non-serializable, handle if needed later
            }
            devices.append(device)
        return devices

    def _generate_device_time_series(self, device: Dict, time_points: pd.DatetimeIndex) -> List[Dict]:
        """Generate time series data for a single device for batch processing."""
        device_data = []
        for i, timestamp in enumerate(time_points):
            # Use the new single-reading generator
            record = self._generate_single_reading(device, timestamp, i)
            device_data.append(record)
        return device_data


    # --- Refactored Single-Reading Generators (No changes needed) ---
    # ... (_generate_single_reading, _generate_single_system_metric, _generate_single_energy_metric) ...
    def _generate_single_reading(self, device: Dict, timestamp: pd.Timestamp, step_index: int) -> Dict:
        """Generates a single, stateful data reading for a device at a given timestamp."""
        config = device['config']
        device_id = device['device_id']

        # Base value with patterns
        base_value = self._calculate_base_value(timestamp, config)

        # Apply long-term temporal trend/drift
        current_drift = self.device_drifts.get(device_id, 0)
        base_value += current_drift * step_index

        # Add noise (Uses configurable noise_factor)
        noise = np.random.normal(0, config['noise_factor'])
        value = base_value + noise

        # Anomaly and Correlation logic
        status = 'normal'
        if random.random() < self.anomaly_probability:
            status = 'anomaly'
            if random.random() < 0.5: # High anomaly
                value = random.uniform(config['critical_range'][1] * 0.8, config['critical_range'][1])
            else: # Low anomaly
                value = random.uniform(config['critical_range'][0], config['critical_range'][0] + (config['critical_range'][1] - config['critical_range'][0]) * 0.2)

            # Correlation Logic
            if device['device_type'] in self.correlation_map:
                correlation = self.correlation_map[device['device_type']]
                target_device_type = correlation['target']
                effect = correlation['effect']

                # Apply effect to other devices
                for dev_id, drift in self.device_drifts.items():
                    if target_device_type in dev_id and dev_id != device_id:
                        self.device_drifts[dev_id] += effect

        # Ensure value is within critical range
        value = max(config['critical_range'][0], min(config['critical_range'][1], value))

        # Assemble record
        record = {
            'timestamp': timestamp, # Keep as datetime object here
            'device_id': device['device_id'],
            'device_name': device['device_name'],
            'device_type': device['device_type'],
            'location': device['location'],
            'location_x': device['location_x'],
            'location_y': device['location_y'],
            'zone': device['zone'],
            'value': round(value, 3),
            'unit': self._get_unit_for_device_type(device['device_type']),
            'status': status,
            'quality': round(random.uniform(0.95, 1.0), 3),
            'signal_strength': random.randint(80, 100),
            'battery_level': round(random.uniform(0.7, 1.0), 2) if 'wireless' in device.get('connection_type', '') else None
        }

        # Add device-specific metrics
        if device['device_type'] == 'temperature_sensor':
            humidity = round(random.uniform(40, 70), 1)
            record.update({
                'humidity': humidity,
                'heat_index': self._calculate_heat_index(value, humidity)
            })
        elif device['device_type'] == 'vibration_sensor':
            record.update({
                'frequency_hz': round(random.uniform(10, 100), 1),
                'amplitude_mm': round(value, 3), # Use main value as amplitude
                'rms_velocity': round(random.uniform(0.5, 5.0), 2)
            })
        elif device['device_type'] == 'power_meter':
            voltage = round(random.uniform(220, 240), 1)
            current = round(value / voltage, 2) if voltage > 0 and value > 0 else 0
            power_factor = round(random.uniform(0.8, 1.0), 2)
            # Estimate energy consumed in this interval (e.g., if interval is 5 mins = 1/12 hour)
            # This is simplified; real energy meters track cumulative
            interval_hours = 5 / 60.0 # Assuming 5 min interval for example
            energy_consumed = round(value * interval_hours / 1000.0, 4) # in kWh
            record.update({
                'voltage': voltage,
                'current': current,
                'power_factor': power_factor,
                'energy_consumed_kwh_interval': energy_consumed
            })

        return record

    def _generate_single_system_metric(self, timestamp: pd.Timestamp) -> Dict:
        """Generates a single system metric record for a given timestamp."""
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5

        if 8 <= hour <= 18 and not is_weekend:
            base_load = random.uniform(60, 90)
        elif 6 <= hour <= 22:
            base_load = random.uniform(30, 70)
        else:
            base_load = random.uniform(10, 40)

        seasonal_factor = math.sin(2 * math.pi * timestamp.dayofyear / 365.25)
        load_adjustment = seasonal_factor * 10
        system_load = max(0, min(100, base_load + load_adjustment))

        record = {
            'timestamp': timestamp,
            'metric_type': 'system_performance',
            'cpu_usage_percent': round(max(0, min(100, system_load + random.uniform(-10, 10))), 2),
            'memory_usage_percent': round(max(0, min(100, system_load * 0.8 + random.uniform(-5, 15))), 2),
            'disk_usage_percent': round(random.uniform(45, 75), 2),
            'network_io_mbps': round(max(0, system_load * 0.5 + random.uniform(0, 20)), 2),
            'active_connections': int(max(0, system_load * 2 + random.randint(0, 50))),
            'response_time_ms': round(max(50, 100 + (system_load - 50) * 2 + random.uniform(-20, 50)), 2),
            'error_rate_percent': round(max(0, min(10, (system_load - 80) * 0.1 + random.uniform(0, 0.5))), 3),
            'throughput_rps': round(max(10, 1000 - (system_load - 50) * 5 + random.uniform(-100, 200)), 2),
            'availability_percent': round(max(95, min(100, 100 - (system_load - 70) * 0.1 + random.uniform(-1, 1))), 3)
        }
        return record

    def _generate_single_energy_metric(self, timestamp: pd.Timestamp) -> Dict:
        """Generates a single energy metric record, updating cumulative state."""
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5

        if 6 <= hour <= 22 and not is_weekend:
            base_consumption = random.uniform(800, 1500) # kW
        elif 22 <= hour or hour <= 6:
            base_consumption = random.uniform(200, 600) # kW
        else:  # Weekend
            base_consumption = random.uniform(300, 800) # kW

        seasonal_factor = math.sin(2 * math.pi * timestamp.dayofyear / 365.25)
        seasonal_adjustment = seasonal_factor * 300 # +/- 300 kW based on season

        total_consumption_kw = max(0, base_consumption + seasonal_adjustment)

        # Update stateful cumulative energy (assuming 15min interval)
        interval_hours = 15 / 60.0 # 15 minutes = 0.25 hours
        self.sim_cumulative_energy += total_consumption_kw * interval_hours # kWh

        voltage = round(random.uniform(220, 240), 1)
        current_a = round(total_consumption_kw * 1000 / voltage, 2) if voltage > 0 else 0 # A = W / V

        record = {
            'timestamp': timestamp,
            'metric_type': 'energy',
            'power_consumption_kw': round(total_consumption_kw, 2),
            'energy_consumed_kwh_cumulative': round(self.sim_cumulative_energy, 2),
            'voltage_v': voltage,
            'current_a': current_a,
            'power_factor': round(random.uniform(0.8, 0.95), 3),
            'frequency_hz': round(random.uniform(49.9, 50.1), 2),
            'energy_cost_usd': round(self.sim_cumulative_energy * 0.12, 2),  # Example: $0.12 per kWh
            'carbon_footprint_kg_co2': round(self.sim_cumulative_energy * 0.4, 2),  # Example: 0.4 kg CO2 per kWh
            'efficiency_percent': round(random.uniform(85, 95), 2),
            'renewable_percent': round(random.uniform(20, 40), 2)
        }
        return record


    # --- Helper Methods (No changes needed) ---
    # ... (_calculate_base_value, _get_unit_for_device_type, _calculate_heat_index) ...
    def _calculate_base_value(self, timestamp: pd.Timestamp, config: Dict) -> float:
        """Calculate base value with seasonal and daily patterns."""
        normal_min, normal_max = config['normal_range']
        base_value = (normal_min + normal_max) / 2

        day_of_year = timestamp.dayofyear
        seasonal_factor = math.sin(2 * math.pi * day_of_year / 365.25)
        seasonal_adjustment = seasonal_factor * config['seasonal_amplitude']

        hour_of_day = timestamp.hour + timestamp.minute / 60.0
        daily_factor = math.sin(2 * math.pi * hour_of_day / 24)
        daily_adjustment = daily_factor * config['daily_amplitude']

        weekly_factor = 1.0
        if timestamp.weekday() >= 5:  # Weekend
            weekly_factor = 0.7

        return base_value + seasonal_adjustment + daily_adjustment * weekly_factor

    def _get_unit_for_device_type(self, device_type: str) -> str:
        """Get measurement unit for device type."""
        unit_mapping = {
            'temperature_sensor': '°C', 'pressure_sensor': 'hPa',
            'vibration_sensor': 'mm/s', 'humidity_sensor': '%RH',
            'power_meter': 'W' # Base unit is Watts for power meters
        }
        return unit_mapping.get(device_type, 'units')

    def _calculate_heat_index(self, temperature: float, humidity: float) -> float:
        """Calculate heat index from temperature and humidity."""
        # Ensure temperature is in Fahrenheit for the formula
        temp_f = temperature * 9/5 + 32

        if temp_f < 80:
             # Formula is generally valid for T >= 80F and RH >= 40%
             # Return original temp in Celsius if below threshold
             return round(temperature, 2)

        # Steadman's formula constants
        c1 = -42.379
        c2 = 2.04901523
        c3 = 10.14333127
        c4 = -0.22475541
        c5 = -6.83783e-3
        c6 = -5.481717e-2
        c7 = 1.22874e-3
        c8 = 8.5282e-4
        c9 = -1.99e-6

        hi_f = (c1 +
                c2 * temp_f +
                c3 * humidity +
                c4 * temp_f * humidity +
                c5 * temp_f**2 +
                c6 * humidity**2 +
                c7 * temp_f**2 * humidity +
                c8 * temp_f * humidity**2 +
                c9 * temp_f**2 * humidity**2)

        # Convert heat index back to Celsius
        hi_c = (hi_f - 32) * 5/9
        return round(hi_c, 2)


    # --- DataFrame Augmentation (No changes needed) ---
    # ... (_add_calculated_fields, _add_maintenance_events) ...
    def _add_calculated_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated fields to the dataset."""
        if df.empty:
            return df
        df_copy = df.copy()

        # Ensure timestamp is datetime
        if 'timestamp' in df_copy.columns and not pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
             df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])

        def calculate_health(row):
            try:
                # Use pre-loaded config if available
                dev_type_config = self.device_types.get(row['device_type'])
                if not dev_type_config: return 0.5 # Default if type unknown

                n_min, n_max = dev_type_config['normal_range']
                c_min, c_max = dev_type_config['critical_range']

                if row['status'] == 'anomaly' or row['status'] == 'maintenance': return 0.2
                if n_min <= row['value'] <= n_max: return round(random.uniform(0.9, 1.0), 3)

                # Calculate deviation normalized by the range outside normal
                if row['value'] > n_max:
                    # Avoid division by zero if critical max equals normal max
                    deviation_range = max(c_max - n_max, 1e-6)
                    deviation = (row['value'] - n_max) / deviation_range
                else: # value < n_min
                    deviation_range = max(n_min - c_min, 1e-6)
                    deviation = (n_min - row['value']) / deviation_range

                # Score decreases from 0.9 based on deviation, clamped at 0.1
                return round(max(0.1, 0.9 - deviation), 3)

            except Exception as e:
                self.logger.warning(f"Error calculating health for row: {row}. Error: {e}")
                return 0.5 # Default score on error

        df_copy['health_score'] = df_copy.apply(calculate_health, axis=1)

        # Efficiency loosely based on health, with some random variation
        df_copy['efficiency_score'] = (df_copy['health_score'] - df_copy.apply(lambda x: random.uniform(0.05, 0.15), axis=1)).clip(0, 1)
        df_copy['efficiency_score'] = df_copy['efficiency_score'].round(3)

        # Calculate operating hours (requires sorted data)
        df_copy = df_copy.sort_values(['device_id', 'timestamp'])
        if 'timestamp' in df_copy.columns and len(df_copy) > 1:
            # Calculate time difference in hours between consecutive readings per device
            time_diff_hours = df_copy.groupby('device_id')['timestamp'].diff().dt.total_seconds() / 3600.0
            # Cumulative sum of hours for each device
            df_copy['operating_hours'] = df_copy.groupby('device_id')['timestamp'].transform(
                 lambda x: (x - x.min()).dt.total_seconds() / 3600.0
            ).round(2)
        else:
             df_copy['operating_hours'] = 0.0

        # Calculate days since maintenance (based on simulated metadata)
        # This requires merging metadata or recalculating based on 'maintenance' status flags
        # Simplified: Days since first record for the device
        if 'timestamp' in df_copy.columns:
            df_copy['days_since_start'] = (
                df_copy['timestamp'] - df_copy.groupby('device_id')['timestamp'].transform('min')
            ).dt.days
        else:
             df_copy['days_since_start'] = 0

        # Replace 'days_since_maintenance' with this simpler version for now
        df_copy['days_since_maintenance'] = df_copy['days_since_start']


        return df_copy

    def _add_maintenance_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add maintenance events to the dataset."""
        if df.empty or self.maintenance_probability <= 0:
            return df
        df = df.copy()
        maintenance_count = 0
        num_events = int(len(df) * self.maintenance_probability)

        if num_events == 0:
             self.logger.info("No maintenance events added based on probability.")
             return df

        # Select random indices for maintenance
        maintenance_indices = np.random.choice(df.index, size=num_events, replace=False)

        for idx in maintenance_indices:
            df.loc[idx, 'status'] = 'maintenance'
            df.loc[idx, 'value'] = 0.0 # Typically sensors read 0 or null during maintenance
            df.loc[idx, 'health_score'] = 1.0 # Assume health resets after maintenance
            df.loc[idx, 'efficiency_score'] = 0.0 # Efficiency is 0 when under maintenance
            df.loc[idx, 'quality'] = 1.0 # Data quality is perfect (it's a known state)
            maintenance_count += 1

        self.logger.info(f"Added {maintenance_count} maintenance events to dataset")
        return df


    # --- Data Generation Orchestration (No changes needed) ---
    # ... (generate_system_metrics, generate_energy_data, save_to_database, ...) ...
    def generate_system_metrics(self, days_of_data: int = 30) -> pd.DataFrame:
        """Generate historical system-level performance metrics."""
        try:
            self.logger.info(f"Generating historical system metrics for {days_of_data} days")
            start_time = datetime.now() - timedelta(days=days_of_data)
            time_points = pd.date_range(start_time, datetime.now(), freq='1H') # Hourly metrics
            system_data = [self._generate_single_system_metric(ts) for ts in time_points]
            df = pd.DataFrame(system_data)
            self.logger.info(f"Generated {len(df):,} system metric records")
            return df
        except Exception as e:
            self.logger.error(f"System metrics generation error: {e}", exc_info=True)
            raise

    def generate_energy_data(self, days_of_data: int = 30) -> pd.DataFrame:
        """Generate historical energy consumption and efficiency data."""
        try:
            self.logger.info(f"Generating historical energy data for {days_of_data} days")
            start_time = datetime.now() - timedelta(days=days_of_data)
            time_points = pd.date_range(start_time, datetime.now(), freq='15min') # 15-min interval

            # Reset cumulative energy for batch generation
            self.sim_cumulative_energy = 0
            energy_data = [self._generate_single_energy_metric(ts) for ts in time_points]

            df = pd.DataFrame(energy_data)
            self.logger.info(f"Generated {len(df):,} energy records")
            return df
        except Exception as e:
            self.logger.error(f"Energy data generation error: {e}", exc_info=True)
            raise

    def save_to_database(self, dataframes: Dict[str, pd.DataFrame]):
        """Save generated dataframes to the configured SQLite database."""
        try:
            db_file = Path(self.db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Connecting to database: {self.db_path}")

            with sqlite3.connect(self.db_path) as conn:
                for table_name, df in dataframes.items():
                    if df.empty:
                        self.logger.warning(f"Skipping empty dataframe for table '{table_name}'")
                        continue

                    # Convert datetime columns to string format suitable for SQLite
                    df_copy = df.copy()
                    for col in df_copy.select_dtypes(include=['datetime64[ns]']).columns:
                        df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S.%f')

                    self.logger.info(f"Saving {len(df_copy):,} records to table '{table_name}'")
                    try:
                        df_copy.to_sql(table_name, conn, if_exists='replace', index=False)
                        self.logger.info(f"Successfully saved to table '{table_name}'")
                    except Exception as sql_e:
                        self.logger.error(f"Failed to save table '{table_name}' to database: {sql_e}", exc_info=True)

            self.logger.info(f"All data saved to database: {self.db_path}")
        except sqlite3.Error as e:
            self.logger.error(f"SQLite database error during save: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"General database save error: {e}", exc_info=True)
            raise


    # --- Simulation Methods (No changes needed) ---
    # ... (setup_simulation, run_device_simulation, run_system_simulation, run_energy_simulation) ...
    def setup_simulation(self, device_count: int, start_time: Optional[datetime] = None):
        """
        Initialize devices and their states for a new simulation.
        """
        self.logger.info(f"Setting up simulation for {device_count} devices...")
        if start_time is None:
            start_time = datetime.now()

        # Generate device metadata
        self.devices = self._generate_device_metadata(device_count, start_time)

        # Initialize device drifts/trends
        self.device_drifts = {}
        for device in self.devices:
            if random.random() < self.trend_probability:
                # Assign a small, random drift factor
                self.device_drifts[device['device_id']] = random.uniform(-0.005, 0.005) * (self.device_types[device['device_type']]['normal_range'][1] - self.device_types[device['device_type']]['normal_range'][0])
            else:
                self.device_drifts[device['device_id']] = 0

        # Reset other state variables
        self.sim_cumulative_energy = 0
        self.logger.info("Simulation setup complete.")

    def run_device_simulation(self,
                              duration_days: float,
                              interval_minutes: int,
                              simulation_speed: float = 1.0,
                              publish_mqtt: bool = False):
        """
        Run a real-time device simulation, yielding data or publishing to MQTT.

        Args:
            duration_days: How long the simulation should run in *simulated time*.
            interval_minutes: The gap between data points in *simulated time*.
            simulation_speed: Multiplier for real-time. 1.0 = real-time,
                              60.0 = 1 min of sim time passes in 1 sec.
                              Use float('inf') for fastest possible generation (no sleep).
            publish_mqtt: If True, publish to MQTT. If False, yield data batches.

        Yields:
            List[Dict]: A batch of readings for one timestamp if publish_mqtt is False.
        """
        if not self.devices:
            self.logger.warning("No devices set up. Call setup_simulation() first. Defaulting to 10 devices.")
            self.setup_simulation(device_count=10)

        sim_start_time = datetime.now()
        sim_end_time = sim_start_time + timedelta(days=duration_days)
        time_delta = timedelta(minutes=interval_minutes)

        # Calculate sleep time, handle infinite speed
        if simulation_speed <= 0 or math.isinf(simulation_speed):
            real_time_sleep = 0
            speed_log = "max (no sleep)"
        else:
            real_time_sleep = (interval_minutes * 60) / simulation_speed
            speed_log = f"{simulation_speed}x"

        self.logger.info(f"Starting device simulation for {duration_days} simulated days...")
        self.logger.info(f"Interval: {interval_minutes} min | Speed: {speed_log} | Sleep: {real_time_sleep:.3f}s | MQTT: {publish_mqtt}")

        current_sim_time = sim_start_time
        step_index = 0

        try:
            last_log_time = time.monotonic()
            while current_sim_time < sim_end_time:
                loop_start_time = time.monotonic()
                readings_batch = []
                for device in self.devices:
                    # Pass pandas Timestamp for consistency
                    reading = self._generate_single_reading(device, pd.Timestamp(current_sim_time), step_index)
                    readings_batch.append(reading)

                if publish_mqtt:
                    if self.mqtt_client and self.mqtt_connected:
                        for reading in readings_batch:
                            topic = f"digital_twin/device/{reading['device_id']}"
                            # Make a copy to avoid modifying batch before potential yield
                            payload_copy = reading.copy()
                            self._publish_mqtt(topic, payload_copy)
                    elif self.mqtt_client and not self.mqtt_connected:
                        # Log warning less frequently
                        current_mono_time = time.monotonic()
                        if current_mono_time - last_log_time > 10: # Log every 10 seconds
                             self.logger.warning("MQTT publishing requested, but client is not connected.")
                             last_log_time = current_mono_time
                    # else: No MQTT client configured, do nothing
                else:
                    yield readings_batch

                # Advance simulated time
                current_sim_time += time_delta
                step_index += 1

                # Sleep if necessary
                if real_time_sleep > 0:
                    loop_end_time = time.monotonic()
                    elapsed_time = loop_end_time - loop_start_time
                    sleep_duration = max(0, real_time_sleep - elapsed_time)
                    if sleep_duration > 0:
                        time.sleep(sleep_duration)

        except KeyboardInterrupt:
            self.logger.info("Simulation stopped by user.")
        except Exception as e:
            self.logger.error(f"Error during device simulation: {e}", exc_info=True)
        finally:
            self.logger.info("Device simulation finished.")

    def run_system_simulation(self,
                              duration_days: float,
                              simulation_speed: float = 1.0,
                              publish_mqtt: bool = False):
        """Run a real-time system metrics simulation (hourly interval)."""
        interval_minutes = 60
        sim_start_time = datetime.now()
        sim_end_time = sim_start_time + timedelta(days=duration_days)
        time_delta = timedelta(minutes=interval_minutes)

        if simulation_speed <= 0 or math.isinf(simulation_speed):
            real_time_sleep = 0
            speed_log = "max"
        else:
            real_time_sleep = (interval_minutes * 60) / simulation_speed
            speed_log = f"{simulation_speed}x"

        self.logger.info(f"Starting system simulation for {duration_days} simulated days...")
        self.logger.info(f"Interval: {interval_minutes} min | Speed: {speed_log} | Sleep: {real_time_sleep:.3f}s | MQTT: {publish_mqtt}")

        current_sim_time = sim_start_time

        try:
            last_log_time = time.monotonic()
            while current_sim_time < sim_end_time:
                loop_start_time = time.monotonic()
                record = self._generate_single_system_metric(pd.Timestamp(current_sim_time))

                if publish_mqtt:
                    if self.mqtt_client and self.mqtt_connected:
                        self._publish_mqtt("digital_twin/system/metrics", record.copy())
                    elif self.mqtt_client and not self.mqtt_connected:
                         current_mono_time = time.monotonic()
                         if current_mono_time - last_log_time > 10:
                              self.logger.warning("MQTT publishing requested, but client is not connected.")
                              last_log_time = current_mono_time
                else:
                    yield record

                current_sim_time += time_delta
                if real_time_sleep > 0:
                    loop_end_time = time.monotonic()
                    elapsed_time = loop_end_time - loop_start_time
                    sleep_duration = max(0, real_time_sleep - elapsed_time)
                    if sleep_duration > 0: time.sleep(sleep_duration)

        except KeyboardInterrupt:
            self.logger.info("System simulation stopped by user.")
        except Exception as e:
            self.logger.error(f"Error during system simulation: {e}", exc_info=True)
        finally:
            self.logger.info("System simulation finished.")

    def run_energy_simulation(self,
                              duration_days: float,
                              simulation_speed: float = 1.0,
                              publish_mqtt: bool = False):
        """Run a real-time energy metrics simulation (15-min interval)."""
        interval_minutes = 15
        sim_start_time = datetime.now()
        sim_end_time = sim_start_time + timedelta(days=duration_days)
        time_delta = timedelta(minutes=interval_minutes)

        if simulation_speed <= 0 or math.isinf(simulation_speed):
            real_time_sleep = 0
            speed_log = "max"
        else:
            real_time_sleep = (interval_minutes * 60) / simulation_speed
            speed_log = f"{simulation_speed}x"

        self.logger.info(f"Starting energy simulation for {duration_days} simulated days...")
        self.logger.info(f"Interval: {interval_minutes} min | Speed: {speed_log} | Sleep: {real_time_sleep:.3f}s | MQTT: {publish_mqtt}")

        current_sim_time = sim_start_time
        # Reset cumulative energy for this simulation run
        self.sim_cumulative_energy = 0

        try:
            last_log_time = time.monotonic()
            while current_sim_time < sim_end_time:
                loop_start_time = time.monotonic()
                # Note: This uses and updates self.sim_cumulative_energy
                record = self._generate_single_energy_metric(pd.Timestamp(current_sim_time))

                if publish_mqtt:
                    if self.mqtt_client and self.mqtt_connected:
                         self._publish_mqtt("digital_twin/energy/metrics", record.copy())
                    elif self.mqtt_client and not self.mqtt_connected:
                         current_mono_time = time.monotonic()
                         if current_mono_time - last_log_time > 10:
                              self.logger.warning("MQTT publishing requested, but client is not connected.")
                              last_log_time = current_mono_time
                else:
                    yield record

                current_sim_time += time_delta
                if real_time_sleep > 0:
                    loop_end_time = time.monotonic()
                    elapsed_time = loop_end_time - loop_start_time
                    sleep_duration = max(0, real_time_sleep - elapsed_time)
                    if sleep_duration > 0: time.sleep(sleep_duration)

        except KeyboardInterrupt:
            self.logger.info("Energy simulation stopped by user.")
        except Exception as e:
            self.logger.error(f"Error during energy simulation: {e}", exc_info=True)
        finally:
            self.logger.info("Energy simulation finished.")


    # --- Summary Method (No changes needed) ---
    # ... (generate_dataset_summary) ...
    def generate_dataset_summary(self, datasets: Dict[str, pd.DataFrame]) -> Dict:
        """Generate summary statistics for the datasets."""
        summary = {
            'generation_timestamp': datetime.now().isoformat(),
            'seed': self.seed,
            'tables': {}
        }
        for table_name, df in datasets.items():
            if df.empty:
                summary['tables'][table_name] = {'record_count': 0, 'columns': [], 'date_range': {}}
                continue

            table_summary = {
                'record_count': len(df),
                'columns': list(df.columns),
                'date_range': {},
                'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
            if 'timestamp' in df.columns:
                 # Ensure timestamp is datetime before min/max
                 if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                      ts_col = pd.to_datetime(df['timestamp'], errors='coerce').dropna()
                 else:
                      ts_col = df['timestamp'].dropna()

                 if not ts_col.empty:
                      table_summary['date_range']['start'] = ts_col.min().isoformat()
                      table_summary['date_range']['end'] = ts_col.max().isoformat()
                 else:
                     table_summary['date_range'] = {'start': None, 'end': None}

            if table_name == 'device_data':
                table_summary.update({
                    'unique_devices': df['device_id'].nunique() if 'device_id' in df.columns else 0,
                    'device_types': df['device_type'].unique().tolist() if 'device_type' in df.columns else [],
                    'status_counts': df['status'].value_counts().to_dict() if 'status' in df.columns else {},
                    'anomaly_count': int((df['status'] == 'anomaly').sum()) if 'status' in df.columns else 0,
                    'maintenance_count': int((df['status'] == 'maintenance').sum()) if 'status' in df.columns else 0
                })
            summary['tables'][table_name] = table_summary
        return summary


# --- UPDATED Main Function ---
def main():
    """Main function to demonstrate data generation."""
    print("🏭 Digital Twin Data Generation System")
    print("=" * 50)

    # --- Batch Generation Example ---
    print("\n--- Batch Generation Mode ---")
    # Uses default db path from CONFIG/app_config.py or fallback
    generator_batch = UnifiedDataGenerator(seed=42)

    print("Generating comprehensive historical dataset...")
    # This also saves to the database automatically now
    datasets = generator_batch.generate_complete_dataset(
        device_count=25,
        days_of_data=45
    )

    print("\n📊 Batch Generation Summary:")
    summary = generator_batch.generate_dataset_summary(datasets)
    print(json.dumps(summary, indent=2, default=str))

    print(f"\n✅ Batch data generation completed!")
    print(f"📁 Database saved to: {generator_batch.db_path}")

    print("\n🔍 Sample Device Data (from DataFrame):")
    if 'device_data' in datasets and not datasets['device_data'].empty:
        sample = datasets['device_data'].sample(min(3, len(datasets['device_data'])))
        print(sample[['timestamp', 'device_id', 'device_type', 'value', 'status', 'health_score']])

    # --- Real-Time Simulation Example (MQTT) ---
    print("\n\n--- Real-Time Simulation Mode (MQTT) ---")
    print("Checking MQTT availability and environment configuration...")

    if PAHO_MQTT_AVAILABLE:
        # Initialize generator, it will automatically try to read from .env
        # We don't need to pass mqtt_broker/port if they are in .env
        generator_mqtt = UnifiedDataGenerator(seed=202)

        if generator_mqtt.mqtt_client:
            print(f"MQTT client created. Attempting connection (check logs for status).")
            # Give a moment for connection attempt
            time.sleep(2)
            if not generator_mqtt.mqtt_connected:
                print("⚠️  Warning: MQTT client not connected after initial attempt. Check broker address/port and logs.")
                print("    Ensure MQTT_BROKER_HOST and MQTT_BROKER_PORT are set in your environment or .env file.")
                print("    Skipping MQTT simulation run.")
            else:
                print("✅ MQTT client connected.")
                print("\n🚀 Starting ~10-second real-time simulation (publishing to MQTT)...")
                generator_mqtt.setup_simulation(device_count=3, start_time=datetime.now())

                # Simulate 1 hour of data (at 1-min intervals) at 360x speed
                # This means 60 steps * (1*60 / 360) = 10 seconds total run time
                duration_days = 1 / 24 # 1 hour
                interval_minutes = 1
                speed = 360 # (1*60 sec) / (10 sec / 60 steps) = 360x

                print(f"Simulating {duration_days*24:.1f} hours of data ({interval_minutes} min interval) at {speed}x speed.")
                print("Check your MQTT client for topics: digital_twin/device/...")

                try:
                    # This is a generator, so we must iterate it to run it
                    for _ in generator_mqtt.run_device_simulation(
                        duration_days=duration_days,
                        interval_minutes=interval_minutes,
                        simulation_speed=speed,
                        publish_mqtt=True
                    ):
                        pass # The work is done by publishing

                    print("Simulation (MQTT) example finished.")
                except Exception as e:
                    print(f"Simulation (MQTT) run failed. Error: {e}")

        else:
            print("❌ MQTT client could not be initialized.")
            print("   Ensure MQTT_BROKER_HOST is set in your environment or .env file.")
            print("   Skipping MQTT simulation example.")
    else:
        print("❌ Skipping MQTT simulation example ('paho-mqtt' library not found).")
        print("   Install it using: pip install paho-mqtt")


if __name__ == "__main__":
    main()