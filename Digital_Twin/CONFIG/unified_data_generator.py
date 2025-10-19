import numpy as np
import pandas as pd
import sqlite3
import json
import logging
import random
import uuid
import os  # <-- Import was missing in original code for sys.path
import sys
import time  # <-- Added for simulation sleep
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import math
import warnings

warnings.filterwarnings('ignore')

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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from CONFIG.app_config import config
except ImportError:
    print("Warning: app_config not found. Using default paths.")
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
    
    # --- Updated __init__ with MQTT support ---
    def __init__(self, 
                 db_path: str = None,
                 seed: int = 42,
                 mqtt_broker: Optional[str] = None,
                 mqtt_port: int = 1883):
        
        self.db_path = db_path or config.database.primary_path
        self.seed = seed
        self.logger = self._setup_logging()
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        # Device configurations
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
        
        # --- MQTT Setup ---
        self.mqtt_client: Optional[mqtt.Client] = None
        self.mqtt_connected = False
        if mqtt_broker and PAHO_MQTT_AVAILABLE:
            self._setup_mqtt(mqtt_broker, mqtt_port)
        elif mqtt_broker and not PAHO_MQTT_AVAILABLE:
            self.logger.error("MQTT broker provided, but 'paho-mqtt' library is not installed.")

    def __del__(self):
        """Clean up resources, especially MQTT connection."""
        if self.mqtt_client and self.mqtt_connected:
            self.logger.info("Disconnecting from MQTT broker.")
            self._disconnect_mqtt()

    def _setup_logging(self):
        """Setup logging for data generator."""
        logger = logging.getLogger('UnifiedDataGenerator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            Path('LOGS').mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler('LOGS/digital_twin_app.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
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
            self.mqtt_client = mqtt.Client(client_id)
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
            
            self.logger.info(f"Connecting to MQTT broker at {broker}:{port}...")
            self.mqtt_client.connect(broker, port, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            self.logger.error(f"Error setting up MQTT client: {e}")
            self.mqtt_client = None

    def _disconnect_mqtt(self):
        """Disconnect the MQTT client."""
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            self.mqtt_connected = False

    def _publish_mqtt(self, topic: str, payload: Dict):
        """Publish a data payload to an MQTT topic."""
        if self.mqtt_client and self.mqtt_connected:
            try:
                # Convert datetime to string for JSON serialization
                json_payload = json.dumps(payload, default=str)
                self.mqtt_client.publish(topic, json_payload)
            except Exception as e:
                self.logger.warning(f"Failed to publish MQTT message to {topic}: {e}")
        
    # --- Batch Generation Methods (Original Logic) ---
    
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
            self.logger.error(f"Device data generation error: {e}")
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
                'installation_date': start_time + timedelta(days=random.randint(0, 30)),
                'manufacturer': random.choice(['Siemens', 'ABB', 'Schneider', 'Rockwell', 'Honeywell']),
                'model': f"Model-{random.randint(1000, 9999)}",
                'firmware_version': f"{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
                'status': 'active',
                'last_maintenance': start_time + timedelta(days=random.randint(0, 60)),
                'config': self.device_types[device_type]
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

    # --- Refactored Single-Reading Generators (for Sim and Batch) ---

    def _generate_single_reading(self, device: Dict, timestamp: pd.Timestamp, step_index: int) -> Dict:
        """Generates a single, stateful data reading for a device at a given timestamp."""
        config = device['config']
        device_id = device['device_id']
        
        # Base value with patterns
        base_value = self._calculate_base_value(timestamp, config)
        
        # Apply long-term temporal trend/drift
        current_drift = self.device_drifts.get(device_id, 0)
        base_value += current_drift * step_index
        
        # Add noise
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
            'timestamp': timestamp,
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
            'quality': random.uniform(0.95, 1.0),
            'signal_strength': random.randint(80, 100),
            'battery_level': random.uniform(0.7, 1.0) if 'wireless' in device.get('connection_type', '') else None
        }
        
        # Add device-specific metrics
        if device['device_type'] == 'temperature_sensor':
            humidity = random.uniform(40, 70)
            record.update({
                'humidity': humidity,
                'heat_index': self._calculate_heat_index(value, humidity)
            })
        elif device['device_type'] == 'vibration_sensor':
            record.update({
                'frequency_hz': random.uniform(10, 100),
                'amplitude_mm': value,
                'rms_velocity': random.uniform(0.5, 5.0)
            })
        elif device['device_type'] == 'power_meter':
            record.update({
                'voltage': random.uniform(220, 240),
                'current': value / random.uniform(220, 240) if value > 0 else 0,
                'power_factor': random.uniform(0.8, 1.0),
                'energy_consumed': value * random.uniform(0.5, 2.0)
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
            'cpu_usage_percent': round(system_load + random.uniform(-10, 10), 2),
            'memory_usage_percent': round(system_load * 0.8 + random.uniform(-5, 15), 2),
            'disk_usage_percent': round(random.uniform(45, 75), 2),
            'network_io_mbps': round(system_load * 0.5 + random.uniform(0, 20), 2),
            'active_connections': int(system_load * 2 + random.randint(0, 50)),
            'response_time_ms': round(max(50, 100 + (system_load - 50) * 2 + random.uniform(-20, 50)), 2),
            'error_rate_percent': round(max(0, (system_load - 80) * 0.1 + random.uniform(0, 0.5)), 3),
            'throughput_rps': round(max(10, 1000 - (system_load - 50) * 5 + random.uniform(-100, 200)), 2),
            'availability_percent': round(max(95, 100 - (system_load - 70) * 0.1 + random.uniform(-1, 1)), 3)
        }
        return record
    
    def _generate_single_energy_metric(self, timestamp: pd.Timestamp) -> Dict:
        """Generates a single energy metric record, updating cumulative state."""
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5
        
        if 6 <= hour <= 22 and not is_weekend:
            base_consumption = random.uniform(800, 1500)
        elif 22 <= hour or hour <= 6:
            base_consumption = random.uniform(200, 600)
        else:  # Weekend
            base_consumption = random.uniform(300, 800)
        
        seasonal_factor = math.sin(2 * math.pi * timestamp.dayofyear / 365.25)
        seasonal_adjustment = seasonal_factor * 300
        
        total_consumption = max(0, base_consumption + seasonal_adjustment)
        # Update stateful cumulative energy (assuming 15min interval)
        self.sim_cumulative_energy += total_consumption * 0.25  # 15 minutes = 0.25 hours
        
        record = {
            'timestamp': timestamp,
            'metric_type': 'energy',
            'power_consumption_kw': round(total_consumption, 2),
            'energy_consumed_kwh': round(self.sim_cumulative_energy, 2),
            'voltage_v': round(random.uniform(220, 240), 1),
            'current_a': round(total_consumption / random.uniform(220, 240), 2),
            'power_factor': round(random.uniform(0.8, 0.95), 3),
            'frequency_hz': round(random.uniform(49.9, 50.1), 2),
            'energy_cost_usd': round(self.sim_cumulative_energy * 0.12, 2),  # $0.12 per kWh
            'carbon_footprint_kg': round(self.sim_cumulative_energy * 0.4, 2),  # 0.4 kg CO2 per kWh
            'efficiency_percent': round(random.uniform(85, 95), 2),
            'renewable_percent': round(random.uniform(20, 40), 2)
        }
        return record

    # --- Helper Methods ---

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
            'power_meter': 'W'
        }
        return unit_mapping.get(device_type, 'units')
    
    def _calculate_heat_index(self, temperature: float, humidity: float) -> float:
        """Calculate heat index from temperature and humidity."""
        if temperature < 26.7: return temperature
        hi = (
            -42.379 + 2.04901523 * temperature + 10.14333127 * humidity -
            0.22475541 * temperature * humidity - 6.83783e-3 * temperature**2 -
            5.481717e-2 * humidity**2 + 1.22874e-3 * temperature**2 * humidity +
            8.5282e-4 * temperature * humidity**2 - 1.99e-6 * temperature**2 * humidity**2
        )
        return round(hi, 2)
    
    def _add_calculated_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated fields to the dataset."""
        df_copy = df.copy()
        
        def calculate_health(row):
            try:
                config = self.device_types[row['device_type']]
                n_min, n_max = config['normal_range']
                c_min, c_max = config['critical_range']
                if row['status'] == 'anomaly': return 0.2
                if n_min <= row['value'] <= n_max: return 0.9 + random.uniform(0, 0.1)
                if row['value'] > n_max:
                    deviation = (row['value'] - n_max) / (c_max - n_max + 1e-6)
                else:
                    deviation = (n_min - row['value']) / (n_min - c_min + 1e-6)
                return max(0.1, 0.9 - deviation)
            except: return 0.5
        
        df_copy['health_score'] = df_copy.apply(calculate_health, axis=1)
        df_copy['efficiency_score'] = (df_copy['health_score'] - random.uniform(0.05, 0.15)).clip(0, 1)
        
        df_copy = df_copy.sort_values(['device_id', 'timestamp'])
        df_copy['operating_hours'] = df_copy.groupby('device_id').cumcount() * (df_copy['timestamp'].diff().dt.total_seconds().mean() / 3600.0)
        df_copy['operating_hours'] = df_copy['operating_hours'].fillna(0)
        
        df_copy['days_since_maintenance'] = (
            df_copy['timestamp'] - df_copy.groupby('device_id')['timestamp'].transform('min')
        ).dt.days
        
        return df_copy
    
    def _add_maintenance_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add maintenance events to the dataset."""
        df = df.copy()
        maintenance_count = 0
        maintenance_indices = random.sample(range(len(df)), int(len(df) * self.maintenance_probability))
        
        for idx in maintenance_indices:
            df.loc[idx, 'status'] = 'maintenance'
            df.loc[idx, 'value'] = 0.0
            df.loc[idx, 'health_score'] = 1.0
            df.loc[idx, 'efficiency_score'] = 0.0
            df.loc[idx, 'quality'] = 1.0
            maintenance_count += 1
        
        self.logger.info(f"Added {maintenance_count} maintenance events to dataset")
        return df
    
    def generate_system_metrics(self, days_of_data: int = 30) -> pd.DataFrame:
        """Generate historical system-level performance metrics."""
        try:
            self.logger.info(f"Generating historical system metrics for {days_of_data} days")
            start_time = datetime.now() - timedelta(days=days_of_data)
            time_points = pd.date_range(start_time, datetime.now(), freq='1H')
            system_data = [self._generate_single_system_metric(ts) for ts in time_points]
            df = pd.DataFrame(system_data)
            self.logger.info(f"Generated {len(df):,} system metric records")
            return df
        except Exception as e:
            self.logger.error(f"System metrics generation error: {e}")
            raise
    
    def generate_energy_data(self, days_of_data: int = 30) -> pd.DataFrame:
        """Generate historical energy consumption and efficiency data."""
        try:
            self.logger.info(f"Generating historical energy data for {days_of_data} days")
            start_time = datetime.now() - timedelta(days=days_of_data)
            time_points = pd.date_range(start_time, datetime.now(), freq='15min')
            
            # Reset cumulative energy for batch generation
            self.sim_cumulative_energy = 0
            energy_data = [self._generate_single_energy_metric(ts) for ts in time_points]
            
            df = pd.DataFrame(energy_data)
            self.logger.info(f"Generated {len(df):,} energy records")
            return df
        except Exception as e:
            self.logger.error(f"Energy data generation error: {e}")
            raise
    
    # --- Database and Summary Methods ---
    
    def save_to_database(self, dataframes: Dict[str, pd.DataFrame]):
        """Save generated data to database."""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                for table_name, df in dataframes.items():
                    self.logger.info(f"Saving {len(df):,} records to table '{table_name}'")
                    df.to_sql(table_name, conn, if_exists='replace', index=False)
            self.logger.info(f"All data saved to database: {self.db_path}")
        except Exception as e:
            self.logger.error(f"Database save error: {e}")
            raise
    
    def generate_complete_dataset(self, 
                                  device_count: int = 20,
                                  days_of_data: int = 30) -> Dict[str, pd.DataFrame]:
        """Generate complete historical dataset with all data types."""
        try:
            self.logger.info(f"Starting complete dataset generation")
            datasets = {}
            datasets['device_data'] = self.generate_device_data(
                device_count=device_count, days_of_data=days_of_data
            )
            datasets['system_metrics'] = self.generate_system_metrics(
                days_of_data=days_of_data
            )
            datasets['energy_data'] = self.generate_energy_data(
                days_of_data=days_of_data
            )
            
            self.save_to_database(datasets)
            summary = self.generate_dataset_summary(datasets)
            self.logger.info(f"Dataset generation completed. Summary: {summary}")
            return datasets
        except Exception as e:
            self.logger.error(f"Complete dataset generation error: {e}")
            raise
    
    def generate_dataset_summary(self, datasets: Dict[str, pd.DataFrame]) -> Dict:
        """Generate summary statistics for the datasets."""
        summary = {
            'generation_timestamp': datetime.now().isoformat(),
            'tables': {}
        }
        for table_name, df in datasets.items():
            table_summary = {
                'record_count': len(df),
                'columns': list(df.columns),
                'date_range': {
                    'start': df['timestamp'].min().isoformat() if 'timestamp' in df.columns and not df.empty else None,
                    'end': df['timestamp'].max().isoformat() if 'timestamp' in df.columns and not df.empty else None
                },
                'data_types': {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}
            }
            if table_name == 'device_data':
                table_summary.update({
                    'unique_devices': df['device_id'].nunique() if 'device_id' in df.columns else 0,
                    'device_types': df['device_type'].unique().tolist() if 'device_type' in df.columns else [],
                    'anomaly_count': len(df[df['status'] == 'anomaly']) if 'status' in df.columns else 0,
                    'maintenance_count': len(df[df['status'] == 'maintenance']) if 'status' in df.columns else 0
                })
            summary['tables'][table_name] = table_summary
        return summary

    # --- New Real-Time Simulation Methods ---

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
                self.device_drifts[device['device_id']] = random.uniform(-0.01, 0.01)
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
        
        :param duration_days: How long the simulation should run in *simulated time*.
        :param interval_minutes: The gap between data points in *simulated time*.
        :param simulation_speed: Multiplier for real-time. 1.0 = real-time, 60.0 = 1 min of sim time passes in 1 sec.
        :param publish_mqtt: If True, publish to MQTT. If False, yield data batches.
        """
        if not self.devices:
            self.logger.warning("No devices set up. Call setup_simulation() first. Defaulting to 10 devices.")
            self.setup_simulation(device_count=10)
            
        start_time = datetime.now()
        end_time = start_time + timedelta(days=duration_days)
        time_delta = timedelta(minutes=interval_minutes)
        real_time_sleep = (interval_minutes * 60) / simulation_speed

        self.logger.info(f"Starting device simulation for {duration_days} simulated days...")
        self.logger.info(f"Interval: {interval_minutes} min | Speed: {simulation_speed}x | Sleep: {real_time_sleep:.2f}s")
        
        current_time = start_time
        step_index = 0
        
        try:
            while current_time < end_time:
                readings_batch = []
                for device in self.devices:
                    reading = self._generate_single_reading(device, current_time, step_index)
                    readings_batch.append(reading)
                
                if publish_mqtt:
                    if not self.mqtt_connected:
                        self.logger.warning("MQTT publishing requested, but client is not connected.")
                    for reading in readings_batch:
                        topic = f"digital_twin/device/{reading['device_id']}"
                        self._publish_mqtt(topic, reading)
                else:
                    yield readings_batch
                
                # Advance time
                current_time += time_delta
                step_index += 1
                time.sleep(real_time_sleep)
                
        except KeyboardInterrupt:
            self.logger.info("Simulation stopped by user.")
        finally:
            self.logger.info("Device simulation finished.")

    def run_system_simulation(self, 
                              duration_days: float, 
                              simulation_speed: float = 1.0, 
                              publish_mqtt: bool = False):
        """Run a real-time system metrics simulation (hourly interval)."""
        interval_minutes = 60
        start_time = datetime.now()
        end_time = start_time + timedelta(days=duration_days)
        time_delta = timedelta(minutes=interval_minutes)
        real_time_sleep = (interval_minutes * 60) / simulation_speed

        self.logger.info(f"Starting system simulation for {duration_days} simulated days...")
        current_time = start_time
        
        try:
            while current_time < end_time:
                record = self._generate_single_system_metric(current_time)
                
                if publish_mqtt:
                    self._publish_mqtt("digital_twin/system/metrics", record)
                else:
                    yield record
                    
                current_time += time_delta
                time.sleep(real_time_sleep)
        except KeyboardInterrupt:
            self.logger.info("Simulation stopped by user.")
        finally:
            self.logger.info("System simulation finished.")

    def run_energy_simulation(self, 
                              duration_days: float, 
                              simulation_speed: float = 1.0, 
                              publish_mqtt: bool = False):
        """Run a real-time energy metrics simulation (15-min interval)."""
        interval_minutes = 15
        start_time = datetime.now()
        end_time = start_time + timedelta(days=duration_days)
        time_delta = timedelta(minutes=interval_minutes)
        real_time_sleep = (interval_minutes * 60) / simulation_speed

        self.logger.info(f"Starting energy simulation for {duration_days} simulated days...")
        current_time = start_time
        
        try:
            while current_time < end_time:
                # Note: This uses and updates self.sim_cumulative_energy
                record = self._generate_single_energy_metric(current_time)
                
                if publish_mqtt:
                    self._publish_mqtt("digital_twin/energy/metrics", record)
                else:
                    yield record
                    
                current_time += time_delta
                time.sleep(real_time_sleep)
        except KeyboardInterrupt:
            self.logger.info("Simulation stopped by user.")
        finally:
            self.logger.info("Energy simulation finished.")


def main():
    """Main function to generate and save data."""
    print("🏭 Digital Twin Data Generation System")
    print("=" * 50)
    
    # --- Batch Generation Example ---
    print("\n--- Batch Generation Mode ---")
    generator_batch = UnifiedDataGenerator()
    
    print("Generating comprehensive historical dataset...")
    datasets = generator_batch.generate_complete_dataset(
        device_count=25,
        days_of_data=45
    )
    
    print("\n📊 Batch Generation Summary:")
    for table_name, df in datasets.items():
        print(f"  {table_name}: {len(df):,} records")
    
    print(f"\n✅ Batch data generation completed!")
    print(f"📁 Database saved to: {generator_batch.db_path}")
    
    print("\n🔍 Sample Device Data:")
    if 'device_data' in datasets and not datasets['device_data'].empty:
        sample = datasets['device_data'].sample(min(3, len(datasets['device_data'])))
        print(sample[['timestamp', 'device_id', 'device_type', 'value', 'status', 'health_score']])

    # --- Real-Time Simulation Example (Yield) ---
    print("\n\n--- Real-Time Simulation Mode (Yield) ---")
    
    # Note: No mqtt_broker passed, so it will use yield
    generator_sim = UnifiedDataGenerator(seed=101) 
    generator_sim.setup_simulation(device_count=5, start_time=datetime.now())
    
    print("\n🚀 Starting 5-step real-time simulation (using 'yield')...")
    sim_steps = 5
    # Run as fast as possible for the example
    simulator = generator_sim.run_device_simulation(
        duration_days=1,  # Simulated duration doesn't matter much here
        interval_minutes=5, 
        simulation_speed=float('inf') 
    )
    
    for i, batch in enumerate(simulator):
        if i >= sim_steps:
            break
        print(f"  [SIM STEP {i+1}] Generated batch of {len(batch)} readings for timestamp {batch[0]['timestamp']}")
        # print(f"    - Device {batch[0]['device_id']}: {batch[0]['value']:.2f}") # Uncomment for more detail
    
    print("Simulation (yield) example finished.")

    # --- Real-Time Simulation Example (MQTT) ---
    print("\n\n--- Real-Time Simulation Mode (MQTT) ---")
    print("This example is commented out. To run it:")
    print("1. Ensure you have 'paho-mqtt' installed: pip install paho-mqtt")
    print("2. Ensure you have an MQTT broker running (e.g., on localhost:1883)")
    print("3. Uncomment the code block below in main()")
    
    """
    if PAHO_MQTT_AVAILABLE:
        print("\n🚀 Starting 10-second real-time simulation (publishing to MQTT)...")
        # Pass broker details to enable MQTT
        generator_mqtt = UnifiedDataGenerator(mqtt_broker='localhost', mqtt_port=1883, seed=202)
        generator_mqtt.setup_simulation(device_count=3, start_time=datetime.now())

        # Simulate 1 minute of data over 10 seconds (6x speed)
        # duration_days = 1 / (24 * 60) # 1 minute
        # We'll run for 10 real-time seconds instead
        
        sim_start_time = time.time()
        
        # We'll run the device and system sims concurrently (in a real app, use threading)
        # For this example, we'll just show the device sim
        
        # Run simulation with 1-second interval, 6x speed (so 6 sim-seconds pass per 1 real-second)
        # This is just an example, a 1-minute interval is more realistic
        
        # Let's simulate 1 hour of data (at 1-min intervals) at 360x speed
        # This means 60 steps * (1*60 / 360) = 10 seconds total run time
        
        duration_days = 1 / 24 # 1 hour
        interval_minutes = 1
        speed = 360 # (1*60) / 0.166s sleep = 360x
        
        print(f"Simulating {duration_days*24} hours of data ({interval_minutes} min interval) at {speed}x speed.")
        print("This will take ~10 seconds. Check your MQTT client for topics: digital_twin/device/...")
        
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
            print(f"Could not run MQTT simulation. Is broker running at localhost:1883? Error: {e}")
    else:
        print("\nSkipping MQTT simulation example ('paho-mqtt' not found).")
    """

if __name__ == "__main__":
    main()