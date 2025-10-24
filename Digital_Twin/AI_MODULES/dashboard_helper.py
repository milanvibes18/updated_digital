# Updated File: Digital_Twin/AI_MODULES/dashboard_helper.py

#!/usr/bin/env python3
"""
Dashboard Helper Module for Digital Twin System v2.0
Provides data processing, aggregation, and formatting for dashboard visualization.
Refactored to operate primarily on Pandas DataFrames, removing direct DB access.
"""

import numpy as np
import pandas as pd
import logging
import json
import os # Added for path handling
import sys # Added for path handling
# import sqlite3 # Removed direct SQLite dependency
import pickle
from datetime import datetime, timedelta, timezone # Added timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Statistical and ML imports
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Time series analysis
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# --- Add project root to path ---
# Use relative pathing to be more robust
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import necessary SQLAlchemy components if direct DB access is needed (less preferred)
# from sqlalchemy.orm import Session # Example if passing session
# from Digital_Twin.WEB_APPLICATION.enhanced_flask_app_v2 import DeviceData # Example if querying models


class DashboardHelper:
    """
    Comprehensive dashboard helper for Digital Twin applications. v2.0
    Provides data processing, aggregation, chart preparation based on input DataFrames.
    """

    def __init__(self, cache_path: str = "ANALYTICS/analysis_cache/"):
        self.cache_path = Path(cache_path)
        self.logger = self._setup_logging()

        # Create cache directory
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # Aggregation settings
        self.aggregation_config = {
            'temperature': {'unit': '°C', 'normal_range': (15, 35), 'precision': 1},
            'pressure': {'unit': 'hPa', 'normal_range': (950, 1050), 'precision': 1},
            'vibration': {'unit': 'mm/s', 'normal_range': (0, 0.5), 'precision': 3},
            'humidity': {'unit': '%RH', 'normal_range': (30, 70), 'precision': 1},
            'power': {'unit': 'W', 'normal_range': (800, 2000), 'precision': 1},
            'efficiency': {'unit': '%', 'normal_range': (70, 100), 'precision': 1}
            # Add device_type specific configs if needed (e.g., 'temperature_sensor')
        }

        # Chart color schemes
        self.color_schemes = {
            'primary': ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'],
            'status': {
                'normal': '#27ae60',
                'warning': '#f39c12',
                'critical': '#e74c3c',
                'offline': '#95a5a6'
            },
            'health': {
                'excellent': '#27ae60',
                'good': '#2ecc71',
                'fair': '#f39c12',
                'poor': '#e67e22',
                'critical': '#e74c3c'
            }
        }

        self._previous_metric_values = {}


    def _setup_logging(self):
        """Setup logging for dashboard helper."""
        logger = logging.getLogger('DashboardHelper')
        logger.setLevel(logging.INFO)

        # Prevent duplicate handlers if logger already exists
        if not logger.handlers:
            log_dir = Path("LOGS")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / 'digital_twin_dashboard.log'
            
            # File Handler
            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            # Console Handler (optional, for visibility during development/debugging)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    def get_dashboard_overview(self, historical_df: pd.DataFrame, alerts_summary: Dict) -> Dict:
        """
        Get comprehensive dashboard overview data based on provided historical data.

        Args:
            historical_df: DataFrame containing historical device data (e.g., last 7 days).
                           Expected columns: device_id, timestamp, value, status, health_score,
                           efficiency_score, device_type, unit, location. Optional: operating_hours,
                           days_since_maintenance. Timestamp should be datetime objects or ISO strings.
            alerts_summary: Dictionary containing current alert summary.

        Returns:
            Dictionary containing dashboard overview metrics.
        """
        try:
            self.logger.info("Generating dashboard overview from DataFrame")

            if historical_df is None or historical_df.empty:
                self.logger.warning("Input historical_df is empty or None. Returning default overview.")
                return self._get_default_overview()

            # Make a copy to avoid modifying the original DataFrame
            historical_df = historical_df.copy()

            # Ensure timestamp is datetime
            if 'timestamp' in historical_df.columns:
                try:
                    # Attempt conversion, handling potential errors for mixed types
                    historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'], errors='coerce')
                    # Drop rows where timestamp conversion failed
                    historical_df.dropna(subset=['timestamp'], inplace=True)
                    if historical_df.empty:
                        self.logger.warning("DataFrame became empty after timestamp conversion/dropna.")
                        return self._get_default_overview()
                except Exception as e:
                    self.logger.error(f"Error converting timestamp column: {e}. Returning default overview.", exc_info=True)
                    return self._get_default_overview()
            else:
                 self.logger.warning("Timestamp column not found in historical_df.")
                 # Decide if you want to proceed without time-based analysis or return default
                 # return self._get_default_overview() # Option 1: Return default
                 # Option 2: Proceed, but time-based calculations will fail gracefully


            # Get the latest snapshot for each device from the provided DataFrame
            # Ensure we handle potential duplicates by taking the absolute last entry per device
            if 'timestamp' in historical_df.columns:
                 latest_df = historical_df.sort_values('timestamp').groupby('device_id').last().reset_index()
            else:
                 # If no timestamp, just take the last occurrence per device_id (less reliable)
                 latest_df = historical_df.groupby('device_id').last().reset_index()


            # Calculate key metrics
            overview = {
                'timestamp': datetime.now(timezone.utc).isoformat(), # Use timezone aware
                'system_metrics': self._calculate_system_metrics(latest_df, historical_df),
                'device_metrics': self._calculate_device_metrics(latest_df),
                'health_metrics': self._calculate_health_metrics(historical_df), # Uses historical
                'performance_metrics': self._calculate_performance_metrics(historical_df), # Uses historical
                'status_distribution': self._calculate_status_distribution(latest_df), # Uses latest
                'trend_analysis': self._calculate_trend_analysis(historical_df), # Uses historical
                'asset_comparisons': self._calculate_asset_comparisons(latest_df), # Uses latest
                'alerts_summary': alerts_summary, # Pass alerts through
                'energy_metrics': self._calculate_energy_metrics(historical_df) # Uses historical
            }

            return overview

        except Exception as e:
            self.logger.error(f"Dashboard overview generation error from DataFrame: {e}", exc_info=True)
            return self._get_default_overview()


    def _generate_sample_device_data(self) -> List[Dict]:
        """Generate sample device data for demonstration (now richer for trends)."""
        try:
            devices = []
            device_types = ['temperature_sensor', 'pressure_sensor', 'vibration_sensor',
                            'humidity_sensor', 'power_meter']
            locations = ['Factory Floor A', 'Factory Floor B', 'Warehouse', 'Quality Lab']

            for i in range(20):  # 20 unique devices
                device_type_base = np.random.choice(device_types)
                location = np.random.choice(locations)

                # Base values for this device
                base_health = np.random.uniform(0.5, 0.9)
                base_efficiency = np.random.uniform(0.6, 0.9)

                for j in range(10): # 10 data points each over the last 7 days
                    timestamp = datetime.now(timezone.utc) - timedelta(days=np.random.uniform(0, 7),
                                                           hours=np.random.uniform(0, 24))

                    device_type = device_type_base

                    # Generate realistic values based on device type
                    config_key = device_type.split('_')[0] # e.g., 'temperature' from 'temperature_sensor'
                    config = self.aggregation_config.get(config_key, {})
                    normal_range = config.get('normal_range', (0, 100))
                    critical_range_low = normal_range[0] * 0.7 # Example range expansion
                    critical_range_high = normal_range[1] * 1.3 # Example range expansion
                    precision = config.get('precision', 1)

                    if 'temperature' in device_type:
                        value = np.random.uniform(normal_range[0] - 5, normal_range[1] + 10)
                        unit = '°C'
                    elif 'pressure' in device_type:
                        value = np.random.uniform(normal_range[0] - 50, normal_range[1] + 50)
                        unit = 'hPa'
                    elif 'vibration' in device_type:
                        value = np.random.uniform(normal_range[0], normal_range[1] + 0.3)
                        unit = 'mm/s'
                    elif 'humidity' in device_type:
                        value = np.random.uniform(normal_range[0] - 10, normal_range[1] + 10)
                        unit = '%RH'
                    else:  # power_meter
                        value = np.random.uniform(normal_range[0] - 200, normal_range[1] + 1000)
                        unit = 'W'

                    # Clamp to a wider critical range to avoid extreme unrealistic values
                    value = max(critical_range_low, min(critical_range_high, value))

                    # Determine status based on value ranges relative to NORMAL range
                    if normal_range[0] <= value <= normal_range[1]:
                        status = 'normal'
                    elif value < normal_range[0] * 0.8 or value > normal_range[1] * 1.2:
                        status = 'critical'
                    else:
                        status = 'warning'

                    # Simulate trend
                    health_score = max(0.2, min(1.0, base_health + (j * 0.01))) # Slowly improving
                    efficiency_score = max(0.2, min(1.0, base_efficiency - (j * 0.005))) # Slowly declining

                    devices.append({
                        'device_id': f'DEVICE_{i+1:03d}',
                        'device_name': f'{device_type.replace("_", " ").title()} {i+1:03d}',
                        'device_type': device_type,
                        'value': round(value, precision),
                        'unit': unit,
                        'status': status,
                        'health_score': health_score,
                        'efficiency_score': efficiency_score,
                        'location': location,
                        'timestamp': timestamp.isoformat(),
                        'operating_hours': np.random.uniform(100, 8000),
                        'days_since_maintenance': np.random.randint(1, 90)
                    })

            return devices

        except Exception as e:
            self.logger.error(f"Sample data generation error: {e}")
            return []

    def _calculate_system_metrics(self, latest_df: pd.DataFrame, historical_df: pd.DataFrame) -> Dict:
        """Calculate system-level metrics from DataFrames."""
        try:
            total_devices = latest_df['device_id'].nunique() if not latest_df.empty else 0

            if total_devices == 0:
                return {'total_devices': 0, 'active_devices': 0, 'offline_devices': 0, 'uptime': 100.0, 'data_points_collected': 0, 'response_time_ms': 0.0, 'last_update': datetime.now(timezone.utc).isoformat()}

            active_devices = 0
            offline_devices = 0
            if 'status' in latest_df.columns:
                status_counts = latest_df['status'].value_counts()
                active_devices = status_counts.get('normal', 0) + status_counts.get('warning', 0) + status_counts.get('critical', 0)
                offline_devices = status_counts.get('offline', 0)
            else:
                 # If no status column, assume all devices seen are active
                 active_devices = total_devices


            uptime = (active_devices / total_devices) * 100 if total_devices > 0 else 100.0

            # Calculate average response time (simulated - replace with real data if available)
            response_time = np.random.uniform(50, 200)

            return {
                'total_devices': total_devices,
                'active_devices': active_devices,
                'offline_devices': offline_devices,
                'uptime': round(uptime, 2),
                'response_time_ms': round(response_time, 1),
                'data_points_collected': len(historical_df), # Total points in the period
                'last_update': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            self.logger.error(f"System metrics calculation error: {e}", exc_info=True)
            return {'total_devices': 0, 'active_devices': 0, 'offline_devices': 0, 'uptime': 0.0, 'data_points_collected': 0, 'response_time_ms': 0.0, 'last_update': datetime.now(timezone.utc).isoformat()}


    def _calculate_device_metrics(self, latest_df: pd.DataFrame) -> Dict:
        """Calculate device-specific metrics based on latest snapshot DataFrame."""
        try:
            if latest_df.empty or 'device_type' not in latest_df.columns:
                return {}

            device_metrics = {}

            for device_type in latest_df['device_type'].unique():
                if pd.isna(device_type): continue # Skip if device_type is NaN

                type_df = latest_df[latest_df['device_type'] == device_type]

                avg_val = 0
                min_val = 0
                max_val = 0
                unit = 'units'
                status_dist = {}

                if 'value' in type_df.columns:
                     value_col = type_df['value'].dropna()
                     if not value_col.empty:
                         avg_val = value_col.mean()
                         min_val = value_col.min()
                         max_val = value_col.max()

                if 'unit' in type_df.columns:
                     unit_col = type_df['unit'].dropna()
                     if not unit_col.empty:
                         unit = unit_col.iloc[0]


                if 'status' in type_df.columns:
                     status_col = type_df['status'].dropna()
                     if not status_col.empty:
                          status_dist = status_col.value_counts().to_dict()


                device_metrics[device_type] = {
                    'count': len(type_df),
                    'average_value': round(avg_val, 2),
                    'min_value': round(min_val, 2),
                    'max_value': round(max_val, 2),
                    'unit': unit,
                    'status_distribution': status_dist
                }

            return device_metrics

        except Exception as e:
            self.logger.error(f"Device metrics calculation error: {e}", exc_info=True)
            return {}

    def _calculate_health_metrics(self, historical_df: pd.DataFrame) -> Dict:
        """Calculate health-related metrics from historical DataFrame."""
        try:
            if historical_df.empty or 'health_score' not in historical_df.columns:
                self.logger.warning("Health score column missing or DataFrame empty for health metrics.")
                return self._get_default_health_metrics()

            # Get latest health scores for current snapshot
            latest_df = historical_df.sort_values('timestamp').groupby('device_id').last().reset_index()
            health_scores = latest_df['health_score'].dropna()

            if health_scores.empty:
                self.logger.warning("No valid health scores found in the latest device snapshot.")
                return self._get_default_health_metrics()

            # Convert to percentages
            health_scores_pct = health_scores * 100

            # Calculate health distribution from latest snapshot
            health_distribution = {
                'excellent': int((health_scores_pct >= 90).sum()),
                'good': int(((health_scores_pct >= 75) & (health_scores_pct < 90)).sum()),
                'fair': int(((health_scores_pct >= 60) & (health_scores_pct < 75)).sum()),
                'poor': int((health_scores_pct < 60).sum()),
                # 'critical' is usually below 'poor', adjust if needed based on config
                # 'critical': int((health_scores_pct < 30).sum()) # Example critical threshold
            }
            devices_needing_attention = health_distribution['poor'] # + health_distribution.get('critical', 0)


            # Calculate trend from historical data
            health_trend_details = self._calculate_detailed_trend(historical_df, 'health_score')

            return {
                'average_health': round(health_scores_pct.mean(), 1) if not np.isnan(health_scores_pct.mean()) else 0.0,
                'median_health': round(health_scores_pct.median(), 1) if not np.isnan(health_scores_pct.median()) else 0.0,
                'min_health': round(health_scores_pct.min(), 1) if not np.isnan(health_scores_pct.min()) else 0.0,
                'max_health': round(health_scores_pct.max(), 1) if not np.isnan(health_scores_pct.max()) else 0.0,
                'std_health': round(health_scores_pct.std(), 1) if not np.isnan(health_scores_pct.std()) else 0.0,
                'health_distribution': health_distribution,
                'health_trend': health_trend_details.get('trend', 'stable'), # Simple trend for overview card
                'devices_needing_attention': devices_needing_attention
            }

        except Exception as e:
            self.logger.error(f"Health metrics calculation error: {e}", exc_info=True)
            return self._get_default_health_metrics()


    def _get_default_health_metrics(self) -> Dict:
        """Get default health metrics when no data is available."""
        return {
            'average_health': 0.0, # Default to 0 if no data
            'median_health': 0.0,
            'min_health': 0.0,
            'max_health': 0.0,
            'std_health': 0.0,
            'health_distribution': {
                'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0, 'critical': 0
            },
            'health_trend': 'stable',
            'devices_needing_attention': 0
        }

    def _calculate_performance_metrics(self, historical_df: pd.DataFrame) -> Dict:
        """
        Calculate performance-related metrics from historical DataFrame.
        Handles missing optional columns 'operating_hours' and 'days_since_maintenance'.
        """
        try:
            if historical_df.empty:
                return self._get_default_performance_metrics()

            # Get latest snapshot for current metrics
            latest_df = historical_df.sort_values('timestamp').groupby('device_id').last().reset_index()

            # --- Efficiency ---
            average_efficiency = 0.0
            min_efficiency = 0.0
            max_efficiency = 0.0
            if 'efficiency_score' in latest_df.columns:
                efficiency_scores = latest_df['efficiency_score'].dropna() * 100
                if not efficiency_scores.empty:
                    average_efficiency = round(efficiency_scores.mean(), 1) if not np.isnan(efficiency_scores.mean()) else 0.0
                    min_efficiency = round(efficiency_scores.min(), 1) if not np.isnan(efficiency_scores.min()) else 0.0
                    max_efficiency = round(efficiency_scores.max(), 1) if not np.isnan(efficiency_scores.max()) else 0.0
            else:
                 self.logger.warning("Efficiency score column not found for performance metrics.")


            # Calculate trend from historical data
            efficiency_trend_details = self._calculate_detailed_trend(historical_df, 'efficiency_score')

            # --- Operating Hours (Optional) ---
            operating_hours_stats = {}
            if 'operating_hours' in latest_df.columns:
                operating_hours = latest_df['operating_hours'].dropna()
                if not operating_hours.empty:
                    operating_hours_stats = {
                        'total_operating_hours': round(operating_hours.sum(), 0) if not np.isnan(operating_hours.sum()) else 0.0,
                        'average_operating_hours': round(operating_hours.mean(), 0) if not np.isnan(operating_hours.mean()) else 0.0,
                        'max_operating_hours': round(operating_hours.max(), 0) if not np.isnan(operating_hours.max()) else 0.0,
                    }
            else:
                self.logger.info("Optional column 'operating_hours' not found in DataFrame.")

            # --- Maintenance Statistics (Optional) ---
            maintenance_stats = {}
            if 'days_since_maintenance' in latest_df.columns:
                maintenance_days = latest_df['days_since_maintenance'].dropna()
                if not maintenance_days.empty:
                    maintenance_stats = {
                        'average_days_since_maintenance': round(maintenance_days.mean(), 0) if not np.isnan(maintenance_days.mean()) else 0.0,
                        'devices_due_maintenance': int((maintenance_days > 60).sum()),
                        'devices_overdue_maintenance': int((maintenance_days > 90).sum())
                    }
            else:
                 self.logger.info("Optional column 'days_since_maintenance' not found in DataFrame.")


            return {
                'average_efficiency': average_efficiency,
                'min_efficiency': min_efficiency,
                'max_efficiency': max_efficiency,
                'efficiency_trend': efficiency_trend_details.get('trend', 'stable'),
                **operating_hours_stats, # Merges dict if exists, otherwise empty
                **maintenance_stats    # Merges dict if exists, otherwise empty
            }

        except Exception as e:
            self.logger.error(f"Performance metrics calculation error: {e}", exc_info=True)
            return self._get_default_performance_metrics()


    def _get_default_performance_metrics(self) -> Dict:
        """Get default performance metrics when no data is available."""
        # Removed operating_hours and maintenance stats from default if they are optional
        return {
            'average_efficiency': 0.0,
            'min_efficiency': 0.0,
            'max_efficiency': 0.0,
            'efficiency_trend': 'stable',
            # 'total_operating_hours': 0,
            # 'average_operating_hours': 0,
            # 'max_operating_hours': 0,
            # 'average_days_since_maintenance': 0,
            # 'devices_due_maintenance': 0,
            # 'devices_overdue_maintenance': 0
        }


    def _calculate_status_distribution(self, latest_df: pd.DataFrame) -> Dict:
        """Calculate device status distribution from latest snapshot DataFrame."""
        try:
            if latest_df.empty or 'status' not in latest_df.columns:
                return {'normal': 0, 'warning': 0, 'critical': 0, 'offline': 0}

            # Ensure 'status' column is treated as string and handle NaNs
            status_counts = latest_df['status'].astype(str).fillna('unknown').value_counts().to_dict()

            # Ensure all standard status types are present
            status_distribution = {
                'normal': status_counts.get('normal', 0),
                'warning': status_counts.get('warning', 0),
                'critical': status_counts.get('critical', 0),
                'offline': status_counts.get('offline', 0),
                'unknown': status_counts.get('unknown', 0) # Include unknown count if any NaNs existed
            }
             # Remove unknown if count is 0
            if status_distribution['unknown'] == 0:
                 del status_distribution['unknown']


            return status_distribution

        except Exception as e:
            self.logger.error(f"Status distribution calculation error: {e}", exc_info=True)
            return {'normal': 0, 'warning': 0, 'critical': 0, 'offline': 0}

    def _calculate_detailed_trend(self, df: pd.DataFrame, metric: str) -> Dict:
        """
        Calculate detailed trend analysis for a specific metric using daily resampling.
        Handles potential missing 'unit' column.
        """
        default_trend = {'trend': 'stable', 'slope': 0.0, 'change_24h': 0.0, 'change_7d': 0.0, 'unit': ''} # Default unit to empty string

        try:
            if df is None or df.empty or metric not in df.columns or df[metric].isnull().all():
                self.logger.warning(f"No valid data for detailed trend on '{metric}'.")
                return default_trend

            df_copy = df.dropna(subset=[metric, 'timestamp']).copy()
            # Ensure timestamp is datetime type
            if not pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
                 df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], errors='coerce')
                 df_copy.dropna(subset=['timestamp'], inplace=True)

            if df_copy.empty:
                self.logger.warning(f"No valid data after dropping NaNs for trend on '{metric}'.")
                return default_trend

            # Ensure index is unique before resampling if needed
            if df_copy['timestamp'].duplicated().any():
                self.logger.debug(f"Handling duplicate timestamps for trend on '{metric}'. Keeping last.")
                df_copy = df_copy.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')

            if df_copy.empty: # Check again after potential drop_duplicates
                 self.logger.warning(f"No data after handling duplicates for trend on '{metric}'.")
                 return default_trend

            df_copy = df_copy.set_index('timestamp')

            # Determine appropriate resampling frequency (e.g., daily)
            resample_freq = 'D'
            try:
                # Use mean for numeric, maybe mode or first for categorical if needed later
                 daily_avg = df_copy[metric].resample(resample_freq).mean().dropna()
            except TypeError: # Handle non-numeric data gracefully if metric isn't number
                 self.logger.warning(f"Metric '{metric}' is not numeric, cannot calculate trend.")
                 return default_trend


            if len(daily_avg) < 2:
                self.logger.warning(f"Not enough resampled data points ({len(daily_avg)}) for detailed trend on '{metric}'.")
                return {'trend': 'stable', 'slope': 0.0, 'change_24h': 0.0, 'change_7d': 0.0, 'unit': ''}


            # Get latest, 24h ago, and 7d ago (or first) values
            latest_val = daily_avg.iloc[-1]
            prev_val_24h = daily_avg.iloc[-2] if len(daily_avg) >= 2 else latest_val
            seven_days_ago_ts = daily_avg.index[-1] - pd.Timedelta(days=7)
            # Find the value closest to 7 days ago, or use the first if none exist before that
            prev_val_7d_series = daily_avg[daily_avg.index <= seven_days_ago_ts]
            prev_val_7d = prev_val_7d_series.iloc[-1] if not prev_val_7d_series.empty else daily_avg.iloc[0]


            # Calculate changes
            change_24h = latest_val - prev_val_24h
            change_7d = latest_val - prev_val_7d

            # Calculate linear regression slope
            slope = 0.0
            r_value_sq = 0.0
            if len(daily_avg) >= 2:
                x = np.arange(len(daily_avg))
                try:
                    slope_val, _, r_val, _, _ = stats.linregress(x, daily_avg.values)
                    slope = slope_val
                    r_value_sq = r_val**2
                except ValueError as e:
                     self.logger.warning(f"Linregress failed for {metric}: {e}")


            # Determine trend string based on normalized slope
            data_range = daily_avg.max() - daily_avg.min() if len(daily_avg) > 1 else 1.0
            normalized_slope = slope / (data_range + 1e-9) # Avoid division by zero

            trend_str = 'stable'
            if normalized_slope > 0.05:
                trend_str = 'improving'
            elif normalized_slope < -0.05:
                trend_str = 'declining'

            # Unit determination
            unit = '%'
            multiplier = 100
            if metric == 'value': # Special case for 'value' trend
                unit = '' # Default if 'unit' column is missing
                if 'unit' in df_copy.columns:
                     unit_series = df_copy['unit'].dropna()
                     if not unit_series.empty:
                         unit = unit_series.iloc[0]
                multiplier = 1 # Don't multiply raw values by 100
            elif metric not in ['health_score', 'efficiency_score']:
                 multiplier = 1
                 unit = ''
                 if 'unit' in df_copy.columns:
                      unit_series = df_copy['unit'].dropna()
                      if not unit_series.empty:
                           unit = unit_series.iloc[0]


            return {
                'trend': trend_str,
                'slope': round(slope * multiplier, 3) if not np.isnan(slope) else 0.0,
                'change_24h': round(change_24h * multiplier, 2) if not np.isnan(change_24h) else 0.0,
                'change_7d': round(change_7d * multiplier, 2) if not np.isnan(change_7d) else 0.0,
                'unit': unit
            }

        except Exception as e:
            self.logger.error(f"Detailed trend calculation error for {metric}: {e}", exc_info=True)
            return default_trend


    def _calculate_trend_analysis(self, df: pd.DataFrame) -> Dict:
        """Calculate detailed trend analysis for key metrics from DataFrame."""
        try:
            trends = {}

            if df is None or df.empty:
                return {
                    'health': self._calculate_detailed_trend(pd.DataFrame(), 'health_score'),
                    'efficiency': self._calculate_detailed_trend(pd.DataFrame(), 'efficiency_score')
                }

            # Health trend
            trends['health'] = self._calculate_detailed_trend(df, 'health_score')

            # Efficiency trend
            trends['efficiency'] = self._calculate_detailed_trend(df, 'efficiency_score')

            # Key metric trend (e.g., Temperature, Pressure) - analyze 'value' for common types
            if 'device_type' in df.columns:
                common_types = df['device_type'].dropna().value_counts().head(3).index.tolist()
                for dev_type in common_types:
                    type_df = df[df['device_type'] == dev_type]
                    if not type_df.empty:
                        # Use a simplified key like 'temperature_trend' instead of 'temperature_sensor_trend'
                        metric_key = dev_type.split('_')[0] + "_trend"
                        trends[metric_key] = self._calculate_detailed_trend(type_df, 'value')
            else:
                 self.logger.warning("Device type column not found, cannot calculate metric-specific trends.")


            return trends

        except Exception as e:
            self.logger.error(f"Trend analysis calculation error: {e}", exc_info=True)
            return {'health': {'trend': 'stable'}, 'efficiency': {'trend': 'stable'}}


    def _calculate_asset_comparisons(self, latest_df: pd.DataFrame) -> Dict:
        """Calculate multi-asset comparisons from latest snapshot DataFrame."""
        try:
            comparisons = {}
            if latest_df is None or latest_df.empty: return comparisons

            # Top 5 / Bottom 5 by Health
            if 'health_score' in latest_df.columns and not latest_df['health_score'].isnull().all():
                health_cols = ['device_id', 'device_name', 'health_score']
                health_df_valid = latest_df.dropna(subset=['health_score'])

                if not health_df_valid.empty:
                    # Make sure health_score is numeric before nlargest/nsmallest
                    health_df_valid['health_score'] = pd.to_numeric(health_df_valid['health_score'], errors='coerce')
                    health_df_valid = health_df_valid.dropna(subset=['health_score'])

                    if not health_df_valid.empty:
                        top_health = health_df_valid.nlargest(5, 'health_score')[health_cols].copy()
                        top_health['health_score'] = (top_health['health_score'] * 100).round(1)

                        bottom_health = health_df_valid.nsmallest(5, 'health_score')[health_cols].copy()
                        bottom_health['health_score'] = (bottom_health['health_score'] * 100).round(1)

                        comparisons['top_5_health'] = top_health.to_dict('records')
                        comparisons['bottom_5_health'] = bottom_health.to_dict('records')

            # Top 5 / Bottom 5 by Efficiency
            if 'efficiency_score' in latest_df.columns and not latest_df['efficiency_score'].isnull().all():
                eff_cols = ['device_id', 'device_name', 'efficiency_score']
                eff_df_valid = latest_df.dropna(subset=['efficiency_score'])

                if not eff_df_valid.empty:
                    eff_df_valid['efficiency_score'] = pd.to_numeric(eff_df_valid['efficiency_score'], errors='coerce')
                    eff_df_valid = eff_df_valid.dropna(subset=['efficiency_score'])

                    if not eff_df_valid.empty:
                        top_eff = eff_df_valid.nlargest(5, 'efficiency_score')[eff_cols].copy()
                        top_eff['efficiency_score'] = (top_eff['efficiency_score'] * 100).round(1)

                        bottom_eff = eff_df_valid.nsmallest(5, 'efficiency_score')[eff_cols].copy()
                        bottom_eff['efficiency_score'] = (bottom_eff['efficiency_score'] * 100).round(1)

                        comparisons['top_5_efficiency'] = top_eff.to_dict('records')
                        comparisons['bottom_5_efficiency'] = bottom_eff.to_dict('records')

            # Aggregates by Location
            if 'location' in latest_df.columns and 'health_score' in latest_df.columns:
                # Ensure health_score is numeric before grouping
                loc_df_valid = latest_df.dropna(subset=['health_score'])
                loc_df_valid['health_score'] = pd.to_numeric(loc_df_valid['health_score'], errors='coerce')
                loc_df_valid = loc_df_valid.dropna(subset=['health_score'])

                if not loc_df_valid.empty:
                    health_by_loc = (loc_df_valid.groupby('location')['health_score'].mean() * 100).round(1).dropna()
                    if not health_by_loc.empty:
                        comparisons['health_by_location'] = health_by_loc.to_dict()

            # Aggregates by Device Type
            if 'device_type' in latest_df.columns and 'efficiency_score' in latest_df.columns:
                type_df_valid = latest_df.dropna(subset=['efficiency_score'])
                type_df_valid['efficiency_score'] = pd.to_numeric(type_df_valid['efficiency_score'], errors='coerce')
                type_df_valid = type_df_valid.dropna(subset=['efficiency_score'])

                if not type_df_valid.empty:
                    eff_by_type = (type_df_valid.groupby('device_type')['efficiency_score'].mean() * 100).round(1).dropna()
                    if not eff_by_type.empty:
                        comparisons['efficiency_by_type'] = eff_by_type.to_dict()

            return comparisons

        except Exception as e:
            self.logger.error(f"Asset comparison calculation error: {e}", exc_info=True)
            return {}

    def _calculate_energy_metrics(self, historical_df: pd.DataFrame) -> Dict:
        """Calculate energy consumption metrics based on historical DataFrame."""
        try:
            if historical_df is None or historical_df.empty or 'device_type' not in historical_df.columns:
                 self.logger.warning("Cannot calculate energy metrics: DataFrame empty or missing 'device_type'.")
                 return self._get_default_energy_metrics()

            # Filter for power meter data
            power_df = historical_df[historical_df['device_type'] == 'power_meter'].copy()

            if power_df.empty:
                self.logger.info("No 'power_meter' device type found for energy metrics.")
                return self._get_default_energy_metrics()

            # Ensure timestamp is datetime and value is numeric
            if 'timestamp' not in power_df.columns:
                 self.logger.warning("Timestamp column missing in power meter data.")
                 return self._get_default_energy_metrics()

            power_df['timestamp'] = pd.to_datetime(power_df['timestamp'], errors='coerce')
            power_df.dropna(subset=['timestamp'], inplace=True)
            if power_df.empty:
                 return self._get_default_energy_metrics()


            if 'value' not in power_df.columns:
                self.logger.warning("Value column missing in power meter data.")
                return self._get_default_energy_metrics()

            power_df['value'] = pd.to_numeric(power_df['value'], errors='coerce')
            power_df.dropna(subset=['value'], inplace=True)
            if power_df.empty:
                 return self._get_default_energy_metrics()


            # Get latest total power by summing the last reading of each power meter
            latest_power_df = power_df.sort_values('timestamp').groupby('device_id').last()
            current_power_w = latest_power_df['value'].sum()
            current_power_kw = current_power_w / 1000.0

            # Calculate daily energy estimate from historical data
            daily_energy_kwh = 0.0 # Default
            if len(power_df) > 1:
                power_df_indexed = power_df.set_index('timestamp')
                # Sum power across all meters at each timestamp
                total_power_series_w = power_df_indexed.groupby(level=0)['value'].sum() # Group by index (timestamp)
                # Resample to hourly average power in kW
                hourly_power_kw = total_power_series_w.resample('H').mean() / 1000.0
                hourly_power_kw.dropna(inplace=True) # Drop hours with no data

                if not hourly_power_kw.empty:
                    last_ts = hourly_power_kw.index.max()
                    start_ts = last_ts - pd.Timedelta(hours=24)
                    last_24h_power = hourly_power_kw[hourly_power_kw.index > start_ts]

                    if not last_24h_power.empty:
                        # Energy = Sum of (Average Power in hour * 1 hour)
                        daily_energy_kwh = last_24h_power.sum()
                    else:
                        daily_energy_kwh = current_power_kw * 24 # Fallback estimate
                else:
                     daily_energy_kwh = current_power_kw * 24 # Fallback estimate
            else:
                 daily_energy_kwh = current_power_kw * 24 # Fallback estimate


            monthly_cost_usd = daily_energy_kwh * 30 * 0.12 # Example cost: $0.12 per kWh

            # Use overall average efficiency score if available
            avg_efficiency = 0.0
            if 'efficiency_score' in historical_df.columns:
                 avg_efficiency_series = historical_df['efficiency_score'].dropna()
                 if not avg_efficiency_series.empty:
                      avg_efficiency = avg_efficiency_series.mean() * 100


            return {
                'current_power_kw': round(current_power_kw, 1) if not np.isnan(current_power_kw) else 0.0,
                'daily_energy_kwh': round(daily_energy_kwh, 1) if not np.isnan(daily_energy_kwh) else 0.0,
                'monthly_cost_usd': round(monthly_cost_usd, 0) if not np.isnan(monthly_cost_usd) else 0.0,
                'energy_efficiency': round(avg_efficiency, 1) if not np.isnan(avg_efficiency) else 0.0,
                'carbon_footprint_kg': round(daily_energy_kwh * 0.4, 1) if not np.isnan(daily_energy_kwh) else 0.0 # Example factor
            }

        except Exception as e:
            self.logger.error(f"Energy metrics calculation error: {e}", exc_info=True)
            return self._get_default_energy_metrics()


    def _get_default_energy_metrics(self) -> Dict:
        """Get default energy metrics when no data is available."""
        return {
            'current_power_kw': 0.0,
            'daily_energy_kwh': 0.0,
            'monthly_cost_usd': 0.0,
            'energy_efficiency': 0.0,
            'carbon_footprint_kg': 0.0
        }

    def _get_default_overview(self) -> Dict:
        """Get default dashboard overview when no data is available."""
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'system_metrics': self._calculate_system_metrics(pd.DataFrame(), pd.DataFrame()),
            'device_metrics': {},
            'health_metrics': self._get_default_health_metrics(),
            'performance_metrics': self._get_default_performance_metrics(),
            'status_distribution': {'normal': 0, 'warning': 0, 'critical': 0, 'offline': 0},
            'trend_analysis': {'health': {'trend': 'stable'}, 'efficiency': {'trend': 'stable'}},
            'asset_comparisons': {},
            'alerts_summary': {'total_alerts': 0, 'critical_alerts': 0, 'warning_alerts': 0, 'info_alerts': 0},
            'energy_metrics': self._get_default_energy_metrics()
        }

    def _prepare_pie_chart_data_from_counts(self, counts: Dict[str, int]) -> Dict:
        """Prepares data suitable for a pie chart from a counts dictionary."""
        # Ensure counts is a dict
        if not isinstance(counts, dict):
            self.logger.warning("Invalid input for pie chart counts, expected dict.")
            return {'labels': [], 'datasets': []}

        labels = list(counts.keys())
        values = list(counts.values())
        # Make sure values are numbers
        values = [v if isinstance(v, (int, float)) else 0 for v in values]

        # Use status colors if labels match, otherwise default
        colors = [self.color_schemes['status'].get(label, self.color_schemes['primary'][i % len(self.color_schemes['primary'])])
                  for i, label in enumerate(labels)]

        return {
            'labels': labels,
            'datasets': [{
                'data': values,
                'backgroundColor': colors,
                'hoverBackgroundColor': colors # Or slightly darker versions
            }]
        }

    def prepare_chart_data(self, chart_type: str, time_range: str = '24h', device_id: Optional[str] = None, data_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Prepares data for various chart types based on input DataFrame or generates sample data.
        NOTE: In production, this should ideally always receive a DataFrame (data_df).
              The internal generator is primarily for demo/fallback.
        """
        self.logger.info(f"Preparing chart data: type={chart_type}, range={time_range}, device={device_id}")

        if data_df is None or data_df.empty:
            self.logger.warning("No DataFrame provided or empty for prepare_chart_data. Cannot generate chart.")
            # Avoid using internal generator, return error/empty structure
            return {'error': 'No data available to generate chart', 'labels': [], 'datasets': []}


        # Make a copy to avoid modifying original
        data_df = data_df.copy()

        # Ensure timestamp is datetime and handle errors
        if 'timestamp' in data_df.columns:
             try:
                 data_df['timestamp'] = pd.to_datetime(data_df['timestamp'], errors='coerce')
                 data_df.dropna(subset=['timestamp'], inplace=True)
                 if data_df.empty:
                      self.logger.warning("DataFrame empty after timestamp conversion/dropna in prepare_chart_data.")
                      return {'error': 'No valid time data available', 'labels': [], 'datasets': []}
             except Exception as e:
                  self.logger.error(f"Error handling timestamp in prepare_chart_data: {e}", exc_info=True)
                  return {'error': 'Invalid time data format', 'labels': [], 'datasets': []}
        else:
             self.logger.warning("Timestamp column missing for chart preparation.")
             # Charts requiring time axis might fail or produce unexpected results


        try:
            if chart_type == 'line':
                # Pass only the relevant time range if possible (more efficient if done earlier)
                # cutoff = datetime.now(timezone.utc) - timedelta(days={'1h': 1/24, '4h': 4/24, '24h': 1, '7d': 7, '30d': 30}.get(time_range, 1))
                # data_df = data_df[data_df['timestamp'] >= cutoff]
                return self._prepare_line_chart_data(data_df, device_id)
            elif chart_type == 'pie': # Example: Use status distribution
                 # Pie chart represents the LATEST status, not historical distribution
                 if data_df.empty: return {'labels': [], 'datasets': []}
                 latest_df = data_df.sort_values('timestamp').groupby('device_id').last().reset_index()
                 status_counts = self._calculate_status_distribution(latest_df) # Use helper
                 return self._prepare_pie_chart_data_from_counts(status_counts)
            elif chart_type == 'bar': # Example: Average health per device type
                if data_df.empty: return {'labels': [], 'datasets': []}
                # Use latest values for bar chart comparison
                latest_df = data_df.sort_values('timestamp').groupby('device_id').last().reset_index()
                return self._prepare_bar_chart_data_grouped(latest_df, group_by_col='device_type', value_col='health_score')

            else:
                self.logger.warning(f"Unsupported chart type requested: {chart_type}")
                return {'error': f'Unsupported chart type: {chart_type}'}
        except Exception as e:
            self.logger.error(f"Chart preparation error for {chart_type}: {e}", exc_info=True)
            return {'error': f'Failed to prepare chart: {str(e)}'}


    def _prepare_line_chart_data(self, df: pd.DataFrame, device_id: Optional[str] = None) -> Dict:
        """Prepares data for a line chart (e.g., value over time)."""
        target_col = 'value' # Could be configurable, e.g., health_score

        if device_id:
            df_filtered = df[df['device_id'] == device_id].copy()
            label_prefix = device_id
        else:
            # Aggregate if no specific device is selected
            if 'timestamp' not in df.columns:
                 self.logger.warning("Timestamp column missing for aggregation.")
                 return {'labels': [], 'datasets': []}
            # Resample to hourly mean if data spans multiple days, else use raw points
            time_span_days = (df['timestamp'].max() - df['timestamp'].min()).days if not df.empty else 0
            if time_span_days > 2:
                 df_filtered = df.set_index('timestamp')[target_col].resample('H').mean().reset_index()
                 df_filtered.dropna(subset=[target_col], inplace=True) # Drop hours with no avg
                 label_prefix = f"Average {target_col}"
            else:
                 # For shorter periods, maybe plot individual devices or just use raw points averaged?
                 # Let's average raw points per timestamp for simplicity
                 df_filtered = df.groupby('timestamp')[target_col].mean().reset_index()
                 label_prefix = f"Average {target_col}"


        if df_filtered.empty or target_col not in df_filtered.columns:
            return {'labels': [], 'datasets': []}

        df_sorted = df_filtered.sort_values('timestamp')

        # Format labels clearly
        time_format = '%H:%M' if time_span_days <= 1 else '%m-%d %H:%M'
        labels = df_sorted['timestamp'].dt.strftime(time_format).tolist()
        values = df_sorted[target_col].round(2).tolist() # Round values

        # Determine Unit - tricky if aggregated, get from original df if possible
        unit = ''
        if device_id and 'unit' in df.columns:
             unit_val = df.loc[df['device_id'] == device_id, 'unit'].iloc[0]
             unit = str(unit_val) if pd.notna(unit_val) else ''
        elif 'unit' in df.columns: # Get common unit if aggregating
             common_unit = df['unit'].mode()
             unit = common_unit[0] if not common_unit.empty else ''


        label_text = f"{label_prefix} ({unit})" if unit else label_prefix

        return {
            'labels': labels,
            'datasets': [{
                'label': label_text,
                'data': values,
                'borderColor': self.color_schemes['primary'][0],
                'tension': 0.1,
                'pointRadius': 2 # Smaller points
            }]
        }

    def _prepare_bar_chart_data_grouped(self, df: pd.DataFrame, group_by_col: str, value_col: str) -> Dict:
        """Prepares data for a grouped bar chart."""
        if df.empty or group_by_col not in df.columns or value_col not in df.columns:
            return {'labels': [], 'datasets': []}

        # Ensure value col is numeric, group by category, calculate mean
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        grouped_data = df.groupby(group_by_col)[value_col].mean().dropna()

        if grouped_data.empty:
            return {'labels': [], 'datasets': []}

        # Apply multiplier if it's a score
        multiplier = 100 if '_score' in value_col else 1
        values = (grouped_data * multiplier).round(1).tolist()
        labels = grouped_data.index.tolist()

        return {
            'labels': labels,
            'datasets': [{
                'label': f'Average {value_col.replace("_", " ").title()}',
                'data': values,
                'backgroundColor': self.color_schemes['primary'][:len(labels)] # Use multiple colors
            }]
        }


    def get_device_summary(self, device_id: str, historical_df: pd.DataFrame, alerts: List[Dict], recommendations: List[Dict]) -> Dict:
        """
        Get comprehensive summary for a specific device based on provided data.
        Ensures timestamps are ISO strings.
        """
        try:
            if historical_df is None or historical_df.empty:
                return {'error': f'No historical data provided for device {device_id}'}

            # Make a copy
            historical_df = historical_df.copy()

            # Ensure timestamp is datetime and handle errors
            if 'timestamp' in historical_df.columns:
                historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'], errors='coerce')
                historical_df.dropna(subset=['timestamp'], inplace=True)
                if historical_df.empty:
                     return {'error': f'No valid time data for device {device_id}'}
            else:
                 return {'error': f'Timestamp column missing for device {device_id}'}


            # Get the latest data point for current metrics
            # Use last() which is slightly safer if index isn't perfectly sequential
            device_latest_series = historical_df.sort_values('timestamp').iloc[-1]
            device = device_latest_series.to_dict()

            # Safely get metrics, providing defaults
            current_metrics = {
                'value': round(device.get('value', 0.0), self.aggregation_config.get(device.get('device_type', '').split('_')[0], {}).get('precision', 2)),
                'unit': device.get('unit', ''),
                'health_score': round(device.get('health_score', 0.0) * 100, 1),
                'efficiency_score': round(device.get('efficiency_score', 0.0) * 100, 1),
                'operating_hours': round(device.get('operating_hours', 0.0)),
                'days_since_maintenance': int(device.get('days_since_maintenance', 0))
            }


            summary = {
                'device_id': device_id,
                'device_name': device.get('device_name', 'Unknown Name'),
                'device_type': device.get('device_type', 'unknown_type'),
                'location': device.get('location', 'Unknown Location'),
                'status': device.get('status', 'unknown'),
                'current_metrics': current_metrics,
                'trends': self._calculate_device_trends(historical_df), # Use full history for trends
                'alerts': alerts, # Pass through provided alerts
                'recommendations': recommendations, # Pass through provided recommendations
                # Ensure last_updated is ISO string
                'last_updated': device_latest_series['timestamp'].isoformat() if pd.notna(device_latest_series['timestamp']) else datetime.now(timezone.utc).isoformat()

            }

            return summary

        except IndexError: # Handles case where iloc[-1] fails on empty df after filtering
             return {'error': f'No data found for device {device_id} after processing'}
        except Exception as e:
            self.logger.error(f"Device summary error for {device_id} using DataFrame: {e}", exc_info=True)
            return {'error': f'Failed to generate summary for {device_id}: {str(e)}'}


    def _generate_sample_historical_data(self, device_id: str, hours: int) -> List[Dict]:
        """Generate sample historical data for demonstration."""
        try:
            data = []
            now = datetime.now(timezone.utc)
            base_health = np.random.uniform(0.6, 0.95)
            health_trend = np.random.uniform(-0.001, 0.0005) # Slight trend

            for i in range(hours):
                timestamp = now - timedelta(hours=hours-1-i)

                # Generate realistic pattern (e.g., sine wave + noise)
                # Adjust based on likely device type if known, otherwise generic
                is_temp = 'temp' in device_id.lower()
                base_value = 25 + (10 * np.sin(i * 2 * np.pi / 24)) if is_temp else 50 + 10*np.random.rand()
                noise = np.random.normal(0, 1)

                value = base_value + noise
                unit = '°C' if is_temp else 'units'

                current_health = max(0.2, min(1.0, base_health + health_trend * i))
                current_efficiency = max(0.3, min(1.0, current_health - np.random.uniform(0.05, 0.1)))

                status = 'normal'
                if current_health < 0.6: status = 'critical'
                elif current_health < 0.75: status = 'warning'


                data.append({
                    'device_id': device_id,
                    'timestamp': timestamp.isoformat(),
                    'value': round(value, 2),
                    'unit': unit, # Include unit
                    'health_score': current_health,
                    'efficiency_score': current_efficiency,
                    'status': status,
                    'operating_hours': 1000 + i, # Example
                    'days_since_maintenance': 30 + i // 24 # Example
                })

            return data

        except Exception as e:
            self.logger.error(f"Sample historical data generation error for {device_id}: {e}")
            return []


    def _calculate_device_trends(self, historical_df: pd.DataFrame) -> Dict:
        """Calculate trends for a single device's metrics from DataFrame."""
        try:
            if historical_df is None or historical_df.empty:
                return {'health': 'stable', 'efficiency': 'stable', 'value': 'stable'}

            trends = {}

            # Calculate trends only if columns exist and have enough data
            for metric in ['health_score', 'efficiency_score', 'value']:
                trend_key = metric.replace('_score', '') # Simplified key
                if metric in historical_df.columns and len(historical_df[metric].dropna()) >= 3:
                    trend_details = self._calculate_detailed_trend(historical_df, metric)
                    trends[trend_key] = trend_details.get('trend', 'stable')
                else:
                    trends[trend_key] = 'unknown' # Indicate insufficient data


            return trends

        except Exception as e:
            self.logger.error(f"Device trends calculation error: {e}", exc_info=True)
            # Return default trends on error
            return {'health': 'unknown', 'efficiency': 'unknown', 'value': 'unknown'}


    def export_dashboard_data(self, overview_data: Dict, latest_devices_df: pd.DataFrame, historical_df: Optional[pd.DataFrame] = None, format_type: str = 'json') -> Dict:
        """
        Export dashboard data (overview, latest devices, optional history) in specified format.
        Ensures DataFrames are handled correctly, including timestamps.
        """
        try:
            self.logger.info(f"Exporting dashboard data in {format_type} format from provided data")

            # Prepare DataFrames for export (handle potential NaNs and timestamps)
            latest_devices_export_df = latest_devices_df.copy() if latest_devices_df is not None else pd.DataFrame()
            historical_export_df = historical_df.copy() if historical_df is not None else pd.DataFrame()

            # Convert timestamps to ISO strings for export consistency
            for df in [latest_devices_export_df, historical_export_df]:
                if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                     df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ') # Use ISO 8601 format

            # Prepare main export dictionary
            export_data = {
                'export_timestamp': datetime.now(timezone.utc).isoformat(),
                'overview': overview_data if overview_data else {},
                'latest_devices': latest_devices_export_df.to_dict('records') if not latest_devices_export_df.empty else []
            }

            if not historical_export_df.empty:
                export_data['historical_data'] = historical_export_df.to_dict('records')


            timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            base_filename = f"dashboard_export_{timestamp_str}"
            export_path_base = self.cache_path / base_filename # Use cache path as base

            if format_type == 'json':
                export_path = export_path_base.with_suffix('.json')
                with open(export_path, 'w', encoding='utf-8') as f: # Specify encoding
                    # Use default=str to handle potential non-serializable types like Timestamps if conversion failed
                    json.dump(export_data, f, indent=2, default=str)
                return {'export_path': str(export_path), 'format': 'json'}

            elif format_type == 'csv':
                # Export latest devices to CSV
                export_path_latest = export_path_base.with_name(f"{base_filename}_latest_devices.csv")
                latest_devices_export_df.to_csv(export_path_latest, index=False, date_format='%Y-%m-%dT%H:%M:%SZ')
                export_paths = {'latest_devices': str(export_path_latest)}

                # Optionally export historical data to a separate CSV
                if not historical_export_df.empty:
                    hist_export_path = export_path_base.with_name(f"{base_filename}_historical.csv")
                    historical_export_df.to_csv(hist_export_path, index=False, date_format='%Y-%m-%dT%H:%M:%SZ')
                    export_paths['historical_data'] = str(hist_export_path)

                return {'export_paths': export_paths, 'format': 'csv'}

            elif format_type == 'excel':
                export_path = export_path_base.with_suffix('.xlsx')
                try:
                    with pd.ExcelWriter(export_path, engine='xlsxwriter',
                                        datetime_format='yyyy-mm-dd hh:mm:ss',
                                        date_format='yyyy-mm-dd') as writer:
                        if not latest_devices_export_df.empty:
                           latest_devices_export_df.to_excel(writer, sheet_name='Latest Devices', index=False)
                        if not historical_export_df.empty:
                           historical_export_df.to_excel(writer, sheet_name='Historical Data', index=False)
                    return {'export_path': str(export_path), 'format': 'excel'}
                except ImportError:
                    self.logger.error("Excel export failed: 'xlsxwriter' library not installed. pip install xlsxwriter")
                    return {'error': "Excel export requires the 'xlsxwriter' library."}
                except Exception as ex_err:
                     self.logger.error(f"Excel export failed during writing: {ex_err}", exc_info=True)
                     return {'error': f"Failed to write Excel file: {str(ex_err)}"}


            else:
                self.logger.warning(f"Unsupported export format requested: {format_type}")
                return {'error': f'Unsupported export format: {format_type}'}

        except Exception as e:
            self.logger.error(f"Dashboard data export error: {e}", exc_info=True)
            return {'error': f'An unexpected error occurred during export: {str(e)}'}