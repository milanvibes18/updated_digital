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
from datetime import datetime, timedelta
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

        # Data caches (consider moving caching layer higher up if helper becomes stateless)
        # self.data_cache = {}
        # self.cache_timestamps = {}
        # self.cache_ttl = 300  # 5 minutes

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

        # Real-time data buffer (Consider if this state belongs here or higher up)
        # self.realtime_buffer = defaultdict(lambda: deque(maxlen=100))

        # Store previous values for delta calculations (again, consider state management)
        self._previous_metric_values = {}


    def _setup_logging(self):
        """Setup logging for dashboard helper."""
        logger = logging.getLogger('DashboardHelper')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            Path("LOGS").mkdir(exist_ok=True)
            handler = logging.FileHandler('LOGS/digital_twin_dashboard.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def get_dashboard_overview(self, historical_df: pd.DataFrame, alerts_summary: Dict) -> Dict:
        """
        Get comprehensive dashboard overview data based on provided historical data.

        Args:
            historical_df: DataFrame containing historical device data (e.g., last 7 days).
                           Expected columns: device_id, timestamp, value, status, health_score,
                           efficiency_score, device_type, unit, location, operating_hours,
                           days_since_maintenance. Timestamp should be datetime objects.
            alerts_summary: Dictionary containing current alert summary.

        Returns:
            Dictionary containing dashboard overview metrics.
        """
        try:
            self.logger.info("Generating dashboard overview from DataFrame")

            if historical_df.empty:
                self.logger.warning("Input historical_df is empty. Returning default overview.")
                return self._get_default_overview()

            # Ensure timestamp is datetime
            if 'timestamp' in historical_df.columns and not pd.api.types.is_datetime64_any_dtype(historical_df['timestamp']):
                historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])


            # Get the latest snapshot for each device from the provided DataFrame
            # Ensure we handle potential duplicates by taking the absolute last entry per device
            latest_df = historical_df.sort_values('timestamp').groupby('device_id').last().reset_index()

            # Calculate key metrics
            overview = {
                'timestamp': datetime.now().isoformat(),
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

    # REMOVED: _fetch_device_data - Data should be fetched via SQLAlchemy in Flask app

    # Kept for fallback/demo purposes if needed, but should not be used in production flow
    def _generate_sample_device_data(self) -> List[Dict]:
        """Generate sample device data for demonstration (now richer for trends)."""
        # (Implementation remains the same as provided)
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
                    timestamp = datetime.now() - timedelta(days=np.random.uniform(0, 7),
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

    # --- Calculation methods remain largely the same, operating on DataFrames ---

    def _calculate_system_metrics(self, latest_df: pd.DataFrame, historical_df: pd.DataFrame) -> Dict:
        """Calculate system-level metrics from DataFrames."""
        # (Implementation remains the same - already uses DataFrames)
        try:
            total_devices = latest_df['device_id'].nunique() if not latest_df.empty else 0

            if total_devices == 0:
                return {'total_devices': 0, 'active_devices': 0, 'uptime': 100.0, 'data_points_collected': 0}

            # Assuming 'status' column exists
            active_devices = latest_df[latest_df['status'] == 'normal'].shape[0] if 'status' in latest_df.columns else total_devices

            # Calculate system uptime based on device status
            uptime = (active_devices / total_devices) * 100 if total_devices > 0 else 100.0

            # Calculate average response time (simulated)
            response_time = np.random.uniform(50, 200)

            return {
                'total_devices': total_devices,
                'active_devices': active_devices,
                'offline_devices': total_devices - active_devices, # Simple assumption
                'uptime': round(uptime, 2),
                'response_time_ms': round(response_time, 1),
                'data_points_collected': len(historical_df), # Total points in the period
                'last_update': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"System metrics calculation error: {e}", exc_info=True)
            return {'total_devices': 0, 'active_devices': 0, 'uptime': 0.0, 'data_points_collected': 0}

    def _calculate_device_metrics(self, latest_df: pd.DataFrame) -> Dict:
        """Calculate device-specific metrics based on latest snapshot DataFrame."""
        # (Implementation remains the same - already uses DataFrames)
        try:
            if latest_df.empty or 'device_type' not in latest_df.columns:
                return {}

            device_metrics = {}

            for device_type in latest_df['device_type'].unique():
                type_df = latest_df[latest_df['device_type'] == device_type]

                avg_val = type_df['value'].mean() if 'value' in type_df.columns and not type_df['value'].isnull().all() else 0
                min_val = type_df['value'].min() if 'value' in type_df.columns and not type_df['value'].isnull().all() else 0
                max_val = type_df['value'].max() if 'value' in type_df.columns and not type_df['value'].isnull().all() else 0
                unit = type_df['unit'].iloc[0] if 'unit' in type_df.columns and not type_df['unit'].isnull().all() else 'units'
                status_dist = type_df['status'].value_counts().to_dict() if 'status' in type_df.columns else {}


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
        # (Implementation remains the same - already uses DataFrames)
        try:
            if historical_df.empty or 'health_score' not in historical_df.columns:
                return self._get_default_health_metrics()

            # Get latest health scores for current snapshot
            latest_df = historical_df.sort_values('timestamp').groupby('device_id').last().reset_index()
            health_scores = latest_df['health_score'].dropna()

            if health_scores.empty:
                return self._get_default_health_metrics()

            # Convert to percentages
            health_scores_pct = health_scores * 100

            # Calculate health distribution from latest snapshot
            health_distribution = {
                'excellent': len(health_scores[health_scores >= 0.9]),
                'good': len(health_scores[(health_scores >= 0.7) & (health_scores < 0.9)]),
                'fair': len(health_scores[(health_scores >= 0.5) & (health_scores < 0.7)]),
                'poor': len(health_scores[(health_scores >= 0.3) & (health_scores < 0.5)]),
                'critical': len(health_scores[health_scores < 0.3])
            }

            # Calculate trend from historical data
            health_trend_details = self._calculate_detailed_trend(historical_df, 'health_score')

            return {
                'average_health': round(health_scores_pct.mean(), 1),
                'median_health': round(health_scores_pct.median(), 1),
                'min_health': round(health_scores_pct.min(), 1),
                'max_health': round(health_scores_pct.max(), 1),
                'std_health': round(health_scores_pct.std(), 1) if not np.isnan(health_scores_pct.std()) else 0.0, # Handle NaN std
                'health_distribution': health_distribution,
                'health_trend': health_trend_details.get('trend', 'stable'), # Simple trend for overview card
                'devices_needing_attention': health_distribution['poor'] + health_distribution['critical']
            }

        except Exception as e:
            self.logger.error(f"Health metrics calculation error: {e}", exc_info=True)
            return self._get_default_health_metrics()


    def _get_default_health_metrics(self) -> Dict:
        """Get default health metrics when no data is available."""
        # (Implementation remains the same)
        return {
            'average_health': 85.0,
            'median_health': 87.0,
            'min_health': 65.0,
            'max_health': 98.0,
            'std_health': 12.5,
            'health_distribution': {
                'excellent': 5, 'good': 8, 'fair': 4, 'poor': 2, 'critical': 1
            },
            'health_trend': 'stable',
            'devices_needing_attention': 3
        }

    def _calculate_performance_metrics(self, historical_df: pd.DataFrame) -> Dict:
        """Calculate performance-related metrics from historical DataFrame."""
        # (Implementation remains the same - already uses DataFrames)
        try:
            if historical_df.empty:
                return self._get_default_performance_metrics()

            # Get latest snapshot for current metrics
            latest_df = historical_df.sort_values('timestamp').groupby('device_id').last().reset_index()

            efficiency_scores = pd.Series([], dtype=float) # Initialize with dtype
            if 'efficiency_score' in latest_df.columns:
                 efficiency_scores = latest_df['efficiency_score'].dropna() * 100


            # Calculate operating hours statistics
            operating_hours_stats = {}
            if 'operating_hours' in latest_df.columns:
                operating_hours = latest_df['operating_hours'].dropna()
                if not operating_hours.empty:
                    operating_hours_stats = {
                        'total_operating_hours': round(operating_hours.sum(), 0),
                        'average_operating_hours': round(operating_hours.mean(), 0),
                        'max_operating_hours': round(operating_hours.max(), 0)
                    }

            # Calculate maintenance statistics
            maintenance_stats = {}
            if 'days_since_maintenance' in latest_df.columns:
                maintenance_days = latest_df['days_since_maintenance'].dropna()
                if not maintenance_days.empty:
                    maintenance_stats = {
                        'average_days_since_maintenance': round(maintenance_days.mean(), 0),
                        'devices_due_maintenance': len(maintenance_days[maintenance_days > 60]),
                        'devices_overdue_maintenance': len(maintenance_days[maintenance_days > 90])
                    }

            # Calculate trend from historical data
            efficiency_trend_details = self._calculate_detailed_trend(historical_df, 'efficiency_score')

            return {
                'average_efficiency': round(efficiency_scores.mean(), 1) if not efficiency_scores.empty else 85.0,
                'min_efficiency': round(efficiency_scores.min(), 1) if not efficiency_scores.empty else 70.0,
                'max_efficiency': round(efficiency_scores.max(), 1) if not efficiency_scores.empty else 98.0,
                'efficiency_trend': efficiency_trend_details.get('trend', 'stable'),
                **operating_hours_stats,
                **maintenance_stats
            }

        except Exception as e:
            self.logger.error(f"Performance metrics calculation error: {e}", exc_info=True)
            return self._get_default_performance_metrics()


    def _get_default_performance_metrics(self) -> Dict:
        """Get default performance metrics when no data is available."""
        # (Implementation remains the same)
        return {
            'average_efficiency': 85.5,
            'min_efficiency': 68.0,
            'max_efficiency': 97.5,
            'efficiency_trend': 'stable',
            'total_operating_hours': 45000,
            'average_operating_hours': 2250,
            'max_operating_hours': 8760,
            'average_days_since_maintenance': 32,
            'devices_due_maintenance': 3,
            'devices_overdue_maintenance': 1
        }


    def _calculate_status_distribution(self, latest_df: pd.DataFrame) -> Dict:
        """Calculate device status distribution from latest snapshot DataFrame."""
        # (Implementation remains the same - already uses DataFrames)
        try:
            if latest_df.empty or 'status' not in latest_df.columns:
                return {'normal': 0, 'warning': 0, 'critical': 0, 'offline': 0}

            status_counts = latest_df['status'].value_counts().to_dict()

            # Ensure all status types are present
            status_distribution = {
                'normal': status_counts.get('normal', 0),
                'warning': status_counts.get('warning', 0),
                'critical': status_counts.get('critical', 0),
                'offline': status_counts.get('offline', 0)
            }

            return status_distribution

        except Exception as e:
            self.logger.error(f"Status distribution calculation error: {e}", exc_info=True)
            return {'normal': 0, 'warning': 0, 'critical': 0, 'offline': 0}

    def _calculate_detailed_trend(self, df: pd.DataFrame, metric: str) -> Dict:
        """
        Calculate detailed trend analysis for a specific metric using daily resampling.
        (Implementation remains the same - already uses DataFrames)
        """
        # (Implementation remains the same)
        default_trend = {'trend': 'stable', 'slope': 0.0, 'change_24h': 0.0, 'change_7d': 0.0, 'unit': '%'}

        try:
            if df.empty or metric not in df.columns or df[metric].isnull().all():
                self.logger.warning(f"No valid data for detailed trend on '{metric}'.")
                return default_trend

            df_copy = df.dropna(subset=[metric, 'timestamp']).copy()
            # Ensure timestamp is datetime type
            if not pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
                 df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])

            if df_copy.empty:
                self.logger.warning(f"No valid data after dropping NaNs for trend on '{metric}'.")
                return default_trend

            # Ensure index is unique before resampling if needed
            if df_copy['timestamp'].duplicated().any():
                self.logger.debug(f"Handling duplicate timestamps for trend on '{metric}'. Keeping last.")
                df_copy = df_copy.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')

            df_copy = df_copy.set_index('timestamp')

            # Determine appropriate resampling frequency (e.g., daily)
            # Adjust freq based on data density if needed
            resample_freq = 'D'
            daily_avg = df_copy[metric].resample(resample_freq).mean().dropna()


            if len(daily_avg) < 2:
                # Need at least two points for trend calculation
                self.logger.warning(f"Not enough resampled data points ({len(daily_avg)}) for detailed trend on '{metric}'.")
                # Return basic trend if possible
                if len(daily_avg) == 1:
                     return {'trend': 'stable', 'slope': 0.0, 'change_24h': 0.0, 'change_7d': 0.0, 'unit': '%'} # Treat single point as stable
                else:
                     return default_trend # No data after resampling

            # Get latest, 24h ago, and 7d ago (or first) values
            latest_val = daily_avg.iloc[-1]
            prev_val_24h = daily_avg.iloc[-2] if len(daily_avg) >= 2 else latest_val # Fallback if only 1 day
            # Use data from 7 days prior if available, otherwise use the first point
            seven_days_ago = daily_avg.index[-1] - pd.Timedelta(days=7)
            prev_val_7d_series = daily_avg[daily_avg.index <= seven_days_ago]
            prev_val_7d = prev_val_7d_series.iloc[-1] if not prev_val_7d_series.empty else daily_avg.iloc[0]

            # Calculate changes
            change_24h = latest_val - prev_val_24h
            change_7d = latest_val - prev_val_7d

            # Calculate linear regression slope
            x = np.arange(len(daily_avg))
            slope, _, _, _, _ = stats.linregress(x, daily_avg.values)

            # Determine trend string based on normalized slope (more robust)
            data_range = daily_avg.max() - daily_avg.min() if len(daily_avg) > 1 else 1.0
            normalized_slope = slope / (data_range + 1e-9) # Avoid division by zero

            trend_str = 'stable'
            if normalized_slope > 0.05:  # Threshold for 'improving' (e.g., >5% change over period)
                trend_str = 'improving'
            elif normalized_slope < -0.05: # Threshold for 'declining'
                trend_str = 'declining'

            # Unit determination
            unit = '%'
            multiplier = 100
            if metric == 'value' and 'unit' in df.columns: # Special case for 'value' trend
                unit_series = df_copy['unit'].dropna() # Use df_copy to access unit after potential dropna
                if not unit_series.empty:
                    unit = unit_series.iloc[0]
                    multiplier = 1 # Don't multiply raw values by 100
                else:
                     unit = '' # Fallback if unit column exists but is empty
                     multiplier = 1
            elif metric != 'health_score' and metric != 'efficiency_score':
                 # Assume other metrics are not 0-1 scores
                 multiplier = 1
                 unit_series = df_copy['unit'].dropna()
                 unit = unit_series.iloc[0] if not unit_series.empty else ''


            return {
                'trend': trend_str,
                'slope': round(slope * multiplier, 3), # Apply multiplier for scores
                'change_24h': round(change_24h * multiplier, 2), # Apply multiplier for scores
                'change_7d': round(change_7d * multiplier, 2), # Apply multiplier for scores
                'unit': unit
            }

        except Exception as e:
            self.logger.error(f"Detailed trend calculation error for {metric}: {e}", exc_info=True)
            return default_trend


    def _calculate_trend_analysis(self, df: pd.DataFrame) -> Dict:
        """Calculate detailed trend analysis for key metrics from DataFrame."""
        # (Implementation remains the same - already uses DataFrames)
        try:
            trends = {}

            if df.empty:
                return {
                    'health': self._calculate_detailed_trend(df, 'health_score'),
                    'efficiency': self._calculate_detailed_trend(df, 'efficiency_score')
                }

            # Health trend
            trends['health'] = self._calculate_detailed_trend(df, 'health_score')

            # Efficiency trend
            trends['efficiency'] = self._calculate_detailed_trend(df, 'efficiency_score')

            # Example: Key metric trend (e.g., Temperature)
            # Find common device types and analyze their 'value' trend
            common_types = df['device_type'].value_counts().head(3).index.tolist()
            for dev_type in common_types:
                 type_df = df[df['device_type'] == dev_type]
                 if not type_df.empty:
                      # Use a simplified key like 'temperature_trend' instead of 'temperature_sensor_trend'
                      metric_key = dev_type.split('_')[0] + "_trend"
                      trends[metric_key] = self._calculate_detailed_trend(type_df, 'value')


            return trends

        except Exception as e:
            self.logger.error(f"Trend analysis calculation error: {e}", exc_info=True)
            return {'health': {'trend': 'stable'}, 'efficiency': {'trend': 'stable'}}


    def _calculate_asset_comparisons(self, latest_df: pd.DataFrame) -> Dict:
        """Calculate multi-asset comparisons from latest snapshot DataFrame."""
        # (Implementation remains the same - already uses DataFrames)
        try:
            comparisons = {}
            if latest_df.empty: return comparisons

            # Top 5 / Bottom 5 by Health
            if 'health_score' in latest_df.columns and not latest_df['health_score'].isnull().all():
                health_cols = ['device_id', 'device_name', 'health_score']
                # Drop rows where health_score is NaN before nlargest/nsmallest
                health_df_valid = latest_df.dropna(subset=['health_score'])

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
                    top_eff = eff_df_valid.nlargest(5, 'efficiency_score')[eff_cols].copy()
                    top_eff['efficiency_score'] = (top_eff['efficiency_score'] * 100).round(1)

                    bottom_eff = eff_df_valid.nsmallest(5, 'efficiency_score')[eff_cols].copy()
                    bottom_eff['efficiency_score'] = (bottom_eff['efficiency_score'] * 100).round(1)

                    comparisons['top_5_efficiency'] = top_eff.to_dict('records')
                    comparisons['bottom_5_efficiency'] = bottom_eff.to_dict('records')

            # Aggregates by Location
            if 'location' in latest_df.columns and 'health_score' in latest_df.columns:
                health_by_loc = (latest_df.groupby('location')['health_score'].mean() * 100).round(1).dropna()
                if not health_by_loc.empty:
                    comparisons['health_by_location'] = health_by_loc.to_dict()

            # Aggregates by Device Type
            if 'device_type' in latest_df.columns and 'efficiency_score' in latest_df.columns:
                eff_by_type = (latest_df.groupby('device_type')['efficiency_score'].mean() * 100).round(1).dropna()
                if not eff_by_type.empty:
                    comparisons['efficiency_by_type'] = eff_by_type.to_dict()

            return comparisons

        except Exception as e:
            self.logger.error(f"Asset comparison calculation error: {e}", exc_info=True)
            return {}

    # REMOVED: _get_alerts_summary - Should be passed in

    def _calculate_energy_metrics(self, historical_df: pd.DataFrame) -> Dict:
        """Calculate energy consumption metrics based on historical DataFrame."""
        # (Implementation remains the same - already uses DataFrames)
        try:
            # Filter for power meter data
            power_df = historical_df[historical_df['device_type'] == 'power_meter'].copy()

            if power_df.empty:
                # Fallback to simulated data if no power meter data
                return self._get_default_energy_metrics()

            # Ensure timestamp is datetime
            if 'timestamp' in power_df.columns and not pd.api.types.is_datetime64_any_dtype(power_df['timestamp']):
                power_df['timestamp'] = pd.to_datetime(power_df['timestamp'])

            # Get latest total power by summing the last reading of each power meter
            latest_power_df = power_df.sort_values('timestamp').groupby('device_id').last()
            current_power_w = latest_power_df['value'].sum()
            current_power_kw = current_power_w / 1000.0

            # Calculate daily energy estimate from historical data (more accurate)
            # Resample to hourly mean power, then sum over 24h
            daily_energy_kwh = 0 # Default
            if 'timestamp' in power_df.columns and len(power_df) > 1:
                power_df_indexed = power_df.set_index('timestamp')
                # Sum power across all meters at each timestamp
                total_power_series_w = power_df_indexed.groupby(power_df_indexed.index)['value'].sum()
                # Resample to hourly average power in kW
                hourly_power_kw = total_power_series_w.resample('H').mean() / 1000.0

                # Get the last 24 hours of data relative to the LATEST timestamp in the data
                if not hourly_power_kw.empty:
                    last_ts = hourly_power_kw.index.max()
                    start_ts = last_ts - pd.Timedelta(hours=24)
                    last_24h_power = hourly_power_kw[(hourly_power_kw.index > start_ts) & (hourly_power_kw.index <= last_ts)]

                    if not last_24h_power.empty:
                        # Energy = Power * Time (assuming hourly samples, so time interval = 1 hour)
                        # Summing the average hourly power gives the total kWh for the period
                        daily_energy_kwh = last_24h_power.sum()
                    else:
                        # Fallback: estimate based on current power if not enough history
                        daily_energy_kwh = current_power_kw * 24
                else:
                     daily_energy_kwh = current_power_kw * 24
            else:
                 # Fallback estimate
                 daily_energy_kwh = current_power_kw * 24

            monthly_cost_usd = daily_energy_kwh * 30 * 0.12 # Example cost: $0.12 per kWh

            # Estimate efficiency and footprint (can be refined)
            # Use efficiency score if available, otherwise simulate
            avg_efficiency_series = historical_df['efficiency_score'].dropna()
            avg_efficiency = avg_efficiency_series.mean() * 100 if not avg_efficiency_series.empty else np.random.uniform(85, 95)

            return {
                'current_power_kw': round(current_power_kw, 1),
                'daily_energy_kwh': round(daily_energy_kwh, 1),
                'monthly_cost_usd': round(monthly_cost_usd, 0),
                'energy_efficiency': round(avg_efficiency, 1), # Use calculated or simulated
                'carbon_footprint_kg': round(daily_energy_kwh * 0.4, 1)  # Example factor: 0.4 kg CO2 per kWh
            }

        except Exception as e:
            self.logger.error(f"Energy metrics calculation error: {e}", exc_info=True)
            return self._get_default_energy_metrics()

    def _get_default_energy_metrics(self) -> Dict:
        """Get default energy metrics when no data is available."""
        # (Implementation remains the same)
        return {
            'current_power_kw': 125.0, # Adjusted default
            'daily_energy_kwh': 3000.0, # Adjusted default
            'monthly_cost_usd': 360.0, # Adjusted default
            'energy_efficiency': 88.5,
            'carbon_footprint_kg': 1200.0 # Adjusted default
        }

    # REMOVED: _fetch_energy_data - Integrated into _calculate_energy_metrics

    def _get_default_overview(self) -> Dict:
        """Get default dashboard overview when no data is available."""
        # (Implementation remains the same)
        return {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': {'total_devices': 0, 'active_devices': 0, 'uptime': 0.0},
            'device_metrics': {},
            'health_metrics': self._get_default_health_metrics(),
            'performance_metrics': self._get_default_performance_metrics(),
            'status_distribution': {'normal': 0, 'warning': 0, 'critical': 0, 'offline': 0},
            'trend_analysis': {'health': {'trend': 'stable'}, 'efficiency': {'trend': 'stable'}},
            'asset_comparisons': {},
            'alerts_summary': {'total_alerts': 0, 'critical_alerts': 0}, # Keep alerts structure
            'energy_metrics': self._get_default_energy_metrics()
        }


    # --- Chart preparation methods remain the same, operating on DataFrames ---
    # (No changes needed for _prepare_line_chart_data, _prepare_bar_chart_data, etc.)
    # ... (rest of the chart methods like _prepare_pie_chart_data_from_counts) ...
    def _prepare_pie_chart_data_from_counts(self, counts: Dict[str, int]) -> Dict:
        """Prepares data suitable for a pie chart from a counts dictionary."""
        labels = list(counts.keys())
        values = list(counts.values())
        colors = [self.color_schemes['status'].get(label, '#bdc3c7') for label in labels] # Use status colors

        return {
            'labels': labels,
            'datasets': [{
                'data': values,
                'backgroundColor': colors,
                'hoverBackgroundColor': colors # Or slightly darker versions
            }]
        }

    # Added prepare_chart_data from example usage for completeness
    def prepare_chart_data(self, chart_type: str, time_range: str = '24h', device_id: Optional[str] = None, data_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Prepares data for various chart types based on input DataFrame or generates sample data.
        NOTE: In production, this should ideally always receive a DataFrame (data_df).
              The internal generator is primarily for demo/fallback.
        """
        self.logger.info(f"Preparing chart data: type={chart_type}, range={time_range}, device={device_id}")

        if data_df is None:
            # Fallback to internal generator ONLY if no data provided
            self.logger.warning("No DataFrame provided to prepare_chart_data. Using internal demo generator.")
            if device_id:
                # Generate sample historical for a specific device
                hours = {'1h': 1, '4h': 4, '24h': 24, '7d': 7*24, '30d': 30*24}.get(time_range, 24)
                sample_records = self._generate_sample_historical_data(device_id, hours)
                data_df = pd.DataFrame(sample_records)
            else:
                 # Generate sample overview data
                 sample_records = self._generate_sample_device_data()
                 data_df = pd.DataFrame(sample_records)
                 # Filter by time range roughly
                 cutoff = datetime.now() - timedelta(days={'1h': 1/24, '4h': 4/24, '24h': 1, '7d': 7, '30d': 30}.get(time_range, 1))
                 data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
                 data_df = data_df[data_df['timestamp'] >= cutoff]


        if data_df.empty:
            return {'error': 'No data available to generate chart'}

        # Ensure timestamp is datetime
        if 'timestamp' in data_df.columns and not pd.api.types.is_datetime64_any_dtype(data_df['timestamp']):
            data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])

        try:
            if chart_type == 'line':
                return self._prepare_line_chart_data(data_df, device_id)
            # Add other chart types like 'bar', 'pie' based on needs
            elif chart_type == 'pie': # Example: Use status distribution
                 latest_df = data_df.sort_values('timestamp').groupby('device_id').last().reset_index()
                 status_counts = latest_df['status'].value_counts().to_dict()
                 return self._prepare_pie_chart_data_from_counts(status_counts)
            else:
                return {'error': f'Unsupported chart type: {chart_type}'}
        except Exception as e:
            self.logger.error(f"Chart preparation error for {chart_type}: {e}", exc_info=True)
            return {'error': str(e)}

    def _prepare_line_chart_data(self, df: pd.DataFrame, device_id: Optional[str] = None) -> Dict:
        """Prepares data for a line chart (e.g., value over time)."""
        df_filtered = df if device_id is None else df[df['device_id'] == device_id]

        if df_filtered.empty:
            return {'labels': [], 'datasets': []}

        df_sorted = df_filtered.sort_values('timestamp')

        labels = df_sorted['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
        values = df_sorted['value'].tolist()
        unit = df_sorted['unit'].iloc[0] if not df_sorted['unit'].empty else ''
        label_text = f"{device_id} Value ({unit})" if device_id else f"Average Value ({unit})"

        return {
            'labels': labels,
            'datasets': [{
                'label': label_text,
                'data': values,
                'borderColor': self.color_schemes['primary'][0],
                'tension': 0.1
            }]
        }


    # --- Real-time updates remain the same, operating on DataFrames ---
    # (No changes needed for get_realtime_updates, _get_metric_with_delta, etc.)
    # ... (rest of real-time update methods) ...

    # --- Device summary needs modification to accept DataFrame input ---

    def get_device_summary(self, device_id: str, historical_df: pd.DataFrame, alerts: List[Dict], recommendations: List[Dict]) -> Dict:
        """
        Get comprehensive summary for a specific device based on provided data.

        Args:
            device_id: Device identifier.
            historical_df: DataFrame with historical data for the specific device (e.g., last 7 days).
            alerts: List of alerts specific to this device.
            recommendations: List of recommendations specific to this device.

        Returns:
            Device summary data dictionary.
        """
        try:
            if historical_df.empty:
                return {'error': f'No historical data provided for device {device_id}'}

            # Ensure timestamp is datetime
            if 'timestamp' in historical_df.columns and not pd.api.types.is_datetime64_any_dtype(historical_df['timestamp']):
                 historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])

            # Get the latest data point for current metrics
            device = historical_df.sort_values('timestamp').iloc[-1].to_dict()

            summary = {
                'device_id': device_id,
                'device_name': device.get('device_name', ''),
                'device_type': device.get('device_type', ''),
                'location': device.get('location', ''),
                'status': device.get('status', 'unknown'),
                'current_metrics': {
                    'value': device.get('value', 0),
                    'unit': device.get('unit', ''),
                    'health_score': round(device.get('health_score', 0) * 100, 1),
                    'efficiency_score': round(device.get('efficiency_score', 0) * 100, 1),
                    'operating_hours': round(device.get('operating_hours', 0)),
                    'days_since_maintenance': int(device.get('days_since_maintenance', 0))
                },
                'trends': self._calculate_device_trends(historical_df), # Use full history for trends
                'alerts': alerts, # Pass through provided alerts
                'recommendations': recommendations, # Pass through provided recommendations
                'last_updated': device.get('timestamp', datetime.now().isoformat())
            }
            # Ensure last_updated is string
            if isinstance(summary['last_updated'], (datetime, pd.Timestamp)):
                 summary['last_updated'] = summary['last_updated'].isoformat()


            return summary

        except Exception as e:
            self.logger.error(f"Device summary error for {device_id} using DataFrame: {e}", exc_info=True)
            return {'error': str(e)}

    # REMOVED: _get_device_historical_data - Data should be fetched via SQLAlchemy

    # Kept for fallback/demo
    def _generate_sample_historical_data(self, device_id: str, hours: int) -> List[Dict]:
        """Generate sample historical data for demonstration."""
        # (Implementation remains the same)
        try:
            data = []
            now = datetime.now()
            base_health = np.random.uniform(0.6, 0.95)
            health_trend = np.random.uniform(-0.001, 0.0005) # Slight trend

            for i in range(hours):
                timestamp = now - timedelta(hours=hours-1-i)

                # Generate realistic pattern
                base_value = 25 + 5 * np.sin(i * 0.1)  # Daily temperature pattern
                noise = np.random.normal(0, 1)

                current_health = max(0.2, min(1.0, base_health + health_trend * i))

                data.append({
                    'device_id': device_id,
                    'timestamp': timestamp.isoformat(),
                    'value': round(base_value + noise, 2),
                    'health_score': current_health,
                    'efficiency_score': max(0.3, min(1.0, current_health - np.random.uniform(0.05, 0.1))),
                    'status': np.random.choice(['normal', 'warning'], p=[0.9, 0.1]) if current_health > 0.6 else 'critical',
                    'operating_hours': 1000 + i, # Example
                    'days_since_maintenance': 30 + i // 24 # Example
                })

            return data

        except Exception as e:
            self.logger.error(f"Sample historical data generation error: {e}")
            return []


    def _calculate_device_trends(self, historical_df: pd.DataFrame) -> Dict:
        """Calculate trends for a single device's metrics from DataFrame."""
        # (Implementation remains the same - already uses DataFrames)
        try:
            if historical_df.empty:
                return {'health': 'stable', 'efficiency': 'stable', 'value': 'stable'}

            trends = {}

            for metric in ['health_score', 'efficiency_score', 'value']:
                if metric in historical_df.columns:
                    # Use the robust trend calculation method
                    trend_details = self._calculate_detailed_trend(historical_df, metric)
                    # Extract only the trend direction string
                    trends[metric.replace('_score', '')] = trend_details.get('trend', 'stable')

            return trends

        except Exception as e:
            self.logger.error(f"Device trends calculation error: {e}", exc_info=True)
            return {'health': 'stable', 'efficiency': 'stable', 'value': 'stable'}


    # REMOVED: _get_device_alerts - Should be passed in
    # REMOVED: _get_device_recommendations - Should be passed in

    # --- Export methods remain the same, but input data source changes ---

    def export_dashboard_data(self, overview_data: Dict, latest_devices_df: pd.DataFrame, historical_df: Optional[pd.DataFrame] = None, format_type: str = 'json') -> Dict:
        """
        Export dashboard data (overview, latest devices, optional history) in specified format.

        Args:
            overview_data: The dictionary returned by get_dashboard_overview.
            latest_devices_df: DataFrame of the latest data for each device.
            historical_df: Optional DataFrame of historical data used for the overview.
            format_type: Export format ('json', 'csv', 'excel').

        Returns:
            Dictionary with export path or error.
        """
        try:
            self.logger.info(f"Exporting dashboard data in {format_type} format from provided data")

            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'overview': overview_data,
                'latest_devices': latest_devices_df.to_dict('records') if not latest_devices_df.empty else []
            }

            if historical_df is not None and not historical_df.empty:
                # Ensure timestamp is string for JSON/CSV compatibility if needed
                historical_df_export = historical_df.copy() # Avoid modifying original
                if 'timestamp' in historical_df_export.columns and pd.api.types.is_datetime64_any_dtype(historical_df_export['timestamp']):
                     historical_df_export['timestamp'] = historical_df_export['timestamp'].dt.isoformat()

                export_data['historical_data'] = historical_df_export.to_dict('records')


            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f"dashboard_export_{timestamp_str}"

            if format_type == 'json':
                export_path = self.cache_path / f"{base_filename}.json"
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                return {'export_path': str(export_path), 'format': 'json'}

            elif format_type == 'csv':
                # Export latest devices to CSV
                export_path = self.cache_path / f"{base_filename}_latest_devices.csv"
                latest_devices_df.to_csv(export_path, index=False, date_format='%Y-%m-%dT%H:%M:%SZ')
                export_paths = {'latest_devices': str(export_path)}

                # Optionally export historical data to a separate CSV
                if historical_df is not None and not historical_df.empty:
                    # Use the same export_df as JSON for consistent timestamp formatting
                    hist_export_path = self.cache_path / f"{base_filename}_historical.csv"
                    historical_df_export.to_csv(hist_export_path, index=False, date_format='%Y-%m-%dT%H:%M:%SZ')
                    export_paths['historical_data'] = str(hist_export_path)

                return {'export_paths': export_paths, 'format': 'csv'}

            # Add 'excel' format if needed using pandas ExcelWriter
            elif format_type == 'excel':
                export_path = self.cache_path / f"{base_filename}.xlsx"
                with pd.ExcelWriter(export_path, engine='xlsxwriter',
                                    datetime_format='yyyy-mm-dd hh:mm:ss',
                                    date_format='yyyy-mm-dd') as writer:
                    latest_devices_df.to_excel(writer, sheet_name='Latest Devices', index=False)
                    if historical_df is not None and not historical_df.empty:
                        # Use export_df for consistent timestamp formatting if exists
                        (historical_df_export if 'historical_df_export' in locals() else historical_df).to_excel(writer, sheet_name='Historical Data', index=False)
                return {'export_path': str(export_path), 'format': 'excel'}


            else:
                return {'error': f'Unsupported export format: {format_type}'}

        except Exception as e:
            self.logger.error(f"Dashboard data export error: {e}", exc_info=True)
            return {'error': str(e)}

    # REMOVED: _get_all_historical_data - Data should be fetched via SQLAlchemy

    # --- Cache methods remain conceptually similar ---
    # ... (clear_cache, get_cache_statistics) ...


# Example usage adjusted for DataFrame input
if __name__ == "__main__":
    # Initialize dashboard helper
    dashboard = DashboardHelper()

    print("=== DIGITAL TWIN DASHBOARD HELPER DEMO (v2.0 - DataFrame Input) ===\n")

    # 1. Generate sample data (simulating data fetched from DB)
    print("1. Generating sample historical DataFrame (simulating DB fetch)...")
    # Use the helper's generator for demo purposes
    sample_records = dashboard._generate_sample_device_data() # Generates ~200 records over 7 days
    historical_df = pd.DataFrame(sample_records)
    if not historical_df.empty:
        historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
        print(f"Generated DataFrame with {len(historical_df)} records.")
    else:
        print("Failed to generate sample data.")
        sys.exit(1) # Use sys.exit


    # 2. Get dashboard overview using the DataFrame
    print("\n2. Getting dashboard overview (using DataFrame)...")
    # Simulate alerts summary
    alerts_summary_demo = {
        'total_alerts': 8, 'critical_alerts': 1, 'warning_alerts': 4, 'info_alerts': 3
    }
    overview = dashboard.get_dashboard_overview(historical_df, alerts_summary_demo)

    # Print summary parts (same as before)
    print(f"System Metrics:")
    system_metrics = overview.get('system_metrics', {})
    print(f"  - Total Devices: {system_metrics.get('total_devices', 0)}")
    print(f"  - Active Devices: {system_metrics.get('active_devices', 0)}")
    # ... (print other overview sections as before) ...
    print(f"\nHealth Metrics:")
    health_metrics = overview.get('health_metrics', {})
    print(f"  - Average Health: {health_metrics.get('average_health', 0)}%")
    print(f"  - Health Trend: {health_metrics.get('health_trend', 'stable')}")


    # 3. Test chart data preparation (these methods might need adjustment based on how data is passed)
    print("\n" + "="*50)
    print("3. Testing chart data preparation (using generated overview data)...")
    # Example: Pie chart using status distribution from overview
    status_dist = overview.get('status_distribution', {})
    if status_dist:
         pie_chart_data = dashboard._prepare_pie_chart_data_from_counts(status_dist)
         print(f"✓ Pie Chart prepared from overview status counts.")
    else:
         print(f"✗ Could not prepare Pie chart (no status distribution in overview).")

    # Example: Prepare line chart data (needs refinement - how to pass specific device history?)
    # For demo, we might call the internal generator
    line_chart = dashboard.prepare_chart_data('line', time_range='7d') # Uses internal generator for demo
    print(f"✓ Line Chart prepared (using internal demo generator).")


    # 4. Test device summary using DataFrame subset
    print("\n" + "="*50)
    print("4. Testing device summary (using DataFrame subset)...")
    device_id_to_test = 'DEVICE_001'
    device_historical_df = historical_df[historical_df['device_id'] == device_id_to_test]

    if not device_historical_df.empty:
        # Simulate alerts and recommendations for the device
        device_alerts_demo = [{'title': 'Sample Alert', 'severity': 'warning'}]
        device_recs_demo = [{'action': 'Perform routine check'}]

        device_summary = dashboard.get_device_summary(
            device_id_to_test,
            device_historical_df,
            device_alerts_demo,
            device_recs_demo
        )

        if 'error' not in device_summary:
            print(f"Device: {device_summary.get('device_name', 'Unknown')}")
            print(f"Status: {device_summary.get('status', 'Unknown')}")
            current_metrics = device_summary.get('current_metrics', {})
            print(f"Health Score: {current_metrics.get('health_score', 0)}%")
            trends = device_summary.get('trends', {})
            print(f"Trends: Health={trends.get('health', 'stable')}")
            print(f"Alerts: {len(device_summary.get('alerts', []))}")
            print(f"Recommendations: {len(device_summary.get('recommendations', []))}")
        else:
            print(f"✗ Device Summary Error: {device_summary['error']}")
    else:
         print(f"✗ Cannot test device summary: No data found for {device_id_to_test}")


    # 5. Test data export using generated data
    print("\n" + "="*50)
    print("5. Testing data export (using generated overview and latest data)...")
    latest_devices_df = historical_df.sort_values('timestamp').groupby('device_id').last().reset_index()

    # JSON Export
    json_export = dashboard.export_dashboard_data(overview, latest_devices_df, historical_df, format_type='json')
    if 'error' not in json_export:
        print(f"✓ JSON Export: {json_export.get('export_path', 'Unknown path')}")
    else:
        print(f"✗ JSON Export Error: {json_export['error']}")

    # CSV Export
    csv_export = dashboard.export_dashboard_data(overview, latest_devices_df, historical_df, format_type='csv')
    if 'error' not in csv_export:
        print(f"✓ CSV Export: {csv_export.get('export_paths', 'Unknown paths')}")
    else:
        print(f"✗ CSV Export Error: {csv_export['error']}")

    # Excel Export
    excel_export = dashboard.export_dashboard_data(overview, latest_devices_df, historical_df, format_type='excel')
    if 'error' not in excel_export:
        print(f"✓ Excel Export: {excel_export.get('export_path', 'Unknown path')}")
    else:
        print(f"✗ Excel Export Error: {excel_export['error']}")

    print("\n=== DEMO COMPLETED (v2.0) ===")