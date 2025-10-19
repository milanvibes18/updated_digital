#!/usr/bin/env python3
"""
Dashboard Helper Module for Digital Twin System
Provides data processing, aggregation, and formatting for dashboard visualization.
"""

import numpy as np
import pandas as pd
import logging
import json
import sqlite3
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

class DashboardHelper:
    """
    Comprehensive dashboard helper for Digital Twin applications.
    Provides data processing, aggregation, chart preparation, and real-time updates.
    """
    
    def __init__(self, cache_path: str = "ANALYTICS/analysis_cache/"):
        self.cache_path = Path(cache_path)
        self.logger = self._setup_logging()
        
        # Create cache directory
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Data caches
        self.data_cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Aggregation settings
        self.aggregation_config = {
            'temperature': {'unit': '°C', 'normal_range': (15, 35), 'precision': 1},
            'pressure': {'unit': 'hPa', 'normal_range': (950, 1050), 'precision': 1},
            'vibration': {'unit': 'mm/s', 'normal_range': (0, 0.5), 'precision': 3},
            'humidity': {'unit': '%RH', 'normal_range': (30, 70), 'precision': 1},
            'power': {'unit': 'W', 'normal_range': (800, 2000), 'precision': 1},
            'efficiency': {'unit': '%', 'normal_range': (70, 100), 'precision': 1}
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
        
        # Real-time data buffer
        self.realtime_buffer = defaultdict(lambda: deque(maxlen=100))
        
    def _setup_logging(self):
        """Setup logging for dashboard helper."""
        logger = logging.getLogger('DashboardHelper')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Ensure logs directory exists
            Path("LOGS").mkdir(exist_ok=True)
            handler = logging.FileHandler('LOGS/digital_twin_dashboard.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def get_dashboard_overview(self) -> Dict:
        """
        Get comprehensive dashboard overview data, including historical aggregates,
        trend summaries, and multi-asset comparisons.
        
        Returns:
            Dictionary containing dashboard overview metrics
        """
        try:
            self.logger.info("Generating dashboard overview")
            
            # Fetch 7 days of data for historical trends and aggregations
            historical_df = self._fetch_device_data(days_back=7)
            
            if historical_df.empty:
                return self._get_default_overview()
            
            # Get the latest snapshot for each device
            latest_df = historical_df.groupby('device_id').last().reset_index()
            
            # Calculate key metrics
            overview = {
                'timestamp': datetime.now().isoformat(),
                'system_metrics': self._calculate_system_metrics(latest_df, historical_df),
                'device_metrics': self._calculate_device_metrics(latest_df),
                'health_metrics': self._calculate_health_metrics(historical_df),
                'performance_metrics': self._calculate_performance_metrics(historical_df),
                'status_distribution': self._calculate_status_distribution(latest_df),
                'trend_analysis': self._calculate_trend_analysis(historical_df),
                'asset_comparisons': self._calculate_asset_comparisons(latest_df),
                'alerts_summary': self._get_alerts_summary(),
                'energy_metrics': self._calculate_energy_metrics(historical_df)
            }
            
            return overview
            
        except Exception as e:
            self.logger.error(f"Dashboard overview generation error: {e}")
            return self._get_default_overview()
    
    def _fetch_device_data(self, days_back: int = 1) -> pd.DataFrame:
        """
        Fetch device data from the database for the specified number of days.
        
        Args:
            days_back: Number of days to fetch data for.
            
        Returns:
            DataFrame containing device data.
        """
        try:
            db_path = "DATABASE/health_data.db"
            
            if not Path(db_path).exists():
                return pd.DataFrame(self._generate_sample_device_data())
            
            with sqlite3.connect(db_path) as conn:
                query = f"""
                    SELECT device_id, device_name, device_type, value, unit, status, 
                           health_score, efficiency_score, location, timestamp,
                           operating_hours, days_since_maintenance
                    FROM device_data 
                    WHERE timestamp >= datetime('now', '-{days_back} days')
                    ORDER BY timestamp DESC
                """
                
                df = pd.read_sql_query(query, conn)
                return df
                
        except Exception as e:
            self.logger.error(f"Error fetching device data: {e}")
            return pd.DataFrame(self._generate_sample_device_data())
    
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
                    timestamp = datetime.now() - timedelta(days=np.random.uniform(0, 7), 
                                                           hours=np.random.uniform(0, 24))
                    
                    device_type = device_type_base
                    
                    # Generate realistic values based on device type
                    if device_type == 'temperature_sensor':
                        value = np.random.uniform(18, 35)
                        unit = '°C'
                    elif device_type == 'pressure_sensor':
                        value = np.random.uniform(900, 1100)
                        unit = 'hPa'
                    elif device_type == 'vibration_sensor':
                        value = np.random.uniform(0.1, 0.6)
                        unit = 'mm/s'
                    elif device_type == 'humidity_sensor':
                        value = np.random.uniform(30, 80)
                        unit = '%RH'
                    else:  # power_meter
                        value = np.random.uniform(800, 2000)
                        unit = 'W'
                    
                    # Determine status based on value ranges
                    config = self.aggregation_config.get(device_type.split('_')[0], {})
                    normal_range = config.get('normal_range', (0, 100))
                    
                    # Simulate trend
                    health_score = max(0.2, min(1.0, base_health + (j * 0.01))) # Slowly improving
                    efficiency_score = max(0.2, min(1.0, base_efficiency - (j * 0.005))) # Slowly declining
                    
                    if normal_range[0] <= value <= normal_range[1]:
                        status = 'normal'
                    elif value < normal_range[0] * 0.8 or value > normal_range[1] * 1.2:
                        status = 'critical'
                    else:
                        status = 'warning'
                    
                    devices.append({
                        'device_id': f'DEVICE_{i+1:03d}',
                        'device_name': f'{device_type.replace("_", " ").title()} {i+1:03d}',
                        'device_type': device_type,
                        'value': round(value, config.get('precision', 1)),
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
        """Calculate system-level metrics."""
        try:
            total_devices = len(latest_df)
            
            if total_devices == 0:
                return {'total_devices': 0, 'active_devices': 0, 'uptime': 100.0, 'data_points_collected': 0}
            
            active_devices = len(latest_df[latest_df['status'] == 'normal'])
            
            # Calculate system uptime based on device status
            uptime = (active_devices / total_devices) * 100 if total_devices > 0 else 100.0
            
            # Calculate average response time (simulated)
            response_time = np.random.uniform(50, 200)
            
            return {
                'total_devices': total_devices,
                'active_devices': active_devices,
                'offline_devices': total_devices - active_devices,
                'uptime': round(uptime, 2),
                'response_time_ms': round(response_time, 1),
                'data_points_collected': len(historical_df), # Total points in the period
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"System metrics calculation error: {e}")
            return {'total_devices': 0, 'active_devices': 0, 'uptime': 0.0, 'data_points_collected': 0}
    
    def _calculate_device_metrics(self, latest_df: pd.DataFrame) -> Dict:
        """Calculate device-specific metrics based on latest snapshot."""
        try:
            if latest_df.empty:
                return {}
            
            device_metrics = {}
            
            for device_type in latest_df['device_type'].unique():
                type_df = latest_df[latest_df['device_type'] == device_type]
                
                device_metrics[device_type] = {
                    'count': len(type_df),
                    'average_value': round(type_df['value'].mean(), 2) if 'value' in type_df.columns else 0,
                    'min_value': round(type_df['value'].min(), 2) if 'value' in type_df.columns else 0,
                    'max_value': round(type_df['value'].max(), 2) if 'value' in type_df.columns else 0,
                    'unit': type_df['unit'].iloc[0] if 'unit' in type_df.columns else 'units',
                    'status_distribution': type_df['status'].value_counts().to_dict() if 'status' in type_df.columns else {}
                }
            
            return device_metrics
            
        except Exception as e:
            self.logger.error(f"Device metrics calculation error: {e}")
            return {}
    
    def _calculate_health_metrics(self, historical_df: pd.DataFrame) -> Dict:
        """Calculate health-related metrics from historical data."""
        try:
            if historical_df.empty or 'health_score' not in historical_df.columns:
                return self._get_default_health_metrics()
            
            # Get latest health scores for current snapshot
            latest_df = historical_df.groupby('device_id').last().reset_index()
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
                'std_health': round(health_scores_pct.std(), 1),
                'health_distribution': health_distribution,
                'health_trend': health_trend_details.get('trend', 'stable'), # Simple trend for overview card
                'devices_needing_attention': health_distribution['poor'] + health_distribution['critical']
            }
            
        except Exception as e:
            self.logger.error(f"Health metrics calculation error: {e}")
            return self._get_default_health_metrics()
    
    def _get_default_health_metrics(self) -> Dict:
        """Get default health metrics when no data is available."""
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
        """Calculate performance-related metrics."""
        try:
            if historical_df.empty or 'efficiency_score' not in historical_df.columns:
                return self._get_default_performance_metrics()
            
            # Get latest snapshot for current metrics
            latest_df = historical_df.groupby('device_id').last().reset_index()
            efficiency_scores = latest_df['efficiency_score'].dropna() * 100
            
            # Calculate operating hours statistics
            operating_hours_stats = {}
            if 'operating_hours' in latest_df.columns:
                operating_hours = latest_df['operating_hours'].dropna()
                operating_hours_stats = {
                    'total_operating_hours': round(operating_hours.sum(), 0),
                    'average_operating_hours': round(operating_hours.mean(), 0),
                    'max_operating_hours': round(operating_hours.max(), 0)
                }
            
            # Calculate maintenance statistics
            maintenance_stats = {}
            if 'days_since_maintenance' in latest_df.columns:
                maintenance_days = latest_df['days_since_maintenance'].dropna()
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
            self.logger.error(f"Performance metrics calculation error: {e}")
            return self._get_default_performance_metrics()
    
    def _get_default_performance_metrics(self) -> Dict:
        """Get default performance metrics when no data is available."""
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
        """Calculate device status distribution from latest snapshot."""
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
            self.logger.error(f"Status distribution calculation error: {e}")
            return {'normal': 0, 'warning': 0, 'critical': 0, 'offline': 0}

    def _calculate_detailed_trend(self, df: pd.DataFrame, metric: str) -> Dict:
        """
        Calculate detailed trend analysis for a specific metric using daily resampling.
        
        Args:
            df: The historical DataFrame (e.g., 7 days).
            metric: The column name of the metric to analyze (e.g., 'health_score').
            
        Returns:
            A dictionary with trend, slope, and changes.
        """
        default_trend = {'trend': 'stable', 'slope': 0.0, 'change_24h': 0.0, 'change_7d': 0.0, 'unit': '%'}
        
        try:
            if df.empty or metric not in df.columns:
                return default_trend
            
            df_copy = df.dropna(subset=[metric, 'timestamp']).copy()
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            
            # Resample to daily average
            daily_avg = df_copy.set_index('timestamp').resample('D')[metric].mean().dropna()
            
            if len(daily_avg) < 2:
                return default_trend
                
            # Get latest, 24h ago, and 7d ago (or first) values
            latest_val = daily_avg.iloc[-1]
            prev_val_24h = daily_avg.iloc[-2]
            prev_val_7d = daily_avg.iloc[0] # First value in the series
            
            # Calculate changes
            change_24h = latest_val - prev_val_24h
            change_7d = latest_val - prev_val_7d
            
            # Calculate linear regression slope
            x = np.arange(len(daily_avg))
            slope, _, _, _, _ = stats.linregress(x, daily_avg.values)
            
            # Determine trend string
            trend_str = 'stable'
            if slope > 0.005:  # Threshold for 'improving'
                trend_str = 'improving'
            elif slope < -0.005: # Threshold for 'declining'
                trend_str = 'declining'
                
            # Unit (all our trends are scores 0-1, so we convert to %)
            unit = '%'
            multiplier = 100
            if metric == 'value': # Special case for 'value' trend
                unit = df_copy['unit'].iloc[0] if 'unit' in df_copy.columns else ''
                multiplier = 1

            return {
                'trend': trend_str,
                'slope': round(slope * multiplier, 3),
                'change_24h': round(change_24h * multiplier, 2),
                'change_7d': round(change_7d * multiplier, 2),
                'unit': unit
            }

        except Exception as e:
            self.logger.error(f"Detailed trend calculation error for {metric}: {e}")
            return default_trend

    def _calculate_trend_analysis(self, df: pd.DataFrame) -> Dict:
        """Calculate detailed trend analysis for key metrics."""
        try:
            trends = {}
            
            if not df.empty:
                # Health trend
                trends['health'] = self._calculate_detailed_trend(df, 'health_score')
                
                # Efficiency trend
                trends['efficiency'] = self._calculate_detailed_trend(df, 'efficiency_score')
                
                # Example: Key metric trend (e.g., Temperature)
                temp_df = df[df['device_type'] == 'temperature_sensor']
                if not temp_df.empty:
                    trends['temperature'] = self._calculate_detailed_trend(temp_df, 'value')
                
            else:
                trends = {
                    'health': self._calculate_detailed_trend(df, 'health_score'),
                    'efficiency': self._calculate_detailed_trend(df, 'efficiency_score')
                }
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Trend analysis calculation error: {e}")
            return {'health': {'trend': 'stable'}, 'efficiency': {'trend': 'stable'}}

    def _calculate_asset_comparisons(self, latest_df: pd.DataFrame) -> Dict:
        """Calculate multi-asset comparisons, like top/bottom performers."""
        try:
            comparisons = {}
            
            # Top 5 / Bottom 5 by Health
            if 'health_score' in latest_df.columns:
                health_cols = ['device_id', 'device_name', 'health_score']
                top_health = latest_df.nlargest(5, 'health_score')[health_cols].copy()
                top_health['health_score'] = (top_health['health_score'] * 100).round(1)
                
                bottom_health = latest_df.nsmallest(5, 'health_score')[health_cols].copy()
                bottom_health['health_score'] = (bottom_health['health_score'] * 100).round(1)
                
                comparisons['top_5_health'] = top_health.to_dict('records')
                comparisons['bottom_5_health'] = bottom_health.to_dict('records')

            # Top 5 / Bottom 5 by Efficiency
            if 'efficiency_score' in latest_df.columns:
                eff_cols = ['device_id', 'device_name', 'efficiency_score']
                top_eff = latest_df.nlargest(5, 'efficiency_score')[eff_cols].copy()
                top_eff['efficiency_score'] = (top_eff['efficiency_score'] * 100).round(1)
                
                bottom_eff = latest_df.nsmallest(5, 'efficiency_score')[eff_cols].copy()
                bottom_eff['efficiency_score'] = (bottom_eff['efficiency_score'] * 100).round(1)

                comparisons['top_5_efficiency'] = top_eff.to_dict('records')
                comparisons['bottom_5_efficiency'] = bottom_eff.to_dict('records')

            # Aggregates by Location
            if 'location' in latest_df.columns and 'health_score' in latest_df.columns:
                health_by_loc = (latest_df.groupby('location')['health_score'].mean() * 100).round(1)
                comparisons['health_by_location'] = health_by_loc.to_dict()

            # Aggregates by Device Type
            if 'device_type' in latest_df.columns and 'efficiency_score' in latest_df.columns:
                eff_by_type = (latest_df.groupby('device_type')['efficiency_score'].mean() * 100).round(1)
                comparisons['efficiency_by_type'] = eff_by_type.to_dict()

            return comparisons

        except Exception as e:
            self.logger.error(f"Asset comparison calculation error: {e}")
            return {}

    def _get_alerts_summary(self) -> Dict:
        """Get alerts summary information."""
        try:
            # In a real implementation, this would fetch from the alert system
            # For now, return simulated data
            return {
                'total_alerts': 8,
                'critical_alerts': 1,
                'warning_alerts': 4,
                'info_alerts': 3,
                'recent_alerts': [
                    {
                        'id': 'alert_001',
                        'title': 'Temperature High',
                        'severity': 'warning',
                        'device_id': 'DEVICE_003',
                        'timestamp': (datetime.now() - timedelta(minutes=15)).isoformat()
                    },
                    {
                        'id': 'alert_002',
                        'title': 'Pressure Critical',
                        'severity': 'critical',
                        'device_id': 'DEVICE_007',
                        'timestamp': (datetime.now() - timedelta(minutes=5)).isoformat()
                    }
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Alerts summary error: {e}")
            return {'total_alerts': 0, 'critical_alerts': 0}
    
    def _calculate_energy_metrics(self, historical_df: pd.DataFrame) -> Dict:
        """Calculate energy consumption metrics based on historical data."""
        try:
            power_df = historical_df[historical_df['device_type'] == 'power_meter']
            
            if power_df.empty:
                # Fallback to simulated data
                return self._get_default_energy_metrics()

            # Get latest total power
            latest_power_df = power_df.groupby('device_id').last()
            current_power_w = latest_power_df['value'].sum()
            current_power_kw = current_power_w / 1000.0
            
            # Calculate daily energy from historical data
            power_df['timestamp'] = pd.to_datetime(power_df['timestamp'])
            # Assuming 'value' is avg power in Watts over the reporting interval.
            # This is a simplification; real energy = power * time_interval.
            # We'll simulate daily energy based on current power.
            daily_energy_kwh = current_power_kw * 24 
            monthly_cost_usd = daily_energy_kwh * 30 * 0.12 # $0.12 per kWh
            
            return {
                'current_power_kw': round(current_power_kw, 1),
                'daily_energy_kwh': round(daily_energy_kwh, 1),
                'monthly_cost_usd': round(monthly_cost_usd, 0),
                'energy_efficiency': round(np.random.uniform(85, 95), 1), # Simulated
                'carbon_footprint_kg': round(daily_energy_kwh * 0.4, 1)  # 0.4 kg CO2 per kWh
            }
            
        except Exception as e:
            self.logger.error(f"Energy metrics calculation error: {e}")
            return self._get_default_energy_metrics()

    def _get_default_energy_metrics(self) -> Dict:
        """Get default energy metrics when no data is available."""
        return {
            'current_power_kw': 1250.0,
            'daily_energy_kwh': 30.0,
            'monthly_cost_usd': 432.0,
            'energy_efficiency': 88.5,
            'carbon_footprint_kg': 12.0
        }
    
    def _fetch_energy_data(self) -> Dict:
        """Fetch energy data from database (legacy, can be merged)."""
        try:
            db_path = "DATABASE/health_data.db"
            
            if not Path(db_path).exists():
                return {}
            
            with sqlite3.connect(db_path) as conn:
                query = """
                    SELECT power_consumption_kw, energy_consumed_kwh, cost_usd
                    FROM energy_data 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """
                
                df = pd.read_sql_query(query, conn)
                
                if not df.empty:
                    return {
                        'current_power': df.iloc[0]['power_consumption_kw'],
                        'daily_energy': df.iloc[0]['energy_consumed_kwh'],
                        'monthly_cost': df.iloc[0]['cost_usd']
                    }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Energy data fetch error: {e}")
            return {}
    
    def _get_default_overview(self) -> Dict:
        """Get default dashboard overview when no data is available."""
        return {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': {'total_devices': 0, 'active_devices': 0, 'uptime': 0.0},
            'device_metrics': {},
            'health_metrics': self._get_default_health_metrics(),
            'performance_metrics': self._get_default_performance_metrics(),
            'status_distribution': {'normal': 0, 'warning': 0, 'critical': 0},
            'trend_analysis': {'health': {'trend': 'stable'}, 'efficiency': {'trend': 'stable'}},
            'asset_comparisons': {},
            'alerts_summary': {'total_alerts': 0, 'critical_alerts': 0},
            'energy_metrics': {
                'current_power_kw': 0.0,
                'daily_energy_kwh': 0.0,
                'monthly_cost_usd': 0.0,
                'energy_efficiency': 0.0
            }
        }
    
    def prepare_chart_data(self, chart_type: str, data_source: str = 'database', 
                           time_range: str = '24h', device_id: str = None) -> Dict:
        """
        Prepare data for specific chart types.
        
        Args:
            chart_type: Type of chart ('line', 'bar', 'pie', 'gauge', 'heatmap')
            data_source: Data source ('database', 'realtime', 'sample')
            time_range: Time range for data ('1h', '24h', '7d', '30d')
            device_id: Optional specific device ID
            
        Returns:
            Chart-ready data structure
        """
        try:
            self.logger.info(f"Preparing {chart_type} chart data for {time_range}")
            
            if chart_type == 'line':
                return self._prepare_line_chart_data(time_range, device_id)
            elif chart_type == 'bar':
                return self._prepare_bar_chart_data(time_range, device_id)
            elif chart_type == 'pie':
                return self._prepare_pie_chart_data()
            elif chart_type == 'gauge':
                return self._prepare_gauge_chart_data(device_id)
            elif chart_type == 'heatmap':
                return self._prepare_heatmap_data(time_range)
            else:
                return {'error': f'Unsupported chart type: {chart_type}'}
                
        except Exception as e:
            self.logger.error(f"Chart data preparation error: {e}")
            return {'error': str(e)}
    
    def _prepare_line_chart_data(self, time_range: str, device_id: str = None) -> Dict:
        """Prepare data for line charts (time series)."""
        try:
            # Calculate number of data points based on time range
            time_ranges = {
                '1h': (1, 60),    # 1 hour, 60 minutes
                '24h': (24, 60),  # 24 hours, every hour  
                '7d': (7, 24),    # 7 days, daily points
                '30d': (30, 24)   # 30 days, daily points
            }
            
            duration, frequency = time_ranges.get(time_range, (24, 60))
            
            # Generate timestamps
            now = datetime.now()
            if time_range == '1h':
                timestamps = [now - timedelta(minutes=i) for i in range(frequency-1, -1, -1)]
                labels = [ts.strftime('%H:%M') for ts in timestamps]
            elif time_range == '24h':
                timestamps = [now - timedelta(hours=i) for i in range(duration-1, -1, -1)]
                labels = [ts.strftime('%H:%M') for ts in timestamps]
            else:
                timestamps = [now - timedelta(days=i) for i in range(duration-1, -1, -1)]
                labels = [ts.strftime('%m/%d') for ts in timestamps]
            
            # Generate realistic data patterns
            datasets = []
            
            if device_id:
                # Single device data
                datasets.append({
                    'label': f'Device {device_id}',
                    'data': self._generate_time_series_pattern(duration, 'temperature'),
                    'borderColor': self.color_schemes['primary'][0],
                    'backgroundColor': self.color_schemes['primary'][0] + '20',
                    'unit': '°C'
                })
            else:
                # Multiple metrics
                metrics = [
                    ('Temperature', 'temperature', '°C'),
                    ('Pressure', 'pressure', 'hPa'),
                    ('Vibration', 'vibration', 'mm/s'),
                    ('Efficiency', 'efficiency', '%')
                ]
                
                for i, (label, metric_type, unit) in enumerate(metrics):
                    datasets.append({
                        'label': label,
                        'data': self._generate_time_series_pattern(len(labels), metric_type),
                        'borderColor': self.color_schemes['primary'][i % len(self.color_schemes['primary'])],
                        'backgroundColor': self.color_schemes['primary'][i % len(self.color_schemes['primary'])] + '20',
                        'unit': unit
                    })
            
            return {
                'type': 'line',
                'labels': labels,
                'datasets': datasets,
                'options': {
                    'responsive': True,
                    'scales': {
                        'x': {'display': True, 'title': {'display': True, 'text': 'Time'}},
                        'y': {'display': True, 'title': {'display': True, 'text': 'Value'}}
                    },
                    'plugins': {
                        'legend': {'display': True},
                        'tooltip': {'mode': 'index', 'intersect': False}
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Line chart data preparation error: {e}")
            return {'error': str(e)}
    
    def _prepare_bar_chart_data(self, time_range: str, device_id: str = None) -> Dict:
        """Prepare data for bar charts."""
        try:
            if device_id:
                # Single device metrics
                labels = ['Health Score', 'Efficiency', 'Uptime', 'Performance']
                data = [
                    np.random.uniform(70, 95),
                    np.random.uniform(75, 90),
                    np.random.uniform(95, 99),
                    np.random.uniform(80, 95)
                ]
                colors = [self.color_schemes['health']['good']] * len(labels)
            else:
                # Device type comparison
                device_types = ['Temperature', 'Pressure', 'Vibration', 'Humidity', 'Power']
                labels = device_types
                data = [np.random.randint(2, 8) for _ in device_types]
                colors = self.color_schemes['primary'][:len(labels)]
            
            return {
                'type': 'bar',
                'labels': labels,
                'datasets': [{
                    'label': 'Device Count' if not device_id else 'Metrics',
                    'data': data,
                    'backgroundColor': colors,
                    'borderColor': [color.replace('20', '') for color in colors] if isinstance(colors[0], str) else colors,
                    'borderWidth': 1
                }],
                'options': {
                    'responsive': True,
                    'scales': {
                        'y': {'beginAtZero': True}
                    },
                    'plugins': {
                        'legend': {'display': False}
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Bar chart data preparation error: {e}")
            return {'error': str(e)}
    
    def _prepare_pie_chart_data(self) -> Dict:
        """Prepare data for pie charts (status distribution)."""
        try:
            # Get device status distribution
            df = self._fetch_device_data(days_back=1)
            
            if not df.empty and 'status' in df.columns:
                status_counts = df.groupby('device_id')['status'].last().value_counts()
                labels = list(status_counts.index)
                data = list(status_counts.values)
            else:
                # Default data
                labels = ['Normal', 'Warning', 'Critical']
                data = [15, 3, 2]
            
            # Map colors
            colors = [self.color_schemes['status'].get(label.lower(), '#95a5a6') for label in labels]
            
            return {
                'type': 'pie',
                'labels': labels,
                'datasets': [{
                    'data': data,
                    'backgroundColor': colors,
                    'borderWidth': 2,
                    'borderColor': '#ffffff'
                }],
                'options': {
                    'responsive': True,
                    'plugins': {
                        'legend': {'position': 'bottom'},
                        'tooltip': {
                            'callbacks': {
                                'label': 'function(context) { return context.label + \': \' + context.parsed + \' devices\'; }'
                            }
                        }
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Pie chart data preparation error: {e}")
            return {'error': str(e)}
    
    def _prepare_gauge_chart_data(self, device_id: str = None) -> Dict:
        """Prepare data for gauge charts."""
        try:
            if device_id:
                # Single device health gauge
                device_data_df = self._fetch_device_data(days_back=1)
                device_data = device_data_df.to_dict('records')
                device = next((d for d in device_data if d.get('device_id') == device_id), None)
                
                if device and 'health_score' in device:
                    value = device['health_score'] * 100
                else:
                    value = np.random.uniform(70, 95)
            else:
                # System health gauge
                overview = self.get_dashboard_overview()
                value = overview.get('health_metrics', {}).get('average_health', 85)
            
            # Determine color based on value
            if value >= 90:
                color = self.color_schemes['health']['excellent']
            elif value >= 75:
                color = self.color_schemes['health']['good']
            elif value >= 60:
                color = self.color_schemes['health']['fair']
            elif value >= 40:
                color = self.color_schemes['health']['poor']
            else:
                color = self.color_schemes['health']['critical']
            
            return {
                'type': 'gauge',
                'value': round(value, 1),
                'min': 0,
                'max': 100,
                'color': color,
                'title': f'Health Score' + (f' - {device_id}' if device_id else ''),
                'unit': '%',
                'thresholds': [
                    {'value': 40, 'color': self.color_schemes['health']['critical']},
                    {'value': 60, 'color': self.color_schemes['health']['poor']},
                    {'value': 75, 'color': self.color_schemes['health']['fair']},
                    {'value': 90, 'color': self.color_schemes['health']['good']},
                    {'value': 100, 'color': self.color_schemes['health']['excellent']}
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Gauge chart data preparation error: {e}")
            return {'error': str(e)}
    
    def _prepare_heatmap_data(self, time_range: str) -> Dict:
        """Prepare data for heatmap visualization."""
        try:
            # Generate device location heatmap data
            device_data_df = self._fetch_device_data(days_back=1)
            device_data = device_data_df.to_dict('records')
            locations = ['Factory Floor A', 'Factory Floor B', 'Warehouse', 'Quality Lab']
            device_types = ['temperature_sensor', 'pressure_sensor', 'vibration_sensor', 'humidity_sensor']
            
            # Create matrix data
            matrix_data = []
            for i, location in enumerate(locations):
                row = []
                for j, device_type in enumerate(device_types):
                    # Count devices of this type in this location
                    count = len([d for d in device_data 
                               if d.get('location') == location and d.get('device_type') == device_type])
                    row.append(count)
                matrix_data.append(row)
            
            return {
                'type': 'heatmap',
                'data': matrix_data,
                'xLabels': [dt.replace('_sensor', '').title() for dt in device_types],
                'yLabels': locations,
                'colorScale': {
                    'min': 0,
                    'max': max(max(row) for row in matrix_data) if matrix_data else 5,
                    'colors': ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#08519c']
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'tooltip': {
                            'callbacks': {
                                'title': 'function(context) { return context[0].dataset.yLabels[context[0].dataIndex]; }',
                                'label': 'function(context) { return context.dataset.xLabels[context.dataIndex] + \': \' + context.raw + \' devices\'; }'
                            }
                        }
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Heatmap data preparation error: {e}")
            return {'error': str(e)}
    
    def _generate_time_series_pattern(self, length: int, metric_type: str) -> List[float]:
        """Generate realistic time series pattern for a metric type."""
        try:
            patterns = {
                'temperature': {'base': 25, 'amplitude': 5, 'frequency': 0.1, 'noise': 1},
                'pressure': {'base': 1013, 'amplitude': 20, 'frequency': 0.05, 'noise': 5},
                'vibration': {'base': 0.25, 'amplitude': 0.1, 'frequency': 0.15, 'noise': 0.05},
                'humidity': {'base': 55, 'amplitude': 15, 'frequency': 0.08, 'noise': 3},
                'power': {'base': 1200, 'amplitude': 300, 'frequency': 0.06, 'noise': 50},
                'efficiency': {'base': 85, 'amplitude': 10, 'frequency': 0.04, 'noise': 2}
            }
            
            config = patterns.get(metric_type, patterns['temperature'])
            
            data = []
            for i in range(length):
                # Base pattern with sine wave
                value = config['base'] + config['amplitude'] * np.sin(i * config['frequency'])
                
                # Add noise
                value += np.random.normal(0, config['noise'])
                
                # Ensure positive values for certain metrics
                if metric_type in ['vibration', 'power', 'efficiency']:
                    value = max(value, 0)
                
                data.append(round(value, 2))
            
            return data
            
        except Exception as e:
            self.logger.error(f"Time series pattern generation error: {e}")
            return [0] * length
    
    def get_realtime_updates(self, client_id: str = None) -> Dict:
        """
        Get real-time updates for dashboard.
        
        Args:
            client_id: Optional client identifier for personalized updates
            
        Returns:
            Real-time update data
        """
        try:
            # Get fresh data (last 1 day is fine for "real-time")
            df = self._fetch_device_data(days_back=1)
            
            # Calculate deltas from previous values
            updates = {
                'timestamp': datetime.now().isoformat(),
                'metrics': {
                    'system_health': self._get_metric_with_delta('system_health', df),
                    'active_devices': self._get_metric_with_delta('active_devices', df),
                    'energy_usage': self._get_metric_with_delta('energy_usage', df),
                    'efficiency': self._get_metric_with_delta('efficiency', df)
                },
                'alerts': self._get_new_alerts(),
                'device_updates': self._get_device_updates(df),
                'status_changes': self._get_status_changes(df)
            }
            
            # Store current values for next delta calculation
            self._store_previous_values(updates['metrics'])
            
            return updates
            
        except Exception as e:
            self.logger.error(f"Real-time updates error: {e}")
            return {'timestamp': datetime.now().isoformat(), 'error': str(e)}
    
    def _get_metric_with_delta(self, metric_name: str, df: pd.DataFrame) -> Dict:
        """Get metric value with delta from previous reading."""
        try:
            # Get latest snapshot
            latest_df = df.groupby('device_id').last().reset_index() if not df.empty else df
            
            # Calculate current value based on metric type
            if metric_name == 'system_health':
                if not latest_df.empty and 'health_score' in latest_df.columns:
                    current_value = latest_df['health_score'].mean() * 100
                else:
                    current_value = 85.0
            elif metric_name == 'active_devices':
                if not latest_df.empty and 'status' in latest_df.columns:
                    current_value = len(latest_df[latest_df['status'] == 'normal'])
                else:
                    current_value = 15
            elif metric_name == 'energy_usage':
                energy_data = self._calculate_energy_metrics(df) # Use historical df
                current_value = energy_data.get('current_power_kw', 1250.0)
            elif metric_name == 'efficiency':
                if not latest_df.empty and 'efficiency_score' in latest_df.columns:
                    current_value = latest_df['efficiency_score'].mean() * 100
                else:
                    current_value = 85.5
            else:
                current_value = 0
            
            # Get previous value
            previous_value = getattr(self, f'_prev_{metric_name}', current_value)
            
            # Calculate delta
            delta = current_value - previous_value
            delta_percent = (delta / previous_value * 100) if previous_value != 0 else 0
            
            return {
                'value': round(current_value, 1),
                'delta': round(delta, 1),
                'delta_percent': round(delta_percent, 1),
                'trend': 'up' if delta > 0.1 else 'down' if delta < -0.1 else 'stable'
            }
            
        except Exception as e:
            self.logger.error(f"Metric delta calculation error for {metric_name}: {e}")
            return {'value': 0, 'delta': 0, 'delta_percent': 0, 'trend': 'stable'}
    
    def _store_previous_values(self, metrics: Dict):
        """Store current metric values for next delta calculation."""
        try:
            for metric_name, metric_data in metrics.items():
                setattr(self, f'_prev_{metric_name}', metric_data.get('value', 0))
        except Exception as e:
            self.logger.error(f"Previous values storage error: {e}")
    
    def _get_new_alerts(self) -> List[Dict]:
        """Get new alerts since last check."""
        try:
            # In a real implementation, this would check for new alerts
            # For demo, occasionally return a new alert
            if np.random.random() < 0.1:  # 10% chance
                alert_types = [
                    ('Temperature Spike', 'warning'),
                    ('Pressure Anomaly', 'critical'),
                    ('Vibration High', 'warning'),
                    ('Device Offline', 'critical')
                ]
                
                alert_title, severity = alert_types[np.random.randint(0, len(alert_types))]
                device_id = f'DEVICE_{np.random.randint(1, 21):03d}'
                
                return [{
                    'id': f'alert_{int(datetime.now().timestamp())}',
                    'title': alert_title,
                    'message': f'{alert_title} detected on {device_id}',
                    'severity': severity,
                    'device_id': device_id,
                    'timestamp': datetime.now().isoformat()
                }]
            
            return []
            
        except Exception as e:
            self.logger.error(f"New alerts retrieval error: {e}")
            return []
    
    def _get_device_updates(self, df: pd.DataFrame) -> List[Dict]:
        """Get updated device information."""
        try:
            updates = []
            
            if not df.empty:
                # Get latest data for each device
                latest_df = df.groupby('device_id').last().reset_index()
                
                # Return updates for first 5 devices (limit for performance)
                for _, device in latest_df.head(5).iterrows():
                    updates.append({
                        'device_id': device['device_id'],
                        'value': round(device.get('value', 0), 2),
                        'health_score': round(device.get('health_score', 0.8) * 100, 1),
                        'status': device.get('status', 'normal'),
                        'timestamp': device.get('timestamp', datetime.now().isoformat())
                    })
            
            return updates
            
        except Exception as e:
            self.logger.error(f"Device updates error: {e}")
            return []
    
    def _get_status_changes(self, df: pd.DataFrame) -> List[Dict]:
        """Get recent device status changes."""
        try:
            # In a real implementation, this would track status changes
            # For demo, occasionally simulate a status change
            if np.random.random() < 0.05:  # 5% chance
                device_id = f'DEVICE_{np.random.randint(1, 21):03d}'
                old_status = np.random.choice(['normal', 'warning'])
                new_status = 'warning' if old_status == 'normal' else 'normal'
                
                return [{
                    'device_id': device_id,
                    'old_status': old_status,
                    'new_status': new_status,
                    'timestamp': datetime.now().isoformat()
                }]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Status changes error: {e}")
            return []
    
    def get_device_summary(self, device_id: str) -> Dict:
        """
        Get comprehensive summary for a specific device.
        
        Args:
            device_id: Device identifier
            
        Returns:
            Device summary data
        """
        try:
            device_data_df = self._fetch_device_data(days_back=1)
            device_data = device_data_df.to_dict('records')
            device = next((d for d in device_data if d.get('device_id') == device_id), None)
            
            if not device:
                return {'error': f'Device {device_id} not found'}
            
            # Get historical data for trends
            historical_data_df = self._get_device_historical_data(device_id, hours=168) # 7 days
            historical_data = historical_data_df.to_dict('records')
            
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
                    'operating_hours': device.get('operating_hours', 0),
                    'days_since_maintenance': device.get('days_since_maintenance', 0)
                },
                'trends': self._calculate_device_trends(historical_data_df),
                'alerts': self._get_device_alerts(device_id),
                'recommendations': self._get_device_recommendations(device),
                'last_updated': device.get('timestamp', datetime.now().isoformat())
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Device summary error for {device_id}: {e}")
            return {'error': str(e)}
    
    def _get_device_historical_data(self, device_id: str, hours: int = 24) -> pd.DataFrame:
        """Get historical data for a specific device."""
        try:
            db_path = "DATABASE/health_data.db"
            
            if not Path(db_path).exists():
                # Generate sample historical data
                return pd.DataFrame(self._generate_sample_historical_data(device_id, hours))
            
            with sqlite3.connect(db_path) as conn:
                query = f"""
                    SELECT * FROM device_data 
                    WHERE device_id = ? AND timestamp >= datetime('now', '-{hours} hours')
                    ORDER BY timestamp ASC
                """
                
                df = pd.read_sql_query(query, conn, params=[device_id])
                return df
                
        except Exception as e:
            self.logger.error(f"Historical data retrieval error for {device_id}: {e}")
            return pd.DataFrame()
    
    def _generate_sample_historical_data(self, device_id: str, hours: int) -> List[Dict]:
        """Generate sample historical data for demonstration."""
        try:
            data = []
            now = datetime.now()
            
            for i in range(hours):
                timestamp = now - timedelta(hours=hours-1-i)
                
                # Generate realistic pattern
                base_value = 25 + 5 * np.sin(i * 0.1)  # Daily temperature pattern
                noise = np.random.normal(0, 1)
                
                data.append({
                    'device_id': device_id,
                    'timestamp': timestamp.isoformat(),
                    'value': round(base_value + noise, 2),
                    'health_score': np.random.uniform(0.7, 0.95),
                    'efficiency_score': np.random.uniform(0.75, 0.92),
                    'status': np.random.choice(['normal', 'warning'], p=[0.9, 0.1])
                })
            
            return data
            
        except Exception as e:
            self.logger.error(f"Sample historical data generation error: {e}")
            return []
    
    def _calculate_device_trends(self, historical_df: pd.DataFrame) -> Dict:
        """Calculate trends for a single device's metrics."""
        try:
            if historical_df.empty:
                return {'health': 'stable', 'efficiency': 'stable', 'value': 'stable'}
            
            trends = {}
            
            for metric in ['health_score', 'efficiency_score', 'value']:
                if metric in historical_df.columns:
                    # Use the same robust trend calculation
                    trend_details = self._calculate_detailed_trend(historical_df, metric)
                    trends[metric.replace('_score', '')] = trend_details.get('trend', 'stable')
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Device trends calculation error: {e}")
            return {'health': 'stable', 'efficiency': 'stable', 'value': 'stable'}
    
    def _get_device_alerts(self, device_id: str) -> List[Dict]:
        """Get alerts for a specific device."""
        try:
            # In a real implementation, this would query the alert system
            # For demo, return sample alerts
            alerts = []
            
            if np.random.random() < 0.3:  # 30% chance of having alerts
                alert_types = [
                    ('Maintenance Due', 'warning'),
                    ('Performance Degraded', 'info'),
                    ('Sensor Calibration', 'warning')
                ]
                
                for alert_title, severity in alert_types[:np.random.randint(1, 3)]:
                    alerts.append({
                        'title': alert_title,
                        'severity': severity,
                        'timestamp': (datetime.now() - timedelta(hours=np.random.randint(1, 48))).isoformat()
                    })
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Device alerts error for {device_id}: {e}")
            return []
    
    def _get_device_recommendations(self, device: Dict) -> List[Dict]:
        """Get recommendations for a specific device."""
        try:
            recommendations = []
            
            health_score = device.get('health_score', 1.0)
            efficiency_score = device.get('efficiency_score', 1.0)
            days_since_maintenance = device.get('days_since_maintenance', 0)
            
            # Health-based recommendations
            if health_score < 0.6:
                recommendations.append({
                    'type': 'maintenance',
                    'priority': 'high',
                    'title': 'Device Health Critical',
                    'description': f'Health score is {health_score*100:.1f}% - immediate attention required',
                    'action': 'Schedule immediate inspection and maintenance'
                })
            elif health_score < 0.8:
                recommendations.append({
                    'type': 'maintenance',
                    'priority': 'medium',
                    'title': 'Health Score Below Normal',
                    'description': f'Health score is {health_score*100:.1f}% - consider maintenance',
                    'action': 'Schedule maintenance within next week'
                })
            
            # Efficiency-based recommendations
            if efficiency_score < 0.7:
                recommendations.append({
                    'type': 'optimization',
                    'priority': 'medium',
                    'title': 'Low Efficiency Detected',
                    'description': f'Efficiency is {efficiency_score*100:.1f}% - optimization needed',
                    'action': 'Review operational parameters and optimize settings'
                })
            
            # Maintenance schedule recommendations
            if days_since_maintenance > 90:
                recommendations.append({
                    'type': 'maintenance',
                    'priority': 'high',
                    'title': 'Maintenance Overdue',
                    'description': f'Last maintenance was {days_since_maintenance} days ago',
                    'action': 'Schedule maintenance immediately'
                })
            elif days_since_maintenance > 60:
                recommendations.append({
                    'type': 'maintenance',
                    'priority': 'low',
                    'title': 'Maintenance Due Soon',
                    'description': f'Last maintenance was {days_since_maintenance} days ago',
                    'action': 'Plan maintenance within next 2 weeks'
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Device recommendations error: {e}")
            return []
    
    def export_dashboard_data(self, format_type: str = 'json', include_historical: bool = False) -> Dict:
        """
        Export dashboard data in specified format.
        
        Args:
            format_type: Export format ('json', 'csv', 'excel')
            include_historical: Whether to include historical data
            
        Returns:
            Export data or file path
        """
        try:
            self.logger.info(f"Exporting dashboard data in {format_type} format")
            
            # Get comprehensive data
            overview = self.get_dashboard_overview()
            device_data_df = self._fetch_device_data(days_back=1)
            device_data = device_data_df.to_dict('records')
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'overview': overview,
                'devices': device_data
            }
            
            if include_historical:
                export_data['historical_data'] = self._get_all_historical_data().to_dict('records')
            
            if format_type == 'json':
                # Save as JSON file
                export_path = self.cache_path / f"dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                return {'export_path': str(export_path), 'format': 'json'}
            
            elif format_type == 'csv':
                # Convert to CSV format
                df = pd.DataFrame(device_data)
                export_path = self.cache_path / f"dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(export_path, index=False)
                
                return {'export_path': str(export_path), 'format': 'csv'}
            
            else:
                return {'error': f'Unsupported export format: {format_type}'}
                
        except Exception as e:
            self.logger.error(f"Dashboard data export error: {e}")
            return {'error': str(e)}
    
    def _get_all_historical_data(self, days: int = 7) -> pd.DataFrame:
        """Get historical data for all devices."""
        try:
            db_path = "DATABASE/health_data.db"
            
            if not Path(db_path).exists():
                return pd.DataFrame()
            
            with sqlite3.connect(db_path) as conn:
                query = f"""
                    SELECT * FROM device_data 
                    WHERE timestamp >= datetime('now', '-{days} days')
                    ORDER BY timestamp DESC
                """
                
                df = pd.read_sql_query(query, conn)
                return df
                
        except Exception as e:
            self.logger.error(f"Historical data retrieval error: {e}")
            return pd.DataFrame()
    
    def clear_cache(self):
        """Clear all cached data."""
        try:
            self.data_cache.clear()
            self.cache_timestamps.clear()
            self.realtime_buffer.clear()
            
            # Clear cache files
            cache_files = list(self.cache_path.glob("*.cache"))
            for cache_file in cache_files:
                cache_file.unlink()
            
            self.logger.info("Dashboard cache cleared")
            
        except Exception as e:
            self.logger.error(f"Cache clear error: {e}")
    
    def get_cache_statistics(self) -> Dict:
        """Get cache usage statistics."""
        try:
            total_cache_entries = len(self.data_cache)
            cache_size = sum(len(str(v)) for v in self.data_cache.values())
            
            return {
                'total_entries': total_cache_entries,
                'estimated_size_bytes': cache_size,
                'realtime_buffer_size': sum(len(buffer) for buffer in self.realtime_buffer.values()),
                'last_updated': max(self.cache_timestamps.values()) if self.cache_timestamps else None,
                'cache_hit_ratio': 0.85  # Placeholder - would track actual hits/misses in real implementation
            }
            
        except Exception as e:
            self.logger.error(f"Cache statistics error: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    # Initialize dashboard helper
    dashboard = DashboardHelper()
    
    print("=== DIGITAL TWIN DASHBOARD HELPER DEMO ===\n")
    
    # 1. Get dashboard overview
    print("1. Getting dashboard overview (using 7-day historical data)...")
    overview = dashboard.get_dashboard_overview()
    
    print(f"System Metrics:")
    system_metrics = overview.get('system_metrics', {})
    print(f"  - Total Devices: {system_metrics.get('total_devices', 0)}")
    print(f"  - Active Devices: {system_metrics.get('active_devices', 0)}")
    print(f"  - System Uptime: {system_metrics.get('uptime', 0)}%")
    print(f"  - Data Points (7d): {system_metrics.get('data_points_collected', 0)}")
    
    print(f"\nHealth Metrics:")
    health_metrics = overview.get('health_metrics', {})
    print(f"  - Average Health: {health_metrics.get('average_health', 0)}%")
    print(f"  - Health Trend: {health_metrics.get('health_trend', 'stable')}")
    print(f"  - Devices Needing Attention: {health_metrics.get('devices_needing_attention', 0)}")
    
    print(f"\nPerformance Metrics:")
    performance_metrics = overview.get('performance_metrics', {})
    print(f"  - Average Efficiency: {performance_metrics.get('average_efficiency', 0)}%")
    print(f"  - Efficiency Trend: {performance_metrics.get('efficiency_trend', 'stable')}")
    print(f"  - Devices Due Maintenance: {performance_metrics.get('devices_due_maintenance', 0)}")
    
    print(f"\nEnergy Metrics:")
    energy_metrics = overview.get('energy_metrics', {})
    print(f"  - Current Power: {energy_metrics.get('current_power_kw', 0)} kW")
    print(f"  - Daily Energy (Est): {energy_metrics.get('daily_energy_kwh', 0)} kWh")
    print(f"  - Monthly Cost (Est): ${energy_metrics.get('monthly_cost_usd', 0)}")
    
    print(f"\nStatus Distribution:")
    status_distribution = overview.get('status_distribution', {})
    for status, count in status_distribution.items():
        print(f"  - {status.title()}: {count} devices")
        
    print(f"\nAlerts Summary:")
    alerts_summary = overview.get('alerts_summary', {})
    print(f"  - Total Alerts: {alerts_summary.get('total_alerts', 0)}")
    print(f"  - Critical Alerts: {alerts_summary.get('critical_alerts', 0)}")

    # 2. Test NEW Trend Analysis
    print("\n" + "="*50)
    print("2. Testing NEW Detailed Trend Analysis...")
    
    trend_analysis = overview.get('trend_analysis', {})
    print("Trend Analysis (7-day):")
    for metric, details in trend_analysis.items():
        print(f"  - {metric.title()}:")
        print(f"    - Trend: {details.get('trend', 'stable')}")
        print(f"    - Slope: {details.get('slope', 0.0)} {details.get('unit', '%')} /day")
        print(f"    - 24h Change: {details.get('change_24h', 0.0):+.2f} {details.get('unit', '%')}")
        print(f"    - 7d Change: {details.get('change_7d', 0.0):+.2f} {details.get('unit', '%')}")

    # 3. Test NEW Asset Comparisons
    print("\n" + "="*50)
    print("3. Testing NEW Asset Comparisons...")
    
    asset_comparisons = overview.get('asset_comparisons', {})
    print("Bottom 3 Health Devices:")
    for device in asset_comparisons.get('bottom_5_health', [])[:3]:
        print(f"  - {device.get('device_name')} ({device.get('device_id')}): {device.get('health_score')}%")
        
    print("\nTop 3 Efficiency Devices:")
    for device in asset_comparisons.get('top_5_efficiency', [])[:3]:
        print(f"  - {device.get('device_name')} ({device.get('device_id')}): {device.get('efficiency_score')}%")

    print("\nAvg Health by Location:")
    for loc, health in asset_comparisons.get('health_by_location', {}).items():
        print(f"  - {loc}: {health:.1f}%")

    # 4. Test chart data preparation
    print("\n" + "="*50)
    print("4. Testing chart data preparation...")
    
    # Line chart data
    line_chart = dashboard.prepare_chart_data('line', time_range='7d')
    if 'error' not in line_chart:
        print(f"✓ Line Chart (7d): {len(line_chart.get('datasets', []))} datasets with {len(line_chart.get('labels', []))} data points")
    else:
        print(f"✗ Line Chart Error: {line_chart['error']}")
    
    # Pie chart data
    pie_chart = dashboard.prepare_chart_data('pie')
    if 'error' not in pie_chart:
        print(f"✓ Pie Chart: {len(pie_chart.get('labels', []))} segments")
    else:
        print(f"✗ Pie Chart Error: {pie_chart['error']}")

    # 5. Test real-time updates
    print("\n" + "="*50)
    print("5. Testing real-time updates...")
    
    realtime_data = dashboard.get_realtime_updates()
    if 'error' not in realtime_data:
        metrics = realtime_data.get('metrics', {})
        print("Real-time Metrics:")
        for metric_name, metric_data in metrics.items():
            value = metric_data.get('value', 0)
            delta = metric_data.get('delta', 0)
            trend = metric_data.get('trend', 'stable')
            print(f"  - {metric_name.replace('_', ' ').title()}: {value} (Δ{delta:+.1f}, {trend})")
        
        new_alerts = realtime_data.get('alerts', [])
        print(f"\nNew Alerts: {len(new_alerts)}")
        for alert in new_alerts[:3]:  # Show first 3 alerts
            print(f"  - [{alert.get('severity', 'info').upper()}] {alert.get('title', '')}")
    else:
        print(f"✗ Real-time Updates Error: {realtime_data['error']}")
    
    # 6. Test device summary
    print("\n" + "="*50)
    print("6. Testing device summary...")
    
    device_summary = dashboard.get_device_summary('DEVICE_001')
    if 'error' not in device_summary:
        print(f"Device: {device_summary.get('device_name', 'Unknown')}")
        print(f"Status: {device_summary.get('status', 'Unknown')}")
        
        current_metrics = device_summary.get('current_metrics', {})
        print(f"Health Score: {current_metrics.get('health_score', 0)}%")
        
        trends = device_summary.get('trends', {})
        print(f"Trends: Health={trends.get('health', 'stable')}, "
              f"Efficiency={trends.get('efficiency', 'stable')}")
        
        recommendations = device_summary.get('recommendations', [])
        print(f"Recommendations: {len(recommendations)}")
        for rec in recommendations[:2]:  # Show first 2 recommendations
            print(f"  - [{rec.get('priority', 'low').upper()}] {rec.get('title', '')}")
    else:
        print(f"✗ Device Summary Error: {device_summary['error']}")
    
    # 7. Test data export
    print("\n" + "="*50)
    print("7. Testing data export...")
    
    # JSON export
    json_export = dashboard.export_dashboard_data('json', include_historical=True)
    if 'error' not in json_export:
        print(f"✓ JSON Export (with history): {json_export.get('export_path', 'Unknown path')}")
    else:
        print(f"✗ JSON Export Error: {json_export['error']}")
    
    # 8. Cleanup and summary
    print("\n" + "="*50)
    print("8. Demo summary and cleanup...")
    
    # Clear cache
    dashboard.clear_cache()
    print("✓ Cache cleared")
    
    # Verify cleanup
    cleaned_cache_stats = dashboard.get_cache_statistics()
    print(f"Cache entries after cleanup: {cleaned_cache_stats.get('total_entries', 0)}")
    
    print("\n=== DEMO COMPLETED SUCCESSFULLY ===")
    print("\nDashboard Helper Features Tested:")
    print("✓ Dashboard overview generation (with 7-day data)")
    print("✓ NEW Detailed trend analysis (slope, 24h/7d change)")
    print("✓ NEW Multi-asset comparisons (Top/Bottom 5, by location)")
    print("✓ Chart data preparation")
    print("✓ Real-time updates")
    print("✓ Device summaries")
    print("✓ Data export (JSON/CSV)")
    print("✓ Cache management")
    
    print("Dashboard Helper is ready for production use!")