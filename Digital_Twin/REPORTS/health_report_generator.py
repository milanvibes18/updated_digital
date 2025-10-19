#!/usr/bin/env python3
"""
Health Report Generator for Digital Twin System
Generates comprehensive HTML and PDF reports for system health analysis,
including data-driven predictions and actionable recommendations.
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from AI_MODULES.health_score import HealthScoreCalculator
    from AI_MODULES.secure_database_manager import SecureDatabaseManager
    from CONFIG.app_config import config
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")

class HealthReportGenerator:
    """
    Comprehensive health report generator for Digital Twin system.
    Generates detailed HTML and PDF reports with charts and analytics.
    """
    
    def __init__(self, output_dir: str = "REPORTS/generated"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logging()
        
        # Initialize components
        try:
            self.health_calculator = HealthScoreCalculator()
            self.db_manager = SecureDatabaseManager()
        except Exception as e:
            self.logger.warning(f"Some components not available: {e}")
            self.health_calculator = None
            self.db_manager = None
        
        # Report configuration
        self.report_config = {
            'company_name': 'Digital Twin Industries',
            'system_name': 'Industrial IoT Monitoring System',
            'version': '2.0.0',
            'logo_path': None,  # Add your logo path here
            'color_scheme': {
                'primary': '#2c3e50',
                'secondary': '#3498db',
                'success': '#27ae60',
                'warning': '#f39c12',
                'danger': '#e74c3c',
                'info': '#17a2b8'
            }
        }
        
        # Matplotlib setup
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def _setup_logging(self):
        """Setup logging for report generator"""
        logger = logging.getLogger('HealthReportGenerator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Ensure LOGS directory exists
            Path("LOGS").mkdir(exist_ok=True)
            handler = logging.FileHandler('LOGS/health_reports.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def generate_comprehensive_report(self, 
                                      device_ids: List[str] = None,
                                      date_range_days: int = 30,
                                      include_predictions: bool = True,
                                      include_recommendations: bool = True) -> str:
        """
        Generate a comprehensive health report.
        
        Args:
            device_ids: Specific devices to include (None for all)
            date_range_days: Number of days to analyze
            include_predictions: Include predictive analysis
            include_recommendations: Include AI recommendations
            
        Returns:
            Path to generated report file
        """
        try:
            self.logger.info("Starting comprehensive health report generation")
            
            # Collect data
            report_data = self._collect_report_data(device_ids, date_range_days)
            
            # Generate charts
            charts = self._generate_charts(report_data)
            
            # Calculate health scores
            health_analysis = self._analyze_health_scores(report_data)
            
            # Generate predictions if requested
            predictions = {}
            if include_predictions:
                # Pass health_analysis to be used for predictions
                predictions = self._generate_predictions(report_data, health_analysis)
            
            # Generate recommendations if requested
            recommendations = {}
            if include_recommendations:
                # Pass health_analysis and new predictions
                recommendations = self._generate_recommendations(report_data, health_analysis, predictions)
            
            # Compile report data
            full_report_data = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'report_type': 'Comprehensive Health Report',
                    'date_range_days': date_range_days,
                    'devices_analyzed': len(set(d['device_id'] for d in report_data.get('devices', []))),
                    'version': self.report_config['version']
                },
                'summary': self._generate_executive_summary(report_data, health_analysis),
                'data': report_data,
                'health_analysis': health_analysis,
                'charts': charts,
                'predictions': predictions,
                'recommendations': recommendations,
                'config': self.report_config
            }
            
            # Generate HTML report
            html_path = self._generate_html_report(full_report_data)
            
            # Generate PDF if possible
            try:
                pdf_path = self._generate_pdf_report(html_path)
                self.logger.info(f"PDF report generated: {pdf_path}")
            except Exception as e:
                self.logger.warning(f"PDF generation failed: {e}")
                pdf_path = None
            
            self.logger.info(f"Comprehensive report generated: {html_path}")
            return html_path
            
        except Exception as e:
            self.logger.error(f"Report generation error: {e}")
            raise
    
    def _collect_report_data(self, device_ids: List[str], date_range_days: int) -> Dict:
        """Collect all necessary data for the report"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=date_range_days)
            
            # Try to get data from database
            data = self._get_database_data(start_date, end_date, device_ids)
            
            # If no database data, generate sample data
            if not data or len(data.get('devices', [])) == 0:
                data = self._generate_sample_data(date_range_days)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Data collection error: {e}")
            return self._generate_sample_data(date_range_days)
    
    def _get_database_data(self, start_date: datetime, end_date: datetime, device_ids: List[str]) -> Dict:
        """Get data from the database"""
        try:
            db_path = "DATABASE/health_data.db"
            if not os.path.exists(db_path):
                return {}
            
            with sqlite3.connect(db_path) as conn:
                # Device data query
                device_query = """
                    SELECT * FROM device_data 
                    WHERE timestamp >= ? AND timestamp <= ?
                """
                params = [start_date.isoformat(), end_date.isoformat()]
                
                if device_ids:
                    device_query += f" AND device_id IN ({','.join(['?' for _ in device_ids])})"
                    params.extend(device_ids)
                
                device_data = pd.read_sql_query(device_query, conn, params=params)
                
                # System metrics if available
                try:
                    metrics_query = "SELECT * FROM system_metrics WHERE timestamp >= ? AND timestamp <= ?"
                    system_metrics = pd.read_sql_query(metrics_query, conn, params=[start_date.isoformat(), end_date.isoformat()])
                except:
                    system_metrics = pd.DataFrame()
                
                # Energy data if available
                try:
                    energy_query = "SELECT * FROM energy_data WHERE timestamp >= ? AND timestamp <= ?"
                    energy_data = pd.read_sql_query(energy_query, conn, params=[start_date.isoformat(), end_date.isoformat()])
                except:
                    energy_data = pd.DataFrame()
                
                return {
                    'devices': device_data.to_dict('records') if not device_data.empty else [],
                    'system_metrics': system_metrics.to_dict('records') if not system_metrics.empty else [],
                    'energy_data': energy_data.to_dict('records') if not energy_data.empty else [],
                    'date_range': {
                        'start': start_date.isoformat(),
                        'end': end_date.isoformat()
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Database query error: {e}")
            return {}
    
    def _generate_sample_data(self, date_range_days: int) -> Dict:
        """Generate sample data for demonstration"""
        try:
            self.logger.info("Generating sample data for report")
            
            # Generate time series
            timestamps = pd.date_range(
                datetime.now() - timedelta(days=date_range_days),
                datetime.now(),
                freq='H'
            )
            
            # Sample devices
            devices = []
            device_names = ['Temperature Sensor 01', 'Pressure Sensor 01', 'Vibration Sensor 01', 
                            'Humidity Sensor 01', 'Power Meter 01']
            
            for i, device_name in enumerate(device_names):
                health = 0.9 - (i * 0.1) # Start with different healths
                efficiency = 0.95 - (i * 0.05)
                
                for timestamp in timestamps[::6]:  # Every 6 hours
                    # Introduce a declining trend for device 3
                    if 'Vibration' in device_name:
                        health -= 0.005
                    
                    # Introduce a slight improvement for device 1
                    if 'Temperature' in device_name:
                         health += 0.001
                         
                    health = max(0.2, min(1.0, health + np.random.normal(0, 0.02)))
                    efficiency = max(0.3, min(1.0, efficiency + np.random.normal(0, 0.01)))
                    
                    device_data = {
                        'device_id': f'DEVICE_{i+1:03d}',
                        'device_name': device_name,
                        'device_type': device_name.split()[0].lower() + '_sensor',
                        'timestamp': timestamp.isoformat(),
                        'value': 50 + 20 * np.sin(i * 0.1) + np.random.normal(0, 5),
                        'health_score': health,
                        'efficiency_score': efficiency,
                        'status': 'critical' if health < 0.6 else ('warning' if health < 0.75 else 'normal'),
                        'location': f'Floor {(i % 3) + 1}',
                        'unit': '°C' if 'temperature' in device_name.lower() else 'units'
                    }
                    devices.append(device_data)
            
            # Sample system metrics
            system_metrics = []
            disk = 70.0
            for timestamp in timestamps[::12]:  # Every 12 hours
                disk = min(95, disk + np.random.uniform(0.1, 0.5)) # Disk usage creeping up
                system_metrics.append({
                    'timestamp': timestamp.isoformat(),
                    'cpu_usage_percent': np.random.uniform(20, 80),
                    'memory_usage_percent': np.random.uniform(40, 85),
                    'disk_usage_percent': disk,
                    'network_io_mbps': np.random.uniform(10, 100),
                    'active_connections': np.random.randint(50, 200)
                })
            
            # Sample energy data
            energy_data = []
            cumulative_energy = 0
            for timestamp in timestamps[::4]:  # Every 4 hours
                power_consumption = np.random.uniform(800, 1500)
                cumulative_energy += power_consumption * 4  # 4 hours
                energy_data.append({
                    'timestamp': timestamp.isoformat(),
                    'power_consumption_kw': power_consumption,
                    'energy_consumed_kwh': cumulative_energy,
                    'efficiency_percent': np.random.uniform(80, 95), # Fluctuate efficiency
                    'cost_usd': cumulative_energy * 0.12
                })
            
            return {
                'devices': devices,
                'system_metrics': system_metrics,
                'energy_data': energy_data,
                'date_range': {
                    'start': (datetime.now() - timedelta(days=date_range_days)).isoformat(),
                    'end': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Sample data generation error: {e}")
            return {}
    
    def _analyze_health_scores(self, data: Dict) -> Dict:
        """Analyze health scores and trends"""
        try:
            devices_data = data.get('devices', [])
            if not devices_data:
                return {}
            
            df = pd.DataFrame(devices_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            analysis = {
                'overall_health': {
                    'current': df['health_score'].mean() * 100,
                    'trend': self._calculate_trend(df.groupby(pd.Grouper(key='timestamp', freq='D'))['health_score'].mean()),
                    'distribution': self._calculate_health_distribution(df)
                },
                'device_analysis': {},
                'critical_devices': [],
                'top_performers': [],
                'health_trends': {}
            }
            
            # Analyze each device
            for device_id in df['device_id'].unique():
                device_df = df[df['device_id'] == device_id].sort_values('timestamp')
                
                if len(device_df) > 0:
                    latest_health = device_df['health_score'].iloc[-1] * 100
                    health_trend = self._calculate_trend(device_df['health_score'])
                    
                    device_analysis = {
                        'current_health': latest_health,
                        'average_health': device_df['health_score'].mean() * 100,
                        'trend': health_trend,
                        'status': self._determine_health_status(latest_health),
                        'data_points': len(device_df),
                        'device_name': device_df['device_name'].iloc[0]
                    }
                    
                    analysis['device_analysis'][device_id] = device_analysis
                    
                    # Identify critical devices
                    if latest_health < 60:
                        analysis['critical_devices'].append({
                            'device_id': device_id,
                            'device_name': device_df['device_name'].iloc[0],
                            'health_score': latest_health,
                            'trend': health_trend
                        })
                    
                    # Identify top performers
                    if latest_health > 90:
                        analysis['top_performers'].append({
                            'device_id': device_id,
                            'device_name': device_df['device_name'].iloc[0],
                            'health_score': latest_health,
                            'trend': health_trend
                        })
            
            # Sort lists
            analysis['critical_devices'].sort(key=lambda x: x['health_score'])
            analysis['top_performers'].sort(key=lambda x: x['health_score'], reverse=True)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Health score analysis error: {e}")
            return {}
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction from a time series"""
        try:
            series = series.dropna()
            if len(series) < 3:
                return 'stable'
            
            # Linear regression for trend
            x = np.arange(len(series))
            coeffs = np.polyfit(x, series.values, 1)
            slope = coeffs[0]
            
            # Normalize slope by the mean to get a relative sense of change
            normalized_slope = slope / (series.mean() + 1e-6)
            
            if normalized_slope > 0.05:
                return 'improving'
            elif normalized_slope < -0.05:
                return 'declining'
            else:
                return 'stable'
                
        except Exception:
            return 'unknown'
    
    def _calculate_health_distribution(self, df: pd.DataFrame) -> Dict:
        """Calculate health score distribution"""
        try:
            latest_scores = df.groupby('device_id')['health_score'].last() * 100
            
            return {
                'excellent': int((latest_scores >= 90).sum()),
                'good': int(((latest_scores >= 75) & (latest_scores < 90)).sum()),
                'fair': int(((latest_scores >= 60) & (latest_scores < 75)).sum()),
                'poor': int((latest_scores < 60).sum())
            }
            
        except Exception:
            return {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
    
    def _determine_health_status(self, health_score: float) -> str:
        """Determine health status from score"""
        if health_score >= 90:
            return 'excellent'
        elif health_score >= 75:
            return 'good'
        elif health_score >= 60:
            return 'fair'
        else:
            return 'poor'
    
    def _generate_charts(self, data: Dict) -> Dict:
        """Generate all charts for the report"""
        try:
            charts = {}
            
            # Health trends chart
            charts['health_trends'] = self._create_health_trends_chart(data)
            
            # Device status distribution
            charts['status_distribution'] = self._create_status_distribution_chart(data)
            
            # System performance chart
            if data.get('system_metrics'):
                charts['system_performance'] = self._create_system_performance_chart(data)
            
            # Energy consumption chart
            if data.get('energy_data'):
                charts['energy_consumption'] = self._create_energy_consumption_chart(data)
            
            # Device comparison chart
            charts['device_comparison'] = self._create_device_comparison_chart(data)
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Chart generation error: {e}")
            return {}
    
    def _create_health_trends_chart(self, data: Dict) -> str:
        """Create health trends over time chart"""
        try:
            devices_data = data.get('devices', [])
            if not devices_data:
                return ""
            
            df = pd.DataFrame(devices_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            plt.figure(figsize=(12, 6))
            
            # Plot health trends for all unique devices (up to 10)
            device_ids = df['device_id'].unique()
            
            for device_id in device_ids[:10]:
                device_data = df[df['device_id'] == device_id].sort_values('timestamp')
                plt.plot(device_data['timestamp'], device_data['health_score'] * 100, 
                         marker='o', markersize=4, linestyle='-', label=device_data['device_name'].iloc[0])
            
            plt.title('Device Health Trends Over Time', fontsize=16, fontweight='bold')
            plt.xlabel('Time')
            plt.ylabel('Health Score (%)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save to base64
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            img_data = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            
            return f"data:image/png;base64,{img_data}"
            
        except Exception as e:
            self.logger.error(f"Health trends chart error: {e}")
            return ""
    
    def _create_status_distribution_chart(self, data: Dict) -> str:
        """Create device status distribution pie chart"""
        try:
            devices_data = data.get('devices', [])
            if not devices_data:
                return ""
            
            df = pd.DataFrame(devices_data)
            
            # Get latest status for each device
            latest_scores = df.loc[df.groupby('device_id')['timestamp'].idxmax()]
            latest_scores['status_label'] = latest_scores['health_score'].apply(lambda x: self._determine_health_status(x * 100))
            status_counts = latest_scores['status_label'].value_counts()
            
            labels = status_counts.index
            sizes = status_counts.values
            
            color_map = {
                'excellent': '#27ae60',
                'good': '#3498db',
                'fair': '#f39c12',
                'poor': '#e74c3c'
            }
            colors = [color_map.get(label, '#bdc3c7') for label in labels]
            
            plt.figure(figsize=(8, 8))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', 
                    colors=colors, startangle=90, pctdistance=0.85)
            
            # Draw a circle at the center to make it a donut chart
            centre_circle = plt.Circle((0,0),0.70,fc='white')
            fig = plt.gcf()
            fig.gca().add_artist(centre_circle)
            
            plt.title('Device Health Distribution', fontsize=16, fontweight='bold')
            plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
            
            # Save to base64
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            img_data = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            
            return f"data:image/png;base64,{img_data}"
            
        except Exception as e:
            self.logger.error(f"Status distribution chart error: {e}")
            return ""
    
    def _create_system_performance_chart(self, data: Dict) -> str:
        """Create system performance metrics chart"""
        try:
            metrics_data = data.get('system_metrics', [])
            if not metrics_data:
                return ""
            
            df = pd.DataFrame(metrics_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('System Performance Metrics Over Time', fontsize=16, fontweight='bold')
            
            # CPU Usage
            axes[0,0].plot(df['timestamp'], df['cpu_usage_percent'], color='#3498db', linewidth=2)
            axes[0,0].set_title('CPU Usage (%)')
            axes[0,0].grid(True, alpha=0.3)
            
            # Memory Usage
            axes[0,1].plot(df['timestamp'], df['memory_usage_percent'], color='#e74c3c', linewidth=2)
            axes[0,1].set_title('Memory Usage (%)')
            axes[0,1].grid(True, alpha=0.3)
            
            # Disk Usage
            axes[1,0].plot(df['timestamp'], df['disk_usage_percent'], color='#f39c12', linewidth=2)
            axes[1,0].set_title('Disk Usage (%)')
            axes[1,0].grid(True, alpha=0.3)
            
            # Network I/O
            axes[1,1].plot(df['timestamp'], df['network_io_mbps'], color='#27ae60', linewidth=2)
            axes[1,1].set_title('Network I/O (Mbps)')
            axes[1,1].grid(True, alpha=0.3)
            
            for ax_row in axes:
                for ax in ax_row:
                    ax.tick_params(axis='x', rotation=30)
                    
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save to base64
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            img_data = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            
            return f"data:image/png;base64,{img_data}"
            
        except Exception as e:
            self.logger.error(f"System performance chart error: {e}")
            return ""
    
    def _create_energy_consumption_chart(self, data: Dict) -> str:
        """Create energy consumption chart"""
        try:
            energy_data = data.get('energy_data', [])
            if not energy_data:
                return ""
            
            df = pd.DataFrame(energy_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            fig, (ax1) = plt.subplots(1, 1, figsize=(12, 6))
            
            # Power consumption
            ax1.plot(df['timestamp'], df['power_consumption_kw'], color='#9b59b6', linewidth=2)
            ax1.set_title('Power Consumption Over Time')
            ax1.set_ylabel('Power (kW)', color='#9b59b6')
            ax1.tick_params(axis='y', labelcolor='#9b59b6')
            ax1.grid(True, alpha=0.3)
            
            # Energy efficiency
            ax2 = ax1.twinx()
            ax2.plot(df['timestamp'], df['efficiency_percent'], color='#16a085', linestyle='--', linewidth=2)
            ax2.set_ylabel('Efficiency (%)', color='#16a085')
            ax2.tick_params(axis='y', labelcolor='#16a085')
            ax2.set_ylim(0, 100)

            fig.suptitle('Energy Consumption and Efficiency', fontsize=16, fontweight='bold')
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save to base64
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            img_data = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            
            return f"data:image/png;base64,{img_data}"
            
        except Exception as e:
            self.logger.error(f"Energy consumption chart error: {e}")
            return ""
    
    def _create_device_comparison_chart(self, data: Dict) -> str:
        """Create device performance comparison chart"""
        try:
            devices_data = data.get('devices', [])
            if not devices_data:
                return ""
            
            df = pd.DataFrame(devices_data)
            
            # Get average scores for each device
            device_summary = df.groupby(['device_id', 'device_name']).agg({
                'health_score': 'mean',
                'efficiency_score': 'mean'
            }).reset_index()
            
            device_summary['health_score'] *= 100
            device_summary['efficiency_score'] *= 100
            device_summary = device_summary.sort_values('health_score', ascending=False)
            
            plt.figure(figsize=(12, 8))
            
            x = np.arange(len(device_summary))
            width = 0.35
            
            plt.bar(x - width/2, device_summary['health_score'], width, 
                    label='Avg. Health Score', color='#3498db', alpha=0.8)
            plt.bar(x + width/2, device_summary['efficiency_score'], width, 
                    label='Avg. Efficiency Score', color='#27ae60', alpha=0.8)
            
            plt.xlabel('Devices')
            plt.ylabel('Score (%)')
            plt.title('Device Performance Comparison (Average)', fontsize=16, fontweight='bold')
            plt.xticks(x, [f"{name}\n({id})" for id, name in zip(device_summary['device_id'], device_summary['device_name'])], 
                       rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            # Save to base64
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            img_data = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            
            return f"data:image/png;base64,{img_data}"
            
        except Exception as e:
            self.logger.error(f"Device comparison chart error: {e}")
            return ""
    
    def _generate_executive_summary(self, data: Dict, health_analysis: Dict) -> Dict:
        """Generate executive summary"""
        try:
            devices_data = data.get('devices', [])
            total_devices = len(set(d['device_id'] for d in devices_data)) if devices_data else 0
            
            overall_health = health_analysis.get('overall_health', {})
            current_health = overall_health.get('current', 0)
            health_trend = overall_health.get('trend', 'unknown')
            
            # Calculate key metrics
            critical_count = len(health_analysis.get('critical_devices', []))
            top_performers_count = len(health_analysis.get('top_performers', []))
            
            # Energy metrics
            energy_data = data.get('energy_data', [])
            total_energy = 0
            total_cost = 0
            if energy_data:
                total_energy = energy_data[-1].get('energy_consumed_kwh', 0)
                total_cost = energy_data[-1].get('cost_usd', 0)
            
            # Status indicators
            if current_health >= 90:
                health_status = "Excellent"
                health_color = "success"
            elif current_health >= 75:
                health_status = "Good"
                health_color = "info"
            elif current_health >= 60:
                health_status = "Fair"
                health_color = "warning"
            else:
                health_status = "Needs Attention"
                health_color = "danger"
            
            return {
                'overall_health': {
                    'score': round(current_health, 1),
                    'status': health_status,
                    'trend': health_trend,
                    'color': health_color
                },
                'key_metrics': {
                    'total_devices': total_devices,
                    'critical_devices': critical_count,
                    'top_performers': top_performers_count,
                    'total_energy_kwh': round(total_energy, 1),
                    'total_cost_usd': round(total_cost, 2)
                },
                'recommendations': self._get_summary_recommendations(health_analysis),
                'alerts': {
                    'critical': critical_count,
                    'total': critical_count  # Simplified for now
                }
            }
            
        except Exception as e:
            self.logger.error(f"Executive summary generation error: {e}")
            return {}
    
    def _get_summary_recommendations(self, health_analysis: Dict) -> List[str]:
        """Get key recommendations for the summary"""
        recommendations = []
        
        critical_devices = health_analysis.get('critical_devices', [])
        if critical_devices:
            recommendations.append(f"Immediate attention required for {len(critical_devices)} critical devices")
        
        overall_trend = health_analysis.get('overall_health', {}).get('trend', 'unknown')
        if overall_trend == 'declining':
            recommendations.append("System health shows declining trend - investigate root causes")
        elif overall_trend == 'improving':
            recommendations.append("System health is improving - maintain current practices")
        
        if not recommendations:
            recommendations.append("System performance is within acceptable parameters")
        
        return recommendations
    
    def _generate_predictions(self, data: Dict, health_analysis: Dict) -> Dict:
        """Generate predictive analysis based on trends."""
        self.logger.info("Generating predictive analysis")
        predictions = {
            'health_forecast': {},
            'maintenance_predictions': []
        }
        
        devices_data = data.get('devices', [])
        if not devices_data:
            return predictions

        df = pd.DataFrame(devices_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Convert timestamps to numerical values (e.g., hours from start) for regression
        df['time_numeric'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 3600.0
        
        # 1. Overall Health Forecast
        try:
            overall_trend = health_analysis.get('overall_health', {}).get('trend', 'stable')
            overall_score = health_analysis.get('overall_health', {}).get('current', 75)
            
            if overall_trend == 'declining':
                forecast_7 = f"Likely to decline further. Current score: {overall_score:.1f}%"
                forecast_30 = "Potential for significant decline if unaddressed."
            elif overall_trend == 'improving':
                forecast_7 = f"Likely to continue improving. Current score: {overall_score:.1f}%"
                forecast_30 = "Positive outlook."
            else:
                forecast_7 = f"Expected to remain stable around {overall_score:.1f}%."
                forecast_30 = "Stable outlook."
            
            predictions['health_forecast'] = {
                'next_7_days': forecast_7,
                'next_30_days': forecast_30,
                'confidence': 0.80 # Simplified confidence
            }
        except Exception as e:
            self.logger.warning(f"Could not generate overall health forecast: {e}")

        # 2. Predictive Maintenance
        critical_threshold = 0.5 # Health score of 50%
        
        for device_id, analysis in health_analysis.get('device_analysis', {}).items():
            if analysis['trend'] == 'declining':
                device_df = df[df['device_id'] == device_id].sort_values('time_numeric')
                
                if len(device_df) < 3: # Not enough data for regression
                    continue
                    
                try:
                    # Perform linear regression
                    x = device_df['time_numeric']
                    y = device_df['health_score']
                    coeffs = np.polyfit(x, y, 1)
                    slope = coeffs[0]
                    intercept = coeffs[1]
                    
                    if slope < -0.001: # Ensure it's a meaningful decline
                        current_time = x.iloc[-1]
                        
                        # Calculate time (in hours) until health hits critical_threshold
                        # y = mx + c => x = (y - c) / m
                        time_to_critical = (critical_threshold - intercept) / slope
                        
                        hours_until_maintenance = time_to_critical - current_time
                        days_until_maintenance = hours_until_maintenance / 24.0
                        
                        if 0 < days_until_maintenance <= 60: # Only report for next 60 days
                            predictions['maintenance_predictions'].append({
                                'device_id': device_id,
                                'device_name': analysis['device_name'], # Add device name
                                'days_until_maintenance': round(days_until_maintenance, 1),
                                'confidence': 0.85 # Simplified
                            })
                            
                except Exception as e:
                    self.logger.warning(f"Could not predict maintenance for {device_id}: {e}")
        
        # Sort predictions by urgency
        predictions['maintenance_predictions'].sort(key=lambda x: x['days_until_maintenance'])
        
        return predictions
    
    def _generate_recommendations(self, data: Dict, health_analysis: Dict, predictions: Dict) -> Dict:
        """Generate AI-powered recommendations based on analysis and predictions."""
        self.logger.info("Generating data-driven recommendations")
        recommendations = {
            'immediate_actions': [],
            'short_term_actions': [],
            'long_term_actions': []
        }
        
        # 1. Immediate Actions (from critical devices and urgent predictions)
        critical_devices = health_analysis.get('critical_devices', [])
        predicted_maintenance = {p['device_id']: p for p in predictions.get('maintenance_predictions', [])}
        
        for device in critical_devices:
            recommendations['immediate_actions'].append({
                'priority': 'Critical',
                'action': f"Inspect {device['device_name']} ({device['device_id']}). Current health is critical at {device['health_score']:.1f}%.",
                'timeline': 'Within 24 hours'
            })
            if device['device_id'] in predicted_maintenance:
                del predicted_maintenance[device['device_id']] # Avoid duplication
        
        # Add predictions that are urgent but not yet in 'critical' list
        for device_id, pred in predicted_maintenance.items():
            if pred['days_until_maintenance'] <= 7:
                recommendations['immediate_actions'].append({
                    'priority': 'Critical',
                    'action': f"Schedule urgent maintenance for {pred['device_name']} ({device_id}). Predicted to fail within {pred['days_until_maintenance']} days.",
                    'timeline': 'Within 48 hours'
                })

        # 2. Short-term Actions (declining trends, 'fair' status)
        for device_id, analysis in health_analysis.get('device_analysis', {}).items():
            if analysis['status'] == 'fair' and device_id not in predicted_maintenance:
                recommendations['short_term_actions'].append({
                    'priority': 'High',
                    'action': f"Monitor {analysis['device_name']} ({device_id}). Health is 'fair' at {analysis['current_health']:.1f}%.",
                    'timeline': 'Within 1 week'
                })
            elif analysis['trend'] == 'declining' and device_id not in predicted_maintenance and analysis['status'] != 'poor':
                    recommendations['short_term_actions'].append({
                    'priority': 'High',
                    'action': f"Investigate {analysis['device_name']} ({device_id}). Health is declining but not yet critical.",
                    'timeline': 'Within 2 weeks'
                })
        
        # Check system metrics
        if data.get('system_metrics'):
            sys_df = pd.DataFrame(data['system_metrics'])
            if not sys_df.empty:
                avg_disk = sys_df['disk_usage_percent'].mean()
                if avg_disk > 85:
                    recommendations['short_term_actions'].append({
                        'priority': 'Medium',
                        'action': f"Archive old data or expand storage. Average disk usage is high at {avg_disk:.1f}%.",
                        'timeline': 'Within 1 month'
                    })

        # 3. Long-term Actions (energy, top performers)
        if data.get('energy_data'):
            energy_df = pd.DataFrame(data['energy_data'])
            if not energy_df.empty:
                avg_efficiency = energy_df['efficiency_percent'].mean()
                if avg_efficiency < 85:
                    recommendations['long_term_actions'].append({
                        'priority': 'Medium',
                        'action': f"Review system for energy optimization. Average energy efficiency is {avg_efficiency:.1f}%.",
                        'timeline': 'Within 3 months'
                    })
        
        if health_analysis.get('top_performers'):
            recommendations['long_term_actions'].append({
                'priority': 'Low',
                'action': "Analyze top-performing devices to establish benchmarks and best-practice maintenance schedules.",
                'timeline': 'Within 6 months'
            })

        if not any(recommendations.values()):
            recommendations['short_term_actions'].append({
                'priority': 'Info',
                'action': 'System operating normally. Continue routine monitoring.',
                'timeline': 'Ongoing'
            })
            
        return recommendations
    
    def _generate_html_report(self, report_data: Dict) -> str:
        """Generate HTML report from template"""
        try:
            # HTML template
            html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Digital Twin Health Report</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 20px;
            background: #f8f9fa;
            color: #333;
        }
        .header {
            background: linear-gradient(135deg, {{ config.color_scheme.primary }}, {{ config.color_scheme.secondary }});
            color: white;
            padding: 30px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        .header h1 {
            margin: 0;
            font-size: 2.5rem;
        }
        .header h2 {
            margin: 10px 0 0;
            font-weight: 300;
        }
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 5px solid {{ config.color_scheme.secondary }};
        }
        .card.success { border-left-color: {{ config.color_scheme.success }}; }
        .card.warning { border-left-color: {{ config.color_scheme.warning }}; }
        .card.danger { border-left-color: {{ config.color_scheme.danger }}; }
        .card.info { border-left-color: {{ config.color_scheme.info }}; }
        
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .chart-container img {
            width: 100%;
            height: auto;
        }
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: {{ config.color_scheme.primary }};
        }
        .metric-label {
            color: #7f8c8d;
            font-size: 0.9rem;
            text-transform: uppercase;
        }
        .recommendations {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .recommendation-item {
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid {{ config.color_scheme.secondary }};
            background: #f8f9fa;
        }
        .priority-critical { border-left-color: {{ config.color_scheme.danger }}; }
        .priority-high { border-left-color: {{ config.color_scheme.warning }}; }
        .priority-medium { border-left-color: {{ config.color_scheme.secondary }}; }
        .priority-low { border-left-color: {{ config.color_scheme.info }}; }
        .priority-info { border-left-color: #bdc3c7; }
        
        .device-table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .device-table th, .device-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        .device-table th {
            background: {{ config.color_scheme.secondary }};
            color: white;
        }
        .status-excellent { background: #d4edda; color: #155724; }
        .status-good { background: #d1ecf1; color: #0c5460; }
        .status-fair { background: #fff3cd; color: #856404; }
        .status-poor { background: #f8d7da; color: #721c24; }
        .footer {
            text-align: center;
            margin-top: 50px;
            color: #7f8c8d;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ config.system_name }}</h1>
        <h2>{{ metadata.report_type }}</h2>
        <p>Generated on {{ metadata.generated_at | format_datetime }}</p>
        <p>Analysis Period: {{ metadata.date_range_days }} days | Devices: {{ metadata.devices_analyzed }}</p>
    </div>
    
    <div class="summary-cards">
        <div class="card {{ summary.overall_health.color }}">
            <div class="metric-value">{{ summary.overall_health.score }}%</div>
            <div class="metric-label">Overall Health</div>
            <div>Status: {{ summary.overall_health.status }}</div>
            <div>Trend: {{ summary.overall_health.trend | title }}</div>
        </div>
        
        <div class="card">
            <div class="metric-value">{{ summary.key_metrics.total_devices }}</div>
            <div class="metric-label">Total Devices</div>
            <div>Critical: {{ summary.key_metrics.critical_devices }}</div>
            <div>Top Performers: {{ summary.key_metrics.top_performers }}</div>
        </div>
        
        <div class="card">
            <div class="metric-value">{{ summary.key_metrics.total_energy_kwh }}</div>
            <div class="metric-label">Energy Consumed (kWh)</div>
            <div>Cost: ${{ summary.key_metrics.total_cost_usd }}</div>
        </div>
        
        <div class="card {% if summary.key_metrics.critical_devices > 0 %}danger{% else %}success{% endif %}">
            <div class="metric-value">{{ summary.alerts.critical }}</div>
            <div class="metric-label">Critical Alerts</div>
            <div>Total: {{ summary.alerts.total }}</div>
        </div>
    </div>
    
    {% if charts.health_trends %}
    <div class="chart-container">
        <h3>Health Trends Over Time</h3>
        <img src="{{ charts.health_trends }}" alt="Health Trends Chart">
    </div>
    {% endif %}
    
    {% if charts.status_distribution %}
    <div class="chart-container">
        <h3>Device Health Distribution</h3>
        <img src="{{ charts.status_distribution }}" alt="Status Distribution Chart">
    </div>
    {% endif %}
    
    {% if charts.device_comparison %}
    <div class="chart-container">
        <h3>Device Performance Comparison</h3>
        <img src="{{ charts.device_comparison }}" alt="Device Comparison Chart">
    </div>
    {% endif %}
    
    {% if charts.system_performance %}
    <div class="chart-container">
        <h3>System Performance Metrics</h3>
        <img src="{{ charts.system_performance }}" alt="System Performance Chart">
    </div>
    {% endif %}
    
    {% if charts.energy_consumption %}
    <div class="chart-container">
        <h3>Energy Consumption Analysis</h3>
        <img src="{{ charts.energy_consumption }}" alt="Energy Consumption Chart">
    </div>
    {% endif %}
    
    <div class="chart-container">
        <h3>Device Health Analysis</h3>
        <table class="device-table">
            <thead>
                <tr>
                    <th>Device ID</th>
                    <th>Device Name</th>
                    <th>Current Health</th>
                    <th>Average Health</th>
                    <th>Status</th>
                    <th>Trend</th>
                </tr>
            </thead>
            <tbody>
                {% for device_id, analysis in health_analysis.device_analysis.items() %}
                <tr>
                    <td>{{ device_id }}</td>
                    <td>{{ analysis.device_name }}</td>
                    <td class="status-{{ analysis.status }}">{{ "%.1f"|format(analysis.current_health) }}%</td>
                    <td>{{ "%.1f"|format(analysis.average_health) }}%</td>
                    <td class="status-{{ analysis.status }}">{{ analysis.status.title() }}</td>
                    <td>{{ analysis.trend.title() }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    
    <div class="recommendations">
        <h3>AI-Powered Recommendations</h3>
        
        {% if recommendations.immediate_actions %}
        <h4>Immediate Actions Required</h4>
        {% for action in recommendations.immediate_actions %}
        <div class="recommendation-item priority-{{ action.priority.lower() }}">
            <strong>{{ action.priority }}:</strong> {{ action.action }}
            <br><small>Timeline: {{ action.timeline }}</small>
        </div>
        {% endfor %}
        {% endif %}
        
        {% if recommendations.short_term_actions %}
        <h4>Short-term Actions</h4>
        {% for action in recommendations.short_term_actions %}
        <div class="recommendation-item priority-{{ action.priority.lower() }}">
            <strong>{{ action.priority }}:</strong> {{ action.action }}
            <br><small>Timeline: {{ action.timeline }}</small>
        </div>
        {% endfor %}
        {% endif %}
        
        {% if recommendations.long_term_actions %}
        <h4>Long-term Strategic Actions</h4>
        {% for action in recommendations.long_term_actions %}
        <div class="recommendation-item priority-{{ action.priority.lower() }}">
            <strong>{{ action.priority }}:</strong> {{ action.action }}
            <br><small>Timeline: {{ action.timeline }}</small>
        </div>
        {% endfor %}
        {% endif %}
    </div>
    
    {% if health_analysis.critical_devices %}
    <div class="chart-container">
        <h3 style="color: {{ config.color_scheme.danger }};">Critical Devices Requiring Immediate Attention</h3>
        <table class="device-table">
            <thead>
                <tr>
                    <th>Device ID</th>
                    <th>Device Name</th>
                    <th>Health Score</th>
                    <th>Trend</th>
                    <th>Recommended Action</th>
                </tr>
            </thead>
            <tbody>
                {% for device in health_analysis.critical_devices %}
                <tr>
                    <td>{{ device.device_id }}</td>
                    <td>{{ device.device_name }}</td>
                    <td class="status-poor">{{ "%.1f"|format(device.health_score) }}%</td>
                    <td>{{ device.trend.title() }}</td>
                    <td>Immediate inspection and maintenance required</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    {% endif %}
    
    {% if health_analysis.top_performers %}
    <div class="chart-container">
        <h3 style="color: {{ config.color_scheme.success }};">Top Performing Devices</h3>
        <table class="device-table">
            <thead>
                <tr>
                    <th>Device ID</th>
                    <th>Device Name</th>
                    <th>Health Score</th>
                    <th>Trend</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {% for device in health_analysis.top_performers %}
                <tr>
                    <td>{{ device.device_id }}</td>
                    <td>{{ device.device_name }}</td>
                    <td class="status-excellent">{{ "%.1f"|format(device.health_score) }}%</td>
                    <td>{{ device.trend.title() }}</td>
                    <td>Excellent Performance</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    {% endif %}
    
    {% if predictions %}
    <div class="chart-container">
        <h3>Predictive Analysis</h3>
        
        {% if predictions.health_forecast %}
        <h4>Health Forecast</h4>
        <div class="recommendation-item priority-medium">
            <strong>Next 7 Days:</strong> {{ predictions.health_forecast.next_7_days }}<br>
            <strong>Next 30 Days:</strong> {{ predictions.health_forecast.next_30_days }}<br>
            <strong>Confidence:</strong> {{ "%.1f"|format(predictions.health_forecast.confidence * 100) }}%
        </div>
        {% endif %}
        
        {% if predictions.maintenance_predictions %}
        <h4>Maintenance Predictions</h4>
        <table class="device-table">
            <thead>
                <tr>
                    <th>Device ID</th>
                    <th>Device Name</th>
                    <th>Days Until Maintenance</th>
                    <th>Confidence</th>
                    <th>Recommendation</th>
                </tr>
            </thead>
            <tbody>
                {% for pred in predictions.maintenance_predictions %}
                <tr>
                    <td>{{ pred.device_id }}</td>
                    <td>{{ pred.device_name }}</td>
                    <td>{{ pred.days_until_maintenance }}</td>
                    <td>{{ "%.1f"|format(pred.confidence * 100) }}%</td>
                    <td class="{% if pred.days_until_maintenance <= 10 %}status-poor{% else %}status-fair{% endif %}">
                        {% if pred.days_until_maintenance <= 10 %}Schedule maintenance soon{% else %}Monitor closely{% endif %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}
    </div>
    {% endif %}
    
    <div class="footer">
        <p>Report generated by {{ config.system_name }} v{{ config.version }}</p>
        <p>Generated on {{ metadata.generated_at | format_datetime }} | Report Type: {{ metadata.report_type }}</p>
        <p>&copy; 2025 {{ config.company_name }}. All rights reserved.</p>
    </div>
</body>
</html>
            """
            
            # Render template
            template = Template(html_template)
            
            # Add a custom filter for formatting datetime
            def format_datetime(value):
                try:
                    return datetime.fromisoformat(value).strftime('%Y-%m-%d %H:%M:%S')
                except:
                    return value
            
            template.environment.filters['format_datetime'] = format_datetime
            
            html_content = template.render(**report_data)
            
            # Save HTML file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"health_report_{timestamp}.html"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML report generated: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"HTML report generation error: {e}")
            raise
    
    def _generate_pdf_report(self, html_path: str) -> str:
        """Generate PDF report from HTML (requires additional library)"""
        try:
            import weasyprint
            pdf_path = html_path.replace('.html', '.pdf')
            weasyprint.HTML(html_path).write_pdf(pdf_path)
            return pdf_path
            
        except ImportError:
            self.logger.warning("PDF generation skipped: 'weasyprint' library not installed.")
            self.logger.warning("To install: pip install weasyprint")
            return None
        except Exception as e:
            self.logger.error(f"PDF generation error: {e}")
            return None
    
    def generate_quick_summary(self, device_ids: List[str] = None) -> Dict:
        """Generate a quick health summary without full report"""
        try:
            self.logger.info("Generating quick health summary")
            
            # Collect minimal data
            data = self._collect_report_data(device_ids, 7)  # Last 7 days
            health_analysis = self._analyze_health_scores(data)
            summary = self._generate_executive_summary(data, health_analysis)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'summary': summary,
                'critical_devices_count': len(health_analysis.get('critical_devices', [])),
                'total_devices': len(set(d['device_id'] for d in data.get('devices', []))),
                'overall_health_trend': health_analysis.get('overall_health', {}).get('trend', 'unknown')
            }
            
        except Exception as e:
            self.logger.error(f"Quick summary generation error: {e}")
            return {}
    
    def schedule_automatic_reports(self, schedule_config: Dict):
        """Schedule automatic report generation (placeholder for scheduler integration)"""
        try:
            self.logger.info("Setting up automatic report scheduling")
            
            # This would integrate with a scheduler like APScheduler
            # For now, just log the configuration
            
            schedule_types = schedule_config.get('types', ['daily', 'weekly', 'monthly'])
            recipients = schedule_config.get('email_recipients', [])
            
            self.logger.info(f"Scheduled reports: {schedule_types}")
            self.logger.info(f"Recipients: {len(recipients)}")
            
            return {
                'status': 'scheduled',
                'types': schedule_types,
                'next_run': 'Not implemented'
            }
            
        except Exception as e:
            self.logger.error(f"Schedule setup error: {e}")
            return {'status': 'error', 'message': str(e)}


# CLI Interface for standalone usage
def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Digital Twin Health Reports')
    parser.add_argument('--devices', nargs='*', help='Specific device IDs to include')
    parser.add_argument('--days', type=int, default=30, help='Number of days to analyze')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Generate quick summary only')
    parser.add_argument('--no-predictions', dest='predictions', action='store_false', help='Disable predictions')
    parser.add_argument('--no-recommendations', dest='recommendations', action='store_false', help='Disable recommendations')
    parser.set_defaults(predictions=True, recommendations=True)
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = HealthReportGenerator(args.output or "REPORTS/generated")
    
    try:
        if args.quick:
            # Generate quick summary
            summary = generator.generate_quick_summary(args.devices)
            print("--- Quick Health Summary ---")
            print(f"  Overall Health: {summary.get('summary', {}).get('overall_health', {}).get('score', 'N/A')}%")
            print(f"  Overall Trend:  {summary.get('overall_health_trend', 'Unknown').title()}")
            print(f"  Total Devices:  {summary.get('total_devices', 0)}")
            print(f"  Critical:       {summary.get('critical_devices_count', 0)}")
            print("----------------------------")
        else:
            # Generate comprehensive report
            print(f"Generating comprehensive report for {args.days} days...")
            report_path = generator.generate_comprehensive_report(
                device_ids=args.devices,
                date_range_days=args.days,
                include_predictions=args.predictions,
                include_recommendations=args.recommendations
            )
            print(f"✅ Comprehensive report generated: {report_path}")
            
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())