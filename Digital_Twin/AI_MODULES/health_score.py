import numpy as np
import pandas as pd
import logging
import json
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
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import IsolationForest

class HealthScoreCalculator:
    """
    Advanced health score calculation system for Digital Twin applications.
    Calculates comprehensive health metrics including overall system health,
    component health, predictive health trends, and risk assessments.
    """
    
    def __init__(self, config_path: str = "CONFIG/health_score_config.json"):
        self.config_path = Path(config_path)
        self.logger = self._setup_logging()
        
        # Health score configuration
        self.config = self._load_config()
        
        # Health score history
        self.health_history = deque(maxlen=1000)
        self.component_history = defaultdict(lambda: deque(maxlen=500))
        
        # Scoring models and scalers
        self.scoring_models = {}
        self.scalers = {}
        
        # Thresholds and weights
        self.health_thresholds = self.config.get('thresholds', {
            'critical': 0.3,
            'warning': 0.6,
            'good': 0.8,
            'excellent': 0.95
        })
        
        # Component weights for overall health calculation
        self.component_weights = self.config.get('component_weights', {})
        
        # Baseline values for comparison
        self.baseline_values = {}
        self.baseline_updated = {}
        
    def _setup_logging(self):
        """Setup logging for health score calculator."""
        logger = logging.getLogger('HealthScoreCalculator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Create logs directory if it doesn't exist
            Path('LOGS').mkdir(exist_ok=True)
            handler = logging.FileHandler('LOGS/digital_twin_health.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_config(self) -> Dict:
        """Load health score configuration."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                self.logger.info("Health score configuration loaded")
                return config
            else:
                # Default configuration
                default_config = {
                    "thresholds": {
                        "critical": 0.3,
                        "warning": 0.6,
                        "good": 0.8,
                        "excellent": 0.95
                    },
                    "component_weights": {
                        "temperature": 0.2,
                        "pressure": 0.15,
                        "vibration": 0.25,
                        "efficiency": 0.2,
                        "uptime": 0.2
                    },
                    "score_components": {
                        "performance": 0.3,
                        "reliability": 0.25,
                        "efficiency": 0.2,
                        "safety": 0.15,
                        "maintenance": 0.1
                    },
                    "anomaly_detection": {
                        "contamination": 0.1,
                        "sensitivity": 0.8
                    },
                    "trend_analysis": {
                        "window_size": 50,
                        "trend_threshold": 0.05
                    },
                    "risk_assessment": {
                        "failure_probability_weight": 0.4,
                        "consequence_weight": 0.6
                    }
                }
                
                # Save default configuration
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                
                self.logger.info("Default health score configuration created")
                return default_config
                
        except Exception as e:
            self.logger.error(f"Health history clear error: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize health score calculator
    health_calculator = HealthScoreCalculator()
    
    # Generate sample data for testing
    np.random.seed(42)
    
    # Sample device data with realistic sensor readings
    device_data = {
        'device_A': pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'temperature': np.random.normal(25, 3, 100),  # Good temperature range
            'pressure': np.random.normal(1013, 5, 100),   # Normal pressure
            'vibration': np.random.exponential(0.1, 100), # Low vibration (good)
            'efficiency': np.random.beta(3, 2, 100) * 100, # High efficiency
            'uptime': np.random.choice([1, 0], 100, p=[0.95, 0.05])  # 95% uptime
        }),
        
        'device_B': pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'temperature': np.random.normal(35, 8, 100),   # Higher temperature
            'pressure': np.random.normal(1020, 15, 100),   # Higher pressure variation
            'vibration': np.random.exponential(0.3, 100),  # Higher vibration
            'efficiency': np.random.beta(2, 3, 100) * 100, # Lower efficiency
            'uptime': np.random.choice([1, 0], 100, p=[0.85, 0.15])  # 85% uptime
        }),
        
        'device_C': pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'temperature': np.random.normal(45, 12, 100),  # High temperature (critical)
            'pressure': np.random.normal(1050, 25, 100),   # High pressure
            'vibration': np.random.exponential(0.8, 100),  # High vibration (bad)
            'efficiency': np.random.beta(1, 4, 100) * 100, # Poor efficiency
            'uptime': np.random.choice([1, 0], 100, p=[0.70, 0.30])  # 70% uptime
        })
    }
    
    print("=== DIGITAL TWIN HEALTH SCORE CALCULATOR DEMO ===\n")
    
    # 1. Calculate individual device health scores
    print("1. Calculating individual device health scores:")
    individual_results = {}
    
    for device_id, data in device_data.items():
        result = health_calculator.calculate_overall_health_score(data, device_id)
        individual_results[device_id] = result
        
        print(f"   {device_id}:")
        print(f"      Health Score: {result['overall_score']:.3f}")
        print(f"      Status: {result['health_status']}")
        print(f"      Trend: {result['trend_analysis']['trend_direction']}")
        print(f"      Risk Level: {result['risk_assessment']['overall_risk_level']}")
        print()
    
    # 2. Calculate fleet health score
    print("2. Calculating fleet health score:")
    fleet_result = health_calculator.calculate_fleet_health_score(device_data)
    
    print(f"   Fleet Health Score: {fleet_result['fleet_health_score']:.3f}")
    print(f"   Fleet Status: {fleet_result['fleet_health_status']}")
    print(f"   Healthy Devices: {fleet_result['fleet_analytics']['healthy_devices']}")
    print(f"   Warning Devices: {fleet_result['fleet_analytics']['warning_devices']}")
    print(f"   Critical Devices: {fleet_result['fleet_analytics']['critical_devices']}")
    print()
    
    # 3. Show best and worst performing devices
    print("3. Device Performance Ranking:")
    print("   Best Performing:")
    for device in fleet_result['best_performing_devices'][:3]:
        print(f"      {device['device_id']}: {device['score']:.3f}")
    
    print("   Worst Performing:")
    for device in fleet_result['worst_performing_devices'][-3:]:
        print(f"      {device['device_id']}: {device['score']:.3f}")
    print()
    
    # 4. Show recommendations
    print("4. Fleet Recommendations:")
    for i, rec in enumerate(fleet_result['fleet_recommendations'][:5], 1):
        print(f"   {i}. [{rec['priority'].upper()}] {rec['action']}")
        print(f"      Timeline: {rec['timeframe']}")
    print()
    
    # 5. Risk assessment summary
    print("5. Fleet Risk Assessment:")
    risk = fleet_result['fleet_risk_assessment']
    print(f"   Fleet Risk Level: {risk['fleet_risk_level']}")
    print(f"   High Risk Devices: {len(risk['high_risk_devices'])}")
    print(f"   Medium Risk Devices: {len(risk['medium_risk_devices'])}")
    print(f"   Low Risk Devices: {len(risk['low_risk_devices'])}")
    print()
    
    # 6. Health history and trends
    print("6. Health History Analysis:")
    history = health_calculator.get_health_history(limit=10)
    print(f"   Total health records: {len(history)}")
    
    if history:
        recent_scores = [h['overall_score'] for h in history[-5:]]
        print(f"   Recent scores: {[round(s, 3) for s in recent_scores]}")
    
    # 7. Export configuration
    print("\n7. Exporting configuration...")
    config_path = health_calculator.export_health_configuration()
    print(f"   Configuration exported to: {config_path}")
    
    # 8. Statistics
    print("\n8. Health Scoring Statistics:")
    stats = health_calculator.get_health_statistics()
    print(f"   Total health records: {stats['history_summary']['total_health_records']}")
    print(f"   Devices tracked: {stats['history_summary']['devices_tracked']}")
    print(f"   Average health score: {stats['history_summary']['average_health_score']:.3f}")
    print(f"   Baseline metrics: {stats['configuration_summary']['baseline_metrics']}")
    
    print("\n=== HEALTH SCORE ANALYSIS COMPLETED ===")
    print("\nKey Insights:")
    print(f"- Device A: Excellent health ({individual_results['device_A']['overall_score']:.3f})")
    print(f"- Device B: Moderate health ({individual_results['device_B']['overall_score']:.3f})")
    print(f"- Device C: Critical health ({individual_results['device_C']['overall_score']:.3f})")
    print(f"- Fleet requires immediate attention for critical devices")
    print(f"- Predictive maintenance recommended for warning-level devices").logger.error(f"Failed to load configuration: {e}")

    
    def calculate_overall_health_score(self, data: pd.DataFrame, 
                                     device_id: str = None,
                                     timestamp: datetime = None) -> Dict:
        """
        Calculate overall health score for a system or device.
        
        Args:
            data: DataFrame with sensor/system data
            device_id: Optional device identifier
            timestamp: Optional timestamp for the calculation
            
        Returns:
            Dictionary containing health score and components
        """
        try:
            self.logger.info(f"Calculating overall health score for device: {device_id or 'system'}")
            
            if timestamp is None:
                timestamp = datetime.now()
            
            # Prepare data
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                return self._create_error_result("No numeric data available")
            
            # Calculate component scores
            component_scores = {}
            
            # 1. Performance Score
            performance_score = self._calculate_performance_score(numeric_data)
            component_scores['performance'] = performance_score
            
            # 2. Reliability Score
            reliability_score = self._calculate_reliability_score(numeric_data, data)
            component_scores['reliability'] = reliability_score
            
            # 3. Efficiency Score
            efficiency_score = self._calculate_efficiency_score(numeric_data)
            component_scores['efficiency'] = efficiency_score
            
            # 4. Safety Score
            safety_score = self._calculate_safety_score(numeric_data)
            component_scores['safety'] = safety_score
            
            # 5. Maintenance Score
            maintenance_score = self._calculate_maintenance_score(numeric_data, data)
            component_scores['maintenance'] = maintenance_score
            
            # Calculate weighted overall score
            score_weights = self.config.get('score_components', {})
            overall_score = 0.0
            total_weight = 0.0
            
            for component, score_data in component_scores.items():
                if isinstance(score_data, dict) and 'score' in score_data:
                    weight = score_weights.get(component, 0.2)
                    overall_score += score_data['score'] * weight
                    total_weight += weight
            
            if total_weight > 0:
                overall_score = overall_score / total_weight
            else:
                overall_score = 0.5  # Default neutral score
            
            # Determine health status
            health_status = self._determine_health_status(overall_score)
            
            # Calculate trend
            trend_analysis = self._calculate_health_trend(device_id, overall_score)
            
            # Risk assessment
            risk_assessment = self._calculate_risk_assessment(component_scores, overall_score)
            
            # Create result
            result = {
                'device_id': device_id,
                'timestamp': timestamp.isoformat(),
                'overall_score': round(overall_score, 3),
                'health_status': health_status,
                'component_scores': component_scores,
                'trend_analysis': trend_analysis,
                'risk_assessment': risk_assessment,
                'recommendations': self._generate_health_recommendations(component_scores, overall_score),
                'data_quality': self._assess_data_quality(data),
                'calculation_metadata': {
                    'data_points': len(data),
                    'numeric_columns': list(numeric_data.columns),
                    'calculation_time': datetime.now().isoformat()
                }
            }
            
            # Store in history
            self._store_health_score(result)
            
            self.logger.info(f"Health score calculated: {overall_score:.3f} ({health_status})")
            return result
            
        except Exception as e:
            self.logger.error(f"Overall health score calculation error: {e}")
            return self._create_error_result(str(e))
    
    def _calculate_performance_score(self, data: pd.DataFrame) -> Dict:
        """Calculate performance-based health score."""
        try:
            performance_metrics = {}
            
            # Key performance indicators
            performance_cols = [col for col in data.columns if any(
                keyword in col.lower() for keyword in 
                ['efficiency', 'throughput', 'output', 'performance', 'speed', 'rate']
            )]
            
            if not performance_cols:
                # Use all numeric data as performance indicators
                performance_cols = data.columns.tolist()
            
            scores = []
            
            for col in performance_cols:
                col_data = data[col].dropna()
                
                if len(col_data) == 0:
                    continue
                
                # Calculate performance metrics
                col_score = self._calculate_metric_score(col_data, col, 'performance')
                
                performance_metrics[col] = {
                    'score': col_score,
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'trend': self._calculate_simple_trend(col_data)
                }
                
                scores.append(col_score)
            
            overall_performance = np.mean(scores) if scores else 0.5
            
            return {
                'score': round(overall_performance, 3),
                'metrics': performance_metrics,
                'summary': self._summarize_score_component(overall_performance, 'performance')
            }
            
        except Exception as e:
            self.logger.error(f"Performance score calculation error: {e}")
            return {'score': 0.5, 'error': str(e)}
    
    def _calculate_reliability_score(self, numeric_data: pd.DataFrame, full_data: pd.DataFrame) -> Dict:
        """Calculate reliability-based health score."""
        try:
            reliability_metrics = {}
            
            # Reliability indicators
            scores = []
            
            # 1. Data consistency (low variance indicates reliability)
            consistency_scores = []
            for col in numeric_data.columns:
                col_data = numeric_data[col].dropna()
                if len(col_data) > 1:
                    cv = col_data.std() / (col_data.mean() + 1e-8)  # Coefficient of variation
                    consistency_score = 1.0 / (1.0 + cv)  # Lower variation = higher score
                    consistency_scores.append(consistency_score)
                    
                    reliability_metrics[f'{col}_consistency'] = {
                        'coefficient_of_variation': float(cv),
                        'consistency_score': round(consistency_score, 3)
                    }
            
            if consistency_scores:
                scores.append(np.mean(consistency_scores))
            
            # 2. Uptime/availability (if timestamp data is available)
            if 'timestamp' in full_data.columns:
                uptime_score = self._calculate_uptime_score(full_data)
                scores.append(uptime_score)
                reliability_metrics['uptime'] = {'score': uptime_score}
            
            # 3. Anomaly-based reliability
            anomaly_score = self._calculate_anomaly_based_reliability(numeric_data)
            scores.append(anomaly_score)
            reliability_metrics['anomaly_based'] = {'score': anomaly_score}
            
            overall_reliability = np.mean(scores) if scores else 0.5
            
            return {
                'score': round(overall_reliability, 3),
                'metrics': reliability_metrics,
                'summary': self._summarize_score_component(overall_reliability, 'reliability')
            }
            
        except Exception as e:
            self.logger.error(f"Reliability score calculation error: {e}")
            return {'score': 0.5, 'error': str(e)}
    
    def _calculate_efficiency_score(self, data: pd.DataFrame) -> Dict:
        """Calculate efficiency-based health score."""
        try:
            efficiency_metrics = {}
            
            # Efficiency indicators
            efficiency_cols = [col for col in data.columns if any(
                keyword in col.lower() for keyword in 
                ['efficiency', 'energy', 'power', 'consumption', 'utilization']
            )]
            
            scores = []
            
            for col in efficiency_cols:
                col_data = data[col].dropna()
                
                if len(col_data) == 0:
                    continue
                
                # For efficiency metrics, higher values are generally better
                # Normalize to 0-1 range
                if col_data.std() > 0:
                    normalized_data = (col_data - col_data.min()) / (col_data.max() - col_data.min())
                    efficiency_score = normalized_data.mean()
                else:
                    efficiency_score = 0.8  # Stable efficiency
                
                efficiency_metrics[col] = {
                    'score': round(efficiency_score, 3),
                    'mean_efficiency': float(col_data.mean()),
                    'efficiency_trend': self._calculate_simple_trend(col_data)
                }
                
                scores.append(efficiency_score)
            
            # If no specific efficiency columns, calculate general efficiency
            if not scores:
                # Use overall data stability as efficiency indicator
                overall_cv = np.mean([data[col].std() / (data[col].mean() + 1e-8) 
                                    for col in data.columns if data[col].std() > 0])
                efficiency_score = 1.0 / (1.0 + overall_cv)
                scores.append(efficiency_score)
                
                efficiency_metrics['general_stability'] = {
                    'score': round(efficiency_score, 3),
                    'coefficient_of_variation': float(overall_cv)
                }
            
            overall_efficiency = np.mean(scores) if scores else 0.5
            
            return {
                'score': round(overall_efficiency, 3),
                'metrics': efficiency_metrics,
                'summary': self._summarize_score_component(overall_efficiency, 'efficiency')
            }
            
        except Exception as e:
            self.logger.error(f"Efficiency score calculation error: {e}")
            return {'score': 0.5, 'error': str(e)}
    
    def _calculate_safety_score(self, data: pd.DataFrame) -> Dict:
        """Calculate safety-based health score."""
        try:
            safety_metrics = {}
            
            # Safety-related indicators
            safety_cols = [col for col in data.columns if any(
                keyword in col.lower() for keyword in 
                ['temperature', 'pressure', 'vibration', 'safety', 'alarm', 'warning']
            )]
            
            scores = []
            
            for col in safety_cols:
                col_data = data[col].dropna()
                
                if len(col_data) == 0:
                    continue
                
                # Check for values within safe operating ranges
                safety_score = self._calculate_safety_range_score(col_data, col)
                
                safety_metrics[col] = {
                    'score': safety_score,
                    'mean_value': float(col_data.mean()),
                    'max_value': float(col_data.max()),
                    'min_value': float(col_data.min()),
                    'outlier_percentage': self._calculate_outlier_percentage(col_data)
                }
                
                scores.append(safety_score)
            
            # If no specific safety columns, use general anomaly detection
            if not scores:
                general_safety = self._calculate_general_safety_score(data)
                scores.append(general_safety)
                safety_metrics['general_safety'] = {'score': general_safety}
            
            overall_safety = np.mean(scores) if scores else 0.8  # Default to good safety
            
            return {
                'score': round(overall_safety, 3),
                'metrics': safety_metrics,
                'summary': self._summarize_score_component(overall_safety, 'safety')
            }
            
        except Exception as e:
            self.logger.error(f"Safety score calculation error: {e}")
            return {'score': 0.8, 'error': str(e)}
    
    def _calculate_maintenance_score(self, numeric_data: pd.DataFrame, full_data: pd.DataFrame) -> Dict:
        """Calculate maintenance-based health score."""
        try:
            maintenance_metrics = {}
            
            # Maintenance indicators
            scores = []
            
            # 1. Degradation indicators (increasing trend in negative metrics)
            degradation_cols = [col for col in numeric_data.columns if any(
                keyword in col.lower() for keyword in 
                ['vibration', 'noise', 'wear', 'degradation', 'fault']
            )]
            
            for col in degradation_cols:
                col_data = numeric_data[col].dropna()
                
                if len(col_data) > 5:
                    # Check for increasing trend (bad for maintenance)
                    trend_score = 1.0 - abs(self._calculate_simple_trend(col_data))
                    scores.append(max(0.0, trend_score))
                    
                    maintenance_metrics[f'{col}_degradation'] = {
                        'trend_score': round(trend_score, 3),
                        'current_level': float(col_data.iloc[-1] if len(col_data) > 0 else 0)
                    }
            
            # 2. Maintenance schedule adherence (if maintenance data is available)
            if 'maintenance_date' in full_data.columns or 'last_maintenance' in full_data.columns:
                schedule_score = self._calculate_maintenance_schedule_score(full_data)
                scores.append(schedule_score)
                maintenance_metrics['schedule_adherence'] = {'score': schedule_score}
            
            # 3. Performance degradation over time
            performance_degradation = self._calculate_performance_degradation(numeric_data)
            scores.append(performance_degradation)
            maintenance_metrics['performance_degradation'] = {'score': performance_degradation}
            
            overall_maintenance = np.mean(scores) if scores else 0.7
            
            return {
                'score': round(overall_maintenance, 3),
                'metrics': maintenance_metrics,
                'summary': self._summarize_score_component(overall_maintenance, 'maintenance')
            }
            
        except Exception as e:
            self.logger.error(f"Maintenance score calculation error: {e}")
            return {'score': 0.7, 'error': str(e)}
    
    def _calculate_metric_score(self, data: pd.Series, metric_name: str, score_type: str) -> float:
        """Calculate score for a specific metric."""
        try:
            # Get baseline if available
            baseline_key = f"{metric_name}_{score_type}"
            
            if baseline_key in self.baseline_values:
                baseline = self.baseline_values[baseline_key]
                
                # Calculate deviation from baseline
                current_mean = data.mean()
                deviation = abs(current_mean - baseline['mean']) / (baseline['std'] + 1e-8)
                
                # Convert deviation to score (0-1 range)
                score = np.exp(-deviation)  # Exponential decay
                
            else:
                # No baseline available, use data quality metrics
                if len(data) > 1:
                    # Use coefficient of variation as quality indicator
                    cv = data.std() / (data.mean() + 1e-8)
                    score = 1.0 / (1.0 + cv)
                else:
                    score = 0.5  # Neutral score for single data point
                
                # Store as baseline for future comparisons
                self.baseline_values[baseline_key] = {
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'updated': datetime.now().isoformat()
                }
            
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
            
        except Exception as e:
            self.logger.error(f"Metric score calculation error: {e}")
            return 0.5
    
    def _calculate_simple_trend(self, data: pd.Series) -> float:
        """Calculate simple linear trend."""
        try:
            if len(data) < 3:
                return 0.0
            
            x = np.arange(len(data))
            slope, _, _, _, _ = stats.linregress(x, data.values)
            
            # Normalize slope by data range
            data_range = data.max() - data.min()
            if data_range > 0:
                normalized_slope = slope / data_range
            else:
                normalized_slope = 0.0
            
            return float(normalized_slope)
            
        except Exception as e:
            self.logger.error(f"Trend calculation error: {e}")
            return 0.0
    
    def _calculate_uptime_score(self, data: pd.DataFrame) -> float:
        """Calculate uptime/availability score."""
        try:
            if 'timestamp' not in data.columns:
                return 0.8  # Default good uptime
            
            timestamps = pd.to_datetime(data['timestamp'])
            timestamps = timestamps.sort_values()
            
            if len(timestamps) < 2:
                return 0.8
            
            # Calculate gaps between data points
            time_diffs = timestamps.diff().dropna()
            expected_interval = time_diffs.median()
            
            # Count gaps larger than 2x expected interval as downtime
            downtime_gaps = time_diffs[time_diffs > 2 * expected_interval]
            total_downtime = downtime_gaps.sum()
            total_time = timestamps.max() - timestamps.min()
            
            if total_time.total_seconds() > 0:
                uptime_ratio = 1.0 - (total_downtime.total_seconds() / total_time.total_seconds())
                return max(0.0, min(1.0, uptime_ratio))
            else:
                return 0.8
                
        except Exception as e:
            self.logger.error(f"Uptime score calculation error: {e}")
            return 0.8
    
    def _calculate_anomaly_based_reliability(self, data: pd.DataFrame) -> float:
        """Calculate reliability based on anomaly detection."""
        try:
            # Use Isolation Forest for anomaly detection
            if len(data) < 10:
                return 0.8
            
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data.fillna(data.mean()))
            
            iso_forest = IsolationForest(
                contamination=self.config.get('anomaly_detection', {}).get('contamination', 0.1),
                random_state=42
            )
            
            anomaly_predictions = iso_forest.fit_predict(data_scaled)
            
            # Calculate reliability as 1 - anomaly_rate
            anomaly_rate = (anomaly_predictions == -1).mean()
            reliability_score = 1.0 - anomaly_rate
            
            return max(0.0, min(1.0, reliability_score))
            
        except Exception as e:
            self.logger.error(f"Anomaly-based reliability calculation error: {e}")
            return 0.7
    
    def _calculate_safety_range_score(self, data: pd.Series, column_name: str) -> float:
        """Calculate safety score based on operating ranges."""
        try:
            # Define safety ranges for common parameters
            safety_ranges = {
                'temperature': {'min': -10, 'max': 80, 'optimal_min': 15, 'optimal_max': 35},
                'pressure': {'min': 900, 'max': 1100, 'optimal_min': 1000, 'optimal_max': 1020},
                'vibration': {'min': 0, 'max': 1.0, 'optimal_min': 0, 'optimal_max': 0.3},
            }
            
            # Find applicable range
            applicable_range = None
            for param, ranges in safety_ranges.items():
                if param in column_name.lower():
                    applicable_range = ranges
                    break
            
            if applicable_range is None:
                # Use statistical approach for unknown parameters
                q25, q75 = data.quantile([0.25, 0.75])
                iqr = q75 - q25
                applicable_range = {
                    'min': q25 - 1.5 * iqr,
                    'max': q75 + 1.5 * iqr,
                    'optimal_min': q25,
                    'optimal_max': q75
                }
            
            # Calculate scores
            in_safe_range = ((data >= applicable_range['min']) & 
                           (data <= applicable_range['max'])).mean()
            
            in_optimal_range = ((data >= applicable_range['optimal_min']) & 
                              (data <= applicable_range['optimal_max'])).mean()
            
            # Weighted score: 70% for safe range, 30% for optimal range
            safety_score = 0.7 * in_safe_range + 0.3 * in_optimal_range
            
            return max(0.0, min(1.0, safety_score))
            
        except Exception as e:
            self.logger.error(f"Safety range score calculation error: {e}")
            return 0.8
    
    def _calculate_outlier_percentage(self, data: pd.Series) -> float:
        """Calculate percentage of outliers in data."""
        try:
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            total_points = len(data)
            
            return float(outliers / total_points) if total_points > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Outlier percentage calculation error: {e}")
            return 0.0
    
    def _calculate_general_safety_score(self, data: pd.DataFrame) -> float:
        """Calculate general safety score using anomaly detection."""
        try:
            # Use isolation forest for general anomaly detection
            if len(data) < 5:
                return 0.8
            
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data.fillna(data.mean()))
            
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            anomalies = iso_forest.fit_predict(data_scaled)
            
            # Higher anomaly rate = lower safety score
            anomaly_rate = (anomalies == -1).mean()
            safety_score = 1.0 - 2 * anomaly_rate  # Amplify impact of anomalies
            
            return max(0.0, min(1.0, safety_score))
            
        except Exception as e:
            self.logger.error(f"General safety score calculation error: {e}")
            return 0.8
    
    def _calculate_maintenance_schedule_score(self, data: pd.DataFrame) -> float:
        """Calculate maintenance schedule adherence score."""
        try:
            # This is a placeholder - would need actual maintenance schedule data
            # For now, return a default score
            return 0.8
            
        except Exception as e:
            self.logger.error(f"Maintenance schedule score calculation error: {e}")
            return 0.8
    
    def _calculate_performance_degradation(self, data: pd.DataFrame) -> float:
        """Calculate performance degradation score."""
        try:
            if len(data) < 10:
                return 0.8
            
            # Calculate degradation for each column
            degradation_scores = []
            
            for col in data.columns:
                col_data = data[col].dropna()
                
                if len(col_data) < 5:
                    continue
                
                # Split data into early and recent periods
                split_point = len(col_data) // 2
                early_data = col_data.iloc[:split_point]
                recent_data = col_data.iloc[split_point:]
                
                # Compare performance (assuming higher values are better for most metrics)
                if len(early_data) > 0 and len(recent_data) > 0:
                    performance_change = recent_data.mean() / (early_data.mean() + 1e-8)
                    
                    # Convert to score (1.0 = no change, >1.0 = improvement, <1.0 = degradation)
                    if performance_change >= 1.0:
                        score = min(1.0, performance_change - 0.5)  # Cap improvement benefit
                    else:
                        score = performance_change
                    
                    degradation_scores.append(score)
            
            if degradation_scores:
                return np.mean(degradation_scores)
            else:
                return 0.8
                
        except Exception as e:
            self.logger.error(f"Performance degradation calculation error: {e}")
            return 0.8
    
    def _determine_health_status(self, score: float) -> str:
        """Determine health status based on score."""
        thresholds = self.health_thresholds
        
        if score >= thresholds['excellent']:
            return 'excellent'
        elif score >= thresholds['good']:
            return 'good'
        elif score >= thresholds['warning']:
            return 'warning'
        elif score >= thresholds['critical']:
            return 'critical'
        else:
            return 'failure'
    
    def _calculate_health_trend(self, device_id: str, current_score: float) -> Dict:
        """Calculate health trend analysis."""
        try:
            if device_id:
                history = list(self.component_history[device_id])
            else:
                history = list(self.health_history)
            
            if len(history) < 3:
                return {
                    'trend_direction': 'stable',
                    'trend_strength': 0.0,
                    'data_points': len(history)
                }
            
            # Extract scores from history
            scores = [entry.get('overall_score', entry.get('score', 0.5)) for entry in history[-10:]]
            scores.append(current_score)
            
            # Calculate trend
            x = np.arange(len(scores))
            slope, _, r_value, _, _ = stats.linregress(x, scores)
            
            # Determine trend direction
            trend_threshold = self.config.get('trend_analysis', {}).get('trend_threshold', 0.01)
            
            if slope > trend_threshold:
                trend_direction = 'improving'
            elif slope < -trend_threshold:
                trend_direction = 'degrading'
            else:
                trend_direction = 'stable'
            
            return {
                'trend_direction': trend_direction,
                'trend_strength': abs(float(slope)),
                'trend_r_squared': float(r_value**2),
                'data_points': len(scores),
                'recent_scores': scores[-5:],  # Last 5 scores
                'prediction_next_period': float(current_score + slope) if slope != 0 else current_score
            }
            
        except Exception as e:
            self.logger.error(f"Health trend calculation error: {e}")
            return {'trend_direction': 'stable', 'trend_strength': 0.0}
    
    def _calculate_risk_assessment(self, component_scores: Dict, overall_score: float) -> Dict:
        """Calculate comprehensive risk assessment."""
        try:
            risk_factors = []
            
            # Component-based risks
            for component, score_data in component_scores.items():
                if isinstance(score_data, dict) and 'score' in score_data:
                    score = score_data['score']
                    
                    if score < self.health_thresholds['critical']:
                        risk_factors.append({
                            'component': component,
                            'risk_level': 'high',
                            'score': score,
                            'impact': 'critical_component_failure'
                        })
                    elif score < self.health_thresholds['warning']:
                        risk_factors.append({
                            'component': component,
                            'risk_level': 'medium',
                            'score': score,
                            'impact': 'performance_degradation'
                        })
            
            # Overall risk level
            if overall_score < self.health_thresholds['critical']:
                overall_risk = 'high'
            elif overall_score < self.health_thresholds['warning']:
                overall_risk = 'medium'
            elif overall_score < self.health_thresholds['good']:
                overall_risk = 'low'
            else:
                overall_risk = 'minimal'
            
            # Failure probability estimation
            failure_probability = self._estimate_failure_probability(overall_score, component_scores)
            
            # Time to failure estimation
            time_to_failure = self._estimate_time_to_failure(overall_score, component_scores)
            
            return {
                'overall_risk_level': overall_risk,
                'risk_factors': risk_factors,
                'failure_probability': failure_probability,
                'estimated_time_to_failure': time_to_failure,
                'risk_score': round(1.0 - overall_score, 3),  # Inverse of health score
                'mitigation_priority': self._determine_mitigation_priority(risk_factors, overall_risk)
            }
            
        except Exception as e:
            self.logger.error(f"Risk assessment calculation error: {e}")
            return {'overall_risk_level': 'unknown', 'risk_factors': []}
    
    def _estimate_failure_probability(self, overall_score: float, component_scores: Dict) -> Dict:
        """Estimate probability of failure."""
        try:
            # Simple probability model based on health score
            base_probability = 1.0 - overall_score
            
            # Adjust based on component scores
            critical_components = 0
            for component, score_data in component_scores.items():
                if isinstance(score_data, dict) and 'score' in score_data:
                    if score_data['score'] < self.health_thresholds['critical']:
                        critical_components += 1
            
            # Increase probability if critical components exist
            adjusted_probability = base_probability * (1.0 + 0.5 * critical_components)
            adjusted_probability = min(1.0, adjusted_probability)
            
            # Time-based probabilities
            probabilities = {
                'next_24_hours': adjusted_probability * 0.1,
                'next_week': adjusted_probability * 0.3,
                'next_month': adjusted_probability * 0.6,
                'next_quarter': adjusted_probability * 0.9,
                'overall': adjusted_probability
            }
            
            return {
                'probabilities': probabilities,
                'confidence': 0.7,  # Model confidence
                'model_type': 'health_score_based'
            }
            
        except Exception as e:
            self.logger.error(f"Failure probability estimation error: {e}")
            return {'probabilities': {}, 'confidence': 0.0}
    
    def _estimate_time_to_failure(self, overall_score: float, component_scores: Dict) -> Dict:
        """Estimate time to failure."""
        try:
            if overall_score > self.health_thresholds['good']:
                return {
                    'estimated_days': None,
                    'confidence': 0.0,
                    'status': 'healthy_no_prediction'
                }
            
            # Simple linear model: time inversely proportional to (1 - health_score)
            degradation_rate = 1.0 - overall_score
            
            if degradation_rate <= 0:
                return {
                    'estimated_days': None,
                    'confidence': 0.0,
                    'status': 'no_degradation_detected'
                }
            
            # Estimate days to critical threshold
            days_to_critical = (overall_score - self.health_thresholds['critical']) / (degradation_rate * 0.01)
            days_to_critical = max(1, days_to_critical)  # At least 1 day
            
            # Estimate days to complete failure
            days_to_failure = overall_score / (degradation_rate * 0.01)
            days_to_failure = max(1, days_to_failure)
            
            return {
                'estimated_days_to_critical': round(days_to_critical),
                'estimated_days_to_failure': round(days_to_failure),
                'confidence': 0.6,  # Medium confidence for simple model
                'model_type': 'linear_degradation'
            }
            
        except Exception as e:
            self.logger.error(f"Time to failure estimation error: {e}")
            return {'estimated_days': None, 'confidence': 0.0}
    
    def _determine_mitigation_priority(self, risk_factors: List[Dict], overall_risk: str) -> str:
        """Determine mitigation priority."""
        try:
            high_risk_factors = [rf for rf in risk_factors if rf.get('risk_level') == 'high']
            medium_risk_factors = [rf for rf in risk_factors if rf.get('risk_level') == 'medium']
            
            if overall_risk == 'high' or len(high_risk_factors) >= 2:
                return 'immediate'
            elif overall_risk == 'medium' or len(high_risk_factors) >= 1:
                return 'urgent'
            elif len(medium_risk_factors) >= 2:
                return 'scheduled'
            else:
                return 'routine'
                
        except Exception as e:
            self.logger.error(f"Mitigation priority determination error: {e}")
            return 'routine'
    
    def _generate_health_recommendations(self, component_scores: Dict, overall_score: float) -> List[Dict]:
        """Generate health improvement recommendations."""
        try:
            recommendations = []
            
            # Component-specific recommendations
            for component, score_data in component_scores.items():
                if isinstance(score_data, dict) and 'score' in score_data:
                    score = score_data['score']
                    
                    if score < self.health_thresholds['critical']:
                        recommendations.append({
                            'priority': 'critical',
                            'component': component,
                            'action': f'Immediate {component} inspection and maintenance required',
                            'expected_impact': 'Prevent system failure',
                            'timeframe': 'immediate'
                        })
                    elif score < self.health_thresholds['warning']:
                        recommendations.append({
                            'priority': 'high',
                            'component': component,
                            'action': f'Schedule {component} maintenance within next week',
                            'expected_impact': 'Improve system reliability',
                            'timeframe': 'within_week'
                        })
                    elif score < self.health_thresholds['good']:
                        recommendations.append({
                            'priority': 'medium',
                            'component': component,
                            'action': f'Monitor {component} closely and plan maintenance',
                            'expected_impact': 'Maintain optimal performance',
                            'timeframe': 'within_month'
                        })
            
            # Overall system recommendations
            if overall_score < self.health_thresholds['critical']:
                recommendations.append({
                    'priority': 'critical',
                    'component': 'system',
                    'action': 'System-wide inspection and immediate corrective action',
                    'expected_impact': 'Prevent catastrophic failure',
                    'timeframe': 'immediate'
                })
            elif overall_score < self.health_thresholds['warning']:
                recommendations.append({
                    'priority': 'high',
                    'component': 'system',
                    'action': 'Comprehensive system health check',
                    'expected_impact': 'Restore system health',
                    'timeframe': 'within_week'
                })
            
            # Sort by priority
            priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))
            
            return recommendations[:10]  # Limit to top 10 recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendation generation error: {e}")
            return []
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict:
        """Assess quality of input data."""
        try:
            total_cells = data.size
            missing_cells = data.isnull().sum().sum()
            missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
            
            # Check for data completeness
            completeness_score = 1.0 - (missing_percentage / 100)
            
            # Check for data freshness (if timestamp available)
            freshness_score = 1.0  # Default
            if 'timestamp' in data.columns:
                try:
                    latest_timestamp = pd.to_datetime(data['timestamp']).max()
                    time_since_last = datetime.now() - latest_timestamp.to_pydatetime()
                    hours_since = time_since_last.total_seconds() / 3600
                    
                    # Fresher data gets higher score
                    if hours_since <= 1:
                        freshness_score = 1.0
                    elif hours_since <= 24:
                        freshness_score = 0.8
                    elif hours_since <= 168:  # 1 week
                        freshness_score = 0.6
                    else:
                        freshness_score = 0.3
                except:
                    pass
            
            # Check for data consistency (low variance in key metrics)
            consistency_scores = []
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                col_data = data[col].dropna()
                if len(col_data) > 1:
                    cv = col_data.std() / (col_data.mean() + 1e-8)
                    consistency_score = 1.0 / (1.0 + cv)
                    consistency_scores.append(consistency_score)
            
            overall_consistency = np.mean(consistency_scores) if consistency_scores else 0.8
            
            # Overall data quality score
            quality_score = 0.4 * completeness_score + 0.3 * freshness_score + 0.3 * overall_consistency
            
            return {
                'overall_quality_score': round(quality_score, 3),
                'completeness_score': round(completeness_score, 3),
                'freshness_score': round(freshness_score, 3),
                'consistency_score': round(overall_consistency, 3),
                'missing_percentage': round(missing_percentage, 2),
                'total_data_points': len(data),
                'numeric_columns': len(numeric_cols),
                'quality_level': 'high' if quality_score > 0.8 else 'medium' if quality_score > 0.6 else 'low'
            }
            
        except Exception as e:
            self.logger.error(f"Data quality assessment error: {e}")
            return {'overall_quality_score': 0.5, 'quality_level': 'unknown'}
    
    def _summarize_score_component(self, score: float, component: str) -> Dict:
        """Summarize a score component."""
        status = self._determine_health_status(score)
        
        summary_messages = {
            'performance': {
                'excellent': 'System performance is exceptional',
                'good': 'System performance is good',
                'warning': 'System performance shows signs of degradation',
                'critical': 'System performance is critically low',
                'failure': 'System performance indicates failure'
            },
            'reliability': {
                'excellent': 'System reliability is excellent',
                'good': 'System reliability is good',
                'warning': 'System reliability is concerning',
                'critical': 'System reliability is at critical levels',
                'failure': 'System reliability indicates failure risk'
            },
            'efficiency': {
                'excellent': 'System efficiency is optimal',
                'good': 'System efficiency is good',
                'warning': 'System efficiency could be improved',
                'critical': 'System efficiency is critically low',
                'failure': 'System efficiency indicates operational failure'
            },
            'safety': {
                'excellent': 'System operates within optimal safety parameters',
                'good': 'System operates within safe parameters',
                'warning': 'Safety parameters show warning signs',
                'critical': 'Safety parameters are at critical levels',
                'failure': 'Safety parameters indicate dangerous conditions'
            },
            'maintenance': {
                'excellent': 'Maintenance requirements are minimal',
                'good': 'Maintenance is up to date',
                'warning': 'Maintenance attention is needed',
                'critical': 'Critical maintenance is required',
                'failure': 'Immediate maintenance intervention required'
            }
        }
        
        message = summary_messages.get(component, {}).get(status, f'{component} status is {status}')
        
        return {
            'status': status,
            'message': message,
            'score_range': f"{score:.3f}",
            'improvement_potential': round((1.0 - score) * 100, 1) if score < 1.0 else 0
        }
    
    def _store_health_score(self, result: Dict):
        """Store health score in history."""
        try:
            # Store overall health score
            self.health_history.append({
                'timestamp': result['timestamp'],
                'overall_score': result['overall_score'],
                'health_status': result['health_status'],
                'device_id': result.get('device_id')
            })
            
            # Store component scores if device-specific
            if result.get('device_id'):
                device_id = result['device_id']
                self.component_history[device_id].append({
                    'timestamp': result['timestamp'],
                    'overall_score': result['overall_score'],
                    'component_scores': result['component_scores'],
                    'health_status': result['health_status']
                })
            
        except Exception as e:
            self.logger.error(f"Health score storage error: {e}")
    
    def _create_error_result(self, error_message: str) -> Dict:
        """Create error result structure."""
        return {
            'error': True,
            'message': error_message,
            'timestamp': datetime.now().isoformat(),
            'overall_score': 0.0,
            'health_status': 'unknown'
        }
    
    def get_health_history(self, device_id: str = None, limit: int = 100) -> List[Dict]:
        """Get health score history."""
        try:
            if device_id:
                history = list(self.component_history.get(device_id, []))
            else:
                history = list(self.health_history)
            
            # Sort by timestamp and limit results
            history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return history[:limit]
            
        except Exception as e:
            self.logger.error(f"Health history retrieval error: {e}")
            return []
    
    def calculate_fleet_health_score(self, fleet_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Calculate health scores for an entire fleet of devices.
        
        Args:
            fleet_data: Dictionary mapping device_ids to their data DataFrames
            
        Returns:
            Dictionary containing fleet health analysis
        """
        try:
            self.logger.info(f"Calculating fleet health score for {len(fleet_data)} devices")
            
            fleet_results = {}
            device_scores = []
            
            # Calculate individual device scores
            for device_id, data in fleet_data.items():
                try:
                    device_result = self.calculate_overall_health_score(data, device_id)
                    fleet_results[device_id] = device_result
                    
                    if not device_result.get('error'):
                        device_scores.append(device_result['overall_score'])
                        
                except Exception as e:
                    self.logger.error(f"Error calculating health for device {device_id}: {e}")
                    fleet_results[device_id] = self._create_error_result(str(e))
            
            # Calculate fleet-level statistics
            if device_scores:
                fleet_health_score = np.mean(device_scores)
                fleet_health_status = self._determine_health_status(fleet_health_score)
                
                # Fleet analytics
                fleet_analytics = {
                    'total_devices': len(fleet_data),
                    'healthy_devices': len([s for s in device_scores if s >= self.health_thresholds['good']]),
                    'warning_devices': len([s for s in device_scores if self.health_thresholds['warning'] <= s < self.health_thresholds['good']]),
                    'critical_devices': len([s for s in device_scores if s < self.health_thresholds['critical']]),
                    'average_score': round(fleet_health_score, 3),
                    'score_std': round(np.std(device_scores), 3),
                    'min_score': round(min(device_scores), 3),
                    'max_score': round(max(device_scores), 3)
                }
                
                # Identify best and worst performing devices
                device_performance = [(device_id, result.get('overall_score', 0)) 
                                    for device_id, result in fleet_results.items() 
                                    if not result.get('error')]
                
                device_performance.sort(key=lambda x: x[1], reverse=True)
                
                best_devices = device_performance[:5]  # Top 5
                worst_devices = device_performance[-5:]  # Bottom 5
                
                # Fleet-wide risk assessment
                fleet_risk = self._calculate_fleet_risk_assessment(fleet_results)
                
                # Fleet recommendations
                fleet_recommendations = self._generate_fleet_recommendations(fleet_results, fleet_analytics)
                
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'fleet_health_score': fleet_health_score,
                    'fleet_health_status': fleet_health_status,
                    'fleet_analytics': fleet_analytics,
                    'device_results': fleet_results,
                    'best_performing_devices': [{'device_id': dev_id, 'score': score} for dev_id, score in best_devices],
                    'worst_performing_devices': [{'device_id': dev_id, 'score': score} for dev_id, score in worst_devices],
                    'fleet_risk_assessment': fleet_risk,
                    'fleet_recommendations': fleet_recommendations
                }
                
            else:
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'fleet_health_score': 0.0,
                    'fleet_health_status': 'unknown',
                    'error': 'No valid device scores calculated',
                    'device_results': fleet_results
                }
            
            self.logger.info(f"Fleet health calculation completed: {result.get('fleet_health_score', 0):.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Fleet health score calculation error: {e}")
            return self._create_error_result(str(e))
    
    def _calculate_fleet_risk_assessment(self, fleet_results: Dict) -> Dict:
        """Calculate risk assessment for entire fleet."""
        try:
            risk_summary = {
                'high_risk_devices': [],
                'medium_risk_devices': [],
                'low_risk_devices': [],
                'fleet_risk_level': 'low'
            }
            
            for device_id, result in fleet_results.items():
                if result.get('error'):
                    continue
                
                risk_assessment = result.get('risk_assessment', {})
                overall_risk = risk_assessment.get('overall_risk_level', 'unknown')
                
                if overall_risk == 'high':
                    risk_summary['high_risk_devices'].append({
                        'device_id': device_id,
                        'score': result.get('overall_score', 0),
                        'risk_factors': risk_assessment.get('risk_factors', [])
                    })
                elif overall_risk == 'medium':
                    risk_summary['medium_risk_devices'].append({
                        'device_id': device_id,
                        'score': result.get('overall_score', 0)
                    })
                else:
                    risk_summary['low_risk_devices'].append({
                        'device_id': device_id,
                        'score': result.get('overall_score', 0)
                    })
            
            # Determine fleet-wide risk level
            total_devices = len([r for r in fleet_results.values() if not r.get('error')])
            high_risk_count = len(risk_summary['high_risk_devices'])
            medium_risk_count = len(risk_summary['medium_risk_devices'])
            
            if total_devices > 0:
                high_risk_ratio = high_risk_count / total_devices
                medium_risk_ratio = medium_risk_count / total_devices
                
                if high_risk_ratio > 0.3 or (high_risk_ratio > 0.1 and medium_risk_ratio > 0.4):
                    risk_summary['fleet_risk_level'] = 'high'
                elif high_risk_ratio > 0.1 or medium_risk_ratio > 0.3:
                    risk_summary['fleet_risk_level'] = 'medium'
                else:
                    risk_summary['fleet_risk_level'] = 'low'
            
            risk_summary['risk_statistics'] = {
                'total_devices': total_devices,
                'high_risk_count': high_risk_count,
                'medium_risk_count': medium_risk_count,
                'low_risk_count': len(risk_summary['low_risk_devices']),
                'high_risk_percentage': round((high_risk_count / total_devices) * 100, 1) if total_devices > 0 else 0
            }
            
            return risk_summary
            
        except Exception as e:
            self.logger.error(f"Fleet risk assessment error: {e}")
            return {'fleet_risk_level': 'unknown'}
    
    def _generate_fleet_recommendations(self, fleet_results: Dict, fleet_analytics: Dict) -> List[Dict]:
        """Generate fleet-wide recommendations."""
        try:
            recommendations = []
            
            # Critical device recommendations
            critical_devices = fleet_analytics.get('critical_devices', 0)
            if critical_devices > 0:
                recommendations.append({
                    'priority': 'critical',
                    'scope': 'fleet',
                    'action': f'Immediate attention required for {critical_devices} critical devices',
                    'impact': 'Prevent fleet-wide failures',
                    'timeframe': 'immediate'
                })
            
            # Warning device recommendations
            warning_devices = fleet_analytics.get('warning_devices', 0)
            if warning_devices > fleet_analytics.get('total_devices', 1) * 0.3:  # >30% in warning
                recommendations.append({
                    'priority': 'high',
                    'scope': 'fleet',
                    'action': f'Schedule maintenance for {warning_devices} devices showing warning signs',
                    'impact': 'Improve overall fleet reliability',
                    'timeframe': 'within_week'
                })
            
            # Performance optimization
            avg_score = fleet_analytics.get('average_score', 0)
            if avg_score < self.health_thresholds['good']:
                recommendations.append({
                    'priority': 'medium',
                    'scope': 'fleet',
                    'action': 'Implement fleet-wide performance optimization program',
                    'impact': 'Enhance overall fleet performance',
                    'timeframe': 'within_month'
                })
            
            # Preventive maintenance
            if fleet_analytics.get('score_std', 0) > 0.2:  # High variation in scores
                recommendations.append({
                    'priority': 'medium',
                    'scope': 'fleet',
                    'action': 'Standardize maintenance procedures across fleet',
                    'impact': 'Reduce performance variation',
                    'timeframe': 'within_month'
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Fleet recommendations generation error: {e}")
            return []
    
    def update_baseline_values(self, data: pd.DataFrame, device_id: str = None):
        """Update baseline values for health score calculations."""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            for col in numeric_data.columns:
                baseline_key = f"{col}_baseline"
                if device_id:
                    baseline_key = f"{device_id}_{baseline_key}"
                
                col_data = numeric_data[col].dropna()
                if len(col_data) > 0:
                    self.baseline_values[baseline_key] = {
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'updated': datetime.now().isoformat(),
                        'sample_size': len(col_data)
                    }
            
            self.logger.info(f"Baseline values updated for {len(numeric_data.columns)} metrics")
            
        except Exception as e:
            self.logger.error(f"Baseline update error: {e}")
    
    def export_health_configuration(self, output_path: str = None) -> str:
        """Export health score configuration and baselines."""
        try:
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"CONFIG/health_config_export_{timestamp}.json"
            
            export_data = {
                'configuration': self.config,
                'health_thresholds': self.health_thresholds,
                'component_weights': self.component_weights,
                'baseline_values': self.baseline_values,
                'export_timestamp': datetime.now().isoformat(),
                'health_history_count': len(self.health_history),
                'component_history_count': {k: len(v) for k, v in self.component_history.items()}
            }
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Health configuration exported to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Configuration export error: {e}")
            raise
    
    def import_health_configuration(self, input_path: str):
        """Import health score configuration and baselines."""
        try:
            with open(input_path, 'r') as f:
                import_data = json.load(f)
            
            # Update configuration
            if 'configuration' in import_data:
                self.config.update(import_data['configuration'])
            
            if 'health_thresholds' in import_data:
                self.health_thresholds.update(import_data['health_thresholds'])
            
            if 'component_weights' in import_data:
                self.component_weights.update(import_data['component_weights'])
            
            if 'baseline_values' in import_data:
                self.baseline_values.update(import_data['baseline_values'])
            
            self.logger.info(f"Health configuration imported from {input_path}")
            
        except Exception as e:
            self.logger.error(f"Configuration import error: {e}")
            raise
    
    def get_health_statistics(self) -> Dict:
        """Get comprehensive health scoring statistics."""
        try:
            stats = {
                'configuration_summary': {
                    'thresholds': self.health_thresholds,
                    'component_weights': self.component_weights,
                    'baseline_metrics': len(self.baseline_values)
                },
                'history_summary': {
                    'total_health_records': len(self.health_history),
                    'devices_tracked': len(self.component_history),
                    'average_health_score': np.mean([h['overall_score'] for h in self.health_history]) if self.health_history else 0
                },
                'recent_activity': list(self.health_history)[-5:] if self.health_history else [],
                'system_status': {
                    'baseline_coverage': len(self.baseline_values),
                    'active_monitoring': len(self.component_history),
                    'last_calculation': self.health_history[-1]['timestamp'] if self.health_history else None
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Health statistics error: {e}")
            return {}
    
    def clear_health_history(self, device_id: str = None):
        """Clear health score history."""
        try:
            if device_id:
                if device_id in self.component_history:
                    self.component_history[device_id].clear()
                    self.logger.info(f"Health history cleared for device: {device_id}")
            else:
                self.health_history.clear()
                self.component_history.clear()
                self.logger.info("All health history cleared")
                
        except Exception as e:
            self