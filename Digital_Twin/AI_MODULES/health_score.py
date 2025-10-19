import numpy as np
import pandas as pd
import logging
import json
import pickle
import joblib
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, deque
import warnings

# --- Add project root to path ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- ML Imports ---
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- Import other AI Modules ---
try:
    # This is key for model-based scoring
    from AI_MODULES.predictive_analytics_engine import PredictiveAnalyticsEngine
    ANALYTICS_ENGINE_AVAILABLE = True
    print("PredictiveAnalyticsEngine loaded for model-based scoring.")
except ImportError:
    PredictiveAnalyticsEngine = None
    ANALYTICS_ENGINE_AVAILABLE = False
    print("Warning: PredictiveAnalyticsEngine not found. Model-based scoring will be disabled.")


class HealthScoreCalculator:
    """
    Advanced health score calculation system for Digital Twin applications.
    Calculates comprehensive health metrics using two methods:
    1. Formula-Based: A weighted average of key performance components.
    2. Model-Based: A trained ML model that uses component scores AND
       live predictions (anomalies, failures) from the analytics engine.
    The final score is a blend of these two methods.
    """
    
    def __init__(self, 
                 config_path: str = "CONFIG/health_score_config.json",
                 model_path: str = "ANALYTICS/models/"):
        
        self.config_path = Path(config_path)
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logging()
        
        # Health score configuration
        self.config = self._load_config()
        
        # Health score history
        # --- REFINED ---
        # Increased history size to provide more data for component calculations
        self.health_history = deque(maxlen=2000) 
        self.component_history = defaultdict(lambda: deque(maxlen=1000)) # Device-specific history
        
        # --- Model-Based Scoring Attributes ---
        self.scoring_model_name = "health_scorer.pkl"
        self.scoring_model = None
        self.scoring_model_metadata = {}
        # Load the analytics engine to get predictive features
        self.analytics_engine = PredictiveAnalyticsEngine() if ANALYTICS_ENGINE_AVAILABLE else None
        self._load_scoring_model()
        
        # Thresholds and weights
        self.health_thresholds = self.config.get('thresholds', {
            'critical': 0.3,
            'warning': 0.6,
            'good': 0.8,
            'excellent': 0.95
        })
        
        # Component weights for overall health calculation
        self.component_weights = self.config.get('component_weights', {})
        
        # --- REFINED: Adaptive EMA Baselines ---
        # Replaces self.baseline_values
        self.baseline_emas = {} # Stores {'metric_key': {'mean': ema_mean, 'std': ema_std}}
        self.baseline_ema_alpha = self.config.get('baseline_ema_alpha', 0.1) # Learning rate for baselines
        
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
                    "component_weights": { # Weights for FORMULA-based score
                        "performance": 0.3,
                        "reliability": 0.25,
                        "efficiency": 0.2,
                        "safety": 0.15,
                        "maintenance": 0.1
                    },
                    "model_blending": { 
                        "formula_weight": 0.5, # 50% formula
                        "model_weight": 0.5    # 50% ML model
                    },
                    # --- REFINED: Added EMA Alpha ---
                    "baseline_ema_alpha": 0.1, # Slow learning rate for adaptive baselines
                    # --- REFINED: Added Safety Ranges ---
                    "safety_ranges": {
                        "temperature": {'min': -10, 'max': 80, 'optimal_min': 15, 'optimal_max': 35},
                        "pressure": {'min': 900, 'max': 1100, 'optimal_min': 1000, 'optimal_max': 1020},
                        "vibration": {'min': 0, 'max': 1.0, 'optimal_min': 0, 'optimal_max': 0.3}
                    },
                    "anomaly_detection": {
                        "contamination": 0.1,
                        "sensitivity": 0.8
                    },
                    "trend_analysis": {
                        "window_size": 50,
                        "trend_threshold": 0.01
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
            self.logger.error(f"Failed to load configuration: {e}")
            return {} # Return empty dict on failure

    # --- Model-Based Scoring Methods ---

    def _load_scoring_model(self):
        """Load the trained health scoring ML model from disk."""
        model_file = self.model_path / self.scoring_model_name
        try:
            if model_file.exists():
                model_data = joblib.load(model_file)
                self.scoring_model = model_data.get('model')
                self.scoring_model_metadata = model_data.get('metadata', {})
                self.logger.info(f"Health scoring model '{self.scoring_model_name}' loaded.")
            else:
                self.logger.info(f"Health scoring model not found. Will use formula-based scoring.")
        except Exception as e:
            self.logger.error(f"Failed to load health scoring model: {e}")

    def _save_scoring_model(self):
        """Save the trained health scoring ML model to disk."""
        if self.scoring_model is None:
            self.logger.warning("No health scoring model to save.")
            return

        model_file = self.model_path / self.scoring_model_name
        try:
            model_data = {
                'model': self.scoring_model,
                'metadata': self.scoring_model_metadata
            }
            joblib.dump(model_data, model_file)
            self.logger.info(f"Health scoring model saved to '{model_file}'")
        except Exception as e:
            self.logger.error(f"Failed to save health scoring model: {e}")

    def train_health_scorer(self, data: pd.DataFrame, failure_labels: pd.Series) -> Dict:
        """
        Trains an ML model to predict health score (1 - failure_prob).
        
        Args:
            data: DataFrame of historical sensor data.
            failure_labels: Series of 0s (healthy) and 1s (failure)
                            corresponding to the data.
                            
        Returns:
            Dictionary with training results.
        """
        if not self.analytics_engine:
            msg = "AnalyticsEngine not available. Cannot train model-based scorer."
            self.logger.error(msg)
            return {'error': msg}

        self.logger.info(f"Starting health scorer training with {len(data)} samples...")
        
        # We want to predict "health" (0 to 1), so we invert failure labels.
        health_labels = 1 - failure_labels
        
        # --- Feature Engineering ---
        # We will train the model on the outputs of our component scores
        # and the outputs of the predictive analytics engine.
        features_list = []
        
        # This is slow (row-by-row) but demonstrates the concept.
        # For production, this should be vectorized.
        # --- REFINED --- 
        # Use a sliding window to give component calculators enough data
        window_size = 50 # Make this larger if more history is needed
        
        for i in range(window_size, len(data)):
            # --- REFINED --- 
            # Use a window of data, not just a single row
            window_data = data.iloc[i-window_size:i]
            current_row = data.iloc[i:i+1] # For predictive features
            
            features = {}
            
            # 1. Get component scores
            num_data = window_data.select_dtypes(include=[np.number])
            features['performance'] = self._calculate_performance_score(num_data)['score']
            features['reliability'] = self._calculate_reliability_score(num_data, window_data)['score']
            features['efficiency'] = self._calculate_efficiency_score(num_data)['score']
            features['safety'] = self._calculate_safety_score(num_data)['score']
            features['maintenance'] = self._calculate_maintenance_score(num_data, window_data)['score']
            
            # 2. Get predictive features (from the *current* state)
            try:
                # Use a fast anomaly model for training
                # This call *must* match the one in calculate_overall_health_score
                anomaly_res = self.analytics_engine.detect_anomalies(current_row, model_name="anomaly_detector")
                features['anomaly_score'] = anomaly_res.get('anomaly_scores', [0.0])[0]
                
                # Use failure predictor
                fail_res = self.analytics_engine.predict_failure(current_row)
                features['failure_prob'] = fail_res.get('failure_probabilities', [0.0])[0]
            except Exception:
                features['anomaly_score'] = 0.0
                features['failure_prob'] = 0.0

            features_list.append(features)

        X = pd.DataFrame(features_list)
        # Adjust labels to match the features (we lost 'window_size' rows)
        y = health_labels.iloc[window_size:] 
        
        # Save feature names for prediction
        self.scoring_model_metadata['features'] = list(X.columns)
        
        # --- Model Training ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # We use a Regressor to predict a continuous health score from 0 to 1
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        self.scoring_model = model
        
        # --- Evaluation ---
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        # Get feature importances
        importances = dict(zip(X.columns, model.feature_importances_))
        
        self.scoring_model_metadata['training_mse'] = mse
        self.scoring_model_metadata['feature_importances'] = importances
        self.scoring_model_metadata['trained_at'] = datetime.now().isoformat()
        
        # Save the trained model
        self._save_scoring_model()
        
        self.logger.info(f"Health scorer training complete. Test MSE: {mse:.4f}")
        
        return {
            'status': 'success',
            'test_mse': mse,
            'feature_importances': importances,
            'features': list(X.columns)
        }

    # --- End New Methods ---

    def calculate_overall_health_score(self, data: pd.DataFrame, 
                                       device_id: str = None,
                                       timestamp: datetime = None,
                                       use_model: bool = True) -> Dict:
        """
        Calculate overall health score for a system or device.
        
        This now calculates a HYBRID score:
        - formula_based_score: Weighted average of components.
        - model_based_score: ML model prediction using components + predictive analytics.
        - overall_score: A blend of the two.
        
        Args:
            data: DataFrame with *new* sensor/system data (can be single row for real-time)
            device_id: Optional device identifier
            timestamp: Optional timestamp for the calculation
            use_model: (bool) Whether to attempt using the ML model
            
        Returns:
            Dictionary containing health score and components
        """
        try:
            self.logger.info(f"Calculating health score for device: {device_id or 'system'}")
            
            if timestamp is None:
                timestamp = datetime.now()
            
            # --- FIX: HISTORY-AWARE CALCULATION ---
            # Component scores (trends, variance, etc.) need history.
            # The 'data' param represents the *newest* data. We combine it with stored history.
            
            historical_data_list = []
            if device_id and device_id in self.component_history:
                # Get historical data points (just the data, not the full result dicts)
                for entry in self.component_history[device_id]:
                    # Assuming data was stored in history (we'll add this in _store_health_score)
                    if 'data' in entry:
                         historical_data_list.append(entry['data'])
            
            if historical_data_list:
                # Combine history + new data
                historical_df = pd.concat(historical_data_list, ignore_index=True)
                # Ensure no duplicates if data is re-sent
                combined_data = pd.concat([historical_df, data], ignore_index=True).drop_duplicates()
            else:
                combined_data = data
            
            # Use the combined (historical) data for component calculations
            # Use the *latest* data (the 'data' param) for predictive features
            
            # Prepare data
            numeric_data_history = combined_data.select_dtypes(include=[np.number]) # For components
            numeric_data_latest = data.select_dtypes(include=[np.number]) # For predictions
            
            if numeric_data_latest.empty:
                return self._create_error_result("No numeric data available in new packet")
            if numeric_data_history.empty:
                 # No history, just use current data
                 numeric_data_history = numeric_data_latest
            
            # --- 1. Calculate Component Scores (using historical data) ---
            component_scores = {}
            component_scores['performance'] = self._calculate_performance_score(numeric_data_history)
            component_scores['reliability'] = self._calculate_reliability_score(numeric_data_history, combined_data)
            component_scores['efficiency'] = self._calculate_efficiency_score(numeric_data_history)
            component_scores['safety'] = self._calculate_safety_score(numeric_data_history)
            component_scores['maintenance'] = self._calculate_maintenance_score(numeric_data_history, combined_data)
            
            # --- 2. Calculate Formula-Based Score ---
            score_weights = self.config.get('component_weights', {})
            formula_based_score = 0.0
            total_weight = 0.0
            
            for component, score_data in component_scores.items():
                if isinstance(score_data, dict) and 'score' in score_data:
                    weight = score_weights.get(component, 0.2) # Default 0.2 if not in config
                    formula_based_score += score_data['score'] * weight
                    total_weight += weight
            
            if total_weight > 0:
                formula_based_score = formula_based_score / total_weight
            else:
                formula_based_score = np.mean([cs['score'] for cs in component_scores.values()]) # Simple average
            
            # --- 3. Calculate Model-Based Score (using latest data) ---
            model_based_score = None
            calculation_method = "formula_based"
            model_inputs = {}

            if use_model and self.scoring_model and self.analytics_engine:
                try:
                    # 3a. Get component score inputs (these are now history-aware)
                    model_inputs['performance'] = component_scores['performance']['score']
                    model_inputs['reliability'] = component_scores['reliability']['score']
                    model_inputs['efficiency'] = component_scores['efficiency']['score']
                    model_inputs['safety'] = component_scores['safety']['score']
                    model_inputs['maintenance'] = component_scores['maintenance']['score']
                    
                    # 3b. Get live predictive inputs (using *latest* data)
                    
                    # --- FIX: Ensure prediction features match training features ---
                    # Call must be identical to the one in train_health_scorer
                    anomaly_res = self.analytics_engine.detect_anomalies(data, model_name="anomaly_detector")
                    fail_res = self.analytics_engine.predict_failure(data)
                    
                    # Data extraction must be identical to the one in train_health_scorer
                    model_inputs['anomaly_score'] = anomaly_res.get('anomaly_scores', [0.0])[0]
                    model_inputs['failure_prob'] = fail_res.get('failure_probabilities', [0.0])[0]
                    # --- END FIX ---

                    # 3c. Predict
                    # Ensure features are in the same order as during training
                    model_features = self.scoring_model_metadata.get('features', list(model_inputs.keys()))
                    input_vector = [model_inputs.get(f, 0.0) for f in model_features]
                    
                    model_based_score = self.scoring_model.predict([input_vector])[0]
                    model_based_score = max(0.0, min(1.0, model_based_score)) # Clamp prediction
                    calculation_method = "hybrid"

                except Exception as e:
                    self.logger.warning(f"Model-based scoring failed, falling back to formula. Error: {e}")
                    model_based_score = None
                    calculation_method = "formula_fallback"

            # --- 4. Blend Scores ---
            if model_based_score is not None:
                blend_weights = self.config.get('model_blending', {'formula_weight': 0.5, 'model_weight': 0.5})
                fw = blend_weights['formula_weight']
                mw = blend_weights['model_weight']
                overall_score = (formula_based_score * fw) + (model_based_score * mw)
            else:
                overall_score = formula_based_score # Fallback
            
            # --- 5. Finalize Result ---
            health_status = self._determine_health_status(overall_score)
            trend_analysis = self._calculate_health_trend(device_id, overall_score)
            risk_assessment = self._calculate_risk_assessment(component_scores, overall_score)
            
            result = {
                'device_id': device_id,
                'timestamp': timestamp.isoformat(),
                'overall_score': round(overall_score, 3),
                'health_status': health_status,
                'calculation_method': calculation_method,
                'formula_score': round(formula_based_score, 3),
                'model_score': round(model_based_score, 3) if model_based_score is not None else None,
                'component_scores': component_scores,
                'model_inputs': model_inputs if model_based_score is not None else {},
                'trend_analysis': trend_analysis,
                'risk_assessment': risk_assessment,
                'recommendations': self._generate_health_recommendations(component_scores, overall_score),
                'data_quality': self._assess_data_quality(data), # Quality of the *new* packet
                'historical_data_points': len(combined_data) # Info
            }
            
            # Store in history
            self._store_health_score(result, data) # --- REFINED: Pass data to be stored
            
            self.logger.info(f"Health score calculated: {overall_score:.3f} ({health_status}) using {calculation_method}")
            return result
            
        except Exception as e:
            self.logger.error(f"Overall health score calculation error: {e}", exc_info=True)
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
            
            scores = []
            
            if not performance_cols:
                # --- REFINED: Safer Fallback ---
                self.logger.warning("No performance columns found. Returning neutral score.")
                return {
                    'score': 0.7, # Assume 'good' if no data
                    'metrics': {},
                    'summary': self._summarize_score_component(0.7, 'performance')
                }
            
            for col in performance_cols:
                col_data = data[col].dropna()
                
                if len(col_data) == 0:
                    continue
                
                # REFINED: Pass 'higher_is_better'
                is_better = 'consumption' not in col.lower() and 'energy' not in col.lower()
                col_score = self._calculate_metric_score(col_data, col, 'performance', higher_is_better=is_better)
                
                performance_metrics[col] = {
                    'score': col_score,
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'trend': self._calculate_simple_trend(col_data)
                }
                
                scores.append(col_score)
            
            overall_performance = np.mean(scores) if scores else 0.7 # Neutral score
            
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
                    consistency_score = 1.0 / (1.0 + abs(cv)) 
                    consistency_scores.append(consistency_score)
                    
                    reliability_metrics[f'{col}_consistency'] = {
                        'coefficient_of_variation': float(cv),
                        'consistency_score': round(consistency_score, 3)
                    }
            
            if consistency_scores:
                scores.append(np.mean(consistency_scores))
            
            # 2. Uptime/availability (if 'uptime' col exists or timestamp gaps)
            if 'uptime' in full_data.columns:
                 uptime_score = full_data['uptime'].mean() # Assumes uptime is 0 or 1
            elif 'timestamp' in full_data.columns:
                 uptime_score = self._calculate_uptime_score(full_data)
            else:
                 uptime_score = 0.8 # Default
                 
            scores.append(uptime_score)
            reliability_metrics['uptime'] = {'score': uptime_score}
            
            # 3. Anomaly-based reliability (using local model)
            # This is intentionally kept separate from the main AI engine
            # to provide an independent, heuristic-based signal for the formula score.
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
                
                higher_is_better = 'consumption' not in col.lower() and 'energy' not in col.lower()
                efficiency_score = self._calculate_metric_score(col_data, col, 'efficiency', higher_is_better)
                
                efficiency_metrics[col] = {
                    'score': round(efficiency_score, 3),
                    'mean_value': float(col_data.mean()),
                    'trend': self._calculate_simple_trend(col_data)
                }
                
                scores.append(efficiency_score)
            
            # If no specific efficiency columns, calculate general efficiency
            if not scores and not data.empty:
                # Use overall data stability as efficiency indicator
                all_cvs = [data[col].std() / (data[col].mean() + 1e-8) 
                           for col in data.columns if data[col].std() > 0]
                overall_cv = np.mean(all_cvs) if all_cvs else 0
                efficiency_score = 1.0 / (1.0 + abs(overall_cv))
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
            
            # If no specific safety columns, use general anomaly detection (local model)
            if not scores and not data.empty:
                general_safety = self._calculate_general_safety_score(data)
                scores.append(general_safety)
                safety_metrics['general_safety'] = {'score': general_safety}
            
            overall_safety = np.mean(scores) if scores else 0.8 # Default to good safety
            
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
                    trend = self._calculate_simple_trend(col_data)
                    trend_score = max(0.0, min(1.0, 1.0 - trend * 5)) 
                    scores.append(trend_score)
                    
                    maintenance_metrics[f'{col}_degradation'] = {
                        'trend_score': round(trend_score, 3),
                        'trend_slope': trend,
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

    # --- REFINED: Sub-Calculator with Adaptive EMA Baseline ---

    def _calculate_metric_score(self, data: pd.Series, metric_name: str, score_type: str, 
                                higher_is_better: bool = True) -> float:
        """
        Calculate score for a specific metric based on deviation from an
        adaptive Exponential Moving Average (EMA) baseline.
        """
        try:
            baseline_key = f"{metric_name}_{score_type}"
            current_mean = data.mean()
            current_std = data.std()
            
            # Get or initialize EMA baseline
            if baseline_key in self.baseline_emas:
                baseline = self.baseline_emas[baseline_key]
                baseline_mean = baseline['mean']
                baseline_std = baseline['std']
                
                # Update EMA
                alpha = self.baseline_ema_alpha
                new_ema_mean = (alpha * current_mean) + ((1 - alpha) * baseline_mean)
                new_ema_std = (alpha * current_std) + ((1 - alpha) * baseline_std)
                
                self.baseline_emas[baseline_key] = {
                    'mean': float(new_ema_mean),
                    'std': float(new_ema_std)
                }
            else:
                # Initialize baseline
                self.baseline_emas[baseline_key] = {
                    'mean': float(current_mean),
                    'std': float(current_std)
                }
                # On first run, use current data for baseline
                baseline_mean = current_mean
                baseline_std = current_std
            
            # Use the baseline *before* the update for calculation
            # This scores the current data against the "known normal"
            safe_std = baseline_std + 1e-8
            
            # Calculate deviation (Z-score)
            deviation = (current_mean - baseline_mean) / safe_std
            
            if higher_is_better:
                # Good if deviation is positive or zero
                # Penalize negative deviation (Gaussian)
                score = np.exp(-0.5 * (min(0, deviation) ** 2))
            else:
                # (e.g., 'consumption')
                # Good if deviation is negative or zero
                # Penalize positive deviation (Gaussian)
                score = np.exp(-0.5 * (max(0, deviation) ** 2))
                
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
            
        except Exception as e:
            self.logger.error(f"Metric score calculation error: {e}")
            return 0.5
    
    # --- Other Sub-Calculators (Largely Unchanged) ---

    def _calculate_simple_trend(self, data: pd.Series) -> float:
        """Calculate simple linear trend slope."""
        try:
            if len(data) < 3:
                return 0.0
            
            x = np.arange(len(data))
            # Filter out NaNs
            valid_indices = ~data.isnull()
            if valid_indices.sum() < 3:
                return 0.0
                
            x_valid = x[valid_indices]
            y_valid = data.values[valid_indices]
            
            slope, _, _, _, _ = stats.linregress(x_valid, y_valid)
            
            data_range = y_valid.max() - y_valid.min()
            if data_range > 0:
                normalized_slope = slope / data_range 
            else:
                normalized_slope = 0.0
            
            return float(normalized_slope)
            
        except Exception as e:
            self.logger.error(f"Trend calculation error: {e}")
            return 0.0
    
    def _calculate_uptime_score(self, data: pd.DataFrame) -> float:
        """Calculate uptime/availability score based on timestamp gaps."""
        try:
            if 'timestamp' not in data.columns:
                return 0.8  # Default good uptime
            
            timestamps = pd.to_datetime(data['timestamp']).dropna()
            timestamps = timestamps.sort_values()
            
            if len(timestamps) < 2:
                return 0.8
            
            time_diffs = timestamps.diff().dropna()
            if time_diffs.empty:
                return 0.8
                
            expected_interval = time_diffs.median()
            
            # Count gaps larger than 3x expected interval as downtime
            downtime_gaps = time_diffs[time_diffs > 3 * expected_interval]
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
        """Calculate reliability based on anomaly detection (local model)."""
        try:
            if len(data) < 10:
                return 0.8
            
            # Fill NaNs before scaling
            data_filled = data.fillna(data.mean())
            if data_filled.empty:
                return 0.8
                
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_filled)
            
            iso_forest = IsolationForest(
                contamination=self.config.get('anomaly_detection', {}).get('contamination', 0.1),
                random_state=42
            )
            
            anomaly_predictions = iso_forest.fit_predict(data_scaled)
            
            anomaly_rate = (anomaly_predictions == -1).mean()
            reliability_score = 1.0 - anomaly_rate
            
            return max(0.0, min(1.0, reliability_score))
            
        except Exception as e:
            self.logger.error(f"Anomaly-based reliability calculation error: {e}")
            return 0.7
    
    def _calculate_safety_range_score(self, data: pd.Series, column_name: str) -> float:
        """Calculate safety score based on operating ranges from config."""
        try:
            # --- REFINED: Read ranges from config ---
            safety_ranges = self.config.get('safety_ranges', {})
            
            # Find applicable range
            applicable_range = None
            for param, ranges in safety_ranges.items():
                if param in column_name.lower():
                    applicable_range = ranges
                    break
            
            if applicable_range is None:
                # Use statistical approach for unknown parameters
                q25, q75 = data.quantile([0.25, 0.75])
                iqr = q75 - q75
                # Handle cases where iqr is 0
                if iqr < 1e-6:
                    iqr = data.std() + 1e-6
                    if iqr < 1e-6:
                        return 0.8 # Not enough variance to judge
                        
                applicable_range = {
                    'min': q25 - 1.5 * iqr,
                    'max': q75 + 1.5 * iqr,
                    'optimal_min': q25,
                    'optimal_max': q75
                }
            
            # Calculate scores
            in_safe_range_pct = ((data >= applicable_range['min']) & 
                                (data <= applicable_range['max'])).mean()
            
            in_optimal_range_pct = ((data >= applicable_range['optimal_min']) & 
                                    (data <= applicable_range['optimal_max'])).mean()
            
            # Weighted score: 50% for just being safe, 50% for being optimal.
            safety_score = 0.5 * in_safe_range_pct + 0.5 * in_optimal_range_pct
            
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
        """Calculate general safety score using anomaly detection (local model)."""
        try:
            if len(data) < 5:
                return 0.8
            
            data_filled = data.fillna(data.mean())
            if data_filled.empty:
                return 0.8
                
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_filled)
            
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            anomalies = iso_forest.fit_predict(data_scaled)
            
            anomaly_rate = (anomalies == -1).mean()
            safety_score = 1.0 - 2 * anomaly_rate # Amplify impact
            
            return max(0.0, min(1.0, safety_score))
            
        except Exception as e:
            self.logger.error(f"General safety score calculation error: {e}")
            return 0.8
    
    def _calculate_maintenance_schedule_score(self, data: pd.DataFrame) -> float:
        """Calculate maintenance schedule adherence score. (Placeholder)"""
        return 0.8
    
    def _calculate_performance_degradation(self, data: pd.DataFrame) -> float:
        """Calculate performance degradation score based on trend."""
        try:
            if len(data) < 10:
                return 0.8
            
            degradation_scores = []
            
            perf_cols = [col for col in data.columns if any(
                keyword in col.lower() for keyword in 
                ['efficiency', 'throughput', 'output', 'performance']
            )]
            
            for col in perf_cols:
                col_data = data[col].dropna()
                if len(col_data) < 5:
                    continue
                
                # Higher is better, so a negative trend is bad.
                trend = self._calculate_simple_trend(col_data)
                # Score = 1.0 for flat/positive trend, decreases with negative trend.
                score = max(0.0, min(1.0, 1.0 + trend * 5)) # *5 amplifies trend impact
                degradation_scores.append(score)

            if degradation_scores:
                return np.mean(degradation_scores)
            else:
                return 0.8 # Default if no perf columns
                
        except Exception as e:
            self.logger.error(f"Performance degradation calculation error: {e}")
            return 0.8

    # --- Helper & Utility Methods (Largely Unchanged) ---

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
            scores = [entry.get('overall_score', 0.5) for entry in history[-20:]] # Use last 20 points
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
            
            failure_probability = self._estimate_failure_probability(overall_score, component_scores)
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
            base_probability = 1.0 - overall_score
            
            critical_components = 0
            for component, score_data in component_scores.items():
                if isinstance(score_data, dict) and 'score' in score_data:
                    if score_data['score'] < self.health_thresholds['critical']:
                        critical_components += 1
            
            adjusted_probability = base_probability * (1.0 + 0.5 * critical_components)
            adjusted_probability = min(1.0, adjusted_probability)
            
            probabilities = {
                'next_24_hours': adjusted_probability * 0.1,
                'next_week': adjusted_probability * 0.3,
                'next_month': adjusted_probability * 0.6,
                'next_quarter': adjusted_probability * 0.9,
                'overall': adjusted_probability
            }
            
            return {
                'probabilities': probabilities,
                'confidence': 0.7, 
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
            
            degradation_rate = 1.0 - overall_score
            
            if degradation_rate <= 0:
                return {
                    'estimated_days': None,
                    'confidence': 0.0,
                    'status': 'no_degradation_detected'
                }
            
            # This is a very simple model, a real RUL model would be better
            days_to_critical = (overall_score - self.health_thresholds['critical']) / (degradation_rate * 0.01)
            days_to_critical = max(1, days_to_critical) 
            
            days_to_failure = overall_score / (degradation_rate * 0.01)
            days_to_failure = max(1, days_to_failure)
            
            return {
                'estimated_days_to_critical': round(days_to_critical),
                'estimated_days_to_failure': round(days_to_failure),
                'confidence': 0.6, 
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
            
            priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))
            
            return recommendations[:10] 
            
        except Exception as e:
            self.logger.error(f"Recommendation generation error: {e}")
            return []
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict:
        """Assess quality of input data."""
        try:
            total_cells = data.size
            if total_cells == 0:
                return {'overall_quality_score': 0.0, 'quality_level': 'no_data'}
                
            missing_cells = data.isnull().sum().sum()
            missing_percentage = (missing_cells / total_cells) * 100
            
            completeness_score = 1.0 - (missing_percentage / 100)
            
            freshness_score = 1.0  # Default
            if 'timestamp' in data.columns:
                try:
                    latest_timestamp = pd.to_datetime(data['timestamp']).max()
                    # Use utcnow() for timezone-aware comparison
                    time_since_last = datetime.utcnow() - latest_timestamp.to_pydatetime().replace(tzinfo=None)
                    hours_since = time_since_last.total_seconds() / 3600
                    
                    if hours_since <= 1:
                        freshness_score = 1.0
                    elif hours_since <= 24:
                        freshness_score = 0.8
                    elif hours_since <= 168:  # 1 week
                        freshness_score = 0.6
                    else:
                        freshness_score = 0.3
                except Exception:
                    pass
            
            consistency_scores = []
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                col_data = data[col].dropna()
                if len(col_data) > 1:
                    cv = col_data.std() / (col_data.mean() + 1e-8)
                    consistency_score = 1.0 / (1.0 + abs(cv))
                    consistency_scores.append(consistency_score)
            
            overall_consistency = np.mean(consistency_scores) if consistency_scores else 0.8
            
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
    
    def _store_health_score(self, result: Dict, data: pd.DataFrame): # --- REFINED: Added data param ---
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
                # --- REFINED: Store the data that was used ---
                # This is critical for the history-aware calculation
                history_entry = {
                    'timestamp': result['timestamp'],
                    'overall_score': result['overall_score'],
                    'component_scores': result['component_scores'], # Store component scores for analysis
                    'health_status': result['health_status'],
                    'data': data # Store the actual data packet
                }
                self.component_history[device_id].append(history_entry)
                
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
            
            history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return history[:limit]
            
        except Exception as e:
            self.logger.error(f"Health history retrieval error: {e}")
            return []
    
    def calculate_fleet_health_score(self, fleet_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Calculate health scores for an entire fleet of devices.
        
        Args:
            fleet_data: Dictionary mapping device_ids to their *latest* data DataFrames
            
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
                    # The calculate_overall_health_score function is now history-aware,
                    # so we just pass the new data and the device_id.
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
                
                device_performance = [(device_id, result.get('overall_score', 0)) 
                                      for device_id, result in fleet_results.items() 
                                      if not result.get('error')]
                
                device_performance.sort(key=lambda x: x[1], reverse=True)
                
                best_devices = device_performance[:5]
                worst_devices = device_performance[-5:]
                
                fleet_risk = self._calculate_fleet_risk_assessment(fleet_results)
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
            
            critical_devices = fleet_analytics.get('critical_devices', 0)
            if critical_devices > 0:
                recommendations.append({
                    'priority': 'critical',
                    'scope': 'fleet',
                    'action': f'Immediate attention required for {critical_devices} critical devices',
                    'impact': 'Prevent fleet-wide failures',
                    'timeframe': 'immediate'
                })
            
            warning_devices = fleet_analytics.get('warning_devices', 0)
            if warning_devices > fleet_analytics.get('total_devices', 1) * 0.3:  # >30% in warning
                recommendations.append({
                    'priority': 'high',
                    'scope': 'fleet',
                    'action': f'Schedule maintenance for {warning_devices} devices showing warning signs',
                    'impact': 'Improve overall fleet reliability',
                    'timeframe': 'within_week'
                })
            
            avg_score = fleet_analytics.get('average_score', 0)
            if avg_score < self.health_thresholds['good']:
                recommendations.append({
                    'priority': 'medium',
                    'scope': 'fleet',
                    'action': 'Implement fleet-wide performance optimization program',
                    'impact': 'Enhance overall fleet performance',
                    'timeframe': 'within_month'
                })
            
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
        """
        Update baseline EMA values for health score calculations.
        This is now handled automatically by _calculate_metric_score.
        This function can be used to "prime" the baselines.
        """
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            for col in numeric_data.columns:
                col_data = numeric_data[col].dropna()
                if len(col_data) > 0:
                    # Prime all component types
                    for score_type in ['performance', 'efficiency', 'safety', 'maintenance']:
                        baseline_key = f"{col}_{score_type}"
                        
                        current_mean = col_data.mean()
                        current_std = col_data.std()
                        
                        if baseline_key not in self.baseline_emas:
                            self.baseline_emas[baseline_key] = {
                                'mean': float(current_mean),
                                'std': float(current_std)
                            }
                        else:
                            # Update existing EMA
                            alpha = self.baseline_ema_alpha
                            baseline_mean = self.baseline_emas[baseline_key]['mean']
                            baseline_std = self.baseline_emas[baseline_key]['std']
                            new_ema_mean = (alpha * current_mean) + ((1 - alpha) * baseline_mean)
                            new_ema_std = (alpha * current_std) + ((1 - alpha) * baseline_std)
                            self.baseline_emas[baseline_key]['mean'] = new_ema_mean
                            self.baseline_emas[baseline_key]['std'] = new_ema_std
            
            self.logger.info(f"Baseline EMA values updated for {len(numeric_data.columns)} metrics")
            
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
                'baseline_emas': self.baseline_emas, # --- REFINED ---
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
            
            if 'configuration' in import_data:
                self.config.update(import_data['configuration'])
            
            if 'health_thresholds' in import_data:
                self.health_thresholds.update(import_data['health_thresholds'])
            
            if 'component_weights' in import_data:
                self.component_weights.update(import_data['component_weights'])
            
            # --- REFINED ---
            if 'baseline_emas' in import_data:
                self.baseline_emas.update(import_data['baseline_emas'])
            elif 'baseline_values' in import_data: # For backwards compatibility
                self.logger.warning("Importing old 'baseline_values'. Converting to 'baseline_emas'.")
                # Simple conversion: treat old baselines as initial EMA values
                for key, vals in import_data['baseline_values'].items():
                    if key not in self.baseline_emas:
                        self.baseline_emas[key] = {'mean': vals.get('mean', 0), 'std': vals.get('std', 1)}

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
                    'model_blending': self.config.get('model_blending'),
                    'model_loaded': self.scoring_model is not None,
                    'baseline_metrics': len(self.baseline_emas) # --- REFINED ---
                },
                'history_summary': {
                    'total_health_records': len(self.health_history),
                    'devices_tracked': len(self.component_history),
                    'average_health_score': np.mean([h['overall_score'] for h in self.health_history]) if self.health_history else 0
                },
                'recent_activity': list(self.health_history)[-5:] if self.health_history else [],
                'system_status': {
                    'baseline_coverage': len(self.baseline_emas), # --- REFINED ---
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
            self.logger.error(f"Health history clear error: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize health score calculator
    health_calculator = HealthScoreCalculator()
    
    # --- NEW: Check if we can train the scoring model ---
    if ANALYTICS_ENGINE_AVAILABLE and health_calculator.analytics_engine:
        print("\n--- Training Health Scorer Model (DEMO) ---")
        # Generate sample data with clear failure labels for training
        np.random.seed(42)
        train_size = 500
        train_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=train_size, freq='H'),
            'temperature': np.random.normal(25, 5, train_size),
            'pressure': np.random.normal(1013, 10, train_size),
            'vibration': np.random.exponential(0.1, train_size),
            'efficiency': np.random.normal(90, 5, train_size),
            'uptime': 1.0
        })
        failure_labels = pd.Series(np.zeros(train_size))
        
        failure_indices = np.random.choice(train_data.index[100:], 50, replace=False)
        train_data.loc[failure_indices, 'temperature'] += np.random.uniform(30, 50, 50)
        train_data.loc[failure_indices, 'vibration'] += np.random.uniform(0.5, 1.5, 50)
        train_data.loc[failure_indices, 'efficiency'] -= np.random.uniform(20, 40, 50)
        failure_labels.loc[failure_indices] = 1
        
        # Train the model
        train_result = health_calculator.train_health_scorer(train_data, failure_labels)
        print(f"Health Scorer trained. MSE: {train_result.get('test_mse', 'N/A'):.4f}")
        print("Feature Importances:")
        for f, imp in train_result.get('feature_importances', {}).items():
            print(f"  - {f}: {imp:.3f}")
    else:
        print("\n--- Skipping Health Scorer Model Training (AnalyticsEngine not found) ---")

    # --- Generate sample data for testing ---
    np.random.seed(42)
    
    device_data = {
        'device_A': pd.DataFrame({ # Healthy device
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'temperature': np.random.normal(25, 3, 100),
            'pressure': np.random.normal(1013, 5, 100),
            'vibration': np.random.exponential(0.1, 100),
            'efficiency': np.random.uniform(90, 95, 100),
            'uptime': 1.0
        }),
        
        'device_B': pd.DataFrame({ # Warning device
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'temperature': np.random.normal(35, 8, 100),
            'pressure': np.random.normal(1020, 15, 100),
            'vibration': np.random.exponential(0.3, 100),
            'efficiency': np.random.uniform(80, 85, 100),
            'uptime': 1.0
        }),
        
        'device_C': pd.DataFrame({ # Critical device
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'temperature': np.random.normal(45, 12, 100),
            'pressure': np.random.normal(1050, 25, 100),
            'vibration': np.random.exponential(0.8, 100),
            'efficiency': np.random.uniform(60, 70, 100),
            'uptime': 1.0
        })
    }
    
    print("\n=== DIGITAL TWIN HEALTH SCORE CALCULATOR DEMO ===\n")
    
    # --- REFINED DEMO ---
    # Prime the history with the first 99 data points
    print("1. Priming device history...")
    for device_id, data in device_data.items():
        # Get all but the last row
        history_data = data.iloc[:-1]
        # We can pass the whole history at once
        # In a real system, this would be a loop
        health_calculator.calculate_overall_health_score(history_data, device_id)
        
    print("History primed.")
    
    # 1. Calculate individual device health scores using *only the last row*
    print("\n2. Calculating individual device health scores (real-time):")
    individual_results = {}
    
    for device_id, data in device_data.items():
        # Use only the *last row* for a "real-time" score
        real_time_data = data.iloc[-1:] 
        
        # The function will automatically pull history for this device_id
        result = health_calculator.calculate_overall_health_score(real_time_data, device_id)
        individual_results[device_id] = result
        
        print(f"    {device_id}:")
        print(f"        Health Score: {result['overall_score']:.3f} (Method: {result['calculation_method']})")
        print(f"        (Formula: {result['formula_score']:.3f}, Model: {result.get('model_score', 'N/A')})")
        print(f"        Status: {result['health_status']}")
        print(f"        Risk Level: {result['risk_assessment']['overall_risk_level']}")
        print(f"        (Used {result['historical_data_points']} data points for component scores)")
        print()
    
    # 2. Calculate fleet health score (using latest data for each)
    print("3. Calculating fleet health score (using latest data):")
    # We already calculated the latest scores, so we can just use the results
    # But for a real demo, we'd call the fleet function
    latest_fleet_data = {dev_id: data.iloc[-1:] for dev_id, data in device_data.items()}
    fleet_result = health_calculator.calculate_fleet_health_score(latest_fleet_data)
    
    print(f"    Fleet Health Score: {fleet_result['fleet_health_score']:.3f}")
    print(f"    Fleet Status: {fleet_result['fleet_health_status']}")
    print(f"    Healthy Devices: {fleet_result['fleet_analytics']['healthy_devices']}")
    print(f"    Warning Devices: {fleet_result['fleet_analytics']['warning_devices']}")
    print(f"    Critical Devices: {fleet_result['fleet_analytics']['critical_devices']}")
    print()
    
    # 3. Show best and worst performing devices
    print("4. Device Performance Ranking:")
    print("    Best Performing:")
    for device in fleet_result['best_performing_devices'][:3]:
        print(f"        {device['device_id']}: {device['score']:.3f}")
    
    print("    Worst Performing:")
    for device in fleet_result['worst_performing_devices'][-3:]:
        print(f"        {device['device_id']}: {device['score']:.3f}")
    print()
    
    # 4. Show recommendations
    print("5. Fleet Recommendations:")
    for i, rec in enumerate(fleet_result['fleet_recommendations'][:5], 1):
        print(f"    {i}. [{rec['priority'].upper()}] {rec['action']}")
        print(f"        Timeline: {rec['timeframe']}")
    print()
    
    # 5. Risk assessment summary
    print("6. Fleet Risk Assessment:")
    risk = fleet_result['fleet_risk_assessment']
    print(f"    Fleet Risk Level: {risk['fleet_risk_level']}")
    print(f"    High Risk Devices: {len(risk['high_risk_devices'])}")
    print(f"    Medium Risk Devices: {len(risk['medium_risk_devices'])}")
    print(f"    Low Risk Devices: {len(risk['low_risk_devices'])}")
    print()
    
    # 6. Statistics
    print("7. Health Scoring Statistics:")
    stats = health_calculator.get_health_statistics()
    print(f"    Total health records: {stats['history_summary']['total_health_records']}")
    print(f"    Devices tracked: {stats['history_summary']['devices_tracked']}")
    print(f"    Model Loaded: {stats['configuration_summary']['model_loaded']}")
    print(f"    Baseline metrics (EMAs): {stats['configuration_summary']['baseline_metrics']}")
    
    print("\n=== HEALTH SCORE ANALYSIS COMPLETED ===")