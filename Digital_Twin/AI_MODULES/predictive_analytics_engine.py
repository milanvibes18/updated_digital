import numpy as np
import pandas as pd
import logging
import joblib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
import sys

# --- Added from New File ---
# Add project root to path to find AI_MODULES
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from AI_MODULES.secure_database_manager import SecureDatabaseManager
except ImportError:
    print("Warning: SecureDatabaseManager not found. Retraining will not work.")
    SecureDatabaseManager = None
# --- End Add ---

warnings.filterwarnings('ignore')

# Machine Learning imports (from Existing File)
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report
from sklearn.cluster import DBSCAN, KMeans
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns

class PredictiveAnalyticsEngine:
    """
    Advanced predictive analytics engine for Digital Twin system.
    Provides anomaly detection, failure prediction, trend analysis, optimization,
    and automated retraining.
    """
    
    # --- Merged __init__ ---
    def __init__(self, model_path="ANALYTICS/models/", cache_path="ANALYTICS/analysis_cache/"):
        self.model_path = Path(model_path)
        self.cache_path = Path(cache_path)
        self.logger = self._setup_logging()
        
        # Create directories
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.model_metadata = {}
        
        # Analysis cache (from Existing File)
        self.analysis_cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Configuration (from Existing File)
        self.config = {
            'anomaly_detection': {
                'contamination': 0.1,
                'n_estimators': 100,
                'random_state': 42
            },
            'time_series': {
                'seasonality_period': 24,
                'forecast_horizon': 12,
                'confidence_interval': 0.95
            },
            'classification': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'regression': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
        }
        
        # --- Added from New File ---
        # Initialize db_manager for retraining
        self.db_manager = SecureDatabaseManager() if SecureDatabaseManager else None
        
        # Load existing models
        self._load_existing_models()
        
    def _setup_logging(self):
        """Setup logging for analytics engine."""
        logger = logging.getLogger('PredictiveAnalyticsEngine')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Ensure logs directory exists
            Path('LOGS').mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler('LOGS/digital_twin_analytics.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_existing_models(self):
        """Load existing models from disk."""
        try:
            model_files = list(self.model_path.glob("*.pkl"))
            
            for model_file in model_files:
                model_name = model_file.stem
                try:
                    model_data = joblib.load(model_file)
                    if isinstance(model_data, dict):
                        self.models[model_name] = model_data.get('model')
                        self.scalers[model_name] = model_data.get('scaler')
                        self.model_metadata[model_name] = model_data.get('metadata', {})
                    else:
                        self.models[model_name] = model_data
                    
                    self.logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    self.logger.error(f"Failed to load model {model_name}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
    
    def _save_model(self, model_name: str, model: Any, scaler: Any = None, metadata: Dict = None):
        """Save model to disk."""
        try:
            model_data = {
                'model': model,
                'scaler': scaler,
                'metadata': metadata or {}
            }
            
            model_file = self.model_path / f"{model_name}.pkl"
            joblib.dump(model_data, model_file)
            
            self.models[model_name] = model
            if scaler:
                self.scalers[model_name] = scaler
            if metadata:
                self.model_metadata[model_name] = metadata
                
            self.logger.info(f"Model saved: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model {model_name}: {e}")
    
    # --- Caching Methods (Kept from Existing File) ---
    def _get_from_cache(self, cache_key: str):
        """Get data from cache if valid."""
        try:
            if cache_key in self.analysis_cache:
                timestamp = self.cache_timestamps.get(cache_key, 0)
                if datetime.now().timestamp() - timestamp < self.cache_ttl:
                    return self.analysis_cache[cache_key]
            return None
        except Exception as e:
            self.logger.error(f"Cache retrieval error: {e}")
            return None
    
    def _set_cache(self, cache_key: str, data: Any):
        """Set data in cache."""
        try:
            self.analysis_cache[cache_key] = data
            self.cache_timestamps[cache_key] = datetime.now().timestamp()
        except Exception as e:
            self.logger.error(f"Cache set error: {e}")
    
    # --- Data Preparation (Kept from Existing File - More Robust) ---
    def prepare_data(self, data: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepare data for machine learning.
        """
        try:
            # Make a copy to avoid modifying original
            df = data.copy()
            
            # Handle missing values
            df = df.fillna(df.mean(numeric_only=True))
            
            # Convert datetime columns
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass
            
            # Extract datetime features if datetime columns exist
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            for col in datetime_cols:
                df[f'{col}_hour'] = df[col].dt.hour
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_weekday'] = df[col].dt.weekday
                df = df.drop(columns=[col])
            
            # Handle categorical variables
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col != target_column:
                    df = pd.get_dummies(df, columns=[col], drop_first=True)
            
            # Separate features and target
            if target_column and target_column in df.columns:
                target = df[target_column]
                features = df.drop(columns=[target_column])
            else:
                target = None
                features = df
            
            # Select only numeric columns
            features = features.select_dtypes(include=[np.number])
            
            self.logger.info(f"Data prepared: {features.shape[0]} rows, {features.shape[1]} features")
            return features, target
            
        except Exception as e:
            self.logger.error(f"Data preparation error: {e}")
            raise
    
    # --- Anomaly Detection (Merged) ---
    
    def train_anomaly_detector(self, data: pd.DataFrame, model_name: str = "anomaly_detector") -> Dict:
        """
        Train anomaly detection model using Isolation Forest. (from Existing File)
        """
        try:
            cache_key = f"anomaly_training_{hash(str(data.values.tobytes()))}"
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result
            
            # Prepare data
            features, _ = self.prepare_data(data)
            if features.empty:
                self.logger.warning("No features to train on for anomaly detector.")
                return {'error': 'No numeric features found'}
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Train Isolation Forest
            model = IsolationForest(
                contamination=self.config['anomaly_detection']['contamination'],
                n_estimators=self.config['anomaly_detection']['n_estimators'],
                random_state=self.config['anomaly_detection']['random_state']
            )
            
            model.fit(features_scaled)
            
            # Evaluate on training data
            anomaly_scores = model.decision_function(features_scaled)
            predictions = model.predict(features_scaled)
            
            # Calculate metrics
            anomaly_ratio = (predictions == -1).sum() / len(predictions)
            
            # Save model
            metadata = {
                'trained_at': datetime.now().isoformat(),
                'training_samples': len(features),
                'features': list(features.columns),
                'anomaly_ratio': anomaly_ratio,
                'contamination': self.config['anomaly_detection']['contamination']
            }
            
            self._save_model(model_name, model, scaler, metadata)
            
            result = {
                'model_name': model_name,
                'training_samples': len(features),
                'anomaly_ratio': float(anomaly_ratio),
                'feature_importance': dict(zip(features.columns, np.abs(anomaly_scores))),
                'metadata': metadata
            }
            
            self._set_cache(cache_key, result)
            self.logger.info(f"Anomaly detector trained: {model_name}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Anomaly detection training error: {e}")
            raise
    
    def detect_anomalies(self, data: pd.DataFrame, model_name: str = "anomaly_detector", 
                         threshold: float = None) -> Dict:
        """
        Detect anomalies in data using trained model.
        (Merged: Auto-train and feature reindex from New File added)
        """
        try:
            # --- Merged Feature: Auto-train if model not found ---
            if model_name not in self.models:
                self.logger.warning(f"Model {model_name} not found. Attempting to train...")
                if self.db_manager:
                    # Fetch data from db to train
                    hist_data = self.db_manager.get_health_data_as_dataframe(limit=5000)
                    if not hist_data.empty and len(hist_data) > 50:
                        self.logger.info(f"Training {model_name} with {len(hist_data)} records from DB.")
                        self.train_anomaly_detector(hist_data, model_name)
                    else:
                        raise ValueError(f"Model {model_name} not found and no data to train.")
                else:
                    raise ValueError(f"Model {model_name} not found and DB manager is not available.")
            # --- End Merged Feature ---

            # Prepare data
            features, _ = self.prepare_data(data)
            
            # --- Merged Feature: Feature matching ---
            model_features = self.model_metadata.get(model_name, {}).get('features', [])
            if model_features:
                # Align columns, filling missing ones with 0 (or mean/median if appropriate)
                features = features.reindex(columns=model_features, fill_value=0)
            
            if features.empty:
                self.logger.warning("No features to predict on for anomaly detector.")
                return {'anomaly_count': 0, 'anomalies': [], 'anomaly_percentage': 0, 'anomaly_indices': []}
            # --- End Merged Feature ---

            # Scale features using saved scaler
            if model_name in self.scalers:
                features_scaled = self.scalers[model_name].transform(features)
            else:
                features_scaled = features.values
            
            # Get model
            model = self.models[model_name]
            
            # Make predictions
            predictions = model.predict(features_scaled)
            anomaly_scores = model.decision_function(features_scaled)
            
            # Apply custom threshold if provided
            if threshold:
                predictions = (anomaly_scores < threshold).astype(int)
                predictions[predictions == 1] = -1  # Convert to anomaly format
                predictions[predictions == 0] = 1
            
            # Identify anomalies
            anomaly_indices = np.where(predictions == -1)[0]
            anomalies = data.iloc[anomaly_indices].copy()
            anomalies['anomaly_score'] = anomaly_scores[anomaly_indices]
            
            # Calculate statistics
            anomaly_count = len(anomaly_indices)
            anomaly_percentage = (anomaly_count / len(data)) * 100 if len(data) > 0 else 0
            
            result = {
                'anomaly_count': anomaly_count,
                'anomaly_percentage': anomaly_percentage,
                'anomaly_indices': anomaly_indices.tolist(),
                'anomaly_scores': anomaly_scores.tolist(),
                'anomalies': anomalies.to_dict('records') if not anomalies.empty else [],
                'summary': {
                    'total_samples': len(data),
                    'normal_samples': len(data) - anomaly_count,
                    'min_score': float(anomaly_scores.min()) if len(anomaly_scores) > 0 else 0,
                    'max_score': float(anomaly_scores.max()) if len(anomaly_scores) > 0 else 0,
                    'mean_score': float(anomaly_scores.mean()) if len(anomaly_scores) > 0 else 0
                }
            }
            
            self.logger.info(f"Anomaly detection completed: {anomaly_count} anomalies found")
            return result
            
        except Exception as e:
            self.logger.error(f"Anomaly detection error: {e}")
            raise
    
    # --- Failure Prediction (Kept from Existing File - Classifier is more robust) ---
    
    def train_failure_predictor(self, data: pd.DataFrame, target_column: str, 
                                model_name: str = "failure_predictor") -> Dict:
        """
        Train failure prediction model. (Using Classifier from Existing File)
        """
        try:
            # Prepare data
            features, target = self.prepare_data(data, target_column)
            if features.empty or target is None:
                self.logger.warning("No features or target for failure predictor.")
                return {'error': 'No features or target'}
            
            # Handle class imbalance for classification
            if target.nunique() < 3: # Check if it's classification
                if target.value_counts(normalize=True).min() < 0.1:
                    self.logger.warning("Imbalanced target variable detected.")
                    # Add logic for SMOTE or class_weight='balanced' if needed
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42, 
                stratify=target if target.nunique() < 10 else None # Stratify for classification
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest classifier
            model = RandomForestClassifier(
                n_estimators=self.config['classification']['n_estimators'],
                max_depth=self.config['classification']['max_depth'],
                random_state=self.config['classification']['random_state'],
                class_weight='balanced' # Good for imbalanced failure data
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # Cross validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            # Feature importance
            feature_importance = dict(zip(features.columns, model.feature_importances_))
            
            # Predictions for detailed metrics
            y_pred = model.predict(X_test_scaled)
            
            # Save model
            metadata = {
                'trained_at': datetime.now().isoformat(),
                'training_samples': len(X_train),
                'features': list(features.columns),
                'train_score': train_score,
                'test_score': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'feature_importance': feature_importance
            }
            
            self._save_model(model_name, model, scaler, metadata)
            
            result = {
                'model_name': model_name,
                'train_score': train_score,
                'test_score': test_score,
                'cv_scores': {
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std(),
                    'scores': cv_scores.tolist()
                },
                'feature_importance': feature_importance,
                'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
                'metadata': metadata
            }
            
            self.logger.info(f"Failure predictor trained: {model_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failure prediction training error: {e}")
            raise
    
    def predict_failure(self, data: pd.DataFrame, model_name: str = "failure_predictor") -> Dict:
        """
        Predict failures using trained model.
        (Merged: Feature reindex from New File added)
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            # Prepare data
            features, _ = self.prepare_data(data)
            
            # --- Merged Feature: Feature matching ---
            model_features = self.model_metadata.get(model_name, {}).get('features', [])
            if model_features:
                features = features.reindex(columns=model_features, fill_value=0)
            
            if features.empty:
                self.logger.warning("No features to predict on for failure predictor.")
                return {'predictions': [], 'failure_probabilities': []}
            # --- End Merged Feature ---

            # Scale features
            if model_name in self.scalers:
                features_scaled = self.scalers[model_name].transform(features)
            else:
                features_scaled = features.values
            
            # Get model
            model = self.models[model_name]
            
            # Make predictions
            predictions = model.predict(features_scaled)
            probabilities = model.predict_proba(features_scaled)
            
            # Get failure probabilities (assuming binary classification, class 1 is failure)
            if probabilities.shape[1] == 2:
                failure_probs = probabilities[:, 1]  # Probability of failure
            else:
                # Handle multiclass by taking prob of highest class
                failure_probs = np.max(probabilities, axis=1)
            
            # Identify high-risk cases
            risk_threshold = 0.7
            high_risk_indices = np.where(failure_probs >= risk_threshold)[0]
            
            result = {
                'predictions': predictions.tolist(),
                'failure_probabilities': failure_probs.tolist(),
                'high_risk_count': len(high_risk_indices),
                'high_risk_indices': high_risk_indices.tolist(),
                'risk_distribution': {
                    'low_risk': int((failure_probs < 0.3).sum()),
                    'medium_risk': int(((failure_probs >= 0.3) & (failure_probs < 0.7)).sum()),
                    'high_risk': int((failure_probs >= 0.7).sum())
                },
                'summary': {
                    'total_samples': len(data),
                    'predicted_failures': int(predictions.sum()), # Assumes 1 is failure
                    'max_failure_probability': float(failure_probs.max()) if len(failure_probs) > 0 else 0,
                    'mean_failure_probability': float(failure_probs.mean()) if len(failure_probs) > 0 else 0
                }
            }
            
            self.logger.info(f"Failure prediction completed: {len(high_risk_indices)} high-risk cases")
            return result
            
        except Exception as e:
            self.logger.error(f"Failure prediction error: {e}")
            raise

    # --- Added from New File: Model Retraining ---
    
    def retrain_models(self):
        """
        Scheduled task to retrain models with fresh data from the database.
        """
        self.logger.info("Starting scheduled model retraining...")
        if not self.db_manager:
            self.logger.error("Database Manager not available. Skipping retraining.")
            return {"status": "error", "message": "Database Manager not available"}

        try:
            # Fetch fresh data (e.g., last 30 days)
            # Using the pandas-native method from the merged db_manager
            data = self.db_manager.get_health_data_as_dataframe(limit=20000) 
            
            if data.empty or len(data) < 100:
                self.logger.warning("Not enough new data to retrain. Skipping.")
                return {"status": "skipped", "message": "Not enough new data"}
            
            results = {}

            # Retrain anomaly detector
            self.logger.info("Retraining anomaly_detector...")
            anomaly_results = self.train_anomaly_detector(data, "anomaly_detector")
            results["anomaly_detector"] = anomaly_results['metadata']

            # Retrain a failure predictor (if target is available)
            if 'failure' in data.columns:
                self.logger.info("Retraining failure_predictor...")
                failure_results = self.train_failure_predictor(data, 'failure', "failure_predictor")
                results["failure_predictor"] = failure_results['metadata']
            else:
                self.logger.info("Skipping failure_predictor: 'failure' column not in data.")

            self.logger.info("Scheduled model retraining completed.")
            return {"status": "success", "results": results}
        
        except Exception as e:
            self.logger.error(f"Scheduled retraining failed: {e}")
            return {"status": "error", "message": str(e)}

    # --- All Advanced Methods Below Kept from Existing File ---

    def time_series_analysis(self, data: pd.DataFrame, value_column: str, 
                             time_column: str = None, forecast_steps: int = 12) -> Dict:
        """
        Perform comprehensive time series analysis.
        """
        try:
            cache_key = f"timeseries_{value_column}_{hash(str(data.values.tobytes()))}"
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result
            
            # Prepare data
            df = data.copy()
            if time_column and time_column in df.columns:
                df[time_column] = pd.to_datetime(df[time_column])
                df.set_index(time_column, inplace=True)
            
            ts = df[value_column].dropna()
            
            # Basic statistics
            basic_stats = {
                'count': len(ts),
                'mean': float(ts.mean()),
                'std': float(ts.std()),
                'min': float(ts.min()),
                'max': float(ts.max()),
                'median': float(ts.median())
            }
            
            # Trend analysis
            trend_analysis = self._analyze_trend(ts)
            
            # Seasonality detection
            seasonality_analysis = self._detect_seasonality(ts)
            
            # Anomaly detection in time series
            ts_anomalies = self._detect_time_series_anomalies(ts)
            
            # Forecasting
            forecast_result = self._forecast_time_series(ts, forecast_steps)
            
            result = {
                'basic_statistics': basic_stats,
                'trend_analysis': trend_analysis,
                'seasonality_analysis': seasonality_analysis,
                'anomalies': ts_anomalies,
                'forecast': forecast_result,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            self._set_cache(cache_key, result)
            self.logger.info(f"Time series analysis completed for {value_column}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Time series analysis error: {e}")
            raise
    
    def _analyze_trend(self, ts: pd.Series) -> Dict:
        """Analyze trend in time series."""
        try:
            # Linear regression for trend
            x = np.arange(len(ts))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, ts.values)
            
            # Determine trend direction
            if abs(slope) < std_err:
                trend_direction = "stable"
            elif slope > 0:
                trend_direction = "increasing"
            else:
                trend_direction = "decreasing"
            
            return {
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_value**2),
                'p_value': float(p_value),
                'trend_direction': trend_direction,
                'trend_strength': abs(float(r_value))
            }
            
        except Exception as e:
            self.logger.error(f"Trend analysis error: {e}")
            return {}
    
    def _detect_seasonality(self, ts: pd.Series) -> Dict:
        """Detect seasonality in time series."""
        try:
            if len(ts) < 2 * self.config['time_series']['seasonality_period']:
                return {'seasonal': False, 'reason': 'Insufficient data'}
            
            # Seasonal decomposition
            decomposition = seasonal_decompose(
                ts, 
                period=self.config['time_series']['seasonality_period'],
                extrapolate_trend='freq'
            )
            
            # Calculate seasonal strength
            seasonal_var = np.var(decomposition.seasonal.dropna())
            residual_var = np.var(decomposition.resid.dropna())
            if (seasonal_var + residual_var) == 0:
                seasonal_strength = 0
            else:
                seasonal_strength = seasonal_var / (seasonal_var + residual_var)
            
            is_seasonal = seasonal_strength > 0.1
            
            return {
                'seasonal': is_seasonal,
                'seasonal_strength': float(seasonal_strength),
                'period': self.config['time_series']['seasonality_period'],
                'seasonal_pattern': decomposition.seasonal.dropna().tolist()[-24:] if is_seasonal else []
            }
            
        except Exception as e:
            self.logger.error(f"Seasonality detection error: {e}")
            return {'seasonal': False, 'reason': str(e)}
    
    def _detect_time_series_anomalies(self, ts: pd.Series) -> Dict:
        """Detect anomalies in time series using statistical methods."""
        try:
            # Z-score based detection
            z_scores = np.abs(stats.zscore(ts.dropna()))
            z_anomalies = np.where(z_scores > 3)[0]
            
            # IQR based detection
            Q1 = ts.quantile(0.25)
            Q3 = ts.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_anomalies = ts[(ts < lower_bound) | (ts > upper_bound)].index
            
            return {
                'z_score_anomalies': {
                    'count': len(z_anomalies),
                    'indices': z_anomalies.tolist(),
                    'threshold': 3.0
                },
                'iqr_anomalies': {
                    'count': len(iqr_anomalies),
                    'indices': iqr_anomalies.tolist(),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Time series anomaly detection error: {e}")
            return {}
    
    def _forecast_time_series(self, ts: pd.Series, steps: int) -> Dict:
        """Forecast time series using multiple methods."""
        try:
            forecasts = {}
            
            # Simple moving average
            window_size = min(12, len(ts) // 4)
            if window_size > 0:
                ma_forecast = ts.rolling(window=window_size).mean().iloc[-1]
                forecasts['moving_average'] = [float(ma_forecast)] * steps
            
            # Exponential smoothing
            if len(ts) > 10:
                try:
                    exp_model = ExponentialSmoothing(
                        ts, 
                        trend='add', 
                        seasonal='add' if len(ts) > 24 else None,
                        seasonal_periods=12 if len(ts) > 24 else None
                    ).fit(optimized=True)
                    
                    exp_forecast = exp_model.forecast(steps)
                    forecasts['exponential_smoothing'] = exp_forecast.tolist()
                    
                except:
                    # Fallback to simple exponential smoothing
                    try:
                        exp_model = ExponentialSmoothing(ts, trend='add').fit()
                        exp_forecast = exp_model.forecast(steps)
                        forecasts['exponential_smoothing'] = exp_forecast.tolist()
                    except Exception as e:
                        self.logger.warning(f"Fallback ExpSmoothing failed: {e}")
            
            # ARIMA (simplified)
            if len(ts) > 50:
                try:
                    arima_model = ARIMA(ts, order=(1, 1, 1)).fit()
                    arima_forecast = arima_model.forecast(steps)
                    forecasts['arima'] = arima_forecast.tolist()
                except:
                    pass
            
            # Calculate ensemble forecast
            if forecasts:
                ensemble = np.mean(list(forecasts.values()), axis=0)
                forecasts['ensemble'] = ensemble.tolist()
            
            return {
                'forecasts': forecasts,
                'forecast_steps': steps,
                'forecast_horizon': f"{steps} periods",
                'confidence': 0.8 if len(forecasts) > 1 else 0.6
            }
            
        except Exception as e:
            self.logger.error(f"Time series forecasting error: {e}")
            return {}
    
    def cluster_analysis(self, data: pd.DataFrame, n_clusters: int = None, 
                         method: str = 'kmeans') -> Dict:
        """
        Perform cluster analysis on data.
        """
        try:
            # Prepare data
            features, _ = self.prepare_data(data)
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            if method == 'kmeans':
                # Determine optimal number of clusters if not provided
                if n_clusters is None:
                    n_clusters = self._find_optimal_clusters(features_scaled)
                
                # Perform K-means clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features_scaled)
                
                # Calculate cluster centers in original space
                cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
                
                result = {
                    'method': 'kmeans',
                    'n_clusters': n_clusters,
                    'cluster_labels': cluster_labels.tolist(),
                    'cluster_centers': cluster_centers.tolist(),
                    'inertia': float(kmeans.inertia_),
                    'silhouette_score': None  # Could add silhouette analysis
                }
                
            elif method == 'dbscan':
                # Perform DBSCAN clustering
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                cluster_labels = dbscan.fit_predict(features_scaled)
                
                n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                n_noise = list(cluster_labels).count(-1)
                
                result = {
                    'method': 'dbscan',
                    'n_clusters': n_clusters_found,
                    'cluster_labels': cluster_labels.tolist(),
                    'n_noise_points': n_noise,
                    'eps': 0.5,
                    'min_samples': 5
                }
            
            # Add cluster statistics
            cluster_stats = self._calculate_cluster_stats(features, cluster_labels)
            result['cluster_statistics'] = cluster_stats
            
            self.logger.info(f"Cluster analysis completed: {result['n_clusters']} clusters found")
            return result
            
        except Exception as e:
            self.logger.error(f"Cluster analysis error: {e}")
            raise
    
    def _find_optimal_clusters(self, data: np.ndarray, max_clusters: int = 10) -> int:
        """Find optimal number of clusters using elbow method."""
        try:
            inertias = []
            k_range = range(1, min(max_clusters + 1, len(data)))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(data)
                inertias.append(kmeans.inertia_)
            
            # Simple elbow detection
            if len(inertias) >= 3:
                # Calculate rate of change
                deltas = np.diff(inertias)
                delta_deltas = np.diff(deltas)
                
                # Find elbow point (maximum change in rate of change)
                if len(delta_deltas) > 0:
                    elbow_idx = np.argmax(delta_deltas) + 2
                    return min(elbow_idx, max_clusters)
            
            # Default to 3 clusters
            return 3
            
        except Exception as e:
            self.logger.error(f"Optimal cluster finding error: {e}")
            return 3
    
    def _calculate_cluster_stats(self, features: pd.DataFrame, labels: np.ndarray) -> Dict:
        """Calculate statistics for each cluster."""
        try:
            stats = {}
            unique_labels = np.unique(labels)
            
            for label in unique_labels:
                label_str = f'cluster_{label}'
                if label == -1:  # Noise points in DBSCAN
                    label_str = 'noise'
                    
                cluster_data = features[labels == label]
                
                stats[label_str] = {
                    'size': len(cluster_data),
                    'mean': cluster_data.mean().to_dict(),
                    'std': cluster_data.std().to_dict(),
                    'min': cluster_data.min().to_dict(),
                    'max': cluster_data.max().to_dict()
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Cluster statistics error: {e}")
            return {}
    
    def pattern_recognition(self, data: pd.DataFrame, pattern_type: str = 'peaks') -> Dict:
        """
        Recognize patterns in data.
        """
        try:
            features, _ = self.prepare_data(data)
            
            if pattern_type == 'peaks':
                return self._detect_peaks(features)
            elif pattern_type == 'cycles':
                return self._detect_cycles(features)
            elif pattern_type == 'correlations':
                return self._analyze_correlations(features)
            else:
                raise ValueError(f"Unknown pattern type: {pattern_type}")
                
        except Exception as e:
            self.logger.error(f"Pattern recognition error: {e}")
            raise
    
    def _detect_peaks(self, data: pd.DataFrame) -> Dict:
        """Detect peaks in data columns."""
        try:
            results = {}
            
            for column in data.columns:
                values = data[column].dropna().values
                if len(values) > 10:
                    peaks, properties = find_peaks(values, height=np.mean(values))
                    
                    results[column] = {
                        'peak_count': len(peaks),
                        'peak_indices': peaks.tolist(),
                        'peak_heights': values[peaks].tolist(),
                        'mean_peak_height': float(np.mean(values[peaks])) if len(peaks) > 0 else 0
                    }
            
            return {'pattern_type': 'peaks', 'results': results}
            
        except Exception as e:
            self.logger.error(f"Peak detection error: {e}")
            return {}
    
    def _detect_cycles(self, data: pd.DataFrame) -> Dict:
        """Detect cyclical patterns in data."""
        try:
            results = {}
            
            for column in data.columns:
                values = data[column].dropna().values
                if len(values) > 50:
                    # Simple autocorrelation-based cycle detection
                    autocorr = np.correlate(values, values, mode='full')
                    autocorr = autocorr[autocorr.size // 2:]
                    
                    # Find peaks in autocorrelation
                    peaks, _ = find_peaks(autocorr[1:], height=np.max(autocorr) * 0.1)
                    
                    if len(peaks) > 0:
                        cycle_length = peaks[0] + 1
                        results[column] = {
                            'cycle_detected': True,
                            'cycle_length': int(cycle_length),
                            'cycle_strength': float(autocorr[cycle_length] / autocorr[0])
                        }
                    else:
                        results[column] = {'cycle_detected': False}
            
            return {'pattern_type': 'cycles', 'results': results}
            
        except Exception as e:
            self.logger.error(f"Cycle detection error: {e}")
            return {}
    
    def _analyze_correlations(self, data: pd.DataFrame) -> Dict:
        """Analyze correlations between variables."""
        try:
            # Calculate correlation matrix
            corr_matrix = data.corr()
            
            # Find strong correlations
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:  # Strong correlation threshold
                        strong_correlations.append({
                            'variable1': corr_matrix.columns[i],
                            'variable2': corr_matrix.columns[j],
                            'correlation': float(corr_val),
                            'strength': 'strong' if abs(corr_val) > 0.8 else 'moderate'
                        })
            
            return {
                'pattern_type': 'correlations',
                'correlation_matrix': corr_matrix.to_dict(),
                'strong_correlations': strong_correlations,
                'summary': {
                    'total_correlations': len(strong_correlations),
                    'strongest_correlation': max(strong_correlations, 
                                                 key=lambda x: abs(x['correlation']), 
                                                 default={'correlation': 0})['correlation']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Correlation analysis error: {e}")
            return {}
    
    def optimization_analysis(self, data: pd.DataFrame, objective_column: str, 
                              constraint_columns: List[str] = None) -> Dict:
        """
        Perform optimization analysis to find optimal parameter settings.
        """
        try:
            features, target = self.prepare_data(data, objective_column)
            
            if constraint_columns:
                constraint_data = features[constraint_columns]
            else:
                constraint_data = None
            
            # Train a model to understand the relationship
            model = RandomForestRegressor(
                n_estimators=100, 
                random_state=42
            )
            model.fit(features, target)
            
            # Feature importance for optimization insights
            feature_importance = dict(zip(features.columns, model.feature_importances_))
            
            # Find best performing cases
            best_indices = target.nlargest(10).index
            best_configs = features.loc[best_indices]
            
            # Statistical analysis of optimal ranges
            optimization_ranges = {}
            for column in features.columns:
                optimization_ranges[column] = {
                    'optimal_min': float(best_configs[column].min()),
                    'optimal_max': float(best_configs[column].max()),
                    'optimal_mean': float(best_configs[column].mean()),
                    'current_range': [float(features[column].min()), float(features[column].max())]
                }
            
            result = {
                'objective_column': objective_column,
                'best_performance': float(target.max()),
                'worst_performance': float(target.min()),
                'mean_performance': float(target.mean()),
                'feature_importance': feature_importance,
                'optimization_ranges': optimization_ranges,
                'best_configurations': best_configs.to_dict('records'),
                'recommendations': self._generate_optimization_recommendations(
                    feature_importance, optimization_ranges
                )
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Optimization analysis error: {e}")
            raise
    
    def _generate_optimization_recommendations(self, feature_importance: Dict, 
                                               optimization_ranges: Dict) -> List[Dict]:
        """Generate optimization recommendations based on analysis."""
        try:
            recommendations = []
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), 
                                     key=lambda x: x[1], reverse=True)
            
            for feature, importance in sorted_features[:5]:  # Top 5 most important
                if importance > 0.1:  # Significant importance threshold
                    rec = {
                        'parameter': feature,
                        'importance': float(importance),
                        'recommendation': f"Optimize {feature} within range "
                                        f"[{optimization_ranges[feature]['optimal_min']:.3f}, "
                                        f"{optimization_ranges[feature]['optimal_max']:.3f}]",
                        'current_optimal_mean': optimization_ranges[feature]['optimal_mean'],
                        'priority': 'high' if importance > 0.2 else 'medium'
                    }
                    recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendation generation error: {e}")
            return []
    
    # --- Utility Methods (Kept from Existing File) ---

    def get_model_info(self, model_name: str = None) -> Dict:
        """Get information about trained models."""
        try:
            if model_name:
                if model_name in self.model_metadata:
                    return self.model_metadata[model_name]
                else:
                    return {}
            else:
                return {
                    'available_models': list(self.models.keys()),
                    'model_metadata': self.model_metadata
                }
                
        except Exception as e:
            self.logger.error(f"Get model info error: {e}")
            return {}
    
    def clear_cache(self):
        """Clear analysis cache."""
        try:
            self.analysis_cache.clear()
            self.cache_timestamps.clear()
            self.logger.info("Analysis cache cleared")
        except Exception as e:
            self.logger.error(f"Cache clear error: {e}")
    
    def export_analysis_results(self, results: Dict, filename: str = None) -> str:
        """Export analysis results to JSON file."""
        try:
            if filename is None:
                filename = f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            filepath = self.cache_path / filename
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Results exported to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Export error: {e}")
            raise
    
    def load_analysis_results(self, filename: str) -> Dict:
        """Load analysis results from JSON file."""
        try:
            filepath = self.cache_path / filename
            
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            self.logger.info(f"Results loaded from: {filepath}")
            return results
            
        except Exception as e:
            self.logger.error(f"Load error: {e}")
            raise


# Example usage and demonstration (Kept from Existing File)
if __name__ == "__main__":
    # Setup basic logging to console for testing
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize analytics engine
    analytics = PredictiveAnalyticsEngine()
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'temperature': 20 + 5 * np.sin(np.arange(1000) * 2 * np.pi / 24) + np.random.normal(0, 1, 1000),
        'pressure': 1013 + np.random.normal(0, 5, 1000),
        'vibration': 0.1 + np.random.exponential(0.05, 1000),
        'failure': np.random.choice([0, 1], 1000, p=[0.9, 0.1])
    })
    
    print("Training anomaly detector...")
    anomaly_result = analytics.train_anomaly_detector(sample_data[['temperature', 'pressure', 'vibration']])
    print(f"Anomaly detector trained. Anomaly ratio: {anomaly_result['anomaly_ratio']:.3f}")
    
    print("\nTraining failure predictor...")
    failure_result = analytics.train_failure_predictor(sample_data, 'failure')
    print(f"Failure predictor trained. Test score: {failure_result['test_score']:.3f}")
    
    print("\nPerforming time series analysis...")
    ts_result = analytics.time_series_analysis(sample_data, 'temperature', 'timestamp')
    print(f"Time series analysis completed. Trend: {ts_result['trend_analysis']['trend_direction']}")
    
    print("\nPerforming cluster analysis...")
    cluster_result = analytics.cluster_analysis(sample_data[['temperature', 'pressure', 'vibration']])
    print(f"Cluster analysis completed. Found {cluster_result['n_clusters']} clusters")
    
    print("\nDetecting patterns...")
    pattern_result = analytics.pattern_recognition(sample_data[['temperature']], 'peaks')
    print(f"Pattern recognition completed. Found peaks in temperature data")
    
    print("\nPerforming optimization analysis...")
    sample_data['efficiency'] = (
        100 - abs(sample_data['temperature'] - 22) * 2 
        - abs(sample_data['pressure'] - 1013) * 0.01 
        - sample_data['vibration'] * 50
        + np.random.normal(0, 5, 1000)
    )
    opt_result = analytics.optimization_analysis(sample_data, 'efficiency')
    print(f"Optimization analysis completed. Best performance: {opt_result['best_performance']:.2f}")
    
    # Test prediction on new data
    print("\nTesting predictions on new data...")
    new_data = sample_data.tail(100)
    
    anomaly_pred = analytics.detect_anomalies(new_data[['temperature', 'pressure', 'vibration']])
    print(f"Anomaly detection: {anomaly_pred['anomaly_count']} anomalies found")
    
    failure_pred = analytics.predict_failure(new_data[['temperature', 'pressure', 'vibration']])
    print(f"Failure prediction: {failure_pred['high_risk_count']} high-risk cases")

    # --- Test New Retraining Feature ---
    print("\nTesting model retraining...")
    if analytics.db_manager:
        print("DB Manager found. Mocking data insertion and retraining...")
        # This is a test, so we can't assume the DB is populated.
        # We'll just call retrain() and let it fail gracefully if no data.
        # In a real scenario, the DB would have data.
        retrain_status = analytics.retrain_models()
        print(f"Retraining status: {retrain_status.get('status')}")
    else:
        print("DB Manager not found. Skipping retraining test.")
    # --- End Test ---
    
    # Export results
    all_results = {
        'anomaly_training': anomaly_result,
        'failure_training': failure_result,
        'time_series': ts_result,
        'clustering': cluster_result,
        'patterns': pattern_result,
        'optimization': opt_result,
        'predictions': {
            'anomalies': anomaly_pred,
            'failures': failure_pred
        }
    }
    
    export_path = analytics.export_analysis_results(all_results)
    print(f"\nAll results exported to: {export_path}")
    
    print("\nAnalytics engine demonstration completed!")