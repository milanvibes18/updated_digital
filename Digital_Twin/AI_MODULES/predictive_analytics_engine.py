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

# --- MLflow Import (FIX 4) ---
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.tensorflow
    MLFLOW_AVAILABLE = True
    print("MLflow loaded successfully.")
except ImportError:
    print("Warning: mlflow not found. Model tracking will be disabled.")
    MLFLOW_AVAILABLE = False
# --- End Add ---

# Add project root to path to find AI_MODULES
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from AI_MODULES.secure_database_manager import SecureDatabaseManager
except ImportError:
    print("Warning: SecureDatabaseManager not found. Retraining will not work.")
    SecureDatabaseManager = None

# --- Celery Import (FIX 1) ---
# This import is safe because celery_app is defined at the module level
# in enhanced_flask_app_v2.py *before* the app instance is created.
try:
    from Digital_Twin.WEB_APPLICATION.enhanced_flask_app_v2 import celery_app
except ImportError:
    print("Warning: Celery app not found. Tasks will not be registered.")
    # Define a dummy decorator if celery isn't available
    class DummyCeleryApp:
        def task(self, *args, **kwargs):
            def decorator(f):
                return f
            return decorator
    celery_app = DummyCeleryApp()
# --- End Add ---

warnings.filterwarnings('ignore')

# --- Machine Learning imports ---
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, classification_report,
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.cluster import DBSCAN, KMeans
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns

# --- Deep Learning Imports (NEW) ---
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
    from tensorflow.keras.callbacks import EarlyStopping
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    tf.get_logger().setLevel('ERROR')
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow loaded successfully.")
except ImportError:
    print("Warning: TensorFlow not found. LSTM and Autoencoder features will be disabled.")
    TENSORFLOW_AVAILABLE = False


class PredictiveAnalyticsEngine:
    """
    Advanced predictive analytics engine for Digital Twin system.
    Provides anomaly detection (IsolationForest, Autoencoder), 
    failure prediction, time-series forecasting (ARIMA, LSTM), 
    and automated retraining.
    """
    
    def __init__(self, model_path="ANALYTICS/models/", cache_path="ANALYTICS/analysis_cache/"):
        self.model_path = Path(model_path)
        self.cache_path = Path(cache_path)
        self.logger = self._setup_logging()
        
        # Create directories
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Model storage
        self.models = {}
        self.preprocessors = {}
        self.model_metadata = {}
        
        # Analysis cache
        self.analysis_cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Configuration
        self.config = {
            'anomaly_detection': {
                'contamination': 0.1,
                'n_estimators': 100,
                'random_state': 42
            },
            'time_series': {
                'seasonality_period': 24,
                'forecast_horizon': 12,
                'confidence_interval': 0.95,
                'lstm_n_steps_in': 24,
                'autoencoder_n_steps': 10
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
            },
            'deep_learning': {
                'epochs': 50,
                'batch_size': 32,
                'validation_split': 0.2,
                'patience': 5
            }
        }
        
        # Initialize db_manager for retraining
        self.db_manager = SecureDatabaseManager() if SecureDatabaseManager else None
        
        # --- MLflow Setup (FIX 4) ---
        if MLFLOW_AVAILABLE:
            self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
            self._setup_mlflow()
        # --- End Add ---

        # Load existing models
        self._load_existing_models()
        
    def _setup_logging(self):
        """Setup logging for analytics engine."""
        logger = logging.getLogger('PredictiveAnalyticsEngine')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            Path('LOGS').mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler('LOGS/digital_twin_analytics.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    # --- NEW: MLflow Setup Method (FIX 4) ---
    def _setup_mlflow(self):
        """Set up MLflow tracking."""
        if not MLFLOW_AVAILABLE:
            return
        try:
            # Check if we are in a Celery worker. If so, setting tracking URI might be problematic.
            # For this setup, we assume it's set globally or via env vars.
            if "celery" not in sys.argv[0]:
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            self.logger.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
        except Exception as e:
            self.logger.error(f"Failed to set up MLflow: {e}")
    # --- End Add ---

    def _load_existing_models(self):
        """Load existing models from disk."""
        try:
            # Load sklearn models
            model_files = list(self.model_path.glob("*.pkl"))
            for model_file in model_files:
                model_name = model_file.stem
                if "_meta" in model_name: # Skip metadata files for TF models
                    continue
                try:
                    model_data = joblib.load(model_file)
                    if isinstance(model_data, dict):
                        self.models[model_name] = model_data.get('model')
                        if 'preprocessors' in model_data:
                            self.preprocessors[model_name] = model_data.get('preprocessors')
                        else:
                            self.preprocessors[model_name] = {
                                'scaler': model_data.get('scaler'), 'imputer': None
                            }
                        self.model_metadata[model_name] = model_data.get('metadata', {})
                    else:
                        self.models[model_name] = model_data
                    
                    self.logger.info(f"Loaded sklearn model: {model_name}")
                except Exception as e:
                    self.logger.error(f"Failed to load .pkl model {model_name}: {e}")
            
            # Load Keras models (NEW)
            if TENSORFLOW_AVAILABLE:
                keras_model_dirs = [d for d in self.model_path.glob("*") if d.is_dir()]
                for model_dir in keras_model_dirs:
                    model_name = model_dir.stem
                    if model_name in self.models: continue 
                    try:
                        self.models[model_name] = tf.keras.models.load_model(model_dir)
                        metadata_file = self.model_path / f"{model_name}_meta.pkl"
                        if metadata_file.exists():
                            meta_data = joblib.load(metadata_file)
                            if 'preprocessors' in meta_data:
                                self.preprocessors[model_name] = meta_data.get('preprocessors')
                            else:
                                self.preprocessors[model_name] = {
                                    'scaler': meta_data.get('scaler'), 'imputer': None
                                }
                            self.model_metadata[model_name] = meta_data.get('metadata', {})
                        self.logger.info(f"Loaded Keras model: {model_name}")
                    except Exception as e:
                        self.logger.error(f"Failed to load Keras model {model_name}: {e}")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
    
    def _save_model(self, model_name: str, model: Any, scaler: Any = None, imputer: Any = None, metadata: Dict = None):
        """Save model to disk."""
        try:
            preprocessors = {'scaler': scaler, 'imputer': imputer}
            if TENSORFLOW_AVAILABLE and isinstance(model, (tf.keras.Model)):
                model_dir = self.model_path / model_name
                model.save(model_dir)
                metadata_file = self.model_path / f"{model_name}_meta.pkl"
                model_data = {
                    'preprocessors': preprocessors,
                    'metadata': metadata or {}
                }
                joblib.dump(model_data, metadata_file)
                self.models[model_name] = model
                self.preprocessors[model_name] = preprocessors
                if metadata:
                    self.model_metadata[model_name] = metadata
            else:
                model_data = {
                    'model': model,
                    'preprocessors': preprocessors,
                    'metadata': metadata or {}
                }
                model_file = self.model_path / f"{model_name}.pkl"
                joblib.dump(model_data, model_file)
                self.models[model_name] = model
                self.preprocessors[model_name] = preprocessors
                if metadata:
                    self.model_metadata[model_name] = metadata
            self.logger.info(f"Model saved: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to save model {model_name}: {e}")
    
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
    
    def prepare_data(self, data: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare data for machine learning."""
        try:
            df = data.copy()
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass
            
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            for col in datetime_cols:
                df[f'{col}_hour'] = df[col].dt.hour
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_weekday'] = df[col].dt.weekday
                df = df.drop(columns=[col])
            
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col != target_column:
                    df = pd.get_dummies(df, columns=[col], drop_first=True)
            
            if target_column and target_column in df.columns:
                target = df[target_column]
                features = df.drop(columns=[target_column])
            else:
                target = None
                features = df
            
            features = features.select_dtypes(include=[np.number])
            self.logger.info(f"Data prepared: {features.shape[0]} rows, {features.shape[1]} features")
            return features, target
        except Exception as e:
            self.logger.error(f"Data preparation error: {e}")
            raise

    # --- NEW: Modular Preprocessing (FIX 5) ---
    def _fit_preprocessors(self, features: pd.DataFrame) -> Dict:
        """Fits imputer and scaler and returns them in a dict."""
        try:
            # 1. Impute
            imputer = SimpleImputer(strategy='mean')
            features_imputed = imputer.fit_transform(features)
            
            # 2. Scale
            scaler = StandardScaler()
            # Fit on the imputed data
            scaler.fit(features_imputed) 
            
            return {'imputer': imputer, 'scaler': scaler}
        except Exception as e:
            self.logger.error(f"Error fitting preprocessors: {e}")
            raise

    def _transform_data(self, features: pd.DataFrame, preprocessors: Dict) -> np.ndarray:
        """Applies fitted imputer and scaler."""
        try:
            imputer = preprocessors['imputer']
            scaler = preprocessors['scaler']
            
            # 1. Impute
            features_imputed = imputer.transform(features)
            
            # 2. Scale
            features_scaled = scaler.transform(features_imputed)
            
            return features_scaled
        except Exception as e:
            self.logger.error(f"Error transforming data: {e}")
            raise
    # --- End Add ---

    def _create_lstm_sequences(self, data: np.ndarray, n_steps_in: int, n_steps_out: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create supervised sequences for LSTM forecasting."""
        X, y = [], []
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        n_features = data.shape[1]
        for i in range(len(data)):
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            if out_end_ix > len(data):
                break
            seq_x = data[i:end_ix, :]
            seq_y = data[end_ix:out_end_ix, 0]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def _create_autoencoder_sequences(self, data: np.ndarray, n_steps: int) -> np.ndarray:
        """Create unsupervised sequences for Autoencoder."""
        X = []
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        for i in range(len(data) - n_steps + 1):
            seq_x = data[i:(i + n_steps), :]
            X.append(seq_x)
        return np.array(X)

    # --- Anomaly Detection (Isolation Forest) ---
    
    @celery_app.task(bind=True, name='analytics.train_anomaly_detector')
    def train_anomaly_detector(self, data: pd.DataFrame, model_name: str = "anomaly_detector") -> Dict:
        """
        Train anomaly detection model (Celery Task).
        - MODULARIZED: Uses _fit_preprocessors, _transform_data
        - TRACKED: Logs params and metrics to MLflow
        """
        # MLflow Tracking (FIX 4)
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_name=f"train_anomaly_detector_{model_name}") as run:
            try:
                cache_key = f"anomaly_training_{hash(str(data.values.tobytes()))}"
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    return cached_result
                
                features, _ = self.prepare_data(data)
                if features.empty:
                    msg = "No numeric features found"
                    self.logger.warning(msg)
                    if MLFLOW_AVAILABLE: mlflow.end_run(status="FAILED")
                    return {'error': msg}
                
                # --- FIX 5: Modular Preprocessing ---
                preprocessors = self._fit_preprocessors(features)
                features_scaled = self._transform_data(features, preprocessors)
                # --- END FIX 5 ---
                
                # Train Isolation Forest
                model_config = self.config['anomaly_detection']
                model = IsolationForest(
                    contamination=model_config['contamination'],
                    n_estimators=model_config['n_estimators'],
                    random_state=model_config['random_state']
                )
                model.fit(features_scaled)
                
                anomaly_scores = model.decision_function(features_scaled)
                predictions = model.predict(features_scaled)
                anomaly_ratio = (predictions == -1).sum() / len(predictions)
                
                # --- FIX 4: MLflow Logging ---
                if MLFLOW_AVAILABLE:
                    mlflow.log_params(model_config)
                    mlflow.log_param("training_samples", len(features))
                    mlflow.log_param("features_count", len(features.columns))
                    mlflow.log_metrics({'anomaly_ratio': anomaly_ratio})
                    mlflow.sklearn.log_model(model, "model")
                # --- END FIX 4 ---

                metadata = {
                    'model_type': 'IsolationForest',
                    'trained_at': datetime.now().isoformat(),
                    'training_samples': len(features),
                    'features': list(features.columns),
                    'anomaly_ratio': anomaly_ratio,
                    'contamination': model_config['contamination'],
                    'mlflow_run_id': run.info.run_id if MLFLOW_AVAILABLE else None
                }
                
                self._save_model(model_name, model, preprocessors['scaler'], preprocessors['imputer'], metadata)
                
                result = {
                    'model_name': model_name,
                    'training_samples': len(features),
                    'anomaly_ratio': float(anomaly_ratio),
                    'feature_importance': dict(zip(features.columns, np.abs(anomaly_scores))),
                    'metadata': metadata
                }
                
                self._set_cache(cache_key, result)
                self.logger.info(f"Anomaly detector trained: {model_name}")
                if MLFLOW_AVAILABLE: mlflow.end_run(status="FINISHED")
                return result
                
            except Exception as e:
                self.logger.error(f"Anomaly detection training error: {e}")
                if MLFLOW_AVAILABLE: mlflow.end_run(status="FAILED")
                raise

    @celery_app.task(bind=True, name='analytics.detect_anomalies')
    def detect_anomalies(self, data: pd.DataFrame, model_name: str = "anomaly_detector", 
                         threshold: float = None) -> Dict:
        """
        Detect anomalies in data (Celery Task).
        - MODULARIZED: Uses _transform_data
        """
        try:
            if model_name not in self.models:
                self.logger.warning(f"Model {model_name} not found. Attempting to train...")
                if self.db_manager:
                    hist_data = self.db_manager.get_health_data_as_dataframe(limit=5000)
                    if not hist_data.empty and len(hist_data) > 50:
                        self.logger.info(f"Training {model_name} with {len(hist_data)} records from DB.")
                        # Call task synchronously for dependency
                        self.train_anomaly_detector(hist_data, model_name) 
                    else:
                        raise ValueError(f"Model {model_name} not found and no data to train.")
                else:
                    raise ValueError(f"Model {model_name} not found and DB manager is not available.")

            features, _ = self.prepare_data(data)
            
            model_features = self.model_metadata.get(model_name, {}).get('features', [])
            if model_features:
                features = features.reindex(columns=model_features, fill_value=0)
            
            if features.empty:
                self.logger.warning("No features to predict on for anomaly detector.")
                return {'anomaly_count': 0, 'anomalies': [], 'anomaly_percentage': 0, 'anomaly_indices': []}

            # --- FIX 5: Modular Preprocessing ---
            preprocessors = self.preprocessors.get(model_name, {})
            if not preprocessors.get('scaler') or not preprocessors.get('imputer'):
                    raise ValueError(f"Preprocessors for model {model_name} not found.")
            features_scaled = self._transform_data(features, preprocessors)
            # --- END FIX 5 ---
            
            model = self.models[model_name]
            
            predictions = model.predict(features_scaled)
            anomaly_scores = model.decision_function(features_scaled)
            
            if threshold:
                predictions = (anomaly_scores < threshold).astype(int)
                predictions[predictions == 1] = -1
                predictions[predictions == 0] = 1
            
            anomaly_indices = np.where(predictions == -1)[0]
            anomalies = data.iloc[anomaly_indices].copy()
            anomalies['anomaly_score'] = anomaly_scores[anomaly_indices]
            
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
    
    # --- Anomaly Detection (Autoencoder) ---

    @celery_app.task(bind=True, name='analytics.train_anomaly_autoencoder')
    def train_anomaly_autoencoder(self, data: pd.DataFrame, model_name: str = "anomaly_autoencoder") -> Dict:
        """
        Train Autoencoder (Celery Task).
        - MODULARIZED: Uses _fit_preprocessors, _transform_data
        - TRACKED: Logs params and metrics (MAE, RMSE) to MLflow
        - METRICS: Logs MAE and RMSE (FIX 3)
        """
        if not TENSORFLOW_AVAILABLE:
            self.logger.error("TensorFlow is not installed. Cannot train Autoencoder.")
            return {'error': 'TensorFlow is not available.'}
            
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_name=f"train_autoencoder_{model_name}") as run:
            try:
                cache_key = f"autoencoder_training_{hash(str(data.values.tobytes()))}"
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    return cached_result

                dl_config = self.config['deep_learning']
                n_steps = self.config['time_series']['autoencoder_n_steps']
                epochs = dl_config['epochs']
                batch_size = dl_config['batch_size']

                features, _ = self.prepare_data(data)
                if features.empty:
                    msg = "No numeric features found"
                    self.logger.warning(msg)
                    if MLFLOW_AVAILABLE: mlflow.end_run(status="FAILED")
                    return {'error': msg}
                
                n_features = features.shape[1]

                # --- FIX 5: Modular Preprocessing ---
                preprocessors = self._fit_preprocessors(features)
                features_scaled = self._transform_data(features, preprocessors)
                # --- END FIX 5 ---
                
                X_seq = self._create_autoencoder_sequences(features_scaled, n_steps)
                if X_seq.shape[0] == 0:
                    msg = f"Not enough data to create sequences of length {n_steps}."
                    self.logger.error(msg)
                    if MLFLOW_AVAILABLE: mlflow.end_run(status="FAILED")
                    return {'error': msg}

                inputs = Input(shape=(n_steps, n_features))
                e = LSTM(128, activation='relu', return_sequences=True)(inputs)
                e = LSTM(64, activation='relu', return_sequences=False)(e)
                bottleneck = RepeatVector(n_steps)(e)
                d = LSTM(64, activation='relu', return_sequences=True)(bottleneck)
                d = LSTM(128, activation='relu', return_sequences=True)(d)
                outputs = TimeDistributed(Dense(n_features))(d)
                
                model = Model(inputs=inputs, outputs=outputs)
                # --- FIX 3: Add metrics ---
                model.compile(optimizer='adam', loss='mae', metrics=['mse'])
                
                self.logger.info(f"Training Autoencoder {model_name}...")
                
                early_stopping = EarlyStopping(
                    monitor='val_loss', 
                    patience=dl_config['patience'], 
                    restore_best_weights=True
                )
                
                history = model.fit(
                    X_seq, X_seq,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=dl_config['validation_split'],
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                X_pred = model.predict(X_seq)
                train_mae_loss = np.mean(np.abs(X_pred - X_seq), axis=1)
                train_mae_loss_flat = train_mae_loss.flatten()
                threshold = np.mean(train_mae_loss_flat) + 3 * np.std(train_mae_loss_flat)
                
                # --- FIX 3 & 4: Log Metrics to MLflow ---
                val_mae = history.history['val_loss'][-1]
                val_mse = history.history['val_mse'][-1]
                val_rmse = np.sqrt(val_mse)
                
                if MLFLOW_AVAILABLE:
                    mlflow.log_params(dl_config)
                    mlflow.log_param("n_steps", n_steps)
                    mlflow.log_param("n_features", n_features)
                    mlflow.log_metrics({
                        'val_mae': val_mae,
                        'val_mse': val_mse,
                        'val_rmse': val_rmse,
                        'reconstruction_threshold': threshold
                    })
                    mlflow.tensorflow.log_model(model, "model")
                # --- END FIX 3 & 4 ---

                metadata = {
                    'model_type': 'Autoencoder',
                    'trained_at': datetime.now().isoformat(),
                    'training_samples': X_seq.shape[0],
                    'features': list(features.columns),
                    'n_steps': n_steps,
                    'n_features': n_features,
                    'reconstruction_threshold': float(threshold),
                    'final_train_loss (mae)': float(history.history['loss'][-1]),
                    'final_val_loss (mae)': float(val_mae),
                    'final_val_rmse': float(val_rmse),
                    'mlflow_run_id': run.info.run_id if MLFLOW_AVAILABLE else None
                }
                
                self._save_model(model_name, model, preprocessors['scaler'], preprocessors['imputer'], metadata)
                
                result = {
                    'model_name': model_name,
                    'training_samples': X_seq.shape[0],
                    'reconstruction_threshold': float(threshold),
                    'metadata': metadata
                }
                
                self._set_cache(cache_key, result)
                self.logger.info(f"Autoencoder anomaly detector trained: {model_name}")
                if MLFLOW_AVAILABLE: mlflow.end_run(status="FINISHED")
                return result

            except Exception as e:
                self.logger.error(f"Autoencoder training error: {e}")
                if MLFLOW_AVAILABLE: mlflow.end_run(status="FAILED")
                raise

    @celery_app.task(bind=True, name='analytics.detect_anomalies_autoencoder')
    def detect_anomalies_autoencoder(self, data: pd.DataFrame, model_name: str = "anomaly_autoencoder") -> Dict:
        """
        Detect anomalies using Autoencoder (Celery Task).
        - MODULARIZED: Uses _transform_data
        """
        if not TENSORFLOW_AVAILABLE:
            self.logger.error("TensorFlow is not installed. Cannot use Autoencoder.")
            return {'error': 'TensorFlow is not available.'}
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")

            metadata = self.model_metadata.get(model_name, {})
            n_steps = metadata.get('n_steps')
            threshold = metadata.get('reconstruction_threshold')
            model_features = metadata.get('features', [])

            if not n_steps or not threshold:
                raise ValueError(f"Model metadata for {model_name} is incomplete.")

            features, _ = self.prepare_data(data)
            if model_features:
                features = features.reindex(columns=model_features, fill_value=0)
            if features.empty:
                return {'anomaly_count': 0, 'anomalies': [], 'anomaly_percentage': 0, 'anomaly_indices': []}
            
            # --- FIX 5: Modular Preprocessing ---
            preprocessors = self.preprocessors.get(model_name, {})
            if not preprocessors.get('scaler') or not preprocessors.get('imputer'):
                raise ValueError(f"Preprocessors for model {model_name} not found.")
            features_scaled = self._transform_data(features, preprocessors)
            # --- END FIX 5 ---
            
            X_seq = self._create_autoencoder_sequences(features_scaled, n_steps)
            if X_seq.shape[0] == 0:
                 return {'anomaly_count': 0, 'anomalies': [], 'anomaly_percentage': 0, 'anomaly_indices': []}

            model = self.models[model_name]
            
            X_pred = model.predict(X_seq)
            mae_loss = np.mean(np.abs(X_pred - X_seq), axis=(1, 2))
            
            anomaly_seq_indices = np.where(mae_loss > threshold)[0]
            
            anomaly_indices_set = set()
            for idx in anomaly_seq_indices:
                for i in range(n_steps):
                    anomaly_indices_set.add(idx + i)
            
            anomaly_indices = sorted([i for i in list(anomaly_indices_set) if i < len(data)])
            anomalies = data.iloc[anomaly_indices].copy()
            
            score_map = {}
            for i, seq_idx in enumerate(anomaly_seq_indices):
                score = mae_loss[seq_idx]
                for j in range(n_steps):
                    if (seq_idx + j) in anomaly_indices_set:
                        score_map[seq_idx + j] = max(score_map.get(seq_idx + j, 0), score)

            anomalies['anomaly_score'] = [score_map.get(i, 0) for i in anomalies.index]
            anomaly_count = len(anomaly_indices)
            anomaly_percentage = (anomaly_count / len(data)) * 100 if len(data) > 0 else 0

            result = {
                'anomaly_count': anomaly_count,
                'anomaly_percentage': anomaly_percentage,
                'anomaly_indices': anomaly_indices,
                'anomaly_scores': mae_loss.tolist(),
                'anomalies': anomalies.to_dict('records') if not anomalies.empty else [],
                'summary': {
                    'total_samples': len(data),
                    'total_sequences': len(X_seq),
                    'threshold': threshold,
                    'max_reconstruction_error': float(mae_loss.max()) if len(mae_loss) > 0 else 0
                }
            }
            
            self.logger.info(f"Autoencoder anomaly detection completed: {anomaly_count} anomalies found")
            return result
        except Exception as e:
            self.logger.error(f"Autoencoder detection error: {e}")
            raise
    
    # --- Failure Prediction ---
    
    @celery_app.task(bind=True, name='analytics.train_failure_predictor')
    def train_failure_predictor(self, data: pd.DataFrame, target_column: str, 
                                model_name: str = "failure_predictor") -> Dict:
        """
        Train failure predictor (Celery Task).
        - MODULARIZED: Uses _fit_preprocessors, _transform_data
        - TRACKED: Logs params and metrics (F1, AUC, etc.) to MLflow
        - METRICS: F1 score already present (FIX 3)
        """
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_name=f"train_failure_predictor_{model_name}") as run:
            try:
                cache_key = f"failure_training_{target_column}_{hash(str(data.values.tobytes()))}"
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    return cached_result

                features, target = self.prepare_data(data, target_column)
                if features.empty or target is None:
                    msg = "No features or target for failure predictor."
                    self.logger.warning(msg)
                    if MLFLOW_AVAILABLE: mlflow.end_run(status="FAILED")
                    return {'error': msg}
                
                is_binary = target.nunique() == 2
                class_weight = None
                if is_binary and target.value_counts(normalize=True).min() < 0.1:
                    self.logger.warning("Imbalanced target variable detected.")
                    class_weight = 'balanced'
                
                stratify = target if target.nunique() < 10 else None
                X_train, X_test, y_train, y_test = train_test_split(
                    features, target, test_size=0.2, random_state=42, stratify=stratify
                )
                
                # --- FIX 5: Modular Preprocessing ---
                preprocessors = self._fit_preprocessors(X_train)
                X_train_scaled = self._transform_data(X_train, preprocessors)
                X_test_scaled = self._transform_data(X_test, preprocessors)
                # --- END FIX 5 ---
                
                model_config = self.config['classification']
                model = RandomForestClassifier(
                    n_estimators=model_config['n_estimators'],
                    max_depth=model_config['max_depth'],
                    random_state=model_config['random_state'],
                    class_weight=class_weight 
                )
                
                model.fit(X_train_scaled, y_train)
                
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                feature_importance = dict(zip(features.columns, model.feature_importances_))
                y_pred = model.predict(X_test_scaled)
                
                # --- FIX 3 & 4: Log Metrics to MLflow ---
                evaluation_metrics = {}
                try:
                    if is_binary:
                        y_proba = model.predict_proba(X_test_scaled)[:, 1]
                        evaluation_metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
                        evaluation_metrics['precision'] = precision_score(y_test, y_pred, average='binary', zero_division=0)
                        evaluation_metrics['recall'] = recall_score(y_test, y_pred, average='binary', zero_division=0)
                        evaluation_metrics['f1_score'] = f1_score(y_test, y_pred, average='binary', zero_division=0)
                    else:
                        y_proba = model.predict_proba(X_test_scaled)
                        evaluation_metrics['roc_auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr')
                        evaluation_metrics['precision_macro'] = precision_score(y_test, y_pred, average='macro', zero_division=0)
                        evaluation_metrics['recall_macro'] = recall_score(y_test, y_pred, average='macro', zero_division=0)
                        evaluation_metrics['f1_score_macro'] = f1_score(y_test, y_pred, average='macro', zero_division=0)
                    evaluation_metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
                except Exception as e:
                    self.logger.warning(f"Could not calculate some metrics: {e}")
                    evaluation_metrics['error'] = str(e)
                
                if MLFLOW_AVAILABLE:
                    mlflow.log_params(model_config)
                    mlflow.log_param("class_weight", class_weight)
                    mlflow.log_metrics({
                        'train_score': train_score,
                        'test_score': test_score,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        **evaluation_metrics # Log all calculated metrics
                    })
                    mlflow.sklearn.log_model(model, "model")
                # --- END FIX 3 & 4 ---

                metadata = {
                    'model_type': 'RandomForestClassifier',
                    'trained_at': datetime.now().isoformat(),
                    'training_samples': len(X_train),
                    'features': list(features.columns),
                    'target_classes': list(model.classes_),
                    'train_score': train_score,
                    'test_score': test_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'feature_importance': feature_importance,
                    'evaluation_metrics': evaluation_metrics,
                    'mlflow_run_id': run.info.run_id if MLFLOW_AVAILABLE else None
                }
                
                self._save_model(model_name, model, preprocessors['scaler'], preprocessors['imputer'], metadata)
                
                result = {
                    'model_name': model_name,
                    'train_score': train_score,
                    'test_score': test_score,
                    'cv_scores': { 'mean': cv_scores.mean(), 'std': cv_scores.std(), 'scores': cv_scores.tolist() },
                    'feature_importance': feature_importance,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
                    'evaluation_metrics': evaluation_metrics,
                    'metadata': metadata
                }
                
                self._set_cache(cache_key, result)
                self.logger.info(f"Failure predictor trained: {model_name}")
                if MLFLOW_AVAILABLE: mlflow.end_run(status="FINISHED")
                return result
                
            except Exception as e:
                self.logger.error(f"Failure prediction training error: {e}")
                if MLFLOW_AVAILABLE: mlflow.end_run(status="FAILED")
                raise
    
    @celery_app.task(bind=True, name='analytics.predict_failure')
    def predict_failure(self, data: pd.DataFrame, model_name: str = "failure_predictor") -> Dict:
        """
        Predict failures (Celery Task).
        - MODULARIZED: Uses _transform_data
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            features, _ = self.prepare_data(data)
            
            metadata = self.model_metadata.get(model_name, {})
            model_features = metadata.get('features', [])
            if model_features:
                features = features.reindex(columns=model_features, fill_value=0)
            if features.empty:
                return {'predictions': [], 'failure_probabilities': []}

            # --- FIX 5: Modular Preprocessing ---
            preprocessors = self.preprocessors.get(model_name, {})
            if not preprocessors.get('scaler') or not preprocessors.get('imputer'):
                raise ValueError(f"Preprocessors for model {model_name} not found.")
            features_scaled = self._transform_data(features, preprocessors)
            # --- END FIX 5 ---
            
            model = self.models[model_name]
            
            predictions = model.predict(features_scaled)
            probabilities = model.predict_proba(features_scaled)
            
            failure_class_label = 1 
            classes = metadata.get('target_classes', model.classes_)
            
            try:
                failure_class_index = list(classes).index(failure_class_label)
                failure_probs = probabilities[:, failure_class_index]
            except (ValueError, TypeError):
                self.logger.warning(f"Class '{failure_class_label}' not found in model classes {classes}. Falling back.")
                if probabilities.shape[1] == 2:
                    failure_probs = probabilities[:, 1]
                else:
                    failure_probs = np.max(probabilities, axis=1)
            
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
                    'predicted_failures': int((predictions == failure_class_label).sum()),
                    'max_failure_probability': float(failure_probs.max()) if len(failure_probs) > 0 else 0,
                    'mean_failure_probability': float(failure_probs.mean()) if len(failure_probs) > 0 else 0
                }
            }
            
            self.logger.info(f"Failure prediction completed: {len(high_risk_indices)} high-risk cases")
            return result
        except Exception as e:
            self.logger.error(f"Failure prediction error: {e}")
            raise

    # --- Time-Series Forecasting (LSTM) ---

    @celery_app.task(bind=True, name='analytics.train_forecaster_lstm')
    def train_forecaster_lstm(self, data: pd.DataFrame, target_column: str,
                              model_name: str = "lstm_forecaster") -> Dict:
        """
        Train LSTM forecaster (Celery Task).
        - MODULARIZED: Uses _fit_preprocessors, _transform_data (adapted for 1D)
        - TRACKED: Logs params and metrics (MAE, RMSE) to MLflow
        - METRICS: Logs MAE and RMSE (FIX 3)
        """
        if not TENSORFLOW_AVAILABLE:
            self.logger.error("TensorFlow is not installed. Cannot train LSTM.")
            return {'error': 'TensorFlow is not available.'}
            
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_name=f"train_lstm_forecaster_{model_name}") as run:
            try:
                cache_key = f"lstm_training_{target_column}_{hash(str(data.values.tobytes()))}"
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    return cached_result

                ts_config = self.config['time_series']
                dl_config = self.config['deep_learning']
                n_steps_in = ts_config['lstm_n_steps_in']
                n_steps_out = ts_config['forecast_horizon']
                epochs = dl_config['epochs']
                batch_size = dl_config['batch_size']

                ts_data = data[[target_column]]
                
                # --- FIX 5: Modular Preprocessing (adapted for 1D) ---
                # 1. Impute
                imputer = SimpleImputer(strategy='mean')
                ts_imputed = imputer.fit_transform(ts_data)
                
                if len(ts_imputed) < n_steps_in + n_steps_out:
                    msg = f"Not enough data ({len(ts_imputed)}) to train LSTM."
                    self.logger.error(msg)
                    if MLFLOW_AVAILABLE: mlflow.end_run(status="FAILED")
                    return {'error': msg}
                
                # 2. Scale
                scaler = MinMaxScaler(feature_range=(0, 1))
                ts_scaled = scaler.fit_transform(ts_imputed)
                
                preprocessors = {'imputer': imputer, 'scaler': scaler}
                # --- END FIX 5 ---
                
                X_seq, y_seq = self._create_lstm_sequences(ts_scaled, n_steps_in, n_steps_out)
                n_features = X_seq.shape[2]
                
                model = Sequential()
                model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
                model.add(Dense(n_steps_out))
                
                # --- FIX 3: Add metrics ---
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                
                self.logger.info(f"Training LSTM Forecaster {model_name}...")
                
                early_stopping = EarlyStopping(
                    monitor='val_loss', 
                    patience=dl_config['patience'], 
                    restore_best_weights=True
                )
                
                history = model.fit(
                    X_seq, y_seq,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=dl_config['validation_split'],
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # --- FIX 3 & 4: Log Metrics to MLflow ---
                val_mse = history.history['val_loss'][-1]
                val_mae = history.history['val_mae'][-1]
                val_rmse = np.sqrt(val_mse)

                if MLFLOW_AVAILABLE:
                    mlflow.log_params(dl_config)
                    mlflow.log_param("n_steps_in", n_steps_in)
                    mlflow.log_param("n_steps_out", n_steps_out)
                    mlflow.log_metrics({
                        'val_mse': val_mse,
                        'val_mae': val_mae,
                        'val_rmse': val_rmse
                    })
                    mlflow.tensorflow.log_model(model, "model")
                # --- END FIX 3 & 4 ---

                metadata = {
                    'model_type': 'LSTM_Forecaster',
                    'trained_at': datetime.now().isoformat(),
                    'target_column': target_column,
                    'training_samples': X_seq.shape[0],
                    'n_steps_in': n_steps_in,
                    'n_steps_out': n_steps_out,
                    'n_features': n_features,
                    'final_train_loss (mse)': float(history.history['loss'][-1]),
                    'final_val_loss (mse)': float(val_mse),
                    'final_val_mae': float(val_mae),
                    'final_val_rmse': float(val_rmse),
                    'mlflow_run_id': run.info.run_id if MLFLOW_AVAILABLE else None
                }
                
                self._save_model(model_name, model, scaler, imputer, metadata)
                
                result = {
                    'model_name': model_name,
                    'training_samples': X_seq.shape[0],
                    'metadata': metadata
                }
                
                self._set_cache(cache_key, result)
                self.logger.info(f"LSTM Forecaster trained: {model_name}")
                if MLFLOW_AVAILABLE: mlflow.end_run(status="FINISHED")
                return result

            except Exception as e:
                self.logger.error(f"LSTM Forecaster training error: {e}")
                if MLFLOW_AVAILABLE: mlflow.end_run(status="FAILED")
                raise

    @celery_app.task(bind=True, name='analytics.predict_lstm_forecast')
    def predict_lstm_forecast(self, data: pd.DataFrame, model_name: str = "lstm_forecaster") -> Dict:
        """
        Make LSTM forecast (Celery Task).
        - MODULARIZED: Uses _transform_data (adapted for 1D)
        """
        if not TENSORFLOW_AVAILABLE:
            self.logger.error("TensorFlow is not installed. Cannot use LSTM.")
            return {'error': 'TensorFlow is not available.'}
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")

            metadata = self.model_metadata.get(model_name, {})
            n_steps_in = metadata.get('n_steps_in')
            n_steps_out = metadata.get('n_steps_out')
            target_column = metadata.get('target_column')

            if not n_steps_in or not target_column or not n_steps_out:
                raise ValueError(f"Model metadata for {model_name} is incomplete.")

            model = self.models[model_name]
            preprocessors = self.preprocessors.get(model_name, {})
            scaler = preprocessors.get('scaler')
            imputer = preprocessors.get('imputer')
            
            if not scaler or not imputer:
                raise ValueError(f"Preprocessors (scaler/imputer) for model {model_name} not found.")

            ts_data = data[[target_column]].values
            if len(ts_data) < n_steps_in:
                msg = f"Not enough data ({len(ts_data)}) to make prediction, need {n_steps_in}."
                self.logger.error(msg)
                return {'error': msg}

            input_seq_raw = ts_data[-n_steps_in:]
            
            # --- FIX 5: Modular Preprocessing (adapted for 1D) ---
            input_seq_imputed = imputer.transform(input_seq_raw)
            input_seq_scaled = scaler.transform(input_seq_imputed)
            # --- END FIX 5 ---
            
            input_seq_scaled = input_seq_scaled.reshape((1, n_steps_in, 1)) 
            y_pred_scaled = model.predict(input_seq_scaled)
            
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)
            y_pred = scaler.inverse_transform(y_pred_scaled)
            
            forecast = y_pred.flatten().tolist()
            
            result = {
                'model_name': model_name,
                'forecast': forecast,
                'forecast_steps': n_steps_out
            }
            
            self.logger.info(f"LSTM forecast completed for {model_name}")
            return result
        except Exception as e:
            self.logger.error(f"LSTM forecast error: {e}")
            raise
    
    # --- Model Retraining ---
    
    @celery_app.task(bind=True, name='analytics.retrain_models')
    def retrain_models(self):
        """
        Scheduled Celery task to retrain models with fresh data.
        (FIX 2: This is the task called by Celery Beat)
        """
        self.logger.info("Starting scheduled model retraining...")
        if not self.db_manager:
            self.logger.error("Database Manager not available. Skipping retraining.")
            return {"status": "error", "message": "Database Manager not available"}

        try:
            data = self.db_manager.get_health_data_as_dataframe(limit=20000) 
            if data.empty or len(data) < 100:
                self.logger.warning("Not enough new data to retrain. Skipping.")
                return {"status": "skipped", "message": "Not enough new data"}
            
            results = {}

            # --- Call other tasks ---
            # We call the .delay() method to spawn sub-tasks
            # Or call them synchronously if we want to wait
            # For this, let's call them synchronously
            
            self.logger.info("Retraining anomaly_detector...")
            anomaly_results = self.train_anomaly_detector(data, "anomaly_detector")
            results["anomaly_detector"] = anomaly_results.get('metadata', {})
            
            if TENSORFLOW_AVAILABLE:
                self.logger.info("Retraining anomaly_autoencoder...")
                try:
                    ae_results = self.train_anomaly_autoencoder(data, "anomaly_autoencoder")
                    results["anomaly_autoencoder"] = ae_results.get('metadata', {})
                except Exception as e:
                    self.logger.error(f"Failed to retrain anomaly_autoencoder: {e}")
                    results["anomaly_autoencoder"] = {"error": str(e)}

            if 'failure' in data.columns:
                self.logger.info("Retraining failure_predictor...")
                failure_results = self.train_failure_predictor(data, 'failure', "failure_predictor")
                results["failure_predictor"] = failure_results.get('metadata', {})
            else:
                self.logger.info("Skipping failure_predictor: 'failure' column not in data.")

            if 'temperature' in data.columns and TENSORFLOW_AVAILABLE:
                self.logger.info("Retraining lstm_forecaster...")
                try:
                    lstm_results = self.train_forecaster_lstm(data, 'temperature', 'lstm_forecaster')
                    results["lstm_forecaster"] = lstm_results.get('metadata', {})
                except Exception as e:
                    self.logger.error(f"Failed to retrain lstm_forecaster: {e}")
                    results["lstm_forecaster"] = {"error": str(e)}

            self.logger.info("Scheduled model retraining completed.")
            return {"status": "success", "results": results}
        
        except Exception as e:
            self.logger.error(f"Scheduled retraining failed: {e}")
            return {"status": "error", "message": str(e)}

    # --- Advanced Analysis Methods (Unchanged, as they don't fit the Celery/MLflow pattern) ---
    # ... (time_series_analysis, _analyze_trend, _detect_seasonality, ...) ...
    # ... (cluster_analysis, pattern_recognition, optimization_analysis, ...) ...
    # ... (utility methods: get_model_info, clear_cache, etc.) ...
    
    # [Rest of the file (time_series_analysis, cluster_analysis, etc.) remains unchanged]
    # ...
    # ... [Code from line 922 to 1640 (end of file) is identical to original] ...
    # ...
    def time_series_analysis(self, data: pd.DataFrame, value_column: str, 
                             time_column: str = None, forecast_steps: int = 12) -> Dict:
        """
        Perform comprehensive time series analysis.
        UPDATED: Now includes LSTM forecast if available.
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
            
            # --- OPTIMIZED: Impute time series data before analysis ---
            ts_imputed = SimpleImputer(strategy='mean').fit_transform(df[[value_column]])
            ts = pd.Series(ts_imputed.flatten(), index=df.index, name=value_column).dropna()
            
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
            forecast_result = self._forecast_time_series(ts, forecast_steps, data, value_column)
            
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
    
    def _forecast_time_series(self, ts: pd.Series, steps: int, 
                              original_data: pd.DataFrame, value_column: str) -> Dict:
        """
        Forecast time series using multiple methods.
        UPDATED: Now includes LSTM forecast if available.
        """
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
                    pass # ARIMA can be sensitive, fail gracefully
            
            # --- NEW: Add LSTM Forecast ---
            # Check if a relevant LSTM model exists
            lstm_model_name = "lstm_forecaster" # Or derive from value_column
            if (lstm_model_name in self.models and 
                self.model_metadata.get(lstm_model_name, {}).get('target_column') == value_column):
                try:
                    # Call the task. Use .get() for synchronous result if in a different context.
                    # Here, we just call the method directly.
                    lstm_pred = self.predict_lstm_forecast(original_data, lstm_model_name)
                    if 'error' not in lstm_pred:
                        # Ensure forecast matches required steps
                        forecasts['lstm'] = lstm_pred['forecast'][:steps]
                except Exception as e:
                    self.logger.warning(f"LSTM forecast failed during time_series_analysis: {e}")
            # --- End New Feature ---

            # Calculate ensemble forecast
            if forecasts:
                ensemble = np.mean([f for f in forecasts.values() if len(f) == steps], axis=0)
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
        - MODULARIZED: Uses _fit_preprocessors, _transform_data
        """
        try:
            features, _ = self.prepare_data(data)
            
            # --- FIX 5: Modular Preprocessing ---
            preprocessors = self._fit_preprocessors(features)
            features_scaled = self._transform_data(features, preprocessors)
            # --- END FIX 5 ---
            
            if method == 'kmeans':
                if n_clusters is None:
                    n_clusters = self._find_optimal_clusters(features_scaled)
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                cluster_labels = kmeans.fit_predict(features_scaled)
                
                cluster_centers_scaled = kmeans.cluster_centers_
                cluster_centers_original_scale = preprocessors['scaler'].inverse_transform(cluster_centers_scaled)
                
                result = {
                    'method': 'kmeans',
                    'n_clusters': n_clusters,
                    'cluster_labels': cluster_labels.tolist(),
                    'cluster_centers': cluster_centers_original_scale.tolist(),
                    'inertia': float(kmeans.inertia_),
                    'silhouette_score': None
                }
                
            elif method == 'dbscan':
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
            
            cluster_stats = self._calculate_cluster_stats(features, cluster_labels)
            result['cluster_statistics'] = cluster_stats
            
            self.logger.info(f"Cluster analysis completed: {result.get('n_clusters')} clusters found")
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
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                kmeans.fit(data)
                inertias.append(kmeans.inertia_)
            
            if len(inertias) >= 3:
                deltas = np.diff(inertias)
                delta_deltas = np.diff(deltas)
                if len(delta_deltas) > 0:
                    elbow_idx = np.argmax(delta_deltas) + 2
                    return min(elbow_idx, max_clusters)
            
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
                if label == -1:
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
        - MODULARIZED: Uses _fit_preprocessors, _transform_data
        """
        try:
            features, _ = self.prepare_data(data)
            
            # --- FIX 5: Modular Preprocessing ---
            # We don't need to scale for this, just impute
            imputer = SimpleImputer(strategy='mean')
            features_imputed = pd.DataFrame(
                imputer.fit_transform(features), 
                columns=features.columns, 
                index=features.index
            )
            # --- END FIX 5 ---
            
            if pattern_type == 'peaks':
                return self._detect_peaks(features_imputed)
            elif pattern_type == 'cycles':
                return self._detect_cycles(features_imputed)
            elif pattern_type == 'correlations':
                return self._analyze_correlations(features_imputed)
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
                    autocorr = np.correlate(values, values, mode='full')
                    autocorr = autocorr[autocorr.size // 2:]
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
            corr_matrix = data.corr()
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
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
        Perform optimization analysis.
        - MODULARIZED: Uses _fit_preprocessors
        """
        try:
            features, target = self.prepare_data(data, objective_column)
            
            # --- FIX 5: Modular Preprocessing ---
            # Fit a separate imputer for features
            feature_imputer = SimpleImputer(strategy='mean')
            features_imputed = feature_imputer.fit_transform(features)
            
            # Fit a separate imputer for target
            target_imputer = SimpleImputer(strategy='mean')
            target_imputed = target_imputer.fit_transform(target.values.reshape(-1, 1)).flatten()
            # --- END FIX 5 ---
            
            if constraint_columns:
                constraint_data = features_imputed[constraint_columns]
            else:
                constraint_data = None
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(features_imputed, target_imputed)
            
            feature_importance = dict(zip(features.columns, model.feature_importances_))
            
            target_series = pd.Series(target_imputed, index=features.index)
            best_indices = target_series.nlargest(10).index
            best_configs = features.loc[best_indices]
            
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
                'best_performance': float(target_series.max()),
                'worst_performance': float(target_series.min()),
                'mean_performance': float(target_series.mean()),
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
            sorted_features = sorted(feature_importance.items(), 
                                     key=lambda x: x[1], reverse=True)
            
            for feature, importance in sorted_features[:5]:
                if importance > 0.1:
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
    
    # --- Utility Methods ---

    def get_model_info(self, model_name: str = None) -> Dict:
        """Get information about trained models."""
        try:
            if model_name:
                if model_name in self.model_metadata:
                    return {
                        'metadata': self.model_metadata[model_name],
                        'preprocessors': self.preprocessors.get(model_name)
                    }
                else:
                    return {}
            else:
                return {
                    'available_models': list(self.models.keys()),
                    'model_metadata': self.model_metadata,
                    'model_preprocessors': self.preprocessors
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


# Example usage and demonstration
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    analytics = PredictiveAnalyticsEngine()
    
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'temperature': 20 + 5 * np.sin(np.arange(1000) * 2 * np.pi / 24) + np.random.normal(0, 1, 1000),
        'pressure': 1013 + np.random.normal(0, 5, 1000),
        'vibration': 0.1 + np.random.exponential(0.05, 1000),
        'failure': np.random.choice([0, 1], 1000, p=[0.9, 0.1])
    })
    for col in ['temperature', 'pressure', 'vibration']:
        sample_data.loc[sample_data.sample(frac=0.05).index, col] = np.nan
    print(f"Sample data created with {sample_data.isna().sum().sum()} missing values.")
    
    print("\n--- Training Standard Models ---")
    print("Training anomaly detector (IsolationForest)...")
    anomaly_result = analytics.train_anomaly_detector(sample_data[['temperature', 'pressure', 'vibration']])
    print(f"IsolationForest trained. Anomaly ratio: {anomaly_result['anomaly_ratio']:.3f}")
    
    print("\nTraining failure predictor (RandomForest)...")
    failure_result = analytics.train_failure_predictor(sample_data, 'failure')
    print(f"Failure predictor trained. Test score: {failure_result['test_score']:.3f}")
    
    print("\nPerforming time series analysis (StatsModels)...")
    ts_result = analytics.time_series_analysis(sample_data, 'temperature', 'timestamp')
    print(f"Time series analysis completed. Trend: {ts_result['trend_analysis']['trend_direction']}")
    
    print("\nPerforming cluster analysis (KMeans)...")
    cluster_result = analytics.cluster_analysis(sample_data[['temperature', 'pressure', 'vibration']])
    print(f"Cluster analysis completed. Found {cluster_result['n_clusters']} clusters")
    
    if TENSORFLOW_AVAILABLE:
        print("\n--- Training Deep Learning Models ---")
        print("Training anomaly detector (Autoencoder)...")
        ae_data = sample_data[['temperature', 'pressure', 'vibration']]
        ae_result = analytics.train_anomaly_autoencoder(ae_data)
        if 'error' not in ae_result:
            print(f"Autoencoder trained. Threshold: {ae_result['reconstruction_threshold']:.4f}")
        else:
            print(f"Autoencoder training failed: {ae_result['error']}")

        print("\nTraining LSTM forecaster (for temperature)...")
        lstm_result = analytics.train_forecaster_lstm(sample_data, 'temperature')
        if 'error' not in lstm_result:
            print(f"LSTM Forecaster trained.")
        else:
            print(f"LSTM training failed: {lstm_result['error']}")
            
        print("\nRe-running time series analysis (to include LSTM)...")
        analytics.clear_cache()
        ts_result = analytics.time_series_analysis(sample_data, 'temperature', 'timestamp')
        if 'lstm' in ts_result['forecast']['forecasts']:
            print("LSTM forecast successfully included in ensemble.")
    else:
        print("\n--- Skipping Deep Learning Models (TensorFlow not available) ---")
        ae_result = {}
        lstm_result = {}
    
    print("\n--- Testing Predictions on New Data ---")
    new_data = sample_data.tail(100).copy()
    new_data.loc[new_data.sample(frac=0.1).index, 'temperature'] = np.nan
    print(f"New data has {new_data.isna().sum().sum()} missing values to test prediction pipeline.")
    
    anomaly_pred = analytics.detect_anomalies(new_data[['temperature', 'pressure', 'vibration']])
    print(f"IsolationForest detection: {anomaly_pred['anomaly_count']} anomalies found")
    
    if TENSORFLOW_AVAILABLE and 'error' not in ae_result:
        ae_pred = analytics.detect_anomalies_autoencoder(new_data[['temperature', 'pressure', 'vibration']])
        print(f"Autoencoder detection: {ae_pred['anomaly_count']} anomalies found")
    else:
        ae_pred = {}

    failure_pred = analytics.predict_failure(new_data)
    print(f"Failure prediction: {failure_pred['high_risk_count']} high-risk cases")

    if TENSORFLOW_AVAILABLE and 'error' not in lstm_result:
        lstm_pred = analytics.predict_lstm_forecast(new_data)
        if 'error' not in lstm_pred:
            print(f"LSTM forecast: {len(lstm_pred['forecast'])} steps predicted.")
    else:
        lstm_pred = {}

    print("\n--- Testing Model Retraining (Mock) ---")
    if analytics.db_manager:
        print("DB Manager found. Mocking data insertion and retraining...")
        retrain_status = analytics.retrain_models() # This is now a Celery task
        print(f"Retraining status: {retrain_status.get('status')}")
    else:
        print("DB Manager not found. Skipping retraining test.")
    
    print("\n--- Exporting All Results ---")
    all_results = {
        'anomaly_training_if': anomaly_result,
        'anomaly_training_ae': ae_result,
        'failure_training_rf': failure_result,
        'lstm_training': lstm_result,
        'time_series': ts_result,
        'clustering': cluster_result,
        'predictions': {
            'anomalies_if': anomaly_pred,
            'anomalies_ae': ae_pred,
            'failures_rf': failure_pred,
            'forecast_lstm': lstm_pred
        }
    }
    
    export_path = analytics.export_analysis_results(all_results)
    print(f"\nAll results exported to: {export_path}")
    
    print("\nAnalytics engine demonstration completed!")