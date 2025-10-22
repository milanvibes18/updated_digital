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
from collections import defaultdict # <-- FIX: Added this import

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
try:
    # Attempt to import from the specific application file
    # Ensure the path is correct relative to where this script might be run from
    # If running as part of the Flask app, this might work directly.
    # If running standalone or from tests, sys.path adjustments might be needed earlier.
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
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV # Added hyperparameter tuning
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

# --- Data Drift and Explainability Imports (NEW) ---
try:
    # Example: Using scipy for basic drift detection (Kolmogorov-Smirnov test)
    from scipy.stats import ks_2samp
    DRIFT_DETECTION_AVAILABLE = True
except ImportError:
    print("Warning: scipy.stats not found. Basic data drift detection disabled.")
    DRIFT_DETECTION_AVAILABLE = False
    # Define dummy ks_2samp if needed
    def ks_2samp(*args, **kwargs):
        return (0.0, 1.0) # Simulate no drift

# Explainability Libraries (Import where needed or globally if preferred)
try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP loaded successfully.")
except ImportError:
    print("Warning: shap not found. SHAP explainability disabled.")
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
    print("LIME loaded successfully.")
except ImportError:
    print("Warning: lime not found. LIME explainability disabled.")
    LIME_AVAILABLE = False
# --- End Data Drift and Explainability Imports ---


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
    automated retraining, data drift monitoring, and model explainability.
    """

    def __init__(self, model_path="ANALYTICS/models/", cache_path="ANALYTICS/analysis_cache/"):
        self.model_path = Path(model_path)
        self.cache_path = Path(cache_path)
        self.logger = self._setup_logging()

        # --- FIX 2: Add versions path ---
        self.versions_path = self.model_path / "versions"

        # Create directories
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.versions_path.mkdir(parents=True, exist_ok=True) # FIX 2

        # Model storage
        self.models = {}
        self.preprocessors = {}
        self.model_metadata = {}
        self.training_data_stats = {} # NEW: Store stats for drift detection

        # Analysis cache
        self.analysis_cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = 300  # 5 minutes

        # Configuration (Consider loading from a file or central config)
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
            },
            # --- NEW: Config for Hyperparameter Tuning ---
            'hyperparameter_tuning': {
                'enabled': True, # Set to False to disable tuning globally
                'method': 'RandomizedSearchCV', # 'GridSearchCV' or 'RandomizedSearchCV'
                'n_iter': 20,       # For RandomizedSearchCV
                'cv': 3,            # Cross-validation folds
                'scoring': 'f1_macro', # Example for classification, adjust as needed (e.g., 'neg_mean_squared_error' for regression)
                'n_jobs': -1        # Use all available CPU cores
            },
            # --- NEW: Config for Data Drift ---
            'data_drift': {
                'enabled': True,
                'check_frequency': 100, # Check every N predictions
                'method': 'ks',         # Kolmogorov-Smirnov test (using t-test as proxy)
                'threshold': 0.05,      # p-value threshold
                'features_to_monitor': None # None = monitor all features used by the model
            },
            # --- NEW: Config for Explainability ---
            'explainability': {
                'enabled': True,
                'shap_sample_size': 100, # Number of background samples for SHAP KernelExplainer
                'lime_num_features': 5   # Number of features for LIME explanation
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

        # Prediction counter for drift checks (NEW)
        self.prediction_counters = defaultdict(int)

    # --- Setup, Load/Save, Cache methods remain largely the same ---
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
            # Add console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

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
            # Load sklearn models (from root model_path, not versions)
            model_files = list(self.model_path.glob("*.pkl"))
            for model_file in model_files:
                model_name = model_file.stem
                if "_meta" in model_name or "_stats" in model_name: # Skip metadata/stats files
                    continue
                try:
                    model_data = joblib.load(model_file)
                    if isinstance(model_data, dict):
                        self.models[model_name] = model_data.get('model')
                        self.preprocessors[model_name] = model_data.get('preprocessors', {}) # Ensure dict
                        self.model_metadata[model_name] = model_data.get('metadata', {})
                        # --- NEW: Load training stats ---
                        stats_file = self.model_path / f"{model_name}_stats.pkl"
                        if stats_file.exists():
                             self.training_data_stats[model_name] = joblib.load(stats_file)
                        # --- End Add ---
                    else: # Legacy format
                        self.models[model_name] = model_data
                    self.logger.info(f"Loaded sklearn model: {model_name}")
                except Exception as e:
                    self.logger.error(f"Failed to load .pkl model {model_name}: {e}")

            # Load Keras models (from root model_path, not versions) (NEW)
            if TENSORFLOW_AVAILABLE:
                keras_model_dirs = [
                    d for d in self.model_path.glob("*")
                    if d.is_dir() and d.name != 'versions'
                ]
                for model_dir in keras_model_dirs:
                    model_name = model_dir.stem
                    if model_name in self.models: continue
                    try:
                        self.models[model_name] = tf.keras.models.load_model(model_dir)
                        metadata_file = self.model_path / f"{model_name}_meta.pkl"
                        if metadata_file.exists():
                            meta_data = joblib.load(metadata_file)
                            self.preprocessors[model_name] = meta_data.get('preprocessors', {})
                            self.model_metadata[model_name] = meta_data.get('metadata', {})
                            # --- NEW: Load training stats ---
                            stats_file = self.model_path / f"{model_name}_stats.pkl"
                            if stats_file.exists():
                                 self.training_data_stats[model_name] = joblib.load(stats_file)
                            # --- End Add ---
                        self.logger.info(f"Loaded Keras model: {model_name}")
                    except Exception as e:
                        self.logger.error(f"Failed to load Keras model {model_name}: {e}")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")

    def _save_model(self, model_name: str, model: Any, preprocessors: Dict,
                    metadata: Dict = None, training_stats: Dict = None, timestamp: str = None):
        """
        Save model, preprocessors, metadata, and training stats. (FIX 2 & NEW)
        Saves latest and versioned copies.
        """
        try:
            if timestamp is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            latest_name = model_name
            versioned_name = f"{model_name}_{timestamp}"

            metadata_full = metadata or {}
            metadata_full['saved_at'] = datetime.now().isoformat()
            metadata_full['save_timestamp'] = timestamp

            if TENSORFLOW_AVAILABLE and isinstance(model, (tf.keras.Model)):
                # --- Save Keras Model ---
                # 1. Save "latest" model and separate meta/stats
                model.save(self.model_path / latest_name)
                joblib.dump({'preprocessors': preprocessors, 'metadata': metadata_full},
                            self.model_path / f"{latest_name}_meta.pkl")
                if training_stats:
                    joblib.dump(training_stats, self.model_path / f"{latest_name}_stats.pkl")

                # 2. Save "versioned" model and separate meta/stats
                model.save(self.versions_path / versioned_name)
                joblib.dump({'preprocessors': preprocessors, 'metadata': metadata_full},
                            self.versions_path / f"{versioned_name}_meta.pkl")
                if training_stats:
                    joblib.dump(training_stats, self.versions_path / f"{versioned_name}_stats.pkl")

                self.models[model_name] = model # Update in-memory model

            else:
                # --- Save Sklearn Model ---
                model_data = {
                    'model': model,
                    'preprocessors': preprocessors,
                    'metadata': metadata_full
                }
                # 1. Save "latest" combined and separate stats
                joblib.dump(model_data, self.model_path / f"{latest_name}.pkl")
                if training_stats:
                    joblib.dump(training_stats, self.model_path / f"{latest_name}_stats.pkl")

                # 2. Save "versioned" combined and separate stats
                joblib.dump(model_data, self.versions_path / f"{versioned_name}.pkl")
                if training_stats:
                    joblib.dump(training_stats, self.versions_path / f"{versioned_name}_stats.pkl")

                self.models[model_name] = model # Update in-memory model

            # Update in-memory stores
            self.preprocessors[model_name] = preprocessors
            self.model_metadata[model_name] = metadata_full
            if training_stats:
                self.training_data_stats[model_name] = training_stats

            self.logger.info(f"Model saved: {latest_name} (latest) and {versioned_name} (versioned)")

        except Exception as e:
            self.logger.error(f"Failed to save model {model_name}: {e}")

    # Cache methods (unchanged)
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


    # --- Data Preparation and Preprocessing (Unchanged logic, just structure) ---
    def prepare_data(self, data: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare data for machine learning."""
        try:
            df = data.copy()
            # Convert potential datetime objects/strings
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        # Attempt datetime conversion, ignore errors for non-datetime strings
                        df[col] = pd.to_datetime(df[col], errors='ignore')
                    except Exception: # Catch broader errors during conversion
                        pass

            # Feature engineering for datetime columns
            datetime_cols = df.select_dtypes(include=['datetime64', 'datetime64[ns]', 'datetime64[us]', 'datetime64[ms]']).columns
            for col in datetime_cols:
                df[f'{col}_hour'] = df[col].dt.hour
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_weekday'] = df[col].dt.weekday
                df = df.drop(columns=[col])

            # One-hot encode categorical features
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if col != target_column:
                    try:
                        df = pd.get_dummies(df, columns=[col], drop_first=True, dummy_na=False) # Avoid NA columns
                    except Exception as e:
                         self.logger.warning(f"Could not one-hot encode column '{col}': {e}")


            # Separate features and target
            if target_column and target_column in df.columns:
                target = df[target_column]
                features = df.drop(columns=[target_column])
            else:
                target = None
                features = df

            # Keep only numeric features
            features = features.select_dtypes(include=[np.number])
            self.logger.info(f"Data prepared: {features.shape[0]} rows, {features.shape[1]} features")
            return features, target
        except Exception as e:
            self.logger.error(f"Data preparation error: {e}", exc_info=True)
            raise

    # --- Modular Preprocessing (FIX 5 - Unchanged logic) ---
    def _fit_preprocessors(self, features: pd.DataFrame) -> Dict:
        """Fits imputer and scaler and returns them in a dict."""
        try:
            # 1. Impute
            imputer = SimpleImputer(strategy='mean')
            features_imputed = imputer.fit_transform(features) # Fit and transform

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
            self.logger.error(f"Error transforming data: {e}", exc_info=True)
            raise
    # --- End Preprocessing ---

    # --- Sequence creation methods (Unchanged) ---
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
            seq_y = data[end_ix:out_end_ix, 0] # Assuming single target feature
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

    # --- Loss Plot Saving (Unchanged) ---
    def _save_loss_plot(self, history: Any, model_name: str, timestamp: str):
        """Save training/validation loss plot."""
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'{model_name} Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()

            # 1. Save "latest"
            latest_path = self.model_path / f"{model_name}_loss.png"
            plt.savefig(latest_path)

            # 2. Save "versioned"
            versioned_path = self.versions_path / f"{model_name}_{timestamp}_loss.png"
            plt.savefig(versioned_path)

            plt.close()
            self.logger.info(f"Loss plot saved to {latest_path} and {versioned_path}")

        except Exception as e:
            self.logger.error(f"Failed to save loss plot for {model_name}: {e}")

    # --- NEW: Hyperparameter Tuning Function ---
    def _perform_hyperparameter_tuning(self, model_class, param_dist, X_train, y_train):
        """Performs hyperparameter tuning using GridSearchCV or RandomizedSearchCV."""
        tuning_config = self.config['hyperparameter_tuning']
        if not tuning_config['enabled']:
            self.logger.info("Hyperparameter tuning is disabled.")
            return model_class(**model_class().get_params()) # Return model with default params

        method = tuning_config['method']
        n_iter = tuning_config['n_iter']
        cv = tuning_config['cv']
        scoring = tuning_config['scoring']
        n_jobs = tuning_config['n_jobs']

        self.logger.info(f"Starting hyperparameter tuning with {method}...")

        if method == 'GridSearchCV':
            # Note: param_dist should be a grid for GridSearchCV
            search = GridSearchCV(
                estimator=model_class(),
                param_grid=param_dist,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1
            )
        elif method == 'RandomizedSearchCV':
            search = RandomizedSearchCV(
                estimator=model_class(),
                param_distributions=param_dist,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                random_state=42,
                verbose=1
            )
        else:
            self.logger.warning(f"Unsupported tuning method: {method}. Using default parameters.")
            return model_class()

        try:
            search.fit(X_train, y_train)
            self.logger.info(f"Best parameters found: {search.best_params_}")
            self.logger.info(f"Best score ({scoring}): {search.best_score_:.4f}")
            return search.best_estimator_
        except Exception as e:
            self.logger.error(f"Hyperparameter tuning failed: {e}. Using default parameters.")
            # Fallback to default model if tuning fails
            # Need to handle potential errors during default instantiation too
            try:
                return model_class(**model_class().get_params())
            except Exception as default_e:
                 self.logger.error(f"Failed to instantiate default model: {default_e}")
                 raise ValueError("Could not create model instance.") from default_e


    # --- Anomaly Detection (Isolation Forest) ---
    @celery_app.task(bind=True, name='analytics.train_anomaly_detector')
    def train_anomaly_detector(self, data: pd.DataFrame, model_name: str = "anomaly_detector") -> Dict:
        """
        Train anomaly detection model (Celery Task).
        - MODULARIZED: Uses _fit_preprocessors, _transform_data
        - TRACKED: Logs params and metrics to MLflow
        - VERSIONED: (FIX 2) Saves latest and timestamped model
        - TUNING: Optionally performs hyperparameter tuning (NEW)
        - DRIFT: Saves training data stats (NEW)
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if MLFLOW_AVAILABLE: mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_name=f"train_anomaly_detector_{model_name}_{timestamp}") as run:
            try:
                # ... (cache check remains the same) ...
                cache_key = f"anomaly_training_{hash(str(data.values.tobytes()))}" # Use hash of data
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    self.logger.info(f"Returning cached training result for {model_name}")
                    return cached_result


                features, _ = self.prepare_data(data)
                if features.empty:
                    # ... (error handling remains the same) ...
                    msg = "No numeric features found"
                    self.logger.warning(msg)
                    if MLFLOW_AVAILABLE: mlflow.end_run(status="FAILED")
                    return {'error': msg}


                # --- Preprocessing & Training Stats ---
                preprocessors = self._fit_preprocessors(features)
                features_scaled = self._transform_data(features, preprocessors)
                # --- NEW: Calculate and store training data statistics ---
                training_stats = {
                    'mean': features.mean().to_dict(),
                    'std': features.std().to_dict(),
                    'min': features.min().to_dict(),
                    'max': features.max().to_dict(),
                    'count': len(features)
                }
                # --- End Add ---

                # --- NEW: Hyperparameter Tuning ---
                model_config = self.config['anomaly_detection']
                if self.config['hyperparameter_tuning']['enabled']:
                    # Define parameter distribution for Isolation Forest
                    param_dist = {
                        'n_estimators': [50, 100, 200],
                        'max_samples': ['auto', 0.5, 0.75],
                        'contamination': [0.01, 0.05, 0.1, 0.15],
                         'max_features': [0.5, 0.75, 1.0]
                    }
                    # Note: Scoring for unsupervised models like IsolationForest is tricky.
                    # We might need a custom scorer or focus on specific metrics if available.
                    # For simplicity, we'll skip scoring specification for IF.
                    # Adjust 'scoring' in config if tuning Regression/Classification
                    tuned_model = self._perform_hyperparameter_tuning(
                        IsolationForest, param_dist, features_scaled, None # No y_train for IF
                    )
                    model = tuned_model
                else:
                     model = IsolationForest(
                        contamination=model_config['contamination'],
                        n_estimators=model_config['n_estimators'],
                        random_state=model_config['random_state']
                    )
                # --- End Tuning ---

                model.fit(features_scaled)

                anomaly_scores = model.decision_function(features_scaled)
                predictions = model.predict(features_scaled)
                anomaly_ratio = (predictions == -1).sum() / len(predictions)

                # --- MLflow Logging (Updated) ---
                if MLFLOW_AVAILABLE:
                    # Log final parameters (best found or default)
                    mlflow.log_params(model.get_params())
                    mlflow.log_param("tuning_enabled", self.config['hyperparameter_tuning']['enabled'])
                    mlflow.log_param("training_samples", len(features))
                    mlflow.log_param("features_count", len(features.columns))
                    mlflow.log_metrics({'anomaly_ratio': anomaly_ratio})
                    # Log training stats artifacts
                    stats_path = Path(mlflow.get_artifact_uri()) / "training_stats.json"
                    stats_path.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists
                    with open(stats_path, 'w') as f:
                        json.dump(training_stats, f)
                    mlflow.log_artifact(str(stats_path))
                    # Log model
                    mlflow.sklearn.log_model(model, "model")
                # --- End MLflow ---

                metadata = {
                    'model_type': 'IsolationForest',
                    'trained_at': datetime.now().isoformat(),
                    'timestamp': timestamp,
                    'training_samples': len(features),
                    'features': list(features.columns),
                    'anomaly_ratio': anomaly_ratio,
                    # Store final model parameters
                    'model_params': model.get_params(),
                    'tuning_enabled': self.config['hyperparameter_tuning']['enabled'],
                    'mlflow_run_id': run.info.run_id if MLFLOW_AVAILABLE else None
                }

                # --- Save model with stats ---
                self._save_model(
                    model_name, model,
                    preprocessors, # Pass the dict directly
                    metadata, training_stats=training_stats, timestamp=timestamp
                )

                result = {
                    'model_name': model_name,
                    'training_samples': len(features),
                    'anomaly_ratio': float(anomaly_ratio),
                    # Feature importance is not directly available for Isolation Forest
                    'metadata': metadata
                }

                self._set_cache(cache_key, result)
                self.logger.info(f"Anomaly detector trained: {model_name}")
                if MLFLOW_AVAILABLE: mlflow.end_run(status="FINISHED")
                return result

            except Exception as e:
                self.logger.error(f"Anomaly detection training error: {e}", exc_info=True)
                if MLFLOW_AVAILABLE: mlflow.end_run(status="FAILED")
                raise

    @celery_app.task(bind=True, name='analytics.detect_anomalies')
    def detect_anomalies(self, data: pd.DataFrame, model_name: str = "anomaly_detector",
                         threshold: float = None) -> Dict:
        """
        Detect anomalies in data (Celery Task).
        - MODULARIZED: Uses _transform_data
        - DRIFT: Performs data drift check (NEW)
        - EXPLAIN: Adds SHAP/LIME explanation (NEW)
        """
        try:
            if model_name not in self.models:
                # ... (model loading/training fallback remains the same) ...
                self.logger.warning(f"Model {model_name} not found. Attempting to train...")
                if self.db_manager:
                    # Fetch more data for training attempt
                    hist_data = self.db_manager.get_health_data_as_dataframe(limit=10000)
                    if not hist_data.empty and len(hist_data) > 50:
                        self.logger.info(f"Training {model_name} with {len(hist_data)} records from DB.")
                        # Call task synchronously for dependency, handle potential errors
                        try:
                            train_result = self.train_anomaly_detector(hist_data, model_name)
                            if 'error' in train_result:
                                raise ValueError(f"Training failed: {train_result['error']}")
                        except Exception as train_e:
                            raise ValueError(f"Model {model_name} not found and auto-training failed: {train_e}") from train_e
                    else:
                        raise ValueError(f"Model {model_name} not found and not enough data to train.")
                else:
                    raise ValueError(f"Model {model_name} not found and DB manager is not available.")

            features, _ = self.prepare_data(data)
            metadata = self.model_metadata.get(model_name, {})
            model_features = metadata.get('features', [])
            if model_features:
                 # Align columns, fill missing with 0 (or mean from training_stats if needed)
                features = features.reindex(columns=model_features, fill_value=0)
            if features.empty:
                # ... (empty features handling remains the same) ...
                self.logger.warning("No features to predict on for anomaly detector.")
                return {'anomaly_count': 0, 'anomalies': [], 'anomaly_percentage': 0, 'anomaly_indices': []}


            # --- NEW: Data Drift Check ---
            drift_detected = False
            drift_report = {}
            if self.config['data_drift']['enabled']:
                 self.prediction_counters[model_name] += len(features)
                 if self.prediction_counters[model_name] >= self.config['data_drift']['check_frequency']:
                     self.logger.info(f"Checking data drift for model {model_name}...")
                     drift_report = self._check_data_drift(features, model_name)
                     drift_detected = drift_report.get('drift_detected', False)
                     if drift_detected:
                         self.logger.warning(f"Data drift detected for model {model_name}!", drift_details=drift_report.get('drifting_features'))
                         # Optional: Trigger retraining task here
                         # self.retrain_models.delay()
                     self.prediction_counters[model_name] = 0 # Reset counter
            # --- End Drift Check ---

            # --- Preprocessing & Prediction ---
            preprocessors = self.preprocessors.get(model_name, {})
            if not preprocessors.get('scaler') or not preprocessors.get('imputer'):
                 raise ValueError(f"Preprocessors for model {model_name} not found.")
            features_scaled = self._transform_data(features, preprocessors)

            model = self.models[model_name]
            predictions = model.predict(features_scaled)
            anomaly_scores = model.decision_function(features_scaled)

            if threshold:
                 # ... (threshold logic remains the same) ...
                 predictions = (anomaly_scores < threshold).astype(int)
                 predictions[predictions == 1] = -1
                 predictions[predictions == 0] = 1


            anomaly_indices = np.where(predictions == -1)[0]
            anomalies = data.iloc[anomaly_indices].copy() # Get original data for anomalies
            anomalies['anomaly_score'] = anomaly_scores[anomaly_indices]

            anomaly_count = len(anomaly_indices)
            anomaly_percentage = (anomaly_count / len(data)) * 100 if len(data) > 0 else 0

            # --- NEW: Explainability ---
            explanations = {}
            if self.config['explainability']['enabled'] and anomaly_count > 0:
                 try:
                     # Explain the first few anomalies
                     explain_indices = anomaly_indices[:min(3, anomaly_count)] # Explain first 3 anomalies
                     if SHAP_AVAILABLE:
                         explanations['shap'] = self._explain_shap(
                             model, features_scaled, explain_indices, model_features
                         )
                     if LIME_AVAILABLE:
                          explanations['lime'] = self._explain_lime(
                             model, features_scaled, explain_indices, model_features
                         )
                 except Exception as explain_e:
                     self.logger.error(f"Error generating explanations: {explain_e}")
                     explanations['error'] = str(explain_e)
            # --- End Explainability ---

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
                },
                'drift_check': { # NEW
                    'performed': self.prediction_counters[model_name] == 0 and self.config['data_drift']['enabled'],
                    'drift_detected': drift_detected,
                    'report': drift_report
                },
                'explanations': explanations # NEW
            }

            self.logger.info(f"Anomaly detection completed: {anomaly_count} anomalies found")
            return result
        except Exception as e:
            self.logger.error(f"Anomaly detection error: {e}", exc_info=True)
            raise

    # --- NEW: Data Drift Check Method ---
    def _check_data_drift(self, new_data: pd.DataFrame, model_name: str) -> Dict:
        """Performs statistical tests to check for data drift."""
        if not DRIFT_DETECTION_AVAILABLE:
            return {'drift_detected': False, 'message': 'Drift detection library not available.'}

        training_stats = self.training_data_stats.get(model_name)
        if not training_stats:
            self.logger.warning(f"No training data statistics found for model {model_name}. Cannot perform drift check.")
            return {'drift_detected': False, 'message': 'Missing training statistics.'}

        drift_report = {
            'drift_detected': False,
            'method': self.config['data_drift']['method'],
            'threshold': self.config['data_drift']['threshold'],
            'drifting_features': [],
            'feature_results': {}
        }

        features_to_monitor = self.config['data_drift']['features_to_monitor'] or list(training_stats.get('mean', {}).keys())

        for feature in features_to_monitor:
            if feature not in new_data.columns or feature not in training_stats.get('mean', {}):
                drift_report['feature_results'][feature] = {'status': 'skipped', 'reason': 'Feature not found in new data or training stats.'}
                continue

            try:
                # Basic KS test requires numerical data
                new_feature_data = new_data[feature].dropna().astype(float)
                if new_feature_data.empty:
                    drift_report['feature_results'][feature] = {'status': 'skipped', 'reason': 'No new data for feature.'}
                    continue

                # We need the original training data distribution, not just stats.
                # For KS, we compare samples. Storing full training data is infeasible.
                # Simplification: Compare means/stds (less robust than KS test on distributions)
                train_mean = training_stats['mean'][feature]
                train_std = training_stats['std'][feature]
                new_mean = new_feature_data.mean()
                new_std = new_feature_data.std()

                # Basic check: T-test for means (assuming normality, might not hold)
                # A more robust approach might involve storing histograms or using dedicated drift libraries.
                # Handle case where new_std is 0 or NaN
                if pd.isna(new_std) or new_std == 0:
                     # If new std is 0, check if new_mean is outside train mean +/- std
                     if new_mean < (train_mean - 3 * train_std) or new_mean > (train_mean + 3 * train_std):
                         p_value_mean = 0.0 # Force drift detection
                     else:
                         p_value_mean = 1.0 # Assume no drift
                     t_stat = 0.0
                else:
                    t_stat, p_value_mean = stats.ttest_ind_from_stats(
                         mean1=train_mean, std1=train_std, nobs1=training_stats['count'],
                         mean2=new_mean, std2=new_std, nobs2=len(new_feature_data),
                         equal_var=False # Welch's t-test
                    )

                drift_detected_feature = p_value_mean < drift_report['threshold']

                drift_report['feature_results'][feature] = {
                    'status': 'drift' if drift_detected_feature else 'no_drift',
                    'p_value': float(p_value_mean),
                    'train_mean': float(train_mean),
                    'new_mean': float(new_mean)
                }
                if drift_detected_feature:
                    drift_report['drifting_features'].append(feature)
                    drift_report['drift_detected'] = True

            except Exception as e:
                self.logger.error(f"Drift check failed for feature '{feature}': {e}")
                drift_report['feature_results'][feature] = {'status': 'error', 'reason': str(e)}

        return drift_report

    # --- NEW: Explainability Methods ---
    def _explain_shap(self, model, X_scaled: np.ndarray, instance_indices: List[int], feature_names: List[str]) -> Dict:
        """Generates SHAP explanations for specific instances."""
        if not SHAP_AVAILABLE:
            return {'error': 'SHAP library not available.'}

        try:
            shap_explanations = {}
            # Use KernelExplainer for models like IsolationForest (model-agnostic)
            # Sample background data for approximation
            background_sample_indices = np.random.choice(X_scaled.shape[0], min(self.config['explainability']['shap_sample_size'], X_scaled.shape[0]), replace=False)
            background_data = X_scaled[background_sample_indices]

            # Use model's decision_function for Isolation Forest scores
            explainer = shap.KernelExplainer(model.decision_function, background_data)

            # Explain the selected instances
            instances_to_explain = X_scaled[instance_indices]
            shap_values = explainer.shap_values(instances_to_explain)

            for i, original_index in enumerate(instance_indices):
                shap_explanations[str(original_index)] = dict(zip(feature_names, shap_values[i].tolist()))

            return {'status': 'success', 'explanations': shap_explanations}

        except Exception as e:
             self.logger.error(f"SHAP explanation failed: {e}", exc_info=True)
             return {'error': str(e)}


    def _explain_lime(self, model, X_scaled: np.ndarray, instance_indices: List[int], feature_names: List[str]) -> Dict:
        """Generates LIME explanations for specific instances."""
        if not LIME_AVAILABLE:
            return {'error': 'LIME library not available.'}

        try:
            lime_explanations = {}
            # LIME requires a prediction function that outputs probabilities (or scores)
            # For Isolation Forest, predict returns 1 (normal) or -1 (anomaly).
            # We can adapt decision_function to a pseudo-probability if needed, or use predict directly.
            # Here, we'll use predict directly.
            predict_fn = model.predict # Use predict for LIME categories (1 or -1)

            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_scaled,
                feature_names=feature_names,
                class_names=['normal', 'anomaly'], # Match predict output (needs mapping if using decision_function)
                mode='classification'
            )

            for i, original_index in enumerate(instance_indices):
                instance = X_scaled[original_index]
                explanation = explainer.explain_instance(
                    data_row=instance,
                    predict_fn=predict_fn, # Pass the predict function
                    num_features=self.config['explainability']['lime_num_features']
                )
                lime_explanations[str(original_index)] = explanation.as_list()

            return {'status': 'success', 'explanations': lime_explanations}

        except Exception as e:
            self.logger.error(f"LIME explanation failed: {e}", exc_info=True)
            return {'error': str(e)}


    # --- Anomaly Detection (Autoencoder) ---
    # Methods train_anomaly_autoencoder and detect_anomalies_autoencoder remain largely the same,
    # but could be updated similarly to include tuning, drift, and explainability if needed.
    # Explainability for autoencoders often involves analyzing reconstruction errors per feature.
    @celery_app.task(bind=True, name='analytics.train_anomaly_autoencoder')
    def train_anomaly_autoencoder(self, data: pd.DataFrame, model_name: str = "anomaly_autoencoder") -> Dict:
        """
        Train Autoencoder (Celery Task).
        - MODULARIZED: Uses _fit_preprocessors, _transform_data
        - TRACKED: Logs params and metrics (MAE, RMSE) to MLflow
        - METRICS: Logs MAE and RMSE (FIX 3)
        - FIX 2: Adds loss viz, threshold tuning, and versioning
        - DRIFT: Saves training data stats (NEW)
        - TUNING: (Note: Tuning DL models is more complex, not added here for simplicity)
        """
        if not TENSORFLOW_AVAILABLE:
            self.logger.error("TensorFlow is not installed. Cannot train Autoencoder.")
            return {'error': 'TensorFlow is not available.'}

        # --- FIX 2: Generate timestamp ---
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_name=f"train_autoencoder_{model_name}_{timestamp}") as run:
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

                # --- Preprocessing & Training Stats ---
                preprocessors = self._fit_preprocessors(features)
                features_scaled = self._transform_data(features, preprocessors)
                # --- NEW: Calculate and store training data statistics ---
                training_stats = {
                    'mean': features.mean().to_dict(),
                    'std': features.std().to_dict(),
                    'min': features.min().to_dict(),
                    'max': features.max().to_dict(),
                    'count': len(features)
                }
                # --- End Add ---

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

                # --- FIX 2: Manual train/val split for threshold tuning ---
                X_train_seq, X_val_seq = train_test_split(
                    X_seq,
                    test_size=dl_config['validation_split'],
                    random_state=42
                )

                history = model.fit(
                    X_train_seq, X_train_seq,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val_seq, X_val_seq), # Use manual split
                    callbacks=[early_stopping],
                    verbose=0
                )

                # --- FIX 2: Loss Visualization ---
                self._save_loss_plot(history, model_name, timestamp)

                # --- FIX 2: Threshold Tuning (on validation set) ---
                X_val_pred = model.predict(X_val_seq)
                val_mae_loss = np.mean(np.abs(X_val_pred - X_val_seq), axis=(1, 2))
                threshold = float(np.percentile(val_mae_loss, 95))
                self.logger.info(f"Autoencoder threshold tuned to {threshold} (95th percentile of val error).")

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
                    # Log training stats artifacts
                    stats_path = Path(mlflow.get_artifact_uri()) / "training_stats.json"
                    stats_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(stats_path, 'w') as f:
                        json.dump(training_stats, f)
                    mlflow.log_artifact(str(stats_path))
                    mlflow.tensorflow.log_model(model, "model")
                # --- END FIX 3 & 4 ---

                metadata = {
                    'model_type': 'Autoencoder',
                    'trained_at': datetime.now().isoformat(),
                    'timestamp': timestamp, # FIX 2
                    'training_samples': X_train_seq.shape[0],
                    'validation_samples': X_val_seq.shape[0],
                    'features': list(features.columns),
                    'n_steps': n_steps,
                    'n_features': n_features,
                    'reconstruction_threshold': threshold, # FIX 2
                    'final_train_loss (mae)': float(history.history['loss'][-1]),
                    'final_val_loss (mae)': float(val_mae),
                    'final_val_rmse': float(val_rmse),
                    'mlflow_run_id': run.info.run_id if MLFLOW_AVAILABLE else None
                }

                # --- Save model with stats ---
                self._save_model(
                    model_name, model,
                    preprocessors, # Pass the dict directly
                    metadata, training_stats=training_stats, timestamp=timestamp
                )

                result = {
                    'model_name': model_name,
                    'training_samples': X_train_seq.shape[0],
                    'reconstruction_threshold': threshold,
                    'metadata': metadata
                }

                self._set_cache(cache_key, result)
                self.logger.info(f"Autoencoder anomaly detector trained: {model_name}")
                if MLFLOW_AVAILABLE: mlflow.end_run(status="FINISHED")
                return result

            except Exception as e:
                self.logger.error(f"Autoencoder training error: {e}", exc_info=True)
                if MLFLOW_AVAILABLE: mlflow.end_run(status="FAILED")
                raise

    @celery_app.task(bind=True, name='analytics.detect_anomalies_autoencoder')
    def detect_anomalies_autoencoder(self, data: pd.DataFrame, model_name: str = "anomaly_autoencoder") -> Dict:
        """
        Detect anomalies using Autoencoder (Celery Task).
        - MODULARIZED: Uses _transform_data
        - DRIFT: Performs data drift check (NEW)
        - EXPLAIN: (Not yet implemented for AE)
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

            if not n_steps or threshold is None: # Allow threshold=0
                raise ValueError(f"Model metadata for {model_name} is incomplete (n_steps or threshold missing).")

            features, _ = self.prepare_data(data)
            if model_features:
                features = features.reindex(columns=model_features, fill_value=0)
            if features.empty:
                return {'anomaly_count': 0, 'anomalies': [], 'anomaly_percentage': 0, 'anomaly_indices': []}

            # --- NEW: Data Drift Check ---
            drift_detected = False
            drift_report = {}
            if self.config['data_drift']['enabled']:
                 self.prediction_counters[model_name] += len(features)
                 if self.prediction_counters[model_name] >= self.config['data_drift']['check_frequency']:
                     self.logger.info(f"Checking data drift for model {model_name}...")
                     drift_report = self._check_data_drift(features, model_name)
                     drift_detected = drift_report.get('drift_detected', False)
                     if drift_detected:
                         self.logger.warning(f"Data drift detected for model {model_name}!", drift_details=drift_report.get('drifting_features'))
                     self.prediction_counters[model_name] = 0
            # --- End Drift Check ---

            # --- Preprocessing ---
            preprocessors = self.preprocessors.get(model_name, {})
            if not preprocessors.get('scaler') or not preprocessors.get('imputer'):
                raise ValueError(f"Preprocessors for model {model_name} not found.")
            features_scaled = self._transform_data(features, preprocessors)
            # --- END ---

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
                },
                'drift_check': { # NEW
                    'performed': self.prediction_counters[model_name] == 0 and self.config['data_drift']['enabled'],
                    'drift_detected': drift_detected,
                    'report': drift_report
                },
                'explanations': {} # NEW (Not implemented for AE yet)
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
        - VERSIONED: (FIX 2) Saves latest and timestamped model
        - TUNING: Optionally performs hyperparameter tuning (NEW)
        - DRIFT: Saves training data stats (NEW)
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if MLFLOW_AVAILABLE: mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_name=f"train_failure_predictor_{model_name}_{timestamp}") as run:
            try:
                # ... (cache check remains the same) ...
                cache_key = f"failure_training_{target_column}_{hash(str(data.values.tobytes()))}"
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    self.logger.info(f"Returning cached training result for {model_name}")
                    return cached_result

                features, target = self.prepare_data(data, target_column)
                if features.empty or target is None:
                    # ... (error handling remains the same) ...
                    msg = "No features or target for failure predictor."
                    self.logger.warning(msg)
                    if MLFLOW_AVAILABLE: mlflow.end_run(status="FAILED")
                    return {'error': msg}

                # ... (imbalance check remains the same) ...
                is_binary = target.nunique() == 2
                class_weight = None
                if is_binary and target.value_counts(normalize=True).min() < 0.1:
                    self.logger.warning("Imbalanced target variable detected.")
                    class_weight = 'balanced'

                # ... (train/test split remains the same) ...
                stratify = target if target.nunique() < 10 else None
                X_train, X_test, y_train, y_test = train_test_split(
                    features, target, test_size=0.2, random_state=42, stratify=stratify
                )


                # --- Preprocessing & Training Stats ---
                preprocessors = self._fit_preprocessors(X_train)
                X_train_scaled = self._transform_data(X_train, preprocessors)
                X_test_scaled = self._transform_data(X_test, preprocessors)
                # --- NEW: Calculate and store training data statistics ---
                training_stats = {
                    'mean': X_train.mean().to_dict(),
                    'std': X_train.std().to_dict(),
                    'min': X_train.min().to_dict(),
                    'max': X_train.max().to_dict(),
                    'count': len(X_train)
                }
                # --- End Add ---

                # --- NEW: Hyperparameter Tuning ---
                model_config = self.config['classification'] # Assuming RandomForestClassifier here
                if self.config['hyperparameter_tuning']['enabled']:
                    param_dist = {
                        'n_estimators': [50, 100, 200, 300],
                        'max_depth': [None, 5, 10, 15, 20],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'class_weight': [class_weight, None] # Tune class weight too
                    }
                    tuned_model = self._perform_hyperparameter_tuning(
                        RandomForestClassifier, param_dist, X_train_scaled, y_train
                    )
                    # Update model with best estimator and handle random_state
                    tuned_model.set_params(random_state=model_config['random_state'])
                    model = tuned_model

                else:
                    model = RandomForestClassifier(
                        n_estimators=model_config['n_estimators'],
                        max_depth=model_config['max_depth'],
                        random_state=model_config['random_state'],
                        class_weight=class_weight
                    )
                # --- End Tuning ---

                model.fit(X_train_scaled, y_train)

                # --- Evaluation & MLflow Logging (Updated) ---
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring=self.config['hyperparameter_tuning']['scoring'])
                feature_importance = dict(zip(features.columns, model.feature_importances_))
                y_pred = model.predict(X_test_scaled)
                # ... (metric calculation logic remains the same) ...
                evaluation_metrics = {}
                try:
                    if is_binary:
                        y_proba = model.predict_proba(X_test_scaled)[:, 1]
                        evaluation_metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
                        evaluation_metrics['precision'] = precision_score(y_test, y_pred, average='binary', zero_division=0)
                        evaluation_metrics['recall'] = recall_score(y_test, y_pred, average='binary', zero_division=0)
                        evaluation_metrics['f1_score'] = f1_score(y_test, y_pred, average='binary', zero_division=0)
                    else: # Multiclass
                        y_proba = model.predict_proba(X_test_scaled)
                        evaluation_metrics['roc_auc_ovr'] = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
                        evaluation_metrics['precision_macro'] = precision_score(y_test, y_pred, average='macro', zero_division=0)
                        evaluation_metrics['recall_macro'] = recall_score(y_test, y_pred, average='macro', zero_division=0)
                        evaluation_metrics['f1_score_macro'] = f1_score(y_test, y_pred, average='macro', zero_division=0)
                    evaluation_metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
                except Exception as e:
                    self.logger.warning(f"Could not calculate some classification metrics: {e}")
                    evaluation_metrics['error'] = str(e)


                if MLFLOW_AVAILABLE:
                    # Log final parameters
                    mlflow.log_params(model.get_params())
                    mlflow.log_param("tuning_enabled", self.config['hyperparameter_tuning']['enabled'])
                    # ... (log other params like sample counts) ...
                    mlflow.log_param("training_samples", len(X_train))
                    mlflow.log_param("features_count", len(features.columns))

                    # Remove non-scalar metrics before logging
                    mlflow_metrics = evaluation_metrics.copy()
                    mlflow_metrics.pop('confusion_matrix', None)
                    mlflow_metrics.pop('error', None)
                    mlflow.log_metrics({
                        'train_score': train_score,
                        'test_score': test_score,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        **mlflow_metrics
                    })
                    # Log training stats artifacts
                    stats_path = Path(mlflow.get_artifact_uri()) / "training_stats.json"
                    stats_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(stats_path, 'w') as f:
                        json.dump(training_stats, f)
                    mlflow.log_artifact(str(stats_path))
                    # Log model
                    mlflow.sklearn.log_model(model, "model")
                # --- End MLflow ---

                metadata = {
                    'model_type': 'RandomForestClassifier',
                    'trained_at': datetime.now().isoformat(),
                    'timestamp': timestamp,
                    'training_samples': len(X_train),
                    'features': list(features.columns),
                    'target_classes': [str(c) for c in model.classes_], # Ensure serializable
                    'train_score': train_score,
                    'test_score': test_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'feature_importance': {k: float(v) for k,v in feature_importance.items()}, # Ensure serializable
                    'evaluation_metrics': evaluation_metrics,
                    # Store final model parameters
                    'model_params': model.get_params(),
                    'tuning_enabled': self.config['hyperparameter_tuning']['enabled'],
                    'mlflow_run_id': run.info.run_id if MLFLOW_AVAILABLE else None
                }

                # --- Save model with stats ---
                self._save_model(
                    model_name, model,
                    preprocessors, # Pass the dict directly
                    metadata, training_stats=training_stats, timestamp=timestamp
                )

                result = {
                    'model_name': model_name,
                    'train_score': train_score,
                    'test_score': test_score,
                    'cv_scores': { 'mean': cv_scores.mean(), 'std': cv_scores.std(), 'scores': cv_scores.tolist() },
                    'feature_importance': {k: float(v) for k,v in feature_importance.items()}, # Ensure serializable
                    'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
                    'evaluation_metrics': evaluation_metrics,
                    'metadata': metadata
                }

                self._set_cache(cache_key, result)
                self.logger.info(f"Failure predictor trained: {model_name}")
                if MLFLOW_AVAILABLE: mlflow.end_run(status="FINISHED")
                return result

            except Exception as e:
                self.logger.error(f"Failure prediction training error: {e}", exc_info=True)
                if MLFLOW_AVAILABLE: mlflow.end_run(status="FAILED")
                raise

    @celery_app.task(bind=True, name='analytics.predict_failure')
    def predict_failure(self, data: pd.DataFrame, model_name: str = "failure_predictor") -> Dict:
        """
        Predict failures (Celery Task).
        - MODULARIZED: Uses _transform_data
        - DRIFT: Performs data drift check (NEW)
        - EXPLAIN: Adds SHAP/LIME explanation (NEW)
        """
        try:
            if model_name not in self.models:
                # ... (model loading/training fallback remains the same) ...
                raise ValueError(f"Model {model_name} not found") # Simplified for now

            features, _ = self.prepare_data(data)
            metadata = self.model_metadata.get(model_name, {})
            model_features = metadata.get('features', [])
            if model_features:
                 features = features.reindex(columns=model_features, fill_value=0)
            if features.empty:
                 # ... (empty features handling remains the same) ...
                 return {'predictions': [], 'failure_probabilities': []}


            # --- NEW: Data Drift Check ---
            drift_detected = False
            drift_report = {}
            if self.config['data_drift']['enabled']:
                 self.prediction_counters[model_name] += len(features)
                 if self.prediction_counters[model_name] >= self.config['data_drift']['check_frequency']:
                     self.logger.info(f"Checking data drift for model {model_name}...")
                     drift_report = self._check_data_drift(features, model_name)
                     drift_detected = drift_report.get('drift_detected', False)
                     if drift_detected:
                         self.logger.warning(f"Data drift detected for model {model_name}!", drift_details=drift_report.get('drifting_features'))
                         # Optional: Trigger retraining
                     self.prediction_counters[model_name] = 0
            # --- End Drift Check ---

            # --- Preprocessing & Prediction ---
            preprocessors = self.preprocessors.get(model_name, {})
            if not preprocessors.get('scaler') or not preprocessors.get('imputer'):
                 raise ValueError(f"Preprocessors for model {model_name} not found.")
            features_scaled = self._transform_data(features, preprocessors)

            model = self.models[model_name]
            predictions = model.predict(features_scaled)
            probabilities = model.predict_proba(features_scaled)

            # ... (probability extraction logic remains the same) ...
            failure_class_label = 1 # Assuming 1 represents failure
            classes = metadata.get('target_classes', model.classes_)
            try:
                # Convert classes to string if stored as string in metadata
                classes_list = [str(c) for c in classes]
                failure_class_index = classes_list.index(str(failure_class_label))
                failure_probs = probabilities[:, failure_class_index]
            except (ValueError, TypeError, IndexError):
                self.logger.warning(f"Could not reliably find failure class index. Using index 1 if binary.", classes=classes)
                # Fallback assuming binary classification and failure is the second class
                if probabilities.shape[1] == 2:
                    failure_probs = probabilities[:, 1]
                else: # Cannot determine failure probability for multi-class without clear index
                    failure_probs = np.zeros(len(predictions)) # Return 0 probability

            # ... (risk threshold logic remains the same) ...
            risk_threshold = 0.7 # Example threshold
            high_risk_indices = np.where(failure_probs >= risk_threshold)[0]

            # --- NEW: Explainability ---
            explanations = {}
            if self.config['explainability']['enabled'] and len(high_risk_indices) > 0:
                 try:
                     # Explain the first few high-risk predictions
                     explain_indices = high_risk_indices[:min(3, len(high_risk_indices))]
                     if SHAP_AVAILABLE:
                         # Use TreeExplainer for RandomForest
                         shap_explainer = shap.TreeExplainer(model)
                         shap_values = shap_explainer.shap_values(features_scaled) # Explain all classes
                         
                         # shap_values is a list [shap_class_0, shap_class_1]
                         shap_vals_failure_class = shap_values[failure_class_index]
                         
                         instances_to_explain = features_scaled[explain_indices]
                         shap_vals_instances = shap_explainer.shap_values(instances_to_explain)[failure_class_index]

                         explanations['shap'] = {}
                         for i, original_index in enumerate(explain_indices):
                             explanations['shap'][str(original_index)] = dict(zip(model_features, shap_vals_instances[i].tolist()))

                     if LIME_AVAILABLE:
                         # Use model.predict_proba for LIME classification
                         predict_proba_fn = model.predict_proba
                         lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                             training_data=features_scaled, # Use scaled data
                             feature_names=model_features,
                             class_names=[str(c) for c in classes],
                             mode='classification'
                         )
                         explanations['lime'] = {}
                         for original_index in explain_indices:
                             instance = features_scaled[original_index]
                             explanation = lime_explainer.explain_instance(
                                 data_row=instance,
                                 predict_fn=predict_proba_fn,
                                 num_features=self.config['explainability']['lime_num_features'],
                                 labels=(failure_class_index,) # Explain the failure class
                             )
                             explanations['lime'][str(original_index)] = explanation.as_list()

                 except Exception as explain_e:
                     self.logger.error(f"Error generating explanations for failure prediction: {explain_e}", exc_info=True)
                     explanations['error'] = str(explain_e)
            # --- End Explainability ---

            result = {
                'predictions': predictions.tolist(),
                'failure_probabilities': failure_probs.tolist(),
                'high_risk_count': len(high_risk_indices),
                'high_risk_indices': high_risk_indices.tolist(),
                'risk_distribution': {
                     # ... (distribution calculation) ...
                     'low_risk': int((failure_probs < 0.3).sum()),
                    'medium_risk': int(((failure_probs >= 0.3) & (failure_probs < 0.7)).sum()),
                    'high_risk': int((failure_probs >= 0.7).sum())
                },
                'summary': {
                     # ... (summary calculation) ...
                     'total_samples': len(data),
                    'predicted_failures': int((predictions == failure_class_label).sum()),
                    'max_failure_probability': float(failure_probs.max()) if len(failure_probs) > 0 else 0,
                    'mean_failure_probability': float(failure_probs.mean()) if len(failure_probs) > 0 else 0

                },
                'drift_check': { # NEW
                    'performed': self.prediction_counters[model_name] == 0 and self.config['data_drift']['enabled'],
                    'drift_detected': drift_detected,
                    'report': drift_report
                },
                'explanations': explanations # NEW
            }

            self.logger.info(f"Failure prediction completed: {len(high_risk_indices)} high-risk cases")
            return result
        except Exception as e:
            self.logger.error(f"Failure prediction error: {e}", exc_info=True)
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
        - FIX 2: Adds loss viz and versioning
        - DRIFT: Saves training data stats (NEW)
        """
        if not TENSORFLOW_AVAILABLE:
            self.logger.error("TensorFlow is not installed. Cannot train LSTM.")
            return {'error': 'TensorFlow is not available.'}

        # --- FIX 2: Generate timestamp ---
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_name=f"train_lstm_forecaster_{model_name}_{timestamp}") as run:
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

                ts_data_df = data[[target_column]] # Keep as DataFrame

                # --- Preprocessing & Training Stats (Adapted for 1D) ---
                # 1. Impute
                imputer = SimpleImputer(strategy='mean')
                ts_imputed = imputer.fit_transform(ts_data_df)

                if len(ts_imputed) < n_steps_in + n_steps_out:
                    msg = f"Not enough data ({len(ts_imputed)}) to train LSTM."
                    self.logger.error(msg)
                    if MLFLOW_AVAILABLE: mlflow.end_run(status="FAILED")
                    return {'error': msg}

                # 2. Scale
                scaler = MinMaxScaler(feature_range=(0, 1))
                ts_scaled = scaler.fit_transform(ts_imputed)

                preprocessors = {'imputer': imputer, 'scaler': scaler}
                # --- NEW: Calculate and store training data statistics (on imputed data) ---
                ts_imputed_df = pd.DataFrame(ts_imputed, columns=[target_column])
                training_stats = {
                    'mean': ts_imputed_df.mean().to_dict(),
                    'std': ts_imputed_df.std().to_dict(),
                    'min': ts_imputed_df.min().to_dict(),
                    'max': ts_imputed_df.max().to_dict(),
                    'count': len(ts_imputed_df)
                }
                # --- End Add ---

                X_seq, y_seq = self._create_lstm_sequences(ts_scaled, n_steps_in, n_steps_out)
                if X_seq.shape[0] == 0:
                    msg = f"Not enough data to create sequences of length {n_steps_in}."
                    self.logger.error(msg)
                    if MLFLOW_AVAILABLE: mlflow.end_run(status="FAILED")
                    return {'error': msg}

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

                # --- FIX 2: Loss Visualization ---
                self._save_loss_plot(history, model_name, timestamp)

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
                    # Log training stats artifacts
                    stats_path = Path(mlflow.get_artifact_uri()) / "training_stats.json"
                    stats_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(stats_path, 'w') as f:
                        json.dump(training_stats, f)
                    mlflow.log_artifact(str(stats_path))
                    mlflow.tensorflow.log_model(model, "model")
                # --- END FIX 3 & 4 ---

                metadata = {
                    'model_type': 'LSTM_Forecaster',
                    'trained_at': datetime.now().isoformat(),
                    'timestamp': timestamp, # FIX 2
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

                # --- Save model with stats ---
                self._save_model(
                    model_name, model,
                    preprocessors, # Pass the dict directly
                    metadata, training_stats=training_stats, timestamp=timestamp
                )

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
                self.logger.error(f"LSTM Forecaster training error: {e}", exc_info=True)
                if MLFLOW_AVAILABLE: mlflow.end_run(status="FAILED")
                raise

    @celery_app.task(bind=True, name='analytics.predict_lstm_forecast')
    def predict_lstm_forecast(self, data: pd.DataFrame, model_name: str = "lstm_forecaster") -> Dict:
        """
        Make LSTM forecast (Celery Task).
        - MODULARIZED: Uses _transform_data (adapted for 1D)
        - DRIFT: Performs data drift check (NEW)
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

            ts_data_df = data[[target_column]] # Keep as DataFrame
            if len(ts_data_df) < n_steps_in:
                msg = f"Not enough data ({len(ts_data_df)}) to make prediction, need {n_steps_in}."
                self.logger.error(msg)
                return {'error': msg}

            input_seq_raw = ts_data_df.tail(n_steps_in) # Get last N samples as DataFrame

            # --- NEW: Data Drift Check (on the input sequence) ---
            drift_detected = False
            drift_report = {}
            if self.config['data_drift']['enabled']:
                 self.prediction_counters[model_name] += 1 # Count per prediction call
                 if self.prediction_counters[model_name] >= self.config['data_drift']['check_frequency']:
                     self.logger.info(f"Checking data drift for model {model_name}...")
                     # Use the raw input sequence for drift check
                     drift_report = self._check_data_drift(input_seq_raw, model_name)
                     drift_detected = drift_report.get('drift_detected', False)
                     if drift_detected:
                         self.logger.warning(f"Data drift detected for model {model_name}!", drift_details=drift_report.get('drifting_features'))
                     self.prediction_counters[model_name] = 0
            # --- End Drift Check ---

            # --- Preprocessing (adapted for 1D) ---
            input_seq_imputed = imputer.transform(input_seq_raw)
            input_seq_scaled = scaler.transform(input_seq_imputed)
            # --- END ---

            input_seq_scaled = input_seq_scaled.reshape((1, n_steps_in, 1))
            y_pred_scaled = model.predict(input_seq_scaled)

            y_pred_scaled = y_pred_scaled.reshape(-1, 1)
            y_pred = scaler.inverse_transform(y_pred_scaled)

            forecast = y_pred.flatten().tolist()

            result = {
                'model_name': model_name,
                'forecast': forecast,
                'forecast_steps': n_steps_out,
                'drift_check': { # NEW
                    'performed': self.prediction_counters[model_name] == 0 and self.config['data_drift']['enabled'],
                    'drift_detected': drift_detected,
                    'report': drift_report
                }
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
            # Fetch a larger dataset for retraining
            data = self.db_manager.get_health_data_as_dataframe(limit=50000) # Increased limit
            if data.empty or len(data) < 100:
                self.logger.warning("Not enough new data to retrain. Skipping.")
                return {"status": "skipped", "message": "Not enough new data"}

            results = {}
            models_retrained = []

            # Define features for each model
            # This should ideally come from config or model metadata
            anomaly_feature_cols = ['temperature', 'pressure', 'vibration', 'efficiency']
            
            # Filter data to only include columns that actually exist
            available_cols = data.columns
            anomaly_features_data = data[[col for col in anomaly_feature_cols if col in available_cols]]


            # Retrain Anomaly Detector (Isolation Forest)
            if not anomaly_features_data.empty:
                try:
                    self.logger.info("Retraining anomaly_detector...")
                    anomaly_results = self.train_anomaly_detector(anomaly_features_data, "anomaly_detector")
                    if 'error' not in anomaly_results:
                         results["anomaly_detector"] = anomaly_results.get('metadata', {})
                         models_retrained.append("anomaly_detector")
                    else: raise ValueError(anomaly_results['error'])
                except Exception as e:
                    self.logger.error(f"Failed to retrain anomaly_detector: {e}")
                    results["anomaly_detector"] = {"error": str(e)}
            else:
                 self.logger.warning("Skipping anomaly_detector: No relevant features found in data.")


            # Retrain Anomaly Detector (Autoencoder)
            if TENSORFLOW_AVAILABLE and not anomaly_features_data.empty:
                try:
                    self.logger.info("Retraining anomaly_autoencoder...")
                    ae_results = self.train_anomaly_autoencoder(anomaly_features_data, "anomaly_autoencoder")
                    if 'error' not in ae_results:
                         results["anomaly_autoencoder"] = ae_results.get('metadata', {})
                         models_retrained.append("anomaly_autoencoder")
                    else: raise ValueError(ae_results['error'])
                except Exception as e:
                    self.logger.error(f"Failed to retrain anomaly_autoencoder: {e}")
                    results["anomaly_autoencoder"] = {"error": str(e)}
            elif not TENSORFLOW_AVAILABLE:
                 self.logger.info("Skipping anomaly_autoencoder retraining: TensorFlow not available.")
            else:
                 self.logger.warning("Skipping anomaly_autoencoder: No relevant features found in data.")


            # Retrain Failure Predictor (if target exists)
            if 'failure' in data.columns:
                try:
                    self.logger.info("Retraining failure_predictor...")
                    failure_results = self.train_failure_predictor(data, 'failure', "failure_predictor")
                    if 'error' not in failure_results:
                         results["failure_predictor"] = failure_results.get('metadata', {})
                         models_retrained.append("failure_predictor")
                    else: raise ValueError(failure_results['error'])
                except Exception as e:
                    self.logger.error(f"Failed to retrain failure_predictor: {e}")
                    results["failure_predictor"] = {"error": str(e)}
            else:
                self.logger.info("Skipping failure_predictor retraining: 'failure' column not in data.")

            # Retrain LSTM Forecaster (if applicable)
            if 'temperature' in data.columns and TENSORFLOW_AVAILABLE:
                try:
                    self.logger.info("Retraining lstm_forecaster...")
                    lstm_results = self.train_forecaster_lstm(data, 'temperature', 'lstm_forecaster')
                    if 'error' not in lstm_results:
                         results["lstm_forecaster"] = lstm_results.get('metadata', {})
                         models_retrained.append("lstm_forecaster")
                    else: raise ValueError(lstm_results['error'])
                except Exception as e:
                    self.logger.error(f"Failed to retrain lstm_forecaster: {e}")
                    results["lstm_forecaster"] = {"error": str(e)}

            self.logger.info(f"Scheduled model retraining completed. Models updated: {', '.join(models_retrained) or 'None'}")
            return {"status": "success", "results": results, "models_retrained": models_retrained}

        except Exception as e:
            self.logger.error(f"Scheduled retraining failed: {e}", exc_info=True)
            # Ensure Celery task shows failure state
            # self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
            # Reraise the exception for Celery to handle
            raise

    # --- Advanced Analysis Methods (Unchanged) ---
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
            
            if ts.empty:
                self.logger.warning(f"Time series analysis failed for {value_column}: No data after imputation.")
                return {'error': f'No data for {value_column} after imputation.'}


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

                except Exception as e_exp:
                    self.logger.warning(f"Complex ExpSmoothing failed: {e_exp}. Falling back.")
                    # Fallback to simple exponential smoothing
                    try:
                        exp_model = ExponentialSmoothing(ts, trend='add').fit()
                        exp_forecast = exp_model.forecast(steps)
                        forecasts['exponential_smoothing'] = exp_forecast.tolist()
                    except Exception as e_simple_exp:
                        self.logger.warning(f"Fallback ExpSmoothing failed: {e_simple_exp}")

            # ARIMA (simplified)
            if len(ts) > 50:
                try:
                    arima_model = ARIMA(ts, order=(1, 1, 1)).fit()
                    arima_forecast = arima_model.forecast(steps)
                    forecasts['arima'] = arima_forecast.tolist()
                except Exception as e_arima:
                    self.logger.warning(f"ARIMA failed: {e_arima}")
                    pass # ARIMA can be sensitive, fail gracefully

            # --- NEW: Add LSTM Forecast ---
            # Check if a relevant LSTM model exists
            lstm_model_name = "lstm_forecaster" # Or derive from value_column
            if (lstm_model_name in self.models and
                self.model_metadata.get(lstm_model_name, {}).get('target_column') == value_column):
                try:
                    # Call the task.
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
            if features.empty:
                self.logger.warning("Cluster analysis failed: No numeric features.")
                return {'error': 'No numeric features for clustering.'}


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
                # Inverse transform centers
                cluster_centers_original_scale = preprocessors['scaler'].inverse_transform(cluster_centers_scaled)
                # Apply inverse imputation (less common, but for completeness)
                # This step is complex as inverse_transform of imputer isn't direct
                # We'll return the inverse-scaled centers, which is usually most interpretable
                
                # Get feature names back from preprocessors (if stored, or from features)
                feature_names = features.columns.tolist()
                centers_df = pd.DataFrame(cluster_centers_original_scale, columns=feature_names)


                result = {
                    'method': 'kmeans',
                    'n_clusters': n_clusters,
                    'cluster_labels': cluster_labels.tolist(),
                    'cluster_centers': centers_df.to_dict('records'),
                    'inertia': float(kmeans.inertia_),
                    'silhouette_score': None # Requires more computation
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
            
            else:
                 raise ValueError(f"Unknown clustering method: {method}")

            cluster_stats = self._calculate_cluster_stats(features, cluster_labels)
            result['cluster_statistics'] = cluster_stats

            self.logger.info(f"Cluster analysis completed: {result.get('n_clusters')} clusters found")
            return result

        except Exception as e:
            self.logger.error(f"Cluster analysis error: {e}", exc_info=True)
            raise

    def _find_optimal_clusters(self, data: np.ndarray, max_clusters: int = 10) -> int:
        """Find optimal number of clusters using elbow method."""
        try:
            inertias = []
            # Ensure max_clusters is not greater than n_samples - 1
            safe_max_clusters = min(max_clusters, data.shape[0] - 1)
            if safe_max_clusters < 1: return 1 # Not enough data
            
            k_range = range(1, safe_max_clusters + 1)

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                kmeans.fit(data)
                inertias.append(kmeans.inertia_)

            if len(inertias) >= 3:
                deltas = np.diff(inertias)
                delta_deltas = np.diff(deltas)
                if len(delta_deltas) > 0:
                    elbow_idx = np.argmax(delta_deltas) + 2 # +1 for diff, +1 for 0-index
                    return min(elbow_idx, safe_max_clusters)
            
            # Fallback if elbow isn't clear
            return 3 if safe_max_clusters >= 3 else safe_max_clusters

        except Exception as e:
            self.logger.error(f"Optimal cluster finding error: {e}")
            return 3

    def _calculate_cluster_stats(self, features: pd.DataFrame, labels: np.ndarray) -> Dict:
        """Calculate statistics for each cluster."""
        try:
            stats = {}
            unique_labels = np.unique(labels)
            
            # Ensure features index matches labels
            if features.shape[0] != len(labels):
                self.logger.error(f"Feature count ({features.shape[0]}) and label count ({len(labels)}) mismatch.")
                return {}

            features_reset = features.reset_index(drop=True)

            for label in unique_labels:
                label_str = f'cluster_{label}'
                if label == -1:
                    label_str = 'noise'

                cluster_data = features_reset[labels == label]
                
                stats[label_str] = {
                    'size': len(cluster_data),
                    'mean': cluster_data.mean().to_dict(),
                    'std': cluster_data.std().to_dict(),
                    'min': cluster_data.min().to_dict(),
                    'max': cluster_data.max().to_dict()
                }
            return stats
        except Exception as e:
            self.logger.error(f"Cluster statistics error: {e}", exc_info=True)
            return {}

    def pattern_recognition(self, data: pd.DataFrame, pattern_type: str = 'peaks') -> Dict:
        """
        Recognize patterns in data.
        - MODULARIZED: Uses _fit_preprocessors
        """
        try:
            features, _ = self.prepare_data(data)
            if features.empty:
                self.logger.warning("Pattern recognition failed: No numeric features.")
                return {'error': 'No numeric features for pattern recognition.'}


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
                    # De-trend data before autocorrelation
                    detrended = values - np.poly1d(np.polyfit(np.arange(len(values)), values, 1))(np.arange(len(values)))
                    autocorr = np.correlate(detrended, detrended, mode='full')
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
            if features.empty or target is None:
                self.logger.warning("Optimization analysis failed: No features or target.")
                return {'error': 'No features or target for optimization.'}


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
                        'model_name': model_name,
                        'metadata': self.model_metadata[model_name],
                        'preprocessors': self.preprocessors.get(model_name),
                        'training_stats': self.training_data_stats.get(model_name)
                    }
                else:
                    return {'error': f'Model {model_name} not found.'}
            else:
                return {
                    'available_models': list(self.models.keys()),
                    'all_model_metadata': self.model_metadata,
                    'all_training_stats': self.training_data_stats
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
                json.dump(results, f, indent=2, default=str) # Use default=str for non-serializable

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


# --- Example Usage (main block) ---
if __name__ == "__main__":
    # Note: This main block is for demonstration and testing.
    # In production, this class is instantiated and its methods called by Celery/Flask.
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    analytics = PredictiveAnalyticsEngine()

    # --- Generate Sample Data ---
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=1000, freq="H")
    sample_data = pd.DataFrame({
        "timestamp": dates,
        "temperature": 20 + 5 * np.sin(np.arange(1000) * 2 * np.pi / 24) + np.random.normal(0, 1, 1000),
        "pressure": 1013 + np.random.normal(0, 5, 1000),
        "vibration": 0.1 + np.random.exponential(0.05, 1000),
        "efficiency": np.random.uniform(70, 95, 1000),
        "failure": np.random.choice([0, 1], 1000, p=[0.95, 0.05])
    })
    # Introduce missing values
    for col in ["temperature", "pressure", "vibration", "efficiency"]:
        sample_data.loc[sample_data.sample(frac=0.05).index, col] = np.nan
    print(f"Sample data created with {sample_data.isna().sum().sum()} missing values.")

    # --- Train Models (with Tuning and Explainability) ---
    print("\n--- Training Standard Models (with potential tuning) ---")
    
    # Define features
    feature_cols = ["temperature", "pressure", "vibration", "efficiency"]

    # Isolation Forest - Anomaly Detection
    print("Training anomaly detector (IsolationForest)...")
    anomaly_result = analytics.train_anomaly_detector(
        sample_data[feature_cols]
    )
    if "error" not in anomaly_result:
        print(f"  IsolationForest trained. Final anomaly ratio: {anomaly_result.get('anomaly_ratio', 'N/A'):.3f}")
        meta = anomaly_result.get("metadata", {})
        print(f"  Tuning Enabled: {meta.get('tuning_enabled')}")
        if meta.get("tuning_enabled"):
            print(f"  Best Params: {meta.get('model_params')}")
    else:
        print(f"  Training failed: {anomaly_result['error']}")

    # Random Forest - Failure Prediction
    print("\nTraining failure predictor (RandomForest)...")
    failure_result = analytics.train_failure_predictor(sample_data, "failure")
    if "error" not in failure_result:
        print(f"  Failure predictor trained. Test score: {failure_result.get('test_score', 'N/A'):.3f}")
        meta = failure_result.get("metadata", {})
        print(f"  Tuning Enabled: {meta.get('tuning_enabled')}")
        if meta.get("tuning_enabled"):
            print(f"  Best Params: {meta.get('model_params')}")
    else:
        print(f"  Training failed: {failure_result['error']}")
        
    # --- Train DL Models (if available) ---
    ae_result = {}
    lstm_result = {}
    if TENSORFLOW_AVAILABLE:
        print("\n--- Training Deep Learning Models ---")
        print("Training anomaly detector (Autoencoder)...")
        ae_result = analytics.train_anomaly_autoencoder(sample_data[feature_cols])
        if 'error' not in ae_result:
            print(f"  Autoencoder trained. Threshold: {ae_result['reconstruction_threshold']:.4f}")
        else:
            print(f"  Autoencoder training failed: {ae_result['error']}")

        print("\nTraining LSTM forecaster (for temperature)...")
        lstm_result = analytics.train_forecaster_lstm(sample_data, 'temperature')
        if 'error' not in lstm_result:
            print(f"  LSTM Forecaster trained.")
        else:
            print(f"  LSTM training failed: {lstm_result['error']}")


    # --- Test Predictions on New Data (Drift + Explainability) ---
    print("\n--- Testing Predictions on New Data (with Drift Check & Explainability) ---")
    new_data = sample_data.tail(150).copy() # Use 150 to trigger check (freq=100)
    # Introduce drift
    new_data["temperature"] += 5
    new_data["vibration"] *= 1.5
    new_data.loc[new_data.sample(frac=0.1).index, "pressure"] = np.nan

    # Anomaly Prediction (IF)
    anomaly_pred = analytics.detect_anomalies(
        new_data[feature_cols]
    )
    print(f"\nIsolationForest Prediction:")
    print(f"  Anomalies Found: {anomaly_pred.get('anomaly_count', 'N/A')}")
    drift_check = anomaly_pred.get("drift_check", {})
    print(f"  Drift Check Performed: {drift_check.get('performed')}")
    if drift_check.get("drift_detected"):
        print(f"  Drift Detected! Features: {drift_check.get('report', {}).get('drifting_features')}")
    explanations = anomaly_pred.get("explanations", {})
    if explanations.get("shap"):
        print("  SHAP Explanations Generated (for first few anomalies).")
    if explanations.get("lime"):
        print("  LIME Explanations Generated (for first few anomalies).")

    # Failure Prediction (RF)
    failure_pred = analytics.predict_failure(new_data)
    print(f"\nFailure Prediction:")
    print(f"  High-Risk Count: {failure_pred.get('high_risk_count', 'N/A')}")
    drift_check_fail = failure_pred.get("drift_check", {})
    print(f"  Drift Check Performed: {drift_check_fail.get('performed')}")
    if drift_check_fail.get("drift_detected"):
        print(f"  Drift Detected! Features: {drift_check_fail.get('report', {}).get('drifting_features')}")
    explanations_fail = failure_pred.get("explanations", {})
    if explanations_fail.get("shap"):
        print("  SHAP Explanations Generated (for first few high-risk).")
    if explanations_fail.get("lime"):
        print("  LIME Explanations Generated (for first few high-risk).")

    # Anomaly Prediction (AE)
    ae_pred = {}
    if TENSORFLOW_AVAILABLE and 'error' not in ae_result:
        ae_pred = analytics.detect_anomalies_autoencoder(new_data[feature_cols])
        print(f"\nAutoencoder Prediction:")
        print(f"  Anomalies Found: {ae_pred.get('anomaly_count', 'N/A')}")
        drift_check_ae = ae_pred.get("drift_check", {})
        print(f"  Drift Check Performed: {drift_check_ae.get('performed')}")
        if drift_check_ae.get("drift_detected"):
            print(f"  Drift Detected! Features: {drift_check_ae.get('report', {}).get('drifting_features')}")
            
    # LSTM Forecast
    lstm_pred = {}
    if TENSORFLOW_AVAILABLE and 'error' not in lstm_result:
        lstm_pred = analytics.predict_lstm_forecast(new_data)
        print(f"\nLSTM Forecast:")
        print(f"  Forecast steps: {len(lstm_pred.get('forecast', []))}")
        drift_check_lstm = lstm_pred.get("drift_check", {})
        print(f"  Drift Check Performed: {drift_check_lstm.get('performed')}")
        if drift_check_lstm.get("drift_detected"):
            print(f"  Drift Detected! Features: {drift_check_lstm.get('report', {}).get('drifting_features')}")

    # --- Other Analyses (optional extensions) ---
    print("\n--- Running Other Analyses ---")
    print("Performing time series analysis (StatsModels + LSTM)...")
    ts_result = analytics.time_series_analysis(sample_data, "temperature", "timestamp")
    print(f"Time series analysis completed. Trend: {ts_result.get('trend_analysis',{}).get('trend_direction')}")
    if 'lstm' in ts_result.get('forecast', {}).get('forecasts', {}):
        print("  LSTM forecast successfully included in time series ensemble.")


    print("\nPerforming cluster analysis (KMeans)...")
    cluster_result = analytics.cluster_analysis(
        sample_data[feature_cols]
    )
    print(f"Cluster analysis completed. Found {cluster_result.get('n_clusters')} clusters.")

    # --- Export Results ---
    print("\n--- Exporting All Results ---")
    all_results = {
        "anomaly_training_if": anomaly_result,
        "failure_training_rf": failure_result,
        "anomaly_training_ae": ae_result,
        "lstm_training": lstm_result,
        "time_series": ts_result,
        "clustering": cluster_result,
        "predictions": {
            "anomalies_if": anomaly_pred,
            "failures_rf": failure_pred,
            "anomalies_ae": ae_pred,
            "forecast_lstm": lstm_pred
        },
    }
    export_path = analytics.export_analysis_results(all_results)
    print(f"\nAll results exported to: {export_path}")

    print("\n✅ Analytics engine demonstration completed with new features!")