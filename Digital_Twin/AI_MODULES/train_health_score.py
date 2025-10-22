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
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder # Added OneHotEncoder
from sklearn.compose import ColumnTransformer # Added ColumnTransformer
from sklearn.pipeline import Pipeline # Added Pipeline
from sklearn.impute import SimpleImputer
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
    The final score is a blend of these two methods. Includes confidence scoring.
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
        self.health_history = deque(maxlen=2000)
        self.component_history = defaultdict(lambda: deque(maxlen=1000))

        # --- Model-Based Scoring Attributes ---
        self.scoring_model_name = "health_scorer" # Base name, extension added later
        self.scoring_model = None
        self.scoring_model_metadata = {}
        # --- NEW: Store the preprocessing pipeline ---
        self.preprocessing_pipeline = None
        # Load the analytics engine to get predictive features
        self.analytics_engine = PredictiveAnalyticsEngine() if ANALYTICS_ENGINE_AVAILABLE else None
        self._load_scoring_model_and_pipeline() # Updated load function

        # Thresholds and weights
        self.health_thresholds = self.config.get('thresholds', {
            'critical': 0.3,
            'warning': 0.6,
            'good': 0.8,
            'excellent': 0.95
        })

        # Component weights for overall health calculation
        self.component_weights = self.config.get('component_weights', {})

        # Adaptive EMA Baselines
        self.baseline_emas = {}
        self.baseline_ema_alpha = self.config.get('baseline_ema_alpha', 0.1)

    def _setup_logging(self):
        """Setup logging for health score calculator."""
        logger = logging.getLogger('HealthScoreCalculator')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
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
        # (Implementation remains the same as provided)
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
                    "baseline_ema_alpha": 0.1, # Slow learning rate for adaptive baselines
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


    # --- Model Loading/Saving ---

    def _load_scoring_model_and_pipeline(self):
        """Load the trained model and preprocessing pipeline."""
        model_file = self.model_path / f"{self.scoring_model_name}.pkl"
        pipeline_file = self.model_path / f"{self.scoring_model_name}_pipeline.pkl"
        metadata_file = self.model_path / f"{self.scoring_model_name}_metadata.json"

        try:
            if model_file.exists():
                self.scoring_model = joblib.load(model_file)
                self.logger.info(f"Health scoring model '{model_file.name}' loaded.")
            else:
                self.logger.info(f"Health scoring model '{model_file.name}' not found.")

            if pipeline_file.exists():
                self.preprocessing_pipeline = joblib.load(pipeline_file)
                self.logger.info(f"Preprocessing pipeline '{pipeline_file.name}' loaded.")
            else:
                self.logger.info(f"Preprocessing pipeline '{pipeline_file.name}' not found.")

            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.scoring_model_metadata = json.load(f)
                self.logger.info(f"Model metadata '{metadata_file.name}' loaded.")
            else:
                self.scoring_model_metadata = {}

        except Exception as e:
            self.logger.error(f"Failed to load model/pipeline/metadata: {e}")
            self.scoring_model = None
            self.preprocessing_pipeline = None
            self.scoring_model_metadata = {}

    def _save_scoring_model_and_pipeline(self):
        """Save the trained model and preprocessing pipeline."""
        if self.scoring_model is None or self.preprocessing_pipeline is None:
            self.logger.warning("No model or pipeline to save.")
            return

        model_file = self.model_path / f"{self.scoring_model_name}.pkl"
        pipeline_file = self.model_path / f"{self.scoring_model_name}_pipeline.pkl"
        metadata_file = self.model_path / f"{self.scoring_model_name}_metadata.json"

        try:
            joblib.dump(self.scoring_model, model_file)
            joblib.dump(self.preprocessing_pipeline, pipeline_file)
            with open(metadata_file, 'w') as f:
                json.dump(self.scoring_model_metadata, f, indent=2)

            self.logger.info(f"Model, pipeline, and metadata saved for '{self.scoring_model_name}'")
        except Exception as e:
            self.logger.error(f"Failed to save model/pipeline/metadata: {e}")

    # --- REFACTORED: prepare_data ---
    def prepare_data(self, data: pd.DataFrame, fit_pipeline: bool = False) -> Optional[pd.DataFrame]:
        """
        Prepare data for machine learning using a consistent pipeline.
        Handles both numeric and categorical features.

        Args:
            data: DataFrame with raw input data (can include non-feature columns).
            fit_pipeline: If True, fit the preprocessing pipeline (used for training).
                          If False, only transform using the existing pipeline (used for prediction).

        Returns:
            DataFrame with preprocessed features, ready for the model, or None if error.
            The columns will match the expected input features of the model.
        """
        try:
            df = data.copy()

            # Define feature types (adjust based on your actual data)
            # Example: Assume 'status_code' is categorical
            potential_categorical_features = ['status_code', 'device_type', 'location_zone'] # Add others as needed
            all_input_cols = list(df.columns)

            # Identify numeric and categorical columns *present in the input data*
            numeric_features = df.select_dtypes(include=np.number).columns.tolist()
            categorical_features = [col for col in potential_categorical_features if col in df.columns]

            # Remove target column if present
            target_col = self.scoring_model_metadata.get('target_column')
            if target_col and target_col in numeric_features:
                numeric_features.remove(target_col)
            if target_col and target_col in categorical_features:
                categorical_features.remove(target_col)

            # Columns used for feature generation (not directly by model pipeline)
            excluded_cols = ['timestamp', 'device_id', 'device_name', 'id'] # Add others like 'unit'
            if target_col:
                excluded_cols.append(target_col)

            # Filter features for the pipeline
            numeric_features = [f for f in numeric_features if f not in excluded_cols]
            categorical_features = [f for f in categorical_features if f not in excluded_cols]

            self.logger.info(f"Preparing data. Numeric: {numeric_features}, Categorical: {categorical_features}")

            if not numeric_features and not categorical_features:
                self.logger.error("No numeric or categorical features identified for preprocessing.")
                return None

            # Define preprocessing steps
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')), # Handle missing categories
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Ignore unknown categories during prediction
            ])

            # Combine transformers using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ],
                remainder='drop' # Drop columns not specified
            )

            if fit_pipeline:
                self.logger.info("Fitting preprocessing pipeline...")
                self.preprocessing_pipeline = preprocessor.fit(df)
                # Store feature names after fitting
                feature_names = self.preprocessing_pipeline.get_feature_names_out()
                self.scoring_model_metadata['pipeline_features'] = feature_names.tolist()
                self.scoring_model_metadata['numeric_features_used'] = numeric_features
                self.scoring_model_metadata['categorical_features_used'] = categorical_features
                self.logger.info(f"Pipeline fitted. Output features: {len(feature_names)}")
            elif self.preprocessing_pipeline is None:
                self.logger.error("Preprocessing pipeline has not been fitted or loaded. Cannot transform data.")
                return None

            # Transform the data
            self.logger.info("Transforming data using pipeline...")
            features_processed = self.preprocessing_pipeline.transform(df)

            # Get feature names from the fitted pipeline
            output_feature_names = self.scoring_model_metadata.get('pipeline_features')
            if not output_feature_names:
                self.logger.error("Could not get feature names from fitted pipeline.")
                return None


            # Create DataFrame with correct column names
            features_df = pd.DataFrame(features_processed, columns=output_feature_names, index=df.index)

            self.logger.info(f"Data prepared: {features_df.shape[0]} rows, {features_df.shape[1]} features")
            return features_df

        except Exception as e:
            self.logger.error(f"Data preparation error: {e}", exc_info=True)
            return None
    # --- END REFACTOR ---


    def train_health_scorer(self, data: pd.DataFrame, failure_labels: pd.Series) -> Dict:
        """
        Trains an ML model (RandomForestRegressor) to predict health score (1 - failure_prob).
        Now uses the refactored prepare_data.

        Args:
            data: DataFrame of historical sensor/feature data. Needs columns used by components
                  and predictive features, plus any raw features for the pipeline (e.g., 'status_code').
                  Should include 'timestamp'.
            failure_labels: Series of 0s (healthy) and 1s (failure) corresponding to the data.

        Returns:
            Dictionary with training results.
        """
        if not self.analytics_engine:
            msg = "AnalyticsEngine not available. Cannot train model-based scorer."
            self.logger.error(msg)
            return {'error': msg}

        self.logger.info(f"Starting health scorer training with {len(data)} samples...")

        if 'timestamp' not in data.columns:
            msg = "Timestamp column missing in input data for training."
            self.logger.error(msg)
            return {'error': msg}
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            try:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            except Exception as e:
                msg = f"Could not convert timestamp column to datetime: {e}"
                self.logger.error(msg)
                return {'error': msg}

        # Sort data by timestamp for rolling operations
        data = data.sort_values('timestamp').reset_index(drop=True)
        failure_labels = failure_labels.loc[data.index] # Ensure labels match sorted data

        health_labels = 1 - failure_labels
        window_size = 50 # Window size for component calculations

        if len(data) <= window_size:
            msg = f"Not enough data ({len(data)}) for window size ({window_size})."
            self.logger.error(msg)
            return {'error': msg}

        # --- Feature Engineering (Component Scores + Predictive) ---
        features_df = pd.DataFrame(index=data.index[window_size:])
        numeric_cols_for_components = data.select_dtypes(include=[np.number]).columns

        self.logger.info("Calculating component scores using rolling windows...")
        # (Component score calculations remain the same)
        rolling_window = data.rolling(window=window_size)
        features_df['performance'] = rolling_window[numeric_cols_for_components].apply(
            lambda x: self._calculate_performance_score(pd.DataFrame(x, columns=numeric_cols_for_components))['score'], raw=False
        )[window_size:]
        features_df['reliability'] = rolling_window[list(data.columns)].apply(
             lambda x: self._calculate_reliability_score(
                 pd.DataFrame(x[:, data.columns.get_indexer(numeric_cols_for_components)], columns=numeric_cols_for_components),
                 pd.DataFrame(x, columns=data.columns)
             )['score'], raw=True
         )[window_size:]
        features_df['efficiency'] = rolling_window[numeric_cols_for_components].apply(
             lambda x: self._calculate_efficiency_score(pd.DataFrame(x, columns=numeric_cols_for_components))['score'], raw=False
         )[window_size:]
        features_df['safety'] = rolling_window[numeric_cols_for_components].apply(
             lambda x: self._calculate_safety_score(pd.DataFrame(x, columns=numeric_cols_for_components))['score'], raw=False
         )[window_size:]
        features_df['maintenance'] = rolling_window[list(data.columns)].apply(
             lambda x: self._calculate_maintenance_score(
                 pd.DataFrame(x[:, data.columns.get_indexer(numeric_cols_for_components)], columns=numeric_cols_for_components),
                 pd.DataFrame(x, columns=data.columns)
             )['score'], raw=True
         )[window_size:]


        self.logger.info("Calculating predictive features...")
        # (Predictive feature calculations remain the same)
        predictive_features = data[window_size:].apply(
             lambda row: pd.Series({
                 'anomaly_score': self.analytics_engine.detect_anomalies(
                                         row.to_frame().T, model_name="anomaly_detector"
                                     ).get('anomaly_scores', [0.0])[0],
                 'failure_prob': self.analytics_engine.predict_failure(
                                         row.to_frame().T
                                     ).get('failure_probabilities', [0.0])[0]
             }),
             axis=1
         )
        features_df = pd.concat([features_df, predictive_features.reset_index(drop=True)], axis=1)

        # --- NEW: Add raw features needed by the pipeline ---
        # Include raw categorical/numeric features from the original data that the pipeline needs
        raw_features_to_include = self.scoring_model_metadata.get('numeric_features_used', []) + \
                                  self.scoring_model_metadata.get('categorical_features_used', [])
        # Ensure we only try to add columns that actually exist in the original `data`
        raw_features_present = [f for f in raw_features_to_include if f in data.columns]
        if raw_features_present:
            features_df = pd.concat([features_df, data.loc[window_size:, raw_features_present].reset_index(drop=True)], axis=1)


        # Fill NaNs before potentially passing to pipeline (safer)
        features_df = features_df.fillna(0.0) # Or use a more sophisticated method

        y = health_labels.iloc[window_size:] # Align labels with features

        # --- Fit preprocessing pipeline AND transform ---
        X_processed = self.prepare_data(features_df, fit_pipeline=True)
        if X_processed is None:
            return {'error': "Data preparation failed during training."}

        # Save feature names *after* pipeline transformation
        self.scoring_model_metadata['model_input_features'] = list(X_processed.columns)
        self.scoring_model_metadata['target_column'] = 'health_label' # Store target name


        # --- Model Training ---
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

        self.logger.info(f"Training RandomForestRegressor with {len(X_train)} samples...")
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_split=5)
        model.fit(X_train, y_train)

        self.scoring_model = model

        # --- Evaluation ---
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = np.mean(np.abs(y_pred - y_test)) # Mean Absolute Error

        # Get feature importances based on processed feature names
        importances = dict(zip(X_processed.columns, model.feature_importances_))

        self.scoring_model_metadata['training_mse'] = mse
        self.scoring_model_metadata['training_mae'] = mae
        self.scoring_model_metadata['feature_importances'] = importances
        self.scoring_model_metadata['trained_at'] = datetime.now().isoformat()

        # Save the trained model AND pipeline
        self._save_scoring_model_and_pipeline()

        self.logger.info(f"Health scorer training complete. Test MSE: {mse:.4f}, MAE: {mae:.4f}")

        return {
            'status': 'success',
            'test_mse': mse,
            'test_mae': mae,
            'feature_importances': {k: round(v, 4) for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=True)},
            'pipeline_features': self.scoring_model_metadata.get('pipeline_features', []),
            'model_input_features': list(X_processed.columns)
        }


    def calculate_overall_health_score(self, data: pd.DataFrame,
                                           device_id: str = None,
                                           timestamp: datetime = None,
                                           use_model: bool = True) -> Dict:
        """
        Calculate overall health score for a system or device (HYBRID score).
        Includes confidence scoring for model predictions.

        Args:
            data: DataFrame with *new* sensor/system data (can be single row for real-time).
                  Must contain columns needed for component scores and raw features for the pipeline.
            device_id: Optional device identifier.
            timestamp: Optional timestamp for the calculation.
            use_model: (bool) Whether to attempt using the ML model.

        Returns:
            Dictionary containing health score, components, confidence, etc.
        """
        try:
            self.logger.info(f"Calculating health score for device: {device_id or 'system'}")

            if timestamp is None:
                timestamp = datetime.now(timezone.utc) # Use timezone-aware

            # --- History-Aware Calculation (remains the same) ---
            historical_data_list = []
            if device_id and device_id in self.component_history:
                 for entry in self.component_history[device_id]:
                     if 'data' in entry:
                          historical_data_list.append(entry['data'])
            if historical_data_list:
                 historical_df = pd.concat(historical_data_list, ignore_index=True)
                 combined_data = pd.concat([historical_df, data], ignore_index=True).drop_duplicates()
            else:
                 combined_data = data
            numeric_data_history = combined_data.select_dtypes(include=[np.number])
            numeric_data_latest = data.select_dtypes(include=[np.number])
            if numeric_data_latest.empty: return self._create_error_result("No numeric data available")
            if numeric_data_history.empty: numeric_data_history = numeric_data_latest
            # --- End History ---

            # --- 1. Calculate Component Scores (using historical data) ---
            # (Calculations remain the same)
            component_scores = {}
            component_scores['performance'] = self._calculate_performance_score(numeric_data_history)
            component_scores['reliability'] = self._calculate_reliability_score(numeric_data_history, combined_data)
            component_scores['efficiency'] = self._calculate_efficiency_score(numeric_data_history)
            component_scores['safety'] = self._calculate_safety_score(numeric_data_history)
            component_scores['maintenance'] = self._calculate_maintenance_score(numeric_data_history, combined_data)


            # --- 2. Calculate Formula-Based Score ---
            # (Calculation remains the same)
            score_weights = self.config.get('component_weights', {})
            formula_based_score = 0.0
            total_weight = 0.0
            for component, score_data in component_scores.items():
                 if isinstance(score_data, dict) and 'score' in score_data:
                     weight = score_weights.get(component, 0.2)
                     formula_based_score += score_data['score'] * weight
                     total_weight += weight
            formula_based_score = formula_based_score / total_weight if total_weight > 0 else np.mean([cs['score'] for cs in component_scores.values()])


            # --- 3. Calculate Model-Based Score (using latest data + pipeline) ---
            model_based_score = None
            prediction_confidence = None # <-- NEW
            calculation_method = "formula_based"
            model_inputs_raw = {}

            if use_model and self.scoring_model and self.preprocessing_pipeline and self.analytics_engine:
                try:
                    # 3a. Prepare raw inputs for the pipeline
                    model_inputs_raw['performance'] = component_scores['performance']['score']
                    model_inputs_raw['reliability'] = component_scores['reliability']['score']
                    model_inputs_raw['efficiency'] = component_scores['efficiency']['score']
                    model_inputs_raw['safety'] = component_scores['safety']['score']
                    model_inputs_raw['maintenance'] = component_scores['maintenance']['score']
                    anomaly_res = self.analytics_engine.detect_anomalies(data, model_name="anomaly_detector")
                    fail_res = self.analytics_engine.predict_failure(data)
                    model_inputs_raw['anomaly_score'] = anomaly_res.get('anomaly_scores', [0.0])[0]
                    model_inputs_raw['failure_prob'] = fail_res.get('failure_probabilities', [0.0])[0]

                    # Add raw features needed by the pipeline from the *latest* data row
                    raw_features_needed = self.scoring_model_metadata.get('numeric_features_used', []) + \
                                          self.scoring_model_metadata.get('categorical_features_used', [])
                    for feature in raw_features_needed:
                        if feature in data.columns:
                            # Ensure we handle potential multi-row input 'data' correctly
                            model_inputs_raw[feature] = data[feature].iloc[-1] # Take the latest value


                    # Convert raw inputs to DataFrame for the pipeline
                    input_df_raw = pd.DataFrame([model_inputs_raw])

                    # 3b. Transform using the *fitted* pipeline
                    input_df_processed = self.prepare_data(input_df_raw, fit_pipeline=False)

                    if input_df_processed is None:
                        raise ValueError("Failed to preprocess input data for model prediction.")

                    # Ensure columns match training order/presence
                    required_model_features = self.scoring_model_metadata.get('model_input_features', [])
                    input_vector_df = input_df_processed.reindex(columns=required_model_features, fill_value=0)


                    # 3c. Predict
                    input_vector = input_vector_df.iloc[0].values.reshape(1, -1)
                    prediction = self.scoring_model.predict(input_vector)[0]
                    model_based_score = max(0.0, min(1.0, prediction)) # Clamp prediction

                    # --- NEW: Calculate Prediction Confidence ---
                    # For RandomForestRegressor, use std dev of tree predictions
                    if isinstance(self.scoring_model, RandomForestRegressor):
                        tree_predictions = np.array([tree.predict(input_vector) for tree in self.scoring_model.estimators_])
                        pred_std_dev = np.std(tree_predictions)
                        # Convert std dev to a confidence score (0-1), lower std dev = higher confidence
                        # This is a heuristic, adjust the scaling (e.g., the '2') as needed
                        prediction_confidence = max(0.0, 1.0 - 2 * pred_std_dev) # Example scaling
                        prediction_confidence = round(prediction_confidence, 3)

                    calculation_method = "hybrid"

                except Exception as e:
                    self.logger.warning(f"Model-based scoring failed, falling back to formula. Error: {e}", exc_info=True)
                    model_based_score = None
                    prediction_confidence = None
                    calculation_method = "formula_fallback"

            # --- 4. Blend Scores ---
            # (Blending remains the same)
            if model_based_score is not None:
                 blend_weights = self.config.get('model_blending', {'formula_weight': 0.5, 'model_weight': 0.5})
                 fw = blend_weights['formula_weight']
                 mw = blend_weights['model_weight']
                 overall_score = (formula_based_score * fw) + (model_based_score * mw)
            else:
                 overall_score = formula_based_score


            # --- 5. Finalize Result ---
            # (Finalization remains largely the same, add confidence)
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
                'prediction_confidence': prediction_confidence, # <-- NEW
                'component_scores': component_scores,
                'model_inputs_raw': model_inputs_raw if calculation_method != "formula_based" else {},
                'trend_analysis': trend_analysis,
                'risk_assessment': risk_assessment,
                'recommendations': self._generate_health_recommendations(component_scores, overall_score),
                'data_quality': self._assess_data_quality(data),
                'historical_data_points': len(combined_data)
            }

            self._store_health_score(result, data.iloc[-1:].copy()) # Store only the latest data point

            self.logger.info(
                 f"Health score: {overall_score:.3f} ({health_status}) | Method: {calculation_method} | Conf: {prediction_confidence}"
             )

            return result

        except Exception as e:
            self.logger.error(f"Overall health score calculation error: {e}", exc_info=True)
            return self._create_error_result(str(e))

    # --- Component Score Calculations (_calculate_performance_score, etc.) ---
    # (These remain the same as provided in your original file)
    # ...
    def _calculate_performance_score(self, data: pd.DataFrame) -> Dict:
        """Calculate performance-based health score."""
        # (Implementation remains the same)
        try:
            performance_metrics = {}
            performance_cols = [col for col in data.columns if any(
                keyword in col.lower() for keyword in
                ['efficiency', 'throughput', 'output', 'performance', 'speed', 'rate']
            )]
            scores = []
            if not performance_cols:
                self.logger.warning("No performance columns found. Returning neutral score.")
                return {'score': 0.7, 'metrics': {}, 'summary': self._summarize_score_component(0.7, 'performance')}

            for col in performance_cols:
                col_data = data[col].dropna()
                if len(col_data) == 0: continue
                is_better = 'consumption' not in col.lower() and 'energy' not in col.lower()
                col_score = self._calculate_metric_score(col_data, col, 'performance', higher_is_better=is_better)
                performance_metrics[col] = {
                    'score': col_score, 'mean': float(col_data.mean()),
                    'std': float(col_data.std()), 'trend': self._calculate_simple_trend(col_data)
                }
                scores.append(col_score)
            overall_performance = np.mean(scores) if scores else 0.7
            return {'score': round(overall_performance, 3), 'metrics': performance_metrics, 'summary': self._summarize_score_component(overall_performance, 'performance')}
        except Exception as e:
            self.logger.error(f"Performance score calculation error: {e}")
            return {'score': 0.5, 'error': str(e)}

    def _calculate_reliability_score(self, numeric_data: pd.DataFrame, full_data: pd.DataFrame) -> Dict:
        """Calculate reliability-based health score."""
        # (Implementation remains the same)
        try:
            reliability_metrics = {}; scores = []
            consistency_scores = []
            for col in numeric_data.columns:
                col_data = numeric_data[col].dropna()
                if len(col_data) > 1:
                    cv = col_data.std() / (col_data.mean() + 1e-8)
                    consistency_score = 1.0 / (1.0 + abs(cv))
                    consistency_scores.append(consistency_score)
                    reliability_metrics[f'{col}_consistency'] = {'coefficient_of_variation': float(cv), 'consistency_score': round(consistency_score, 3)}
            if consistency_scores: scores.append(np.mean(consistency_scores))
            if 'uptime' in full_data.columns: uptime_score = full_data['uptime'].mean()
            elif 'timestamp' in full_data.columns: uptime_score = self._calculate_uptime_score(full_data)
            else: uptime_score = 0.8
            scores.append(uptime_score); reliability_metrics['uptime'] = {'score': uptime_score}
            anomaly_score = self._calculate_anomaly_based_reliability(numeric_data)
            scores.append(anomaly_score); reliability_metrics['anomaly_based'] = {'score': anomaly_score}
            overall_reliability = np.mean(scores) if scores else 0.5
            return {'score': round(overall_reliability, 3), 'metrics': reliability_metrics, 'summary': self._summarize_score_component(overall_reliability, 'reliability')}
        except Exception as e:
            self.logger.error(f"Reliability score calculation error: {e}")
            return {'score': 0.5, 'error': str(e)}

    def _calculate_efficiency_score(self, data: pd.DataFrame) -> Dict:
        """Calculate efficiency-based health score."""
        # (Implementation remains the same)
        try:
            efficiency_metrics = {}
            efficiency_cols = [col for col in data.columns if any(keyword in col.lower() for keyword in ['efficiency', 'energy', 'power', 'consumption', 'utilization'])]
            scores = []
            for col in efficiency_cols:
                col_data = data[col].dropna()
                if len(col_data) == 0: continue
                higher_is_better = 'consumption' not in col.lower() and 'energy' not in col.lower()
                efficiency_score = self._calculate_metric_score(col_data, col, 'efficiency', higher_is_better)
                efficiency_metrics[col] = {'score': round(efficiency_score, 3), 'mean_value': float(col_data.mean()), 'trend': self._calculate_simple_trend(col_data)}
                scores.append(efficiency_score)
            if not scores and not data.empty:
                all_cvs = [data[col].std() / (data[col].mean() + 1e-8) for col in data.columns if data[col].std() > 0]
                overall_cv = np.mean(all_cvs) if all_cvs else 0
                efficiency_score = 1.0 / (1.0 + abs(overall_cv)); scores.append(efficiency_score)
                efficiency_metrics['general_stability'] = {'score': round(efficiency_score, 3), 'coefficient_of_variation': float(overall_cv)}
            overall_efficiency = np.mean(scores) if scores else 0.5
            return {'score': round(overall_efficiency, 3), 'metrics': efficiency_metrics, 'summary': self._summarize_score_component(overall_efficiency, 'efficiency')}
        except Exception as e:
            self.logger.error(f"Efficiency score calculation error: {e}")
            return {'score': 0.5, 'error': str(e)}

    def _calculate_safety_score(self, data: pd.DataFrame) -> Dict:
        """Calculate safety-based health score."""
        # (Implementation remains the same)
        try:
            safety_metrics = {}
            safety_cols = [col for col in data.columns if any(keyword in col.lower() for keyword in ['temperature', 'pressure', 'vibration', 'safety', 'alarm', 'warning'])]
            scores = []
            for col in safety_cols:
                col_data = data[col].dropna();
                if len(col_data) == 0: continue
                safety_score = self._calculate_safety_range_score(col_data, col)
                safety_metrics[col] = {'score': safety_score, 'mean_value': float(col_data.mean()), 'max_value': float(col_data.max()), 'min_value': float(col_data.min()), 'outlier_percentage': self._calculate_outlier_percentage(col_data)}
                scores.append(safety_score)
            if not scores and not data.empty:
                general_safety = self._calculate_general_safety_score(data); scores.append(general_safety)
                safety_metrics['general_safety'] = {'score': general_safety}
            overall_safety = np.mean(scores) if scores else 0.8
            return {'score': round(overall_safety, 3), 'metrics': safety_metrics, 'summary': self._summarize_score_component(overall_safety, 'safety')}
        except Exception as e:
            self.logger.error(f"Safety score calculation error: {e}")
            return {'score': 0.8, 'error': str(e)}

    def _calculate_maintenance_score(self, numeric_data: pd.DataFrame, full_data: pd.DataFrame) -> Dict:
        """Calculate maintenance-based health score."""
        # (Implementation remains the same)
        try:
            maintenance_metrics = {}; scores = []
            degradation_cols = [col for col in numeric_data.columns if any(keyword in col.lower() for keyword in ['vibration', 'noise', 'wear', 'degradation', 'fault'])]
            for col in degradation_cols:
                col_data = numeric_data[col].dropna()
                if len(col_data) > 5:
                    trend = self._calculate_simple_trend(col_data)
                    trend_score = max(0.0, min(1.0, 1.0 - trend * 5)); scores.append(trend_score)
                    maintenance_metrics[f'{col}_degradation'] = {'trend_score': round(trend_score, 3), 'trend_slope': trend, 'current_level': float(col_data.iloc[-1] if len(col_data) > 0 else 0)}
            if 'maintenance_date' in full_data.columns or 'last_maintenance' in full_data.columns:
                schedule_score = self._calculate_maintenance_schedule_score(full_data); scores.append(schedule_score)
                maintenance_metrics['schedule_adherence'] = {'score': schedule_score}
            performance_degradation = self._calculate_performance_degradation(numeric_data); scores.append(performance_degradation)
            maintenance_metrics['performance_degradation'] = {'score': performance_degradation}
            overall_maintenance = np.mean(scores) if scores else 0.7
            return {'score': round(overall_maintenance, 3), 'metrics': maintenance_metrics, 'summary': self._summarize_score_component(overall_maintenance, 'maintenance')}
        except Exception as e:
            self.logger.error(f"Maintenance score calculation error: {e}")
            return {'score': 0.7, 'error': str(e)}

    # --- Other Sub-Calculators & Helpers ---
    # (_calculate_metric_score, _calculate_simple_trend, etc.)
    # (These remain the same as provided in your original file)
    # ...
    def _calculate_metric_score(self, data: pd.Series, metric_name: str, score_type: str,
                                higher_is_better: bool = True) -> float:
        """Calculate score based on deviation from adaptive EMA baseline."""
        # (Implementation remains the same)
        try:
            baseline_key = f"{metric_name}_{score_type}"
            current_mean = data.mean(); current_std = data.std()
            if baseline_key in self.baseline_emas:
                baseline = self.baseline_emas[baseline_key]; baseline_mean = baseline['mean']; baseline_std = baseline['std']
                alpha = self.baseline_ema_alpha
                new_ema_mean = (alpha * current_mean) + ((1 - alpha) * baseline_mean)
                new_ema_std = (alpha * current_std) + ((1 - alpha) * baseline_std)
                self.baseline_emas[baseline_key] = {'mean': float(new_ema_mean), 'std': float(new_ema_std)}
            else:
                self.baseline_emas[baseline_key] = {'mean': float(current_mean), 'std': float(current_std)}
                baseline_mean = current_mean; baseline_std = current_std
            safe_std = baseline_std + 1e-8
            deviation = (current_mean - baseline_mean) / safe_std
            score = np.exp(-0.5 * (min(0, deviation) ** 2)) if higher_is_better else np.exp(-0.5 * (max(0, deviation) ** 2))
            return max(0.0, min(1.0, score))
        except Exception as e:
            self.logger.error(f"Metric score calculation error: {e}")
            return 0.5

    def _calculate_simple_trend(self, data: pd.Series) -> float:
        """Calculate simple linear trend slope, normalized by range."""
        # (Implementation remains the same)
        try:
            if len(data) < 3: return 0.0
            x = np.arange(len(data)); valid_indices = ~data.isnull()
            if valid_indices.sum() < 3: return 0.0
            x_valid = x[valid_indices]; y_valid = data.values[valid_indices]
            slope, _, _, _, _ = stats.linregress(x_valid, y_valid)
            data_range = y_valid.max() - y_valid.min()
            normalized_slope = slope / data_range if data_range > 0 else 0.0
            return float(normalized_slope)
        except Exception as e:
            self.logger.error(f"Trend calculation error: {e}")
            return 0.0

    def _calculate_uptime_score(self, data: pd.DataFrame) -> float:
        """Calculate uptime score based on timestamp gaps."""
        # (Implementation remains the same)
        try:
            if 'timestamp' not in data.columns: return 0.8
            timestamps = pd.to_datetime(data['timestamp']).dropna().sort_values()
            if len(timestamps) < 2: return 0.8
            time_diffs = timestamps.diff().dropna();
            if time_diffs.empty: return 0.8
            expected_interval = time_diffs.median()
            downtime_gaps = time_diffs[time_diffs > 3 * expected_interval]
            total_downtime = downtime_gaps.sum(); total_time = timestamps.max() - timestamps.min()
            if total_time.total_seconds() > 0:
                uptime_ratio = 1.0 - (total_downtime.total_seconds() / total_time.total_seconds())
                return max(0.0, min(1.0, uptime_ratio))
            else: return 0.8
        except Exception as e:
            self.logger.error(f"Uptime score calculation error: {e}")
            return 0.8

    def _calculate_anomaly_based_reliability(self, data: pd.DataFrame) -> float:
        """Calculate reliability based on local Isolation Forest anomaly detection."""
        # (Implementation remains the same)
        try:
            if len(data) < 10: return 0.8
            data_filled = data.fillna(data.mean());
            if data_filled.empty: return 0.8
            scaler = StandardScaler(); data_scaled = scaler.fit_transform(data_filled)
            iso_forest = IsolationForest(contamination=self.config.get('anomaly_detection', {}).get('contamination', 0.1), random_state=42)
            anomaly_predictions = iso_forest.fit_predict(data_scaled)
            anomaly_rate = (anomaly_predictions == -1).mean()
            reliability_score = 1.0 - anomaly_rate
            return max(0.0, min(1.0, reliability_score))
        except Exception as e:
            self.logger.error(f"Anomaly-based reliability calculation error: {e}")
            return 0.7

    def _calculate_safety_range_score(self, data: pd.Series, column_name: str) -> float:
        """Calculate safety score based on configured operating ranges."""
        # (Implementation remains the same)
        try:
            safety_ranges = self.config.get('safety_ranges', {})
            applicable_range = None
            for param, ranges in safety_ranges.items():
                if param in column_name.lower(): applicable_range = ranges; break
            if applicable_range is None:
                q25, q75 = data.quantile([0.25, 0.75]); iqr = q75 - q25 # Corrected IQR calculation
                if iqr < 1e-6: iqr = data.std() + 1e-6
                if iqr < 1e-6: return 0.8
                applicable_range = {'min': q25 - 1.5 * iqr, 'max': q75 + 1.5 * iqr, 'optimal_min': q25, 'optimal_max': q75}
            in_safe_range_pct = ((data >= applicable_range['min']) & (data <= applicable_range['max'])).mean()
            in_optimal_range_pct = ((data >= applicable_range['optimal_min']) & (data <= applicable_range['optimal_max'])).mean()
            safety_score = 0.5 * in_safe_range_pct + 0.5 * in_optimal_range_pct
            return max(0.0, min(1.0, safety_score))
        except Exception as e:
            self.logger.error(f"Safety range score calculation error: {e}")
            return 0.8

    def _calculate_outlier_percentage(self, data: pd.Series) -> float:
        """Calculate percentage of outliers using IQR method."""
        # (Implementation remains the same)
        try:
            Q1 = data.quantile(0.25); Q3 = data.quantile(0.75); IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR; upper_bound = Q3 + 1.5 * IQR
            outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            total_points = len(data)
            return float(outliers / total_points) if total_points > 0 else 0.0
        except Exception as e:
            self.logger.error(f"Outlier percentage calculation error: {e}")
            return 0.0

    def _calculate_general_safety_score(self, data: pd.DataFrame) -> float:
        """Calculate general safety score using local Isolation Forest."""
        # (Implementation remains the same)
        try:
            if len(data) < 5: return 0.8
            data_filled = data.fillna(data.mean());
            if data_filled.empty: return 0.8
            scaler = StandardScaler(); data_scaled = scaler.fit_transform(data_filled)
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            anomalies = iso_forest.fit_predict(data_scaled)
            anomaly_rate = (anomalies == -1).mean(); safety_score = 1.0 - 2 * anomaly_rate
            return max(0.0, min(1.0, safety_score))
        except Exception as e:
            self.logger.error(f"General safety score calculation error: {e}")
            return 0.8

    def _calculate_maintenance_schedule_score(self, data: pd.DataFrame) -> float:
        """Placeholder for maintenance schedule adherence calculation."""
        # (Implementation remains the same)
        return 0.8 # Placeholder implementation

    def _calculate_performance_degradation(self, data: pd.DataFrame) -> float:
        """Calculate performance degradation based on trend of performance metrics."""
        # (Implementation remains the same)
        try:
            if len(data) < 10: return 0.8
            degradation_scores = []
            perf_cols = [col for col in data.columns if any(keyword in col.lower() for keyword in ['efficiency', 'throughput', 'output', 'performance'])]
            for col in perf_cols:
                col_data = data[col].dropna()
                if len(col_data) < 5: continue
                trend = self._calculate_simple_trend(col_data)
                score = max(0.0, min(1.0, 1.0 + trend * 5))
                degradation_scores.append(score)
            return np.mean(degradation_scores) if degradation_scores else 0.8
        except Exception as e:
            self.logger.error(f"Performance degradation calculation error: {e}")
            return 0.8

    def _determine_health_status(self, score: float) -> str:
        """Determine health status string based on score."""
        # (Implementation remains the same)
        thresholds = self.health_thresholds
        if score >= thresholds['excellent']: return 'excellent'
        elif score >= thresholds['good']: return 'good'
        elif score >= thresholds['warning']: return 'warning'
        elif score >= thresholds['critical']: return 'critical'
        else: return 'failure'

    def _calculate_health_trend(self, device_id: str, current_score: float) -> Dict:
        """Calculate health trend analysis based on history."""
        # (Implementation remains the same)
        try:
            history = list(self.component_history[device_id]) if device_id else list(self.health_history)
            if len(history) < 3: return {'trend_direction': 'stable', 'trend_strength': 0.0, 'data_points': len(history)}
            scores = [entry.get('overall_score', 0.5) for entry in history[-20:]] + [current_score]
            x = np.arange(len(scores)); slope, _, r_value, _, _ = stats.linregress(x, scores)
            trend_threshold = self.config.get('trend_analysis', {}).get('trend_threshold', 0.01)
            if slope > trend_threshold: trend_direction = 'improving'
            elif slope < -trend_threshold: trend_direction = 'degrading'
            else: trend_direction = 'stable'
            return {'trend_direction': trend_direction, 'trend_strength': abs(float(slope)), 'trend_r_squared': float(r_value**2), 'data_points': len(scores), 'recent_scores': scores[-5:], 'prediction_next_period': float(current_score + slope) if slope != 0 else current_score}
        except Exception as e:
            self.logger.error(f"Health trend calculation error: {e}")
            return {'trend_direction': 'stable', 'trend_strength': 0.0}

    def _calculate_risk_assessment(self, component_scores: Dict, overall_score: float) -> Dict:
        """Calculate risk assessment based on scores."""
        # (Implementation remains the same)
        try:
            risk_factors = []
            for component, score_data in component_scores.items():
                if isinstance(score_data, dict) and 'score' in score_data:
                    score = score_data['score']
                    if score < self.health_thresholds['critical']: risk_factors.append({'component': component, 'risk_level': 'high', 'score': score, 'impact': 'critical_component_failure'})
                    elif score < self.health_thresholds['warning']: risk_factors.append({'component': component, 'risk_level': 'medium', 'score': score, 'impact': 'performance_degradation'})
            if overall_score < self.health_thresholds['critical']: overall_risk = 'high'
            elif overall_score < self.health_thresholds['warning']: overall_risk = 'medium'
            elif overall_score < self.health_thresholds['good']: overall_risk = 'low'
            else: overall_risk = 'minimal'
            failure_probability = self._estimate_failure_probability(overall_score, component_scores)
            time_to_failure = self._estimate_time_to_failure(overall_score, component_scores)
            return {'overall_risk_level': overall_risk, 'risk_factors': risk_factors, 'failure_probability': failure_probability, 'estimated_time_to_failure': time_to_failure, 'risk_score': round(1.0 - overall_score, 3), 'mitigation_priority': self._determine_mitigation_priority(risk_factors, overall_risk)}
        except Exception as e:
            self.logger.error(f"Risk assessment calculation error: {e}")
            return {'overall_risk_level': 'unknown', 'risk_factors': []}

    def _estimate_failure_probability(self, overall_score: float, component_scores: Dict) -> Dict:
        """Estimate failure probability based on scores."""
        # (Implementation remains the same)
        try:
            base_probability = 1.0 - overall_score; critical_components = 0
            for component, score_data in component_scores.items():
                if isinstance(score_data, dict) and 'score' in score_data and score_data['score'] < self.health_thresholds['critical']: critical_components += 1
            adjusted_probability = min(1.0, base_probability * (1.0 + 0.5 * critical_components))
            probabilities = {'next_24_hours': adjusted_probability * 0.1, 'next_week': adjusted_probability * 0.3, 'next_month': adjusted_probability * 0.6, 'next_quarter': adjusted_probability * 0.9, 'overall': adjusted_probability}
            return {'probabilities': probabilities, 'confidence': 0.7, 'model_type': 'health_score_based'}
        except Exception as e:
            self.logger.error(f"Failure probability estimation error: {e}")
            return {'probabilities': {}, 'confidence': 0.0}

    def _estimate_time_to_failure(self, overall_score: float, component_scores: Dict) -> Dict:
        """Estimate time to failure (simple linear model)."""
        # (Implementation remains the same)
        try:
            if overall_score > self.health_thresholds['good']: return {'estimated_days': None, 'confidence': 0.0, 'status': 'healthy_no_prediction'}
            degradation_rate = 1.0 - overall_score
            if degradation_rate <= 0: return {'estimated_days': None, 'confidence': 0.0, 'status': 'no_degradation_detected'}
            days_to_critical = max(1, (overall_score - self.health_thresholds['critical']) / (degradation_rate * 0.01))
            days_to_failure = max(1, overall_score / (degradation_rate * 0.01))
            return {'estimated_days_to_critical': round(days_to_critical), 'estimated_days_to_failure': round(days_to_failure), 'confidence': 0.6, 'model_type': 'linear_degradation'}
        except Exception as e:
            self.logger.error(f"Time to failure estimation error: {e}")
            return {'estimated_days': None, 'confidence': 0.0}

    def _determine_mitigation_priority(self, risk_factors: List[Dict], overall_risk: str) -> str:
        """Determine mitigation priority based on risk."""
        # (Implementation remains the same)
        try:
            high_risk_factors = [rf for rf in risk_factors if rf.get('risk_level') == 'high']; medium_risk_factors = [rf for rf in risk_factors if rf.get('risk_level') == 'medium']
            if overall_risk == 'high' or len(high_risk_factors) >= 2: return 'immediate'
            elif overall_risk == 'medium' or len(high_risk_factors) >= 1: return 'urgent'
            elif len(medium_risk_factors) >= 2: return 'scheduled'
            else: return 'routine'
        except Exception as e:
            self.logger.error(f"Mitigation priority determination error: {e}")
            return 'routine'

    def _generate_health_recommendations(self, component_scores: Dict, overall_score: float) -> List[Dict]:
        """Generate health recommendations based on scores."""
        # (Implementation remains the same)
        try:
            recommendations = []
            for component, score_data in component_scores.items():
                if isinstance(score_data, dict) and 'score' in score_data:
                    score = score_data['score']
                    if score < self.health_thresholds['critical']: recommendations.append({'priority': 'critical', 'component': component, 'action': f'Immediate {component} inspection and maintenance required', 'expected_impact': 'Prevent system failure', 'timeframe': 'immediate'})
                    elif score < self.health_thresholds['warning']: recommendations.append({'priority': 'high', 'component': component, 'action': f'Schedule {component} maintenance within next week', 'expected_impact': 'Improve system reliability', 'timeframe': 'within_week'})
                    elif score < self.health_thresholds['good']: recommendations.append({'priority': 'medium', 'component': component, 'action': f'Monitor {component} closely and plan maintenance', 'expected_impact': 'Maintain optimal performance', 'timeframe': 'within_month'})
            if overall_score < self.health_thresholds['critical']: recommendations.append({'priority': 'critical', 'component': 'system', 'action': 'System-wide inspection and immediate corrective action', 'expected_impact': 'Prevent catastrophic failure', 'timeframe': 'immediate'})
            elif overall_score < self.health_thresholds['warning']: recommendations.append({'priority': 'high', 'component': 'system', 'action': 'Comprehensive system health check', 'expected_impact': 'Restore system health', 'timeframe': 'within_week'})
            priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}; recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))
            return recommendations[:10]
        except Exception as e:
            self.logger.error(f"Recommendation generation error: {e}")
            return []

    def _assess_data_quality(self, data: pd.DataFrame) -> Dict:
        """Assess quality of input data."""
        # (Implementation remains the same)
        try:
            total_cells = data.size;
            if total_cells == 0: return {'overall_quality_score': 0.0, 'quality_level': 'no_data'}
            missing_cells = data.isnull().sum().sum(); missing_percentage = (missing_cells / total_cells) * 100
            completeness_score = 1.0 - (missing_percentage / 100); freshness_score = 1.0
            if 'timestamp' in data.columns:
                try:
                    latest_timestamp = pd.to_datetime(data['timestamp']).max()
                    time_since_last = datetime.utcnow() - latest_timestamp.to_pydatetime().replace(tzinfo=None)
                    hours_since = time_since_last.total_seconds() / 3600
                    if hours_since <= 1: freshness_score = 1.0
                    elif hours_since <= 24: freshness_score = 0.8
                    elif hours_since <= 168: freshness_score = 0.6
                    else: freshness_score = 0.3
                except Exception: pass
            consistency_scores = []; numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                col_data = data[col].dropna()
                if len(col_data) > 1:
                    cv = col_data.std() / (col_data.mean() + 1e-8)
                    consistency_score = 1.0 / (1.0 + abs(cv)); consistency_scores.append(consistency_score)
            overall_consistency = np.mean(consistency_scores) if consistency_scores else 0.8
            quality_score = 0.4 * completeness_score + 0.3 * freshness_score + 0.3 * overall_consistency
            return {'overall_quality_score': round(quality_score, 3), 'completeness_score': round(completeness_score, 3), 'freshness_score': round(freshness_score, 3), 'consistency_score': round(overall_consistency, 3), 'missing_percentage': round(missing_percentage, 2), 'total_data_points': len(data), 'numeric_columns': len(numeric_cols), 'quality_level': 'high' if quality_score > 0.8 else 'medium' if quality_score > 0.6 else 'low'}
        except Exception as e:
            self.logger.error(f"Data quality assessment error: {e}")
            return {'overall_quality_score': 0.5, 'quality_level': 'unknown'}

    def _summarize_score_component(self, score: float, component: str) -> Dict:
        """Summarize a score component."""
        # (Implementation remains the same)
        status = self._determine_health_status(score)
        summary_messages = { 'performance': {'excellent': 'Perf exceptional', 'good': 'Perf good', 'warning': 'Perf degrading', 'critical': 'Perf critical', 'failure': 'Perf failure'}, 'reliability': {'excellent': 'Reliability excellent', 'good': 'Reliability good', 'warning': 'Reliability concerning', 'critical': 'Reliability critical', 'failure': 'Reliability failure risk'}, 'efficiency': {'excellent': 'Efficiency optimal', 'good': 'Efficiency good', 'warning': 'Efficiency improvement needed', 'critical': 'Efficiency critical', 'failure': 'Efficiency failure'}, 'safety': {'excellent': 'Safety optimal', 'good': 'Safety good', 'warning': 'Safety warning', 'critical': 'Safety critical', 'failure': 'Safety dangerous'}, 'maintenance': {'excellent': 'Maint minimal', 'good': 'Maint up-to-date', 'warning': 'Maint needed', 'critical': 'Maint critical', 'failure': 'Maint immediate'} }
        message = summary_messages.get(component, {}).get(status, f'{component} status {status}')
        return {'status': status, 'message': message, 'score_range': f"{score:.3f}", 'improvement_potential': round((1.0 - score) * 100, 1) if score < 1.0 else 0}

    def _store_health_score(self, result: Dict, data: pd.DataFrame):
        """Store health score and related data in history."""
        # (Implementation remains the same)
        try:
            self.health_history.append({'timestamp': result['timestamp'], 'overall_score': result['overall_score'], 'health_status': result['health_status'], 'device_id': result.get('device_id')})
            if result.get('device_id'):
                device_id = result['device_id']
                history_entry = {'timestamp': result['timestamp'], 'overall_score': result['overall_score'], 'component_scores': result['component_scores'], 'health_status': result['health_status'], 'data': data}
                self.component_history[device_id].append(history_entry)
        except Exception as e:
            self.logger.error(f"Health score storage error: {e}")

    def _create_error_result(self, error_message: str) -> Dict:
        """Create a standardized error result dictionary."""
        # (Implementation remains the same)
        return {'error': True, 'message': error_message, 'timestamp': datetime.now().isoformat(), 'overall_score': 0.0, 'health_status': 'unknown'}

    # --- Fleet & Other Methods ---
    # (get_health_history, calculate_fleet_health_score, etc.)
    # (These remain the same as provided in your original file)
    # ...


# Example usage and testing (can be updated to test confidence and new prepare_data)
if __name__ == "__main__":
    # (Your existing __main__ block can be kept or adapted)
    # Note: To test the new prepare_data and confidence score,
    # you'd need sample data with categorical columns and run
    # train_health_scorer followed by calculate_overall_health_score.
    # Example usage and testing

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
        
        print(f"     {device_id}:")
        print(f"         Health Score: {result['overall_score']:.3f} (Method: {result['calculation_method']})")
        print(f"         (Formula: {result['formula_score']:.3f}, Model: {result.get('model_score', 'N/A')})")
        print(f"         Status: {result['health_status']}")
        print(f"         Risk Level: {result['risk_assessment']['overall_risk_level']}")
        print(f"         (Used {result['historical_data_points']} data points for component scores)")
        print()
    
    # 2. Calculate fleet health score (using latest data for each)
    print("3. Calculating fleet health score (using latest data):")
    # We already calculated the latest scores, so we can just use the results
    # But for a real demo, we'd call the fleet function
    latest_fleet_data = {dev_id: data.iloc[-1:] for dev_id, data in device_data.items()}
    fleet_result = health_calculator.calculate_fleet_health_score(latest_fleet_data)
    
    print(f"     Fleet Health Score: {fleet_result['fleet_health_score']:.3f}")
    print(f"     Fleet Status: {fleet_result['fleet_health_status']}")
    print(f"     Healthy Devices: {fleet_result['fleet_analytics']['healthy_devices']}")
    print(f"     Warning Devices: {fleet_result['fleet_analytics']['warning_devices']}")
    print(f"     Critical Devices: {fleet_result['fleet_analytics']['critical_devices']}")
    print()
    
    # 3. Show best and worst performing devices
    print("4. Device Performance Ranking:")
    print("     Best Performing:")
    for device in fleet_result['best_performing_devices'][:3]:
        print(f"         {device['device_id']}: {device['score']:.3f}")
    
    print("     Worst Performing:")
    for device in fleet_result['worst_performing_devices'][-3:]:
        print(f"         {device['device_id']}: {device['score']:.3f}")
    print()
    
    # 4. Show recommendations
    print("5. Fleet Recommendations:")
    for i, rec in enumerate(fleet_result['fleet_recommendations'][:5], 1):
        print(f"     {i}. [{rec['priority'].upper()}] {rec['action']}")
        print(f"         Timeline: {rec['timeframe']}")
    print()
    
    # 5. Risk assessment summary
    print("6. Fleet Risk Assessment:")
    risk = fleet_result['fleet_risk_assessment']
    print(f"     Fleet Risk Level: {risk['fleet_risk_level']}")
    print(f"     High Risk Devices: {len(risk['high_risk_devices'])}")
    print(f"     Medium Risk Devices: {len(risk['medium_risk_devices'])}")
    print(f"     Low Risk Devices: {len(risk['low_risk_devices'])}")
    print()
    
    # 6. Statistics
    print("7. Health Scoring Statistics:")
    stats = health_calculator.get_health_statistics()
    print(f"     Total health records: {stats['history_summary']['total_health_records']}")
    print(f"     Devices tracked: {stats['history_summary']['devices_tracked']}")
    print(f"     Model Loaded: {stats['configuration_summary']['model_loaded']}")
    print(f"     Baseline metrics (EMAs): {stats['configuration_summary']['baseline_metrics']}")
    
    print("\n=== HEALTH SCORE ANALYSIS COMPLETED ===")