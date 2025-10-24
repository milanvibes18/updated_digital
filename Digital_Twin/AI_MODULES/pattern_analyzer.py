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

# Scientific computing libraries for signal processing, stats, clustering, etc.
from scipy import signal, stats, fft
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit

# Machine Learning libraries for preprocessing, dimensionality reduction, clustering, etc.
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import IsolationForest

# Time series analysis libraries
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox

# --- Aggregation Helper Functions for Feature Extraction ---

def _pa_agg_slope(series: pd.Series) -> float:
    """
    (Helper) Calculate the slope of the linear trend for a given series.

    Args:
        series (pd.Series): Input time series data.

    Returns:
        float: The calculated slope of the linear trend. Returns 0.0 if calculation fails or not enough data.
    """
    # Check if there are enough data points for trend calculation
    if len(series) < 2:
        return 0.0
    try:
        # Drop missing values
        series = series.dropna()
        if len(series) < 2:
            return 0.0
        # Create x-axis values (indices)
        x = np.arange(len(series))
        # Fit a linear polynomial (degree 1)
        coeffs = np.polyfit(x, series.values, 1)
        # Return the slope (first coefficient)
        return float(coeffs[0])
    except (np.linalg.LinAlgError, ValueError):
        # Handle potential errors during polyfit
        return 0.0

def _pa_agg_autocorr(series: pd.Series) -> float:
    """
    (Helper) Calculate the lag-1 autocorrelation of a series.

    Args:
        series (pd.Series): Input time series data.

    Returns:
        float: The lag-1 autocorrelation coefficient. Returns 0.0 if calculation fails or not enough data.
    """
    # Check for sufficient data points
    if len(series) < 2:
        return 0.0
    # Calculate autocorrelation for lag 1
    val = series.dropna().autocorr(lag=1)
    # Return the value or 0.0 if it's NaN
    return float(val) if pd.notna(val) else 0.0

def _pa_agg_cv(series: pd.Series) -> float:
    """
    (Helper) Calculate the coefficient of variation (std / mean).

    Args:
        series (pd.Series): Input time series data.

    Returns:
        float: The coefficient of variation. Returns 0.0 if mean is zero or calculation fails.
    """
    mean = series.mean()
    std = series.std()
    # Check for NaN or zero mean to avoid division errors
    if pd.isna(mean) or pd.isna(std) or mean == 0:
        return 0.0
    # Calculate CV
    return float(std / mean)

def _pa_agg_entropy(series: pd.Series) -> float:
    """
    (Helper) Calculate the Shannon entropy of a categorical series.

    Args:
        series (pd.Series): Input categorical data.

    Returns:
        float: The calculated entropy. Returns 0.0 for empty series.
    """
    if series.empty:
        return 0.0
    # Calculate probability distribution of categories
    probs = series.dropna().value_counts(normalize=True)
    # Calculate entropy
    return float(stats.entropy(probs))

# --- End Helper Functions ---


class PatternAnalyzer:
    """
    Advanced pattern analysis system for Digital Twin applications.
    Detects temporal patterns (cycles, seasonality, trends), spatial relationships (clustering, correlation),
    behavioral patterns (entity clustering, sequences, anomalies), and system-level correlations
    in industrial IoT data. Provides methods for analyzing various pattern types and exporting results.
    """

    def __init__(self, cache_path="ANALYTICS/analysis_cache/"):
        """
        Initialize the PatternAnalyzer.

        Args:
            cache_path (str): Path to store cache files and pattern templates.
        """
        self.cache_path = Path(cache_path)
        self.logger = self._setup_logging()

        # Create cache directory if it doesn't exist
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # In-memory storage for detected patterns and history
        self.detected_patterns: Dict[str, Any] = {} # Stores results of different analyses (temporal, spatial, etc.)
        self.pattern_templates: Dict[str, Dict] = {} # Stores user-saved patterns for later matching
        self.pattern_history: deque = deque(maxlen=1000) # Tracks recent analysis runs

        # Default configuration for various analyses
        self.config = {
            'temporal_patterns': {
                'min_pattern_length': 5,      # Minimum length for subsequence detection
                'max_pattern_length': 100,     # Maximum length for subsequence detection
                'similarity_threshold': 0.8,  # Threshold for subsequence similarity (not currently used)
                'frequency_threshold': 3      # Min occurrences for a subsequence to be considered recurring
            },
            'spatial_patterns': {
                'clustering_eps': 0.5,        # DBSCAN epsilon parameter
                'min_samples': 5,             # DBSCAN minimum samples parameter
                'correlation_threshold': 0.7  # Threshold for strong spatial correlation
            },
            'anomaly_patterns': {
                'z_score_threshold': 3.0,     # Threshold for Z-score anomaly detection
                'isolation_contamination': 0.1,# Assumed contamination for Isolation Forest
                'moving_window_size': 50      # Window size for moving anomaly checks (not currently used)
            },
            'seasonal_patterns': {
                'min_periods': 24,            # Minimum data points for seasonality check (e.g., hours in a day)
                'seasonality_test_alpha': 0.05,# Significance level for seasonality tests (not currently used)
                'decomposition_model': 'additive' # 'additive' or 'multiplicative' for seasonal_decompose
            }
        }

        # Storage for potential ML models used in pattern recognition (currently unused)
        self.pattern_models = {}
        self.scalers = {}

        # Load previously saved pattern templates on initialization
        self._load_pattern_templates()

    def _setup_logging(self) -> logging.Logger:
        """
        Sets up logging for the PatternAnalyzer class.

        Returns:
            logging.Logger: Configured logger instance.
        """
        logger = logging.getLogger('PatternAnalyzer')
        logger.setLevel(logging.INFO) # Set default logging level

        # Add file handler only if no handlers are already configured
        if not logger.handlers:
            log_dir = Path('LOGS')
            log_dir.mkdir(exist_ok=True) # Ensure LOGS directory exists
            handler = logging.FileHandler(log_dir / 'digital_twin_patterns.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            # Optional: Add console handler for immediate feedback
            # console_handler = logging.StreamHandler(sys.stdout)
            # console_handler.setFormatter(formatter)
            # logger.addHandler(console_handler)

        return logger

    def _load_pattern_templates(self) -> None:
        """Load existing pattern templates from a pickle file in the cache directory."""
        try:
            template_file = self.cache_path / "pattern_templates.pkl"
            if template_file.exists():
                with open(template_file, 'rb') as f:
                    # Load the dictionary of templates from the file
                    self.pattern_templates = pickle.load(f)
                self.logger.info(f"Loaded {len(self.pattern_templates)} pattern templates from {template_file}")
        except Exception as e:
            # Log error and reset templates if loading fails
            self.logger.error(f"Failed to load pattern templates: {e}", exc_info=True)
            self.pattern_templates = {}

    def _save_pattern_templates(self) -> None:
        """Save the current pattern templates dictionary to a pickle file."""
        try:
            template_file = self.cache_path / "pattern_templates.pkl"
            with open(template_file, 'wb') as f:
                # Save the current templates dictionary
                pickle.dump(self.pattern_templates, f)
            self.logger.info(f"Pattern templates saved to {template_file}")
        except Exception as e:
            self.logger.error(f"Failed to save pattern templates: {e}", exc_info=True)

    def analyze_temporal_patterns(self, data: pd.DataFrame,
                                  timestamp_col: str = 'timestamp',
                                  value_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze temporal patterns (cycles, seasonality, trends, recurrence) in time series data.

        Args:
            data (pd.DataFrame): DataFrame containing time series data.
            timestamp_col (str): Name of the column containing timestamps.
            value_cols (Optional[List[str]]): List of numeric column names to analyze.
                                                If None, analyzes all numeric columns.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis results for each specified value column,
                            including cyclical, seasonal, trend, recurring subsequence, periodicity,
                            and change point information.
        """
        try:
            self.logger.info("Starting temporal pattern analysis...")

            # Ensure data is not modified in place
            df = data.copy()

            # Convert timestamp column to datetime and set as index
            if timestamp_col in df.columns:
                try:
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                    df.set_index(timestamp_col, inplace=True)
                except Exception as e:
                    self.logger.error(f"Failed to process timestamp column '{timestamp_col}': {e}")
                    raise ValueError(f"Invalid timestamp column: {timestamp_col}") from e
            else:
                 raise ValueError(f"Timestamp column '{timestamp_col}' not found in data.")

            # Automatically select numeric columns if value_cols is not specified
            if value_cols is None:
                value_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            # Initialize results dictionary
            results = {
                'analysis_timestamp': datetime.now().isoformat(),
                'analyzed_columns': value_cols,
                'patterns_found': {}
            }

            # Analyze each specified value column
            for col in value_cols:
                if col not in df.columns:
                    self.logger.warning(f"Column '{col}' not found in data. Skipping.")
                    continue

                # Get the series data, drop NaNs
                series = df[col].dropna()

                # Check if enough data points exist for analysis
                min_pattern_length = self.config['temporal_patterns']['min_pattern_length']
                if len(series) < min_pattern_length:
                    self.logger.warning(f"Insufficient data points ({len(series)} < {min_pattern_length}) for column '{col}'. Skipping.")
                    continue

                # Store patterns found for the current column
                col_patterns = {}

                # --- Perform various temporal analyses ---
                col_patterns['cyclical'] = self._detect_cyclical_patterns(series)
                col_patterns['seasonal'] = self._detect_seasonal_patterns(series)
                col_patterns['trend'] = self._detect_trend_patterns(series)
                col_patterns['recurring_subsequences'] = self._detect_recurring_subsequences(series)
                col_patterns['periodicity'] = self._analyze_periodicity(series)
                col_patterns['change_points'] = self._detect_change_points(series)

                results['patterns_found'][col] = col_patterns

            # Store results in memory and update history
            self.detected_patterns['temporal'] = results
            self.pattern_history.append(('temporal_analysis', datetime.now().isoformat()))

            self.logger.info(f"Temporal pattern analysis completed for columns: {value_cols}")
            return results

        except Exception as e:
            self.logger.error(f"Temporal pattern analysis error: {e}", exc_info=True)
            # Re-raise to indicate failure to the caller
            raise

    def _detect_cyclical_patterns(self, series: pd.Series) -> Dict[str, Any]:
        """
        Detect cyclical patterns using Autocorrelation Function (ACF).

        Args:
            series (pd.Series): Input time series data.

        Returns:
            Dict[str, Any]: Dictionary containing detected cycles, their lengths, strengths, and the dominant cycle.
        """
        try:
            # Determine maximum lag for ACF calculation (heuristic)
            max_lag = min(len(series) // 4, 100) # Check up to 1/4 of series length or 100 points
            if max_lag < 1:
                return {'cycles_detected': 0, 'cycles': []} # Not enough data for lag calculation

            # Calculate ACF using statsmodels, using FFT for efficiency
            autocorr = acf(series.values, nlags=max_lag, fft=True)

            # Find significant peaks in the ACF plot (excluding lag 0)
            # height: minimum peak height
            # distance: minimum horizontal distance between peaks
            # prominence: minimum vertical drop from peak to surrounding lows
            peaks, properties = signal.find_peaks(
                autocorr[1:], # Exclude lag 0 (always 1.0)
                height=0.3, # Minimum correlation strength
                distance=5, # Minimum separation between cycle lengths
                prominence=0.1 # Peak must stand out from surroundings
            )

            # Process found peaks into cycle information
            cycles = []
            for peak_idx in peaks:
                cycle_length = peak_idx + 1 # +1 because we excluded lag 0
                cycle_strength = autocorr[cycle_length] # ACF value at the peak lag

                cycles.append({
                    'cycle_length': int(cycle_length), # Lag represents cycle length in data points
                    'strength': float(cycle_strength), # How strong the correlation is at this lag
                    # Estimate period in hours if series index has frequency info
                    'period_hours': float(cycle_length * pd.Timedelta(series.index.freq).total_seconds() / 3600) if series.index.freq else None
                })

            # Sort cycles by strength (descending) to find the most dominant ones
            cycles.sort(key=lambda x: x['strength'], reverse=True)

            return {
                'cycles_detected': len(cycles),
                'cycles': cycles[:5], # Return top 5 strongest cycles
                'dominant_cycle': cycles[0] if cycles else None # The strongest cycle
            }

        except Exception as e:
            self.logger.error(f"Cyclical pattern detection error: {e}", exc_info=True)
            # Return empty result on error
            return {'cycles_detected': 0, 'cycles': []}

    def _detect_seasonal_patterns(self, series: pd.Series) -> Dict[str, Any]:
        """
        Detect seasonal patterns using time series decomposition.

        Args:
            series (pd.Series): Input time series data.

        Returns:
            Dict[str, Any]: Dictionary indicating if seasonality was found, details of detected patterns,
                            and the strongest seasonal pattern identified.
        """
        try:
            min_periods_config = self.config['seasonal_patterns']['min_periods']

            # Check if there's enough data for seasonality analysis (at least 2 full periods)
            if len(series) < 2 * min_periods_config:
                return {'seasonal': False, 'reason': f'Insufficient data (less than 2 * {min_periods_config} periods)'}

            seasonal_results = {}
            # Define potential seasonal periods to check (e.g., daily, weekly, monthly in hours)
            potential_periods = [24, 168, 720] # Assuming hourly data

            for period in potential_periods:
                # Ensure enough data for the current period check
                if len(series) < 2 * period:
                    continue

                try:
                    # Perform seasonal decomposition using statsmodels
                    decomposition = seasonal_decompose(
                        series,
                        model=self.config['seasonal_patterns']['decomposition_model'], # 'additive' or 'multiplicative'
                        period=period,
                        extrapolate_trend='freq' # Method to handle NaNs at ends due to filtering
                    )

                    # Calculate seasonal strength (variance of seasonal / (variance of seasonal + variance of residual))
                    seasonal_comp = decomposition.seasonal.dropna()
                    residual_comp = decomposition.resid.dropna()
                    if seasonal_comp.empty or residual_comp.empty: continue # Skip if components are empty

                    seasonal_var = np.var(seasonal_comp)
                    residual_var = np.var(residual_comp)

                    # Avoid division by zero
                    total_var = seasonal_var + residual_var
                    seasonal_strength = seasonal_var / total_var if total_var > 0 else 0

                    # Consider seasonality significant if strength is above a threshold (e.g., 0.1)
                    if seasonal_strength > 0.1:
                        seasonal_results[f'period_{period}'] = {
                            'period': period,
                            'strength': float(seasonal_strength),
                            # Store the last full cycle of the seasonal component
                            'seasonal_component': seasonal_comp.tolist()[-period:],
                            # Store the last full cycle of the trend component
                            'trend_component': decomposition.trend.dropna().tolist()[-period:]
                        }

                except Exception as decomp_e:
                    # Log errors during decomposition for a specific period but continue checking others
                    self.logger.debug(f"Decomposition failed for period {period}: {decomp_e}")
                    continue

            # Find the seasonal pattern with the highest strength
            best_seasonal = None
            if seasonal_results:
                best_seasonal = max(seasonal_results.values(), key=lambda x: x['strength'])

            return {
                'seasonal': len(seasonal_results) > 0, # True if any significant seasonality was found
                'seasonal_patterns': seasonal_results, # Details for each significant period found
                'best_seasonal': best_seasonal # The pattern with the highest strength
            }

        except Exception as e:
            self.logger.error(f"Seasonal pattern detection error: {e}", exc_info=True)
            return {'seasonal': False, 'reason': str(e)}

    def _detect_trend_patterns(self, series: pd.Series) -> Dict:
        """Detect trend patterns using statistical methods."""
        try:
            # Linear trend
            x = np.arange(len(series))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, series.values)
            
            # Determine trend type
            trend_type = 'stable'
            if abs(slope) > 2 * std_err:
                trend_type = 'increasing' if slope > 0 else 'decreasing'
            
            # Polynomial trend analysis
            poly_trends = {}
            for degree in [2, 3]:
                try:
                    coeffs = np.polyfit(x, series.values, degree)
                    poly_fit = np.polyval(coeffs, x)
                    r_squared = 1 - (np.sum((series.values - poly_fit) ** 2) / 
                                     np.sum((series.values - np.mean(series.values)) ** 2))
                    
                    poly_trends[f'degree_{degree}'] = {
                        'coefficients': coeffs.tolist(),
                        'r_squared': float(r_squared)
                    }
                except:
                    continue
            
            # Trend change points
            trend_changes = self._detect_trend_changes(series)
            
            return {
                'linear_trend': {
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'r_squared': float(r_value**2),
                    'p_value': float(p_value),
                    'trend_type': trend_type,
                    'significance': float(abs(slope) / std_err) if std_err > 0 else 0
                },
                'polynomial_trends': poly_trends,
                'trend_changes': trend_changes
            }
            
        except Exception as e:
            self.logger.error(f"Trend pattern detection error: {e}")
            return {}
    
    def _detect_trend_changes(self, series: pd.Series) -> Dict:
        """Detect trend change points."""
        try:
            # Simple approach using moving windows
            window_size = max(10, len(series) // 20)
            if len(series) < 2 * window_size:
                 return {'change_points_detected': 0, 'change_points': []}
                
            trends = []
            
            for i in range(window_size, len(series) - window_size):
                before = series.iloc[i-window_size:i]
                after = series.iloc[i:i+window_size]
                
                # Calculate slopes
                x_before = np.arange(len(before))
                x_after = np.arange(len(after))
                
                slope_before, _, _, _, _ = stats.linregress(x_before, before.values)
                slope_after, _, _, _, _ = stats.linregress(x_after, after.values)
                
                # Check for significant change
                slope_diff = abs(slope_after - slope_before)
                trends.append((i, slope_diff))
            
            # Find significant changes
            if trends:
                trend_values = [t[1] for t in trends]
                threshold = np.mean(trend_values) + 2 * np.std(trend_values)
                
                change_points = [(idx, diff) for idx, diff in trends if diff > threshold]
                change_points.sort(key=lambda x: x[1], reverse=True)
                
                return {
                    'change_points_detected': len(change_points),
                    'change_points': [
                        {
                            'index': int(cp[0]),
                            'timestamp': series.index[cp[0]].isoformat() if hasattr(series.index[cp[0]], 'isoformat') else str(series.index[cp[0]]),
                            'magnitude': float(cp[1])
                        }
                        for cp in change_points[:5]  # Top 5 changes
                    ]
                }
            
            return {'change_points_detected': 0, 'change_points': []}
            
        except Exception as e:
            self.logger.error(f"Trend change detection error: {e}")
            return {}
    
    def _detect_recurring_subsequences(self, series: pd.Series) -> Dict:
        """Detect recurring subsequences using pattern matching."""
        try:
            min_length = self.config['temporal_patterns']['min_pattern_length']
            max_length = min(self.config['temporal_patterns']['max_pattern_length'], len(series) // 4)
            
            if min_length > max_length:
                 return {'recurring_patterns_found': 0, 'patterns': []}

            recurring_patterns = []
            
            # Normalize series for pattern matching
            normalized = (series - series.mean()) / series.std()
            
            for pattern_length in range(min_length, max_length + 1):
                pattern_counts = defaultdict(list)
                
                # Extract all subsequences of this length
                for i in range(len(normalized) - pattern_length + 1):
                    subsequence = normalized.iloc[i:i + pattern_length]
                    
                    # Convert to tuple for hashing (rounded for fuzzy matching)
                    pattern_key = tuple(np.round(subsequence.values, 2))
                    pattern_counts[pattern_key].append(i)
                
                # Find frequently occurring patterns
                for pattern, occurrences in pattern_counts.items():
                    if len(occurrences) >= self.config['temporal_patterns']['frequency_threshold']:
                        # Calculate pattern statistics
                        avg_interval = np.mean(np.diff(occurrences)) if len(occurrences) > 1 else 0
                        
                        recurring_patterns.append({
                            'pattern_length': pattern_length,
                            'occurrences': len(occurrences),
                            'positions': occurrences,
                            'average_interval': float(avg_interval),
                            'pattern_values': list(pattern)
                        })
            
            # Sort by frequency and pattern length
            recurring_patterns.sort(key=lambda x: (x['occurrences'], x['pattern_length']), reverse=True)
            
            return {
                'recurring_patterns_found': len(recurring_patterns),
                'patterns': recurring_patterns[:10]  # Top 10 patterns
            }
            
        except Exception as e:
            self.logger.error(f"Recurring subsequence detection error: {e}")
            return {'recurring_patterns_found': 0, 'patterns': []}
    
    def _analyze_periodicity(self, series: pd.Series) -> Dict:
        """Analyze periodicity using frequency domain analysis."""
        try:
            # Remove trend
            detrended = signal.detrend(series.values)
            
            # Apply window function
            windowed = detrended * signal.windows.hann(len(detrended))
            
            # FFT analysis
            fft_result = fft.fft(windowed)
            frequencies = fft.fftfreq(len(windowed))
            
            # Get positive frequencies only
            positive_freq_idx = frequencies > 0
            if not np.any(positive_freq_idx):
                return {'dominant_periods_found': 0, 'periods': []}

            positive_freqs = frequencies[positive_freq_idx]
            positive_magnitudes = np.abs(fft_result[positive_freq_idx])
            
            # Find dominant frequencies
            peak_indices, _ = signal.find_peaks(positive_magnitudes, height=np.max(positive_magnitudes) * 0.1)
            
            dominant_periods = []
            for idx in peak_indices:
                if positive_freqs[idx] > 0:
                    period = 1.0 / positive_freqs[idx]
                    magnitude = positive_magnitudes[idx]
                    
                    dominant_periods.append({
                        'period': float(period),
                        'frequency': float(positive_freqs[idx]),
                        'magnitude': float(magnitude),
                        'strength': float(magnitude / np.max(positive_magnitudes))
                    })
            
            # Sort by strength
            dominant_periods.sort(key=lambda x: x['strength'], reverse=True)
            
            return {
                'dominant_periods_found': len(dominant_periods),
                'periods': dominant_periods[:5],  # Top 5 periods
                'spectral_energy': float(np.sum(positive_magnitudes**2)),
                'frequency_resolution': float(1.0 / len(series))
            }
            
        except Exception as e:
            self.logger.error(f"Periodicity analysis error: {e}")
            return {'dominant_periods_found': 0, 'periods': []}
    
    def _detect_change_points(self, series: pd.Series) -> Dict:
        """Detect change points using statistical methods."""
        try:
            # CUSUM (Cumulative Sum) change point detection
            values = series.values
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val == 0:
                return {'mean_change_points': [], 'variance_change_points': [], 'total_change_points': 0}

            # Standardize
            standardized = (values - mean_val) / std_val
            
            # Calculate CUSUM
            cusum_pos = np.zeros(len(standardized))
            cusum_neg = np.zeros(len(standardized))
            
            h = 2.0  # Decision interval
            k = 0.5  # Reference value
            
            for i in range(1, len(standardized)):
                cusum_pos[i] = max(0, cusum_pos[i-1] + standardized[i] - k)
                cusum_neg[i] = max(0, cusum_neg[i-1] - standardized[i] - k)
            
            # Find change points
            change_points = []
            for i in range(len(cusum_pos)):
                if cusum_pos[i] > h or cusum_neg[i] > h:
                    change_points.append({
                        'index': int(i),
                        'timestamp': series.index[i].isoformat() if hasattr(series.index[i], 'isoformat') else str(series.index[i]),
                        'cusum_positive': float(cusum_pos[i]),
                        'cusum_negative': float(cusum_neg[i]),
                        'direction': 'upward' if cusum_pos[i] > cusum_neg[i] else 'downward'
                    })
            
            # Variance change point detection using F-test
            variance_changes = self._detect_variance_changes(series)
            
            return {
                'mean_change_points': change_points,
                'variance_change_points': variance_changes,
                'total_change_points': len(change_points) + len(variance_changes)
            }
            
        except Exception as e:
            self.logger.error(f"Change point detection error: {e}")
            return {}
    
    def _detect_variance_changes(self, series: pd.Series) -> List[Dict]:
        """Detect variance change points using F-test."""
        try:
            values = series.values
            window_size = max(10, len(values) // 20)
            if len(values) < 2 * window_size:
                return []
                
            change_points = []
            
            for i in range(window_size, len(values) - window_size):
                # Split data
                before = values[i-window_size:i]
                after = values[i:i+window_size]
                
                # Calculate variances
                var_before = np.var(before, ddof=1)
                var_after = np.var(after, ddof=1)
                
                if var_before > 0 and var_after > 0:
                    # F-test
                    f_statistic = var_after / var_before if var_after > var_before else var_before / var_after
                    
                    # Critical value for F-test (approximate)
                    critical_f = 2.0  # Simplified threshold
                    
                    if f_statistic > critical_f:
                        change_points.append({
                            'index': int(i),
                            'timestamp': series.index[i].isoformat() if hasattr(series.index[i], 'isoformat') else str(series.index[i]),
                            'f_statistic': float(f_statistic),
                            'variance_before': float(var_before),
                            'variance_after': float(var_after),
                            'change_type': 'increase' if var_after > var_before else 'decrease'
                        })
            
            return change_points
            
        except Exception as e:
            self.logger.error(f"Variance change detection error: {e}")
            return []
    
    def analyze_spatial_patterns(self, data: pd.DataFrame, 
                                 location_cols: List[str] = None,
                                 value_cols: List[str] = None) -> Dict:
        """
        Analyze spatial patterns in data.

        Args:
            data: DataFrame with spatial data
            location_cols: Columns containing location information (e.g., ['x', 'y'])
            value_cols: Value columns to analyze spatially
            
        Returns:
            Dictionary containing spatial pattern analysis results
        """
        try:
            self.logger.info("Starting spatial pattern analysis")
            
            df = data.copy()
            
            # Auto-detect location columns if not provided
            if location_cols is None:
                location_cols = [col for col in df.columns if any(
                    keyword in col.lower() for keyword in ['lat', 'lon', 'x', 'y', 'location']
                )]
            
            # Auto-detect value columns if not provided
            if value_cols is None:
                value_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                              if col not in location_cols]

            if not location_cols or len(location_cols) < 2:
                self.logger.warning("Insufficient location columns for spatial analysis.")
                return {'analysis_timestamp': datetime.now().isoformat(), 'spatial_patterns': {}}
            
            # Ensure data is numeric and clean
            spatial_data = df[location_cols + value_cols].dropna()
            
            if len(spatial_data) < self.config['spatial_patterns']['min_samples']:
                self.logger.warning("Insufficient data for spatial analysis after cleaning.")
                return {'analysis_timestamp': datetime.now().isoformat(), 'spatial_patterns': {}}

            results = {
                'analysis_timestamp': datetime.now().isoformat(),
                'analyzed_location_cols': location_cols,
                'analyzed_value_cols': value_cols,
                'spatial_patterns': {}
            }

            # 1. Spatial Clustering (DBSCAN)
            clustering_results = {}
            try:
                coords = spatial_data[location_cols].values
                coords_scaled = StandardScaler().fit_transform(coords)
                
                dbscan = DBSCAN(
                    eps=self.config['spatial_patterns']['clustering_eps'],
                    min_samples=self.config['spatial_patterns']['min_samples']
                )
                clusters = dbscan.fit_predict(coords_scaled)
                
                n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                n_noise = np.sum(clusters == -1)
                
                spatial_data['cluster'] = clusters
                
                cluster_summary = {}
                for c in range(n_clusters):
                    cluster_data = spatial_data[spatial_data['cluster'] == c]
                    cluster_summary[f'cluster_{c}'] = {
                        'size': len(cluster_data),
                        'center': cluster_data[location_cols].mean().to_dict(),
                        'value_means': cluster_data[value_cols].mean().to_dict()
                    }

                clustering_results = {
                    'n_clusters': n_clusters,
                    'n_noise_points': int(n_noise),
                    'cluster_summary': cluster_summary,
                    'silhouette_score': float(silhouette_score(coords_scaled, clusters)) if n_clusters > 1 else 0.0
                }
            except Exception as e:
                self.logger.error(f"Spatial clustering error: {e}")
                clustering_results = {'error': str(e)}

            results['spatial_patterns']['clustering'] = clustering_results

            # 2. Spatial Correlation (based on values)
            correlation_results = {}
            try:
                corr_matrix = spatial_data[value_cols].corr()
                
                # Find strong correlations
                strong_correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        corr_val = corr_matrix.iloc[i, j]
                        
                        if abs(corr_val) > self.config['spatial_patterns']['correlation_threshold']:
                            strong_correlations.append({
                                'variable_1': col1,
                                'variable_2': col2,
                                'correlation': float(corr_val),
                                'type': 'positive' if corr_val > 0 else 'negative'
                            })
                
                correlation_results = {
                    'correlation_matrix': corr_matrix.to_dict(),
                    'strong_correlations': strong_correlations
                }
            except Exception as e:
                self.logger.error(f"Spatial correlation error: {e}")
                correlation_results = {'error': str(e)}
            
            results['spatial_patterns']['correlation'] = correlation_results
            
            self.detected_patterns['spatial'] = results
            self.pattern_history.append(('spatial_analysis', datetime.now().isoformat()))
            self.logger.info("Spatial pattern analysis completed")
            return results

        except Exception as e:
            self.logger.error(f"Error in spatial pattern analysis: {e}")
            return {'analysis_timestamp': datetime.now().isoformat(), 'spatial_patterns': {}, 'error': str(e)}

    def analyze_behavioral_patterns(self, data: pd.DataFrame, 
                                    entity_col: str,
                                    behavior_cols: List[str],
                                    timestamp_col: str) -> Dict:
        """
        Analyze behavioral patterns of different entities.

        Args:
            data: DataFrame with behavioral data
            entity_col: Column name identifying the entity (e.g., 'device_id')
            behavior_cols: Columns representing behaviors (e.g., ['usage', 'temp'])
            timestamp_col: Column name for timestamps
            
        Returns:
            Dictionary containing behavioral pattern analysis results
        """
        try:
            self.logger.info("Starting behavioral pattern analysis")
            df = data.copy()

            if entity_col not in df.columns:
                self.logger.error(f"Entity column '{entity_col}' not found in data.")
                return {}
            
            results = {
                'analysis_timestamp': datetime.now().isoformat(),
                'analyzed_entity_col': entity_col,
                'analyzed_behavior_cols': behavior_cols,
                'total_entities': int(df[entity_col].nunique()),
                'behavioral_patterns': {}
            }

            # 1. Create aggregated behavioral features for clustering
            features_df = self._create_behavioral_features(df, entity_col, behavior_cols)
            
            if features_df.empty:
                 self.logger.warning("No behavioral features could be created.")
                 return results
            
            feature_cols = [col for col in features_df.columns if col != entity_col]
            features_scaled = StandardScaler().fit_transform(features_df[feature_cols])

            # 2. Entity Clustering (KMeans)
            try:
                n_clusters = self._find_optimal_clusters_behavioral(features_scaled)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(features_scaled)
                
                features_df['cluster'] = clusters
                
                cluster_summaries = {}
                for c in range(n_clusters):
                    cluster_data = features_df[features_df['cluster'] == c]
                    cluster_summaries[f'cluster_{c}'] = {
                        'size': len(cluster_data),
                        'entities': cluster_data[entity_col].tolist(),
                        'behavior_summary': self._summarize_cluster_behavior(cluster_data, behavior_cols)
                    }
                
                results['behavioral_patterns']['entity_clustering'] = {
                    'n_clusters': n_clusters,
                    'cluster_summaries': cluster_summaries,
                    'silhouette_score': float(silhouette_score(features_scaled, clusters)) if n_clusters > 1 else 0.0
                }
            except Exception as e:
                self.logger.error(f"Behavioral clustering error: {e}")
                results['behavioral_patterns']['entity_clustering'] = {'error': str(e)}

            # 3. Behavioral Sequence Analysis
            try:
                sequence_results = self._analyze_behavioral_sequences(
                    df, entity_col, behavior_cols, timestamp_col
                )
                results['behavioral_patterns']['sequences'] = sequence_results
            except Exception as e:
                self.logger.error(f"Behavioral sequence analysis error: {e}")
                results['behavioral_patterns']['sequences'] = {'error': str(e)}

            # 4. Anomalous Behavior Detection
            try:
                anomaly_results = self._detect_anomalous_behavior(
                    features_df.drop(columns=['cluster'], errors='ignore'), entity_col, feature_cols
                )
                results['behavioral_patterns']['anomalies'] = anomaly_results
            except Exception as e:
                self.logger.error(f"Anomalous behavior detection error: {e}")
                results['behavioral_patterns']['anomalies'] = {'error': str(e)}

            # 5. State Transition Analysis
            try:
                transition_results = self._analyze_state_transitions(
                    df, entity_col, behavior_cols, timestamp_col
                )
                results['behavioral_patterns']['state_transitions'] = transition_results
            except Exception as e:
                self.logger.error(f"State transition analysis error: {e}")
                results['behavioral_patterns']['state_transitions'] = {'error': str(e)}

            # 6. Collective Behavior Analysis
            try:
                collective_results = self._analyze_collective_behavior(
                    df, entity_col, behavior_cols
                )
                results['behavioral_patterns']['collective_behavior'] = collective_results
            except Exception as e:
                self.logger.error(f"Collective behavior analysis error: {e}")
                results['behavioral_patterns']['collective_behavior'] = {'error': str(e)}

            self.detected_patterns['behavioral'] = results
            self.pattern_history.append(('behavioral_analysis', datetime.now().isoformat()))
            self.logger.info("Behavioral pattern analysis completed")
            return results

        except Exception as e:
            self.logger.error(f"Error in behavioral pattern analysis: {e}")
            return {'analysis_timestamp': datetime.now().isoformat(), 'behavioral_patterns': {}, 'error': str(e)}

    def _create_behavioral_features(self, data: pd.DataFrame, 
                                    entity_col: str, 
                                    behavior_cols: List[str]) -> pd.DataFrame:
        """
        Create aggregated behavioral features for each entity.
        (Improved version with temporal and categorical features)

        Args:
            data: The input DataFrame.
            entity_col: The column identifying entities.
            behavior_cols: The columns representing behaviors.

        Returns:
            A DataFrame where each row is an entity and columns are aggregated features.
        """
        try:
            self.logger.info("Creating improved behavioral features")
            df = data.copy()
            
            # Separate numeric and categorical columns
            numeric_cols = []
            categorical_cols = []
            
            for col in behavior_cols:
                if col not in df.columns:
                    continue
                # Exclude boolean from numeric, treat as categorical
                if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                    numeric_cols.append(col)
                else:
                    # Treat boolean, object, category types as categorical
                    categorical_cols.append(col)
            
            if not numeric_cols and not categorical_cols:
                self.logger.warning("No valid behavior columns found for feature creation.")
                return pd.DataFrame()

            all_features = []

            # --- 1. Process Numeric Features ---
            if numeric_cols:
                self.logger.info(f"Processing numeric features: {numeric_cols}")
                # Define aggregations
                numeric_aggregations = {
                    'mean': 'mean',
                    'std': 'std',
                    'median': 'median',
                    'min': 'min',
                    'max': 'max',
                    'q25': lambda x: x.quantile(0.25),
                    'q75': lambda x: x.quantile(0.75),
                    'skew': 'skew',
                    'kurt': 'kurtosis',
                    'cv': _pa_agg_cv,
                    'slope': _pa_agg_slope,
                    'autocorr_lag1': _pa_agg_autocorr
                }
                
                # Build the full aggregation dictionary
                full_numeric_agg = {}
                for col in numeric_cols:
                    full_numeric_agg[col] = list(numeric_aggregations.values())
                
                numeric_features_df = df.groupby(entity_col).agg(full_numeric_agg)
                
                # Flatten multi-index columns
                numeric_features_df.columns = ['_'.join(col).strip() for col in numeric_features_df.columns.values]
                all_features.append(numeric_features_df)

            # --- 2. Process Categorical Features ---
            if categorical_cols:
                self.logger.info(f"Processing categorical features: {categorical_cols}")
                cat_aggregations = {
                    'nunique': 'nunique',
                    'top': lambda x: x.mode()[0] if not x.empty and not x.mode().empty else None,
                    'entropy': _pa_agg_entropy
                }
                
                full_cat_agg = {}
                for col in categorical_cols:
                    full_cat_agg[col] = list(cat_aggregations.values())
                    
                categorical_features_df = df.groupby(entity_col).agg(full_cat_agg)
                categorical_features_df.columns = ['_'.join(col).strip() for col in categorical_features_df.columns.values]
                
                # One-hot encode the 'top' (mode) feature
                dfs_to_concat = [categorical_features_df]
                cols_to_drop = []
                for col in categorical_cols:
                    top_col = f"{col}_top"
                    if top_col in categorical_features_df.columns:
                        # pd.get_dummies handles NaNs (from empty groups)
                        top_dummies = pd.get_dummies(categorical_features_df[top_col], prefix=top_col, dummy_na=True)
                        dfs_to_concat.append(top_dummies)
                        cols_to_drop.append(top_col)
                
                # Drop the original 'top' columns
                categorical_features_df = categorical_features_df.drop(columns=cols_to_drop)
                dfs_to_concat[0] = categorical_features_df # Update the base df
                
                all_features.append(pd.concat(dfs_to_concat, axis=1))

            # --- 3. Combine Features ---
            if not all_features:
                return pd.DataFrame() # Should be covered, but as a safeguard
                
            features_df = pd.concat(all_features, axis=1)
            
            # Handle potential NaNs (e.g., from std on 1 record, or empty groups)
            features_df = features_df.fillna(0)
            
            features_df.reset_index(inplace=True)
            
            self.logger.info(f"Created {len(features_df.columns) - 1} features for {len(features_df)} entities.")
            return features_df

        except Exception as e:
            self.logger.error(f"Improved behavioral feature creation error: {e}")
            return pd.DataFrame()

    def _find_optimal_clusters_behavioral(self, data: np.ndarray, max_k: int = 8) -> int:
        """
        Find optimal number of clusters for behavioral data using the Elbow method.

        Args:
            data: Scaled feature data for clustering.
            max_k: Maximum number of clusters to test.

        Returns:
            Optimal number of clusters (k).
        """
        try:
            if len(data) <= 3:
                return min(2, len(data))
            
            inertias = []
            k_range = range(2, min(max_k + 1, len(data)))
            
            if not k_range:
                return 2
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(data)
                inertias.append(kmeans.inertia_)
            
            if len(inertias) >= 2:
                # Calculate rate of change
                deltas = np.diff(inertias)
                
                # Find elbow (point of maximum change in rate)
                if len(deltas) >= 2:
                    delta_deltas = np.diff(deltas)
                    elbow_idx = np.argmax(np.abs(delta_deltas)) + 2 # +2 to account for diffs and 0-index
                    return min(elbow_idx + 1, max_k) # +1 to align with k_range
            
            return min(3, len(data))
            
        except Exception as e:
            self.logger.error(f"Optimal cluster finding error: {e}")
            return 3
    
    def _summarize_cluster_behavior(self, cluster_data: pd.DataFrame, behavior_cols: List[str]) -> Dict:
        """
        Summarize behavioral characteristics of a cluster.
        (Improved to handle new features)

        Args:
            cluster_data: DataFrame containing feature data for a single cluster.
            behavior_cols: List of base behavior column names (e.g., 'temperature').

        Returns:
            A dictionary summarizing the statistics for each behavior in the cluster.
        """
        try:
            summary = {}
            
            for col in behavior_cols:
                # Find all feature columns derived from this base behavior
                metric_cols = [c for c in cluster_data.columns if c.startswith(f"{col}_")]
                
                if metric_cols:
                    col_summary = {}
                    for metric_col in metric_cols:
                        # Skip one-hot-encoded 'top' features, as their mean is just a proportion
                        if '_top_' in metric_col: 
                            continue
                            
                        # Extract metric (e.g., mean, std, slope)
                        metric_name = metric_col[len(col)+1:] 
                        
                        if metric_col not in cluster_data.columns:
                            continue

                        col_summary[metric_name] = {
                            'mean': float(cluster_data[metric_col].mean()),
                            'std': float(cluster_data[metric_col].std()),
                            'median': float(cluster_data[metric_col].median())
                        }
                    
                    # Add proportion for one-hot-encoded 'top' features
                    top_cols = [c for c in metric_cols if '_top_' in c]
                    if top_cols:
                        # Calculate mean proportion for each 'top' feature
                        top_proportions = cluster_data[top_cols].mean()
                        # Filter out proportions that are 0 (not relevant)
                        relevant_proportions = top_proportions[top_proportions > 0].to_dict()
                        if relevant_proportions:
                            col_summary['top_proportions'] = relevant_proportions

                    summary[col] = col_summary
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Cluster behavior summary error: {e}")
            return {}
    
    def _analyze_behavioral_sequences(self, data: pd.DataFrame, 
                                      entity_col: str, 
                                      behavior_cols: List[str], 
                                      timestamp_col: str) -> Dict:
        """
        Analyze behavioral sequences over time for entities.

        Args:
            data: The input DataFrame.
            entity_col: The column identifying entities.
            behavior_cols: The columns representing behaviors.
            timestamp_col: The timestamp column.

        Returns:
            A dictionary with individual and global sequence patterns.
        """
        try:
            df = data.copy()
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.sort_values([entity_col, timestamp_col])
            
            sequence_patterns = {}
            
            # Analyze sequences for each entity
            for entity_id in df[entity_col].unique():
                entity_data = df[df[entity_col] == entity_id].copy()
                
                if len(entity_data) < 5:  # Need minimum sequence length
                    continue
                
                # Create behavioral states based on quantiles
                entity_sequences = {}
                for col in behavior_cols:
                    if col not in entity_data.columns or not pd.api.types.is_numeric_dtype(entity_data[col]):
                        continue
                    
                    values = entity_data[col].dropna()
                    if values.empty:
                        continue
                    
                    # Create discrete states based on quartiles
                    try:
                        quartiles = values.quantile([0.25, 0.5, 0.75])
                    except ValueError:
                        continue # Handle case with all equal values
                        
                    states = []
                    
                    for val in values:
                        if pd.isna(val):
                            states.append('nan')
                        elif val <= quartiles[0.25]:
                            states.append('low')
                        elif val <= quartiles[0.5]:
                            states.append('medium_low')
                        elif val <= quartiles[0.75]:
                            states.append('medium_high')
                        else:
                            states.append('high')
                    
                    entity_sequences[col] = states
                
                if not entity_sequences:
                    continue

                # Find common subsequences
                common_subsequences = self._find_common_subsequences(entity_sequences)
                
                if common_subsequences:
                    sequence_patterns[str(entity_id)] = {
                        'sequence_length': len(entity_data),
                        'common_patterns': common_subsequences[:5],  # Top 5 patterns
                        'behavioral_states': {k: v[:50] for k, v in entity_sequences.items()} # Truncate for readability
                    }
            
            # Find global patterns across all entities
            global_patterns = self._find_global_sequence_patterns(sequence_patterns)
            
            return {
                'individual_sequences': len(sequence_patterns),
                'entity_patterns': dict(list(sequence_patterns.items())[:10]),  # Limit for readability
                'global_patterns': global_patterns
            }
            
        except Exception as e:
            self.logger.error(f"Behavioral sequence analysis error: {e}")
            return {}
    
    def _find_common_subsequences(self, sequences: Dict[str, List[str]]) -> List[Dict]:
        """
        Find common subsequences in behavioral states.

        Args:
            sequences: A dictionary mapping behavior columns to their state sequences.

        Returns:
            A list of common patterns found.
        """
        try:
            if not sequences:
                return []
            
            # Focus on the first behavioral variable for simplicity
            first_col = list(sequences.keys())[0]
            states = sequences[first_col]
            
            pattern_counts = defaultdict(int)
            min_length = 3
            max_length = min(10, len(states) // 2)

            if min_length > max_length:
                return []
            
            # Extract all subsequences
            for length in range(min_length, max_length + 1):
                for i in range(len(states) - length + 1):
                    subsequence = tuple(states[i:i + length])
                    pattern_counts[subsequence] += 1
            
            # Find frequent patterns
            common_patterns = []
            for pattern, count in pattern_counts.items():
                if count >= 2:  # Appears at least twice
                    common_patterns.append({
                        'pattern': list(pattern),
                        'frequency': count,
                        'length': len(pattern)
                    })
            
            # Sort by frequency
            common_patterns.sort(key=lambda x: x['frequency'], reverse=True)
            
            return common_patterns
            
        except Exception as e:
            self.logger.error(f"Common subsequence finding error: {e}")
            return []
    
    def _find_global_sequence_patterns(self, sequence_patterns: Dict) -> Dict:
        """
        Find patterns that occur across multiple entities.

        Args:
            sequence_patterns: The dictionary of patterns found per entity.

        Returns:
            A dictionary of global patterns.
        """
        try:
            if not sequence_patterns:
                return {}
            
            # Collect all patterns from all entities
            all_patterns = []
            for entity_patterns in sequence_patterns.values():
                for pattern in entity_patterns.get('common_patterns', []):
                    all_patterns.append(tuple(pattern['pattern']))
            
            # Count global pattern frequencies
            global_pattern_counts = defaultdict(int)
            for pattern in all_patterns:
                global_pattern_counts[pattern] += 1
            
            # Find patterns that occur in multiple entities
            global_patterns = []
            for pattern, count in global_pattern_counts.items():
                if count >= 2:  # Appears in at least 2 entities
                    global_patterns.append({
                        'pattern': list(pattern),
                        'entity_frequency': count,
                        'pattern_length': len(pattern)
                    })
            
            # Sort by frequency
            global_patterns.sort(key=lambda x: x['entity_frequency'], reverse=True)
            
            return {
                'cross_entity_patterns': global_patterns[:10],
                'total_global_patterns': len(global_patterns)
            }
            
        except Exception as e:
            self.logger.error(f"Global sequence pattern finding error: {e}")
            return {}
    
    def _detect_anomalous_behavior(self, data: pd.DataFrame, 
                                   entity_col: str, 
                                   behavior_cols: List[str]) -> Dict:
        """
        Detect anomalous behavioral patterns among entities.

        Args:
            data: DataFrame of aggregated entity features.
            entity_col: The column identifying entities.
            behavior_cols: The feature columns to use for anomaly detection.

        Returns:
            A dictionary of anomaly detection results.
        """
        try:
            # Create entity profiles
            entity_profiles = data.set_index(entity_col)[behavior_cols].fillna(0)
            
            if len(entity_profiles) < 5:
                return {'anomaly_detection': False, 'reason': 'Insufficient data'}
            
            # Standardize profiles
            scaler = StandardScaler()
            profiles_scaled = scaler.fit_transform(entity_profiles)
            
            # Use Isolation Forest for anomaly detection
            isolation_forest = IsolationForest(
                contamination=self.config['anomaly_patterns']['isolation_contamination'],
                random_state=42
            )
            
            anomaly_predictions = isolation_forest.fit_predict(profiles_scaled)
            anomaly_scores = isolation_forest.decision_function(profiles_scaled)
            
            # Identify anomalous entities
            anomalous_entities = []
            for i, (entity_id, pred, score) in enumerate(zip(entity_profiles.index, anomaly_predictions, anomaly_scores)):
                if pred == -1:  # Anomaly
                    entity_profile = entity_profiles.iloc[i].to_dict()
                    anomalous_entities.append({
                        'entity_id': str(entity_id),
                        'anomaly_score': float(score),
                        'behavioral_profile': entity_profile,
                        'deviation_analysis': self._analyze_behavioral_deviation(entity_profiles.iloc[i], entity_profiles)
                    })
            
            # Sort by anomaly score
            anomalous_entities.sort(key=lambda x: x['anomaly_score'])
            
            return {
                'anomaly_detection': True,
                'anomalous_entities_count': len(anomalous_entities),
                'anomalous_entities': anomalous_entities,
                'anomaly_rate': len(anomalous_entities) / len(entity_profiles),
                'detection_parameters': {
                    'contamination': self.config['anomaly_patterns']['isolation_contamination'],
                    'total_entities': len(entity_profiles)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Anomalous behavior detection error: {e}")
            return {'anomaly_detection': False, 'reason': str(e)}
    
    def _analyze_behavioral_deviation(self, entity_profile: pd.Series, all_profiles: pd.DataFrame) -> Dict:
        """
        Analyze how an entity's behavior deviates from the norm.

        Args:
            entity_profile: The feature Series for the anomalous entity.
            all_profiles: The DataFrame of features for all entities.

        Returns:
            A dictionary analyzing the deviation for each feature.
        """
        try:
            deviations = {}
            
            for col in entity_profile.index:
                entity_value = entity_profile[col]
                population_mean = all_profiles[col].mean()
                population_std = all_profiles[col].std()
                
                if population_std > 0:
                    z_score = (entity_value - population_mean) / population_std
                    deviations[col] = {
                        'entity_value': float(entity_value),
                        'population_mean': float(population_mean),
                        'z_score': float(z_score),
                        'deviation_type': 'high' if z_score > 0 else 'low',
                        'severity': 'extreme' if abs(z_score) > 3 else 'moderate' if abs(z_score) > 2 else 'mild'
                    }
            
            return deviations
            
        except Exception as e:
            self.logger.error(f"Behavioral deviation analysis error: {e}")
            return {}
    
    def _analyze_state_transitions(self, data: pd.DataFrame, 
                                   entity_col: str, 
                                   behavior_cols: List[str], 
                                   timestamp_col: str) -> Dict:
        """
        Analyze state transitions in behavioral data.

        Args:
            data: The input DataFrame.
            entity_col: The column identifying entities.
            behavior_cols: The columns representing behaviors.
            timestamp_col: The timestamp column.

        Returns:
            A dictionary of transition matrices and insights.
        """
        try:
            df = data.copy()
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.sort_values([entity_col, timestamp_col])
            
            transition_analysis = {}
            
            for col in behavior_cols:
                if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                
                # *** BUG FIX ***: Initialize transition counter for each column
                entity_transitions = defaultdict(lambda: defaultdict(int))
                
                # Create behavioral states
                values = df[col].dropna()
                if values.empty:
                    continue

                try:
                    quartiles = values.quantile([0.33, 0.67])
                except ValueError:
                    continue # Handle case with all equal values

                
                df[f'{col}_state'] = pd.cut(
                    df[col], 
                    bins=[-np.inf, quartiles[0.33], quartiles[0.67], np.inf],
                    labels=['low', 'medium', 'high']
                )
                
                
                for entity_id in df[entity_col].unique():
                    entity_data = df[df[entity_col] == entity_id].copy()
                    
                    if len(entity_data) < 2:
                        continue
                    
                    states = entity_data[f'{col}_state'].values
                    
                    # Count state transitions
                    for i in range(len(states) - 1):
                        current_state = states[i]
                        next_state = states[i + 1]
                        
                        if pd.notna(current_state) and pd.notna(next_state):
                            entity_transitions[current_state][next_state] += 1
                
                # Calculate transition probabilities
                transition_matrix = {}
                for from_state in ['low', 'medium', 'high']:
                    transition_matrix[from_state] = {}
                    total_transitions = sum(entity_transitions[from_state].values())
                    if total_transitions == 0:
                        total_transitions = 1 # Avoid division by zero
                    
                    for to_state in ['low', 'medium', 'high']:
                        count = entity_transitions[from_state][to_state]
                        probability = count / total_transitions
                        transition_matrix[from_state][to_state] = {
                            'count': count,
                            'probability': float(probability)
                        }
                
                transition_analysis[col] = {
                    'transition_matrix': transition_matrix,
                    'most_common_transitions': self._find_most_common_transitions(transition_matrix),
                    'state_stability': self._calculate_state_stability(transition_matrix)
                }
            
            return transition_analysis
            
        except Exception as e:
            self.logger.error(f"State transition analysis error: {e}")
            return {}
    
    def _find_most_common_transitions(self, transition_matrix: Dict) -> List[Dict]:
        """
        Find most common state transitions (excluding self-transitions).

        Args:
            transition_matrix: The calculated transition matrix.

        Returns:
            A list of the most common transitions.
        """
        try:
            transitions = []
            
            for from_state, to_states in transition_matrix.items():
                for to_state, data in to_states.items():
                    if from_state != to_state:  # Exclude self-transitions
                        transitions.append({
                            'from_state': from_state,
                            'to_state': to_state,
                            'probability': data['probability'],
                            'count': data['count']
                        })
            
            # Sort by probability
            transitions.sort(key=lambda x: x['probability'], reverse=True)
            
            return transitions[:5]  # Top 5 transitions
            
        except Exception as e:
            self.logger.error(f"Most common transitions finding error: {e}")
            return []
    
    def _calculate_state_stability(self, transition_matrix: Dict) -> Dict:
        """
        Calculate stability of each state (self-transition probability).

        Args:
            transition_matrix: The calculated transition matrix.

        Returns:
            A dictionary of state stabilities.
        """
        try:
            stability = {}
            
            for state in transition_matrix.keys():
                # Self-transition probability indicates stability
                self_transition_prob = transition_matrix[state][state]['probability']
                stability[state] = {
                    'self_transition_probability': self_transition_prob,
                    'stability_level': 'high' if self_transition_prob > 0.7 else 'medium' if self_transition_prob > 0.4 else 'low'
                }
            
            return stability
            
        except Exception as e:
            self.logger.error(f"State stability calculation error: {e}")
            return {}
    
    def _analyze_collective_behavior(self, data: pd.DataFrame, 
                                    entity_col: str, 
                                    behavior_cols: List[str]) -> Dict:
        """
        Analyze collective behavioral patterns across all entities.

        Args:
            data: The input DataFrame.
            entity_col: The column identifying entities.
            behavior_cols: The columns representing behaviors.

        Returns:
            A dictionary of collective statistics.
        """
        try:
            collective_stats = {}
            
            for col in behavior_cols:
                if col not in data.columns or not pd.api.types.is_numeric_dtype(data[col]):
                    continue
                
                # Overall distribution analysis
                values = data[col].dropna()
                
                if len(values) == 0:
                    continue
                
                # Calculate collective statistics
                collective_stats[col] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'median': float(values.median()),
                    'skewness': float(values.skew()),
                    'kurtosis': float(values.kurtosis()),
                    'coefficient_of_variation': float(values.std() / values.mean()) if values.mean() != 0 else 0
                }
                
                # Distribution shape analysis
                collective_stats[col]['distribution_shape'] = self._analyze_distribution_shape(values)
                
                # Identify collective trends
                entity_means = data.groupby(entity_col)[col].mean().dropna()
                if entity_means.empty:
                    continue

                collective_stats[col]['entity_diversity'] = {
                    'entity_count': len(entity_means),
                    'mean_variation': float(entity_means.std()),
                    'min_entity_mean': float(entity_means.min()),
                    'max_entity_mean': float(entity_means.max()),
                    'outlier_entities': self._find_outlier_entities(entity_means)
                }
            
            return collective_stats
            
        except Exception as e:
            self.logger.error(f"Collective behavior analysis error: {e}")
            return {}
    
    def _analyze_distribution_shape(self, values: pd.Series) -> Dict:
        """
        Analyze the shape of a distribution using skewness and kurtosis.

        Args:
            values: A pandas Series of numeric data.

        Returns:
            A dictionary interpreting the distribution shape.
        """
        try:
            skewness = values.skew()
            kurtosis = values.kurtosis()
            
            # Interpret skewness
            if abs(skewness) < 0.5:
                skew_interpretation = 'approximately_symmetric'
            elif skewness > 0.5:
                skew_interpretation = 'right_skewed'
            else:
                skew_interpretation = 'left_skewed'
            
            # Interpret kurtosis (Fisher's definition, 0 is normal)
            if kurtosis > 1:
                kurtosis_interpretation = 'heavy_tailed (leptokurtic)'
            elif kurtosis < -1:
                kurtosis_interpretation = 'light_tailed (platykurtic)'
            else:
                kurtosis_interpretation = 'normal_tailed (mesokurtic)'
            
            return {
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'skew_interpretation': skew_interpretation,
                'kurtosis_interpretation': kurtosis_interpretation,
                'is_normal_like': abs(skewness) < 0.5 and -1 < kurtosis < 1
            }
            
        except Exception as e:
            self.logger.error(f"Distribution shape analysis error: {e}")
            return {}
    
    def _find_outlier_entities(self, entity_means: pd.Series) -> List[Dict]:
        """
        Find outlier entities based on their mean behavior using IQR.

        Args:
            entity_means: A Series of mean values, indexed by entity_id.

        Returns:
            A list of outlier entity details.
        """
        try:
            # Use IQR method
            Q1 = entity_means.quantile(0.25)
            Q3 = entity_means.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0: # Avoid division by zero or no spread
                return []

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = entity_means[(entity_means < lower_bound) | (entity_means > upper_bound)]
            
            outlier_list = []
            for entity_id, value in outliers.items():
                outlier_type = 'high' if value > upper_bound else 'low'
                outlier_list.append({
                    'entity_id': str(entity_id),
                    'value': float(value),
                    'outlier_type': outlier_type,
                    'deviation_magnitude': float(abs(value - entity_means.median()))
                })
            
            # Sort by deviation magnitude
            outlier_list.sort(key=lambda x: x['deviation_magnitude'], reverse=True)
            
            return outlier_list[:10]  # Top 10 outliers
            
        except Exception as e:
            self.logger.error(f"Outlier entity finding error: {e}")
            return []
    
    def get_pattern_summary(self) -> Dict:
        """
        Get summary of all detected patterns.

        Returns:
            A summary dictionary.
        """
        try:
            summary = {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_pattern_types': len(self.detected_patterns),
                'pattern_types': list(self.detected_patterns.keys()),
                'pattern_templates_count': len(self.pattern_templates),
                'recent_analyses': {}
            }
            
            # Summarize each pattern type
            for pattern_type, pattern_data in self.detected_patterns.items():
                if isinstance(pattern_data, dict) and 'analysis_timestamp' in pattern_data:
                    summary['recent_analyses'][pattern_type] = {
                        'timestamp': pattern_data['analysis_timestamp'],
                        'patterns_found': self._count_patterns_in_analysis(pattern_data)
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Pattern summary error: {e}")
            return {}
    
    def _count_patterns_in_analysis(self, analysis_data: Dict) -> int:
        """
        Count total patterns found in a specific analysis result.

        Args:
            analysis_data: The result dictionary from an analysis method.

        Returns:
            The total count of significant patterns found.
        """
        try:
            count = 0
            
            # Temporal patterns
            if 'patterns_found' in analysis_data:
                for col_patterns in analysis_data['patterns_found'].values():
                    if isinstance(col_patterns, dict):
                        for pattern_category in col_patterns.values():
                            if isinstance(pattern_category, dict):
                                if 'cycles' in pattern_category:
                                    count += len(pattern_category.get('cycles', []))
                                if 'patterns' in pattern_category:
                                    count += len(pattern_category.get('patterns', []))
                                if 'change_points' in pattern_category:
                                    count += pattern_category.get('change_points_detected', 0)
            
            # Spatial patterns
            if 'spatial_patterns' in analysis_data:
                count += analysis_data['spatial_patterns'].get('clustering', {}).get('n_clusters', 0)
                count += len(analysis_data['spatial_patterns'].get('correlation', {}).get('strong_correlations', []))

            # Behavioral patterns
            if 'behavioral_patterns' in analysis_data:
                count += analysis_data['behavioral_patterns'].get('entity_clustering', {}).get('n_clusters', 0)
                count += analysis_data['behavioral_patterns'].get('anomalies', {}).get('anomalous_entities_count', 0)
                count += analysis_data['behavioral_patterns'].get('sequences', {}).get('individual_sequences', 0)

            return count
            
        except Exception as e:
            self.logger.error(f"Pattern counting error: {e}")
            return 0
    
    def save_patterns_to_template(self, pattern_name: str, patterns: Dict):
        """
        Save detected patterns as reusable templates.

        Args:
            pattern_name: A unique name for the template.
            patterns: The pattern dictionary to save (e.g., from results['patterns_found']).
        """
        try:
            self.pattern_templates[pattern_name] = {
                'patterns': patterns,
                'created_at': datetime.now().isoformat(),
                'usage_count': 0
            }
            
            self._save_pattern_templates()
            self.logger.info(f"Pattern template saved: {pattern_name}")
            
        except Exception as e:
            self.logger.error(f"Pattern template saving error: {e}")
    
    def match_against_templates(self, data: pd.DataFrame, template_name: str = None) -> Dict:
        """
        Match new data against existing pattern templates.

        Args:
            data: The new DataFrame to check.
            template_name: Specific template to match. If None, matches all.

        Returns:
            A dictionary of matching results.
        """
        try:
            if not self.pattern_templates:
                return {'matches': [], 'message': 'No templates available'}
            
            templates_to_check = [template_name] if template_name else list(self.pattern_templates.keys())
            matches = []
            
            for tmpl_name in templates_to_check:
                if tmpl_name not in self.pattern_templates:
                    continue
                
                template = self.pattern_templates[tmpl_name]
                
                # Simple template matching (can be enhanced)
                match_score = self._calculate_template_match_score(data, template['patterns'])
                
                if match_score > 0.5:  # Threshold for significant match
                    matches.append({
                        'template_name': tmpl_name,
                        'match_score': match_score,
                        'template_created': template['created_at'],
                        'matching_elements': self._identify_matching_elements(data, template['patterns'])
                    })
                
                # Update usage count
                self.pattern_templates[tmpl_name]['usage_count'] += 1
            
            # Sort by match score
            matches.sort(key=lambda x: x['match_score'], reverse=True)
            
            return {
                'matches': matches,
                'templates_checked': len(templates_to_check),
                'best_match': matches[0] if matches else None
            }
            
        except Exception as e:
            self.logger.error(f"Template matching error: {e}")
            return {'matches': [], 'error': str(e)}
    
    def _calculate_template_match_score(self, data: pd.DataFrame, template_patterns: Dict) -> float:
        """
        Calculate how well data matches a pattern template (simplified).

        Args:
            data: The new DataFrame.
            template_patterns: The patterns dictionary from a template.

        Returns:
            A match score between 0.0 and 1.0.
        """
        try:
            # Simplified matching - compare statistical properties
            score = 0.0
            comparisons = 0
            
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col in template_patterns:
                    col_data = data[col].dropna()
                    if len(col_data) > 0:
                        # Compare basic statistics
                        data_mean = col_data.mean()
                        data_std = col_data.std()
                        
                        # Get template stats (e.g., from temporal trend)
                        template_mean = template_patterns[col].get('trend', {}).get('linear_trend', {}).get('intercept', data_mean)
                        template_std = data_std # Placeholder, better comparison needed
                        
                        # Calculate similarity (inverse of normalized difference)
                        mean_diff = abs(data_mean - template_mean) / (abs(template_mean) + 1e-8)
                        # std_diff = abs(data_std - template_std) / (template_std + 1e-8) # std comparison is tricky
                        
                        col_similarity = 1.0 / (1.0 + mean_diff) # Simplified
                        score += col_similarity
                        comparisons += 1
            
            return score / comparisons if comparisons > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Template match score calculation error: {e}")
            return 0.0
    
    def _identify_matching_elements(self, data: pd.DataFrame, template_patterns: Dict) -> List[str]:
        """
        Identify which elements (e.g., columns) match the template.

        Args:
            data: The new DataFrame.
            template_patterns: The patterns dictionary from a template.

        Returns:
            A list of strings describing matching elements.
        """
        try:
            matching_elements = []
            
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col in template_patterns:
                    # Simple check - if column exists
                    col_data = data[col].dropna()
                    if len(col_data) > 0:
                        matching_elements.append(f"Column: {col}")
            
            return matching_elements
            
        except Exception as e:
            self.logger.error(f"Matching elements identification error: {e}")
            return []
    
    def export_patterns(self, output_path: str = None) -> str:
        """
        Export all detected patterns and templates to a JSON file.

        Args:
            output_path: Optional path to save the file.

        Returns:
            The final path of the exported file.
        """
        try:
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = self.cache_path / f"pattern_export_{timestamp}.json"
            else:
                output_path = Path(output_path)
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'detected_patterns': self.detected_patterns,
                'pattern_templates': self.pattern_templates,
                'configuration': self.config
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Patterns exported to {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Pattern export error: {e}")
            raise
    
    def import_patterns(self, input_path: str):
        """
        Import patterns and templates from a JSON file.

        Args:
            input_path: The path to the JSON file to import.
        """
        try:
            input_path = Path(input_path)
            if not input_path.exists():
                self.logger.error(f"Import file not found: {input_path}")
                return

            with open(input_path, 'r') as f:
                import_data = json.load(f)
            
            # Merge imported patterns
            if 'detected_patterns' in import_data:
                self.detected_patterns.update(import_data['detected_patterns'])
            
            if 'pattern_templates' in import_data:
                self.pattern_templates.update(import_data['pattern_templates'])
            
            if 'configuration' in import_data:
                # Merge configuration (imported values override existing ones)
                self.config.update(import_data['configuration'])
            
            self.logger.info(f"Patterns imported from {input_path}")
            
        except Exception as e:
            self.logger.error(f"Pattern import error: {e}")
            raise
    
    def clear_pattern_cache(self):
        """Clear all in-memory detected patterns and history."""
        try:
            self.detected_patterns.clear()
            self.pattern_history.clear()
            
            # Note: This does not clear saved template files, only in-memory state.
            
            self.logger.info("In-memory pattern cache cleared")
            
        except Exception as e:
            self.logger.error(f"Pattern cache clear error: {e}")
    
    def get_pattern_statistics(self) -> Dict:
        """
        Get comprehensive statistics about pattern analysis.

        Returns:
            A dictionary of statistics.
        """
        try:
            stats = {
                'analysis_summary': {
                    'total_analyses_performed': len(self.detected_patterns),
                    'pattern_types_analyzed': list(self.detected_patterns.keys()),
                    'template_count': len(self.pattern_templates),
                    'history_length': len(self.pattern_history)
                },
                'configuration': self.config,
                'recent_activity': list(self.pattern_history)[-10:] if self.pattern_history else []
            }
            
            # Add detailed statistics for each pattern type
            pattern_type_stats = {}
            for pattern_type, data in self.detected_patterns.items():
                pattern_type_stats[pattern_type] = {
                    'last_analysis': data.get('analysis_timestamp', 'Unknown'),
                    'patterns_detected': self._count_patterns_in_analysis(data)
                }
            
            stats['pattern_type_statistics'] = pattern_type_stats
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Pattern statistics error: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    # Initialize pattern analyzer
    analyzer = PatternAnalyzer()
    
    # Generate sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='H')
    
    # Temporal data with patterns
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'device_id': np.random.choice(['device_A', 'device_B', 'device_C'], 1000),
        'temperature': 20 + 5 * np.sin(np.arange(1000) * 2 * np.pi / 24) + np.random.normal(0, 1, 1000),
        'pressure': 1013 + 2 * np.sin(np.arange(1000) * 2 * np.pi / 168) + np.random.normal(0, 2, 1000),
        'vibration': 0.1 + 0.05 * np.sin(np.arange(1000) * 2 * np.pi / 12) + np.random.exponential(0.02, 1000),
        'location_x': np.random.uniform(0, 10, 1000), # Reduced range for better clustering
        'location_y': np.random.uniform(0, 10, 1000),
        'usage_hours': np.random.uniform(0, 24, 1000),
        'efficiency': np.random.beta(2, 2, 1000) * 100,
        'status_code': np.random.choice(['OK', 'WARN', 'FAIL', 'OK', 'OK'], 1000) # Added categorical feature
    })

    # Add some anomalies for behavioral analysis
    sample_data.loc[sample_data['device_id'] == 'device_A', 'temperature'] *= 1.5
    sample_data.loc[sample_data['device_id'] == 'device_B', 'efficiency'] = 10
    
    print("=== DIGITAL TWIN PATTERN ANALYSIS DEMO ===\n")
    
    # 1. Temporal Pattern Analysis
    print("1. Analyzing temporal patterns...")
    temporal_results = analyzer.analyze_temporal_patterns(
        sample_data, 
        timestamp_col='timestamp',
        value_cols=['temperature', 'pressure', 'vibration']
    )
    
    print(f"    - Analyzed {len(temporal_results['analyzed_columns'])} variables")
    for col, patterns in temporal_results.get('patterns_found', {}).items():
        cyclical = patterns.get('cyclical', {})
        seasonal = patterns.get('seasonal', {})
        print(f"    - {col}: {cyclical.get('cycles_detected', 0)} cycles, seasonal: {seasonal.get('seasonal', False)}")
    
    # 2. Spatial Pattern Analysis
    print("\n2. Analyzing spatial patterns...")
    # Update config for demo data
    analyzer.config['spatial_patterns']['clustering_eps'] = 1.0 
    spatial_results = analyzer.analyze_spatial_patterns(
        sample_data,
        location_cols=['location_x', 'location_y'],
        value_cols=['temperature', 'pressure', 'vibration']
    )
    
    clustering = spatial_results.get('spatial_patterns', {}).get('clustering', {})
    correlation = spatial_results.get('spatial_patterns', {}).get('correlation', {})
    print(f"    - Spatial clustering: {clustering.get('n_clusters', 0)} clusters found")
    print(f"    - Strong correlations: {len(correlation.get('strong_correlations', []))}")
    
    # 3. Behavioral Pattern Analysis
    print("\n3. Analyzing behavioral patterns...")
    behavioral_results = analyzer.analyze_behavioral_patterns(
        sample_data,
        entity_col='device_id',
        behavior_cols=['temperature', 'pressure', 'vibration', 'efficiency', 'status_code'], # Include categorical
        timestamp_col='timestamp'
    )
    
    clustering_info = behavioral_results.get('behavioral_patterns', {}).get('entity_clustering', {})
    anomalies = behavioral_results.get('behavioral_patterns', {}).get('anomalies', {})
    print(f"    - Entity clusters: {clustering_info.get('n_clusters', 0)}")
    print(f"    - Anomalous entities: {anomalies.get('anomalous_entities_count', 0)}")
    
    # Show summary for one cluster to demo new features
    if 'cluster_summaries' in clustering_info:
        print("\n    --- Example Cluster 0 Behavior Summary ---")
        summary_c0 = clustering_info['cluster_summaries'].get('cluster_0', {}).get('behavior_summary', {})
        for behavior, stats in summary_c0.items():
            print(f"        - {behavior}:")
            for stat_name, values in stats.items():
                if stat_name == 'top_proportions':
                    print(f"            - {stat_name}: {values}")
                else:
                    print(f"            - {stat_name} (mean): {values.get('mean', 0.0):.2f}")
        print("    ------------------------------------------")

    # 4. Pattern Summary
    print("\n4. Pattern Analysis Summary:")
    summary = analyzer.get_pattern_summary()
    print(f"    - Total pattern types analyzed: {summary['total_pattern_types']}")
    print(f"    - Pattern types: {', '.join(summary['pattern_types'])}")
    
    # 5. Save patterns as template
    print("\n5. Saving patterns as template...")
    if 'patterns_found' in temporal_results:
        analyzer.save_patterns_to_template('industrial_sensor_patterns', temporal_results['patterns_found'])
        print("    - Template saved successfully")
    else:
        print("    - No temporal patterns to save")
    
    # 6. Export patterns
    print("\n6. Exporting patterns...")
    export_path = analyzer.export_patterns()
    print(f"    - Patterns exported to: {export_path}")
    
    # 7. Get statistics
    print("\n7. Pattern Analysis Statistics:")
    stats = analyzer.get_pattern_statistics()
    print(f"    - Total analyses: {stats['analysis_summary']['total_analyses_performed']}")
    print(f"    - Templates available: {stats['analysis_summary']['template_count']}")
    
    print("\n=== PATTERN ANALYSIS COMPLETED ===")
    print("\nKey Findings:")
    print("- Temporal patterns detected in temperature (daily cycles)")
    print("- Spatial clustering potentially revealed device groupings")
    print("- Behavioral analysis identified device performance patterns (e.g., device_A/B)")
    print(f"- Anomaly detection found {anomalies.get('anomalous_entities_count', 0)} outlier devices")
    print("- All patterns saved for future reference and comparison")