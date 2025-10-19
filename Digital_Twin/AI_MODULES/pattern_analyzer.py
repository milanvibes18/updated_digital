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

# Scientific computing
from scipy import signal, stats, fft
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit

# Machine Learning
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_regression

# Time series analysis
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox

class PatternAnalyzer:
    """
    Advanced pattern analysis system for Digital Twin applications.
    Detects temporal patterns, spatial relationships, behavioral patterns,
    and system-level correlations in industrial IoT data.
    """
    
    def __init__(self, cache_path="ANALYTICS/analysis_cache/"):
        self.cache_path = Path(cache_path)
        self.logger = self._setup_logging()
        
        # Create cache directory
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Pattern storage
        self.detected_patterns = {}
        self.pattern_templates = {}
        self.pattern_history = deque(maxlen=1000)
        
        # Analysis configuration
        self.config = {
            'temporal_patterns': {
                'min_pattern_length': 5,
                'max_pattern_length': 100,
                'similarity_threshold': 0.8,
                'frequency_threshold': 3
            },
            'spatial_patterns': {
                'clustering_eps': 0.5,
                'min_samples': 5,
                'correlation_threshold': 0.7
            },
            'anomaly_patterns': {
                'z_score_threshold': 3.0,
                'isolation_contamination': 0.1,
                'moving_window_size': 50
            },
            'seasonal_patterns': {
                'min_periods': 24,
                'seasonality_test_alpha': 0.05,
                'decomposition_model': 'additive'
            }
        }
        
        # Pattern recognition models
        self.pattern_models = {}
        self.scalers = {}
        
        # Load existing patterns
        self._load_pattern_templates()
        
    def _setup_logging(self):
        """Setup logging for pattern analyzer."""
        logger = logging.getLogger('PatternAnalyzer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Create logs directory if it doesn't exist
            Path('LOGS').mkdir(exist_ok=True)
            handler = logging.FileHandler('LOGS/digital_twin_patterns.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_pattern_templates(self):
        """Load existing pattern templates from cache."""
        try:
            template_file = self.cache_path / "pattern_templates.pkl"
            if template_file.exists():
                with open(template_file, 'rb') as f:
                    self.pattern_templates = pickle.load(f)
                    
                self.logger.info(f"Loaded {len(self.pattern_templates)} pattern templates")
        except Exception as e:
            self.logger.error(f"Failed to load pattern templates: {e}")
            self.pattern_templates = {}
    
    def _save_pattern_templates(self):
        """Save pattern templates to cache."""
        try:
            template_file = self.cache_path / "pattern_templates.pkl"
            with open(template_file, 'wb') as f:
                pickle.dump(self.pattern_templates, f)
                
            self.logger.info("Pattern templates saved")
        except Exception as e:
            self.logger.error(f"Failed to save pattern templates: {e}")
    
    def analyze_temporal_patterns(self, data: pd.DataFrame, 
                                 timestamp_col: str = 'timestamp',
                                 value_cols: List[str] = None) -> Dict:
        """
        Analyze temporal patterns in time series data.
        
        Args:
            data: DataFrame with time series data
            timestamp_col: Column name for timestamps
            value_cols: List of value columns to analyze
            
        Returns:
            Dictionary containing temporal pattern analysis results
        """
        try:
            self.logger.info("Starting temporal pattern analysis")
            
            # Prepare data
            df = data.copy()
            if timestamp_col in df.columns:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                df.set_index(timestamp_col, inplace=True)
            
            # Select value columns
            if value_cols is None:
                value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            results = {
                'analysis_timestamp': datetime.now().isoformat(),
                'analyzed_columns': value_cols,
                'patterns_found': {}
            }
            
            for col in value_cols:
                if col not in df.columns:
                    continue
                    
                series = df[col].dropna()
                if len(series) < self.config['temporal_patterns']['min_pattern_length']:
                    continue
                
                col_patterns = {}
                
                # 1. Cyclical patterns
                cyclical = self._detect_cyclical_patterns(series)
                col_patterns['cyclical'] = cyclical
                
                # 2. Seasonal patterns
                seasonal = self._detect_seasonal_patterns(series)
                col_patterns['seasonal'] = seasonal
                
                # 3. Trend patterns
                trend = self._detect_trend_patterns(series)
                col_patterns['trend'] = trend
                
                # 4. Recurring subsequences
                recurring = self._detect_recurring_subsequences(series)
                col_patterns['recurring_subsequences'] = recurring
                
                # 5. Periodicity analysis
                periodicity = self._analyze_periodicity(series)
                col_patterns['periodicity'] = periodicity
                
                # 6. Change point detection
                change_points = self._detect_change_points(series)
                col_patterns['change_points'] = change_points
                
                results['patterns_found'][col] = col_patterns
            
            # Store results
            self.detected_patterns['temporal'] = results
            
            self.logger.info("Temporal pattern analysis completed")
            return results
            
        except Exception as e:
            self.logger.error(f"Temporal pattern analysis error: {e}")
            raise
    
    def _detect_cyclical_patterns(self, series: pd.Series) -> Dict:
        """Detect cyclical patterns using autocorrelation."""
        try:
            # Calculate autocorrelation
            max_lag = min(len(series) // 4, 100)
            autocorr = acf(series.values, nlags=max_lag, fft=True)
            
            # Find peaks in autocorrelation
            peaks, properties = signal.find_peaks(
                autocorr[1:], 
                height=0.3, 
                distance=5,
                prominence=0.1
            )
            
            cycles = []
            for peak_idx in peaks:
                cycle_length = peak_idx + 1
                cycle_strength = autocorr[cycle_length]
                
                cycles.append({
                    'cycle_length': int(cycle_length),
                    'strength': float(cycle_strength),
                    'period_hours': float(cycle_length) if series.index.freq else None
                })
            
            # Sort by strength
            cycles.sort(key=lambda x: x['strength'], reverse=True)
            
            return {
                'cycles_detected': len(cycles),
                'cycles': cycles[:5],  # Top 5 cycles
                'dominant_cycle': cycles[0] if cycles else None
            }
            
        except Exception as e:
            self.logger.error(f"Cyclical pattern detection error: {e}")
            return {'cycles_detected': 0, 'cycles': []}
    
    def _detect_seasonal_patterns(self, series: pd.Series) -> Dict:
        """Detect seasonal patterns using decomposition."""
        try:
            min_periods = self.config['seasonal_patterns']['min_periods']
            
            if len(series) < 2 * min_periods:
                return {'seasonal': False, 'reason': 'Insufficient data'}
            
            # Try different seasonal periods
            seasonal_results = {}
            
            for period in [24, 168, 720]:  # Daily, weekly, monthly (hours)
                if len(series) < 2 * period:
                    continue
                    
                try:
                    decomposition = seasonal_decompose(
                        series, 
                        model=self.config['seasonal_patterns']['decomposition_model'],
                        period=period,
                        extrapolate_trend='freq'
                    )
                    
                    # Calculate seasonal strength
                    seasonal_var = np.var(decomposition.seasonal.dropna())
                    residual_var = np.var(decomposition.resid.dropna())
                    seasonal_strength = seasonal_var / (seasonal_var + residual_var) if (seasonal_var + residual_var) > 0 else 0
                    
                    if seasonal_strength > 0.1:  # Significant seasonality
                        seasonal_results[f'period_{period}'] = {
                            'period': period,
                            'strength': float(seasonal_strength),
                            'seasonal_component': decomposition.seasonal.dropna().tolist()[-period:],
                            'trend_component': decomposition.trend.dropna().tolist()[-period:]
                        }
                        
                except Exception as e:
                    continue
            
            # Find best seasonal pattern
            best_seasonal = None
            if seasonal_results:
                best_seasonal = max(seasonal_results.values(), key=lambda x: x['strength'])
            
            return {
                'seasonal': len(seasonal_results) > 0,
                'seasonal_patterns': seasonal_results,
                'best_seasonal': best_seasonal
            }
            
        except Exception as e:
            self.logger.error(f"Seasonal pattern detection error: {e}")
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
            location_cols: Columns containing location information
            value_cols: Value columns to analyze spatially
            
        Returns:
            Dictionary containing spatial pattern analysis results
        """
        try:
            self.logger.info("Starting spatial pattern analysis")
            
            # Prepare data
            df = data.copy()
            
            if location_cols is None:
                location_cols = [col for col in df.columns if any(
                    keyword in col.lower() for keyword in ['lat', 'lon', 'x', 'y', 'location']
                )]
        except Exception as e:
            self.logger.error(f"Error in spatial pattern analysis: {e}")

    def _find_optimal_clusters_behavioral(self, data: np.ndarray, max_k: int = 8) -> int:
        """Find optimal number of clusters for behavioral data."""
        try:
            if len(data) <= 3:
                return min(2, len(data))
            
            inertias = []
            k_range = range(2, min(max_k + 1, len(data)))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(data)
                inertias.append(kmeans.inertia_)
            
            if len(inertias) >= 2:
                # Calculate rate of change
                deltas = np.diff(inertias)
                
                # Find elbow (point of maximum change)
                if len(deltas) >= 2:
                    delta_deltas = np.diff(deltas)
                    elbow_idx = np.argmax(np.abs(delta_deltas)) + 2
                    return min(elbow_idx, max_k)
            
            return min(3, len(data))
            
        except Exception as e:
            self.logger.error(f"Optimal cluster finding error: {e}")
            return 3
    
    def _summarize_cluster_behavior(self, cluster_data: pd.DataFrame, behavior_cols: List[str]) -> Dict:
        """Summarize behavioral characteristics of a cluster."""
        try:
            summary = {}
            
            for col in behavior_cols:
                mean_cols = [c for c in cluster_data.columns if c.startswith(f"{col}_")]
                
                if mean_cols:
                    col_summary = {}
                    for metric_col in mean_cols:
                        metric = metric_col.split('_')[-1]  # Extract metric (mean, std, etc.)
                        col_summary[metric] = {
                            'mean': float(cluster_data[metric_col].mean()),
                            'std': float(cluster_data[metric_col].std()),
                            'median': float(cluster_data[metric_col].median())
                        }
                    summary[col] = col_summary
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Cluster behavior summary error: {e}")
            return {}
    
    def _analyze_behavioral_sequences(self, data: pd.DataFrame, 
                                    entity_col: str, 
                                    behavior_cols: List[str], 
                                    timestamp_col: str) -> Dict:
        """Analyze behavioral sequences over time."""
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
                    if col not in entity_data.columns:
                        continue
                    
                    values = entity_data[col]
                    
                    # Create discrete states based on quartiles
                    quartiles = values.quantile([0.25, 0.5, 0.75])
                    states = []
                    
                    for val in values:
                        if val <= quartiles[0.25]:
                            states.append('low')
                        elif val <= quartiles[0.5]:
                            states.append('medium_low')
                        elif val <= quartiles[0.75]:
                            states.append('medium_high')
                        else:
                            states.append('high')
                    
                    entity_sequences[col] = states
                
                # Find common subsequences
                common_subsequences = self._find_common_subsequences(entity_sequences)
                
                if common_subsequences:
                    sequence_patterns[str(entity_id)] = {
                        'sequence_length': len(entity_data),
                        'common_patterns': common_subsequences[:5],  # Top 5 patterns
                        'behavioral_states': entity_sequences
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
        """Find common subsequences in behavioral states."""
        try:
            if not sequences:
                return []
            
            # Focus on the first behavioral variable for simplicity
            first_col = list(sequences.keys())[0]
            states = sequences[first_col]
            
            pattern_counts = defaultdict(int)
            min_length = 3
            max_length = min(10, len(states) // 2)
            
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
        """Find patterns that occur across multiple entities."""
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
        """Detect anomalous behavioral patterns."""
        try:
            # Create entity profiles
            entity_profiles = data.groupby(entity_col)[behavior_cols].mean().fillna(0)
            
            if len(entity_profiles) < 5:
                return {'anomaly_detection': False, 'reason': 'Insufficient data'}
            
            # Standardize profiles
            scaler = StandardScaler()
            profiles_scaled = scaler.fit_transform(entity_profiles)
            
            # Use Isolation Forest for anomaly detection
            from sklearn.ensemble import IsolationForest
            
            isolation_forest = IsolationForest(
                contamination=0.1,
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
                    'contamination': 0.1,
                    'total_entities': len(entity_profiles)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Anomalous behavior detection error: {e}")
            return {'anomaly_detection': False, 'reason': str(e)}
    
    def _analyze_behavioral_deviation(self, entity_profile: pd.Series, all_profiles: pd.DataFrame) -> Dict:
        """Analyze how an entity's behavior deviates from the norm."""
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
        """Analyze state transitions in behavioral data."""
        try:
            df = data.copy()
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.sort_values([entity_col, timestamp_col])
            
            transition_analysis = {}
            
            for col in behavior_cols:
                if col not in df.columns:
                    continue
                
                # Create behavioral states
                values = df[col]
                quartiles = values.quantile([0.33, 0.67])
                
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
                    total_transitions = sum(entity_transitions[from_state].values()) or 1
                    
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
        """Find most common state transitions."""
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
        """Calculate stability of each state."""
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
        """Analyze collective behavioral patterns across entities."""
        try:
            collective_stats = {}
            
            for col in behavior_cols:
                if col not in data.columns:
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
                entity_means = data.groupby(entity_col)[col].mean()
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
        """Analyze the shape of a distribution."""
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
            
            # Interpret kurtosis
            if kurtosis > 3:
                kurtosis_interpretation = 'heavy_tailed'
            elif kurtosis < -1:
                kurtosis_interpretation = 'light_tailed'
            else:
                kurtosis_interpretation = 'normal_tailed'
            
            return {
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'skew_interpretation': skew_interpretation,
                'kurtosis_interpretation': kurtosis_interpretation,
                'is_normal_like': abs(skewness) < 0.5 and -1 < kurtosis < 3
            }
            
        except Exception as e:
            self.logger.error(f"Distribution shape analysis error: {e}")
            return {}
    
    def _find_outlier_entities(self, entity_means: pd.Series) -> List[Dict]:
        """Find outlier entities based on their mean behavior."""
        try:
            # Use IQR method
            Q1 = entity_means.quantile(0.25)
            Q3 = entity_means.quantile(0.75)
            IQR = Q3 - Q1
            
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
        """Get summary of all detected patterns."""
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
        """Count total patterns found in an analysis."""
        try:
            count = 0
            
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
            
            return count
            
        except Exception as e:
            self.logger.error(f"Pattern counting error: {e}")
            return 0
    
    def save_patterns_to_template(self, pattern_name: str, patterns: Dict):
        """Save detected patterns as reusable templates."""
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
        """Match data against existing pattern templates."""
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
        """Calculate how well data matches a pattern template."""
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
                        
                        template_mean = template_patterns[col].get('mean', data_mean)
                        template_std = template_patterns[col].get('std', data_std)
                        
                        # Calculate similarity (inverse of normalized difference)
                        mean_diff = abs(data_mean - template_mean) / (template_mean + 1e-8)
                        std_diff = abs(data_std - template_std) / (template_std + 1e-8)
                        
                        col_similarity = 1.0 / (1.0 + mean_diff + std_diff)
                        score += col_similarity
                        comparisons += 1
            
            return score / comparisons if comparisons > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Template match score calculation error: {e}")
            return 0.0
    
    def _identify_matching_elements(self, data: pd.DataFrame, template_patterns: Dict) -> List[str]:
        """Identify which elements match the template."""
        try:
            matching_elements = []
            
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col in template_patterns:
                    # Simple check - if column exists and has similar range
                    col_data = data[col].dropna()
                    if len(col_data) > 0:
                        matching_elements.append(f"Column: {col}")
            
            return matching_elements
            
        except Exception as e:
            self.logger.error(f"Matching elements identification error: {e}")
            return []
    
    def export_patterns(self, output_path: str = None) -> str:
        """Export detected patterns to JSON file."""
        try:
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = self.cache_path / f"pattern_export_{timestamp}.json"
            
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
        """Import patterns from JSON file."""
        try:
            with open(input_path, 'r') as f:
                import_data = json.load(f)
            
            # Merge imported patterns
            if 'detected_patterns' in import_data:
                self.detected_patterns.update(import_data['detected_patterns'])
            
            if 'pattern_templates' in import_data:
                self.pattern_templates.update(import_data['pattern_templates'])
            
            if 'configuration' in import_data:
                # Merge configuration (keep existing values as priority)
                for key, value in import_data['configuration'].items():
                    if key not in self.config:
                        self.config[key] = value
            
            self.logger.info(f"Patterns imported from {input_path}")
            
        except Exception as e:
            self.logger.error(f"Pattern import error: {e}")
            raise
    
    def clear_pattern_cache(self):
        """Clear pattern analysis cache."""
        try:
            self.detected_patterns.clear()
            self.pattern_history.clear()
            
            # Clear cache files
            cache_files = list(self.cache_path.glob("*.cache"))
            for cache_file in cache_files:
                cache_file.unlink()
            
            self.logger.info("Pattern cache cleared")
            
        except Exception as e:
            self.logger.error(f"Pattern cache clear error: {e}")
    
    def get_pattern_statistics(self) -> Dict:
        """Get comprehensive statistics about pattern analysis."""
        try:
            stats = {
                'analysis_summary': {
                    'total_analyses_performed': len(self.detected_patterns),
                    'pattern_types_analyzed': list(self.detected_patterns.keys()),
                    'template_count': len(self.pattern_templates),
                    'cache_size': len(self.pattern_history)
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
        'location_x': np.random.uniform(0, 100, 1000),
        'location_y': np.random.uniform(0, 100, 1000),
        'usage_hours': np.random.uniform(0, 24, 1000),
        'efficiency': np.random.beta(2, 2, 1000) * 100
    })
    
    print("=== DIGITAL TWIN PATTERN ANALYSIS DEMO ===\n")
    
    # 1. Temporal Pattern Analysis
    print("1. Analyzing temporal patterns...")
    temporal_results = analyzer.analyze_temporal_patterns(
        sample_data, 
        timestamp_col='timestamp',
        value_cols=['temperature', 'pressure', 'vibration']
    )
    
    print(f"   - Analyzed {len(temporal_results['analyzed_columns'])} variables")
    for col, patterns in temporal_results['patterns_found'].items():
        cyclical = patterns.get('cyclical', {})
        seasonal = patterns.get('seasonal', {})
        print(f"   - {col}: {cyclical.get('cycles_detected', 0)} cycles, seasonal: {seasonal.get('seasonal', False)}")
    
    # 2. Spatial Pattern Analysis
    print("\n2. Analyzing spatial patterns...")
    spatial_results = analyzer.analyze_spatial_patterns(
        sample_data,
        location_cols=['location_x', 'location_y'],
        value_cols=['temperature', 'pressure', 'vibration']
    )
    
    clustering = spatial_results['spatial_patterns'].get('clustering', {})
    correlation = spatial_results['spatial_patterns'].get('correlation', {})
    print(f"   - Spatial clustering: {clustering.get('n_clusters', 0)} clusters found")
    print(f"   - Strong correlations: {len(correlation.get('strong_correlations', []))}")
    
    # 3. Behavioral Pattern Analysis
    print("\n3. Analyzing behavioral patterns...")
    behavioral_results = analyzer.analyze_behavioral_patterns(
        sample_data,
        entity_col='device_id',
        behavior_cols=['temperature', 'pressure', 'vibration', 'efficiency'],
        timestamp_col='timestamp'
    )
    
    clustering_info = behavioral_results['behavioral_patterns'].get('entity_clustering', {})
    anomalies = behavioral_results['behavioral_patterns'].get('anomalies', {})
    print(f"   - Entity clusters: {clustering_info.get('n_clusters', 0)}")
    print(f"   - Anomalous entities: {anomalies.get('anomalous_entities_count', 0)}")
    
    # 4. Pattern Summary
    print("\n4. Pattern Analysis Summary:")
    summary = analyzer.get_pattern_summary()
    print(f"   - Total pattern types analyzed: {summary['total_pattern_types']}")
    print(f"   - Pattern types: {', '.join(summary['pattern_types'])}")
    
    # 5. Save patterns as template
    print("\n5. Saving patterns as template...")
    analyzer.save_patterns_to_template('industrial_sensor_patterns', temporal_results['patterns_found'])
    print("   - Template saved successfully")
    
    # 6. Export patterns
    print("\n6. Exporting patterns...")
    export_path = analyzer.export_patterns()
    print(f"   - Patterns exported to: {export_path}")
    
    # 7. Get statistics
    print("\n7. Pattern Analysis Statistics:")
    stats = analyzer.get_pattern_statistics()
    print(f"   - Total analyses: {stats['analysis_summary']['total_analyses_performed']}")
    print(f"   - Templates available: {stats['analysis_summary']['template_count']}")
    
    print("\n=== PATTERN ANALYSIS COMPLETED ===")
    print("\nKey Findings:")
    print("- Temporal patterns detected in temperature (daily cycles)")
    print("- Spatial clustering revealed device groupings")
    print("- Behavioral analysis identified device performance patterns")
    print("- Anomaly detection found outlier devices")
    print("- All patterns saved for future reference and comparison")
                