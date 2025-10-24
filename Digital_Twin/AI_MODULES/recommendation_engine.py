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

# Machine Learning and Analytics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import networkx as nx

class RecommendationEngine:
    """
    Advanced recommendation engine for Digital Twin systems.

    Provides maintenance recommendations, optimization suggestions,
    operational improvements, and predictive actions based on
    system health, analytical patterns, historical data, and contextual rules.

    Attributes:
        config_path (Path): Path to the JSON configuration file.
        logger (logging.Logger): Logger instance for the engine.
        config (Dict): Loaded configuration from the JSON file.
        recommendation_history (deque): A history of generated recommendations.
        knowledge_base (Dict): Rule-based logic loaded from config.
        models (Dict): In-memory storage for ML models (not used in this version).
        scalers (Dict): In-memory storage for data scalers (not used in this version).
        scoring_weights (Dict): Weights for calculating composite scores.
        current_context (Dict): Current operational context (e.g., system load).
    """
    
    def __init__(self, config_path: str = "CONFIG/recommendation_config.json"):
        """
        Initializes the RecommendationEngine.

        Args:
            config_path (str): Path to the configuration file.
        """
        self.config_path = Path(config_path)
        self.logger = self._setup_logging()
        
        # Load configuration
        self.config = self._load_config()
        
        # Recommendation storage
        self.recommendation_history = deque(maxlen=self.config.get("history_maxlen", 1000))
        self.recommendation_templates = {}
        self.action_effectiveness = defaultdict(list)
        
        # Knowledge base for recommendations (now loaded from config)
        self.knowledge_base = self.config.get('knowledge_base', {})
        
        # Machine learning models for recommendations
        self.models = {}
        self.scalers = {}
        
        # Recommendation scoring weights
        self.scoring_weights = self.config.get('scoring_weights', {
            'urgency': 0.3,
            'impact': 0.25,
            'feasibility': 0.2,
            'cost_effectiveness': 0.15,
            'risk_reduction': 0.1
        })
        
        # Context for recommendations
        self.current_context = {}
        
    def _setup_logging(self) -> logging.Logger:
        """
        Setup logging for recommendation engine.
        
        Returns:
            logging.Logger: A configured logger instance.
        """
        logger = logging.getLogger('RecommendationEngine')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Ensure LOGS directory exists
            Path("LOGS").mkdir(exist_ok=True)
            handler = logging.FileHandler('LOGS/digital_twin_recommendations.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load recommendation engine configuration from the specified JSON file.
        If the file doesn't exist, create a default configuration.
        
        Returns:
            Dict[str, Any]: The loaded or default configuration dictionary.
        """
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                self.logger.info(f"Recommendation configuration loaded from {self.config_path}")
                return config
            else:
                self.logger.warning(f"Configuration file not found at {self.config_path}. Creating default config.")
                # Default configuration
                default_config = {
                    "history_maxlen": 1000,
                    "recommendation_types": {
                        "maintenance": {
                            "priority_weight": 0.8,
                            "time_horizon_days": 30,
                            "confidence_threshold": 0.7
                        },
                        "optimization": {
                            "priority_weight": 0.6,
                            "time_horizon_days": 90,
                            "confidence_threshold": 0.6
                        },
                        "operational": {
                            "priority_weight": 0.7,
                            "time_horizon_days": 7,
                            "confidence_threshold": 0.8
                        },
                        "preventive": {
                            "priority_weight": 0.5,
                            "time_horizon_days": 180,
                            "confidence_threshold": 0.5
                        }
                    },
                    "scoring_weights": {
                        "urgency": 0.3,
                        "impact": 0.25,
                        "feasibility": 0.2,
                        "cost_effectiveness": 0.15,
                        "risk_reduction": 0.1
                    },
                    "thresholds": {
                        "critical_score": 0.8,
                        "high_priority": 0.7,
                        "medium_priority": 0.5,
                        "low_priority": 0.3
                    },
                    "filters": {
                        "min_confidence": 0.4,
                        "max_recommendations": 20,
                        "duplicate_threshold_hours": 24
                    },
                    "knowledge_base": {
                        "maintenance_rules": {
                            "critical_maintenance": {
                                "conditions": {
                                    "AND": [
                                        {"key": "component_scores.maintenance.score", "op": "<", "value": 0.4}
                                    ]
                                },
                                "recommendations": [
                                    "Perform immediate inspection of maintenance-related components",
                                    "Check all fluid levels and critical seals",
                                    "Review maintenance logs for overdue tasks"
                                ],
                                "urgency": 0.9,
                                "category": "maintenance"
                            },
                            "low_efficiency": {
                                "conditions": {
                                    "AND": [
                                        {"key": "component_scores.efficiency.score", "op": "<", "value": 0.5}
                                    ]
                                },
                                "recommendations": [
                                    "Performance tuning required",
                                    "Check operational parameters for efficiency loss",
                                    "Inspect wear components related to efficiency"
                                ],
                                "urgency": 0.6,
                                "category": "optimization"
                            },
                            "degrading_performance": {
                                "conditions": {
                                    "AND": [
                                        {"key": "overall_score", "op": "<", "value": 0.7},
                                        {"key": "trend_analysis.trend_strength", "op": ">", "value": 0.05},
                                        {"key": "trend_analysis.trend_direction", "op": "==", "value": "degrading"}
                                    ]
                                },
                                "recommendations": [
                                    "Investigate root cause of performance degradation trend",
                                    "Analyze components with the lowest scores",
                                    "Update control algorithms to stabilize performance"
                                ],
                                "urgency": 0.7,
                                "category": "operational"
                            },
                            "high_system_load": {
                                "conditions": {
                                    "AND": [
                                        {"key": "system_load", "op": ">", "value": 0.9}
                                    ]
                                },
                                "recommendations": [
                                    "Implement load balancing strategies",
                                    "Identify and optimize high-load processes",
                                    "Consider scheduling non-critical tasks for off-peak hours"
                                ],
                                "urgency": 0.8,
                                "category": "operational"
                            },
                            "high_risk_factor": {
                                "conditions": {
                                    "AND": [
                                        {"key": "component_scores.reliability.score", "op": "<", "value": 0.6}
                                    ]
                                },
                                "recommendations": [
                                    "Inspect components related to reliability",
                                    "Implement redundancy checks",
                                    "Schedule preventive maintenance for high-risk components"
                                ],
                                "urgency": 0.7,
                                "category": "preventive"
                            },
                            "critical_system_state": {
                                "conditions": {
                                    "OR": [
                                        {"key": "overall_score", "op": "<", "value": 0.3},
                                        {"key": "health_status", "op": "==", "value": "critical"},
                                        {"key": "risk_assessment.overall_risk_level", "op": "==", "value": "critical"}
                                    ]
                                },
                                "recommendations": [
                                    "Initiate immediate system-wide safety check", 
                                    "Alert critical response team",
                                    "Review all active critical alerts"
                                ],
                                "urgency": 0.95,
                                "category": "emergency"
                            }
                        },
                        "optimization_patterns": {
                            "energy_efficiency": [
                                "Optimize operating schedules",
                                "Implement load balancing",
                                "Upgrade to efficient components"
                            ]
                        },
                        "operational_guidelines": {
                            "best_practices": [
                                "Regular system health checks",
                                "Maintain optimal operating conditions"
                            ]
                        }
                    }
                }
                
                # Save default configuration
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                
                self.logger.info(f"Default recommendation configuration created at {self.config_path}")
                return default_config
                
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}", exc_info=True)
            return {}

    def _flatten_dict(self, d: Union[Dict, List], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """
        Flattens a nested dictionary or list into a single-level dictionary.
        List items are keyed by their index.
        
        Example:
            _flatten_dict({'a': {'b': 1, 'c': [10, 20]}})
            Returns: {'a.b': 1, 'a.c.0': 10, 'a.c.1': 20}

        Args:
            d (Union[Dict, List]): The dictionary or list to flatten.
            parent_key (str): The prefix to prepend to keys.
            sep (str): The separator to use between nested keys.

        Returns:
            Dict[str, Any]: A flattened dictionary.
        """
        items = {}
        if isinstance(d, dict):
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, (dict, list)):
                    items.update(self._flatten_dict(v, new_key, sep=sep))
                else:
                    items[new_key] = v
        elif isinstance(d, list):
            for i, v in enumerate(d):
                new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
                if isinstance(v, (dict, list)):
                    items.update(self._flatten_dict(v, new_key, sep=sep))
                else:
                    items[new_key] = v
        return items

    def generate_recommendations(self, 
                                 health_data: Dict[str, Any],
                                 pattern_analysis: Optional[Dict[str, Any]] = None,
                                 historical_data: Optional[pd.DataFrame] = None,
                                 context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive recommendations based on system state and analysis.
        
        This is the main entry point for the engine. It orchestrates several
        sub-generators (health-based, pattern-based, etc.) and then
        scores, filters, and summarizes the results.
        
        Args:
            health_data (Dict[str, Any]): Health score and component analysis results
                                          from the HealthScoreCalculator.
            pattern_analysis (Optional[Dict[str, Any]]): Pattern analysis results
                                          from the PatternAnalyzer. Defaults to None.
            historical_data (Optional[pd.DataFrame]): Historical system data for
                                          trend analysis. Defaults to None.
            context (Optional[Dict[str, Any]]): Additional context information
                                          (e.g., system load, location). Defaults to None.
            
        Returns:
            Dict[str, Any]: A dictionary containing categorized recommendations,
                            a summary, and the context used.
        """
        try:
            self.logger.info("Generating system recommendations...")
            
            # Update context
            if context:
                self.current_context.update(context)
            
            recommendations: Dict[str, Any] = {
                'timestamp': datetime.now().isoformat(),
                'maintenance_recommendations': [],
                'optimization_recommendations': [],
                'operational_recommendations': [],
                'preventive_recommendations': [],
                'emergency_recommendations': [], # For critical, immediate actions
                'summary': {},
                'context': self.current_context.copy()
            }
            
            all_generated_recs: List[Dict[str, Any]] = []

            # 1. Health-based recommendations
            if health_data:
                all_generated_recs.extend(self._generate_health_based_recommendations(health_data))
            
            # 2. Pattern-based recommendations
            if pattern_analysis:
                all_generated_recs.extend(self._generate_pattern_based_recommendations(pattern_analysis))
            
            # 3. Historical data recommendations
            if historical_data is not None and not historical_data.empty:
                all_generated_recs.extend(self._generate_historical_recommendations(historical_data))
            
            # 4. Context-based recommendations
            if self.current_context:
                all_generated_recs.extend(self._generate_context_recommendations(self.current_context))
            
            # 5. Knowledge base recommendations (Dynamic Rule Engine)
            kb_recs = self._generate_knowledge_base_recommendations(
                health_data, pattern_analysis, self.current_context
            )
            all_generated_recs.extend(kb_recs)
            
            # 6. Categorize all recommendations
            self._categorize_recommendations(recommendations, all_generated_recs)
            
            # 7. Calculate composite scores and prioritize
            self._calculate_composite_scores(recommendations)
            
            # 8. Filter and limit recommendations
            self._filter_recommendations(recommendations)
            
            # 9. Generate summary
            recommendations['summary'] = self._generate_recommendations_summary(recommendations)
            
            # 10. Store recommendations for learning
            self._store_recommendations(recommendations)
            
            self.logger.info(f"Generated {self._count_total_recommendations(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendation generation error: {e}", exc_info=True)
            return self._create_error_result(f"Recommendation generation failed: {e}")
    
    def _generate_health_based_recommendations(self, health_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on health analysis data.

        Args:
            health_data (Dict[str, Any]): The health score analysis dictionary.

        Returns:
            List[Dict[str, Any]]: A list of recommendation dictionaries.
        """
        try:
            recommendations = []
            overall_score = health_data.get('overall_score', 1.0)
            health_status = health_data.get('health_status', 'good')
            
            # Critical health issues
            if overall_score < 0.4 or health_status == 'critical':
                recommendations.append({
                    'type': 'emergency',
                    'title': 'Critical System Health Detected',
                    'description': f'Overall system health critically low: {overall_score:.3f}',
                    'action': 'Immediate system inspection and intervention required',
                    'urgency': 0.95,
                    'impact': 0.9,
                    'timeframe': 'immediate',
                    'source': 'health_analysis',
                    'priority_level': 'critical'
                })
            
            # Component-specific recommendations
            component_scores = health_data.get('component_scores', {})
            for component, data in component_scores.items():
                score = data.get('score', 1.0)
                
                if score < 0.5:  # Poor component health
                    recommendations.append({
                        'type': 'maintenance',
                        'title': f'{component.title()} Performance Issue',
                        'description': f'{component.title()} score is {score:.3f} - below acceptable threshold',
                        'action': f'Inspect and service {component} components',
                        'urgency': 0.8,
                        'impact': 0.7,
                        'timeframe': 'within_week',
                        'component': component,
                        'source': 'health_analysis',
                        'priority_level': 'high'
                    })
                elif score < 0.7:  # Degrading component health
                    recommendations.append({
                        'type': 'preventive',
                        'title': f'{component.title()} Optimization Needed',
                        'description': f'{component.title()} score is {score:.3f} - preventive action recommended',
                        'action': f'Schedule preventive maintenance for {component}',
                        'urgency': 0.5,
                        'impact': 0.6,
                        'timeframe': 'within_month',
                        'component': component,
                        'source': 'health_analysis',
                        'priority_level': 'medium'
                    })
            
            # Trend-based recommendations
            trend_analysis = health_data.get('trend_analysis', {})
            trend_direction = trend_analysis.get('trend_direction')
            
            if trend_direction == 'degrading':
                trend_strength = trend_analysis.get('trend_strength', 0)
                if trend_strength > 0.05:  # Strong degrading trend
                    recommendations.append({
                        'type': 'operational',
                        'title': 'System Performance Degradation Trend',
                        'description': f'System showing degrading trend with strength {trend_strength:.3f}',
                        'action': 'Investigate root causes of performance degradation',
                        'urgency': 0.7,
                        'impact': 0.8,
                        'timeframe': 'within_week',
                        'source': 'trend_analysis',
                        'priority_level': 'high'
                    })
            
            # Risk-based recommendations
            risk_assessment = health_data.get('risk_assessment', {})
            overall_risk = risk_assessment.get('overall_risk_level')
            
            if overall_risk in ['high', 'critical']:
                risk_factors = risk_assessment.get('risk_factors', [])
                for risk_factor in risk_factors[:3]:  # Top 3 risk factors
                    component = risk_factor.get('component', 'unknown')
                    risk_level = risk_factor.get('risk_level', 'medium')
                    impact = risk_factor.get('impact', 'performance_degradation')
                    
                    recommendations.append({
                        'type': 'maintenance',
                        'title': f'High Risk: {component.title()} Issue',
                        'description': f'{component.title()} presents {risk_level} risk of {impact}',
                        'action': f'Address {component} risk factors immediately',
                        'urgency': 0.8 if risk_level == 'high' else 0.9,
                        'impact': 0.8,
                        'timeframe': 'within_week' if risk_level == 'high' else 'immediate',
                        'component': component,
                        'source': 'risk_analysis',
                        'priority_level': 'critical' if risk_level == 'critical' else 'high'
                    })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Health-based recommendation generation error: {e}", exc_info=True)
            return []
    
    def _generate_pattern_based_recommendations(self, pattern_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on pattern analysis results.

        Args:
            pattern_analysis (Dict[str, Any]): The results from the PatternAnalyzer.

        Returns:
            List[Dict[str, Any]]: A list of recommendation dictionaries.
        """
        try:
            recommendations = []
            
            # Temporal pattern recommendations
            temporal = pattern_analysis.get('temporal', {})
            patterns_found = temporal.get('patterns_found', {})
            
            for metric, patterns in patterns_found.items():
                # Cyclical patterns
                cyclical = patterns.get('cyclical', {})
                if cyclical.get('cycles_detected', 0) > 0:
                    cycle_info = cyclical.get('dominant_cycle', {})
                    cycle_length = cycle_info.get('cycle_length', 0)
                    strength = cycle_info.get('strength', 0)
                    
                    if strength > 0.7 and cycle_length == 24:  # Strong daily cycle
                        recommendations.append({
                            'type': 'optimization',
                            'title': f'{metric.title()} Daily Cycle Optimization',
                            'description': f'Strong daily pattern detected in {metric} (strength: {strength:.2f})',
                            'action': f'Optimize {metric} schedule based on daily patterns',
                            'urgency': 0.4,
                            'impact': 0.6,
                            'timeframe': 'within_month',
                            'source': 'pattern_analysis',
                            'priority_level': 'medium'
                        })
                    elif strength > 0.6 and cycle_length == 168:  # Strong weekly cycle
                        recommendations.append({
                            'type': 'optimization',
                            'title': f'{metric.title()} Weekly Cycle Optimization',
                            'description': f'Strong weekly pattern detected in {metric} (strength: {strength:.2f})',
                            'action': f'Adjust {metric} based on weekly operational patterns',
                            'urgency': 0.3,
                            'impact': 0.5,
                            'timeframe': 'within_month',
                            'source': 'pattern_analysis',
                            'priority_level': 'low'
                        })
                
                # Change point detection
                change_points = patterns.get('change_points', {})
                if change_points.get('change_points_detected', 0) > 2:
                    recommendations.append({
                        'type': 'operational',
                        'title': f'{metric.title()} Instability Detected',
                        'description': f'Multiple change points detected in {metric}',
                        'action': f'Investigate causes of {metric} variability',
                        'urgency': 0.6,
                        'impact': 0.5,
                        'timeframe': 'within_week',
                        'source': 'pattern_analysis',
                        'priority_level': 'medium'
                    })
            
            # Behavioral pattern recommendations
            behavioral = pattern_analysis.get('behavioral', {})
            behavioral_patterns = behavioral.get('behavioral_patterns', {})
            
            # Anomaly patterns
            anomalies = behavioral_patterns.get('anomalies', {})
            if anomalies.get('anomaly_detection') and anomalies.get('anomalous_entities_count', 0) > 0:
                count = anomalies.get('anomalous_entities_count')
                recommendations.append({
                    'type': 'maintenance',
                    'title': 'Behavioral Anomalies Detected',
                    'description': f'{count} anomalous behavioral patterns identified',
                    'action': 'Investigate and address anomalous system behaviors',
                    'urgency': 0.7,
                    'impact': 0.6,
                    'timeframe': 'within_week',
                    'source': 'behavioral_analysis',
                    'priority_level': 'high'
                })
            
            # Clustering patterns
            clustering = behavioral_patterns.get('clustering', {})
            if clustering.get('clusters_found', 0) > 1:
                cluster_count = clustering.get('clusters_found')
                recommendations.append({
                    'type': 'optimization',
                    'title': 'Operational Clustering Patterns Identified',
                    'description': f'{cluster_count} distinct operational clusters detected',
                    'action': 'Analyze clusters to optimize operational strategies',
                    'urgency': 0.4,
                    'impact': 0.5,
                    'timeframe': 'within_month',
                    'source': 'clustering_analysis',
                    'priority_level': 'medium'
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Pattern-based recommendation generation error: {e}", exc_info=True)
            return []
    
    def _generate_historical_recommendations(self, historical_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on historical data analysis.

        Args:
            historical_data (pd.DataFrame): DataFrame of historical data.

        Returns:
            List[Dict[str, Any]]: A list of recommendation dictionaries.
        """
        try:
            recommendations = []
            
            # Analyze data quality
            data_quality = self._assess_historical_data_quality(historical_data)
            if data_quality.get('missing_percentage', 0) > 20:
                recommendations.append({
                    'type': 'operational',
                    'title': 'Data Quality Issues Detected',
                    'description': f'Historical data has {data_quality["missing_percentage"]:.1f}% missing values',
                    'action': 'Improve data collection and sensor reliability',
                    'urgency': 0.5,
                    'impact': 0.7,
                    'timeframe': 'within_month',
                    'source': 'data_analysis',
                    'priority_level': 'medium'
                })
            
            # Analyze trends in key metrics
            for column in historical_data.select_dtypes(include=[np.number]).columns:
                if column in ['timestamp', 'id']:
                    continue
                
                series = historical_data[column].dropna()
                if len(series) < 10:  # Not enough data
                    continue
                
                # Calculate trend
                trend = self._calculate_simple_trend(series)
                
                # Check for concerning trends
                if abs(trend) > 0.1:  # Significant trend
                    if 'efficiency' in column.lower() and trend < -0.05:
                        recommendations.append({
                            'type': 'maintenance',
                            'title': f'Declining {column.title()} Trend',
                            'description': f'{column.title()} showing declining trend over time',
                            'action': f'Investigate and address {column} degradation',
                            'urgency': 0.7,
                            'impact': 0.8,
                            'timeframe': 'within_week',
                            'source': 'historical_analysis',
                            'priority_level': 'high'
                        })
                    elif any(term in column.lower() for term in ['temperature', 'vibration', 'pressure']) and trend > 0.05:
                        recommendations.append({
                            'type': 'maintenance',
                            'title': f'Increasing {column.title()} Trend',
                            'description': f'{column.title()} showing increasing trend - potential issue',
                            'action': f'Monitor and investigate {column} increase',
                            'urgency': 0.6,
                            'impact': 0.7,
                            'timeframe': 'within_week',
                            'source': 'historical_analysis',
                            'priority_level': 'medium'
                        })
            
            # Analyze historical failures
            failure_recommendations = self._analyze_historical_failures(historical_data)
            recommendations.extend(failure_recommendations)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Historical recommendation generation error: {e}", exc_info=True)
            return []
    
    def _generate_context_recommendations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on operational context.

        Args:
            context (Dict[str, Any]): The current operational context.

        Returns:
            List[Dict[str, Any]]: A list of recommendation dictionaries.
        """
        try:
            recommendations = []
            
            # High system load recommendations
            system_load = context.get('system_load', 0)
            if system_load > 0.9:
                recommendations.append({
                    'type': 'operational',
                    'title': 'High System Load Detected',
                    'description': f'System load at {system_load*100:.1f}% - approaching capacity',
                    'action': 'Consider load balancing or capacity expansion',
                    'urgency': 0.8,
                    'impact': 0.7,
                    'timeframe': 'immediate',
                    'source': 'context_analysis',
                    'priority_level': 'high'
                })
            elif system_load > 0.8:
                recommendations.append({
                    'type': 'optimization',
                    'title': 'High System Load Warning',
                    'description': f'System load at {system_load*100:.1f}% - optimization recommended',
                    'action': 'Optimize system performance to reduce load',
                    'urgency': 0.6,
                    'impact': 0.6,
                    'timeframe': 'within_week',
                    'source': 'context_analysis',
                    'priority_level': 'medium'
                })
            
            # Maintenance scheduling recommendations
            last_maintenance = context.get('last_maintenance')
            if last_maintenance:
                try:
                    last_date = datetime.fromisoformat(last_maintenance)
                    days_since = (datetime.now() - last_date).days
                    
                    if days_since > 90:  # More than 3 months
                        recommendations.append({
                            'type': 'maintenance',
                            'title': 'Scheduled Maintenance Overdue',
                            'description': f'Last maintenance was {days_since} days ago',
                            'action': 'Schedule routine maintenance inspection',
                            'urgency': 0.7,
                            'impact': 0.6,
                            'timeframe': 'within_week',
                            'source': 'maintenance_schedule',
                            'priority_level': 'high'
                        })
                    elif days_since > 60:  # More than 2 months
                        recommendations.append({
                            'type': 'preventive',
                            'title': 'Maintenance Due Soon',
                            'description': f'Last maintenance was {days_since} days ago - approaching due date',
                            'action': 'Plan upcoming maintenance activities',
                            'urgency': 0.4,
                            'impact': 0.5,
                            'timeframe': 'within_month',
                            'source': 'maintenance_schedule',
                            'priority_level': 'medium'
                        })
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Could not parse last_maintenance date '{last_maintenance}': {e}")
            
            # Operating hours recommendations
            operating_hours = context.get('operating_hours', 0)
            if operating_hours > 10000:  # Very high usage
                recommendations.append({
                    'type': 'maintenance',
                    'title': 'Very High Operating Hours',
                    'description': f'System has {operating_hours} operating hours - major service recommended',
                    'action': 'Schedule major maintenance service',
                    'urgency': 0.7,
                    'impact': 0.8,
                    'timeframe': 'within_week',
                    'source': 'usage_analysis',
                    'priority_level': 'high'
                })
            elif operating_hours > 8000:  # High usage
                recommendations.append({
                    'type': 'preventive',
                    'title': 'High Operating Hours',
                    'description': f'System has {operating_hours} operating hours',
                    'action': 'Consider comprehensive system inspection',
                    'urgency': 0.5,
                    'impact': 0.7,
                    'timeframe': 'within_month',
                    'source': 'usage_analysis',
                    'priority_level': 'medium'
                })
            
            # Environmental context
            location = context.get('location', '')
            if 'harsh' in location.lower() or 'extreme' in location.lower():
                recommendations.append({
                    'type': 'preventive',
                    'title': 'Harsh Environment Operation',
                    'description': f'System operating in challenging environment: {location}',
                    'action': 'Increase monitoring and maintenance frequency for harsh conditions',
                    'urgency': 0.6,
                    'impact': 0.7,
                    'timeframe': 'within_month',
                    'source': 'environmental_analysis',
                    'priority_level': 'medium'
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Context recommendation generation error: {e}", exc_info=True)
            return []
    
    def _generate_knowledge_base_recommendations(self, 
                                               health_data: Optional[Dict[str, Any]], 
                                               pattern_analysis: Optional[Dict[str, Any]], 
                                               context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on knowledge base rules loaded from config.
        Uses a flattened data context and a recursive rule evaluator.

        Args:
            health_data: The health score analysis dictionary.
            pattern_analysis: The results from the PatternAnalyzer.
            context: The current operational context.

        Returns:
            List[Dict[str, Any]]: A list of recommendation dictionaries.
        """
        try:
            recommendations = []
            rules = self.knowledge_base.get('maintenance_rules', {})
            
            # Create a comprehensive, flattened data context
            # This makes *all* data from all sources available to the rule engine
            data_context: Dict[str, Any] = {}
            if health_data:
                data_context.update(self._flatten_dict(health_data))
            if pattern_analysis:
                data_context.update(self._flatten_dict(pattern_analysis))
            if context:
                # Context is assumed to be flat, so just update
                data_context.update(context)
            
            if not data_context:
                self.logger.info("Knowledge base check skipped: no data context provided.")
                return []
                
            # Evaluate each rule from the knowledge base
            for rule_name, rule_config in rules.items():
                conditions = rule_config.get('conditions', {})
                
                # Call the enhanced recursive rule evaluator
                rule_triggered = self._evaluate_rule(conditions, data_context)
                
                if rule_triggered:
                    rule_recommendations = rule_config.get('recommendations', [])
                    urgency = rule_config.get('urgency', 0.5)
                    category = rule_config.get('category', 'maintenance')
                    
                    for rec_text in rule_recommendations:
                        recommendations.append({
                            'type': category,
                            'title': f'{rule_name.replace("_", " ").title()} Action Required',
                            'description': f'Knowledge base rule triggered: {rule_name}',
                            'action': rec_text,
                            'urgency': urgency,
                            'impact': 0.7,  # Default impact for KB rules
                            'timeframe': 'within_week' if urgency > 0.7 else 'within_month',
                            'source': 'knowledge_base',
                            'priority_level': 'high' if urgency > 0.7 else 'medium'
                        })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Knowledge base recommendation generation error: {e}", exc_info=True)
            return []

    def _evaluate_rule(self, conditions: Dict[str, Any], data_context: Dict[str, Any]) -> bool:
        """
        Recursively evaluate a set of conditions (AND/OR) against the data context.

        Args:
            conditions (Dict[str, Any]): The condition block (e.g., {'AND': [...]}).
            data_context (Dict[str, Any]): The flattened data context.

        Returns:
            bool: True if the condition block evaluates to true, False otherwise.
        """
        try:
            if "AND" in conditions:
                # All conditions in the list must be true
                if not conditions["AND"]: return False # Empty AND is false
                return all(self._evaluate_rule(cond, data_context) for cond in conditions["AND"])
            
            elif "OR" in conditions:
                # Any condition in the list can be true
                if not conditions["OR"]: return False # Empty OR is false
                return any(self._evaluate_rule(cond, data_context) for cond in conditions["OR"])
            
            elif "key" in conditions:
                # This is a base condition to check
                return self._check_condition(conditions, data_context)
            
            else:
                # Malformed rule structure
                self.logger.warning(f"Skipping malformed rule condition block: {conditions}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error evaluating rule block '{conditions}': {e}", exc_info=True)
            return False

    def _check_condition(self, condition: Dict[str, Any], data_context: Dict[str, Any]) -> bool:
        """
        Check a single condition (e.g., {"key": "x", "op": ">", "value": 5}).
        Handles type conversion for numeric and string comparisons.

        Args:
            condition (Dict[str, Any]): The single condition dictionary.
            data_context (Dict[str, Any]): The flattened data context.

        Returns:
            bool: True if the condition is met, False otherwise.
        """
        key = condition.get("key")
        op = condition.get("op")
        rule_value = condition.get("value")

        if key is None or op is None or rule_value is None:
            self.logger.warning(f"Skipping malformed condition: {condition}")
            return False

        actual_value = data_context.get(key)
        
        if actual_value is None:
            # Data required by the rule is not available in the context
            # self.logger.debug(f"Rule key '{key}' not in data context.")
            return False
        
        # --- Robust Comparison Logic ---
        try:
            # Try numeric comparison first
            num_actual = float(actual_value)
            num_rule = float(rule_value)
            
            if op == '<':
                return num_actual < num_rule
            if op == '>':
                return num_actual > num_rule
            if op == '<=':
                return num_actual <= num_rule
            if op == '>=':
                return num_actual >= num_rule
            if op == '==':
                # Use np.isclose for float equality
                return np.isclose(num_actual, num_rule)
            if op == '!=':
                return not np.isclose(num_actual, num_rule)

        except (ValueError, TypeError):
            # Fallback to string comparison for '==' and '!='
            str_actual = str(actual_value)
            str_rule = str(rule_value)
            
            if op == '==':
                return str_actual == str_rule
            if op == '!=':
                return str_actual != str_rule
            
            # Cannot perform <, > on non-numeric types
            self.logger.warning(
                f"Cannot perform numeric op '{op}' on non-numeric values: "
                f"'{key}' (Value: {actual_value}, Type: {type(actual_value)}) "
                f"vs Rule Value: {rule_value} (Type: {type(rule_value)})"
            )
            return False
        
        self.logger.warning(f"Unknown operator in rule: {op}")
        return False
        
    def _categorize_recommendations(self, recommendations: Dict[str, Any], new_recommendations: List[Dict[str, Any]]):
        """
        Categorize new recommendations into the main recommendations dictionary.

        Args:
            recommendations (Dict[str, Any]): The main dictionary to update.
            new_recommendations (List[Dict[str, Any]]): The list of new recommendations.
        """
        try:
            for rec in new_recommendations:
                rec_type = rec.get('type', 'operational')
                category_key = f"{rec_type}_recommendations"
                
                if category_key in recommendations:
                    recommendations[category_key].append(rec)
                else:
                    # Fallback to operational if type is unknown
                    recommendations['operational_recommendations'].append(rec)
                    
        except Exception as e:
            self.logger.error(f"Recommendation categorization error: {e}", exc_info=True)
    
    def _calculate_composite_scores(self, recommendations: Dict[str, Any]):
        """
        Calculate composite scores for all recommendations in place.

        Args:
            recommendations (Dict[str, Any]): The main recommendations dictionary.
        """
        try:
            all_categories = [
                'emergency_recommendations', 'maintenance_recommendations',
                'optimization_recommendations', 'operational_recommendations',
                'preventive_recommendations'
            ]
            
            thresholds = self.config.get('thresholds', {})
            
            for category in all_categories:
                for rec in recommendations.get(category, []):
                    # Get individual scores
                    urgency = rec.get('urgency', 0.5)
                    impact = rec.get('impact', 0.5)
                    
                    # Estimate missing scores based on type and urgency
                    feasibility = rec.get('feasibility', 0.8 if rec.get('type') in ['operational', 'optimization'] else 0.6)
                    cost_effectiveness = rec.get('cost_effectiveness', 0.7 if urgency > 0.7 else 0.5)
                    risk_reduction = rec.get('risk_reduction', urgency * 0.8)
                    
                    # Calculate composite score
                    composite_score = (
                        urgency * self.scoring_weights.get('urgency', 0.3) + 
                        impact * self.scoring_weights.get('impact', 0.25) + 
                        feasibility * self.scoring_weights.get('feasibility', 0.2) + 
                        cost_effectiveness * self.scoring_weights.get('cost_effectiveness', 0.15) + 
                        risk_reduction * self.scoring_weights.get('risk_reduction', 0.1)
                    )
                    
                    # Update recommendation dictionary in-place
                    rec['composite_score'] = round(composite_score, 3)
                    rec['feasibility'] = feasibility
                    rec['cost_effectiveness'] = cost_effectiveness
                    rec['risk_reduction'] = risk_reduction
                    
                    # Set priority level if not already set
                    if 'priority_level' not in rec:
                        if composite_score >= thresholds.get('critical_score', 0.8):
                            rec['priority_level'] = 'critical'
                        elif composite_score >= thresholds.get('high_priority', 0.7):
                            rec['priority_level'] = 'high'
                        elif composite_score >= thresholds.get('medium_priority', 0.5):
                            rec['priority_level'] = 'medium'
                        else:
                            rec['priority_level'] = 'low'
                    
                    # Set timeframe if not already set
                    if 'timeframe' not in rec:
                        if rec['priority_level'] == 'critical':
                            rec['timeframe'] = 'immediate'
                        elif rec['priority_level'] == 'high':
                            rec['timeframe'] = 'within_week'
                        elif rec['priority_level'] == 'medium':
                            rec['timeframe'] = 'within_month'
                        else:
                            rec['timeframe'] = 'within_quarter'
                
                # Sort recommendations by composite score
                recommendations[category] = sorted(
                    recommendations.get(category, []), 
                    key=lambda x: x.get('composite_score', 0), 
                    reverse=True
                )
                
        except Exception as e:
            self.logger.error(f"Composite score calculation error: {e}", exc_info=True)
    
    def _filter_recommendations(self, recommendations: Dict[str, Any]):
        """
        Filter and limit recommendations based on configuration.

        Args:
            recommendations (Dict[str, Any]): The main recommendations dictionary.
        """
        try:
            filters = self.config.get('filters', {})
            min_confidence = filters.get('min_confidence', 0.4)
            max_recommendations = filters.get('max_recommendations', 20)
            
            all_categories = [
                'emergency_recommendations', 'maintenance_recommendations',
                'optimization_recommendations', 'operational_recommendations',
                'preventive_recommendations'
            ]
            
            for category in all_categories:
                category_recs = recommendations.get(category, [])
                
                # Filter by confidence (using composite_score as proxy)
                filtered_recs = [rec for rec in category_recs if rec.get('composite_score', 0) >= min_confidence]
                
                # Limit number of recommendations per category
                # (More sophisticated logic could limit total while preserving priority)
                category_limit = max(1, max_recommendations // len(all_categories))
                filtered_recs = filtered_recs[:category_limit]
                
                recommendations[category] = filtered_recs
                
        except Exception as e:
            self.logger.error(f"Recommendation filtering error: {e}", exc_info=True)
    
    def _generate_recommendations_summary(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of the generated recommendations.

        Args:
            recommendations (Dict[str, Any]): The main recommendations dictionary.

        Returns:
            Dict[str, Any]: A summary dictionary.
        """
        try:
            total_recommendations = self._count_total_recommendations(recommendations)
            
            # Count by category
            category_counts = {
                'emergency': len(recommendations.get('emergency_recommendations', [])),
                'maintenance': len(recommendations.get('maintenance_recommendations', [])),
                'optimization': len(recommendations.get('optimization_recommendations', [])),
                'operational': len(recommendations.get('operational_recommendations', [])),
                'preventive': len(recommendations.get('preventive_recommendations', []))
            }
            
            # Priority distribution
            priority_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
            
            all_recs = []
            for category in ['emergency_recommendations', 'maintenance_recommendations', 
                           'optimization_recommendations', 'operational_recommendations',
                           'preventive_recommendations']:
                all_recs.extend(recommendations.get(category, []))
            
            for rec in all_recs:
                priority = rec.get('priority_level', 'low')
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
            # Top recommendations
            all_recs.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
            top_recommendations = all_recs[:5]
            
            # Urgency analysis
            urgent_count = len([rec for rec in all_recs if rec.get('urgency', 0) > 0.7])
            
            return {
                'total_recommendations': total_recommendations,
                'category_distribution': category_counts,
                'priority_distribution': priority_counts,
                'urgent_recommendations': urgent_count,
                'top_recommendations': [
                    {
                        'title': rec.get('title', 'Unknown'),
                        'priority': rec.get('priority_level', 'low'),
                        'score': rec.get('composite_score', 0),
                        'timeframe': rec.get('timeframe', 'unknown')
                    }
                    for rec in top_recommendations
                ],
                'next_actions': self._identify_next_actions(all_recs),
                'estimated_impact': self._estimate_overall_impact(all_recs)
            }
            
        except Exception as e:
            self.logger.error(f"Recommendations summary generation error: {e}", exc_info=True)
            return {}
    
    def _count_total_recommendations(self, recommendations: Dict[str, Any]) -> int:
        """Count total number of recommendations across all categories."""
        try:
            total = 0
            categories = ['emergency_recommendations', 'maintenance_recommendations',
                        'optimization_recommendations', 'operational_recommendations',
                        'preventive_recommendations']
            
            for category in categories:
                total += len(recommendations.get(category, []))
            
            return total
            
        except Exception as e:
            self.logger.error(f"Recommendation counting error: {e}", exc_info=True)
            return 0
    
    def _identify_next_actions(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify immediate next actions from recommendations."""
        try:
            # Sort by urgency and impact
            urgent_recs = [rec for rec in recommendations 
                           if rec.get('urgency', 0) > 0.7 or rec.get('priority_level') == 'critical']
            
            urgent_recs.sort(key=lambda x: (x.get('urgency', 0), x.get('impact', 0)), reverse=True)
            
            next_actions = []
            for rec in urgent_recs[:3]:  # Top 3 urgent actions
                next_actions.append({
                    'action': rec.get('action', 'Unknown action'),
                    'timeframe': rec.get('timeframe', 'unknown'),
                    'priority': rec.get('priority_level', 'low'),
                    'category': rec.get('type', 'operational')
                })
            
            return next_actions
            
        except Exception as e:
            self.logger.error(f"Next actions identification error: {e}", exc_info=True)
            return []
    
    def _estimate_overall_impact(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate overall impact of implementing recommendations."""
        try:
            if not recommendations:
                return {'estimated_improvement': 0, 'confidence': 0, 'impact_areas': []}
            
            # Calculate weighted impact
            total_weighted_impact = 0
            total_weight = 0
            
            for rec in recommendations:
                impact = rec.get('impact', 0.5)
                urgency = rec.get('urgency', 0.5)
                weight = urgency  # Use urgency as weight
                
                total_weighted_impact += impact * weight
                total_weight += weight
            
            estimated_improvement = total_weighted_impact / total_weight if total_weight > 0 else 0
            
            # Estimate confidence based on number of recommendations and their consistency
            confidence = min(0.9, 0.5 + (len(recommendations) * 0.05))
            
            return {
                'estimated_improvement': round(estimated_improvement, 3),
                'confidence': round(confidence, 2),
                'impact_areas': self._identify_impact_areas(recommendations)
            }
            
        except Exception as e:
            self.logger.error(f"Overall impact estimation error: {e}", exc_info=True)
            return {'estimated_improvement': 0, 'confidence': 0, 'impact_areas': []}
    
    def _identify_impact_areas(self, recommendations: List[Dict[str, Any]]) -> List[str]:
        """Identify main areas of impact from recommendations."""
        try:
            impact_areas = set()
            
            for rec in recommendations:
                rec_type = rec.get('type', 'operational')
                component = rec.get('component', '')
                
                if component:
                    impact_areas.add(component.title())
                else:
                    impact_areas.add(rec_type.title())
            
            return list(impact_areas)
            
        except Exception as e:
            self.logger.error(f"Impact areas identification error: {e}", exc_info=True)
            return []
    
    def _store_recommendations(self, recommendations: Dict[str, Any]):
        """Store recommendations in history for learning."""
        try:
            # Create a serializable copy of the recommendations
            serializable_recs = json.loads(json.dumps(recommendations, default=str))

            history_entry = {
                'timestamp': serializable_recs['timestamp'],
                'recommendations': serializable_recs,
                'context': serializable_recs.get('context', {}),
                'total_count': self._count_total_recommendations(serializable_recs)
            }
            
            self.recommendation_history.append(history_entry)
            
        except Exception as e:
            self.logger.error(f"Recommendation storage error: {e}", exc_info=True)
    
    def _assess_historical_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess quality of historical data."""
        try:
            total_cells = data.size
            missing_cells = data.isnull().sum().sum()
            missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
            
            return {
                'missing_percentage': missing_percentage,
                'total_records': len(data),
                'completeness_score': 1.0 - (missing_percentage / 100)
            }
            
        except Exception as e:
            self.logger.error(f"Data quality assessment error: {e}", exc_info=True)
            return {'missing_percentage': 0, 'total_records': 0, 'completeness_score': 0.0}
    
    def _calculate_simple_trend(self, data: pd.Series) -> float:
        """Calculate simple trend coefficient."""
        try:
            if len(data) < 3:
                return 0.0
            
            x = np.arange(len(data))
            coeffs = np.polyfit(x, data.values, 1)
            
            # Normalize by data range
            data_range = data.max() - data.min()
            normalized_trend = coeffs[0] / data_range if data_range > 1e-6 else 0
            
            return float(normalized_trend)
            
        except (np.linalg.LinAlgError, ValueError, TypeError) as e:
            self.logger.error(f"Trend calculation error: {e}", exc_info=True)
            return 0.0
    
    def _analyze_historical_failures(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze historical data for failure patterns."""
        try:
            recommendations = []
            
            # Look for failure indicators
            failure_indicators = ['error', 'fault', 'alarm', 'failure', 'down']
            
            for col in data.columns:
                if any(indicator in col.lower() for indicator in failure_indicators):
                    # Ensure column is numeric for .sum()
                    if pd.api.types.is_numeric_dtype(data[col]):
                        col_data = data[col].dropna()
                        
                        if len(col_data) > 0 and col_data.sum() > 0:  # Failures detected
                            failure_rate = col_data.mean()
                            
                            if failure_rate > 0.1:  # High failure rate
                                recommendations.append({
                                    'type': 'maintenance',
                                    'title': f'High {col} Rate Detected',
                                    'description': f'Historical data shows {failure_rate*100:.1f}% {col} rate',
                                    'action': f'Investigate and address root causes of {col}',
                                    'urgency': 0.8,
                                    'impact': 0.7,
                                    'timeframe': 'within_week',
                                    'source': 'failure_analysis',
                                    'priority_level': 'high'
                                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Historical failure analysis error: {e}", exc_info=True)
            return []
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result structure."""
        return {
            'error': True,
            'message': error_message,
            'timestamp': datetime.now().isoformat(),
            'maintenance_recommendations': [],
            'optimization_recommendations': [],
            'operational_recommendations': [],
            'preventive_recommendations': [],
            'emergency_recommendations': []
        }
    
    def update_recommendation_effectiveness(self, recommendation_id: str, effectiveness_score: float):
        """
        Update effectiveness score for a recommendation (feedback loop).

        Args:
            recommendation_id (str): The unique ID of the recommendation.
            effectiveness_score (float): A score (e.g., 0.0 to 1.0) indicating
                                         how effective the action was.
        """
        try:
            self.action_effectiveness[recommendation_id].append({
                'effectiveness_score': effectiveness_score,
                'timestamp': datetime.now().isoformat()
            })
            
            self.logger.info(f"Updated effectiveness for recommendation {recommendation_id}: {effectiveness_score}")
            
        except Exception as e:
            self.logger.error(f"Effectiveness update error: {e}", exc_info=True)
    
    def get_recommendation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get historical recommendations.

        Args:
            limit (int): The maximum number of historical entries to return.

        Returns:
            List[Dict[str, Any]]: A list of historical recommendation entries.
        """
        try:
            history = list(self.recommendation_history)
            history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return history[:limit]
            
        except Exception as e:
            self.logger.error(f"Recommendation history retrieval error: {e}", exc_info=True)
            return []
    
    def export_recommendations(self, recommendations: Dict[str, Any], output_path: str = None) -> str:
        """
        Export recommendations to a JSON file.

        Args:
            recommendations (Dict[str, Any]): The recommendations dictionary to export.
            output_path (str, optional): The file path to save to. Defaults to None.

        Returns:
            str: The path to the saved file.
        """
        try:
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_dir = Path("REPORTS")
                output_dir.mkdir(exist_ok=True)
                output_path_obj = output_dir / f"recommendations_{timestamp}.json"
            else:
                output_path_obj = Path(output_path)
                
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path_obj, 'w') as f:
                json.dump(recommendations, f, indent=2, default=str)
            
            self.logger.info(f"Recommendations exported to {output_path_obj}")
            return str(output_path_obj)
            
        except Exception as e:
            self.logger.error(f"Recommendation export error: {e}", exc_info=True)
            raise
    
    def get_recommendation_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive recommendation statistics from history.

        Returns:
            Dict[str, Any]: A dictionary of statistics.
        """
        try:
            if not self.recommendation_history:
                return {'total_sessions': 0, 'statistics': {}}
            
            total_recs = sum(entry['total_count'] for entry in self.recommendation_history)
            
            # Category statistics
            category_stats = defaultdict(int)
            priority_stats = defaultdict(int)
            
            for entry in self.recommendation_history:
                recs = entry.get('recommendations', {})
                for category in ['emergency_recommendations', 'maintenance_recommendations',
                               'optimization_recommendations', 'operational_recommendations', 
                               'preventive_recommendations']:
                    category_recs = recs.get(category, [])
                    category_name = category.replace('_recommendations', '')
                    category_stats[category_name] += len(category_recs)
                    
                    for rec in category_recs:
                        priority = rec.get('priority_level', 'low')
                        priority_stats[priority] += 1
            
            # Effectiveness statistics
            effectiveness_data = []
            for rec_id, effectiveness_list in self.action_effectiveness.items():
                if effectiveness_list:
                    avg_effectiveness = np.mean([e['effectiveness_score'] for e in effectiveness_list])
                    effectiveness_data.append(avg_effectiveness)
            
            avg_effectiveness = np.mean(effectiveness_data) if effectiveness_data else 0
            
            return {
                'total_sessions': len(self.recommendation_history),
                'total_recommendations': total_recs,
                'average_per_session': total_recs / len(self.recommendation_history) if self.recommendation_history else 0,
                'category_distribution': dict(category_stats),
                'priority_distribution': dict(priority_stats),
                'average_effectiveness': round(avg_effectiveness, 3),
                'effectiveness_samples': len(effectiveness_data),
                'recent_activity': self.recommendation_history[-5:] if len(self.recommendation_history) >= 5 else list(self.recommendation_history)
            }
            
        except Exception as e:
            self.logger.error(f"Recommendation statistics error: {e}", exc_info=True)
            return {}


# Example usage and testing
if __name__ == "__main__":
    """
    Test scenarios for the RecommendationEngine.
    This block is executed when the script is run directly.
    It demonstrates how to use the engine and validates its output.
    """
    
    # Initialize recommendation engine
    # This will create a 'CONFIG/recommendation_config.json' if it doesn't exist.
    print("Initializing RecommendationEngine...")
    rec_engine = RecommendationEngine()
    
    # === Test Scenario 1: Critical Failure Imminent ===
    print("\n" + "="*30)
    print("=== Test Scenario 1: Critical Failure Imminent ===")
    print("="*30)
    
    # This data should trigger multiple 'critical' and 'emergency' rules
    sample_health_data_critical = {
        'overall_score': 0.25, # Triggers 'critical_system_state' (OR)
        'health_status': 'critical', # Triggers 'critical_system_state' (OR)
        'component_scores': {
            'performance': {'score': 0.4},
            'reliability': {'score': 0.5},  # Triggers 'high_risk_factor'
            'efficiency': {'score': 0.3},  # Triggers 'low_efficiency'
            'safety': {'score': 0.9},
            'maintenance': {'score': 0.1}   # Triggers 'critical_maintenance'
        },
        'trend_analysis': {
            'trend_direction': 'degrading', # Triggers 'degrading_performance'
            'trend_strength': 0.15
        },
        'risk_assessment': {
            'overall_risk_level': 'critical', # Triggers 'critical_system_state' (OR)
            'risk_factors': [
                {'component': 'maintenance', 'risk_level': 'critical', 'impact': 'system_failure'}
            ]
        }
    }
    
    sample_pattern_analysis_critical = {
        'behavioral': {
            'behavioral_patterns': {
                'anomalies': {'anomaly_detection': True, 'anomalous_entities_count': 5}
            }
        }
    }
    
    sample_context_critical = {
        'system_load': 0.98, # Triggers 'high_system_load'
        'operating_hours': 12000, # Triggers 'Very High Operating Hours'
        'last_maintenance': (datetime.now() - timedelta(days=100)).isoformat(), # Triggers 'Overdue'
        'location': 'harsh_environment_zone' # Triggers 'Harsh Environment'
    }
    
    print("1. Generating recommendations for CRITICAL state...")
    recommendations_critical = rec_engine.generate_recommendations(
        health_data=sample_health_data_critical,
        pattern_analysis=sample_pattern_analysis_critical,
        historical_data=None, # Skipping historical for this test
        context=sample_context_critical
    )
    
    print(f"\n    Total recommendations generated: {recommendations_critical['summary']['total_recommendations']}")
    print(f"    Emergency: {len(recommendations_critical['emergency_recommendations'])}")
    print(f"    Maintenance: {len(recommendations_critical['maintenance_recommendations'])}")
    print(f"    Operational: {len(recommendations_critical['operational_recommendations'])}")

    print("\n    Top 3 Priority Recommendations:")
    top_recs_critical = recommendations_critical['summary']['top_recommendations']
    for i, rec in enumerate(top_recs_critical[:3], 1):
        print(f"    {i}. [{rec['priority'].upper()}] {rec['title']} (Score: {rec['score']:.3f})")
        
    print("\n    Immediate Next Actions:")
    next_actions_critical = recommendations_critical['summary']['next_actions']
    for i, action in enumerate(next_actions_critical, 1):
        print(f"    {i}. {action['action']} (Priority: {action['priority']})")


    # === Test Scenario 2: Healthy System ===
    print("\n" + "="*30)
    print("=== Test Scenario 2: Healthy System ===")
    print("="*30)
    
    # This data should trigger few or no recommendations
    sample_health_data_healthy = {
        'overall_score': 0.92,
        'health_status': 'excellent',
        'component_scores': {
            'performance': {'score': 0.9},
            'reliability': {'score': 0.95},
            'efficiency': {'score': 0.88},
            'safety': {'score': 0.98},
            'maintenance': {'score': 0.9}
        },
        'trend_analysis': {
            'trend_direction': 'improving',
            'trend_strength': 0.02
        },
        'risk_assessment': {
            'overall_risk_level': 'low',
            'risk_factors': []
        }
    }
    
    sample_pattern_analysis_healthy = {
        'behavioral': {
            'behavioral_patterns': {
                'anomalies': {'anomaly_detection': True, 'anomalous_entities_count': 0}
            }
        }
    }
    
    sample_context_healthy = {
        'system_load': 0.35,
        'operating_hours': 1500,
        'last_maintenance': (datetime.now() - timedelta(days=10)).isoformat(),
        'location': 'clean_room'
    }

    print("1. Generating recommendations for HEALTHY state...")
    recommendations_healthy = rec_engine.generate_recommendations(
        health_data=sample_health_data_healthy,
        pattern_analysis=sample_pattern_analysis_healthy,
        historical_data=None,
        context=sample_context_healthy
    )
    
    print(f"\n    Total recommendations generated: {recommendations_healthy['summary']['total_recommendations']}")
    
    print("\n    Top Priority Recommendations (if any):")
    top_recs_healthy = recommendations_healthy['summary']['top_recommendations']
    if not top_recs_healthy:
        print("    No significant recommendations generated, as expected.")
    for i, rec in enumerate(top_recs_healthy, 1):
        print(f"    {i}. [{rec['priority'].upper()}] {rec['title']} (Score: {rec['score']:.3f})")
        
    print("\n    Immediate Next Actions (if any):")
    next_actions_healthy = recommendations_healthy['summary']['next_actions']
    if not next_actions_healthy:
        print("    No immediate actions required.")
    for i, action in enumerate(next_actions_healthy, 1):
        print(f"    {i}. {action['action']} (Priority: {action['priority']})")
        

    # === Test Scenario 3: Historical Data Trend ===
    print("\n" + "="*30)
    print("=== Test Scenario 3: Historical Data Trend ===")
    print("="*30)

    # Sample historical data
    np.random.seed(42)
    sample_historical_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
        'temperature': np.random.normal(30, 5, 100) + np.arange(100) * 0.1, # Increasing temp
        'efficiency': np.random.normal(75, 10, 100) - np.arange(100) * 0.2,  # Declining efficiency
        'vibration': np.random.exponential(0.2, 100),
        'error_count': np.random.poisson(0.5, 100)
    })
    
    print("1. Generating recommendations from HISTORICAL data...")
    recommendations_hist = rec_engine.generate_recommendations(
        health_data={}, # No current health
        pattern_analysis=None,
        historical_data=sample_historical_data,
        context=None
    )
    
    print(f"\n    Total recommendations generated: {recommendations_hist['summary']['total_recommendations']}")
    print("    Recommendations from historical analysis:")
    all_recs_hist = []
    for cat in recommendations_hist:
        if cat.endswith('_recommendations'):
            all_recs_hist.extend(recommendations_hist[cat])
            
    for rec in all_recs_hist:
        if rec['source'] == 'historical_analysis':
            print(f"    - [{rec['priority_level'].upper()}] {rec['title']}")

    # === Final Statistics ===
    print("\n" + "="*30)
    print("=== Final Engine Statistics ===")
    print("="*30)
    stats = rec_engine.get_recommendation_statistics()
    print(f"    Total recommendation sessions run: {stats['total_sessions']}")
    print(f"    Total recommendations generated: {stats['total_recommendations']}")
    print(f"    Average per session: {stats['average_per_session']:.1f}")
    print(f"    Category Distribution: {stats['category_distribution']}")
    print(f"    Priority Distribution: {stats['priority_distribution']}")
    
    print("\n=== RECOMMENDATION ENGINE DEMO COMPLETED ===")