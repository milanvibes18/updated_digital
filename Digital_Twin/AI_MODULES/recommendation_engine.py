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
    system health, patterns, and historical data.
    """
    
    def __init__(self, config_path: str = "CONFIG/recommendation_config.json"):
        self.config_path = Path(config_path)
        self.logger = self._setup_logging()
        
        # Load configuration
        self.config = self._load_config()
        
        # Recommendation storage
        self.recommendation_history = deque(maxlen=1000)
        self.recommendation_templates = {}
        self.action_effectiveness = defaultdict(list)
        
        # Knowledge base for recommendations
        self.knowledge_base = self._initialize_knowledge_base()
        
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
        
    def _setup_logging(self):
        """Setup logging for recommendation engine."""
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
    
    def _load_config(self) -> Dict:
        """Load recommendation engine configuration."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                self.logger.info("Recommendation configuration loaded")
                return config
            else:
                # Default configuration
                default_config = {
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
                    }
                }
                
                # Save default configuration
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                
                self.logger.info("Default recommendation configuration created")
                return default_config
                
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return {}
    
    def _initialize_knowledge_base(self) -> Dict:
        """Initialize knowledge base with recommendation rules and patterns."""
        return {
            'maintenance_rules': {
                'high_temperature': {
                    'conditions': ['temperature > 80', 'temperature_trend > 0.05'],
                    'recommendations': [
                        'Check cooling system',
                        'Inspect thermal sensors',
                        'Clean air filters',
                        'Verify ventilation'
                    ],
                    'urgency': 0.8,
                    'category': 'maintenance'
                },
                'high_vibration': {
                    'conditions': ['vibration > 0.5', 'vibration_trend > 0.02'],
                    'recommendations': [
                        'Check bearing alignment',
                        'Inspect mounting bolts',
                        'Balance rotating components',
                        'Lubricate moving parts'
                    ],
                    'urgency': 0.9,
                    'category': 'maintenance'
                },
                'low_efficiency': {
                    'conditions': ['efficiency < 70', 'efficiency_trend < -0.03'],
                    'recommendations': [
                        'Performance tuning required',
                        'Check operational parameters',
                        'Inspect wear components',
                        'Update control algorithms'
                    ],
                    'urgency': 0.6,
                    'category': 'optimization'
                },
                'pressure_anomaly': {
                    'conditions': ['pressure_deviation > 20', 'pressure_instability > 0.1'],
                    'recommendations': [
                        'Check pressure regulators',
                        'Inspect seals and gaskets',
                        'Calibrate pressure sensors',
                        'Verify system integrity'
                    ],
                    'urgency': 0.7,
                    'category': 'maintenance'
                }
            },
            'optimization_patterns': {
                'energy_efficiency': [
                    'Optimize operating schedules',
                    'Implement load balancing',
                    'Upgrade to efficient components',
                    'Fine-tune control parameters'
                ],
                'performance_improvement': [
                    'Streamline operational workflows',
                    'Implement predictive control',
                    'Optimize resource allocation',
                    'Enhance monitoring systems'
                ],
                'cost_reduction': [
                    'Consolidate maintenance schedules',
                    'Implement condition-based maintenance',
                    'Optimize inventory levels',
                    'Reduce operational overhead'
                ]
            },
            'operational_guidelines': {
                'best_practices': [
                    'Regular system health checks',
                    'Maintain optimal operating conditions',
                    'Follow recommended maintenance schedules',
                    'Monitor key performance indicators'
                ],
                'safety_protocols': [
                    'Ensure safety system functionality',
                    'Maintain emergency procedures',
                    'Regular safety training',
                    'Compliance with safety standards'
                ]
            }
        }
    
    def generate_recommendations(self, 
                               health_data: Dict,
                               pattern_analysis: Dict = None,
                               historical_data: pd.DataFrame = None,
                               context: Dict = None) -> Dict:
        """
        Generate comprehensive recommendations based on system state and analysis.
        
        Args:
            health_data: Health score and component analysis results
            pattern_analysis: Pattern analysis results
            historical_data: Historical system data
            context: Additional context information
            
        Returns:
            Dictionary containing categorized recommendations
        """
        try:
            self.logger.info("Generating system recommendations")
            
            # Update context
            if context:
                self.current_context.update(context)
            
            recommendations = {
                'timestamp': datetime.now().isoformat(),
                'maintenance_recommendations': [],
                'optimization_recommendations': [],
                'operational_recommendations': [],
                'preventive_recommendations': [],
                'emergency_recommendations': [],
                'summary': {},
                'context': self.current_context.copy()
            }
            
            # 1. Health-based recommendations
            if health_data:
                health_recommendations = self._generate_health_based_recommendations(health_data)
                self._categorize_recommendations(recommendations, health_recommendations)
            
            # 2. Pattern-based recommendations
            if pattern_analysis:
                pattern_recommendations = self._generate_pattern_based_recommendations(pattern_analysis)
                self._categorize_recommendations(recommendations, pattern_recommendations)
            
            # 3. Historical data recommendations
            if historical_data is not None and not historical_data.empty:
                historical_recommendations = self._generate_historical_recommendations(historical_data)
                self._categorize_recommendations(recommendations, historical_recommendations)
            
            # 4. Context-based recommendations
            if self.current_context:
                context_recommendations = self._generate_context_recommendations(self.current_context)
                self._categorize_recommendations(recommendations, context_recommendations)
            
            # 5. Knowledge base recommendations
            kb_recommendations = self._generate_knowledge_base_recommendations(health_data, self.current_context)
            self._categorize_recommendations(recommendations, kb_recommendations)
            
            # 6. Calculate composite scores and prioritize
            self._calculate_composite_scores(recommendations)
            
            # 7. Filter and limit recommendations
            self._filter_recommendations(recommendations)
            
            # 8. Generate summary
            recommendations['summary'] = self._generate_recommendations_summary(recommendations)
            
            # 9. Store recommendations for learning
            self._store_recommendations(recommendations)
            
            self.logger.info(f"Generated {self._count_total_recommendations(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendation generation error: {e}")
            return self._create_error_result(f"Recommendation generation failed: {e}")
    
    def _generate_health_based_recommendations(self, health_data: Dict) -> List[Dict]:
        """Generate recommendations based on health analysis."""
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
            self.logger.error(f"Health-based recommendation generation error: {e}")
            return []
    
    def _generate_pattern_based_recommendations(self, pattern_analysis: Dict) -> List[Dict]:
        """Generate recommendations based on pattern analysis."""
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
            self.logger.error(f"Pattern-based recommendation generation error: {e}")
            return []
    
    def _generate_historical_recommendations(self, historical_data: pd.DataFrame) -> List[Dict]:
        """Generate recommendations based on historical data analysis."""
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
            self.logger.error(f"Historical recommendation generation error: {e}")
            return []
    
    def _generate_context_recommendations(self, context: Dict) -> List[Dict]:
        """Generate recommendations based on operational context."""
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
                except ValueError:
                    pass
            
            # Operating hours recommendations
            operating_hours = context.get('operating_hours', 0)
            if operating_hours > 8000:  # High usage
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
            elif operating_hours > 10000:  # Very high usage
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
            self.logger.error(f"Context recommendation generation error: {e}")
            return []
    
    def _generate_knowledge_base_recommendations(self, health_data: Dict, context: Dict) -> List[Dict]:
        """Generate recommendations based on knowledge base rules."""
        try:
            recommendations = []
            maintenance_rules = self.knowledge_base.get('maintenance_rules', {})
            
            # Check each maintenance rule
            for rule_name, rule_config in maintenance_rules.items():
                conditions = rule_config.get('conditions', [])
                rule_recommendations = rule_config.get('recommendations', [])
                urgency = rule_config.get('urgency', 0.5)
                category = rule_config.get('category', 'maintenance')
                
                # Simple condition checking (would be more sophisticated in real implementation)
                rule_triggered = False
                
                if rule_name == 'high_temperature' and health_data:
                    component_scores = health_data.get('component_scores', {})
                    temp_related = any('temp' in comp.lower() for comp in component_scores.keys())
                    if temp_related:
                        temp_score = min([data.get('score', 1.0) for comp, data in component_scores.items() if 'temp' in comp.lower()], default=1.0)
                        if temp_score < 0.6:
                            rule_triggered = True
                
                elif rule_name == 'high_vibration' and health_data:
                    component_scores = health_data.get('component_scores', {})
                    vibration_related = any('vibr' in comp.lower() for comp in component_scores.keys())
                    if vibration_related:
                        vib_score = min([data.get('score', 1.0) for comp, data in component_scores.items() if 'vibr' in comp.lower()], default=1.0)
                        if vib_score < 0.6:
                            rule_triggered = True
                
                elif rule_name == 'low_efficiency' and health_data:
                    component_scores = health_data.get('component_scores', {})
                    efficiency_score = component_scores.get('efficiency', {}).get('score', 1.0)
                    if efficiency_score < 0.7:
                        rule_triggered = True
                
                if rule_triggered:
                    for rec_text in rule_recommendations:
                        recommendations.append({
                            'type': category,
                            'title': f'{rule_name.replace("_", " ").title()} Action Required',
                            'description': f'Knowledge base rule triggered: {rule_name}',
                            'action': rec_text,
                            'urgency': urgency,
                            'impact': 0.7,
                            'timeframe': 'within_week' if urgency > 0.7 else 'within_month',
                            'source': 'knowledge_base',
                            'priority_level': 'high' if urgency > 0.7 else 'medium'
                        })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Knowledge base recommendation generation error: {e}")
            return []
    
    def _categorize_recommendations(self, recommendations: Dict, new_recommendations: List[Dict]):
        """Categorize recommendations into appropriate buckets."""
        try:
            for rec in new_recommendations:
                rec_type = rec.get('type', 'operational')
                category_key = f"{rec_type}_recommendations"
                
                if category_key in recommendations:
                    recommendations[category_key].append(rec)
                else:
                    recommendations['operational_recommendations'].append(rec)
                    
        except Exception as e:
            self.logger.error(f"Recommendation categorization error: {e}")
    
    def _calculate_composite_scores(self, recommendations: Dict):
        """Calculate composite scores for all recommendations."""
        try:
            all_categories = ['emergency_recommendations', 'maintenance_recommendations',
                            'optimization_recommendations', 'operational_recommendations',
                            'preventive_recommendations']
            
            for category in all_categories:
                for rec in recommendations.get(category, []):
                    # Get individual scores
                    urgency = rec.get('urgency', 0.5)
                    impact = rec.get('impact', 0.5)
                    
                    # Estimate missing scores based on type and urgency
                    feasibility = 0.8 if rec.get('type') in ['operational', 'optimization'] else 0.6
                    cost_effectiveness = 0.7 if urgency > 0.7 else 0.5
                    risk_reduction = urgency * 0.8  # Higher urgency usually means higher risk reduction
                    
                    # Calculate composite score
                    composite_score = (
                        urgency * self.scoring_weights.get('urgency', 0.3) + 
                        impact * self.scoring_weights.get('impact', 0.25) + 
                        feasibility * self.scoring_weights.get('feasibility', 0.2) + 
                        cost_effectiveness * self.scoring_weights.get('cost_effectiveness', 0.15) + 
                        risk_reduction * self.scoring_weights.get('risk_reduction', 0.1)
                    )
                    
                    rec['composite_score'] = round(composite_score, 3)
                    rec['feasibility'] = feasibility
                    rec['cost_effectiveness'] = cost_effectiveness
                    rec['risk_reduction'] = risk_reduction
                    
                    # Set priority level if not already set
                    if 'priority_level' not in rec:
                        thresholds = self.config.get('thresholds', {})
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
            self.logger.error(f"Composite score calculation error: {e}")
    
    def _filter_recommendations(self, recommendations: Dict):
        """Filter and limit recommendations based on configuration."""
        try:
            filters = self.config.get('filters', {})
            min_confidence = filters.get('min_confidence', 0.4)
            max_recommendations = filters.get('max_recommendations', 20)
            
            all_categories = ['emergency_recommendations', 'maintenance_recommendations',
                            'optimization_recommendations', 'operational_recommendations',
                            'preventive_recommendations']
            
            for category in all_categories:
                category_recs = recommendations.get(category, [])
                
                # Filter by confidence (using composite_score as proxy)
                filtered_recs = [rec for rec in category_recs if rec.get('composite_score', 0) >= min_confidence]
                
                # Limit number of recommendations per category
                category_limit = max_recommendations // len(all_categories)
                filtered_recs = filtered_recs[:category_limit]
                
                recommendations[category] = filtered_recs
                
        except Exception as e:
            self.logger.error(f"Recommendation filtering error: {e}")
    
    def _generate_recommendations_summary(self, recommendations: Dict) -> Dict:
        """Generate summary of recommendations."""
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
            self.logger.error(f"Recommendations summary generation error: {e}")
            return {}
    
    def _count_total_recommendations(self, recommendations: Dict) -> int:
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
            self.logger.error(f"Recommendation counting error: {e}")
            return 0
    
    def _identify_next_actions(self, recommendations: List[Dict]) -> List[Dict]:
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
            self.logger.error(f"Next actions identification error: {e}")
            return []
    
    def _estimate_overall_impact(self, recommendations: List[Dict]) -> Dict:
        """Estimate overall impact of implementing recommendations."""
        try:
            if not recommendations:
                return {'estimated_improvement': 0, 'confidence': 0}
            
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
            self.logger.error(f"Overall impact estimation error: {e}")
            return {'estimated_improvement': 0, 'confidence': 0}
    
    def _identify_impact_areas(self, recommendations: List[Dict]) -> List[str]:
        """Identify main areas of impact from recommendations."""
        try:
            impact_areas = set()
            
            for rec in recommendations:
                rec_type = rec.get('type', 'operational')
                component = rec.get('component', '')
                
                if component:
                    impact_areas.add(component)
                else:
                    impact_areas.add(rec_type)
            
            return list(impact_areas)
            
        except Exception as e:
            self.logger.error(f"Impact areas identification error: {e}")
            return []
    
    def _store_recommendations(self, recommendations: Dict):
        """Store recommendations in history for learning."""
        try:
            history_entry = {
                'timestamp': recommendations['timestamp'],
                'recommendations': recommendations,
                'context': recommendations.get('context', {}),
                'total_count': self._count_total_recommendations(recommendations)
            }
            
            self.recommendation_history.append(history_entry)
            
        except Exception as e:
            self.logger.error(f"Recommendation storage error: {e}")
    
    def _assess_historical_data_quality(self, data: pd.DataFrame) -> Dict:
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
            self.logger.error(f"Data quality assessment error: {e}")
            return {'missing_percentage': 0, 'total_records': 0}
    
    def _calculate_simple_trend(self, data: pd.Series) -> float:
        """Calculate simple trend coefficient."""
        try:
            if len(data) < 3:
                return 0.0
            
            x = np.arange(len(data))
            coeffs = np.polyfit(x, data.values, 1)
            
            # Normalize by data range
            data_range = data.max() - data.min()
            normalized_trend = coeffs[0] / data_range if data_range > 0 else 0
            
            return float(normalized_trend)
            
        except Exception as e:
            self.logger.error(f"Trend calculation error: {e}")
            return 0.0
    
    def _analyze_historical_failures(self, data: pd.DataFrame) -> List[Dict]:
        """Analyze historical data for failure patterns."""
        try:
            recommendations = []
            
            # Look for failure indicators
            failure_indicators = ['error', 'fault', 'alarm', 'failure', 'down']
            
            for col in data.columns:
                if any(indicator in col.lower() for indicator in failure_indicators):
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
            self.logger.error(f"Historical failure analysis error: {e}")
            return []
    
    def _create_error_result(self, error_message: str) -> Dict:
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
        """Update effectiveness score for a recommendation."""
        try:
            self.action_effectiveness[recommendation_id].append({
                'effectiveness_score': effectiveness_score,
                'timestamp': datetime.now().isoformat()
            })
            
            self.logger.info(f"Updated effectiveness for recommendation {recommendation_id}: {effectiveness_score}")
            
        except Exception as e:
            self.logger.error(f"Effectiveness update error: {e}")
    
    def get_recommendation_history(self, limit: int = 50) -> List[Dict]:
        """Get historical recommendations."""
        try:
            history = list(self.recommendation_history)
            history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return history[:limit]
            
        except Exception as e:
            self.logger.error(f"Recommendation history retrieval error: {e}")
            return []
    
    def export_recommendations(self, recommendations: Dict, output_path: str = None) -> str:
        """Export recommendations to file."""
        try:
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"REPORTS/recommendations_{timestamp}.json"
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(recommendations, f, indent=2, default=str)
            
            self.logger.info(f"Recommendations exported to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Recommendation export error: {e}")
            raise
    
    def get_recommendation_statistics(self) -> Dict:
        """Get comprehensive recommendation statistics."""
        try:
            if not self.recommendation_history:
                return {'total_sessions': 0, 'statistics': {}}
            
            total_recs = sum(entry['total_count'] for entry in self.recommendation_history)
            
            # Category statistics
            category_stats = defaultdict(int)
            priority_stats = defaultdict(int)
            
            for entry in self.recommendation_history:
                recs = entry['recommendations']
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
                'average_per_session': total_recs / len(self.recommendation_history),
                'category_distribution': dict(category_stats),
                'priority_distribution': dict(priority_stats),
                'average_effectiveness': round(avg_effectiveness, 3),
                'effectiveness_samples': len(effectiveness_data),
                'recent_activity': self.recommendation_history[-5:] if len(self.recommendation_history) >= 5 else list(self.recommendation_history)
            }
            
        except Exception as e:
            self.logger.error(f"Recommendation statistics error: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    # Initialize recommendation engine
    rec_engine = RecommendationEngine()
    
    # Sample health data
    sample_health_data = {
        'overall_score': 0.65,
        'health_status': 'warning',
        'component_scores': {
            'performance': {'score': 0.7},
            'reliability': {'score': 0.8},
            'efficiency': {'score': 0.4},  # Low efficiency
            'safety': {'score': 0.9},
            'maintenance': {'score': 0.3}   # Critical maintenance needed
        },
        'trend_analysis': {
            'trend_direction': 'degrading',
            'trend_strength': 0.08
        },
        'risk_assessment': {
            'overall_risk_level': 'medium',
            'risk_factors': [
                {
                    'component': 'efficiency',
                    'risk_level': 'high',
                    'impact': 'performance_degradation'
                },
                {
                    'component': 'maintenance',
                    'risk_level': 'high',
                    'impact': 'critical_component_failure'
                }
            ]
        }
    }
    
    # Sample pattern analysis
    sample_pattern_analysis = {
        'temporal': {
            'patterns_found': {
                'temperature': {
                    'cyclical': {
                        'cycles_detected': 2,
                        'dominant_cycle': {
                            'cycle_length': 24,
                            'strength': 0.8
                        }
                    },
                    'change_points': {
                        'change_points_detected': 3
                    }
                }
            }
        },
        'behavioral': {
            'behavioral_patterns': {
                'anomalies': {
                    'anomaly_detection': True,
                    'anomalous_entities_count': 2
                },
                'clustering': {
                    'clusters_found': 3
                }
            }
        }
    }
    
    # Sample historical data
    np.random.seed(42)
    sample_historical_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
        'temperature': np.random.normal(30, 5, 100),
        'efficiency': np.random.normal(75, 10, 100) - np.arange(100) * 0.1,  # Declining efficiency
        'vibration': np.random.exponential(0.2, 100) + np.arange(100) * 0.001,  # Increasing vibration
        'error_count': np.random.poisson(0.5, 100)
    })
    
    # Context information
    sample_context = {
        'system_load': 0.95,  # High load
        'operating_hours': 8760,  # Full year
        'last_maintenance': '2024-01-01',
        'location': 'factory_floor_a'
    }
    
    print("=== DIGITAL TWIN RECOMMENDATION ENGINE DEMO ===\n")
    
    # Generate recommendations
    print("1. Generating comprehensive recommendations...")
    recommendations = rec_engine.generate_recommendations(
        health_data=sample_health_data,
        pattern_analysis=sample_pattern_analysis,
        historical_data=sample_historical_data,
        context=sample_context
    )
    
    # Display results
    print(f"   Total recommendations generated: {recommendations['summary']['total_recommendations']}")
    print(f"   Emergency: {len(recommendations['emergency_recommendations'])}")
    print(f"   Maintenance: {len(recommendations['maintenance_recommendations'])}")
    print(f"   Optimization: {len(recommendations['optimization_recommendations'])}")
    print(f"   Operational: {len(recommendations['operational_recommendations'])}")
    print(f"   Preventive: {len(recommendations['preventive_recommendations'])}")
    print()
    
    # Show top recommendations
    print("2. Top Priority Recommendations:")
    top_recs = recommendations['summary']['top_recommendations']
    for i, rec in enumerate(top_recs, 1):
        print(f"   {i}. [{rec['priority'].upper()}] {rec['title']}")
        print(f"      Score: {rec['score']:.3f} | Timeframe: {rec['timeframe']}")
    print()
    
    # Show emergency recommendations
    if recommendations['emergency_recommendations']:
        print("3. Emergency Recommendations:")
        for rec in recommendations['emergency_recommendations']:
            print(f"   • {rec['title']}")
            print(f"     Action: {rec['action']}")
            print(f"     Urgency: {rec['urgency']:.2f} | Impact: {rec['impact']:.2f}")
        print()
    
    # Show maintenance recommendations
    if recommendations['maintenance_recommendations']:
        print("4. Critical Maintenance Recommendations:")
        for rec in recommendations['maintenance_recommendations'][:3]:
            print(f"   • {rec['title']}")
            print(f"     Action: {rec['action']}")
            print(f"     Priority: {rec.get('priority_level', 'unknown')}")
        print()
    
    # Show next actions
    print("5. Immediate Next Actions:")
    next_actions = recommendations['summary']['next_actions']
    for i, action in enumerate(next_actions, 1):
        print(f"   {i}. {action['action']}")
        print(f"      Category: {action['category']} | Timeframe: {action['timeframe']}")
    print()
    
    # Impact estimation
    print("6. Estimated Impact of Recommendations:")
    impact = recommendations['summary']['estimated_impact']
    print(f"   Estimated Improvement: {impact['estimated_improvement']*100:.1f}%")
    print(f"   Confidence: {impact['confidence']*100:.1f}%")
    print(f"   Impact Areas: {', '.join(impact['impact_areas'])}")
    print()
    
    # Export recommendations
    print("7. Exporting recommendations...")
    export_path = rec_engine.export_recommendations(recommendations)
    print(f"   Recommendations exported to: {export_path}")
    
    # Statistics
    print("\n8. Recommendation Engine Statistics:")
    stats = rec_engine.get_recommendation_statistics()
    print(f"   Total recommendation sessions: {stats['total_sessions']}")
    print(f"   Total recommendations generated: {stats['total_recommendations']}")
    if stats['total_sessions'] > 0:
        print(f"   Average recommendations per session: {stats['average_per_session']:.1f}")
    print()
    
    # Simulate effectiveness feedback
    print("9. Simulating recommendation effectiveness feedback...")
    if recommendations['maintenance_recommendations']:
        # Simulate high effectiveness for first maintenance recommendation
        rec_engine.update_recommendation_effectiveness("maintenance_001", 0.85)
        print("   Updated effectiveness score for maintenance recommendation")
    
    print("\n=== RECOMMENDATION ENGINE DEMO COMPLETED ===")
    print("\nKey Findings:")
    print("- Critical maintenance issues identified requiring immediate attention")
    print("- Efficiency optimization opportunities detected")
    print("- Pattern-based recommendations for preventive actions")
    print("- Context-aware suggestions for current operating conditions")
    print("- Comprehensive scoring and prioritization of all recommendations")
    print("- Actionable next steps with clear timeframes and priorities")