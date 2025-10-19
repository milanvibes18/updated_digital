import numpy as np
import pandas as pd
import logging
import json
import smtplib
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict, deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import threading
import queue
import warnings
warnings.filterwarnings('ignore')

# Notification libraries
import requests  # For webhooks and Slack notifications

class AlertManager:
    """
    Advanced alert management system for Digital Twin applications.
    Handles real-time monitoring, intelligent alerting, notification routing,
    alert correlation, escalation, and suppression.
    """
    
    def __init__(self, config_path: str = "CONFIG/alert_config.json"):
        self.config_path = Path(config_path)
        self.logger = self._setup_logging()
        
        # Load configuration
        self.config = self._load_config()
        
        # Alert storage and tracking
        self.active_alerts = {}
        self.alert_history = deque(maxlen=10000)
        self.suppressed_alerts = {}
        self.escalated_alerts = {}
        
        # Alert processing queue
        self.alert_queue = queue.Queue()
        self.processing_thread = None
        self.should_stop = threading.Event()
        
        # Alert rules and thresholds
        self.alert_rules = self.config.get('alert_rules', {})
        self.thresholds = self.config.get('thresholds', {})
        self.notification_channels = self.config.get('notification_channels', {})
        
        # Alert correlation and grouping
        self.correlation_window = timedelta(minutes=5)
        self.correlation_rules = self.config.get('correlation_rules', {})
        
        # Rate limiting and suppression
        self.rate_limits = defaultdict(list)
        self.suppression_rules = self.config.get('suppression_rules', {})
        
        # Escalation management
        self.escalation_policies = self.config.get('escalation_policies', {})
        self.escalation_timers = {}
        
        # Notification handlers
        self.notification_handlers = {
            'email': self._send_email_notification,
            'slack': self._send_slack_notification,
            'webhook': self._send_webhook_notification,
            'sms': self._send_sms_notification
        }
        
        # Start alert processing
        self._start_alert_processing()
        
    def _setup_logging(self):
        """Setup logging for alert manager."""
        logger = logging.getLogger('AlertManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Ensure LOGS directory exists
            Path("LOGS").mkdir(exist_ok=True)
            handler = logging.FileHandler('LOGS/digital_twin_alerts.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_config(self) -> Dict:
        """Load alert manager configuration."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                self.logger.info("Alert configuration loaded")
                return config
            else:
                # Default configuration
                default_config = {
                    "alert_rules": {
                        "temperature_high": {
                            "metric": "temperature",
                            "operator": ">",
                            "threshold": 80,
                            "severity": "warning",
                            "duration": 300,
                            "description": "Temperature above normal operating range"
                        },
                        "temperature_critical": {
                            "metric": "temperature",
                            "operator": ">",
                            "threshold": 95,
                            "severity": "critical",
                            "duration": 60,
                            "description": "Temperature at critical level"
                        },
                        "vibration_high": {
                            "metric": "vibration",
                            "operator": ">",
                            "threshold": 0.5,
                            "severity": "warning",
                            "duration": 180,
                            "description": "High vibration detected"
                        },
                        "efficiency_low": {
                            "metric": "efficiency",
                            "operator": "<",
                            "threshold": 70,
                            "severity": "info",
                            "duration": 600,
                            "description": "System efficiency below optimal"
                        }
                    },
                    "thresholds": {
                        "health_score_critical": 0.3,
                        "health_score_warning": 0.6,
                        "anomaly_threshold": 0.8,
                        "failure_probability_high": 0.7
                    },
                    "notification_channels": {
                        "email": {
                            "enabled": True,
                            "smtp_server": "smtp.gmail.com",
                            "smtp_port": 587,
                            "username": "alerts@company.com",
                            "password": "your_password",
                            "recipients": ["admin@company.com", "maintenance@company.com"]
                        },
                        "slack": {
                            "enabled": False,
                            "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
                            "channel": "#alerts"
                        },
                        "webhook": {
                            "enabled": False,
                            "url": "http://your-webhook-endpoint.com/alerts",
                            "method": "POST"
                        }
                    },
                    "escalation_policies": {
                        "default": {
                            "levels": [
                                {"delay_minutes": 0, "channels": ["email"]},
                                {"delay_minutes": 15, "channels": ["email", "slack"]}
                            ]
                        },
                        "critical": {
                            "levels": [
                                {"delay_minutes": 0, "channels": ["email", "slack"]},
                                {"delay_minutes": 5, "channels": ["email", "slack"]}
                            ]
                        }
                    },
                    "suppression_rules": {
                        "duplicate_window_minutes": 5,
                        "burst_limit": 10,
                        "burst_window_minutes": 1
                    },
                    "correlation_rules": {
                        "temperature_vibration": {
                            "metrics": ["temperature", "vibration"],
                            "correlation_threshold": 0.7,
                            "window_minutes": 10,
                            "action": "create_correlated_alert"
                        }
                    }
                }
                
                # Save default configuration
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                
                self.logger.info("Default alert configuration created")
                return default_config
                
        except Exception as e:
            self.logger.error(f"Failed to load alert configuration: {e}")
            return {}
    
    def _start_alert_processing(self):
        """Start background alert processing thread."""
        try:
            self.processing_thread = threading.Thread(target=self._process_alerts, daemon=True)
            self.processing_thread.start()
            self.logger.info("Alert processing thread started")
        except Exception as e:
            self.logger.error(f"Failed to start alert processing: {e}")
    
    def _process_alerts(self):
        """Main alert processing loop."""
        while not self.should_stop.is_set():
            try:
                # Get alert from queue with timeout
                alert = self.alert_queue.get(timeout=1)
                
                # Process the alert
                self._handle_alert(alert)
                
                # Mark task as done
                self.alert_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Alert processing error: {e}")
    
    def evaluate_conditions(self, data: Dict, device_id: str = None) -> List[Dict]:
        """
        Evaluate alert conditions against current data.
        
        Args:
            data: Current system/device data
            device_id: Optional device identifier
            
        Returns:
            List of triggered alerts
        """
        try:
            triggered_alerts = []
            current_time = datetime.now()
            
            # Evaluate each alert rule
            for rule_name, rule_config in self.alert_rules.items():
                try:
                    alert = self._evaluate_single_rule(rule_name, rule_config, data, device_id, current_time)
                    if alert:
                        triggered_alerts.append(alert)
                        
                except Exception as e:
                    self.logger.error(f"Error evaluating rule {rule_name}: {e}")
                    continue
            
            # Evaluate health score thresholds
            if 'health_score' in data:
                health_alerts = self._evaluate_health_score_alerts(data, device_id, current_time)
                triggered_alerts.extend(health_alerts)
            
            # Evaluate anomaly detection alerts
            if 'anomaly_score' in data:
                anomaly_alerts = self._evaluate_anomaly_alerts(data, device_id, current_time)
                triggered_alerts.extend(anomaly_alerts)
            
            # Queue triggered alerts for processing
            for alert in triggered_alerts:
                self.alert_queue.put(alert)
            
            self.logger.info(f"Evaluated conditions: {len(triggered_alerts)} alerts triggered")
            return triggered_alerts
            
        except Exception as e:
            self.logger.error(f"Alert condition evaluation error: {e}")
            return []
    
    def _evaluate_single_rule(self, rule_name: str, rule_config: Dict, data: Dict, 
                             device_id: str, current_time: datetime) -> Optional[Dict]:
        """Evaluate a single alert rule."""
        try:
            metric = rule_config.get('metric')
            operator = rule_config.get('operator')
            threshold = rule_config.get('threshold')
            duration = rule_config.get('duration', 0)
            severity = rule_config.get('severity', 'info')
            
            if metric not in data:
                return None
            
            current_value = data[metric]
            
            # Check if condition is met
            condition_met = self._check_condition(current_value, operator, threshold)
            
            if not condition_met:
                # Clear any existing alert for this rule
                alert_key = f"{rule_name}_{device_id or 'system'}"
                if alert_key in self.active_alerts:
                    self._resolve_alert(alert_key)
                return None
            
            # Check duration requirement
            if duration > 0:
                alert_key = f"{rule_name}_{device_id or 'system'}"
                
                if alert_key in self.active_alerts:
                    # Check if duration threshold is met
                    alert_start = datetime.fromisoformat(self.active_alerts[alert_key]['first_triggered'])
                    if (current_time - alert_start).total_seconds() < duration:
                        return None  # Duration not yet met
                else:
                    # First time condition is met, start tracking
                    self.active_alerts[alert_key] = {
                        'first_triggered': current_time.isoformat(),
                        'rule_name': rule_name,
                        'condition_met': True
                    }
                    return None  # Wait for duration
            
            # Create alert
            alert = {
                'id': f"{rule_name}_{device_id or 'system'}_{int(current_time.timestamp())}",
                'rule_name': rule_name,
                'device_id': device_id,
                'metric': metric,
                'current_value': float(current_value),
                'threshold': threshold,
                'operator': operator,
                'severity': severity,
                'description': rule_config.get('description', f'{metric} {operator} {threshold}'),
                'timestamp': current_time.isoformat(),
                'status': 'active',
                'acknowledged': False,
                'source': 'rule_evaluation'
            }
            
            return alert
            
        except Exception as e:
            self.logger.error(f"Single rule evaluation error: {e}")
            return None
    
    def _check_condition(self, value: float, operator: str, threshold: float) -> bool:
        """Check if a condition is met."""
        try:
            if operator == '>':
                return value > threshold
            elif operator == '<':
                return value < threshold
            elif operator == '>=':
                return value >= threshold
            elif operator == '<=':
                return value <= threshold
            elif operator == '==':
                return abs(value - threshold) < 1e-6
            elif operator == '!=':
                return abs(value - threshold) >= 1e-6
            else:
                self.logger.warning(f"Unknown operator: {operator}")
                return False
                
        except Exception as e:
            self.logger.error(f"Condition check error: {e}")
            return False
    
    def _evaluate_health_score_alerts(self, data: Dict, device_id: str, current_time: datetime) -> List[Dict]:
        """Evaluate health score-based alerts."""
        try:
            alerts = []
            health_score = data.get('health_score', 1.0)
            
            # Critical health score alert
            if health_score <= self.thresholds.get('health_score_critical', 0.3):
                alerts.append({
                    'id': f"health_critical_{device_id or 'system'}_{int(current_time.timestamp())}",
                    'rule_name': 'health_score_critical',
                    'device_id': device_id,
                    'metric': 'health_score',
                    'current_value': health_score,
                    'threshold': self.thresholds.get('health_score_critical', 0.3),
                    'operator': '<=',
                    'severity': 'critical',
                    'description': f'System health score critically low: {health_score:.3f}',
                    'timestamp': current_time.isoformat(),
                    'status': 'active',
                    'acknowledged': False,
                    'source': 'health_evaluation'
                })
            
            # Warning health score alert
            elif health_score <= self.thresholds.get('health_score_warning', 0.6):
                alerts.append({
                    'id': f"health_warning_{device_id or 'system'}_{int(current_time.timestamp())}",
                    'rule_name': 'health_score_warning',
                    'device_id': device_id,
                    'metric': 'health_score',
                    'current_value': health_score,
                    'threshold': self.thresholds.get('health_score_warning', 0.6),
                    'operator': '<=',
                    'severity': 'warning',
                    'description': f'System health score below normal: {health_score:.3f}',
                    'timestamp': current_time.isoformat(),
                    'status': 'active',
                    'acknowledged': False,
                    'source': 'health_evaluation'
                })
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Health score alert evaluation error: {e}")
            return []
    
    def _evaluate_anomaly_alerts(self, data: Dict, device_id: str, current_time: datetime) -> List[Dict]:
        """Evaluate anomaly detection alerts."""
        try:
            alerts = []
            
            # Check for anomaly scores
            if 'anomaly_score' in data:
                anomaly_score = data['anomaly_score']
                anomaly_threshold = self.thresholds.get('anomaly_threshold', 0.8)
                
                if abs(anomaly_score) >= anomaly_threshold:
                    alerts.append({
                        'id': f"anomaly_{device_id or 'system'}_{int(current_time.timestamp())}",
                        'rule_name': 'anomaly_detection',
                        'device_id': device_id,
                        'metric': 'anomaly_score',
                        'current_value': anomaly_score,
                        'threshold': anomaly_threshold,
                        'operator': '>=',
                        'severity': 'warning',
                        'description': f'Anomaly detected with score: {anomaly_score:.3f}',
                        'timestamp': current_time.isoformat(),
                        'status': 'active',
                        'acknowledged': False,
                        'source': 'anomaly_detection'
                    })
            
            # Check for failure probability
            if 'failure_probability' in data:
                failure_prob = data['failure_probability']
                prob_threshold = self.thresholds.get('failure_probability_high', 0.7)
                
                if failure_prob >= prob_threshold:
                    alerts.append({
                        'id': f"failure_risk_{device_id or 'system'}_{int(current_time.timestamp())}",
                        'rule_name': 'high_failure_probability',
                        'device_id': device_id,
                        'metric': 'failure_probability',
                        'current_value': failure_prob,
                        'threshold': prob_threshold,
                        'operator': '>=',
                        'severity': 'critical',
                        'description': f'High failure probability detected: {failure_prob:.3f}',
                        'timestamp': current_time.isoformat(),
                        'status': 'active',
                        'acknowledged': False,
                        'source': 'predictive_analysis'
                    })
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Anomaly alert evaluation error: {e}")
            return []
    
    def _handle_alert(self, alert: Dict):
        """Handle a triggered alert through the processing pipeline."""
        try:
            # Check if alert should be suppressed
            if self._should_suppress_alert(alert):
                self.logger.info(f"Alert suppressed: {alert['id']}")
                return
            
            # Apply correlation rules
            correlated_alert = self._apply_correlation_rules(alert)
            if correlated_alert != alert:
                alert = correlated_alert
                self.logger.info(f"Alert correlated: {alert['id']}")
            
            # Check rate limiting
            if self._is_rate_limited(alert):
                self.logger.info(f"Alert rate limited: {alert['id']}")
                return
            
            # Store alert
            self._store_alert(alert)
            
            # Send notifications
            self._send_notifications(alert)
            
            # Setup escalation if needed
            self._setup_escalation(alert)
            
            self.logger.info(f"Alert processed: {alert['id']} - {alert['severity']}")
            
        except Exception as e:
            self.logger.error(f"Alert handling error: {e}")
    
    def _should_suppress_alert(self, alert: Dict) -> bool:
        """Check if alert should be suppressed."""
        try:
            current_time = datetime.now()
            
            # Check duplicate suppression
            duplicate_window = timedelta(minutes=self.suppression_rules.get('duplicate_window_minutes', 5))
            alert_key = f"{alert['rule_name']}_{alert.get('device_id', 'system')}"
            
            if alert_key in self.suppressed_alerts:
                last_suppressed = datetime.fromisoformat(self.suppressed_alerts[alert_key]['last_suppressed'])
                if current_time - last_suppressed < duplicate_window:
                    return True
            
            # Check burst suppression
            burst_limit = self.suppression_rules.get('burst_limit', 10)
            burst_window = timedelta(minutes=self.suppression_rules.get('burst_window_minutes', 1))
            
            # Count recent alerts of same type
            recent_alerts = [
                a for a in self.alert_history
                if (a.get('rule_name') == alert['rule_name'] and
                    datetime.fromisoformat(a['timestamp']) > current_time - burst_window)
            ]
            
            if len(recent_alerts) >= burst_limit:
                # Suppress and record
                self.suppressed_alerts[alert_key] = {
                    'last_suppressed': current_time.isoformat(),
                    'suppression_count': self.suppressed_alerts.get(alert_key, {}).get('suppression_count', 0) + 1
                }
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Alert suppression check error: {e}")
            return False
    
    def _apply_correlation_rules(self, alert: Dict) -> Dict:
        """Apply correlation rules to potentially group related alerts."""
        try:
            current_time = datetime.now()
            
            # Check each correlation rule
            for rule_name, rule_config in self.correlation_rules.items():
                metrics = rule_config.get('metrics', [])
                correlation_threshold = rule_config.get('correlation_threshold', 0.7)
                window_minutes = rule_config.get('window_minutes', 10)
                
                if alert['metric'] not in metrics:
                    continue
                
                # Find related alerts within time window
                window_start = current_time - timedelta(minutes=window_minutes)
                related_alerts = []
                
                for historical_alert in reversed(list(self.alert_history)):
                    alert_time = datetime.fromisoformat(historical_alert['timestamp'])
                    
                    if alert_time < window_start:
                        break
                        
                    if (historical_alert['metric'] in metrics and
                        historical_alert['metric'] != alert['metric'] and
                        historical_alert.get('device_id') == alert.get('device_id')):
                        related_alerts.append(historical_alert)
                
                # If related alerts found, create correlated alert
                if related_alerts:
                    correlated_alert = self._create_correlated_alert(alert, related_alerts, rule_name)
                    return correlated_alert
            
            return alert
            
        except Exception as e:
            self.logger.error(f"Alert correlation error: {e}")
            return alert
    
    def _create_correlated_alert(self, primary_alert: Dict, related_alerts: List[Dict], rule_name: str) -> Dict:
        """Create a correlated alert from multiple related alerts."""
        try:
            all_metrics = [primary_alert['metric']] + [a['metric'] for a in related_alerts]
            
            correlated_alert = primary_alert.copy()
            correlated_alert.update({
                'id': f"correlated_{rule_name}_{primary_alert.get('device_id', 'system')}_{int(time.time())}",
                'rule_name': f'correlated_{rule_name}',
                'description': f"Correlated alert: {', '.join(set(all_metrics))} anomalies detected",
                'severity': 'warning' if primary_alert['severity'] == 'info' else primary_alert['severity'],
                'correlated_alerts': [a['id'] for a in related_alerts],
                'correlation_rule': rule_name,
                'source': 'correlation_engine'
            })
            
            return correlated_alert
            
        except Exception as e:
            self.logger.error(f"Correlated alert creation error: {e}")
            return primary_alert
    
    def _is_rate_limited(self, alert: Dict) -> bool:
        """Check if alert should be rate limited."""
        try:
            current_time = time.time()
            rate_key = f"{alert['rule_name']}_{alert.get('device_id', 'system')}"
            
            # Clean old entries
            cutoff_time = current_time - 3600  # 1 hour window
            self.rate_limits[rate_key] = [
                t for t in self.rate_limits[rate_key] if t > cutoff_time
            ]
            
            # Check rate limit (max 10 alerts per hour for same rule/device)
            if len(self.rate_limits[rate_key]) >= 10:
                return True
            
            # Add current alert timestamp
            self.rate_limits[rate_key].append(current_time)
            return False
            
        except Exception as e:
            self.logger.error(f"Rate limiting check error: {e}")
            return False
    
    def _store_alert(self, alert: Dict):
        """Store alert in active alerts and history."""
        try:
            # Add to active alerts
            self.active_alerts[alert['id']] = alert
            
            # Add to history
            self.alert_history.append(alert.copy())
            
            # Clean up old resolved alerts from active alerts
            current_time = datetime.now()
            resolved_alerts = []
            
            for alert_id, stored_alert in self.active_alerts.items():
                if stored_alert.get('status') == 'resolved':
                    resolved_time = datetime.fromisoformat(stored_alert.get('resolved_at', stored_alert['timestamp']))
                    if current_time - resolved_time > timedelta(hours=1):
                        resolved_alerts.append(alert_id)
            
            for alert_id in resolved_alerts:
                del self.active_alerts[alert_id]
                
        except Exception as e:
            self.logger.error(f"Alert storage error: {e}")
    
    def _send_notifications(self, alert: Dict):
        """Send notifications for the alert."""
        try:
            severity = alert['severity']
            
            # Determine notification channels based on severity
            if severity == 'critical':
                channels = ['email', 'slack']
            elif severity == 'warning':
                channels = ['email']
            else:  # info
                channels = []
            
            # Send notifications
            for channel in channels:
                if channel in self.notification_channels and self.notification_channels[channel].get('enabled', False):
                    try:
                        handler = self.notification_handlers.get(channel)
                        if handler:
                            handler(alert)
                    except Exception as e:
                        self.logger.error(f"Notification sending error ({channel}): {e}")
                        
        except Exception as e:
            self.logger.error(f"Notification handling error: {e}")
    
    def _setup_escalation(self, alert: Dict):
        """Setup escalation for the alert if needed."""
        try:
            severity = alert['severity']
            
            # Determine escalation policy
            if severity == 'critical':
                policy_name = 'critical'
            else:
                policy_name = 'default'
            
            escalation_policy = self.escalation_policies.get(policy_name)
            if not escalation_policy:
                return
            
            # Schedule escalation levels
            alert_id = alert['id']
            self.escalation_timers[alert_id] = []
            
            for level_index, level in enumerate(escalation_policy.get('levels', [])[1:], 1):
                delay_minutes = level.get('delay_minutes', 15)
                
                # Schedule escalation
                timer = threading.Timer(
                    delay_minutes * 60,
                    self._escalate_alert,
                    args=[alert_id, level_index, level]
                )
                timer.start()
                self.escalation_timers[alert_id].append(timer)
                
        except Exception as e:
            self.logger.error(f"Escalation setup error: {e}")
    
    def _escalate_alert(self, alert_id: str, level: int, escalation_config: Dict):
        """Escalate an alert to the next level."""
        try:
            # Check if alert is still active and unacknowledged
            if (alert_id not in self.active_alerts or
                self.active_alerts[alert_id].get('acknowledged', False) or
                self.active_alerts[alert_id].get('status') == 'resolved'):
                return
            
            alert = self.active_alerts[alert_id]
            
            # Update alert with escalation info
            alert['escalation_level'] = level
            alert['last_escalated'] = datetime.now().isoformat()
            
            # Send escalation notifications
            channels = escalation_config.get('channels', [])
            for channel in channels:
                if channel in self.notification_channels and self.notification_channels[channel].get('enabled', False):
                    try:
                        # Create escalation alert
                        escalation_alert = alert.copy()
                        escalation_alert['description'] = f"ESCALATED (Level {level}): {alert['description']}"
                        escalation_alert['escalated'] = True
                        
                        handler = self.notification_handlers.get(channel)
                        if handler:
                            handler(escalation_alert)
                            
                    except Exception as e:
                        self.logger.error(f"Escalation notification error ({channel}): {e}")
            
            self.logger.warning(f"Alert escalated to level {level}: {alert_id}")
            
        except Exception as e:
            self.logger.error(f"Alert escalation error: {e}")
    
    def _send_email_notification(self, alert: Dict):
        """Send email notification for alert."""
        try:
            email_config = self.notification_channels.get('email', {})
            
            if not email_config.get('enabled', False):
                return
            
            # For demo purposes, just log the notification
            self.logger.info(f"Email notification sent for alert {alert['id']}: {alert['description']}")
            
            # Full email implementation would go here:
            # - Create email message with proper formatting
            # - Connect to SMTP server
            # - Send to configured recipients
            
        except Exception as e:
            self.logger.error(f"Email notification error: {e}")
    
    def _send_slack_notification(self, alert: Dict):
        """Send Slack notification for alert."""
        try:
            slack_config = self.notification_channels.get('slack', {})
            
            if not slack_config.get('enabled', False):
                return
            
            # For demo purposes, just log the notification
            self.logger.info(f"Slack notification sent for alert {alert['id']}: {alert['description']}")
            
            # Full Slack implementation would go here:
            # - Format message for Slack
            # - Send to webhook URL
            
        except Exception as e:
            self.logger.error(f"Slack notification error: {e}")
    
    def _send_webhook_notification(self, alert: Dict):
        """Send webhook notification for alert."""
        try:
            webhook_config = self.notification_channels.get('webhook', {})
            
            if not webhook_config.get('enabled', False):
                return
            
            # For demo purposes, just log the notification
            self.logger.info(f"Webhook notification sent for alert {alert['id']}: {alert['description']}")
            
            # Full webhook implementation would go here:
            # - Format payload
            # - Send HTTP request to configured endpoint
                
        except Exception as e:
            self.logger.error(f"Webhook notification error: {e}")
    
    def _send_sms_notification(self, alert: Dict):
        """Send SMS notification for alert."""
        try:
            # For demo purposes, just log the notification
            self.logger.info(f"SMS notification sent for alert {alert['id']}: {alert['description']}")
            
            # Full SMS implementation would integrate with service like Twilio
            
        except Exception as e:
            self.logger.error(f"SMS notification error: {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = None) -> bool:
        """Acknowledge an alert."""
        try:
            if alert_id not in self.active_alerts:
                self.logger.warning(f"Alert not found for acknowledgment: {alert_id}")
                return False
            
            alert = self.active_alerts[alert_id]
            alert['acknowledged'] = True
            alert['acknowledged_by'] = acknowledged_by or 'system'
            alert['acknowledged_at'] = datetime.now().isoformat()
            
            # Cancel escalation timers
            if alert_id in self.escalation_timers:
                for timer in self.escalation_timers[alert_id]:
                    timer.cancel()
                del self.escalation_timers[alert_id]
            
            self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
            return True
            
        except Exception as e:
            self.logger.error(f"Alert acknowledgment error: {e}")
            return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str = None) -> bool:
        """Resolve an alert."""
        try:
            return self._resolve_alert(alert_id, resolved_by)
            
        except Exception as e:
            self.logger.error(f"Alert resolution error: {e}")
            return False
    
    def _resolve_alert(self, alert_id: str, resolved_by: str = None) -> bool:
        """Internal method to resolve an alert."""
        try:
            if alert_id not in self.active_alerts:
                return False
            
            alert = self.active_alerts[alert_id]
            alert['status'] = 'resolved'
            alert['resolved_by'] = resolved_by or 'system'
            alert['resolved_at'] = datetime.now().isoformat()
            
            # Cancel escalation timers
            if alert_id in self.escalation_timers:
                for timer in self.escalation_timers[alert_id]:
                    timer.cancel()
                del self.escalation_timers[alert_id]
            
            self.logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
            return True
            
        except Exception as e:
            self.logger.error(f"Internal alert resolution error: {e}")
            return False
    
    def get_active_alerts(self, device_id: str = None, severity: str = None) -> List[Dict]:
        """Get list of active alerts with optional filtering."""
        try:
            active_alerts = []
            
            for alert in self.active_alerts.values():
                if alert.get('status') != 'active':
                    continue
                    
                if device_id and alert.get('device_id') != device_id:
                    continue
                    
                if severity and alert.get('severity') != severity:
                    continue
                
                active_alerts.append(alert.copy())
            
            # Sort by severity and timestamp
            severity_order = {'critical': 0, 'warning': 1, 'info': 2}
            active_alerts.sort(key=lambda x: (
                severity_order.get(x['severity'], 3),
                x['timestamp']
            ))
            
            return active_alerts
            
        except Exception as e:
            self.logger.error(f"Get active alerts error: {e}")
            return []
    
    def get_alert_statistics(self) -> Dict:
        """Get comprehensive alert statistics."""
        try:
            current_time = datetime.now()
            
            # Active alerts statistics
            active_alerts = list(self.active_alerts.values())
            active_count = len([a for a in active_alerts if a.get('status') == 'active'])
            
            severity_counts = {'critical': 0, 'warning': 0, 'info': 0}
            for alert in active_alerts:
                if alert.get('status') == 'active':
                    severity = alert.get('severity', 'info')
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Historical statistics (last 24 hours)
            day_ago = current_time - timedelta(days=1)
            recent_alerts = [
                a for a in self.alert_history
                if datetime.fromisoformat(a['timestamp']) > day_ago
            ]
            
            # Alert rate
            alert_rate_per_hour = len(recent_alerts) / 24 if recent_alerts else 0
            
            # Top alerting rules
            rule_counts = defaultdict(int)
            for alert in recent_alerts:
                rule_counts[alert.get('rule_name', 'unknown')] += 1
            
            top_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Escalation statistics
            escalated_alerts = [a for a in recent_alerts if a.get('escalation_level', 0) > 0]
            
            return {
                'current_status': {
                    'active_alerts': active_count,
                    'severity_breakdown': severity_counts,
                    'escalated_alerts': len(self.escalation_timers)
                },
                'recent_activity': {
                    'alerts_last_24h': len(recent_alerts),
                    'alert_rate_per_hour': round(alert_rate_per_hour, 2),
                    'escalations_last_24h': len(escalated_alerts)
                },
                'top_alerting_rules': [
                    {'rule_name': rule, 'count': count} for rule, count in top_rules
                ],
                'suppressed_alerts': len(self.suppressed_alerts)
            }
            
        except Exception as e:
            self.logger.error(f"Alert statistics error: {e}")
            return {}
    
    def stop(self):
        """Stop the alert manager."""
        try:
            self.should_stop.set()
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5)
            
            # Cancel all escalation timers
            for alert_id, timers in self.escalation_timers.items():
                for timer in timers:
                    timer.cancel()
            
            self.logger.info("Alert manager stopped")
            
        except Exception as e:
            self.logger.error(f"Alert manager stop error: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize alert manager
    alert_manager = AlertManager()
    
    print("=== DIGITAL TWIN ALERT MANAGER DEMO ===\n")
    
    # Sample data with various alert conditions
    sample_data_critical = {
        'temperature': 98,  # Critical temperature
        'vibration': 0.8,   # High vibration
        'efficiency': 45,   # Low efficiency
        'health_score': 0.25,  # Critical health score
        'anomaly_score': 0.85   # High anomaly score
    }
    
    sample_data_warning = {
        'temperature': 85,  # Warning temperature
        'vibration': 0.3,   # Normal vibration
        'efficiency': 65,   # Below optimal efficiency
        'health_score': 0.55,  # Warning health score
        'anomaly_score': 0.4    # Normal anomaly score
    }
    
    sample_data_normal = {
        'temperature': 70,  # Normal temperature
        'vibration': 0.2,   # Normal vibration
        'efficiency': 85,   # Good efficiency
        'health_score': 0.85,  # Good health score
        'anomaly_score': 0.1    # Low anomaly score
    }
    
    print("1. Testing Critical Alert Conditions...")
    critical_alerts = alert_manager.evaluate_conditions(sample_data_critical, device_id="device_001")
    print(f"   Critical alerts triggered: {len(critical_alerts)}")
    for alert in critical_alerts:
        print(f"   - {alert['severity'].upper()}: {alert['description']}")
    print()
    
    # Wait for processing
    import time
    time.sleep(2)
    
    print("2. Testing Warning Alert Conditions...")
    warning_alerts = alert_manager.evaluate_conditions(sample_data_warning, device_id="device_002")
    print(f"   Warning alerts triggered: {len(warning_alerts)}")
    for alert in warning_alerts:
        print(f"   - {alert['severity'].upper()}: {alert['description']}")
    print()
    
    time.sleep(2)
    
    print("3. Testing Normal Conditions...")
    normal_alerts = alert_manager.evaluate_conditions(sample_data_normal, device_id="device_003")
    print(f"   Normal condition alerts triggered: {len(normal_alerts)}")
    print()
    
    time.sleep(1)
    
    print("4. Current Active Alerts:")
    active_alerts = alert_manager.get_active_alerts()
    print(f"   Total active alerts: {len(active_alerts)}")
    for alert in active_alerts[:5]:  # Show first 5
        print(f"   - [{alert['severity'].upper()}] {alert['description']} (Device: {alert.get('device_id', 'N/A')})")
    print()
    
    print("5. Alert Statistics:")
    stats = alert_manager.get_alert_statistics()
    print(f"   Current active alerts: {stats['current_status']['active_alerts']}")
    print(f"   Severity breakdown: {stats['current_status']['severity_breakdown']}")
    print(f"   Recent activity (24h): {stats['recent_activity']['alerts_last_24h']} alerts")
    print(f"   Alert rate: {stats['recent_activity']['alert_rate_per_hour']} alerts/hour")
    print()
    
    print("6. Testing Alert Management...")
    if active_alerts:
        test_alert_id = active_alerts[0]['id']
        
        # Test acknowledgment
        print(f"   Acknowledging alert: {test_alert_id}")
        success = alert_manager.acknowledge_alert(test_alert_id, "test_user")
        print(f"   Acknowledgment successful: {success}")
        
        # Test resolution
        print(f"   Resolving alert: {test_alert_id}")
        success = alert_manager.resolve_alert(test_alert_id, "test_user")
        print(f"   Resolution successful: {success}")
    print()
    
    print("7. Final Statistics:")
    final_stats = alert_manager.get_alert_statistics()
    print(f"   Remaining active alerts: {final_stats['current_status']['active_alerts']}")
    print()
    
    # Stop the alert manager
    alert_manager.stop()
    
    print("=== ALERT MANAGER DEMO COMPLETED ===")
    print("\nKey Features Demonstrated:")
    print("- Rule-based alert evaluation")
    print("- Health score and anomaly detection alerts")
    print("- Alert suppression and rate limiting")
    print("- Alert correlation and grouping")
    print("- Notification routing (placeholder implementations)")
    print("- Escalation management")
    print("- Alert acknowledgment and resolution")
    print("- Comprehensive statistics and monitoring")