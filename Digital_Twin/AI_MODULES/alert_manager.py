# Digital_Twin/AI_MODULES/alert_manager.py
import numpy as np
import pandas as pd
import logging
import json
import smtplib
import time
import sys # Added for sys.stdout
import os # Added for path handling
from datetime import datetime, timedelta, timezone # Added timezone
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
import unittest # Added for unit tests
from unittest.mock import patch, MagicMock # Added for mocking

warnings.filterwarnings('ignore')

# Notification libraries
import requests  # For webhooks and Slack notifications

# --- Helper Function for Environment Variables ---
def get_env_var(var_name: str, default: Any = None) -> Any:
    """Gets an environment variable, returning a default if not set."""
    return os.environ.get(var_name, default)

def get_list_env_var(var_name: str, default: List = None, separator: str = ',') -> List:
    """Gets a list environment variable (comma-separated)."""
    value = os.environ.get(var_name)
    if value:
        return [item.strip() for item in value.split(separator)]
    return default if default is not None else []

# --- Standardized Logging Setup ---
def setup_alert_manager_logging(log_dir="LOGS", level=logging.INFO):
    """Setup standardized logging for the Alert Manager."""
    logger = logging.getLogger('AlertManager')
    
    # Prevent duplicate handlers if called multiple times
    if logger.hasHandlers() and len(logger.handlers) > 0:
        return logger

    logger.setLevel(level)
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    log_file = log_path / 'digital_twin_alerts.log'
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.propagate = False # Prevent double logging in Flask setup
    logger.info("AlertManager logging initialized.")
    return logger

class AlertManager:
    """
    Advanced alert management system for Digital Twin applications.
    Handles real-time monitoring, intelligent alerting, notification routing,
    alert correlation, escalation, and suppression.
    """
    
    def __init__(self, config_path: str = "CONFIG/alert_config.json"):
        self.config_path = Path(config_path)
        # --- Use standardized logger ---
        self.logger = setup_alert_manager_logging() 
        
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
        self.alert_rules = self.config.get('configuration', {}).get('alert_rules', {})
        self.thresholds = self.config.get('configuration', {}).get('thresholds', {})
        self.notification_channels = self.config.get('configuration', {}).get('notification_channels', {})
        
        # Alert correlation and grouping
        self.correlation_window = timedelta(minutes=self.config.get('configuration', {}).get('correlation_rules', {}).get('window_minutes', 5))
        self.correlation_rules = self.config.get('configuration', {}).get('correlation_rules', {})
        
        # Rate limiting and suppression
        self.rate_limits = defaultdict(list)
        self.suppression_rules = self.config.get('configuration', {}).get('suppression_rules', {})
        
        # Escalation management
        self.escalation_policies = self.config.get('configuration', {}).get('escalation_policies', {})
        self.escalation_timers = {} # Store Timer objects
        
        # --- Added Notification Retry Logic ---
        self.notification_retry_queue = queue.Queue()
        self.notification_retry_thread = None
        self.max_notification_retries = 3
        self.notification_retry_delay = 60 # seconds
        # --- End Added ---
        
        # Notification handlers
        self.notification_handlers = {
            'email': self._send_email_notification,
            'slack': self._send_slack_notification,
            'webhook': self._send_webhook_notification
            # 'sms': self._send_sms_notification # SMS needs specific setup
        }
        
        # Start background processing threads
        self._start_alert_processing()
        self._start_notification_retry_thread()
        
    def _setup_logging(self):
        """Deprecated: Use setup_alert_manager_logging function instead."""
        # Ensure logger exists even if called directly initially
        if not hasattr(self, 'logger'):
             self.logger = setup_alert_manager_logging()
        return self.logger
    
    def _load_config(self) -> Dict:
        """Load alert manager configuration."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                self.logger.info("Alert configuration loaded", path=str(self.config_path))
                # Resolve environment variables referenced in config
                self._resolve_env_vars_in_config(config_data)
                return config_data
            else:
                self.logger.warning(f"Configuration file not found: {self.config_path}. Using defaults.")
                default_config = {
                    "log_level": "INFO",
                    "output_dir": "./output",
                    "retry_count": 3
                }
                return default_config

                
        except json.JSONDecodeError as e:
            self.logger.error("Failed to parse configuration file", path=str(self.config_path), error=str(e))
            return {} # Return empty dict on failure
        except Exception as e:
            self.logger.error("Failed to load configuration", path=str(self.config_path), error=str(e))
            return {}

    def _resolve_env_vars_in_config(self, config_node: Any):
        """Recursively resolve 'env_VARNAME' strings in config dictionary/list."""
        if isinstance(config_node, dict):
            for key, value in config_node.items():
                if isinstance(value, str):
                    if value.startswith("env_"):
                        env_var_name = value[4:]
                        resolved_value = get_env_var(env_var_name)
                        if resolved_value is not None:
                            config_node[key] = resolved_value
                            self.logger.debug(f"Resolved config key '{key}' from env var '{env_var_name}'")
                        else:
                            self.logger.warning(f"Environment variable '{env_var_name}' referenced in config not found for key '{key}'. Keeping placeholder.")
                    # Handle comma-separated lists from env vars
                    elif key.endswith("_env_var") and isinstance(config_node.get(key[:-8]), list):
                        env_var_name = value
                        resolved_list = get_list_env_var(env_var_name)
                        if resolved_list:
                            config_node[key[:-8]] = resolved_list # Overwrite the list key
                            self.logger.debug(f"Resolved list config key '{key[:-8]}' from env var '{env_var_name}'")
                        else:
                            self.logger.warning(f"List environment variable '{env_var_name}' not found for key '{key[:-8]}'. Keeping default/empty list.")

                elif isinstance(value, (dict, list)):
                    self._resolve_env_vars_in_config(value)
        elif isinstance(config_node, list):
            for i, item in enumerate(config_node):
                if isinstance(item, str) and item.startswith("env_"):
                     env_var_name = item[4:]
                     resolved_value = get_env_var(env_var_name)
                     if resolved_value is not None:
                         config_node[i] = resolved_value
                     else:
                          self.logger.warning(f"Env var '{env_var_name}' in list not found.")
                elif isinstance(item, (dict, list)):
                    self._resolve_env_vars_in_config(item)

    def _start_alert_processing(self):
        """Start background alert processing thread."""
        if self.processing_thread and self.processing_thread.is_alive():
             self.logger.debug("Alert processing thread already running.")
             return
        try:
            self.processing_thread = threading.Thread(target=self._process_alerts, daemon=True)
            self.processing_thread.start()
            self.logger.info("Alert processing thread started")
        except Exception as e:
            self.logger.exception("Failed to start alert processing thread")

    # --- Added Notification Retry Thread ---
    def _start_notification_retry_thread(self):
        """Start background thread for retrying failed notifications."""
        if self.notification_retry_thread and self.notification_retry_thread.is_alive():
            self.logger.debug("Notification retry thread already running.")
            return
        try:
            self.notification_retry_thread = threading.Thread(target=self._process_notification_retries, daemon=True)
            self.notification_retry_thread.start()
            self.logger.info("Notification retry thread started")
        except Exception as e:
            self.logger.exception("Failed to start notification retry thread")
    # --- End Added ---

    def _process_alerts(self):
        """Main alert processing loop."""
        self.logger.info("Alert processing loop entering.")
        while not self.should_stop.is_set():
            try:
                alert = self.alert_queue.get(timeout=1)
                self.logger.debug("Processing alert from queue", alert_id=alert.get('id'))
                self._handle_alert(alert)
                self.alert_queue.task_done()
            except queue.Empty:
                continue # Normal timeout, continue loop
            except Exception as e:
                self.logger.exception("Unexpected error in alert processing loop")
        self.logger.info("Alert processing loop exiting.")

    # --- Added Notification Retry Processing ---
    def _process_notification_retries(self):
        """Process failed notifications from the retry queue."""
        self.logger.info("Notification retry loop entering.")
        while not self.should_stop.is_set():
            try:
                channel, alert, attempt = self.notification_retry_queue.get(timeout=1)
                self.logger.debug(f"Retrying notification", channel=channel, alert_id=alert.get('id'), attempt=attempt)

                if attempt >= self.max_notification_retries:
                    self.logger.error(f"Max retries reached for notification", channel=channel, alert_id=alert.get('id'))
                    self.notification_retry_queue.task_done()
                    continue

                try:
                    handler = self.notification_handlers.get(channel)
                    if handler:
                        success = handler(alert, is_retry=True) # Pass retry flag
                        if success:
                            self.logger.info(f"Notification retry successful", channel=channel, alert_id=alert.get('id'), attempt=attempt)
                            # Update alert history/status if needed
                        else:
                            # Re-queue with increased attempt count and delay
                            self.logger.warning(f"Notification retry failed, requeueing", channel=channel, alert_id=alert.get('id'), attempt=attempt + 1)
                            # Use a Timer for delay to avoid blocking the retry thread
                            retry_delay = self.notification_retry_delay * (2 ** attempt) # Exponential backoff
                            threading.Timer(retry_delay, self.notification_retry_queue.put, args=[(channel, alert, attempt + 1)]).start()
                    else:
                        self.logger.error(f"No handler found for retry", channel=channel, alert_id=alert.get('id'))

                except Exception as e:
                    self.logger.exception(f"Error during notification retry execution")
                    # Re-queue after error
                    retry_delay = self.notification_retry_delay * (2 ** attempt)
                    threading.Timer(retry_delay, self.notification_retry_queue.put, args=[(channel, alert, attempt + 1)]).start()

                self.notification_retry_queue.task_done()

            except queue.Empty:
                continue # Normal timeout
            except Exception as e:
                self.logger.exception("Unexpected error in notification retry loop")
        self.logger.info("Notification retry loop exiting.")
    # --- End Added ---

    def evaluate_conditions(self, data: Dict, device_id: str = None) -> List[Dict]:
        """Evaluate alert conditions against current data."""
        triggered_alerts = []
        current_time = datetime.now(timezone.utc) # Use timezone-aware UTC
        
        # Evaluate each alert rule
        for rule_name, rule_config in self.alert_rules.items():
            try:
                alert = self._evaluate_single_rule(rule_name, rule_config, data, device_id, current_time)
                if alert:
                    triggered_alerts.append(alert)
            except Exception as e:
                self.logger.exception("Error evaluating rule", rule_name=rule_name, device_id=device_id)

        # Evaluate health score thresholds
        if 'health_score' in data:
            try:
                health_alerts = self._evaluate_health_score_alerts(data, device_id, current_time)
                triggered_alerts.extend(health_alerts)
            except Exception as e:
                self.logger.exception("Error evaluating health score alerts", device_id=device_id)
        
        # Evaluate anomaly/failure alerts
        if 'anomaly_score' in data or 'failure_probability' in data:
             try:
                 pred_alerts = self._evaluate_predictive_alerts(data, device_id, current_time)
                 triggered_alerts.extend(pred_alerts)
             except Exception as e:
                 self.logger.exception("Error evaluating predictive alerts", device_id=device_id)

        # Queue triggered alerts for processing
        for alert in triggered_alerts:
            self.alert_queue.put(alert)
        
        if triggered_alerts:
             self.logger.info(f"Conditions evaluated", triggered_count=len(triggered_alerts), device_id=device_id)
        else:
             self.logger.debug(f"Conditions evaluated, no alerts triggered.", device_id=device_id)
             
        return triggered_alerts

    def _evaluate_single_rule(self, rule_name: str, rule_config: Dict, data: Dict, 
                              device_id: str, current_time: datetime) -> Optional[Dict]:
        """Evaluate a single alert rule."""
        metric = rule_config.get('metric')
        operator = rule_config.get('operator')
        threshold = rule_config.get('threshold')
        duration_sec = rule_config.get('duration', 0) # Duration in seconds
        severity = rule_config.get('severity', 'info')
        
        if metric not in data or data[metric] is None:
            # Metric not present in data, clear any existing alert state
            alert_key = f"{rule_name}_{device_id or 'system'}"
            if alert_key in self.active_alerts:
                self._resolve_alert(alert_key, resolved_by="condition_not_met")
            return None
        
        try:
             current_value = float(data[metric]) # Ensure numeric comparison
        except (ValueError, TypeError):
             self.logger.warning(f"Non-numeric value for metric", metric=metric, value=data[metric], rule=rule_name)
             return None # Cannot evaluate non-numeric

        condition_met = self._check_condition(current_value, operator, threshold)
        alert_key = f"{rule_name}_{device_id or 'system'}"

        if not condition_met:
            # Condition no longer met, resolve if active
            if alert_key in self.active_alerts and self.active_alerts[alert_key].get('status') == 'active':
                self._resolve_alert(alert_key, resolved_by="condition_cleared")
            return None
        
        # Condition is met
        if duration_sec > 0:
            if alert_key in self.active_alerts:
                # Alert already exists, check duration
                alert_state = self.active_alerts[alert_key]
                if alert_state.get('status') == 'active':
                     return None # Already active and notified (or escalated)
                     
                first_triggered_str = alert_state.get('first_triggered')
                if first_triggered_str:
                    first_triggered = datetime.fromisoformat(first_triggered_str)
                    if (current_time - first_triggered).total_seconds() >= duration_sec:
                        # Duration met, trigger the alert fully
                        pass # Continue to create alert dict below
                    else:
                        # Duration not yet met, keep tracking
                        self.logger.debug("Duration condition not yet met", alert_key=alert_key)
                        return None 
                else:
                     # Should not happen if state exists, but handle defensively
                     alert_state['first_triggered'] = current_time.isoformat()
                     self.logger.debug("Tracking duration start", alert_key=alert_key)
                     return None
            else:
                # First time condition met, start tracking duration
                self.active_alerts[alert_key] = {
                    'rule_name': rule_name,
                    'device_id': device_id,
                    'metric': metric,
                    'first_triggered': current_time.isoformat(),
                    'status': 'pending_duration' # New status
                }
                self.logger.debug("Tracking duration start", alert_key=alert_key)
                return None
        
        # Condition met (and duration met or not required)
        # Check if alert is already active and notified to prevent re-triggering notifications
        if alert_key in self.active_alerts and self.active_alerts[alert_key].get('status') == 'active':
             self.logger.debug("Alert already active", alert_key=alert_key)
             return None # Don't re-queue if already active

        alert = {
            'id': f"ALERT_{rule_name}_{device_id or 'system'}_{int(current_time.timestamp())}",
            'rule_name': rule_name,
            'device_id': device_id,
            'metric': metric,
            'current_value': current_value,
            'threshold': threshold,
            'operator': operator,
            'severity': severity,
            'description': rule_config.get('description', f'{metric} {operator} {threshold}'),
            'timestamp': current_time.isoformat(),
            'status': 'active', # Will be set to active when handled
            'acknowledged': False,
            'source': 'rule_evaluation',
            'notifications_sent': [], # Initialize notification tracking
            'first_triggered': self.active_alerts.get(alert_key, {}).get('first_triggered', current_time.isoformat()) # Keep original trigger time
        }
        
        # Update state immediately for duration tracking or if re-triggered
        self.active_alerts[alert_key] = alert.copy() # Store a copy
        
        return alert

    def _check_condition(self, value: float, operator: str, threshold: float) -> bool:
        """Check if a condition is met."""
        op_map: Dict[str, Callable[[float, float], bool]] = {
            '>': lambda v, t: v > t,
            '<': lambda v, t: v < t,
            '>=': lambda v, t: v >= t,
            '<=': lambda v, t: v <= t,
            '==': lambda v, t: abs(v - t) < 1e-9, # Use tolerance for float equality
            '!=': lambda v, t: abs(v - t) >= 1e-9
        }
        if operator in op_map:
            try:
                return op_map[operator](value, threshold)
            except (TypeError, ValueError):
                 self.logger.warning("Type error during condition check", value=value, op=operator, threshold=threshold)
                 return False
        else:
            self.logger.warning(f"Unknown operator: {operator}")
            return False

    def _evaluate_health_score_alerts(self, data: Dict, device_id: str, current_time: datetime) -> List[Dict]:
        """Evaluate health score-based alerts."""
        alerts = []
        health_score = data.get('health_score')
        if health_score is None: return []

        try:
             health_score = float(health_score)
        except (ValueError, TypeError):
             self.logger.warning("Invalid health score value", value=health_score, device_id=device_id)
             return []

        # Use thresholds from config, provide defaults
        crit_thresh = self.thresholds.get('health_score_critical', 0.3)
        warn_thresh = self.thresholds.get('health_score_warning', 0.6)

        if health_score <= crit_thresh:
            rule_name = 'health_score_critical'
            severity = 'critical'
            threshold = crit_thresh
            operator = '<='
            description = f'Device health score critically low: {health_score:.3f}'
        elif health_score <= warn_thresh:
            rule_name = 'health_score_warning'
            severity = 'warning'
            threshold = warn_thresh
            operator = '<='
            description = f'Device health score below normal: {health_score:.3f}'
        else:
            # Health is OK, resolve any active health score alerts for this device
            self._resolve_alert(f"health_score_critical_{device_id or 'system'}", "health_improved")
            self._resolve_alert(f"health_score_warning_{device_id or 'system'}", "health_improved")
            return [] # No alert needed

        alert_key = f"{rule_name}_{device_id or 'system'}"
        if alert_key in self.active_alerts and self.active_alerts[alert_key].get('status') == 'active':
             return [] # Already active

        alert = {
            'id': f"ALERT_{rule_name}_{device_id or 'system'}_{int(current_time.timestamp())}",
            'rule_name': rule_name,
            'device_id': device_id,
            'metric': 'health_score',
            'current_value': health_score,
            'threshold': threshold,
            'operator': operator,
            'severity': severity,
            'description': description,
            'timestamp': current_time.isoformat(),
            'status': 'active',
            'acknowledged': False,
            'source': 'health_evaluation',
            'notifications_sent': [],
            'first_triggered': self.active_alerts.get(alert_key, {}).get('first_triggered', current_time.isoformat())
        }
        self.active_alerts[alert_key] = alert.copy()
        alerts.append(alert)
        return alerts

    def _evaluate_predictive_alerts(self, data: Dict, device_id: str, current_time: datetime) -> List[Dict]:
        """Evaluate anomaly and failure prediction alerts."""
        alerts = []
        
        # Anomaly Score Alert
        anomaly_score = data.get('anomaly_score')
        if anomaly_score is not None:
             try:
                 anomaly_score = float(anomaly_score)
                 anomaly_threshold = self.thresholds.get('anomaly_threshold', 0.8) # Configurable?
                 rule_name = 'anomaly_detection'
                 alert_key = f"{rule_name}_{device_id or 'system'}"
                 
                 if abs(anomaly_score) >= anomaly_threshold: # Check if score exceeds threshold
                      if alert_key not in self.active_alerts or self.active_alerts[alert_key].get('status') != 'active':
                          alerts.append({
                              'id': f"ALERT_{rule_name}_{device_id or 'system'}_{int(current_time.timestamp())}",
                              'rule_name': rule_name, 'device_id': device_id, 'metric': 'anomaly_score',
                              'current_value': anomaly_score, 'threshold': anomaly_threshold, 'operator': '>=',
                              'severity': 'warning', 'description': f'Anomaly detected with score: {anomaly_score:.3f}',
                              'timestamp': current_time.isoformat(), 'status': 'active', 'acknowledged': False,
                              'source': 'predictive_analysis', 'notifications_sent': [],
                              'first_triggered': self.active_alerts.get(alert_key, {}).get('first_triggered', current_time.isoformat())
                          })
                          self.active_alerts[alert_key] = alerts[-1].copy() # Add/update state
                 else:
                      # Anomaly condition cleared
                      self._resolve_alert(alert_key, "anomaly_cleared")
             except (ValueError, TypeError):
                  self.logger.warning("Invalid anomaly score value", value=anomaly_score, device_id=device_id)

        # Failure Probability Alert
        failure_prob = data.get('failure_probability')
        if failure_prob is not None:
            try:
                failure_prob = float(failure_prob)
                prob_threshold = self.thresholds.get('failure_probability_high', 0.7)
                rule_name = 'high_failure_probability'
                alert_key = f"{rule_name}_{device_id or 'system'}"

                if failure_prob >= prob_threshold:
                    if alert_key not in self.active_alerts or self.active_alerts[alert_key].get('status') != 'active':
                        alerts.append({
                            'id': f"ALERT_{rule_name}_{device_id or 'system'}_{int(current_time.timestamp())}",
                            'rule_name': rule_name, 'device_id': device_id, 'metric': 'failure_probability',
                            'current_value': failure_prob, 'threshold': prob_threshold, 'operator': '>=',
                            'severity': 'critical', 'description': f'High failure probability detected: {failure_prob:.3f}',
                            'timestamp': current_time.isoformat(), 'status': 'active', 'acknowledged': False,
                            'source': 'predictive_analysis', 'notifications_sent': [],
                            'first_triggered': self.active_alerts.get(alert_key, {}).get('first_triggered', current_time.isoformat())
                        })
                        self.active_alerts[alert_key] = alerts[-1].copy()
                else:
                    # Failure condition cleared
                    self._resolve_alert(alert_key, "failure_risk_reduced")
            except (ValueError, TypeError):
                 self.logger.warning("Invalid failure probability value", value=failure_prob, device_id=device_id)

        return alerts
    
    def _handle_alert(self, alert: Dict):
        """Handle a triggered alert through the processing pipeline."""
        alert_id = alert.get('id', 'unknown_id')
        self.logger.debug("Handling alert", alert_id=alert_id)
        
        # Check suppression
        if self._should_suppress_alert(alert):
            self.logger.info("Alert suppressed", alert_id=alert_id, rule=alert.get('rule_name'), device=alert.get('device_id'))
            return

        # Apply correlation
        correlated_alert = self._apply_correlation_rules(alert)
        if correlated_alert['id'] != alert['id']:
             self.logger.info("Alert correlated", original_id=alert_id, new_id=correlated_alert['id'])
             alert = correlated_alert # Use the correlated alert going forward

        # Check rate limiting
        if self._is_rate_limited(alert):
            self.logger.info("Alert rate limited", alert_id=alert['id'], rule=alert.get('rule_name'), device=alert.get('device_id'))
            return
            
        # Store alert state (before notifications in case they fail)
        self._store_alert(alert) 

        # Send initial notifications
        self._send_notifications(alert, level=0) # Send level 0 notifications

        # Setup escalation timers
        self._setup_escalation(alert)
            
        self.logger.info("Alert processed successfully", alert_id=alert['id'], severity=alert['severity'])

    def _should_suppress_alert(self, alert: Dict) -> bool:
        """Check if alert should be suppressed based on rules."""
        current_time = datetime.now(timezone.utc)
        duplicate_window = timedelta(minutes=self.suppression_rules.get('duplicate_window_minutes', 5))
        burst_limit = self.suppression_rules.get('burst_limit', 10)
        burst_window = timedelta(minutes=self.suppression_rules.get('burst_window_minutes', 1))
        alert_key = f"{alert['rule_name']}_{alert.get('device_id', 'system')}"

        # Check for active acknowledged/resolved alert of the same type
        if alert_key in self.active_alerts:
             state = self.active_alerts[alert_key]
             if state.get('acknowledged') or state.get('status') == 'resolved':
                  # Allow re-trigger after cooldown (e.g., 1 hour)
                  last_event_time_str = state.get('resolved_at') or state.get('acknowledged_at') or state.get('timestamp')
                  if last_event_time_str:
                      last_event_time = datetime.fromisoformat(last_event_time_str)
                      if current_time - last_event_time < timedelta(hours=1):
                           self.logger.debug("Suppressed: Recently acknowledged/resolved", alert_key=alert_key)
                           return True

        # Check duplicate time window
        if alert_key in self.suppressed_alerts:
            last_suppressed = datetime.fromisoformat(self.suppressed_alerts[alert_key]['last_suppressed'])
            if current_time - last_suppressed < duplicate_window:
                 self.logger.debug("Suppressed: Duplicate within window", alert_key=alert_key)
                 return True # Suppress as duplicate

        # Check burst rate
        recent_alert_timestamps = [
            datetime.fromisoformat(a['timestamp']) for a in self.alert_history
            if a.get('rule_name') == alert['rule_name'] and a.get('device_id') == alert.get('device_id')
        ]
        alerts_in_burst_window = [ts for ts in recent_alert_timestamps if current_time - ts <= burst_window]

        if len(alerts_in_burst_window) >= burst_limit:
            self.logger.warning("Suppressed: Burst limit reached", alert_key=alert_key, count=len(alerts_in_burst_window), limit=burst_limit)
            self.suppressed_alerts[alert_key] = {
                'last_suppressed': current_time.isoformat(),
                'reason': 'burst_limit'
            }
            return True

        # Not suppressed, update last seen time for duplicate check
        self.suppressed_alerts[alert_key] = {'last_suppressed': current_time.isoformat(), 'reason': 'not_suppressed'}
        return False

    def _apply_correlation_rules(self, alert: Dict) -> Dict:
        """Apply correlation rules to potentially group related alerts."""
        current_time = datetime.now(timezone.utc)
        window_start = current_time - self.correlation_window
        
        # Find potentially related active alerts (same device, within window, different metric)
        related_active = []
        for active_alert_id, active_alert in self.active_alerts.items():
            if active_alert.get('status') == 'active' and \
               active_alert.get('device_id') == alert.get('device_id') and \
               active_alert['metric'] != alert['metric'] and \
               datetime.fromisoformat(active_alert['timestamp']) >= window_start:
                related_active.append(active_alert)

        if not related_active:
            return alert # No active related alerts to correlate with

        # Check defined correlation rules
        for rule_name, rule_config in self.correlation_rules.items():
             required_metrics = set(rule_config.get('metrics', []))
             current_metrics = {alert['metric']} | {a['metric'] for a in related_active}
             
             # If the current alert + active related alerts satisfy a rule's metrics
             if required_metrics.issubset(current_metrics):
                  self.logger.info("Correlation rule matched", rule_name=rule_name, device_id=alert.get('device_id'))
                  # Create a new correlated alert or update an existing one
                  # For simplicity, create a new one referencing the others
                  correlated_id = f"CORR_{rule_name}_{alert.get('device_id', 'system')}_{int(current_time.timestamp())}"
                  correlated_alert = {
                      'id': correlated_id,
                      'rule_name': f'correlated_{rule_name}',
                      'device_id': alert.get('device_id'),
                      'metric': ', '.join(sorted(required_metrics)), # Combine metrics
                      'current_value': None, # Value doesn't make sense for combined
                      'threshold': None,
                      'operator': None,
                      'severity': 'critical' if any(a['severity'] == 'critical' for a in [alert] + related_active) else 'warning',
                      'description': f"Correlated event involving: {', '.join(sorted(required_metrics))}",
                      'timestamp': current_time.isoformat(),
                      'status': 'active',
                      'acknowledged': False,
                      'source': 'correlation_engine',
                      'notifications_sent': [],
                      'correlated_alerts': [alert['id']] + [a['id'] for a in related_active],
                      'first_triggered': min([alert['timestamp']] + [a['timestamp'] for a in related_active])
                  }
                  # Optionally resolve the individual alerts being correlated
                  # self._resolve_alert(alert['id'], resolved_by=f"correlated_to_{correlated_id}")
                  # for related in related_active:
                  #     self._resolve_alert(related['id'], resolved_by=f"correlated_to_{correlated_id}")
                      
                  return correlated_alert # Return the new correlated alert

        return alert # No correlation rule matched

    def _is_rate_limited(self, alert: Dict) -> bool:
        """Check alert rate limiting based on rule and device."""
        current_time = time.monotonic() # Use monotonic clock for rate limiting intervals
        # Example: Limit same rule on same device to once per minute
        rate_limit_window = 60 # seconds
        rate_limit_max = 1 # count
        
        rate_key = f"{alert['rule_name']}_{alert.get('device_id', 'system')}"
        
        # Clean old timestamps
        timestamps = self.rate_limits[rate_key]
        while timestamps and timestamps[0] < current_time - rate_limit_window:
            timestamps.pop(0)
            
        if len(timestamps) >= rate_limit_max:
            return True # Rate limited
        
        timestamps.append(current_time)
        return False

    def _store_alert(self, alert: Dict):
        """Store alert in active alerts and history."""
        alert_id = alert['id']
        # Store in active alerts (overwrites if ID exists, which is intended on re-trigger before resolve)
        self.active_alerts[alert_id] = alert.copy() # Store a copy
        
        # Add to history
        self.alert_history.append(alert.copy())
        
        # Basic cleanup of old *resolved* alerts from active_alerts to prevent memory leak
        # More robust cleanup could run periodically
        if len(self.active_alerts) > 5000: # Example limit
            resolved_cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
            keys_to_remove = [
                k for k, v in self.active_alerts.items()
                if v.get('status') == 'resolved' and datetime.fromisoformat(v.get('resolved_at', v['timestamp'])) < resolved_cutoff
            ]
            for key in keys_to_remove:
                del self.active_alerts[key]
            if keys_to_remove:
                 self.logger.info("Cleaned up old resolved alerts from active memory", count=len(keys_to_remove))

    def _send_notifications(self, alert: Dict, level: int = 0):
        """Send notifications based on alert severity and escalation level."""
        severity = alert['severity']
        policy_name = 'critical' if severity == 'critical' else 'default'
        policy = self.escalation_policies.get(policy_name, self.escalation_policies.get('default'))

        if not policy or not policy.get('levels'):
             self.logger.warning("No valid escalation policy found", policy_name=policy_name)
             return

        # Ensure level is within bounds
        level = min(level, len(policy['levels']) - 1)
        
        channels_to_notify = policy['levels'][level].get('channels', [])
        
        self.logger.info("Sending notifications", alert_id=alert['id'], level=level, channels=channels_to_notify)
        
        notification_successful = True # Track overall success for this level
        for channel in channels_to_notify:
            channel_config = self.notification_channels.get(channel, {})
            if channel_config.get('enabled', False):
                handler = self.notification_handlers.get(channel)
                if handler:
                    try:
                        success = handler(alert) # Handler returns True/False
                        if success:
                             # Record successful notification timestamp
                             alert['notifications_sent'].append({
                                 'channel': channel,
                                 'timestamp': datetime.now(timezone.utc).isoformat(),
                                 'level': level
                             })
                        else:
                             # --- Queue for retry ---
                             self.logger.warning(f"Initial notification failed, queueing retry", channel=channel, alert_id=alert['id'])
                             self.notification_retry_queue.put((channel, alert.copy(), 1)) # attempt 1
                             notification_successful = False # Mark overall as failed for this level
                    except Exception as e:
                        self.logger.exception(f"Notification handler failed unexpectedly", channel=channel, alert_id=alert['id'])
                        self.notification_retry_queue.put((channel, alert.copy(), 1)) # Retry on exception too
                        notification_successful = False
                else:
                    self.logger.warning(f"No handler configured for channel", channel=channel)
            else:
                 self.logger.debug(f"Channel disabled", channel=channel)
                 
        # Update alert state in active_alerts with notification info
        if alert['id'] in self.active_alerts:
             self.active_alerts[alert['id']]['notifications_sent'] = alert['notifications_sent']

    def _setup_escalation(self, alert: Dict):
        """Setup escalation timers for an active, unacknowledged alert."""
        alert_id = alert['id']
        severity = alert['severity']
        
        # Clear any existing timers for this alert ID first
        if alert_id in self.escalation_timers:
            for timer in self.escalation_timers[alert_id]:
                timer.cancel()
            del self.escalation_timers[alert_id]

        policy_name = 'critical' if severity == 'critical' else 'default'
        policy = self.escalation_policies.get(policy_name, self.escalation_policies.get('default'))

        if not policy or not policy.get('levels'):
             self.logger.warning("No valid escalation policy for setup", policy_name=policy_name)
             return

        self.escalation_timers[alert_id] = []
        cumulative_delay = 0
        
        # Start from level 1 (level 0 notifications already sent)
        for level_index, level_config in enumerate(policy['levels'][1:], 1):
             delay_minutes = level_config.get('delay_minutes', 15) # Delay *since the previous level*
             # Calculate total delay from alert trigger time
             total_delay_seconds = delay_minutes * 60 
             
             self.logger.debug(f"Scheduling escalation", alert_id=alert_id, level=level_index, delay_seconds=total_delay_seconds)

             # Schedule escalation using threading.Timer
             timer = threading.Timer(
                 total_delay_seconds,
                 self._escalate_alert,
                 args=[alert_id, level_index, level_config] # Pass config for this level
             )
             timer.start()
             self.escalation_timers[alert_id].append(timer)
             # Update cumulative delay for next level's baseline (if needed, but simple delay works too)
             # cumulative_delay += delay_minutes 

    def _escalate_alert(self, alert_id: str, level: int, escalation_config: Dict):
        """Escalate an alert to the next level (called by Timer)."""
        self.logger.info(f"Escalation timer triggered", alert_id=alert_id, level=level)
        # Check if alert is still active and unacknowledged
        if alert_id not in self.active_alerts:
             self.logger.info("Escalation cancelled: Alert no longer active", alert_id=alert_id)
             return
             
        alert = self.active_alerts[alert_id]
        if alert.get('acknowledged') or alert.get('status') != 'active':
             self.logger.info("Escalation cancelled: Alert already acknowledged or resolved", alert_id=alert_id)
             return

        # Update alert state (important to do this *before* sending notifications)
        alert['escalation_level'] = level
        alert['last_escalated'] = datetime.now(timezone.utc).isoformat()
        
        # Send notifications for this escalation level
        self._send_notifications(alert, level=level)

        self.logger.warning(f"Alert escalated", alert_id=alert_id, level=level)

    # --- Notification Handlers (with retry logic support) ---
    def _send_email_notification(self, alert: Dict, is_retry: bool = False) -> bool:
        """Send email notification with retry logic."""
        email_config = self.notification_channels.get('email', {})
        if not email_config.get('enabled', False):
            return True # Not enabled, count as success (nothing to do)

        # Resolve recipient list (potentially from env var)
        recipients = email_config.get('recipients', [])

        subject = f"[{alert['severity'].upper()}] Alert: {alert.get('rule_name', 'System Event')} - {alert.get('device_id', 'System')}"
        if alert.get('escalated'): subject = f"[ESCALATED L{alert.get('escalation_level', '?')}] " + subject
        if is_retry: subject = f"[RETRY] " + subject

        body = f"""
        Digital Twin Alert:
        Severity: {alert['severity'].upper()}
        Rule: {alert.get('rule_name', 'N/A')}
        Device: {alert.get('device_id', 'System')}
        Timestamp: {alert['timestamp']}
        Description: {alert.get('description', 'No details')}
        Metric: {alert.get('metric', 'N/A')}
        Value: {alert.get('current_value', 'N/A')}
        Threshold: {alert.get('threshold', 'N/A')} {alert.get('operator', '')}
        """

        sender = email_config.get('username')
        password = email_config.get('password')
        smtp_server = email_config.get('smtp_server', 'smtp.gmail.com')
        smtp_port = email_config.get('smtp_port', 587)

        if not sender or not password or not recipients:
             self.logger.error("Email config incomplete (sender/password/recipients)", alert_id=alert['id'])
             return False # Config error, don't retry indefinitely

        try:
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender, password)
                server.sendmail(sender, recipients, msg.as_string())
            
            self.logger.info("Email notification sent", alert_id=alert['id'], recipients=recipients)
            return True # Success

        except smtplib.SMTPAuthenticationError:
            self.logger.error("SMTP Authentication failed. Check username/password.", alert_id=alert['id'])
            # Don't retry auth errors usually
            return False 
        except smtplib.SMTPConnectError:
             self.logger.error("SMTP Connection failed. Check server/port.", alert_id=alert['id'])
             return False # Likely config error, don't keep retrying
        except Exception as e:
            self.logger.exception("Failed to send email notification", alert_id=alert['id'])
            return False # Failed, allow retry

    def _send_slack_notification(self, alert: Dict, is_retry: bool = False) -> bool:
        """Send Slack notification via webhook."""
        slack_config = self.notification_channels.get('slack', {})
        webhook_url = slack_config.get('webhook_url')
        channel = slack_config.get('channel')

        if not slack_config.get('enabled') or not webhook_url:
            return True # Not configured or enabled

        severity_map = {'critical': 'danger', 'warning': 'warning', 'info': '#439FE0'}
        color = severity_map.get(alert['severity'], '#CCCCCC')
        
        prefix = "[ESCALATED]" if alert.get('escalated') else ""
        if is_retry: prefix = "[RETRY] " + prefix

        payload = {
            "channel": channel,
            "username": "DigitalTwinAlertBot",
            "icon_emoji": ":warning:",
            "attachments": [
                {
                    "fallback": f"{prefix}[{alert['severity'].upper()}] {alert['description']}",
                    "color": color,
                    "title": f"{prefix} Alert: {alert.get('rule_name', 'System Event')}",
                    "fields": [
                        {"title": "Severity", "value": alert['severity'].upper(), "short": True},
                        {"title": "Device", "value": alert.get('device_id', 'System'), "short": True},
                        {"title": "Metric", "value": f"{alert.get('metric', 'N/A')} = {alert.get('current_value', 'N/A')}", "short": True},
                        {"title": "Timestamp", "value": alert['timestamp'], "short": True},
                        {"title": "Description", "value": alert['description'], "short": False}
                    ],
                    "ts": int(datetime.fromisoformat(alert['timestamp']).timestamp())
                }
            ]
        }
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status() # Raise exception for bad status codes
            self.logger.info("Slack notification sent", alert_id=alert['id'])
            return True
        except requests.exceptions.RequestException as e:
            self.logger.error("Failed to send Slack notification", alert_id=alert['id'], error=str(e))
            return False # Failed, allow retry

    def _send_webhook_notification(self, alert: Dict, is_retry: bool = False) -> bool:
        """Send notification to a generic webhook."""
        webhook_config = self.notification_channels.get('webhook', {})
        url = webhook_config.get('url')
        method = webhook_config.get('method', 'POST').upper()

        if not webhook_config.get('enabled') or not url:
            return True

        # Include retry/escalation info in payload
        payload = alert.copy()
        payload['is_retry'] = is_retry
        payload['is_escalated'] = alert.get('escalated', False)

        try:
            if method == 'POST':
                response = requests.post(url, json=payload, timeout=10)
            elif method == 'GET': # GET is less common for notifications
                response = requests.get(url, params=payload, timeout=10)
            else:
                 self.logger.error("Unsupported webhook method", method=method)
                 return False # Config error

            response.raise_for_status()
            self.logger.info("Webhook notification sent", alert_id=alert['id'], url=url, method=method)
            return True
        except requests.exceptions.RequestException as e:
            self.logger.error("Failed to send webhook notification", alert_id=alert['id'], url=url, error=str(e))
            return False # Failed, allow retry

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = None) -> bool:
        """Acknowledge an active alert."""
        if alert_id not in self.active_alerts:
            self.logger.warning("Attempted to acknowledge non-existent or inactive alert", alert_id=alert_id)
            return False

        alert = self.active_alerts[alert_id]
        if alert.get('acknowledged'):
             self.logger.info("Alert already acknowledged", alert_id=alert_id)
             return True # Already done

        alert['acknowledged'] = True
        alert['acknowledged_by'] = acknowledged_by or 'system'
        alert['acknowledged_at'] = datetime.now(timezone.utc).isoformat()
        
        # Cancel any pending escalation timers for this alert
        if alert_id in self.escalation_timers:
            self.logger.info("Cancelling escalation timers for acknowledged alert", alert_id=alert_id)
            for timer in self.escalation_timers[alert_id]:
                timer.cancel()
            del self.escalation_timers[alert_id] # Remove from tracking

        self.logger.info("Alert acknowledged", alert_id=alert_id, user=acknowledged_by)
        # Optionally, notify others about acknowledgment
        # self._send_notification(alert, notification_type='acknowledgment')
        return True

    def resolve_alert(self, alert_id: str, resolved_by: str = None) -> bool:
        """Mark an alert as resolved (condition cleared)."""
        return self._resolve_alert(alert_id, resolved_by)

    def _resolve_alert(self, alert_key_or_id: str, resolved_by: str = None) -> bool:
        """Internal: Resolve alert by key (rule_device) or specific ID."""
        alert = None
        if alert_key_or_id in self.active_alerts:
             alert = self.active_alerts[alert_key_or_id]
        
        # If not found by ID, maybe it was a key for duration tracking
        elif alert_key_or_id in self.active_alerts and self.active_alerts[alert_key_or_id].get('status') == 'pending_duration':
             alert = self.active_alerts[alert_key_or_id]
             alert['status'] = 'resolved_before_trigger' # Special status
             alert_id_actual = alert_key_or_id # Use the key as the ID in this case

        if not alert or alert.get('status') == 'resolved':
             self.logger.debug("Attempted to resolve non-active or non-existent alert", alert_key=alert_key_or_id)
             return False

        alert_id_actual = alert.get('id', alert_key_or_id) # Get the actual ID if available

        alert['status'] = 'resolved'
        alert['resolved_by'] = resolved_by or 'system'
        alert['resolved_at'] = datetime.now(timezone.utc).isoformat()
        
        # Cancel escalations for the specific alert ID
        if alert_id_actual in self.escalation_timers:
            self.logger.info("Cancelling escalation timers for resolved alert", alert_id=alert_id_actual)
            for timer in self.escalation_timers[alert_id_actual]:
                timer.cancel()
            del self.escalation_timers[alert_id_actual]
            
        self.logger.info("Alert resolved", alert_id=alert_id_actual, resolved_by=resolved_by)
        # Optionally, notify about resolution
        # self._send_notification(alert, notification_type='resolution')
        return True

    def get_active_alerts(self, device_id: str = None, severity: str = None) -> List[Dict]:
        """Get list of *currently active* alerts."""
        active_now = []
        for alert_id, alert in list(self.active_alerts.items()): # Iterate copy
             if alert.get('status') == 'active':
                  if device_id and alert.get('device_id') != device_id: continue
                  if severity and alert.get('severity') != severity: continue
                  active_now.append(alert.copy())

        # Sort by severity and timestamp
        severity_order = {'critical': 0, 'warning': 1, 'info': 2}
        active_now.sort(key=lambda x: (severity_order.get(x['severity'], 3), x['timestamp']))
        
        return active_now

    def get_alert_history(self, limit: int = 100, device_id: str = None, severity: str = None) -> List[Dict]:
        """Get recent alert history (active and resolved)."""
        history = list(reversed(self.alert_history)) # Newest first
        filtered = []
        count = 0
        for alert in history:
            if device_id and alert.get('device_id') != device_id: continue
            if severity and alert.get('severity') != severity: continue
            filtered.append(alert.copy())
            count += 1
            if count >= limit: break
        return filtered

    def get_alert_statistics(self) -> Dict:
        """Get comprehensive alert statistics."""
        active_alerts_list = [a for a in self.active_alerts.values() if a.get('status') == 'active']
        active_count = len(active_alerts_list)
        severity_counts = defaultdict(int)
        for alert in active_alerts_list:
            severity_counts[alert.get('severity', 'info')] += 1

        now = datetime.now(timezone.utc)
        day_ago = now - timedelta(days=1)
        recent_history = [a for a in self.alert_history if datetime.fromisoformat(a['timestamp']) > day_ago]
        
        rule_counts = defaultdict(int)
        for alert in recent_history:
             rule_counts[alert.get('rule_name', 'unknown')] += 1
        top_rules = sorted(rule_counts.items(), key=lambda item: item[1], reverse=True)[:5]

        suppressed_count = sum(1 for info in self.suppressed_alerts.values() if info.get('reason') != 'not_suppressed')

        return {
            'current_status': {
                'active_alerts': active_count,
                'severity_breakdown': dict(severity_counts),
                'escalated_alerts_pending': len(self.escalation_timers) # Alerts with active timers
            },
            'recent_activity': {
                'alerts_last_24h': len(recent_history),
                'alert_rate_per_hour': round(len(recent_history) / 24.0, 2) if recent_history else 0.0,
            },
            'top_alerting_rules_24h': [{'rule': name, 'count': count} for name, count in top_rules],
            'suppressed_alerts_count': suppressed_count # Count suppressed in last window
        }

    def stop(self):
        """Stop the alert manager and background threads."""
        self.logger.info("Shutting down Alert Manager...")
        self.should_stop.set()
        
        # Cancel all pending escalation timers
        self.logger.info("Cancelling pending escalation timers...")
        for alert_id, timers in list(self.escalation_timers.items()):
            for timer in timers:
                timer.cancel()
        self.escalation_timers.clear()

        # Wait for threads to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.logger.info("Waiting for alert processing thread to finish...")
            self.processing_thread.join(timeout=5)
            if self.processing_thread.is_alive():
                 self.logger.warning("Alert processing thread did not exit cleanly.")

        if self.notification_retry_thread and self.notification_retry_thread.is_alive():
             self.logger.info("Waiting for notification retry thread to finish...")
             # Optionally signal retry queue processing to stop quickly
             self.notification_retry_thread.join(timeout=5)
             if self.notification_retry_thread.is_alive():
                  self.logger.warning("Notification retry thread did not exit cleanly.")
                  
        self.logger.info("Alert Manager stopped.")

# --- Unit Tests ---
class TestAlertManager(unittest.TestCase):

    def setUp(self):
        """Set up test environment"""
        # Use a temporary config file or mock loading
        self.test_config = {
            "configuration": {
                "alert_rules": {
                    "temp_high": {"metric": "temp", "operator": ">", "threshold": 30, "severity": "warning", "duration": 5},
                    "pressure_low": {"metric": "pressure", "operator": "<", "threshold": 900, "severity": "critical"}
                },
                "escalation_policies": {
                    "default": {"levels": [{"delay_minutes": 0, "channels": ["mock"]}]},
                    "critical": {"levels": [
                        {"delay_minutes": 0, "channels": ["mock"]},
                        {"delay_minutes": 0.01, "channels": ["mock2"]} # Short delay for testing
                    ]}
                },
                "notification_channels": {
                    "mock": {"enabled": True},
                    "mock2": {"enabled": True}
                },
                "correlation_rules": {
                     "temp_pressure": {"metrics": ["temp", "pressure"], "window_minutes": 1}
                }
            }
        }
        # Mock config loading
        patcher = patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(self.test_config)))
        self.mock_open = patcher.start()
        self.addCleanup(patcher.stop)
        
        # Mock notification handlers
        self.mock_sender = MagicMock(return_value=True) # Assume success initially
        
        self.manager = AlertManager(config_path="dummy_config.json")
        self.manager.notification_handlers['mock'] = self.mock_sender
        self.manager.notification_handlers['mock2'] = self.mock_sender
        self.manager.logger.setLevel(logging.DEBUG) # More verbose logging for tests

    def tearDown(self):
        """Clean up after tests"""
        self.manager.stop()
        # Ensure timers are cancelled if stop didn't catch them
        for timers in self.manager.escalation_timers.values():
            for timer in timers:
                timer.cancel()
        # Give threads a moment to potentially exit if stop failed
        time.sleep(0.1)

    def test_01_alert_trigger_no_duration(self):
        """Test alert trigger without duration."""
        test_data = {'pressure': 850}
        alerts = self.manager.evaluate_conditions(test_data, device_id="dev1")
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]['rule_name'], 'pressure_low')
        
        # Check processing via queue
        time.sleep(0.1) # Allow queue processing
        self.assertEqual(len(self.manager.get_active_alerts()), 1)
        self.mock_sender.assert_called_once() # Check notification sent

    def test_02_alert_trigger_with_duration_fail(self):
        """Test alert trigger fails if duration not met."""
        test_data = {'temp': 35}
        alerts = self.manager.evaluate_conditions(test_data, device_id="dev2")
        self.assertEqual(len(alerts), 0) # Should be pending duration
        
        time.sleep(0.1)
        self.assertEqual(len(self.manager.get_active_alerts()), 0) # Not active yet
        alert_key = "temp_high_dev2"
        self.assertIn(alert_key, self.manager.active_alerts)
        self.assertEqual(self.manager.active_alerts[alert_key]['status'], 'pending_duration')
        self.mock_sender.assert_not_called()

    def test_03_alert_trigger_with_duration_success(self):
        """Test alert trigger succeeds when duration is met."""
        test_data = {'temp': 35}
        alert_key = "temp_high_dev3"
        
        # 1. Initial trigger (starts duration timer)
        self.manager.evaluate_conditions(test_data, device_id="dev3")
        time.sleep(0.1)
        self.assertEqual(len(self.manager.get_active_alerts()), 0)
        self.assertIn(alert_key, self.manager.active_alerts)
        self.assertEqual(self.manager.active_alerts[alert_key]['status'], 'pending_duration')
        
        # 2. Wait longer than duration and trigger again
        time.sleep(6) # Wait > 5 seconds
        alerts = self.manager.evaluate_conditions(test_data, device_id="dev3")
        self.assertEqual(len(alerts), 1) # Now it should trigger
        self.assertEqual(alerts[0]['rule_name'], 'temp_high')
        
        # 3. Check queue processing
        time.sleep(0.1)
        active = self.manager.get_active_alerts()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]['rule_name'], 'temp_high')
        self.mock_sender.assert_called_once() # Notification sent

    def test_04_alert_resolve(self):
        """Test alert resolves when condition clears."""
        # 1. Trigger alert
        self.manager.evaluate_conditions({'pressure': 850}, device_id="dev4")
        time.sleep(0.1)
        self.assertEqual(len(self.manager.get_active_alerts(device_id="dev4")), 1)
        alert_id = self.manager.get_active_alerts(device_id="dev4")[0]['id']
        
        # 2. Condition clears
        self.manager.evaluate_conditions({'pressure': 950}, device_id="dev4")
        time.sleep(0.1)
        
        # 3. Check if resolved
        self.assertEqual(len(self.manager.get_active_alerts(device_id="dev4")), 0)
        self.assertIn(alert_id, self.manager.active_alerts) # Still in main dict
        self.assertEqual(self.manager.active_alerts[alert_id]['status'], 'resolved')

    def test_05_correlation(self):
        """Test alert correlation."""
        # 1. Trigger first alert (pressure)
        self.manager.evaluate_conditions({'pressure': 850}, device_id="dev5")
        time.sleep(0.1) # Process pressure alert
        self.assertEqual(len(self.manager.get_active_alerts(device_id="dev5")), 1)
        self.mock_sender.reset_mock()
        
        # 2. Trigger second related alert (temp) within window
        alerts = self.manager.evaluate_conditions({'temp': 35, 'pressure': 850}, device_id="dev5")
        # Evaluate conditions queues the raw alert, _handle_alert does correlation
        
        # 3. Check processing - should result in a *correlated* alert
        time.sleep(0.1) # Allow queue processing
        active = self.manager.get_active_alerts(device_id="dev5")
        
        # Expecting ONE active alert, which is the correlated one
        self.assertEqual(len(active), 1) 
        correlated_alert = active[0]
        self.assertTrue(correlated_alert['rule_name'].startswith('correlated_temp_pressure'))
        self.assertIn('temp, pressure', correlated_alert['metric'])
        self.assertIn('correlated_alerts', correlated_alert)
        self.assertGreater(len(correlated_alert['correlated_alerts']), 0) # Includes original pressure alert ID
        
        # Check that only ONE notification was sent (for the correlated alert)
        self.mock_sender.assert_called_once() 

    def test_06_escalation(self):
        """Test alert escalation."""
        # 1. Trigger critical alert
        alerts = self.manager.evaluate_conditions({'pressure': 850}, device_id="dev6")
        self.assertEqual(len(alerts), 1)
        alert_id = alerts[0]['id']

        # 2. Process initial alert (level 0 notification)
        time.sleep(0.1) 
        self.assertEqual(len(self.manager.get_active_alerts(device_id="dev6")), 1)
        self.assertEqual(self.mock_sender.call_count, 1) # Initial notification
        self.assertIn(alert_id, self.manager.escalation_timers) # Timer should be set

        # 3. Wait for escalation delay (0.01 mins = 0.6 secs)
        self.logger.debug("Waiting for escalation...")
        time.sleep(0.8) 

        # 4. Check if escalation occurred (mock_sender called again)
        self.assertGreaterEqual(self.mock_sender.call_count, 2, "Escalation notification not sent")
        self.assertIn(alert_id, self.manager.active_alerts)
        self.assertEqual(self.manager.active_alerts[alert_id]['escalation_level'], 1)
        
        # 5. Acknowledge to stop further escalations
        self.manager.acknowledge_alert(alert_id, "test_ack")
        self.assertNotIn(alert_id, self.manager.escalation_timers) # Timers should be cancelled

    # --- Added Notification Retry Test ---
    def test_07_notification_retry(self):
         """Test notification retry mechanism."""
         # 1. Configure mock sender to fail initially
         fail_count = [0] # Use a list to modify mutable object in closure
         max_fails = 2
         def failing_sender(alert, is_retry=False):
             if fail_count[0] < max_fails:
                 self.logger.debug(f"Simulating notification failure {fail_count[0]+1}/{max_fails}")
                 fail_count[0] += 1
                 return False # Simulate failure
             else:
                 self.logger.debug("Simulating notification success on retry")
                 return True # Simulate success on retry

         self.manager.notification_handlers['mock'] = failing_sender
         self.manager.max_notification_retries = 3
         self.manager.notification_retry_delay = 0.1 # Short delay for test

         # 2. Trigger an alert
         alerts = self.manager.evaluate_conditions({'pressure': 850}, device_id="dev_retry")
         self.assertEqual(len(alerts), 1)

         # 3. Wait for initial processing and retries
         self.logger.debug("Waiting for notification retries...")
         time.sleep(1.0) # Allow time for initial attempt + 2 retries with backoff (0.1, 0.2)

         # 4. Assertions
         # Check if the sender was called multiple times (initial + retries)
         self.assertEqual(fail_count[0], max_fails, f"Sender should have failed {max_fails} times")
         # Check that the final call (which succeeded) was made
         self.assertEqual(self.manager.notification_handlers['mock'].call_count, max_fails + 1, "Sender not called enough times for success")
         # Check retry queue is empty
         self.assertTrue(self.manager.notification_retry_queue.empty())


# --- Main execution ---
if __name__ == '__main__':
    # Configure logging for test output if run directly
    setup_alert_manager_logging(level=logging.DEBUG)
    unittest.main()