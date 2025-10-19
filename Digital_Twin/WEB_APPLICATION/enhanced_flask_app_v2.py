#!/usr/bin/env python3
"""
Enhanced Flask Application for Digital Twin System v2.1 (Merged)
Main web application with JWT security, real-time capabilities, 
advanced analytics integration, scheduled retraining, reporting, and export.
"""

import os
import sys
import json
import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import threading
import time
import uuid
from functools import wraps
import hashlib
import hmac
import secrets
import math
import random

# Flask imports
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
import eventlet # Still needed for background tasks

# Security imports (Merged)
from flask_jwt_extended import (
    JWTManager, jwt_required, create_access_token,
    get_jwt_identity, unset_jwt_cookies
)
from werkzeug.security import generate_password_hash, check_password_hash

# Background Scheduler (from File 2)
from apscheduler.schedulers.background import BackgroundScheduler

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules (Kept comprehensive imports from File 1)
# Made imports more robust - log critical if essential ones fail
try:
    from CONFIG.app_config import config
    # Essential modules - log critical if missing
    try:
        from AI_MODULES.secure_database_manager import SecureDatabaseManager
    except ImportError as e:
        logging.critical(f"CRITICAL: Cannot import SecureDatabaseManager: {e}")
        SecureDatabaseManager = None # Allow fallback for limited functionality
    try:
        from AI_MODULES.predictive_analytics_engine import PredictiveAnalyticsEngine
    except ImportError as e:
        logging.critical(f"CRITICAL: Cannot import PredictiveAnalyticsEngine: {e}")
        PredictiveAnalyticsEngine = None
    try:
        from CONFIG.unified_data_generator import UnifiedDataGenerator
    except ImportError as e:
        logging.warning(f"Warning: Cannot import UnifiedDataGenerator: {e}")
        UnifiedDataGenerator = None # Use fallback

    # Optional/Supporting modules - log warning if missing
    try:
        from REPORTS.health_report_generator import HealthReportGenerator
    except ImportError as e:
        logging.warning(f"Warning: Cannot import HealthReportGenerator: {e}")
        HealthReportGenerator = None
    try:
        from AI_MODULES.health_score import HealthScoreCalculator
    except ImportError as e:
        logging.warning(f"Warning: Cannot import HealthScoreCalculator: {e}")
        HealthScoreCalculator = None
    try:
        from AI_MODULES.alert_manager import AlertManager
    except ImportError as e:
        logging.warning(f"Warning: Cannot import AlertManager: {e}")
        AlertManager = None
    try:
        from AI_MODULES.pattern_analyzer import PatternAnalyzer
    except ImportError as e:
        logging.warning(f"Warning: Cannot import PatternAnalyzer: {e}")
        PatternAnalyzer = None
    try:
        from AI_MODULES.recommendation_engine import RecommendationEngine
    except ImportError as e:
        logging.warning(f"Warning: Cannot import RecommendationEngine: {e}")
        RecommendationEngine = None

except ImportError as e:
    # Handle failure to import config
    logging.critical(f"CRITICAL: Could not import app_config or essential directories: {e}")
    sys.exit(1)

class DigitalTwinApp:
    """Main Digital Twin Flask Application Class"""
    
    def __init__(self):
        self.app = None
        self.socketio = None
        self.db_manager = None
        self.analytics_engine = None
        self.health_calculator = None
        self.alert_manager = None
        self.pattern_analyzer = None
        self.recommendation_engine = None
        self.data_generator = None
        self.scheduler = None # Added from File 2
        self.jwt = None # Added from File 2
        
        # Application state (Kept from File 1)
        self.connected_clients = {}
        self.data_cache = {}
        self.last_update = datetime.now()
        self.start_time = datetime.now()
        
        # Initialize logging
        self.setup_logging()
        
        # Initialize Flask app (Merged)
        self.create_app()
        
        # Initialize all modules (Merged)
        self.initialize_modules()
        
        # Setup routes (Merged)
        self.setup_routes()
        
        # Setup WebSocket events (Kept from File 1)
        self.setup_websocket_events()
        
        # Start background tasks (Merged)
        self.start_background_tasks()
    
    def setup_logging(self):
        """Setup comprehensive logging system"""
        os.makedirs('LOGS', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('LOGS/digital_twin_app.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DigitalTwinApp')
        self.logger.info("Digital Twin Application starting...")
    
    # --- Merged create_app ---
    def create_app(self):
        """Create and configure Flask application with JWT"""
        self.app = Flask(__name__, static_folder='static', template_folder='templates')
        
        # Configuration
        self.app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
        self.app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
        self.app.config['TESTING'] = False
        
        # --- Added from File 2: JWT Configuration ---
        self.app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', secrets.token_hex(32))
        self.app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
        self.jwt = JWTManager(self.app)
        
        # CORS configuration (Merged - added supports_credentials)
        CORS(self.app, origins="*", supports_credentials=True, 
             allow_headers=["Content-Type", "Authorization", "X-CSRF-Token"])
        
        # SocketIO initialization (Kept from File 1 - eventlet)
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            async_mode='eventlet',
            logger=False,
            engineio_logger=False,
            ping_timeout=60,
            ping_interval=25
        )
        
        self.logger.info("Flask application created successfully with JWT and SocketIO (Eventlet)")
    
    # --- Merged initialize_modules ---
    def initialize_modules(self):
        """Initialize AI, analytics, and data generation modules with fallbacks"""
        try:
            # Initialize core modules (check if imported)
            if SecureDatabaseManager:
                self.db_manager = SecureDatabaseManager()
            else:
                self.logger.critical("SecureDatabaseManager failed to import. Authentication and DB operations will fail.")
            
            if PredictiveAnalyticsEngine:
                self.analytics_engine = PredictiveAnalyticsEngine()
            else:
                 self.logger.critical("PredictiveAnalyticsEngine failed to import. Analytics and retraining will fail.")

            if HealthScoreCalculator:
                self.health_calculator = HealthScoreCalculator()
            else:
                self.logger.warning("HealthScoreCalculator not available. Using fallback logic.")
            
            if AlertManager:
                self.alert_manager = AlertManager()
            else:
                self.logger.warning("AlertManager not available. Using fallback logic.")
                self.alert_manager = self._create_fallback_alert_manager() # Kept fallback
            
            if PatternAnalyzer:
                self.pattern_analyzer = PatternAnalyzer()
            else:
                self.logger.warning("PatternAnalyzer not available. Some recommendations may be limited.")
            
            if RecommendationEngine:
                self.recommendation_engine = RecommendationEngine()
            else:
                self.logger.warning("RecommendationEngine not available. Recommendations will be basic.")

            if UnifiedDataGenerator:
                self.data_generator = UnifiedDataGenerator()
            else:
                self.logger.warning("UnifiedDataGenerator not available. Using basic fallback data generator.")
                self.data_generator = self._create_fallback_data_generator() # Kept fallback

            # --- Added from File 2: Initialize Scheduler ---
            self.scheduler = BackgroundScheduler(daemon=True)
            
            self.logger.info("Modules initialized (with potential fallbacks).")
            
        except Exception as e:
            self.logger.error(f"Critical error during module initialization: {e}", exc_info=True)
            # Attempt to initialize fallbacks even on general error
            self._initialize_fallback_modules() 
            if not self.db_manager or not self.analytics_engine or not self.data_generator:
                 self.logger.critical("Essential modules (DB, Analytics, Generator) failed. Exiting.")
                 sys.exit(1) # Exit if essential modules totally failed

    # Kept Fallback methods from File 1
    def _create_fallback_alert_manager(self):
        """Create a fallback alert manager"""
        # ... (Implementation from File 1) ...
        class FallbackAlertManager:
            def __init__(self):
                self.alert_conditions = {
                    'temperature_high': {'threshold': 80, 'operator': '>', 'severity': 'warning'},
                    'temperature_critical': {'threshold': 100, 'operator': '>', 'severity': 'critical'},
                    # ... other conditions
                }
            
            def evaluate_conditions(self, data, device_id):
                alerts = []
                # ... Simplified alert logic from File 1 ...
                device_type = data.get('device_type', '')
                value = data.get('value', 0)
                health_score = data.get('health_score', 1.0)
                
                if 'temperature' in device_type and value > 80:
                    severity = 'critical' if value > 100 else 'warning'
                    alerts.append({ 'id': str(uuid.uuid4()), 'device_id': device_id, 'type': 'temperature_alert', 'severity': severity, 'description': f'Temp {value}°C high', 'timestamp': datetime.now().isoformat(), 'value': value })
                elif 'pressure' in device_type and value > 1050:
                    alerts.append({ 'id': str(uuid.uuid4()), 'device_id': device_id, 'type': 'pressure_alert', 'severity': 'warning', 'description': f'Pressure {value} hPa high', 'timestamp': datetime.now().isoformat(), 'value': value })
                # ... Add more simple rules ...
                if health_score < 0.5:
                     alerts.append({ 'id': str(uuid.uuid4()), 'device_id': device_id, 'type': 'health_alert', 'severity': 'critical', 'description': f'Health {health_score:.1%} low', 'timestamp': datetime.now().isoformat(), 'value': health_score })
                return alerts
        self.logger.info("Using FallbackAlertManager.")
        return FallbackAlertManager()
    
    def _create_fallback_data_generator(self):
        """Create a fallback data generator"""
        # ... (Implementation from File 1) ...
        class FallbackDataGenerator:
             def __init__(self):
                 self.device_types = ['temperature_sensor', 'pressure_sensor', 'vibration_sensor']
                 self.locations = ['Factory A', 'Warehouse B']
             
             def generate_device_data(self, device_count=10, days_of_data=0.01, interval_minutes=1):
                 data = []
                 now = datetime.now()
                 start_time = now - timedelta(days=days_of_data)
                 current_time = start_time
                 while current_time <= now:
                     for i in range(device_count):
                         dev_id=f'F_DEVICE_{i+1:03d}'
                         dev_type=random.choice(self.device_types)
                         val = random.uniform(10,50)
                         status = 'normal' if random.random() > 0.1 else 'warning'
                         health = random.uniform(0.7, 1.0) if status == 'normal' else random.uniform(0.4, 0.7)
                         eff = health * random.uniform(0.8, 0.95)
                         data.append({'device_id': dev_id, 'device_type': dev_type, 'timestamp': current_time, 'value': round(val,2), 'status': status, 'health_score': round(health,3), 'efficiency_score': round(eff,3), 'location': random.choice(self.locations), 'unit': 'N/A', 'device_name': f'Fallback {dev_type} {i+1}'})
                     current_time += timedelta(minutes=interval_minutes)
                 return pd.DataFrame(data)

             def generate_complete_dataset(self, device_count=10, days_of_data=1):
                  # Simplified: just generate device data
                  return {'device_data': self.generate_device_data(device_count, days_of_data)}

        self.logger.info("Using FallbackDataGenerator.")
        return FallbackDataGenerator()
    
    def _initialize_fallback_modules(self):
        """Initialize fallback modules when imports fail"""
        if not hasattr(self, 'alert_manager') or self.alert_manager is None:
             self.alert_manager = self._create_fallback_alert_manager()
        if not hasattr(self, 'data_generator') or self.data_generator is None:
             self.data_generator = self._create_fallback_data_generator()
        # Add fallbacks for others if needed, though core functionality might be lost
        if not hasattr(self, 'health_calculator') or self.health_calculator is None:
             self.logger.warning("Health calculation fallback: Using basic score from generator.")
             # No separate fallback class, rely on generator's score
        if not hasattr(self, 'recommendation_engine') or self.recommendation_engine is None:
             self.logger.warning("Recommendation engine fallback: Providing static recommendations.")
             # No separate fallback class, rely on static data in get_recommendations
        # Analytics, Pattern Analyzer require more complex fallbacks or disabling features

    # --- Merged setup_routes ---
    def setup_routes(self):
        """Setup all Flask routes including authentication and protected endpoints"""
        
        # --- Authentication Endpoints (from File 2) ---
        @self.app.route('/api/register', methods=['POST'])
        def register():
            """User registration endpoint."""
            if not self.db_manager:
                return jsonify({"error": "Database manager not available"}), 503
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')
            email = data.get('email')
            
            if not username or not password:
                return jsonify({"error": "Username and password are required"}), 400
            
            try:
                if self.db_manager.create_user(username, password, email):
                    return jsonify({"message": "User created successfully"}), 201
                else:
                    return jsonify({"error": "Username or email already exists"}), 409
            except Exception as e:
                self.logger.error(f"Registration error: {e}")
                return jsonify({"error": "Registration failed"}), 500

        @self.app.route('/api/login', methods=['POST'])
        def login():
            """User login endpoint."""
            if not self.db_manager:
                return jsonify({"error": "Database manager not available"}), 503
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')
            
            try:
                if not self.db_manager.authenticate_user(username, password):
                    return jsonify({"error": "Invalid credentials"}), 401
                
                access_token = create_access_token(identity=username)
                return jsonify(access_token=access_token), 200
            except Exception as e:
                self.logger.error(f"Login error: {e}")
                return jsonify({"error": "Login failed"}), 500

        @self.app.route('/api/logout', methods=['POST'])
        def logout():
            """User logout endpoint."""
            response = jsonify({"message": "Logout successful"})
            unset_jwt_cookies(response)
            return response, 200

        @self.app.route('/api/whoami')
        @jwt_required()
        def protected():
            """Protected endpoint to check user identity."""
            return jsonify(logged_in_as=get_jwt_identity()), 200

        # --- Main Pages (Kept from File 1) ---
        # These might need @jwt_required() if you want to protect the HTML pages too,
        # or handle redirection to a login page in the templates/JS.
        # For simplicity, keeping them public for now.
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            try:
                return render_template('index.html')
            except:
                 return "Digital Twin API is running. Dashboard templates not found."
        
        @self.app.route('/dashboard')
        def enhanced_dashboard():
             try: return render_template('enhanced_dashboard.html')
             except: return "Enhanced Dashboard - Templates not found."
        
        @self.app.route('/analytics')
        def analytics():
             try: return render_template('analytics.html')
             except: return "Analytics page - Templates not found."
        
        @self.app.route('/devices')
        def devices():
             try: return render_template('devices_view.html')
             except: return "Devices page - Templates not found."
        
        # --- Health Check (Kept from File 1 - Public) ---
        @self.app.route('/health')
        def health_check():
            """Health check endpoint for monitoring"""
            # ... (Existing health check logic from File 1) ...
            try:
                db_status = self.check_database_health()
                ai_status = self.check_ai_modules_health()
                status = {
                    'status': 'healthy' if db_status.get('status') == 'healthy' and ai_status.get('status') == 'healthy' else 'partial',
                    'timestamp': datetime.now().isoformat(),
                    'database': db_status,
                    'ai_modules': ai_status,
                    'uptime': self.get_uptime(),
                    'version': '2.1.0', # Updated version
                    'connected_clients': len(self.connected_clients)
                }
                return jsonify(status), 200
            except Exception as e:
                 self.logger.error(f"Health check failed: {e}")
                 return jsonify({'status': 'unhealthy', 'error': str(e), 'timestamp': datetime.now().isoformat()}), 503
        
        # --- Core API Endpoints (Kept structure from File 1, added @jwt_required(), integrated REAL AI calls from File 2) ---
        
        @self.app.route('/api/dashboard_data')
        @jwt_required()
        def get_dashboard_data():
            """Get main dashboard data (from cache)"""
            try:
                data = self.get_cached_dashboard_data()
                return jsonify(data)
            except Exception as e:
                self.logger.error(f"Error getting dashboard data: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/devices')
        @jwt_required()
        def get_devices():
            """Get all devices data from cache"""
            try:
                cached_data = self.get_cached_dashboard_data()
                return jsonify(cached_data.get('devices', []))
            except Exception as e:
                self.logger.error(f"Error getting devices data: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/device/<device_id>')
        @jwt_required()
        def get_device(device_id):
            """Get specific device data from cache"""
            try:
                device = self.get_device_data(device_id) # Uses cached data
                if device:
                    return jsonify(device)
                else:
                    return jsonify({'error': 'Device not found'}), 404
            except Exception as e:
                self.logger.error(f"Error getting device {device_id}: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/analytics')
        @jwt_required()
        def get_analytics_data():
            """Get analytics data for charts (Using simple generation for now)"""
            # Kept File 1's placeholder analytics data generation.
            # A real implementation would fetch/calculate this from DB or cache.
            try:
                analytics_data = self.get_analytics_data() # Uses placeholder method
                return jsonify(analytics_data)
            except Exception as e:
                self.logger.error(f"Error getting analytics data: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/alerts')
        @jwt_required()
        def get_alerts():
            """Get system alerts using the AlertManager"""
            try:
                limit = request.args.get('limit', 10, type=int)
                severity = request.args.get('severity', None)
                
                # Use REAL AlertManager or fallback
                devices_df = pd.DataFrame(self.get_cached_dashboard_data().get('devices', []))
                all_alerts = []
                if not devices_df.empty:
                    for _, row in devices_df.iterrows():
                        all_alerts.extend(
                            self.alert_manager.evaluate_conditions(row.to_dict(), row['device_id'])
                        )
                
                # Sort by timestamp (most recent first) and apply limit/filter
                all_alerts.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                if severity:
                     filtered_alerts = [a for a in all_alerts if a.get('severity') == severity]
                else:
                     filtered_alerts = all_alerts

                return jsonify(filtered_alerts[:limit])

            except Exception as e:
                self.logger.error(f"Error getting alerts: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/system_metrics')
        @jwt_required()
        def get_system_metrics():
            """Get system performance metrics (Kept psutil/fallback from File 1)"""
            try:
                metrics = self.get_system_metrics()
                return jsonify(metrics)
            except Exception as e:
                self.logger.error(f"Error getting system metrics: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/historical_data')
        @jwt_required()
        def get_historical_data_endpoint(): # Renamed to avoid conflict
            """Get historical data for trends (Using simple generation for now)"""
            # Kept File 1's placeholder historical data.
            # A real implementation would query the DB.
            try:
                device_id = request.args.get('device_id')
                hours = request.args.get('hours', 24, type=int)
                metric = request.args.get('metric', 'value')
                
                data = self.get_historical_data(device_id, hours, metric) # Uses placeholder
                return jsonify(data)
            except Exception as e:
                self.logger.error(f"Error getting historical data: {e}")
                return jsonify({'error': str(e)}), 500
        
        # --- Merged /api/predictions ---
        @self.app.route('/api/predictions')
        @jwt_required()
        def get_predictions():
            """Get predictive analytics data using REAL Analytics Engine"""
            if not self.analytics_engine:
                 return jsonify({'error': 'Analytics engine not available'}), 503
            try:
                device_id = request.args.get('device_id')
                # Get the latest data point for this device from cache
                device_data = self.get_device_data(device_id)
                
                if not device_data:
                    return jsonify({'error': 'Device not found or no data'}), 404
                
                device_df = pd.DataFrame([device_data])
                
                # Call REAL analytics engine methods
                anomalies = self.analytics_engine.detect_anomalies(device_df)
                
                # Call failure prediction (handle potential errors)
                try:
                     failure_pred = self.analytics_engine.predict_failure(device_df)
                except ValueError as ve: # Handle case where model might not exist yet
                     self.logger.warning(f"Failure prediction failed for {device_id}: {ve}")
                     failure_pred = {'error': str(ve), 'predictions': []}
                except Exception as fe:
                     self.logger.error(f"Failure prediction failed unexpectedly for {device_id}: {fe}")
                     failure_pred = {'error': 'Prediction failed', 'predictions': []}

                # Add time series forecast if needed (example)
                # forecast = self.analytics_engine.time_series_analysis(device_df, 'value', 'timestamp')

                return jsonify({
                    'device_id': device_id,
                    'anomaly_prediction': anomalies,
                    'failure_prediction': failure_pred,
                    # 'forecast': forecast # Optionally add forecast results
                })
            except Exception as e:
                self.logger.error(f"Error getting predictions: {e}")
                return jsonify({'error': str(e)}), 500

        # --- Merged /api/health_scores (using REAL HealthScoreCalculator) ---
        @self.app.route('/api/health_scores')
        @jwt_required()
        def get_health_scores():
            """Get health scores using REAL HealthScoreCalculator"""
            if not self.health_calculator:
                 return jsonify({'error': 'Health calculator not available'}), 503
            try:
                devices_df = pd.DataFrame(self.get_cached_dashboard_data().get('devices', []))
                if devices_df.empty:
                    return jsonify({}), 200 # Return empty if no devices

                # Use the REAL health calculator
                health_scores = self.health_calculator.calculate_device_health_scores(devices_df)
                
                # Format for API response if needed (assuming calculate_device_health_scores returns a dict {device_id: score})
                formatted_scores = {
                    dev_id: {'overall_health': score * 100} # Adjust structure as needed
                    for dev_id, score in health_scores.items()
                }
                return jsonify(formatted_scores)

            except Exception as e:
                self.logger.error(f"Error getting health scores: {e}")
                return jsonify({'error': str(e)}), 500

        # --- Merged /api/recommendations (using REAL RecommendationEngine) ---
        @self.app.route('/api/recommendations')
        @jwt_required()
        def get_recommendations():
            """Get AI recommendations using REAL engines"""
            if not self.recommendation_engine or not self.health_calculator or not self.pattern_analyzer:
                 self.logger.warning("One or more AI modules for recommendations are unavailable.")
                 # Provide static fallback recommendations if RecommendationEngine itself is missing
                 if not self.recommendation_engine:
                      return jsonify(self.get_static_recommendations()) # Use static fallback
                 # Proceed if RecommendationEngine exists, even if others are missing (it might handle partial data)

            try:
                cached_data = self.get_cached_dashboard_data()
                devices_df = pd.DataFrame(cached_data.get('devices', []))
                
                if devices_df.empty:
                    return jsonify([]), 200 # No recommendations if no data

                # Prepare inputs for RecommendationEngine
                health_data = {}
                if self.health_calculator:
                     health_data = self.health_calculator.calculate_device_health_scores(devices_df)
                     # Convert health_data format if needed by RecommendationEngine
                     health_data = {'device_scores': health_data} # Example adaptation
                
                pattern_data = {}
                if self.pattern_analyzer:
                     # Analyze recent historical data if available, otherwise use latest snapshot
                     # Placeholder: Use snapshot for now
                     pattern_data = self.pattern_analyzer.analyze_temporal_patterns(
                          devices_df, 'timestamp', ['value'] # Adapt as needed
                     )
                
                recommendations = self.recommendation_engine.generate_recommendations(
                    health_data=health_data,
                    pattern_analysis=pattern_data,
                    historical_data=devices_df # Provide snapshot or fetch historical if needed
                )
                return jsonify(recommendations)
            except Exception as e:
                self.logger.error(f"Error getting recommendations: {e}")
                # Fallback to static recommendations on error
                return jsonify(self.get_static_recommendations())


        # --- Report Generation & Export Endpoints (Kept from File 1) ---
        # Added @jwt_required()
        @self.app.route('/api/generate_report')
        @jwt_required()
        def generate_report_endpoint():
             """Generate a health report and return its path"""
             # ... (Existing logic from File 1, including fallback) ...
             try:
                self.logger.info("Generating health report via API request.")
                if HealthReportGenerator:
                    report_generator = HealthReportGenerator()
                    # Fetch recent data for the report
                    recent_data_df = self.db_manager.get_health_data_as_dataframe(limit=5000) if self.db_manager else pd.DataFrame()
                    if recent_data_df.empty:
                        self.logger.warning("No recent data found for report generation.")
                        # Proceed with fallback or generate report with limited data
                    
                    html_path = report_generator.generate_comprehensive_report(data_df=recent_data_df, date_range_days=7)
                    report_filename = os.path.basename(html_path)
                    return jsonify({ 'success': True, 'report_path': f'/reports/{report_filename}', 'message': 'Health report generated successfully' })
                else:
                    self.logger.warning("HealthReportGenerator not available. Generating fallback report.")
                    return self._generate_fallback_report()
             except Exception as e:
                 self.logger.error(f"Error generating report: {e}")
                 return jsonify({'success': False, 'error': 'Failed to generate report', 'details': str(e)}), 500

        @self.app.route('/reports/<filename>')
        # No JWT needed if reports are public, add @jwt_required() if they should be protected
        def serve_report(filename):
             """Serve the generated report file"""
             # ... (Existing logic from File 1) ...
             try:
                reports_dir = os.path.join(os.path.dirname(__file__), '..', 'REPORTS', 'generated')
                return send_from_directory(reports_dir, filename)
             except Exception as e:
                self.logger.error(f"Error serving report: {e}")
                return jsonify({'error': 'Report not found'}), 404

        @self.app.route('/api/export_data')
        @jwt_required()
        def export_data_endpoint():
             """Export data and provide download link"""
             # ... (Existing logic from File 1) ...
             try:
                format_type = request.args.get('format', 'json')
                date_range = request.args.get('days', 7, type=int)
                self.logger.info(f"Exporting data. Format: {format_type}, Days: {date_range}")
                
                export_data_content = self.export_data(format_type, date_range) # Uses placeholder method
                
                exports_dir = os.path.join(os.path.dirname(__file__), '..', 'EXPORTS')
                os.makedirs(exports_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                if format_type.lower() == 'csv':
                    filename = f'export_{timestamp}.csv'
                    filepath = os.path.join(exports_dir, filename)
                    if 'devices' in export_data_content and export_data_content['devices']:
                        pd.DataFrame(export_data_content['devices']).to_csv(filepath, index=False)
                    else:
                         pd.DataFrame([{'message': 'No data available'}]).to_csv(filepath, index=False)
                else: # JSON
                    filename = f'export_{timestamp}.json'
                    filepath = os.path.join(exports_dir, filename)
                    with open(filepath, 'w') as f:
                        json.dump(export_data_content, f, indent=2, default=str)
                        
                return jsonify({'success': True, 'message': 'Data export completed', 'export_path': f'/exports/{filename}', 'filename': filename})

             except Exception as e:
                 self.logger.error(f"Error exporting data: {e}")
                 return jsonify({'success': False, 'error': 'Failed to export data', 'details': str(e)}), 500


        @self.app.route('/exports/<filename>')
        # No JWT needed if exports are public links, add @jwt_required() otherwise
        def serve_export(filename):
             """Serve exported data files"""
             # ... (Existing logic from File 1) ...
             try:
                exports_dir = os.path.join(os.path.dirname(__file__), '..', 'EXPORTS')
                return send_from_directory(exports_dir, filename)
             except Exception as e:
                self.logger.error(f"Error serving export: {e}")
                return jsonify({'error': 'Export file not found'}), 404
        
        self.logger.info("All routes setup completed")

    # Kept fallback report generation from File 1
    def _generate_fallback_report(self):
         # ... (Implementation from File 1) ...
         try:
            reports_dir = os.path.join(os.path.dirname(__file__), '..', 'REPORTS', 'generated')
            os.makedirs(reports_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'fallback_report_{timestamp}.html'
            filepath = os.path.join(reports_dir, filename)
            dashboard_data = self.get_cached_dashboard_data()
            # Generate simple HTML content based on dashboard_data
            html_content = "<html><head><title>Fallback Report</title></head><body>"
            html_content += f"<h1>Fallback Health Report - {datetime.now():%Y-%m-%d %H:%M}</h1>"
            html_content += f"<p>System Health: {dashboard_data.get('systemHealth', 'N/A')}%</p>"
            html_content += f"<p>Active Devices: {dashboard_data.get('activeDevices', 'N/A')}/{dashboard_data.get('totalDevices', 'N/A')}</p>"
            html_content += "<h2>Device Samples:</h2><ul>"
            for device in dashboard_data.get('devices', [])[:5]:
                html_content += f"<li>{device.get('device_name', 'Unknown')}: Status - {device.get('status', 'N/A')}, Health - {device.get('health_score', 0):.1%}</li>"
            html_content += "</ul></body></html>"
            with open(filepath, 'w') as f:
                f.write(html_content)
            return jsonify({'success': True, 'report_path': f'/reports/{filename}', 'message': 'Simplified fallback report generated'})
         except Exception as e:
             self.logger.error(f"Error generating fallback report: {e}")
             return jsonify({'success': False, 'error': 'Failed to generate fallback report'}), 500


    # Kept WebSocket setup from File 1
    def setup_websocket_events(self):
        """Setup WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            # ... (Existing logic from File 1) ...
             client_id = request.sid
             self.connected_clients[client_id] = {'connected_at': datetime.now(), 'last_ping': datetime.now()}
             self.logger.info(f"Client {client_id} connected. Total: {len(self.connected_clients)}")
             emit('initial_data', self.get_cached_dashboard_data())
             join_room('device_updates')
             join_room('alerts')
             join_room('system_metrics')
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
             # ... (Existing logic from File 1) ...
            client_id = request.sid
            if client_id in self.connected_clients:
                del self.connected_clients[client_id]
                self.logger.info(f"Client {client_id} disconnected. Total: {len(self.connected_clients)}")
        
        @self.socketio.on('ping')
        def handle_ping():
            # ... (Existing logic from File 1) ...
            client_id = request.sid
            if client_id in self.connected_clients:
                 self.connected_clients[client_id]['last_ping'] = datetime.now()
                 emit('pong', {'timestamp': datetime.now().isoformat()})
        
        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            # ... (Existing logic from File 1) ...
            try:
                client_id = request.sid
                sub_type = data.get('type')
                if sub_type in ['device_updates', 'alerts', 'system_metrics']:
                    join_room(sub_type)
                    emit('subscription_confirmed', {'type': sub_type})
                    self.logger.info(f"Client {client_id} subscribed to {sub_type}")
                else:
                    emit('error', {'message': 'Invalid subscription type'})
            except Exception as e:
                 self.logger.error(f"Subscription error: {e}")
                 emit('error', {'message': 'Subscription failed'})
        
        self.logger.info("WebSocket events setup completed")
    
    # --- Merged start_background_tasks ---
    def start_background_tasks(self):
        """Start background tasks for real-time updates, cleanup, and retraining"""
        
        # Data Update Task (Kept core logic from File 1, using eventlet)
        def data_update_task():
            """Background task to update data and send to clients"""
            self.logger.info("Starting real-time data simulation task.")
            while True:
                try:
                    # Generate data using the initialized generator (real or fallback)
                    new_devices_df = self.data_generator.generate_device_data(
                        device_count=25, # Use consistent count
                        days_of_data=0.003,  # ~5 minutes of data
                        interval_minutes=1
                    )
                    
                    latest_devices_df = new_devices_df.loc[new_devices_df.groupby('device_id')['timestamp'].idxmax()]
                    
                    self.update_data_cache(latest_devices_df)
                    
                    if self.connected_clients:
                        dashboard_data = self.get_cached_dashboard_data()
                        self.socketio.emit('data_update', dashboard_data, room='device_updates')
                    
                    self.check_and_send_alerts(latest_devices_df)
                    
                    eventlet.sleep(5)  # Update every 5 seconds
                    
                except Exception as e:
                    self.logger.error(f"Error in data update task: {e}", exc_info=True)
                    eventlet.sleep(60) # Wait longer on error
        
        # Cleanup Task (Kept from File 1, using eventlet)
        def cleanup_task():
            """Background task to cleanup disconnected clients"""
            self.logger.info("Starting client cleanup task.")
            while True:
                try:
                    current_time = datetime.now()
                    timeout_threshold = current_time - timedelta(minutes=5)
                    disconnected_clients = [
                         client_id for client_id, info in self.connected_clients.items()
                         if info['last_ping'] < timeout_threshold
                    ]
                    
                    for client_id in disconnected_clients:
                        del self.connected_clients[client_id]
                        self.logger.info(f"Cleaned up inactive client {client_id}")
                    
                    eventlet.sleep(300)  # Cleanup every 5 minutes
                    
                except Exception as e:
                    self.logger.error(f"Error in cleanup task: {e}")
                    eventlet.sleep(600) # Wait longer on error

        # --- Added from File 2: Retraining Scheduler ---
        try:
            if self.scheduler and self.analytics_engine:
                 # Check if job already exists to prevent duplicates on reload
                 if not self.scheduler.get_job('model_retraining_task'):
                      self.scheduler.add_job(
                           id='model_retraining_task',
                           func=self.analytics_engine.retrain_models, # Call method on the instance
                           trigger='interval',
                           hours=24  # Retrain once a day
                      )
                      self.scheduler.start()
                      self.logger.info("Started model retraining scheduler (runs every 24h).")
                 else:
                      self.logger.info("Model retraining scheduler already running.")
            elif not self.analytics_engine:
                 self.logger.warning("Analytics engine not available, cannot start retraining scheduler.")
            else: # Scheduler itself failed init
                 self.logger.error("Scheduler not initialized, cannot start retraining.")
        except Exception as e:
            self.logger.error(f"Failed to start scheduler: {e}")

        # Start Eventlet-based tasks
        self.socketio.start_background_task(data_update_task)
        self.socketio.start_background_task(cleanup_task)
        
        self.logger.info("Background tasks (Data Update, Cleanup, Retraining) initiated.")

    # --- Data Retrieval/Processing Methods (Kept comprehensive versions from File 1) ---

    def get_cached_dashboard_data(self):
        """Get cached dashboard data, generate if empty"""
        cache_age = datetime.now() - self.data_cache.get('dashboard_updated', datetime.min)
        if 'dashboard' not in self.data_cache or cache_age > timedelta(minutes=1): # Refresh if older than 1 min or missing
             self.logger.info("Dashboard cache miss or expired. Generating initial/fresh data...")
             # Use generate_complete_dataset for richer initial data if available
             try:
                 initial_datasets = self.data_generator.generate_complete_dataset(device_count=25, days_of_data=0.1) # Generate small initial set
                 initial_df = initial_datasets.get('device_data', pd.DataFrame())
                 if not initial_df.empty:
                      latest_df = initial_df.loc[initial_df.groupby('device_id')['timestamp'].idxmax()]
                      self.update_data_cache(latest_df)
                 else: # Fallback if generator failed
                      fallback_df = self._create_fallback_data_generator().generate_device_data(10, 0.01, 1)
                      latest_fallback = fallback_df.loc[fallback_df.groupby('device_id')['timestamp'].idxmax()]
                      self.update_data_cache(latest_fallback)

             except Exception as e:
                  self.logger.error(f"Failed to generate initial data: {e}. Using fallback.")
                  fallback_df = self._create_fallback_data_generator().generate_device_data(10, 0.01, 1)
                  latest_fallback = fallback_df.loc[fallback_df.groupby('device_id')['timestamp'].idxmax()]
                  self.update_data_cache(latest_fallback)


        return self.data_cache.get('dashboard', {}) # Return empty dict if cache update failed

    # Kept richer fetch_dashboard_data from File 1
    def fetch_dashboard_data(self, devices_df):
        """Fetch fresh dashboard data from a DataFrame"""
        try:
             # Ensure essential columns exist, add them with defaults if missing
             required_cols = ['device_id', 'status', 'health_score', 'efficiency_score', 'device_type', 'value']
             for col in required_cols:
                  if col not in devices_df.columns:
                       default_val = 0.0 if 'score' in col or col == 'value' else 'unknown'
                       devices_df[col] = default_val
                       self.logger.warning(f"Column '{col}' missing in devices_df, added default.")

             devices_data = devices_df.to_dict('records')
             total_devices = len(devices_data)
             active_devices = devices_df[devices_df['status'] != 'offline'].shape[0]

             health_scores = devices_df['health_score'].dropna()
             avg_health = health_scores.mean() * 100 if not health_scores.empty else 0

             efficiency_scores = devices_df['efficiency_score'].dropna()
             avg_efficiency = efficiency_scores.mean() * 100 if not efficiency_scores.empty else 0

             energy_usage_series = devices_df.loc[devices_df['device_type'] == 'power_meter', 'value'].dropna()
             energy_usage = energy_usage_series.sum() if not energy_usage_series.empty else 0
             # Fallback energy if no power meters
             if total_devices > 0 and energy_usage == 0:
                  energy_usage = random.uniform(800, 1500) * (active_devices / total_devices)

             performance_data = self.get_performance_chart_data(avg_health, avg_efficiency, energy_usage)
             status_distribution = self.calculate_status_distribution(devices_data)

             return {
                  'timestamp': datetime.now().isoformat(),
                  'systemHealth': round(avg_health),
                  'activeDevices': active_devices,
                  'totalDevices': total_devices,
                  'efficiency': round(avg_efficiency),
                  'energyUsage': round(energy_usage),
                  'energyCost': round(energy_usage * 0.12), # Assuming constant cost
                  'performanceData': performance_data,
                  'statusDistribution': status_distribution,
                  'devices': devices_data
             }
        except Exception as e:
            self.logger.error(f"Error fetching dashboard data: {e}", exc_info=True)
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}


    # Kept update_data_cache from File 1
    def update_data_cache(self, devices_df):
        """Update cached dashboard data"""
        try:
            self.data_cache['dashboard'] = self.fetch_dashboard_data(devices_df)
            self.data_cache['dashboard_updated'] = datetime.now()
            self.last_update = datetime.now()
        except Exception as e:
            self.logger.error(f"Error updating data cache: {e}")

    # Kept check_and_send_alerts from File 1 (uses real AlertManager now)
    def check_and_send_alerts(self, devices_df):
        """Check for new alerts using AlertManager and send to clients"""
        # ... (Implementation from File 1 - already uses self.alert_manager) ...
        if not self.alert_manager: return # Skip if no alert manager
        try:
             all_alerts = []
             for _, device_row in devices_df.iterrows():
                  device_data_dict = device_row.to_dict()
                  try:
                       triggered_alerts = self.alert_manager.evaluate_conditions(
                           data=device_data_dict,
                           device_id=device_data_dict.get('device_id')
                       )
                       all_alerts.extend(triggered_alerts)
                  except Exception as eval_e:
                       self.logger.error(f"Error evaluating alerts for {device_data_dict.get('device_id')}: {eval_e}")

             # Emit alerts (consider debouncing or rate limiting in a production system)
             for alert in all_alerts:
                  self.socketio.emit('alert_update', alert, room='alerts')
                  # Limit logging frequency if needed
                  # self.logger.info(f"Alert sent: {alert.get('description')} for {alert.get('device_id')}")

        except Exception as e:
             self.logger.error(f"Error in check_and_send_alerts: {e}")


    # Kept get_performance_chart_data from File 1
    def get_performance_chart_data(self, system_health, efficiency, energy_usage):
        # ... (Implementation from File 1) ...
         try:
            chart_data = []
            base_time = datetime.now() - timedelta(hours=23)
            for i in range(24):
                timestamp = base_time + timedelta(hours=i)
                hour_factor = math.sin(2 * math.pi * timestamp.hour / 24)
                health_var = system_health + (10 * hour_factor) + random.gauss(0, 3)
                eff_var = efficiency + (8 * hour_factor) + random.gauss(0, 4)
                energy_var = energy_usage + (energy_usage * 0.2 * hour_factor) + random.gauss(0, max(1, energy_usage * 0.05)) # Avoid gauss(0,0)
                chart_data.append({
                    'timestamp': timestamp.strftime('%H:%M'),
                    'systemHealth': max(0, min(100, health_var)),
                    'efficiency': max(0, min(100, eff_var)),
                    'energyUsage': max(0, energy_var)
                })
            return chart_data
         except Exception as e:
             self.logger.error(f"Error generating perf chart data: {e}")
             return []


    # Kept calculate_status_distribution from File 1
    def calculate_status_distribution(self, devices_data):
        # ... (Implementation from File 1) ...
         status_counts = {'normal': 0, 'warning': 0, 'critical': 0, 'offline': 0, 'maintenance': 0, 'anomaly': 0 }
         for device in devices_data:
             status = device.get('status', 'normal').lower()
             if status in status_counts:
                 status_counts[status] += 1
             else: # Handle unexpected statuses
                 status_counts['normal'] += 1 # Default to normal
         # Consolidate anomaly into critical for dashboard view if needed
         final_distribution = {
              'normal': status_counts['normal'],
              'warning': status_counts['warning'],
              'critical': status_counts['critical'] + status_counts['anomaly'],
              'offline': status_counts['offline'],
              'maintenance': status_counts['maintenance']
         }
         return final_distribution


    # Kept get_devices_data from File 1
    def get_devices_data(self):
         try:
             dashboard_data = self.get_cached_dashboard_data()
             return dashboard_data.get('devices', [])
         except Exception as e:
             self.logger.error(f"Error getting devices data: {e}")
             return []

    # Kept get_device_data from File 1
    def get_device_data(self, device_id):
         try:
             devices = self.get_devices_data()
             return next((d for d in devices if d.get('device_id') == device_id), None)
         except Exception as e:
             self.logger.error(f"Error getting device data for {device_id}: {e}")
             return None

    # Kept get_analytics_data (placeholder) from File 1
    def get_analytics_data(self):
         # ... (Placeholder Implementation from File 1) ...
         # Replace with real data fetching/calculation if needed for dedicated analytics page
         try:
             now = datetime.now()
             ts = [(now - timedelta(hours=i)).strftime('%H:%M') for i in range(23,-1,-1)]
             return {
                 'temperature': {'labels': ts, 'values': [20+5*math.sin(i*0.1)+random.gauss(0,1) for i in range(24)]},
                 'pressure': {'labels': ts, 'values': [1013+20*math.sin(i*0.05)+random.gauss(0,5) for i in range(24)]},
                 # Add others
             }
         except Exception as e:
             self.logger.error(f"Error getting analytics placeholder data: {e}")
             return {}


    # Kept get_alerts_data (placeholder) from File 1 - Used by export, API uses real one
    def get_alerts_data(self, limit=10, severity=None):
         # ... (Placeholder Implementation from File 1) ...
         # This is now mainly used for the EXPORT function as a fallback/example
         # The /api/alerts endpoint uses the real alert manager
         try:
             types = [('Temp high', 'warning'), ('Pressure low', 'critical'), ('Offline', 'critical')]
             alerts = []
             for i in range(limit * 2): # Generate more to allow filtering
                 t, s = types[i % len(types)]
                 if severity and s != severity: continue
                 alerts.append({'id': str(uuid.uuid4()), 'title': t, 'severity': s, 'device_id': f'DEVICE_{i%5+1:03d}', 'timestamp': (datetime.now()-timedelta(minutes=i*10)).isoformat()})
                 if len(alerts) >= limit: break
             return alerts
         except Exception as e:
             self.logger.error(f"Error getting placeholder alerts: {e}")
             return []


    # Kept get_system_metrics (psutil/fallback) from File 1
    def get_system_metrics(self):
        # ... (psutil/fallback Implementation from File 1) ...
        try:
             import psutil
             return {
                  'timestamp': datetime.now().isoformat(),
                  'cpu_percent': psutil.cpu_percent(interval=0.1),
                  'memory_percent': psutil.virtual_memory().percent,
                  'disk_percent': psutil.disk_usage('/').percent,
                  'active_connections': len(self.connected_clients)
             }
        except ImportError:
             return {
                  'timestamp': datetime.now().isoformat(),
                  'cpu_percent': random.uniform(10, 60),
                  'memory_percent': random.uniform(30, 70),
                  'disk_percent': random.uniform(40, 80),
                  'active_connections': len(self.connected_clients)
             }
        except Exception as e:
             self.logger.error(f"Error getting system metrics: {e}")
             return {}

    # Kept get_historical_data (placeholder) from File 1
    def get_historical_data(self, device_id, hours, metric):
        # ... (Placeholder Implementation from File 1) ...
        # Replace with DB query in a real application
         try:
             now = datetime.now()
             ts = [(now - timedelta(hours=hours-1-i)).isoformat() for i in range(hours)]
             vals = [50 + 20*math.sin(i*0.1) + random.gauss(0,5) for i in range(hours)]
             return {'device_id': device_id, 'metric': metric, 'timestamps': ts, 'values': [round(v,2) for v in vals]}
         except Exception as e:
             self.logger.error(f"Error getting placeholder historical data: {e}")
             return {}

    # Kept get_predictions_data (placeholder) from File 1 - Used by export, API uses real one
    def get_predictions_data(self, device_id, horizon):
         # ... (Placeholder Implementation from File 1) ...
         # This is now mainly used for the EXPORT function as a fallback/example
         # The /api/predictions endpoint uses the real analytics engine
        try:
            now = datetime.now()
            ts = [(now + timedelta(hours=i)).isoformat() for i in range(horizon)]
            preds = [50 + 15*math.sin(i*0.05) + random.gauss(0,2) for i in range(horizon)]
            conf = [max(0.5, 0.95 - (i*0.02)) for i in range(horizon)]
            return {'device_id': device_id, 'horizon_hours': horizon, 'timestamps': ts, 'predictions': [round(p,2) for p in preds], 'confidence': [round(c,3) for c in conf]}
        except Exception as e:
             self.logger.error(f"Error getting placeholder predictions: {e}")
             return {}

    # Kept calculate_health_scores (placeholder) from File 1 - Used by export, API uses real one
    def calculate_health_scores(self):
         # ... (Placeholder Implementation from File 1) ...
         # This is now mainly used for the EXPORT function as a fallback/example
         # The /api/health_scores endpoint uses the real health calculator
        try:
            devices = self.get_devices_data()
            scores = {}
            for dev in devices:
                 dev_id = dev.get('device_id')
                 health = dev.get('health_score', random.uniform(0.7, 1.0))
                 scores[dev_id] = {'overall_health': health*100, 'components': {}} # Simplified
            return scores
        except Exception as e:
            self.logger.error(f"Error calculating placeholder health scores: {e}")
            return {}

    # Kept get_recommendations (static) from File 1 - Used as fallback now
    def get_static_recommendations(self):
         # Renamed from get_recommendations to avoid conflict
         # ... (Static Recommendations from File 1) ...
         self.logger.info("Providing static fallback recommendations.")
         return [
              {'id': str(uuid.uuid4()), 'type': 'maintenance', 'priority': 'high', 'title': 'Check DEVICE_003 Vibration', 'description': 'Vibration levels appear elevated based on recent trends.', 'confidence': 0.75},
              {'id': str(uuid.uuid4()), 'type': 'optimization', 'priority': 'medium', 'title': 'Review Energy Usage', 'description': 'Overall energy consumption seems higher during off-peak hours.', 'confidence': 0.65}
         ]

    # Kept export_data from File 1 (uses placeholder methods)
    def export_data(self, format_type, date_range):
        """Export data for analysis (uses placeholder data getters)"""
        # ... (Implementation from File 1 - uses the placeholder getters) ...
        try:
             end_date = datetime.now()
             start_date = end_date - timedelta(days=date_range)
             # Generate sample data for export using fallback generator
             export_df = self._create_fallback_data_generator().generate_device_data(
                  device_count=15, days_of_data=date_range, interval_minutes=60
             )
             return {
                  'metadata': {'export_timestamp': datetime.now().isoformat(), 'date_range': {'start': start_date.isoformat(), 'end': end_date.isoformat()}, 'format': format_type},
                  'devices': export_df.to_dict('records'),
                  'alerts': self.get_alerts_data(limit=50), # Placeholder alerts
                  'system_metrics': self.get_system_metrics() # Placeholder/psutil metrics
             }
        except Exception as e:
             self.logger.error(f"Error exporting data: {e}")
             return {'error': str(e)}


    # --- Health Check Methods (Kept from File 1) ---
    def check_database_health(self):
        """Check database health"""
        # ... (Implementation from File 1) ...
        if not self.db_manager: return {'status': 'error', 'message': 'DB Manager not initialized'}
        try:
            db_path_primary = getattr(self.db_manager, 'db_path', 'N/A')
            db_path_users = getattr(self.db_manager, 'users_db_path', 'N/A')
            # Basic check: try connecting
            conn_primary = sqlite3.connect(db_path_primary)
            conn_primary.close()
            conn_users = sqlite3.connect(db_path_users)
            conn_users.close()
            return {'status': 'healthy', 'primary_path': db_path_primary, 'users_path': db_path_users}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def check_ai_modules_health(self):
        """Check AI modules health (more accurate now)"""
        modules = {
             'database_manager': self.db_manager is not None,
             'analytics_engine': self.analytics_engine is not None,
             'health_calculator': self.health_calculator is not None,
             'alert_manager': self.alert_manager is not None and not isinstance(self.alert_manager, self._create_fallback_alert_manager().__class__), # Check if not fallback
             'pattern_analyzer': self.pattern_analyzer is not None,
             'recommendation_engine': self.recommendation_engine is not None,
             'data_generator': self.data_generator is not None and not isinstance(self.data_generator, self._create_fallback_data_generator().__class__) # Check if not fallback
        }
        status = 'healthy'
        if not all(modules.values()):
            status = 'partial'
        if not modules['database_manager'] or not modules['analytics_engine']:
             status = 'critical_error' # Essential modules missing

        return {'modules': modules, 'status': status}

    # Kept get_uptime from File 1
    def get_uptime(self):
        if hasattr(self, 'start_time'):
            uptime = datetime.now() - self.start_time
            return str(uptime).split('.')[0]
        return "Unknown"
    
    # --- Merged run method ---
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application using SocketIO"""
        self.start_time = datetime.now()
        self.logger.info(f"Starting Digital Twin Application v2.1 on {host}:{port}")
        self.logger.info(f"Debug mode: {debug}")
        
        try:
            self.socketio.run(
                self.app,
                host=host,
                port=port,
                debug=debug,
                use_reloader=False, # Important for background tasks
                log_output=True # Log SocketIO messages
            )
        except KeyboardInterrupt:
            self.logger.info("Application stopped by user.")
            if self.scheduler and self.scheduler.running:
                 self.scheduler.shutdown()
                 self.logger.info("Scheduler shut down.")
        except Exception as e:
            self.logger.critical(f"Application failed to run: {e}", exc_info=True)
            if self.scheduler and self.scheduler.running:
                 self.scheduler.shutdown()
            raise

# Kept Error Handlers from File 1
def setup_error_handlers(app):
    @app.errorhandler(404)
    def not_found_error(error): return jsonify({'error': 'Not found'}), 404
    @app.errorhandler(500)
    def internal_error(error): return jsonify({'error': 'Internal server error'}), 500
    @app.errorhandler(Exception)
    def handle_exception(e):
        app.logger.error(f"Unhandled exception: {e}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred'}), 500

# Kept Application Factory from File 1
def create_app():
    """Application factory function"""
    try:
        app_instance = DigitalTwinApp()
        setup_error_handlers(app_instance.app)
        return app_instance
    except Exception as e:
        logging.critical(f"Failed to create application instance: {e}", exc_info=True)
        sys.exit(1) # Exit if app creation fails fundamentally

# Kept Main Execution Block from File 1
if __name__ == '__main__':
    os.environ.setdefault('FLASK_ENV', 'development')
    try:
        digital_twin_app = create_app()
        host = os.environ.get('HOST', '0.0.0.0')
        port = int(os.environ.get('PORT', 5000))
        debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
        digital_twin_app.run(host=host, port=port, debug=debug)
    except Exception as e:
        # Logging might not be fully set up here, so print as well
        print(f"CRITICAL: Failed to start application: {e}")
        logging.critical(f"Failed to start application: {e}", exc_info=True)
        sys.exit(1)