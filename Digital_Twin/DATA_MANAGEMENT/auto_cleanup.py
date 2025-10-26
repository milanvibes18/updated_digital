#!/usr/bin/env python3
"""
Periodic Data Cleanup (Celery Task)

This module defines a Celery task responsible for periodically cleaning up
old data from the database to maintain performance and manage storage.

This task is designed to be run on a schedule (e.g., nightly) via Celery Beat.

To schedule this task, add it to the `celery_app.conf.beat_schedule`
in `enhanced_flask_app_v2.py`:

.. code-block:: python

    celery_app.conf.beat_schedule = {
        # ... other tasks ...
        'auto-cleanup-daily': {
            'task': 'tasks.auto_cleanup_data',
            'schedule': crontab(hour=3, minute=0),  # Run every day at 3 AM UTC
        },
    }

"""

import os
import sys
import logging
from datetime import datetime, timedelta, timezone

# --- Add project root to path ---
# This allows imports from other modules like WEB_APPLICATION and AI_MODULES
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    # Import the Celery app and session manager from the main Flask app
    from WEB_APPLICATION.enhanced_flask_app_v2 import (
        celery_app, get_session, DeviceData, Alert, User
    )
    # Import other models that may need cleanup
    from AI_MODULES.secure_database_manager import AuditLog, UserSession
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Error: Could not import necessary modules. {e}")
    print("Please ensure enhanced_flask_app_v2.py and secure_database_manager.py exist and are importable.")
    MODELS_AVAILABLE = False
    # Define dummy task if imports fail
    class DummyCeleryApp:
        def task(self, *args, **kwargs):
            def decorator(f):
                return f
            return decorator
    celery_app = DummyCeleryApp()


# --- Configuration ---
# Data retention policies (in days)
DATA_RETENTION_DAYS = int(os.getenv('DATA_RETENTION_DAYS', 90))
LOG_RETENTION_DAYS = int(os.getenv('LOG_RETENTION_DAYS', 30))

# Get a logger
logger = logging.getLogger('celery.auto_cleanup')


@celery_app.task(bind=True, name='tasks.auto_cleanup_data')
def auto_cleanup_data_task(self):
    """
    Celery task to perform periodic cleanup of old database records.
    """
    if not MODELS_AVAILABLE:
        logger.error("Cleanup task cannot run: Required models or Celery app not imported.")
        return {'status': 'FAILURE', 'error': 'Module import failed'}

    logger.info(f"Starting data cleanup task. Data retention: {DATA_RETENTION_DAYS} days. Log retention: {LOG_RETENTION_DAYS} days.")
    
    counts = {
        'device_data': 0,
        'alerts': 0,
        'audit_logs': 0,
        'user_sessions': 0
    }
    
    try:
        # Calculate cutoff dates
        data_cutoff = datetime.now(timezone.utc) - timedelta(days=DATA_RETENTION_DAYS)
        log_cutoff = datetime.now(timezone.utc) - timedelta(days=LOG_RETENTION_DAYS)
        session_cutoff = datetime.now(timezone.utc)

        with get_session() as session:
            
            # 1. Clean old DeviceData
            logger.info(f"Cleaning DeviceData older than {data_cutoff.isoformat()}...")
            deleted_device_data = session.query(DeviceData).filter(
                DeviceData.timestamp < data_cutoff
            ).delete(synchronize_session=False)
            counts['device_data'] = deleted_device_data
            logger.info(f"Deleted {deleted_device_data} old device_data records.")

            # 2. Clean old, acknowledged Alerts
            logger.info(f"Cleaning acknowledged Alerts older than {data_cutoff.isoformat()}...")
            deleted_alerts = session.query(Alert).filter(
                Alert.acknowledged == True,
                Alert.timestamp < data_cutoff
            ).delete(synchronize_session=False)
            counts['alerts'] = deleted_alerts
            logger.info(f"Deleted {deleted_alerts} old acknowledged alert records.")

            # 3. Clean old AuditLogs
            logger.info(f"Cleaning AuditLogs older than {log_cutoff.isoformat()}...")
            deleted_audit_logs = session.query(AuditLog).filter(
                AuditLog.timestamp < log_cutoff
            ).delete(synchronize_session=False)
            counts['audit_logs'] = deleted_audit_logs
            logger.info(f"Deleted {deleted_audit_logs} old audit_log records.")

            # 4. Clean expired UserSessions
            logger.info(f"Cleaning expired UserSessions...")
            deleted_sessions = session.query(UserSession).filter(
                UserSession.expires_at < session_cutoff
            ).delete(synchronize_session=False)
            counts['user_sessions'] = deleted_sessions
            logger.info(f"Deleted {deleted_sessions} expired user_session records.")

            # Commit changes (handled by get_session context manager)
            logger.info("Cleanup task commit.")
            
        logger.info("Database cleanup task completed successfully.")
        return {'status': 'SUCCESS', 'deleted_counts': counts}

    except Exception as e:
        logger.error(f"Error during data cleanup task: {e}", exc_info=True)
        # Rollback is handled by get_session context manager on exception
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        return {'status': 'FAILURE', 'error': str(e)}

if __name__ == "__main__":
    """
    Allows running the task directly for testing purposes.
    Make sure Celery, Redis, and DB are running.
    
    Example (from Digital_Twin directory):
    python DATA_MANAGEMENT/auto_cleanup.py
    """
    print("Running cleanup task directly for testing...")
    
    # This requires the Celery worker to be running to actually execute
    # Or, to test the function directly without Celery:
    
    print("Simulating direct function call (not as Celery task)...")
    try:
        # We need a mock 'self' for a bound task
        class MockTask:
            def update_state(self, *args, **kwargs):
                print(f"MockTask.update_state: {kwargs}")
        
        mock_self = MockTask()
        result = auto_cleanup_data_task(mock_self)
        print(f"Direct call result: {result}")
    except Exception as e:
        print(f"Direct call failed: {e}")
        print("Note: This likely failed because the Flask app context is not available.")
        print("To test, run via Celery worker or a dedicated test script with app context.")