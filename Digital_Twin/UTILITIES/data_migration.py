#!/usr/bin/env python3
"""
Data Migration Utility for Digital Twin System
Handles database migrations, data backups, and system upgrades.
"""

import sqlite3
import json
import logging
import shutil
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import zipfile
import tempfile
import subprocess
import os

class DataMigration:
    """
    Comprehensive data migration utility for Digital Twin system.
    Handles schema migrations, data transformations, and version upgrades.
    """
    
    def __init__(self, database_path: str = "DATABASE/health_data.db", 
                 backup_path: str = "SECURITY/data_backups/"):
        self.database_path = Path(database_path)
        self.backup_path = Path(backup_path)
        self.logger = self._setup_logging()
        
        # Migration history
        self.migration_log = Path("DATABASE/migration_log.json")
        self.version_file = Path("DATABASE/db_version.txt")
        
        # Create directories
        self.backup_path.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Migration definitions
        self.migrations = {
            "1.0.0": self._migration_v1_0_0,
            "1.1.0": self._migration_v1_1_0,
            "2.0.0": self._migration_v2_0_0,
        }
        
        # Current schema version
        self.current_version = "2.0.0"
    
    def _setup_logging(self):
        """Setup logging for data migration."""
        logger = logging.getLogger('DataMigration')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            Path("LOGS").mkdir(exist_ok=True)
            handler = logging.FileHandler('LOGS/data_migration.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def get_current_version(self) -> str:
        """Get current database version."""
        try:
            if self.version_file.exists():
                with open(self.version_file, 'r') as f:
                    return f.read().strip()
            return "0.0.0"  # Initial version
        except Exception as e:
            self.logger.error(f"Error reading version file: {e}")
            return "0.0.0"
    
    def set_version(self, version: str):
        """Set database version."""
        try:
            with open(self.version_file, 'w') as f:
                f.write(version)
            self.logger.info(f"Database version set to {version}")
        except Exception as e:
            self.logger.error(f"Error setting version: {e}")
    
    def create_backup(self, backup_name: str = None) -> str:
        """Create a complete system backup."""
        try:
            if backup_name is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_name = f"backup_{timestamp}"
            
            backup_file = self.backup_path / f"{backup_name}.zip"
            
            with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Backup databases
                for db_file in Path("DATABASE").glob("*.db"):
                    zipf.write(db_file, f"DATABASE/{db_file.name}")
                
                # Backup configuration
                for config_file in Path("CONFIG").glob("*.json"):
                    zipf.write(config_file, f"CONFIG/{config_file.name}")
                
                # Backup security keys
                if Path("CONFIG/encryption.key").exists():
                    zipf.write("CONFIG/encryption.key", "CONFIG/encryption.key")
                if Path("CONFIG/salt.key").exists():
                    zipf.write("CONFIG/salt.key", "CONFIG/salt.key")
                
                # Backup AI models
                if Path("ANALYTICS/models").exists():
                    for model_file in Path("ANALYTICS/models").glob("*.pkl"):
                        zipf.write(model_file, f"ANALYTICS/models/{model_file.name}")
                
                # Backup reports
                if Path("REPORTS").exists():
                    for report_file in Path("REPORTS").glob("*.html"):
                        zipf.write(report_file, f"REPORTS/{report_file.name}")
            
            # Calculate backup checksum
            checksum = self._calculate_file_checksum(backup_file)
            
            # Log backup
            self._log_migration({
                'type': 'backup',
                'backup_name': backup_name,
                'backup_file': str(backup_file),
                'checksum': checksum,
                'timestamp': datetime.now().isoformat(),
                'size_mb': backup_file.stat().st_size / (1024*1024)
            })
            
            self.logger.info(f"Backup created: {backup_file} ({checksum})")
            return str(backup_file)
            
        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            raise
    
    def restore_backup(self, backup_file: str) -> bool:
        """Restore system from backup."""
        try:
            backup_path = Path(backup_file)
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_file}")
            
            # Verify backup integrity
            if not self._verify_backup_integrity(backup_path):
                raise ValueError("Backup file integrity check failed")
            
            # Create current backup before restore
            current_backup = self.create_backup("pre_restore")
            
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                # Extract all files
                zipf.extractall(Path("."))
            
            self.logger.info(f"System restored from backup: {backup_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            return False
    
    def migrate_database(self, target_version: str = None) -> bool:
        """Migrate database to target version."""
        try:
            if target_version is None:
                target_version = self.current_version
            
            current_db_version = self.get_current_version()
            
            if current_db_version == target_version:
                self.logger.info(f"Database is already at version {target_version}")
                return True
            
            # Create backup before migration
            backup_file = self.create_backup(f"pre_migration_{target_version}")
            
            # Get migration path
            migration_path = self._get_migration_path(current_db_version, target_version)
            
            if not migration_path:
                raise ValueError(f"No migration path from {current_db_version} to {target_version}")
            
            # Execute migrations
            for version in migration_path:
                self.logger.info(f"Applying migration to version {version}")
                
                if version in self.migrations:
                    success = self.migrations[version]()
                    
                    if success:
                        self.set_version(version)
                        self._log_migration({
                            'type': 'migration',
                            'from_version': current_db_version,
                            'to_version': version,
                            'timestamp': datetime.now().isoformat(),
                            'success': True,
                            'backup_file': backup_file
                        })
                    else:
                        raise Exception(f"Migration to {version} failed")
                else:
                    raise ValueError(f"Migration for version {version} not found")
            
            self.logger.info(f"Database migrated successfully to version {target_version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            # Attempt to restore from backup
            try:
                self.restore_backup(backup_file)
                self.logger.info("Restored from backup after migration failure")
            except:
                self.logger.error("Failed to restore from backup")
            return False
    
    def _get_migration_path(self, from_version: str, to_version: str) -> List[str]:
        """Get ordered list of migrations to apply."""
        available_versions = sorted(self.migrations.keys())
        
        try:
            from_index = available_versions.index(from_version) + 1
        except ValueError:
            from_index = 0
        
        try:
            to_index = available_versions.index(to_version) + 1
        except ValueError:
            return []
        
        return available_versions[from_index:to_index]
    
    def _migration_v1_0_0(self) -> bool:
        """Migration to version 1.0.0 - Initial schema."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Create device_data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS device_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        device_id TEXT NOT NULL,
                        device_name TEXT,
                        device_type TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        value REAL,
                        unit TEXT,
                        status TEXT DEFAULT 'normal',
                        location TEXT,
                        health_score REAL DEFAULT 0.8,
                        efficiency_score REAL DEFAULT 0.8
                    )
                ''')
                
                # Create indices
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_device_timestamp ON device_data(device_id, timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON device_data(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON device_data(status)')
                
                conn.commit()
                
            self.logger.info("Migration v1.0.0 completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Migration v1.0.0 failed: {e}")
            return False
    
    def _migration_v1_1_0(self) -> bool:
        """Migration to version 1.1.0 - Add energy and system tables."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Add new columns to device_data
                cursor.execute('ALTER TABLE device_data ADD COLUMN operating_hours REAL DEFAULT 0')
                cursor.execute('ALTER TABLE device_data ADD COLUMN days_since_maintenance INTEGER DEFAULT 0')
                
                # Create energy_data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS energy_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        power_consumption_kw REAL,
                        energy_consumed_kwh REAL,
                        voltage_v REAL,
                        current_a REAL,
                        power_factor REAL,
                        energy_cost_usd REAL,
                        carbon_footprint_kg REAL
                    )
                ''')
                
                # Create system_metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        cpu_usage_percent REAL,
                        memory_usage_percent REAL,
                        disk_usage_percent REAL,
                        network_io_mbps REAL,
                        response_time_ms REAL,
                        active_connections INTEGER,
                        throughput_rps REAL
                    )
                ''')
                
                # Create indices
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_energy_timestamp ON energy_data(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_system_timestamp ON system_metrics(timestamp)')
                
                conn.commit()
                
            self.logger.info("Migration v1.1.0 completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Migration v1.1.0 failed: {e}")
            return False
    
    def _migration_v2_0_0(self) -> bool:
        """Migration to version 2.0.0 - Enhanced schema for AI features."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Create alerts table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_id TEXT UNIQUE,
                        device_id TEXT,
                        rule_name TEXT,
                        severity TEXT,
                        message TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        acknowledged BOOLEAN DEFAULT FALSE,
                        resolved BOOLEAN DEFAULT FALSE,
                        metadata TEXT
                    )
                ''')
                
                # Create patterns table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern_id TEXT UNIQUE,
                        pattern_type TEXT,
                        device_id TEXT,
                        pattern_data TEXT,
                        confidence REAL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        active BOOLEAN DEFAULT TRUE
                    )
                ''')
                
                # Create recommendations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS recommendations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        recommendation_id TEXT UNIQUE,
                        device_id TEXT,
                        type TEXT,
                        priority TEXT,
                        title TEXT,
                        description TEXT,
                        action TEXT,
                        impact REAL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        status TEXT DEFAULT 'active'
                    )
                ''')
                
                # Add AI-related columns to device_data
                try:
                    cursor.execute('ALTER TABLE device_data ADD COLUMN anomaly_score REAL DEFAULT 0')
                    cursor.execute('ALTER TABLE device_data ADD COLUMN failure_probability REAL DEFAULT 0')
                    cursor.execute('ALTER TABLE device_data ADD COLUMN maintenance_priority INTEGER DEFAULT 0')
                except sqlite3.OperationalError:
                    # Columns might already exist
                    pass
                
                # Create indices for new tables
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_device ON alerts(device_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_device ON patterns(device_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_recommendations_device ON recommendations(device_id)')
                
                conn.commit()
                
            self.logger.info("Migration v2.0.0 completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Migration v2.0.0 failed: {e}")
            return False
    
    def export_data(self, output_path: str = None, format: str = 'json') -> str:
        """Export all system data."""
        try:
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"REPORTS/data_export_{timestamp}.{format}"
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Export database data
            with sqlite3.connect(self.database_path) as conn:
                # Get all table names
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                export_data = {
                    'export_timestamp': datetime.now().isoformat(),
                    'database_version': self.get_current_version(),
                    'tables': {}
                }
                
                for table_name, in tables:
                    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                    
                    if format.lower() == 'json':
                        export_data['tables'][table_name] = df.to_dict('records')
                    elif format.lower() == 'csv':
                        csv_file = output_file.parent / f"{table_name}.csv"
                        df.to_csv(csv_file, index=False)
                
                # Save main export file
                if format.lower() == 'json':
                    with open(output_file, 'w') as f:
                        json.dump(export_data, f, indent=2, default=str)
                
            self.logger.info(f"Data exported to: {output_file}")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Data export failed: {e}")
            raise
    
    def import_data(self, import_path: str) -> bool:
        """Import data from export file."""
        try:
            import_file = Path(import_path)
            if not import_file.exists():
                raise FileNotFoundError(f"Import file not found: {import_path}")
            
            # Create backup before import
            backup_file = self.create_backup("pre_import")
            
            with open(import_file, 'r') as f:
                import_data = json.load(f)
            
            with sqlite3.connect(self.database_path) as conn:
                for table_name, records in import_data.get('tables', {}).items():
                    if not records:
                        continue
                    
                    # Convert records to DataFrame
                    df = pd.DataFrame(records)
                    
                    # Import data (append mode)
                    df.to_sql(table_name, conn, if_exists='append', index=False)
            
            self.logger.info(f"Data imported from: {import_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Data import failed: {e}")
            return False
    
    def optimize_database(self) -> bool:
        """Optimize database performance."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Analyze database
                cursor.execute('ANALYZE')
                
                # Vacuum database
                cursor.execute('VACUUM')
                
                # Update statistics
                cursor.execute('PRAGMA optimize')
                
                conn.commit()
            
            self.logger.info("Database optimization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Database optimization failed: {e}")
            return False
    
    def cleanup_old_data(self, retention_days: int = 90) -> int:
        """Clean up old data based on retention policy."""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            total_deleted = 0
            
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Tables with timestamp columns
                tables_with_timestamp = [
                    'device_data', 'energy_data', 'system_metrics', 
                    'alerts', 'patterns', 'recommendations'
                ]
                
                for table in tables_with_timestamp:
                    try:
                        cursor.execute(f'''
                            DELETE FROM {table} 
                            WHERE timestamp < ?
                        ''', (cutoff_date.isoformat(),))
                        
                        deleted_count = cursor.rowcount
                        total_deleted += deleted_count
                        
                        if deleted_count > 0:
                            self.logger.info(f"Deleted {deleted_count} old records from {table}")
                            
                    except sqlite3.OperationalError:
                        # Table might not exist
                        continue
                
                conn.commit()
            
            # Optimize after cleanup
            self.optimize_database()
            
            self.logger.info(f"Cleanup completed: {total_deleted} total records deleted")
            return total_deleted
            
        except Exception as e:
            self.logger.error(f"Data cleanup failed: {e}")
            return 0
    
    def verify_data_integrity(self) -> Dict:
        """Verify database integrity and consistency."""
        try:
            integrity_report = {
                'timestamp': datetime.now().isoformat(),
                'database_file': str(self.database_path),
                'file_exists': self.database_path.exists(),
                'file_size_mb': 0,
                'integrity_check': False,
                'foreign_key_check': False,
                'table_counts': {},
                'issues': []
            }
            
            if not self.database_path.exists():
                integrity_report['issues'].append("Database file does not exist")
                return integrity_report
            
            # File size
            integrity_report['file_size_mb'] = self.database_path.stat().st_size / (1024*1024)
            
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Integrity check
                cursor.execute('PRAGMA integrity_check')
                integrity_result = cursor.fetchone()
                integrity_report['integrity_check'] = integrity_result[0] == 'ok'
                
                if not integrity_report['integrity_check']:
                    integrity_report['issues'].append(f"Integrity check failed: {integrity_result[0]}")
                
                # Foreign key check
                cursor.execute('PRAGMA foreign_key_check')
                fk_violations = cursor.fetchall()
                integrity_report['foreign_key_check'] = len(fk_violations) == 0
                
                if fk_violations:
                    integrity_report['issues'].append(f"Foreign key violations: {len(fk_violations)}")
                
                # Table counts
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                for table_name, in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    integrity_report['table_counts'][table_name] = count
            
            self.logger.info("Data integrity verification completed")
            return integrity_report
            
        except Exception as e:
            self.logger.error(f"Data integrity verification failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'integrity_check': False
            }
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.error(f"Checksum calculation failed: {e}")
            return ""
    
    def _verify_backup_integrity(self, backup_file: Path) -> bool:
        """Verify backup file integrity."""
        try:
            # Check if file exists and is a valid zip
            if not backup_file.exists():
                return False
            
            with zipfile.ZipFile(backup_file, 'r') as zipf:
                # Test zip file integrity
                bad_file = zipf.testzip()
                if bad_file:
                    self.logger.error(f"Corrupted file in backup: {bad_file}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Backup integrity check failed: {e}")
            return False
    
    def _log_migration(self, migration_info: Dict):
        """Log migration activity."""
        try:
            if self.migration_log.exists():
                with open(self.migration_log, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = {'migrations': []}
            
            log_data['migrations'].append(migration_info)
            
            with open(self.migration_log, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to log migration: {e}")
    
    def get_migration_history(self) -> List[Dict]:
        """Get migration history."""
        try:
            if self.migration_log.exists():
                with open(self.migration_log, 'r') as f:
                    log_data = json.load(f)
                return log_data.get('migrations', [])
            return []
        except Exception as e:
            self.logger.error(f"Failed to read migration history: {e}")
            return []


# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Digital Twin Data Migration Utility')
    parser.add_argument('--migrate', type=str, help='Migrate to specific version')
    parser.add_argument('--backup', action='store_true', help='Create backup')
    parser.add_argument('--restore', type=str, help='Restore from backup file')
    parser.add_argument('--export', type=str, help='Export data to file')
    parser.add_argument('--import', type=str, help='Import data from file')
    parser.add_argument('--cleanup', type=int, help='Clean up data older than X days')
    parser.add_argument('--verify', action='store_true', help='Verify data integrity')
    parser.add_argument('--optimize', action='store_true', help='Optimize database')
    parser.add_argument('--version', action='store_true', help='Show current version')
    parser.add_argument('--history', action='store_true', help='Show migration history')
    
    args = parser.parse_args()
    
    migration = DataMigration()
    
    if args.version:
        print(f"Current database version: {migration.get_current_version()}")
        
    elif args.migrate:
        print(f"Migrating database to version {args.migrate}...")
        success = migration.migrate_database(args.migrate)
        print("Migration completed successfully" if success else "Migration failed")
        
    elif args.backup:
        print("Creating backup...")
        backup_file = migration.create_backup()
        print(f"Backup created: {backup_file}")
        
    elif args.restore:
        print(f"Restoring from backup: {args.restore}")
        success = migration.restore_backup(args.restore)
        print("Restore completed successfully" if success else "Restore failed")
        
    elif args.export:
        print(f"Exporting data to: {args.export}")
        export_file = migration.export_data(args.export)
        print(f"Data exported: {export_file}")
        
    elif getattr(args, 'import'):
        import_file = getattr(args, 'import')
        print(f"Importing data from: {import_file}")
        success = migration.import_data(import_file)
        print("Import completed successfully" if success else "Import failed")
        
    elif args.cleanup:
        print(f"Cleaning up data older than {args.cleanup} days...")
        deleted_count = migration.cleanup_old_data(args.cleanup)
        print(f"Deleted {deleted_count} old records")
        
    elif args.verify:
        print("Verifying data integrity...")
        report = migration.verify_data_integrity()
        print(json.dumps(report, indent=2))
        
    elif args.optimize:
        print("Optimizing database...")
        success = migration.optimize_database()
        print("Optimization completed successfully" if success else "Optimization failed")
        
    elif args.history:
        print("Migration History:")
        history = migration.get_migration_history()
        for entry in history:
            print(f"  {entry['timestamp']}: {entry['type']} - {entry}")
        
    else:
        parser.print_help()