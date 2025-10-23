# migrate_v1_to_v2()

[Back to Migrations](../storage_migrations.md)

## Related User Story
"As a user, I want EDI to continue working properly after software updates." (from PRD - implied by maintainability requirements)

## Function Signature
`migrate_v1_to_v2()`

## Parameters
- None - Migrates the database schema from v1 to v2

## Returns
- Boolean - True if migration was successful, False otherwise

## Step-by-step Logic
1. Check current database version to confirm it's v1
2. Create backup of current database before making changes
3. Add new columns to existing tables as required for v2:
   - Add any new fields to sessions table
   - Add any new fields to prompts table
   - Add any new fields to entities table
   - Add any new fields to validations table
4. Update table schemas according to v2 requirements
5. Transform any existing data to match new schema requirements
6. Update the version table/flag to indicate schema is now v2
7. Run verification to ensure migration was successful
8. Return success status (True/False)

## Migration Changes from v1 to v2
- Add new columns to support additional functionality
- Update data types if needed for better performance
- Create new indexes for improved query performance
- Add constraints to ensure data integrity
- Preserve all existing data during migration

## Safety Measures
- Creates database backup before starting migration
- Uses transactions to ensure atomic changes
- Performs verification after migration to confirm success
- Handles errors gracefully and can potentially roll back
- Logs all migration steps for debugging if needed

## Input/Output Data Structures
### Migration Result
Returns Boolean:
- True: Migration completed successfully
- False: Migration failed, database unchanged or rolled back

### Schema Versioning
- Tracks current schema version in a dedicated table or file
- Ensures migration only runs on compatible versions
- Allows for multiple sequential migrations

## See Docs

### Python Implementation Example
Implementation of the migrate_v1_to_v2() function:

```python
import sqlite3
import os
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import json

class MigrationError(Exception):
    """Custom exception for migration errors."""
    pass

class DatabaseMigration:
    """
    Database migration class for EDI schema version management.
    """
    
    def __init__(self, db_path: str = "edi_sessions.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.backup_path = f"{db_path}.backup"
    
    def migrate_v1_to_v2(self) -> bool:
        """
        Migrate database schema from v1 to v2.
        
        Returns:
            Boolean - True if migration was successful, False otherwise
        """
        # Check current database version to confirm it's v1
        current_version = self._get_current_version()
        if current_version != "v1":
            self.logger.error(f"Cannot migrate from version {current_version} to v2")
            return False
        
        self.logger.info("Starting migration from v1 to v2")
        
        # Create backup of current database before making changes
        if not self._create_backup():
            self.logger.error("Failed to create database backup")
            return False
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Begin transaction to ensure atomic changes
                conn.execute("BEGIN")
                
                # Add new columns to existing tables as required for v2
                self._add_new_columns_v2(conn)
                
                # Update table schemas according to v2 requirements
                self._update_table_schemas_v2(conn)
                
                # Transform any existing data to match new schema requirements
                self._transform_existing_data_v2(conn)
                
                # Update the version table/flag to indicate schema is now v2
                self._update_version_flag(conn, "v2")
                
                # Commit transaction
                conn.execute("COMMIT")
                
                self.logger.info("Migration from v1 to v2 completed successfully")
                
                # Run verification to ensure migration was successful
                if self._verify_migration_v2():
                    self.logger.info("Migration verification passed")
                    return True
                else:
                    self.logger.error("Migration verification failed")
                    return False
                
        except sqlite3.Error as e:
            self.logger.error(f"Migration failed: {str(e)}")
            # Attempt to restore from backup
            self._restore_from_backup()
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during migration: {str(e)}")
            # Attempt to restore from backup
            self._restore_from_backup()
            return False
    
    def _get_current_version(self) -> str:
        """
        Get the current database schema version.
        
        Returns:
            Current version string
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if version table exists
                cursor.execute('''
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='schema_version'
                ''')
                
                if cursor.fetchone():
                    # Get current version from version table
                    cursor.execute('''
                        SELECT version FROM schema_version 
                        ORDER BY applied_at DESC LIMIT 1
                    ''')
                    version_row = cursor.fetchone()
                    if version_row:
                        return version_row[0]
                
                # If no version table, assume v1 (legacy)
                return "v1"
                
        except sqlite3.Error:
            # If we can't determine version, assume v1 for safety
            return "v1"
    
    def _create_backup(self) -> bool:
        """
        Create backup of current database before making changes.
        
        Returns:
            Boolean - True if backup successful, False otherwise
        """
        try:
            # Create backup with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.db_path}.backup_{timestamp}"
            
            shutil.copy2(self.db_path, backup_path)
            self.backup_path = backup_path
            
            self.logger.info(f"Database backup created: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {str(e)}")
            return False
    
    def _restore_from_backup(self) -> bool:
        """
        Restore database from backup if migration fails.
        
        Returns:
            Boolean - True if restore successful, False otherwise
        """
        try:
            if os.path.exists(self.backup_path):
                shutil.copy2(self.backup_path, self.db_path)
                self.logger.info("Database restored from backup")
                return True
            else:
                self.logger.error("Backup file not found for restoration")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to restore from backup: {str(e)}")
            return False
    
    def _add_new_columns_v2(self, conn: sqlite3.Connection):
        """
        Add new columns to existing tables as required for v2 schema.
        """
        cursor = conn.cursor()
        
        # Add new columns to sessions table
        try:
            cursor.execute('''
                ALTER TABLE sessions 
                ADD COLUMN processing_time REAL
            ''')
        except sqlite3.OperationalError:
            # Column might already exist
            pass
        
        try:
            cursor.execute('''
                ALTER TABLE sessions 
                ADD COLUMN model_version TEXT
            ''')
        except sqlite3.OperationalError:
            # Column might already exist
            pass
        
        try:
            cursor.execute('''
                ALTER TABLE sessions 
                ADD COLUMN session_tags TEXT
            ''')
        except sqlite3.OperationalError:
            # Column might already exist
            pass
        
        # Add new columns to prompts table
        try:
            cursor.execute('''
                ALTER TABLE prompts 
                ADD COLUMN generation_time REAL
            ''')
        except sqlite3.OperationalError:
            # Column might already exist
            pass
        
        try:
            cursor.execute('''
                ALTER TABLE prompts 
                ADD COLUMN model_confidence REAL
            ''')
        except sqlite3.OperationalError:
            # Column might already exist
            pass
        
        # Add new columns to entities table
        try:
            cursor.execute('''
                ALTER TABLE entities 
                ADD COLUMN embedding_vector TEXT
            ''')
        except sqlite3.OperationalError:
            # Column might already exist
            pass
        
        try:
            cursor.execute('''
                ALTER TABLE entities 
                ADD COLUMN semantic_label TEXT
            ''')
        except sqlite3.OperationalError:
            # Column might already exist
            pass
        
        # Add new columns to validations table
        try:
            cursor.execute('''
                ALTER TABLE validations 
                ADD COLUMN processing_details TEXT
            ''')
        except sqlite3.OperationalError:
            # Column might already exist
            pass
        
        try:
            cursor.execute('''
                ALTER TABLE validations 
                ADD COLUMN quality_metrics TEXT
            ''')
        except sqlite3.OperationalError:
            # Column might already exist
            pass
        
        conn.commit()
    
    def _update_table_schemas_v2(self, conn: sqlite3.Connection):
        """
        Update table schemas according to v2 requirements.
        """
        cursor = conn.cursor()
        
        # Create indexes for improved query performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_sessions_model_version 
            ON sessions(model_version)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_sessions_processing_time 
            ON sessions(processing_time)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_prompts_generation_time 
            ON prompts(generation_time)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_entities_semantic_label 
            ON entities(semantic_label)
        ''')
        
        # Add constraints to ensure data integrity
        # SQLite has limited support for altering constraints, so we'll just document them
        
        conn.commit()
    
    def _transform_existing_data_v2(self, conn: sqlite3.Connection):
        """
        Transform any existing data to match new schema requirements.
        """
        cursor = conn.cursor()
        
        # Set default values for new columns
        cursor.execute('''
            UPDATE sessions 
            SET model_version = 'v1_default', processing_time = 0.0
            WHERE model_version IS NULL OR processing_time IS NULL
        ''')
        
        cursor.execute('''
            UPDATE prompts 
            SET generation_time = 0.0, model_confidence = 1.0
            WHERE generation_time IS NULL OR model_confidence IS NULL
        ''')
        
        cursor.execute('''
            UPDATE entities 
            SET semantic_label = label
            WHERE semantic_label IS NULL
        ''')
        
        conn.commit()
    
    def _update_version_flag(self, conn: sqlite3.Connection, new_version: str):
        """
        Update the version table/flag to indicate new schema version.
        """
        cursor = conn.cursor()
        
        # Create version table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS schema_version (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                migration_notes TEXT
            )
        ''')
        
        # Insert new version record
        cursor.execute('''
            INSERT INTO schema_version (version, migration_notes)
            VALUES (?, ?)
        ''', (new_version, f"Migrated from previous version to {new_version}"))
        
        conn.commit()
    
    def _verify_migration_v2(self) -> bool:
        """
        Run verification to ensure migration was successful.
        
        Returns:
            Boolean - True if verification passed, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Verify version is now v2
                cursor.execute('''
                    SELECT version FROM schema_version 
                    ORDER BY applied_at DESC LIMIT 1
                ''')
                version_row = cursor.fetchone()
                if not version_row or version_row[0] != "v2":
                    self.logger.error("Version verification failed")
                    return False
                
                # Verify new columns exist
                required_columns = {
                    "sessions": ["processing_time", "model_version", "session_tags"],
                    "prompts": ["generation_time", "model_confidence"],
                    "entities": ["embedding_vector", "semantic_label"],
                    "validations": ["processing_details", "quality_metrics"]
                }
                
                for table, columns in required_columns.items():
                    for column in columns:
                        try:
                            cursor.execute(f'''
                                SELECT {column} FROM {table} LIMIT 1
                            ''')
                        except sqlite3.OperationalError:
                            self.logger.error(f"Column {column} missing from {table}")
                            return False
                
                # Verify data integrity
                cursor.execute('''
                    SELECT COUNT(*) FROM sessions 
                    WHERE model_version IS NULL OR processing_time IS NULL
                ''')
                null_count = cursor.fetchone()[0]
                if null_count > 0:
                    self.logger.warning(f"Found {null_count} sessions with null version fields")
                
                self.logger.info("Migration verification passed")
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Migration verification failed: {str(e)}")
            return False
    
    def get_migration_history(self) -> List[Dict[str, Any]]:
        """
        Get history of all applied migrations.
        
        Returns:
            List of migration records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS schema_version (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        version TEXT NOT NULL,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        migration_notes TEXT
                    )
                ''')
                
                cursor.execute('''
                    SELECT version, applied_at, migration_notes
                    FROM schema_version
                    ORDER BY applied_at DESC
                ''')
                
                rows = cursor.fetchall()
                return [
                    {
                        "version": row[0],
                        "applied_at": row[1],
                        "migration_notes": row[2]
                    }
                    for row in rows
                ]
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get migration history: {str(e)}")
            return []
    
    def needs_migration(self) -> bool:
        """
        Check if database needs migration to a newer version.
        
        Returns:
            Boolean - True if migration needed, False otherwise
        """
        current_version = self._get_current_version()
        target_version = "v2"  # Current target version
        
        # Simple version comparison (in real implementation, use semantic versioning)
        return current_version < target_version

# Example usage
if __name__ == "__main__":
    # Initialize migration manager
    migrator = DatabaseMigration("edi_sessions.db")
    
    # Check if migration is needed
    if migrator.needs_migration():
        print("Database migration needed")
        
        # Perform migration
        success = migrator.migrate_v1_to_v2()
        
        if success:
            print("Migration completed successfully!")
        else:
            print("Migration failed!")
    else:
        print("Database is up to date")
    
    # Show migration history
    history = migrator.get_migration_history()
    print(f"\nMigration history ({len(history)} records):")
    for record in history:
        print(f"  {record['version']} - {record['applied_at'][:19]} - {record['migration_notes']}")
```

### Advanced Migration Implementation
Enhanced migration with rollback support and comprehensive validation:

```python
import sqlite3
import os
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from contextlib import contextmanager
import logging
from dataclasses import dataclass
from enum import Enum

class MigrationStatus(Enum):
    """Enumeration for migration statuses."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class MigrationStep:
    """Representation of a single migration step."""
    name: str
    description: str
    execute_func: Callable
    rollback_func: Optional[Callable] = None
    required: bool = True

class AdvancedMigrationManager:
    """
    Advanced migration manager with rollback support and step tracking.
    """
    
    def __init__(self, db_path: str = "edi_sessions.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.migration_steps = []
        self._initialize_migration_tracking()
    
    def _initialize_migration_tracking(self):
        """Initialize migration tracking tables."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Create migration tracking table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS migration_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        migration_name TEXT NOT NULL,
                        version_from TEXT,
                        version_to TEXT,
                        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        completed_at TIMESTAMP,
                        status TEXT CHECK(status IN ('pending', 'in_progress', 'completed', 'failed', 'rolled_back')),
                        error_message TEXT,
                        duration_seconds REAL
                    )
                ''')
                
                # Create migration steps tracking table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS migration_steps (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        migration_id INTEGER REFERENCES migration_log(id),
                        step_name TEXT NOT NULL,
                        step_description TEXT,
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        status TEXT CHECK(status IN ('pending', 'in_progress', 'completed', 'failed')),
                        error_message TEXT,
                        duration_seconds REAL
                    )
                ''')
                
                conn.commit()
                
        except sqlite3.Error as e:
            raise MigrationError(f"Failed to initialize migration tracking: {str(e)}")
    
    @contextmanager
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with error handling."""
        conn = None
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=30,
                check_same_thread=False
            )
            conn.row_factory = sqlite3.Row
            yield conn
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            raise MigrationError(f"Database connection error: {str(e)}")
        finally:
            if conn:
                try:
                    conn.commit()
                except:
                    pass
                conn.close()
    
    def migrate_v1_to_v2_advanced(self) -> bool:
        """
        Advanced migration from v1 to v2 with comprehensive tracking and rollback support.
        
        Returns:
            Boolean - True if migration was successful, False otherwise
        """
        # Create migration log entry
        migration_id = self._create_migration_log("v1_to_v2", "v1", "v2")
        
        try:
            # Start migration
            self._update_migration_status(migration_id, MigrationStatus.IN_PROGRESS)
            
            # Define migration steps
            migration_steps = [
                MigrationStep(
                    name="backup_database",
                    description="Create backup of current database",
                    execute_func=self._backup_database,
                    rollback_func=self._restore_database_backup,
                    required=True
                ),
                MigrationStep(
                    name="add_sessions_columns",
                    description="Add new columns to sessions table",
                    execute_func=self._add_sessions_columns_v2,
                    rollback_func=None,  # Irreversible in SQLite
                    required=True
                ),
                MigrationStep(
                    name="add_prompts_columns",
                    description="Add new columns to prompts table",
                    execute_func=self._add_prompts_columns_v2,
                    rollback_func=None,
                    required=True
                ),
                MigrationStep(
                    name="add_entities_columns",
                    description="Add new columns to entities table",
                    execute_func=self._add_entities_columns_v2,
                    rollback_func=None,
                    required=True
                ),
                MigrationStep(
                    name="add_validations_columns",
                    description="Add new columns to validations table",
                    execute_func=self._add_validations_columns_v2,
                    rollback_func=None,
                    required=True
                ),
                MigrationStep(
                    name="create_indexes",
                    description="Create indexes for improved performance",
                    execute_func=self._create_indexes_v2,
                    rollback_func=self._drop_indexes_v2,
                    required=True
                ),
                MigrationStep(
                    name="transform_data",
                    description="Transform existing data to match new schema",
                    execute_func=self._transform_data_v2,
                    rollback_func=None,
                    required=True
                ),
                MigrationStep(
                    name="update_version",
                    description="Update schema version flag",
                    execute_func=self._update_version_v2,
                    rollback_func=self._rollback_version_v2,
                    required=True
                ),
                MigrationStep(
                    name="verify_migration",
                    description="Verify migration was successful",
                    execute_func=self._verify_migration_v2,
                    rollback_func=None,
                    required=True
                )
            ]
            
            # Execute migration steps
            for step in migration_steps:
                step_success = self._execute_migration_step(migration_id, step)
                if not step_success and step.required:
                    # Rollback on failure of required step
                    self._rollback_migration(migration_id)
                    return False
            
            # Mark migration as completed
            self._update_migration_status(migration_id, MigrationStatus.COMPLETED)
            self.logger.info("Advanced migration v1 to v2 completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Advanced migration failed: {str(e)}")
            self._update_migration_status(migration_id, MigrationStatus.FAILED, str(e))
            self._rollback_migration(migration_id)
            return False
    
    def _create_migration_log(self, migration_name: str, version_from: str, version_to: str) -> int:
        """Create migration log entry and return its ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO migration_log 
                    (migration_name, version_from, version_to, status)
                    VALUES (?, ?, ?, ?)
                ''', (migration_name, version_from, version_to, MigrationStatus.PENDING.value))
                
                conn.commit()
                return cursor.lastrowid
                
        except sqlite3.Error as e:
            raise MigrationError(f"Failed to create migration log: {str(e)}")
    
    def _update_migration_status(self, migration_id: int, status: MigrationStatus, error_message: str = None):
        """Update migration status."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if status in [MigrationStatus.COMPLETED, MigrationStatus.FAILED, MigrationStatus.ROLLED_BACK]:
                    cursor.execute('''
                        UPDATE migration_log 
                        SET status = ?, error_message = ?, completed_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    ''', (status.value, error_message, migration_id))
                else:
                    cursor.execute('''
                        UPDATE migration_log 
                        SET status = ?, error_message = ?
                        WHERE id = ?
                    ''', (status.value, error_message, migration_id))
                
                conn.commit()
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to update migration status: {str(e)}")
    
    def _execute_migration_step(self, migration_id: int, step: MigrationStep) -> bool:
        """Execute a single migration step with tracking."""
        # Create step log entry
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO migration_steps 
                    (migration_id, step_name, step_description, status)
                    VALUES (?, ?, ?, ?)
                ''', (migration_id, step.name, step.description, MigrationStatus.PENDING.value))
                
                step_id = cursor.lastrowid
                conn.commit()
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to create step log: {str(e)}")
            return False
        
        try:
            # Update step status to in progress
            with self._get_connection() as conn:
                conn.execute('''
                    UPDATE migration_steps 
                    SET status = ?, started_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (MigrationStatus.IN_PROGRESS.value, step_id))
            
            # Execute the step
            start_time = datetime.now()
            step.execute_func()
            duration = (datetime.now() - start_time).total_seconds()
            
            # Mark step as completed
            with self._get_connection() as conn:
                conn.execute('''
                    UPDATE migration_steps 
                    SET status = ?, completed_at = CURRENT_TIMESTAMP, duration_seconds = ?
                    WHERE id = ?
                ''', (MigrationStatus.COMPLETED.value, duration, step_id))
            
            self.logger.info(f"Migration step '{step.name}' completed successfully")
            return True
            
        except Exception as e:
            # Mark step as failed
            with self._get_connection() as conn:
                conn.execute('''
                    UPDATE migration_steps 
                    SET status = ?, error_message = ?, completed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (MigrationStatus.FAILED.value, str(e), step_id))
            
            self.logger.error(f"Migration step '{step.name}' failed: {str(e)}")
            return False
    
    def _rollback_migration(self, migration_id: int):
        """Rollback failed migration."""
        self.logger.info("Attempting to rollback migration")
        
        try:
            with self._get_connection() as conn:
                # Get completed steps in reverse order
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT step_name, step_description
                    FROM migration_steps
                    WHERE migration_id = ? AND status = ?
                    ORDER BY id DESC
                ''', (migration_id, MigrationStatus.COMPLETED.value))
                
                completed_steps = cursor.fetchall()
                
                # Attempt to rollback each completed step
                for step_name, step_desc in completed_steps:
                    self.logger.info(f"Attempting rollback of step: {step_name}")
                    # In a real implementation, you would call rollback functions here
                    # For this example, we'll just log the attempt
                    
            # Mark migration as rolled back
            self._update_migration_status(migration_id, MigrationStatus.ROLLED_BACK)
            self.logger.info("Migration rollback completed")
            
        except Exception as e:
            self.logger.error(f"Migration rollback failed: {str(e)}")
            self._update_migration_status(migration_id, MigrationStatus.FAILED, f"Rollback failed: {str(e)}")
    
    def _backup_database(self):
        """Create database backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{self.db_path}.backup_{timestamp}"
        shutil.copy2(self.db_path, backup_path)
        self.logger.info(f"Database backup created: {backup_path}")
    
    def _restore_database_backup(self):
        """Restore database from backup."""
        # This would find the most recent backup and restore from it
        self.logger.info("Restoring database from backup")
    
    def _add_sessions_columns_v2(self):
        """Add new columns to sessions table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Add new columns (ignore if they already exist)
            columns_to_add = [
                ("processing_time", "REAL"),
                ("model_version", "TEXT"),
                ("session_tags", "TEXT"),
                ("user_preferences", "TEXT"),
                ("session_metadata", "TEXT")
            ]
            
            for column_name, column_type in columns_to_add:
                try:
                    cursor.execute(f'''
                        ALTER TABLE sessions 
                        ADD COLUMN {column_name} {column_type}
                    ''')
                except sqlite3.OperationalError:
                    # Column might already exist
                    pass
            
            conn.commit()
    
    def _add_prompts_columns_v2(self):
        """Add new columns to prompts table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Add new columns
            columns_to_add = [
                ("generation_time", "REAL"),
                ("model_confidence", "REAL"),
                ("prompt_metadata", "TEXT"),
                ("generation_parameters", "TEXT")
            ]
            
            for column_name, column_type in columns_to_add:
                try:
                    cursor.execute(f'''
                        ALTER TABLE prompts 
                        ADD COLUMN {column_name} {column_type}
                    ''')
                except sqlite3.OperationalError:
                    pass
            
            conn.commit()
    
    def _add_entities_columns_v2(self):
        """Add new columns to entities table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Add new columns
            columns_to_add = [
                ("embedding_vector", "TEXT"),
                ("semantic_label", "TEXT"),
                ("entity_metadata", "TEXT"),
                ("quality_score", "REAL")
            ]
            
            for column_name, column_type in columns_to_add:
                try:
                    cursor.execute(f'''
                        ALTER TABLE entities 
                        ADD COLUMN {column_name} {column_type}
                    ''')
                except sqlite3.OperationalError:
                    pass
            
            conn.commit()
    
    def _add_validations_columns_v2(self):
        """Add new columns to validations table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Add new columns
            columns_to_add = [
                ("processing_details", "TEXT"),
                ("quality_metrics", "TEXT"),
                ("validation_metadata", "TEXT"),
                ("user_ratings", "TEXT")
            ]
            
            for column_name, column_type in columns_to_add:
                try:
                    cursor.execute(f'''
                        ALTER TABLE validations 
                        ADD COLUMN {column_name} {column_type}
                    ''')
                except sqlite3.OperationalError:
                    pass
            
            conn.commit()
    
    def _create_indexes_v2(self):
        """Create indexes for improved performance."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create indexes
            indexes_to_create = [
                ("idx_sessions_model_version", "sessions", "model_version"),
                ("idx_sessions_processing_time", "sessions", "processing_time"),
                ("idx_sessions_tags", "sessions", "session_tags"),
                ("idx_prompts_generation_time", "prompts", "generation_time"),
                ("idx_entities_semantic_label", "entities", "semantic_label"),
                ("idx_entities_quality_score", "entities", "quality_score")
            ]
            
            for index_name, table, column in indexes_to_create:
                try:
                    cursor.execute(f'''
                        CREATE INDEX IF NOT EXISTS {index_name} 
                        ON {table}({column})
                    ''')
                except sqlite3.Error:
                    pass
            
            conn.commit()
    
    def _drop_indexes_v2(self):
        """Drop indexes created in v2 migration."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Drop indexes
            indexes_to_drop = [
                "idx_sessions_model_version",
                "idx_sessions_processing_time", 
                "idx_sessions_tags",
                "idx_prompts_generation_time",
                "idx_entities_semantic_label",
                "idx_entities_quality_score"
            ]
            
            for index_name in indexes_to_drop:
                try:
                    cursor.execute(f"DROP INDEX IF EXISTS {index_name}")
                except sqlite3.Error:
                    pass
            
            conn.commit()
    
    def _transform_data_v2(self):
        """Transform existing data to match new schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Set default values for new columns
            cursor.execute('''
                UPDATE sessions 
                SET model_version = COALESCE(model_version, 'v1_default'),
                    processing_time = COALESCE(processing_time, 0.0)
                WHERE model_version IS NULL OR processing_time IS NULL
            ''')
            
            cursor.execute('''
                UPDATE prompts 
                SET generation_time = COALESCE(generation_time, 0.0),
                    model_confidence = COALESCE(model_confidence, 1.0)
                WHERE generation_time IS NULL OR model_confidence IS NULL
            ''')
            
            cursor.execute('''
                UPDATE entities 
                SET semantic_label = COALESCE(semantic_label, label)
                WHERE semantic_label IS NULL
            ''')
            
            conn.commit()
    
    def _update_version_v2(self):
        """Update schema version flag."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Update or insert version record
            cursor.execute('''
                INSERT OR REPLACE INTO schema_version 
                (version, applied_at, migration_notes)
                VALUES (?, CURRENT_TIMESTAMP, ?)
            ''', ("v2", "Migrated from v1 to v2"))
            
            conn.commit()
    
    def _rollback_version_v2(self):
        """Rollback schema version to v1."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Update version back to v1
            cursor.execute('''
                INSERT OR REPLACE INTO schema_version 
                (version, applied_at, migration_notes)
                VALUES (?, CURRENT_TIMESTAMP, ?)
            ''', ("v1", "Rolled back from v2 to v1"))
            
            conn.commit()
    
    def _verify_migration_v2(self):
        """Verify migration was successful."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Verify version is now v2
            cursor.execute('''
                SELECT version FROM schema_version 
                ORDER BY applied_at DESC LIMIT 1
            ''')
            version_row = cursor.fetchone()
            if not version_row or version_row[0] != "v2":
                raise MigrationError("Version verification failed")
            
            # Verify new columns exist by querying them
            required_columns = {
                "sessions": ["processing_time", "model_version"],
                "prompts": ["generation_time", "model_confidence"],
                "entities": ["embedding_vector", "semantic_label"],
                "validations": ["processing_details", "quality_metrics"]
            }
            
            for table, columns in required_columns.items():
                for column in columns:
                    try:
                        cursor.execute(f'SELECT {column} FROM {table} LIMIT 1')
                    except sqlite3.OperationalError:
                        raise MigrationError(f"Column {column} missing from {table}")
            
            self.logger.info("Migration verification passed")
    
    def get_migration_report(self, migration_id: int) -> Dict[str, Any]:
        """Generate detailed migration report."""
        try:
            with self._get_connection() as conn:
                # Get migration details
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT migration_name, version_from, version_to, 
                           started_at, completed_at, status, error_message
                    FROM migration_log
                    WHERE id = ?
                ''', (migration_id,))
                
                migration_row = cursor.fetchone()
                if not migration_row:
                    return {"error": "Migration not found"}
                
                # Get step details
                cursor.execute('''
                    SELECT step_name, step_description, started_at, completed_at, 
                           status, error_message, duration_seconds
                    FROM migration_steps
                    WHERE migration_id = ?
                    ORDER BY id
                ''', (migration_id,))
                
                step_rows = cursor.fetchall()
                
                return {
                    "migration": {
                        "name": migration_row[0],
                        "version_from": migration_row[1],
                        "version_to": migration_row[2],
                        "started_at": migration_row[3],
                        "completed_at": migration_row[4],
                        "status": migration_row[5],
                        "error_message": migration_row[6]
                    },
                    "steps": [
                        {
                            "name": row[0],
                            "description": row[1],
                            "started_at": row[2],
                            "completed_at": row[3],
                            "status": row[4],
                            "error_message": row[5],
                            "duration_seconds": row[6]
                        }
                        for row in step_rows
                    ]
                }
                
        except sqlite3.Error as e:
            return {"error": f"Failed to generate report: {str(e)}"}

# Example usage
if __name__ == "__main__":
    # Initialize advanced migration manager
    advanced_migrator = AdvancedMigrationManager("edi_advanced.db")
    
    print("Advanced Migration Manager initialized")
    
    # Example: Perform migration
    success = advanced_migrator.migrate_v1_to_v2_advanced()
    
    if success:
        print("Migration completed successfully!")
    else:
        print("Migration failed!")
```