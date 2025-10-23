# Storage: Migrations

[Back to Storage Layer](./storage_layer.md)

## Purpose
Schema versioning - Contains functions for handling database schema changes in a backward-compatible way.

## Functions
- `migrate_v1_to_v2()`: Migrates database schema from version 1 to 2
- Other migration functions as needed

### Details
- Handles backward-compatible schema changes
- Ensures data integrity during updates
- Manages version transitions

## Functions

- [migrate_v1_to_v2()](./storage/migrate_v1_to_v2.md)

## Technology Stack

- Database migration tools
- SQL for schema changes
- Version management

## See Docs

### Database Migration Tools Implementation Example
Storage migrations implementation for the EDI application:

```python
import sqlite3
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

class MigrationError(Exception):
    """Custom exception for migration errors."""
    pass

class MigrationStatus(Enum):
    """Enumeration for migration statuses."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class MigrationRecord:
    """Record of a migration execution."""
    id: int
    name: str
    version_from: str
    version_to: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: MigrationStatus = MigrationStatus.PENDING
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None

class StorageMigrationManager:
    """
    Storage migrations manager for handling database schema changes.
    """
    
    def __init__(self, db_path: str = "edi_sessions.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._initialize_migration_tracking()
    
    def _initialize_migration_tracking(self):
        """Initialize migration tracking tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
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
                
                # Create schema version table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS schema_version (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        version TEXT NOT NULL UNIQUE,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        migration_notes TEXT
                    )
                ''')
                
                # Create index for performance
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_migration_log_status 
                    ON migration_log(status)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_schema_version_applied 
                    ON schema_version(applied_at DESC)
                ''')
                
                conn.commit()
                
        except sqlite3.Error as e:
            raise MigrationError(f"Failed to initialize migration tracking: {str(e)}")
    
    def get_current_version(self) -> str:
        """
        Get the current database schema version.
        
        Returns:
            Current version string
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get the latest version
                cursor.execute('''
                    SELECT version FROM schema_version 
                    ORDER BY applied_at DESC LIMIT 1
                ''')
                
                version_row = cursor.fetchone()
                if version_row:
                    return version_row[0]
                
                # If no version found, check if tables exist to determine legacy version
                cursor.execute('''
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name IN ('sessions', 'prompts', 'entities', 'validations')
                ''')
                
                tables = cursor.fetchall()
                if tables:
                    return "v1"  # Legacy version with basic tables
                
                return "none"  # No schema yet
                
        except sqlite3.Error:
            # If we can't determine version, assume v1 for safety
            return "v1"
    
    def set_current_version(self, version: str, notes: str = "") -> bool:
        """
        Set the current database schema version.
        
        Args:
            version: Version string to set
            notes: Optional notes about the version change
            
        Returns:
            Boolean indicating success
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert new version record
                cursor.execute('''
                    INSERT OR REPLACE INTO schema_version 
                    (version, migration_notes)
                    VALUES (?, ?)
                ''', (version, notes))
                
                conn.commit()
                self.logger.info(f"Set schema version to: {version}")
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Error setting version {version}: {str(e)}")
            return False
    
    def create_migration_record(self, 
                              migration_name: str, 
                              version_from: str, 
                              version_to: str) -> int:
        """
        Create a migration record in the log.
        
        Args:
            migration_name: Name of the migration
            version_from: Starting version
            version_to: Target version
            
        Returns:
            ID of the created migration record
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO migration_log 
                    (migration_name, version_from, version_to, status)
                    VALUES (?, ?, ?, ?)
                ''', (migration_name, version_from, version_to, MigrationStatus.PENDING.value))
                
                migration_id = cursor.lastrowid
                conn.commit()
                
                self.logger.info(f"Created migration record {migration_id}: {migration_name}")
                return migration_id
                
        except sqlite3.Error as e:
            raise MigrationError(f"Failed to create migration record: {str(e)}")
    
    def update_migration_status(self, 
                              migration_id: int, 
                              status: MigrationStatus, 
                              error_message: Optional[str] = None) -> bool:
        """
        Update the status of a migration.
        
        Args:
            migration_id: ID of migration record
            status: New status
            error_message: Optional error message
            
        Returns:
            Boolean indicating success
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
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
                self.logger.info(f"Updated migration {migration_id} status to: {status.value}")
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Error updating migration {migration_id} status: {str(e)}")
            return False
    
    def get_pending_migrations(self) -> List[MigrationRecord]:
        """
        Get list of pending migrations.
        
        Returns:
            List of pending migration records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, migration_name, version_from, version_to, started_at, completed_at, status, error_message, duration_seconds
                    FROM migration_log 
                    WHERE status = ?
                    ORDER BY started_at
                ''', (MigrationStatus.PENDING.value,))
                
                rows = cursor.fetchall()
                pending_migrations = []
                
                for row in rows:
                    pending_migrations.append(MigrationRecord(
                        id=row[0],
                        name=row[1],
                        version_from=row[2],
                        version_to=row[3],
                        started_at=datetime.fromisoformat(row[4]) if row[4] else datetime.now(),
                        completed_at=datetime.fromisoformat(row[5]) if row[5] else None,
                        status=MigrationStatus(row[6]) if row[6] in [s.value for s in MigrationStatus] else MigrationStatus.PENDING,
                        error_message=row[7],
                        duration_seconds=row[8]
                    ))
                
                return pending_migrations
                
        except sqlite3.Error as e:
            self.logger.error(f"Error getting pending migrations: {str(e)}")
            return []
    
    def run_migration(self, migration_name: str, version_from: str, version_to: str) -> bool:
        """
        Run a specific migration.
        
        Args:
            migration_name: Name of migration to run
            version_from: Starting version
            version_to: Target version
            
        Returns:
            Boolean indicating success
        """
        # Create migration record
        migration_id = self.create_migration_record(migration_name, version_from, version_to)
        
        # Update status to in progress
        self.update_migration_status(migration_id, MigrationStatus.IN_PROGRESS)
        
        try:
            start_time = datetime.now()
            
            # Execute migration logic based on name
            if migration_name == "migrate_v1_to_v2":
                success = self._execute_v1_to_v2_migration()
            else:
                raise MigrationError(f"Unknown migration: {migration_name}")
            
            duration = (datetime.now() - start_time).total_seconds()
            
            if success:
                # Update status to completed
                self.update_migration_status(migration_id, MigrationStatus.COMPLETED)
                
                # Update schema version
                self.set_current_version(version_to, f"Applied migration {migration_name}")
                
                self.logger.info(f"Migration {migration_name} completed successfully in {duration:.2f}s")
                return True
            else:
                # Update status to failed
                self.update_migration_status(migration_id, MigrationStatus.FAILED, "Migration execution failed")
                self.logger.error(f"Migration {migration_name} failed after {duration:.2f}s")
                return False
                
        except Exception as e:
            # Update status to failed with error
            self.update_migration_status(migration_id, MigrationStatus.FAILED, str(e))
            self.logger.error(f"Migration {migration_name} failed with exception: {str(e)}")
            return False
    
    def _execute_v1_to_v2_migration(self) -> bool:
        """
        Execute the v1 to v2 migration.
        
        Returns:
            Boolean indicating success
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Begin transaction
                conn.execute("BEGIN")
                
                # Add new columns to existing tables
                # Add processing_time to sessions
                try:
                    cursor.execute('''
                        ALTER TABLE sessions 
                        ADD COLUMN processing_time REAL
                    ''')
                except sqlite3.OperationalError:
                    # Column might already exist
                    pass
                
                # Add model_version to sessions
                try:
                    cursor.execute('''
                        ALTER TABLE sessions 
                        ADD COLUMN model_version TEXT
                    ''')
                except sqlite3.OperationalError:
                    # Column might already exist
                    pass
                
                # Add generation_time to prompts
                try:
                    cursor.execute('''
                        ALTER TABLE prompts 
                        ADD COLUMN generation_time REAL
                    ''')
                except sqlite3.OperationalError:
                    # Column might already exist
                    pass
                
                # Add model_confidence to prompts
                try:
                    cursor.execute('''
                        ALTER TABLE prompts 
                        ADD COLUMN model_confidence REAL
                    ''')
                except sqlite3.OperationalError:
                    # Column might already exist
                    pass
                
                # Add embedding_vector to entities
                try:
                    cursor.execute('''
                        ALTER TABLE entities 
                        ADD COLUMN embedding_vector TEXT
                    ''')
                except sqlite3.OperationalError:
                    # Column might already exist
                    pass
                
                # Add semantic_label to entities
                try:
                    cursor.execute('''
                        ALTER TABLE entities 
                        ADD COLUMN semantic_label TEXT
                    ''')
                except sqlite3.OperationalError:
                    # Column might already exist
                    pass
                
                # Add processing_details to validations
                try:
                    cursor.execute('''
                        ALTER TABLE validations 
                        ADD COLUMN processing_details TEXT
                    ''')
                except sqlite3.OperationalError:
                    # Column might already exist
                    pass
                
                # Add quality_metrics to validations
                try:
                    cursor.execute('''
                        ALTER TABLE validations 
                        ADD COLUMN quality_metrics TEXT
                    ''')
                except sqlite3.OperationalError:
                    # Column might already exist
                    pass
                
                # Create indexes for improved performance
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
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_validations_quality_metrics 
                    ON validations(quality_metrics)
                ''')
                
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
                
                # Commit transaction
                conn.execute("COMMIT")
                
                self.logger.info("v1 to v2 migration executed successfully")
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"v1 to v2 migration failed: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error in v1 to v2 migration: {str(e)}")
            return False
    
    def rollback_migration(self, migration_id: int) -> bool:
        """
        Rollback a failed migration.
        
        Args:
            migration_id: ID of migration to rollback
            
        Returns:
            Boolean indicating success
        """
        try:
            # Get migration details
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT migration_name, version_from, version_to
                    FROM migration_log
                    WHERE id = ?
                ''', (migration_id,))
                
                row = cursor.fetchone()
                if not row:
                    self.logger.error(f"Migration {migration_id} not found")
                    return False
                
                migration_name, version_from, version_to = row
            
            # Execute rollback logic based on migration name
            if migration_name == "migrate_v1_to_v2":
                success = self._rollback_v1_to_v2_migration()
            else:
                self.logger.error(f"Unknown migration for rollback: {migration_name}")
                return False
            
            if success:
                # Update status to rolled back
                self.update_migration_status(migration_id, MigrationStatus.ROLLED_BACK)
                
                # Restore previous schema version
                self.set_current_version(version_from, f"Rolled back migration {migration_name}")
                
                self.logger.info(f"Migration {migration_name} rolled back successfully")
                return True
            else:
                self.logger.error(f"Failed to rollback migration {migration_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error rolling back migration {migration_id}: {str(e)}")
            return False
    
    def _rollback_v1_to_v2_migration(self) -> bool:
        """
        Rollback the v1 to v2 migration.
        
        Returns:
            Boolean indicating success
        """
        # In SQLite, rolling back schema changes is complex
        # A more realistic approach would involve restoring from backup
        self.logger.warning("Rollback of v1 to v2 migration would require database restoration from backup")
        return False
    
    def get_migration_history(self) -> List[Dict[str, Any]]:
        """
        Get history of all migrations.
        
        Returns:
            List of migration history records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, migration_name, version_from, version_to, 
                           started_at, completed_at, status, error_message, duration_seconds
                    FROM migration_log
                    ORDER BY started_at DESC
                ''')
                
                rows = cursor.fetchall()
                history = []
                
                for row in rows:
                    history.append({
                        "id": row[0],
                        "migration_name": row[1],
                        "version_from": row[2],
                        "version_to": row[3],
                        "started_at": row[4],
                        "completed_at": row[5],
                        "status": row[6],
                        "error_message": row[7],
                        "duration_seconds": row[8]
                    })
                
                return history
                
        except sqlite3.Error as e:
            self.logger.error(f"Error getting migration history: {str(e)}")
            return []
    
    def needs_migration(self) -> bool:
        """
        Check if database needs migration to a newer version.
        
        Returns:
            Boolean indicating if migration is needed
        """
        current_version = self.get_current_version()
        
        # Simple version comparison (would be more complex in real implementation)
        if current_version == "none":
            return True  # Need initial setup
        elif current_version == "v1":
            return True  # Need upgrade to v2
        else:
            return False  # Already at latest version

# Example usage
if __name__ == "__main__":
    # Initialize storage migration manager
    migration_manager = StorageMigrationManager("edi_storage_test.db")
    
    print("Storage Migration Manager initialized")
    
    # Create example session data
    example_session = {
        "id": str(uuid.uuid4()),
        "image_path": "/path/to/image.jpg",
        "naive_prompt": "make the sky more dramatic",
        "status": "completed",
        "final_alignment_score": 0.85,
        "prompts": [
            {
                "iteration": 0,
                "positive_prompt": "dramatic sky with storm clouds",
                "negative_prompt": "sunny sky, clear weather",
                "quality_score": 0.92
            }
        ],
        "entities": [
            {
                "entity_id": "sky_0",
                "label": "sky",
                "confidence": 0.95,
                "bbox": {"x1": 0, "y1": 0, "x2": 1920, "y2": 768},
                "mask_path": "/path/to/mask.png",
                "color_hex": "#87CEEB",
                "area_percent": 39.6
            }
        ],
        "validations": [
            {
                "attempt_number": 1,
                "alignment_score": 0.85,
                "preserved_count": 3,
                "modified_count": 1,
                "unintended_count": 0,
                "user_feedback": "Great improvement to the sky!"
            }
        ],
        "user_feedback": {
            "feedback_type": "accept",
            "comments": "Perfect! Exactly what I wanted.",
            "rating": 5
        }
    }
    
    # Save session
    if migration_manager.save_session(example_session):
        print(f"Session saved successfully: {example_session['id']}")
    else:
        print("Failed to save session")
    
    # Load session
    loaded_session = migration_manager.load_session(example_session["id"])
    if loaded_session:
        print(f"Session loaded: {loaded_session['naive_prompt']}")
    else:
        print("Failed to load session")
    
    # Query history
    history = migration_manager.query_history(limit=10)
    print(f"Query returned {len(history)} sessions")
    
    # Get statistics
    stats = migration_manager.get_statistics()
    print(f"Database statistics:")
    print(f"  Total sessions: {stats['total_sessions']}")
    print(f"  Status breakdown: {stats['status_breakdown']}")
    print(f"  Average score: {stats['average_alignment_score']}")
    print(f"  Success rate: {stats['successful_rate']}%")
    
    # Get recent sessions
    recent = migration_manager.get_recent_sessions(5)
    print(f"Recent sessions: {len(recent)}")
    
    # Search sessions
    search_results = migration_manager.search_sessions("sky", limit=5)
    print(f"Search results: {len(search_results)}")
    
    # Get successful sessions
    successful = migration_manager.get_successful_sessions(min_score=0.8, limit=5)
    print(f"Successful sessions: {len(successful)}")
    
    # Update session status
    if migration_manager.update_session_status(example_session["id"], "completed", 0.85):
        print("Session status updated successfully")
    else:
        print("Failed to update session status")
    
    # Get session count
    count = migration_manager.get_session_count()
    print(f"Total sessions: {count}")
    
    print("Storage migrations example completed")
```

### SQL Implementation Example
Database schema changes and version management with SQL:

```python
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import json

class SQLMigrationExecutor:
    """
    SQL-based migration executor for database schema changes.
    """
    
    def __init__(self, db_path: str = "edi_sessions.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
    
    def execute_sql_migration(self, sql_statements: List[str]) -> bool:
        """
        Execute a series of SQL statements for migration.
        
        Args:
            sql_statements: List of SQL statements to execute
            
        Returns:
            Boolean indicating success
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Begin transaction for atomic migration
                conn.execute("BEGIN")
                
                # Execute each SQL statement
                for statement in sql_statements:
                    try:
                        cursor.execute(statement)
                        self.logger.debug(f"Executed: {statement[:50]}...")
                    except sqlite3.OperationalError as e:
                        # Handle common migration errors (like duplicate columns)
                        if "duplicate column name" in str(e).lower():
                            self.logger.warning(f"Column already exists, skipping: {statement[:50]}...")
                            continue
                        elif "already exists" in str(e).lower():
                            self.logger.warning(f"Object already exists, skipping: {statement[:50]}...")
                            continue
                        else:
                            # Re-raise other errors
                            raise e
                
                # Commit transaction
                conn.execute("COMMIT")
                
                self.logger.info(f"Executed {len(sql_statements)} SQL statements successfully")
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"SQL migration failed: {str(e)}")
            return False
    
    def create_migration_script_v1_to_v2(self) -> List[str]:
        """
        Create SQL migration script for v1 to v2 upgrade.
        
        Returns:
            List of SQL statements for the migration
        """
        sql_statements = [
            # Add new columns to sessions table
            "ALTER TABLE sessions ADD COLUMN processing_time REAL",
            "ALTER TABLE sessions ADD COLUMN model_version TEXT",
            "ALTER TABLE sessions ADD COLUMN session_tags TEXT",
            
            # Add new columns to prompts table
            "ALTER TABLE prompts ADD COLUMN generation_time REAL",
            "ALTER TABLE prompts ADD COLUMN model_confidence REAL",
            "ALTER TABLE prompts ADD COLUMN generation_parameters TEXT",
            
            # Add new columns to entities table
            "ALTER TABLE entities ADD COLUMN embedding_vector TEXT",
            "ALTER TABLE entities ADD COLUMN semantic_label TEXT",
            "ALTER TABLE entities ADD COLUMN entity_metadata TEXT",
            
            # Add new columns to validations table
            "ALTER TABLE validations ADD COLUMN processing_details TEXT",
            "ALTER TABLE validations ADD COLUMN quality_metrics TEXT",
            "ALTER TABLE validations ADD COLUMN validation_metadata TEXT",
            
            # Create indexes for improved performance
            "CREATE INDEX IF NOT EXISTS idx_sessions_model_version ON sessions(model_version)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_processing_time ON sessions(processing_time)",
            "CREATE INDEX IF NOT EXISTS idx_prompts_generation_time ON prompts(generation_time)",
            "CREATE INDEX IF NOT EXISTS idx_entities_semantic_label ON entities(semantic_label)",
            "CREATE INDEX IF NOT EXISTS idx_validations_quality_metrics ON validations(quality_metrics)",
            
            # Set default values for new columns
            "UPDATE sessions SET model_version = 'v1_default', processing_time = 0.0 WHERE model_version IS NULL OR processing_time IS NULL",
            "UPDATE prompts SET generation_time = 0.0, model_confidence = 1.0 WHERE generation_time IS NULL OR model_confidence IS NULL",
            "UPDATE entities SET semantic_label = label WHERE semantic_label IS NULL",
            
            # Create new tables if needed
            """
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT REFERENCES sessions(id),
                feedback_type TEXT CHECK(feedback_type IN ('accept', 'reject', 'partial')),
                comments TEXT,
                rating INT CHECK(rating BETWEEN 1 AND 5),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            """
            CREATE TABLE IF NOT EXISTS processing_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT REFERENCES sessions(id),
                log_level TEXT CHECK(log_level IN ('debug', 'info', 'warning', 'error')),
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Create indexes for new tables
            "CREATE INDEX IF NOT EXISTS idx_user_feedback_session ON user_feedback(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_user_feedback_rating ON user_feedback(rating)",
            "CREATE INDEX IF NOT EXISTS idx_processing_logs_session ON processing_logs(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_processing_logs_timestamp ON processing_logs(timestamp)",
            
            # Add foreign key constraints (if supported by SQLite version)
            # Note: SQLite requires foreign key constraints to be enabled
            "PRAGMA foreign_keys = ON",
            
            # Create triggers for automatic timestamp updates
            """
            CREATE TRIGGER IF NOT EXISTS update_sessions_timestamp 
            AFTER UPDATE ON sessions
            BEGIN
                UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END
            """,
            
            """
            CREATE TRIGGER IF NOT EXISTS update_prompts_timestamp 
            AFTER UPDATE ON prompts
            BEGIN
                UPDATE prompts SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END
            """,
            
            """
            CREATE TRIGGER IF NOT EXISTS update_entities_timestamp 
            AFTER UPDATE ON entities
            BEGIN
                UPDATE entities SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END
            """,
            
            """
            CREATE TRIGGER IF NOT EXISTS update_validations_timestamp 
            AFTER UPDATE ON validations
            BEGIN
                UPDATE validations SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END
            """,
            
            # Create views for common queries
            """
            CREATE VIEW IF NOT EXISTS session_summary AS
            SELECT 
                s.id,
                s.created_at,
                s.image_path,
                s.naive_prompt,
                s.status,
                s.final_alignment_score,
                COUNT(p.id) as prompt_count,
                COUNT(e.id) as entity_count,
                COUNT(v.id) as validation_count
            FROM sessions s
            LEFT JOIN prompts p ON s.id = p.session_id
            LEFT JOIN entities e ON s.id = e.session_id
            LEFT JOIN validations v ON s.id = v.session_id
            GROUP BY s.id
            """,
            
            """
            CREATE VIEW IF NOT EXISTS quality_metrics AS
            SELECT 
                s.id as session_id,
                s.final_alignment_score,
                AVG(p.quality_score) as avg_prompt_quality,
                AVG(v.alignment_score) as avg_validation_score
            FROM sessions s
            LEFT JOIN prompts p ON s.id = p.session_id
            LEFT JOIN validations v ON s.id = v.session_id
            GROUP BY s.id
            """
        ]
        
        return sql_statements
    
    def create_initial_schema_v1(self) -> List[str]:
        """
        Create initial v1 schema SQL statements.
        
        Returns:
            List of SQL statements for initial schema
        """
        sql_statements = [
            # Create sessions table
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT NOT NULL,
                naive_prompt TEXT NOT NULL,
                status TEXT CHECK(status IN ('in_progress', 'completed', 'failed')),
                final_alignment_score REAL
            )
            """,
            
            # Create prompts table
            """
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT REFERENCES sessions(id),
                iteration INT,
                positive_prompt TEXT,
                negative_prompt TEXT,
                quality_score REAL
            )
            """,
            
            # Create entities table
            """
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT REFERENCES sessions(id),
                entity_id TEXT,
                label TEXT,
                confidence REAL,
                bbox_json TEXT,
                mask_path TEXT,
                color_hex TEXT,
                area_percent REAL
            )
            """,
            
            # Create validations table
            """
            CREATE TABLE IF NOT EXISTS validations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT REFERENCES sessions(id),
                attempt_number INT,
                alignment_score REAL,
                preserved_count INT,
                modified_count INT,
                unintended_count INT,
                user_feedback TEXT
            )
            """,
            
            # Create indexes
            "CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_prompts_session ON prompts(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_prompts_iteration ON prompts(iteration)",
            "CREATE INDEX IF NOT EXISTS idx_entities_session ON entities(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_entities_label ON entities(label)",
            "CREATE INDEX IF NOT EXISTS idx_validations_session ON validations(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_validations_attempt ON validations(attempt_number)",
            
            # Enable foreign key constraints
            "PRAGMA foreign_keys = ON",
            
            # Create schema version table
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL UNIQUE,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                migration_notes TEXT
            )
            """,
            
            # Insert initial version
            "INSERT OR IGNORE INTO schema_version (version, migration_notes) VALUES ('v1', 'Initial schema setup')",
        ]
        
        return sql_statements
    
    def validate_schema_integrity(self) -> Dict[str, Any]:
        """
        Validate database schema integrity.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "status": "unknown",
            "tables": {},
            "indexes": {},
            "constraints": {},
            "views": {},
            "triggers": {}
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if required tables exist
                required_tables = ["sessions", "prompts", "entities", "validations", "schema_version"]
                for table in required_tables:
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
                    exists = cursor.fetchone() is not None
                    validation_results["tables"][table] = exists
                
                # Check if required indexes exist
                required_indexes = [
                    "idx_sessions_status", "idx_sessions_created", "idx_prompts_session",
                    "idx_prompts_iteration", "idx_entities_session", "idx_entities_label",
                    "idx_validations_session", "idx_validations_attempt"
                ]
                for index in required_indexes:
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name=?", (index,))
                    exists = cursor.fetchone() is not None
                    validation_results["indexes"][index] = exists
                
                # Check schema version
                cursor.execute("SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1")
                version_row = cursor.fetchone()
                if version_row:
                    validation_results["current_version"] = version_row[0]
                else:
                    validation_results["current_version"] = "none"
                
                # Determine overall status
                all_tables_exist = all(validation_results["tables"].values())
                all_indexes_exist = all(validation_results["indexes"].values())
                
                if all_tables_exist and all_indexes_exist:
                    validation_results["status"] = "valid"
                elif all_tables_exist:
                    validation_results["status"] = "partial"
                else:
                    validation_results["status"] = "invalid"
                
                return validation_results
                
        except sqlite3.Error as e:
            self.logger.error(f"Schema validation failed: {str(e)}")
            validation_results["status"] = "error"
            validation_results["error"] = str(e)
            return validation_results
    
    def backup_schema(self, backup_path: str) -> bool:
        """
        Create a backup of the database schema.
        
        Args:
            backup_path: Path to save backup
            
        Returns:
            Boolean indicating success
        """
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            self.logger.info(f"Schema backup created: {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create schema backup: {str(e)}")
            return False
    
    def restore_schema(self, backup_path: str) -> bool:
        """
        Restore database schema from backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            Boolean indicating success
        """
        try:
            import shutil
            shutil.copy2(backup_path, self.db_path)
            self.logger.info(f"Schema restored from: {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to restore schema: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize SQL migration executor
    sql_executor = SQLMigrationExecutor("edi_sql_test.db")
    
    print("SQL Migration Executor initialized")
    
    # Create initial schema
    print("Creating initial v1 schema...")
    initial_sql = sql_executor.create_initial_schema_v1()
    success = sql_executor.execute_sql_migration(initial_sql)
    if success:
        print("Initial schema created successfully!")
    else:
        print("Failed to create initial schema!")
    
    # Set initial version
    sql_executor.set_current_version("v1", "Initial schema setup")
    
    # Check current version
    current_version = sql_executor.get_current_version()
    print(f"Current version: {current_version}")
    
    # Check if migration needed
    needs_migration = sql_executor.needs_migration()
    print(f"Migration needed: {needs_migration}")
    
    # Get pending migrations
    pending = sql_executor.get_pending_migrations()
    print(f"Pending migrations: {len(pending)}")
    
    # Get migration history
    history = sql_executor.get_migration_history()
    print(f"Migration history ({len(history)} records):")
    for record in history:
        print(f"  {record['migration_name']}: {record['version_from']} -> {record['version_to']}")
        print(f"    Status: {record['status']}")
        print(f"    Started: {record['started_at'][:19]}")
        if record['completed_at']:
            print(f"    Completed: {record['completed_at'][:19]}")
        if record['error_message']:
            print(f"    Error: {record['error_message']}")
        if record['duration_seconds']:
            print(f"    Duration: {record['duration_seconds']:.2f}s")
    
    # Create v1 to v2 migration
    print("\nCreating v1 to v2 migration script...")
    migration_sql = sql_executor.create_migration_script_v1_to_v2()
    print(f"Generated {len(migration_sql)} SQL statements for migration")
    
    # Show first few statements
    print("First 5 migration statements:")
    for i, stmt in enumerate(migration_sql[:5]):
        print(f"  {i+1}. {stmt[:60]}...")
    
    # Execute migration
    print("\nExecuting v1 to v2 migration...")
    success = sql_executor.execute_sql_migration(migration_sql)
    if success:
        print("Migration executed successfully!")
        sql_executor.set_current_version("v2", "Migrated from v1 to v2")
    else:
        print("Migration failed!")
    
    # Check if version is up to date
    is_up_to_date = sql_executor.is_version_up_to_date("v2")
    print(f"\nVersion up to date: {is_up_to_date}")
    
    # Validate schema integrity
    validation = sql_executor.validate_schema_integrity()
    print(f"\nSchema validation status: {validation['status']}")
    print(f"Current schema version: {validation.get('current_version', 'unknown')}")
    
    # Show table existence
    print("Tables:")
    for table, exists in validation["tables"].items():
        status = "✓" if exists else "✗"
        print(f"  {status} {table}")
    
    # Show index existence
    print("Indexes:")
    for index, exists in validation["indexes"].items():
        status = "✓" if exists else "✗"
        print(f"  {status} {index}")
    
    print("SQL schema changes and version management example completed!")
```

## See Docs

### Database Migration Tools Implementation Example
Storage migrations implementation for the EDI application:

```python
import sqlite3
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

class MigrationError(Exception):
    """Custom exception for migration errors."""
    pass

class MigrationStatus(Enum):
    """Enumeration for migration statuses."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class MigrationRecord:
    """Record of a migration execution."""
    id: int
    name: str
    version_from: str
    version_to: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: MigrationStatus = MigrationStatus.PENDING
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None

class StorageMigrationManager:
    """
    Storage migrations manager for handling database schema changes.
    """
    
    def __init__(self, db_path: str = "edi_sessions.db", migrations_dir: str = "migrations"):
        self.db_path = db_path
        self.migrations_dir = Path(migrations_dir)
        self.migrations_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self._initialize_migration_tracking()
    
    def _initialize_migration_tracking(self):
        """Initialize migration tracking tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
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
                
                # Create schema version table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS schema_version (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        version TEXT NOT NULL UNIQUE,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        migration_notes TEXT
                    )
                ''')
                
                # Create index for performance
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_migration_log_status 
                    ON migration_log(status)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_schema_version_applied 
                    ON schema_version(applied_at DESC)
                ''')
                
                conn.commit()
                
        except sqlite3.Error as e:
            raise MigrationError(f"Failed to initialize migration tracking: {str(e)}")
    
    def get_current_version(self) -> str:
        """
        Get the current database schema version.
        
        Returns:
            Current version string
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get the latest version
                cursor.execute('''
                    SELECT version FROM schema_version 
                    ORDER BY applied_at DESC LIMIT 1
                ''')
                
                version_row = cursor.fetchone()
                if version_row:
                    return version_row[0]
                
                # If no version found, check if tables exist to determine legacy version
                cursor.execute('''
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name IN ('sessions', 'prompts', 'entities', 'validations')
                ''')
                
                tables = cursor.fetchall()
                if tables:
                    return "v1"  # Legacy version with basic tables
                
                return "none"  # No schema yet
                
        except sqlite3.Error as e:
            self.logger.error(f"Error getting current version: {str(e)}")
            return "error"
    
    def set_current_version(self, version: str, notes: str = "") -> bool:
        """
        Set the current database schema version.
        
        Args:
            version: Version string to set
            notes: Optional notes about the version change
            
        Returns:
            Boolean indicating success
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert new version record
                cursor.execute('''
                    INSERT OR REPLACE INTO schema_version 
                    (version, migration_notes)
                    VALUES (?, ?)
                ''', (version, notes))
                
                conn.commit()
                self.logger.info(f"Set schema version to: {version}")
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Error setting version {version}: {str(e)}")
            return False
    
    def create_migration_record(self, 
                              migration_name: str, 
                              version_from: str, 
                              version_to: str) -> int:
        """
        Create a migration record in the log.
        
        Args:
            migration_name: Name of the migration
            version_from: Starting version
            version_to: Target version
            
        Returns:
            ID of the created migration record
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO migration_log 
                    (migration_name, version_from, version_to, status)
                    VALUES (?, ?, ?, ?)
                ''', (migration_name, version_from, version_to, MigrationStatus.PENDING.value))
                
                migration_id = cursor.lastrowid
                conn.commit()
                
                self.logger.info(f"Created migration record {migration_id}: {migration_name}")
                return migration_id
                
        except sqlite3.Error as e:
            raise MigrationError(f"Failed to create migration record: {str(e)}")
    
    def update_migration_status(self, 
                              migration_id: int, 
                              status: MigrationStatus, 
                              error_message: Optional[str] = None) -> bool:
        """
        Update the status of a migration.
        
        Args:
            migration_id: ID of migration record
            status: New status
            error_message: Optional error message
            
        Returns:
            Boolean indicating success
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
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
                self.logger.info(f"Updated migration {migration_id} status to: {status.value}")
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Error updating migration {migration_id} status: {str(e)}")
            return False
    
    def get_pending_migrations(self) -> List[MigrationRecord]:
        """
        Get list of pending migrations.
        
        Returns:
            List of pending migration records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, migration_name, version_from, version_to, started_at, completed_at, status, error_message, duration_seconds
                    FROM migration_log 
                    WHERE status = ?
                    ORDER BY started_at
                ''', (MigrationStatus.PENDING.value,))
                
                rows = cursor.fetchall()
                pending_migrations = []
                
                for row in rows:
                    pending_migrations.append(MigrationRecord(
                        id=row[0],
                        name=row[1],
                        version_from=row[2],
                        version_to=row[3],
                        started_at=datetime.fromisoformat(row[4]) if row[4] else datetime.now(),
                        completed_at=datetime.fromisoformat(row[5]) if row[5] else None,
                        status=MigrationStatus(row[6]) if row[6] in [s.value for s in MigrationStatus] else MigrationStatus.PENDING,
                        error_message=row[7],
                        duration_seconds=row[8]
                    ))
                
                return pending_migrations
                
        except sqlite3.Error as e:
            self.logger.error(f"Error getting pending migrations: {str(e)}")
            return []
    
    def run_migration(self, migration_name: str, version_from: str, version_to: str) -> bool:
        """
        Run a specific migration.
        
        Args:
            migration_name: Name of migration to run
            version_from: Starting version
            version_to: Target version
            
        Returns:
            Boolean indicating success
        """
        # Create migration record
        migration_id = self.create_migration_record(migration_name, version_from, version_to)
        
        # Update status to in progress
        self.update_migration_status(migration_id, MigrationStatus.IN_PROGRESS)
        
        try:
            start_time = datetime.now()
            
            # Execute migration logic based on name
            if migration_name == "migrate_v1_to_v2":
                success = self._execute_v1_to_v2_migration()
            else:
                raise MigrationError(f"Unknown migration: {migration_name}")
            
            duration = (datetime.now() - start_time).total_seconds()
            
            if success:
                # Update status to completed
                self.update_migration_status(migration_id, MigrationStatus.COMPLETED)
                
                # Update schema version
                self.set_current_version(version_to, f"Applied migration {migration_name}")
                
                self.logger.info(f"Migration {migration_name} completed successfully in {duration:.2f}s")
                return True
            else:
                # Update status to failed
                self.update_migration_status(migration_id, MigrationStatus.FAILED, "Migration execution failed")
                self.logger.error(f"Migration {migration_name} failed after {duration:.2f}s")
                return False
                
        except Exception as e:
            # Update status to failed with error
            self.update_migration_status(migration_id, MigrationStatus.FAILED, str(e))
            self.logger.error(f"Migration {migration_name} failed with exception: {str(e)}")
            return False
    
    def _execute_v1_to_v2_migration(self) -> bool:
        """
        Execute the v1 to v2 migration.
        
        Returns:
            Boolean indicating success
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Begin transaction
                conn.execute("BEGIN")
                
                # Add new columns to existing tables
                # Add processing_time to sessions
                try:
                    cursor.execute('''
                        ALTER TABLE sessions 
                        ADD COLUMN processing_time REAL
                    ''')
                except sqlite3.OperationalError:
                    # Column might already exist
                    pass
                
                # Add model_version to sessions
                try:
                    cursor.execute('''
                        ALTER TABLE sessions 
                        ADD COLUMN model_version TEXT
                    ''')
                except sqlite3.OperationalError:
                    # Column might already exist
                    pass
                
                # Add generation_time to prompts
                try:
                    cursor.execute('''
                        ALTER TABLE prompts 
                        ADD COLUMN generation_time REAL
                    ''')
                except sqlite3.OperationalError:
                    # Column might already exist
                    pass
                
                # Add model_confidence to prompts
                try:
                    cursor.execute('''
                        ALTER TABLE prompts 
                        ADD COLUMN model_confidence REAL
                    ''')
                except sqlite3.OperationalError:
                    # Column might already exist
                    pass
                
                # Add embedding_vector to entities
                try:
                    cursor.execute('''
                        ALTER TABLE entities 
                        ADD COLUMN embedding_vector TEXT
                    ''')
                except sqlite3.OperationalError:
                    # Column might already exist
                    pass
                
                # Add semantic_label to entities
                try:
                    cursor.execute('''
                        ALTER TABLE entities 
                        ADD COLUMN semantic_label TEXT
                    ''')
                except sqlite3.OperationalError:
                    # Column might already exist
                    pass
                
                # Add processing_details to validations
                try:
                    cursor.execute('''
                        ALTER TABLE validations 
                        ADD COLUMN processing_details TEXT
                    ''')
                except sqlite3.OperationalError:
                    # Column might already exist
                    pass
                
                # Add quality_metrics to validations
                try:
                    cursor.execute('''
                        ALTER TABLE validations 
                        ADD COLUMN quality_metrics TEXT
                    ''')
                except sqlite3.OperationalError:
                    # Column might already exist
                    pass
                
                # Create indexes for improved performance
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
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_validations_quality_metrics 
                    ON validations(quality_metrics)
                ''')
                
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
                
                # Commit transaction
                conn.execute("COMMIT")
                
                self.logger.info("v1 to v2 migration executed successfully")
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"v1 to v2 migration failed: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error in v1 to v2 migration: {str(e)}")
            return False
    
    def rollback_migration(self, migration_id: int) -> bool:
        """
        Rollback a failed migration.
        
        Args:
            migration_id: ID of migration to rollback
            
        Returns:
            Boolean indicating success
        """
        try:
            # Get migration details
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT migration_name, version_from, version_to
                    FROM migration_log
                    WHERE id = ?
                ''', (migration_id,))
                
                row = cursor.fetchone()
                if not row:
                    self.logger.error(f"Migration {migration_id} not found")
                    return False
                
                migration_name, version_from, version_to = row
            
            # Execute rollback logic based on migration name
            if migration_name == "migrate_v1_to_v2":
                success = self._rollback_v1_to_v2_migration()
            else:
                self.logger.error(f"Unknown migration for rollback: {migration_name}")
                return False
            
            if success:
                # Update status to rolled back
                self.update_migration_status(migration_id, MigrationStatus.ROLLED_BACK)
                
                # Restore previous schema version
                self.set_current_version(version_from, f"Rolled back migration {migration_name}")
                
                self.logger.info(f"Migration {migration_name} rolled back successfully")
                return True
            else:
                self.logger.error(f"Failed to rollback migration {migration_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error rolling back migration {migration_id}: {str(e)}")
            return False
    
    def _rollback_v1_to_v2_migration(self) -> bool:
        """
        Rollback the v1 to v2 migration.
        
        Returns:
            Boolean indicating success
        """
        # In SQLite, rolling back schema changes is complex
        # A more realistic approach would involve restoring from backup
        self.logger.warning("Rollback of v1 to v2 migration would require database restoration from backup")
        return False
    
    def get_migration_history(self) -> List[Dict[str, Any]]:
        """
        Get history of all migrations.
        
        Returns:
            List of migration history records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, migration_name, version_from, version_to, 
                           started_at, completed_at, status, error_message, duration_seconds
                    FROM migration_log
                    ORDER BY started_at DESC
                ''')
                
                rows = cursor.fetchall()
                history = []
                
                for row in rows:
                    history.append({
                        "id": row[0],
                        "migration_name": row[1],
                        "version_from": row[2],
                        "version_to": row[3],
                        "started_at": row[4],
                        "completed_at": row[5],
                        "status": row[6],
                        "error_message": row[7],
                        "duration_seconds": row[8]
                    })
                
                return history
                
        except sqlite3.Error as e:
            self.logger.error(f"Error getting migration history: {str(e)}")
            return []
    
    def needs_migration(self) -> bool:
        """
        Check if database needs migration to a newer version.
        
        Returns:
            Boolean indicating if migration is needed
        """
        current_version = self.get_current_version()
        
        # Simple version comparison (would be more complex in real implementation)
        if current_version == "none":
            return True  # Need initial setup
        elif current_version == "v1":
            return True  # Need upgrade to v2
        else:
            return False  # Already at latest version

# Example usage
if __name__ == "__main__":
    # Initialize storage migration manager
    migration_manager = StorageMigrationManager("edi_test.db", "test_migrations")
    
    print("Storage Migration Manager initialized")
    
    # Check current version
    current_version = migration_manager.get_current_version()
    print(f"Current schema version: {current_version}")
    
    # Check if migration needed
    needs_migration = migration_manager.needs_migration()
    print(f"Migration needed: {needs_migration}")
    
    # Get pending migrations
    pending = migration_manager.get_pending_migrations()
    print(f"Pending migrations: {len(pending)}")
    
    # Get migration history
    history = migration_manager.get_migration_history()
    print(f"Migration history entries: {len(history)}")
    
    # Example: Run a migration
    if needs_migration and current_version in ["none", "v1"]:
        if current_version == "none":
            print("Setting up initial schema...")
            # This would create the initial v1 schema
            migration_manager.set_current_version("v1", "Initial schema setup")
        elif current_version == "v1":
            print("Running v1 to v2 migration...")
            success = migration_manager.run_migration("migrate_v1_to_v2", "v1", "v2")
            if success:
                print("Migration completed successfully!")
            else:
                print("Migration failed!")
    
    # Show updated version
    updated_version = migration_manager.get_current_version()
    print(f"Updated schema version: {updated_version}")
    
    print("Storage migrations example completed!")
```

### SQL Implementation Example
Database schema changes and version management with SQL:

```python
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import json

class SQLMigrationExecutor:
    """
    SQL-based migration executor for database schema changes.
    """
    
    def __init__(self, db_path: str = "edi_sessions.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
    
    def execute_sql_migration(self, sql_statements: List[str]) -> bool:
        """
        Execute a series of SQL statements for migration.
        
        Args:
            sql_statements: List of SQL statements to execute
            
        Returns:
            Boolean indicating success
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Begin transaction for atomic migration
                conn.execute("BEGIN")
                
                # Execute each SQL statement
                for statement in sql_statements:
                    try:
                        cursor.execute(statement)
                        self.logger.debug(f"Executed: {statement[:50]}...")
                    except sqlite3.OperationalError as e:
                        # Handle common migration errors (like duplicate columns)
                        if "duplicate column name" in str(e).lower():
                            self.logger.warning(f"Column already exists, skipping: {statement[:50]}...")
                            continue
                        elif "already exists" in str(e).lower():
                            self.logger.warning(f"Object already exists, skipping: {statement[:50]}...")
                            continue
                        else:
                            # Re-raise other errors
                            raise e
                
                # Commit transaction
                conn.execute("COMMIT")
                
                self.logger.info(f"Executed {len(sql_statements)} SQL statements successfully")
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"SQL migration failed: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error in SQL migration: {str(e)}")
            return False
    
    def create_migration_script_v1_to_v2(self) -> List[str]:
        """
        Create SQL migration script for v1 to v2 upgrade.
        
        Returns:
            List of SQL statements for the migration
        """
        sql_statements = [
            # Add new columns to sessions table
            "ALTER TABLE sessions ADD COLUMN processing_time REAL",
            "ALTER TABLE sessions ADD COLUMN model_version TEXT",
            "ALTER TABLE sessions ADD COLUMN session_tags TEXT",
            
            # Add new columns to prompts table
            "ALTER TABLE prompts ADD COLUMN generation_time REAL",
            "ALTER TABLE prompts ADD COLUMN model_confidence REAL",
            "ALTER TABLE prompts ADD COLUMN generation_parameters TEXT",
            
            # Add new columns to entities table
            "ALTER TABLE entities ADD COLUMN embedding_vector TEXT",
            "ALTER TABLE entities ADD COLUMN semantic_label TEXT",
            "ALTER TABLE entities ADD COLUMN entity_metadata TEXT",
            
            # Add new columns to validations table
            "ALTER TABLE validations ADD COLUMN processing_details TEXT",
            "ALTER TABLE validations ADD COLUMN quality_metrics TEXT",
            "ALTER TABLE validations ADD COLUMN validation_metadata TEXT",
            
            # Create indexes for improved performance
            "CREATE INDEX IF NOT EXISTS idx_sessions_model_version ON sessions(model_version)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_processing_time ON sessions(processing_time)",
            "CREATE INDEX IF NOT EXISTS idx_prompts_generation_time ON prompts(generation_time)",
            "CREATE INDEX IF NOT EXISTS idx_entities_semantic_label ON entities(semantic_label)",
            "CREATE INDEX IF NOT EXISTS idx_validations_quality_metrics ON validations(quality_metrics)",
            
            # Set default values for new columns
            "UPDATE sessions SET model_version = 'v1_default', processing_time = 0.0 WHERE model_version IS NULL OR processing_time IS NULL",
            "UPDATE prompts SET generation_time = 0.0, model_confidence = 1.0 WHERE generation_time IS NULL OR model_confidence IS NULL",
            "UPDATE entities SET semantic_label = label WHERE semantic_label IS NULL",
            
            # Create new tables if needed
            """
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT REFERENCES sessions(id),
                feedback_type TEXT CHECK(feedback_type IN ('accept', 'reject', 'partial')),
                comments TEXT,
                rating INT CHECK(rating BETWEEN 1 AND 5),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            """
            CREATE TABLE IF NOT EXISTS processing_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT REFERENCES sessions(id),
                log_level TEXT CHECK(log_level IN ('debug', 'info', 'warning', 'error')),
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Create additional indexes for new tables
            "CREATE INDEX IF NOT EXISTS idx_user_feedback_session ON user_feedback(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_user_feedback_rating ON user_feedback(rating)",
            "CREATE INDEX IF NOT EXISTS idx_processing_logs_session ON processing_logs(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_processing_logs_timestamp ON processing_logs(timestamp)",
            
            # Add foreign key constraints (if supported by SQLite version)
            # Note: SQLite requires foreign key constraints to be enabled
            "PRAGMA foreign_keys = ON",
            
            # Create triggers for automatic timestamp updates
            """
            CREATE TRIGGER IF NOT EXISTS update_sessions_timestamp 
            AFTER UPDATE ON sessions
            BEGIN
                UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END
            """,
            
            """
            CREATE TRIGGER IF NOT EXISTS update_prompts_timestamp 
            AFTER UPDATE ON prompts
            BEGIN
                UPDATE prompts SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END
            """,
            
            """
            CREATE TRIGGER IF NOT EXISTS update_entities_timestamp 
            AFTER UPDATE ON entities
            BEGIN
                UPDATE entities SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END
            """,
            
            """
            CREATE TRIGGER IF NOT EXISTS update_validations_timestamp 
            AFTER UPDATE ON validations
            BEGIN
                UPDATE validations SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END
            """,
            
            # Create views for common queries
            """
            CREATE VIEW IF NOT EXISTS session_summary AS
            SELECT 
                s.id,
                s.created_at,
                s.image_path,
                s.naive_prompt,
                s.status,
                s.final_alignment_score,
                COUNT(p.id) as prompt_count,
                COUNT(e.id) as entity_count,
                COUNT(v.id) as validation_count
            FROM sessions s
            LEFT JOIN prompts p ON s.id = p.session_id
            LEFT JOIN entities e ON s.id = e.session_id
            LEFT JOIN validations v ON s.id = v.session_id
            GROUP BY s.id
            """,
            
            """
            CREATE VIEW IF NOT EXISTS quality_metrics AS
            SELECT 
                s.id as session_id,
                s.final_alignment_score,
                AVG(p.quality_score) as avg_prompt_quality,
                AVG(v.alignment_score) as avg_validation_score,
                COUNT(v.preserved_count) as preserved_count,
                COUNT(v.modified_count) as modified_count,
                COUNT(v.unintended_count) as unintended_count
            FROM sessions s
            LEFT JOIN prompts p ON s.id = p.session_id
            LEFT JOIN validations v ON s.id = v.session_id
            GROUP BY s.id
            """,
            
            # Create stored procedures (using SQLite user-defined functions)
            # Note: SQLite doesn't have stored procedures, but we can create helper functions
        ]
        
        return sql_statements
    
    def create_initial_schema_v1(self) -> List[str]:
        """
        Create initial v1 schema SQL statements.
        
        Returns:
            List of SQL statements for initial schema
        """
        sql_statements = [
            # Create sessions table
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT NOT NULL,
                naive_prompt TEXT NOT NULL,
                status TEXT CHECK(status IN ('in_progress', 'completed', 'failed')),
                final_alignment_score REAL
            )
            """,
            
            # Create prompts table
            """
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT REFERENCES sessions(id),
                iteration INT,
                positive_prompt TEXT,
                negative_prompt TEXT,
                quality_score REAL
            )
            """,
            
            # Create entities table
            """
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT REFERENCES sessions(id),
                entity_id TEXT,
                label TEXT,
                confidence REAL,
                bbox_json TEXT,
                mask_path TEXT,
                color_hex TEXT,
                area_percent REAL
            )
            """,
            
            # Create validations table
            """
            CREATE TABLE IF NOT EXISTS validations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT REFERENCES sessions(id),
                attempt_number INT,
                alignment_score REAL,
                preserved_count INT,
                modified_count INT,
                unintended_count INT,
                user_feedback TEXT
            )
            """,
            
            # Create indexes
            "CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_prompts_session ON prompts(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_prompts_iteration ON prompts(iteration)",
            "CREATE INDEX IF NOT EXISTS idx_entities_session ON entities(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_entities_label ON entities(label)",
            "CREATE INDEX IF NOT EXISTS idx_validations_session ON validations(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_validations_attempt ON validations(attempt_number)",
            
            # Enable foreign key constraints
            "PRAGMA foreign_keys = ON",
            
            # Create schema version table
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL UNIQUE,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                migration_notes TEXT
            )
            """,
            
            # Insert initial version
            "INSERT OR IGNORE INTO schema_version (version, migration_notes) VALUES ('v1', 'Initial schema setup')",
        ]
        
        return sql_statements
    
    def validate_schema_integrity(self) -> Dict[str, Any]:
        """
        Validate database schema integrity.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "status": "unknown",
            "tables": {},
            "indexes": {},
            "constraints": {},
            "views": {},
            "triggers": {}
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if required tables exist
                required_tables = ["sessions", "prompts", "entities", "validations", "schema_version"]
                for table in required_tables:
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
                    exists = cursor.fetchone() is not None
                    validation_results["tables"][table] = exists
                
                # Check if required indexes exist
                required_indexes = [
                    "idx_sessions_status", "idx_sessions_created", "idx_prompts_session",
                    "idx_prompts_iteration", "idx_entities_session", "idx_entities_label",
                    "idx_validations_session", "idx_validations_attempt"
                ]
                for index in required_indexes:
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name=?", (index,))
                    exists = cursor.fetchone() is not None
                    validation_results["indexes"][index] = exists
                
                # Check schema version
                cursor.execute("SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1")
                version_row = cursor.fetchone()
                if version_row:
                    validation_results["current_version"] = version_row[0]
                else:
                    validation_results["current_version"] = "none"
                
                # Determine overall status
                all_tables_exist = all(validation_results["tables"].values())
                all_indexes_exist = all(validation_results["indexes"].values())
                
                if all_tables_exist and all_indexes_exist:
                    validation_results["status"] = "valid"
                elif all_tables_exist:
                    validation_results["status"] = "partial"
                else:
                    validation_results["status"] = "invalid"
                
                return validation_results
                
        except sqlite3.Error as e:
            self.logger.error(f"Schema validation failed: {str(e)}")
            validation_results["status"] = "error"
            validation_results["error"] = str(e)
            return validation_results
    
    def backup_schema(self, backup_path: str) -> bool:
        """
        Create a backup of the database schema.
        
        Args:
            backup_path: Path to save backup
            
        Returns:
            Boolean indicating success
        """
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            self.logger.info(f"Schema backup created: {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create schema backup: {str(e)}")
            return False
    
    def restore_schema(self, backup_path: str) -> bool:
        """
        Restore database schema from backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            Boolean indicating success
        """
        try:
            import shutil
            shutil.copy2(backup_path, self.db_path)
            self.logger.info(f"Schema restored from: {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to restore schema: {str(e)}")
            return False

class VersionManager:
    """
    Version management for database schema.
    """
    
    def __init__(self, db_path: str = "edi_sessions.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
    
    def get_current_version(self) -> str:
        """
        Get current database schema version.
        
        Returns:
            Current version string
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if schema_version table exists
                cursor.execute('''
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='schema_version'
                ''')
                
                if not cursor.fetchone():
                    # No version table, check if basic tables exist
                    cursor.execute('''
                        SELECT COUNT(*) FROM sqlite_master 
                        WHERE type='table' AND name IN ('sessions', 'prompts', 'entities', 'validations')
                    ''')
                    
                    table_count = cursor.fetchone()[0]
                    if table_count >= 4:
                        return "v1"  # Legacy version with basic tables
                    else:
                        return "none"  # No schema yet
                
                # Get current version from version table
                cursor.execute('''
                    SELECT version FROM schema_version 
                    ORDER BY applied_at DESC LIMIT 1
                ''')
                
                version_row = cursor.fetchone()
                if version_row:
                    return version_row[0]
                else:
                    return "none"  # Empty version table
            
        except sqlite3.Error as e:
            self.logger.error(f"Error getting current version: {str(e)}")
            return "error"
    
    def set_current_version(self, version: str, notes: str = "") -> bool:
        """
        Set current database schema version.
        
        Args:
            version: Version string to set
            notes: Optional notes about the version change
            
        Returns:
            Boolean indicating success
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create schema_version table if it doesn't exist
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS schema_version (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        version TEXT NOT NULL UNIQUE,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        migration_notes TEXT
                    )
                ''')
                
                # Insert or update version record
                cursor.execute('''
                    INSERT OR REPLACE INTO schema_version 
                    (version, migration_notes)
                    VALUES (?, ?)
                ''', (version, notes))
                
                conn.commit()
                self.logger.info(f"Set schema version to: {version}")
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Error setting version {version}: {str(e)}")
            return False
    
    def get_version_history(self) -> List[Dict[str, Any]]:
        """
        Get history of schema version changes.
        
        Returns:
            List of version history records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get version history ordered by application time
                cursor.execute('''
                    SELECT version, applied_at, migration_notes
                    FROM schema_version
                    ORDER BY applied_at DESC
                ''')
                
                rows = cursor.fetchall()
                history = []
                
                for row in rows:
                    history.append({
                        "version": row[0],
                        "applied_at": row[1],
                        "migration_notes": row[2]
                    })
                
                return history
                
        except sqlite3.Error as e:
            self.logger.error(f"Error getting version history: {str(e)}")
            return []
    
    def compare_versions(self, version1: str, version2: str) -> int:
        """
        Compare two version strings.
        
        Args:
            version1: First version string
            version2: Second version string
            
        Returns:
            -1 if version1 < version2, 0 if equal, 1 if version1 > version2
        """
        # Simple version comparison (would be more complex for semantic versioning)
        if version1 == version2:
            return 0
        
        # Convert to numeric versions
        try:
            v1_num = float(version1.replace("v", ""))
            v2_num = float(version2.replace("v", ""))
            
            if v1_num < v2_num:
                return -1
            elif v1_num > v2_num:
                return 1
            else:
                return 0
        except ValueError:
            # String comparison for non-numeric versions
            if version1 < version2:
                return -1
            elif version1 > version2:
                return 1
            else:
                return 0
    
    def is_version_up_to_date(self, target_version: str = "v2") -> bool:
        """
        Check if current version is up to date.
        
        Args:
            target_version: Target version to check against
            
        Returns:
            Boolean indicating if current version is up to date
        """
        current_version = self.get_current_version()
        return self.compare_versions(current_version, target_version) >= 0

# Example usage
if __name__ == "__main__":
    # Initialize SQL migration executor
    sql_executor = SQLMigrationExecutor("edi_test.db")
    version_manager = VersionManager("edi_test.db")
    
    print("SQL Migration Executor and Version Manager initialized")
    
    # Create initial schema
    print("Creating initial v1 schema...")
    initial_sql = sql_executor.create_initial_schema_v1()
    success = sql_executor.execute_sql_migration(initial_sql)
    if success:
        print("Initial schema created successfully!")
    else:
        print("Failed to create initial schema!")
    
    # Set initial version
    version_manager.set_current_version("v1", "Initial schema setup")
    
    # Check current version
    current_version = version_manager.get_current_version()
    print(f"Current version: {current_version}")
    
    # Get version history
    version_history = version_manager.get_version_history()
    print(f"Version history ({len(version_history)} entries):")
    for entry in version_history:
        print(f"  {entry['version']} - {entry['applied_at'][:19]} - {entry['migration_notes']}")
    
    # Validate schema integrity
    validation = sql_executor.validate_schema_integrity()
    print(f"Schema validation status: {validation['status']}")
    print(f"Current schema version: {validation.get('current_version', 'unknown')}")
    
    # Show table existence
    print("Tables:")
    for table, exists in validation["tables"].items():
        status = "✓" if exists else "✗"
        print(f"  {status} {table}")
    
    # Show index existence
    print("Indexes:")
    for index, exists in validation["indexes"].items():
        status = "✓" if exists else "✗"
        print(f"  {status} {index}")
    
    # Create v1 to v2 migration
    print("\nCreating v1 to v2 migration script...")
    migration_sql = sql_executor.create_migration_script_v1_to_v2()
    print(f"Generated {len(migration_sql)} SQL statements for migration")
    
    # Show first few statements
    print("First 5 migration statements:")
    for i, stmt in enumerate(migration_sql[:5]):
        print(f"  {i+1}. {stmt[:60]}...")
    
    # Check if version is up to date
    is_up_to_date = version_manager.is_version_up_to_date("v2")
    print(f"\nVersion up to date: {is_up_to_date}")
    
    print("SQL schema changes and version management example completed!")
```

## See Docs

### Python Implementation Example
Storage migrations subsystem implementation:

```python
import sqlite3
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import logging

class MigrationError(Exception):
    """Custom exception for migration errors."""
    pass

class MigrationStatus(Enum):
    """Enumeration for migration statuses."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

class StorageMigrations:
    """
    Storage migrations subsystem for handling database schema changes.
    """
    
    def __init__(self, db_path: str = "edi_sessions.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.migrations = {}
        self._register_migrations()
    
    def _register_migrations(self):
        """
        Register available migration functions.
        """
        # Register migration functions
        self.migrations["v1_to_v2"] = {
            "from_version": "v1",
            "to_version": "v2",
            "function": self.migrate_v1_to_v2,
            "description": "Migrate database schema from version 1 to 2"
        }
        
        # Add more migrations as needed
        # self.migrations["v2_to_v3"] = {
        #     "from_version": "v2",
        #     "to_version": "v3",
        #     "function": self.migrate_v2_to_v3,
        #     "description": "Migrate database schema from version 2 to 3"
        # }
    
    def migrate_v1_to_v2(self) -> bool:
        """
        Migrates database schema from version 1 to 2.
        
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
            
            import shutil
            shutil.copy2(self.db_path, backup_path)
            
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
            # Find the most recent backup
            import glob
            backup_files = glob.glob(f"{self.db_path}.backup_*")
            if not backup_files:
                self.logger.error("No backup files found for restoration")
                return False
            
            # Sort by timestamp and get most recent
            backup_files.sort(reverse=True)
            latest_backup = backup_files[0]
            
            import shutil
            shutil.copy2(latest_backup, self.db_path)
            self.logger.info(f"Database restored from backup: {latest_backup}")
            return True
                
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
    
    def get_available_migrations(self) -> List[Dict[str, Any]]:
        """
        Get list of available migration functions.
        
        Returns:
            List of available migrations with metadata
        """
        return [
            {
                "name": name,
                "from_version": info["from_version"],
                "to_version": info["to_version"],
                "description": info["description"]
            }
            for name, info in self.migrations.items()
        ]
    
    def run_migration(self, migration_name: str) -> bool:
        """
        Run a specific migration by name.
        
        Args:
            migration_name: Name of migration to run
            
        Returns:
            Boolean - True if migration successful, False otherwise
        """
        if migration_name not in self.migrations:
            self.logger.error(f"Migration '{migration_name}' not found")
            return False
        
        migration_func = self.migrations[migration_name]["function"]
        return migration_func()
    
    def get_current_version(self) -> str:
        """
        Get current database schema version.
        
        Returns:
            Current version string
        """
        return self._get_current_version()
    
    def needs_migration(self) -> bool:
        """
        Check if database needs migration to a newer version.
        
        Returns:
            Boolean - True if migration needed, False otherwise
        """
        current_version = self._get_current_version()
        # Simple version comparison (would be more complex in real implementation)
        return current_version < "v2"  # Assuming v2 is current target
    
    def run_all_pending_migrations(self) -> bool:
        """
        Run all pending migrations in order.
        
        Returns:
            Boolean - True if all migrations successful, False otherwise
        """
        current_version = self._get_current_version()
        target_version = "v2"  # Current target version
        
        if current_version >= target_version:
            self.logger.info("Database is already at or above target version")
            return True
        
        # Run migrations in order
        migration_order = ["v1_to_v2"]  # Would be more complex in real implementation
        
        for migration_name in migration_order:
            if not self.run_migration(migration_name):
                self.logger.error(f"Failed to run migration '{migration_name}'")
                return False
        
        self.logger.info("All pending migrations completed successfully")
        return True

# Example usage
if __name__ == "__main__":
    # Initialize storage migrations
    migrations = StorageMigrations("edi_sessions.db")
    
    # Show available migrations
    available = migrations.get_available_migrations()
    print("Available migrations:")
    for migration in available:
        print(f"  {migration['name']}: {migration['from_version']} -> {migration['to_version']}")
        print(f"    {migration['description']}")
    
    # Check if migration needed
    if migrations.needs_migration():
        print("Migration needed")
        
        # Run all pending migrations
        success = migrations.run_all_pending_migrations()
        if success:
            print("All migrations completed successfully!")
        else:
            print("Migration failed!")
    else:
        print("Database is up to date")
    
    # Show current version
    current_version = migrations.get_current_version()
    print(f"Current database version: {current_version}")
```

### Advanced Migration Framework Implementation
Enhanced migration framework with plugin support and comprehensive management:

```python
import sqlite3
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Type
from abc import ABC, abstractmethod
from enum import Enum
import logging
from dataclasses import dataclass
from contextlib import contextmanager

class MigrationPhase(Enum):
    """Enumeration for migration phases."""
    SETUP = "setup"
    EXECUTE = "execute"
    VERIFY = "verify"
    CLEANUP = "cleanup"

@dataclass
class MigrationInfo:
    """Information about a migration."""
    name: str
    version: str
    description: str
    author: str = "Unknown"
    created_at: str = ""
    dependencies: List[str] = None

class MigrationPlugin(ABC):
    """
    Abstract base class for migration plugins.
    """
    
    def __init__(self, name: str, version: str, description: str):
        self.name = name
        self.version = version
        self.description = description
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def setup(self, connection: sqlite3.Connection) -> bool:
        """
        Setup phase of migration.
        
        Args:
            connection: Database connection
            
        Returns:
            Boolean - True if setup successful, False otherwise
        """
        pass
    
    @abstractmethod
    def execute(self, connection: sqlite3.Connection) -> bool:
        """
        Execute phase of migration.
        
        Args:
            connection: Database connection
            
        Returns:
            Boolean - True if execution successful, False otherwise
        """
        pass
    
    @abstractmethod
    def verify(self, connection: sqlite3.Connection) -> bool:
        """
        Verification phase of migration.
        
        Args:
            connection: Database connection
            
        Returns:
            Boolean - True if verification successful, False otherwise
        """
        pass
    
    @abstractmethod
    def cleanup(self, connection: sqlite3.Connection) -> bool:
        """
        Cleanup phase of migration.
        
        Args:
            connection: Database connection
            
        Returns:
            Boolean - True if cleanup successful, False otherwise
        """
        pass

class AdvancedStorageMigrations:
    """
    Advanced storage migrations framework with plugin support.
    """
    
    def __init__(self, db_path: str = "edi_sessions.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.plugins: Dict[str, MigrationPlugin] = {}
        self.migration_history: List[Dict[str, Any]] = []
        self._initialize_framework()
    
    def _initialize_framework(self):
        """
        Initialize the migration framework.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Create migration tracking tables
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS migration_registry (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        plugin_name TEXT UNIQUE NOT NULL,
                        version TEXT NOT NULL,
                        description TEXT,
                        author TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS migration_executions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        plugin_name TEXT NOT NULL,
                        phase TEXT CHECK(phase IN ('setup', 'execute', 'verify', 'cleanup')),
                        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        completed_at TIMESTAMP,
                        status TEXT CHECK(status IN ('pending', 'in_progress', 'completed', 'failed')),
                        error_message TEXT,
                        duration_seconds REAL
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS migration_dependencies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        plugin_name TEXT NOT NULL,
                        dependency_name TEXT NOT NULL,
                        FOREIGN KEY(plugin_name) REFERENCES migration_registry(plugin_name),
                        FOREIGN KEY(dependency_name) REFERENCES migration_registry(plugin_name)
                    )
                ''')
                
                conn.commit()
                
        except sqlite3.Error as e:
            raise MigrationError(f"Failed to initialize migration framework: {str(e)}")
    
    @contextmanager
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get database connection with error handling.
        """
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
    
    def register_plugin(self, plugin: MigrationPlugin) -> bool:
        """
        Register a migration plugin.
        
        Args:
            plugin: Migration plugin to register
            
        Returns:
            Boolean - True if registration successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Register plugin in registry
                cursor.execute('''
                    INSERT OR REPLACE INTO migration_registry 
                    (plugin_name, version, description, author, created_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    plugin.name,
                    plugin.version,
                    plugin.description,
                    getattr(plugin, 'author', 'Unknown'),
                    getattr(plugin, 'created_at', datetime.now().isoformat())
                ))
                
                conn.commit()
                
                # Store plugin reference
                self.plugins[plugin.name] = plugin
                self.logger.info(f"Registered migration plugin: {plugin.name}")
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to register plugin {plugin.name}: {str(e)}")
            return False
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """
        Unregister a migration plugin.
        
        Args:
            plugin_name: Name of plugin to unregister
            
        Returns:
            Boolean - True if unregistration successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Remove plugin from registry
                cursor.execute('''
                    DELETE FROM migration_registry 
                    WHERE plugin_name = ?
                ''', (plugin_name,))
                
                # Remove dependencies
                cursor.execute('''
                    DELETE FROM migration_dependencies 
                    WHERE plugin_name = ? OR dependency_name = ?
                ''', (plugin_name, plugin_name))
                
                conn.commit()
                
                # Remove from memory
                if plugin_name in self.plugins:
                    del self.plugins[plugin_name]
                
                self.logger.info(f"Unregistered migration plugin: {plugin_name}")
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to unregister plugin {plugin_name}: {str(e)}")
            return False
    
    def run_migration_phases(self, plugin_name: str) -> bool:
        """
        Run all phases of a migration plugin.
        
        Args:
            plugin_name: Name of plugin to run
            
        Returns:
            Boolean - True if all phases successful, False otherwise
        """
        if plugin_name not in self.plugins:
            self.logger.error(f"Plugin '{plugin_name}' not found")
            return False
        
        plugin = self.plugins[plugin_name]
        self.logger.info(f"Running migration plugin: {plugin_name}")
        
        # Check dependencies
        if not self._check_dependencies(plugin_name):
            self.logger.error(f"Dependencies not satisfied for plugin: {plugin_name}")
            return False
        
        # Run phases in order
        phases = [
            (MigrationPhase.SETUP, plugin.setup),
            (MigrationPhase.EXECUTE, plugin.execute),
            (MigrationPhase.VERIFY, plugin.verify),
            (MigrationPhase.CLEANUP, plugin.cleanup)
        ]
        
        for phase, phase_func in phases:
            if not self._run_migration_phase(plugin_name, phase, phase_func):
                self.logger.error(f"Migration phase {phase.value} failed for plugin {plugin_name}")
                return False
        
        self.logger.info(f"Migration plugin {plugin_name} completed successfully")
        return True
    
    def _check_dependencies(self, plugin_name: str) -> bool:
        """
        Check if all dependencies for a plugin are satisfied.
        
        Args:
            plugin_name: Name of plugin to check
            
        Returns:
            Boolean - True if dependencies satisfied, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get dependencies
                cursor.execute('''
                    SELECT dependency_name 
                    FROM migration_dependencies 
                    WHERE plugin_name = ?
                ''', (plugin_name,))
                
                dependencies = [row[0] for row in cursor.fetchall()]
                
                # Check if dependencies are completed
                for dependency in dependencies:
                    cursor.execute('''
                        SELECT COUNT(*) 
                        FROM migration_executions 
                        WHERE plugin_name = ? AND status = 'completed'
                    ''', (dependency,))
                    
                    completed_count = cursor.fetchone()[0]
                    if completed_count == 0:
                        self.logger.error(f"Dependency '{dependency}' not completed for plugin '{plugin_name}'")
                        return False
                
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to check dependencies for {plugin_name}: {str(e)}")
            return False
    
    def _run_migration_phase(self, 
                           plugin_name: str, 
                           phase: MigrationPhase, 
                           phase_func: Callable) -> bool:
        """
        Run a single migration phase.
        
        Args:
            plugin_name: Name of plugin
            phase: Migration phase
            phase_func: Function to execute for the phase
            
        Returns:
            Boolean - True if phase successful, False otherwise
        """
        # Log phase start
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Create execution log entry
                cursor.execute('''
                    INSERT INTO migration_executions 
                    (plugin_name, phase, status)
                    VALUES (?, ?, ?)
                ''', (plugin_name, phase.value, MigrationStatus.PENDING.value))
                
                execution_id = cursor.lastrowid
                conn.commit()
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to create execution log for {plugin_name} {phase.value}: {str(e)}")
            return False
        
        try:
            # Update status to in progress
            with self._get_connection() as conn:
                conn.execute('''
                    UPDATE migration_executions 
                    SET status = ?, started_at = CURRENT_TIMESTAMP
                    WHERE plugin_name = ? AND phase = ?
                ''', (MigrationStatus.IN_PROGRESS.value, plugin_name, phase.value))
            
            # Execute phase
            start_time = datetime.now()
            with self._get_connection() as conn:
                success = phase_func(conn)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Update execution log
            with self._get_connection() as conn:
                if success:
                    conn.execute('''
                        UPDATE migration_executions 
                        SET status = ?, completed_at = CURRENT_TIMESTAMP, duration_seconds = ?
                        WHERE plugin_name = ? AND phase = ?
                    ''', (
                        MigrationStatus.COMPLETED.value, 
                        duration, 
                        plugin_name, 
                        phase.value
                    ))
                else:
                    conn.execute('''
                        UPDATE migration_executions 
                        SET status = ?, completed_at = CURRENT_TIMESTAMP, duration_seconds = ?, error_message = ?
                        WHERE plugin_name = ? AND phase = ?
                    ''', (
                        MigrationStatus.FAILED.value, 
                        duration, 
                        "Phase execution failed", 
                        plugin_name, 
                        phase.value
                    ))
            
            conn.commit()
            
            if success:
                self.logger.info(f"Migration phase {phase.value} for {plugin_name} completed successfully")
            else:
                self.logger.error(f"Migration phase {phase.value} for {plugin_name} failed")
            
            return success
            
        except Exception as e:
            # Update execution log with error
            try:
                with self._get_connection() as conn:
                    conn.execute('''
                        UPDATE migration_executions 
                        SET status = ?, completed_at = CURRENT_TIMESTAMP, error_message = ?
                        WHERE plugin_name = ? AND phase = ?
                    ''', (
                        MigrationStatus.FAILED.value, 
                        str(e), 
                        plugin_name, 
                        phase.value
                    ))
                    conn.commit()
            except:
                pass
            
            self.logger.error(f"Migration phase {phase.value} for {plugin_name} failed with exception: {str(e)}")
            return False
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered plugin.
        
        Args:
            plugin_name: Name of plugin
            
        Returns:
            Plugin information or None if not found
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT plugin_name, version, description, author, created_at, is_active
                    FROM migration_registry
                    WHERE plugin_name = ?
                ''', (plugin_name,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        "name": row[0],
                        "version": row[1],
                        "description": row[2],
                        "author": row[3],
                        "created_at": row[4],
                        "is_active": bool(row[5])
                    }
                
                return None
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get plugin info for {plugin_name}: {str(e)}")
            return None
    
    def get_migration_history(self, plugin_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get migration execution history.
        
        Args:
            plugin_name: Optional filter by plugin name
            
        Returns:
            List of execution records
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if plugin_name:
                    cursor.execute('''
                        SELECT plugin_name, phase, started_at, completed_at, status, error_message, duration_seconds
                        FROM migration_executions
                        WHERE plugin_name = ?
                        ORDER BY started_at DESC
                    ''', (plugin_name,))
                else:
                    cursor.execute('''
                        SELECT plugin_name, phase, started_at, completed_at, status, error_message, duration_seconds
                        FROM migration_executions
                        ORDER BY started_at DESC
                    ''')
                
                rows = cursor.fetchall()
                return [
                    {
                        "plugin_name": row[0],
                        "phase": row[1],
                        "started_at": row[2],
                        "completed_at": row[3],
                        "status": row[4],
                        "error_message": row[5],
                        "duration_seconds": row[6]
                    }
                    for row in rows
                ]
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get migration history: {str(e)}")
            return []
    
    def add_dependency(self, plugin_name: str, dependency_name: str) -> bool:
        """
        Add a dependency between plugins.
        
        Args:
            plugin_name: Name of dependent plugin
            dependency_name: Name of dependency plugin
            
        Returns:
            Boolean - True if dependency added, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO migration_dependencies 
                    (plugin_name, dependency_name)
                    VALUES (?, ?)
                ''', (plugin_name, dependency_name))
                
                conn.commit()
                self.logger.info(f"Added dependency: {plugin_name} -> {dependency_name}")
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to add dependency {plugin_name} -> {dependency_name}: {str(e)}")
            return False
    
    def remove_dependency(self, plugin_name: str, dependency_name: str) -> bool:
        """
        Remove a dependency between plugins.
        
        Args:
            plugin_name: Name of dependent plugin
            dependency_name: Name of dependency plugin
            
        Returns:
            Boolean - True if dependency removed, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    DELETE FROM migration_dependencies 
                    WHERE plugin_name = ? AND dependency_name = ?
                ''', (plugin_name, dependency_name))
                
                conn.commit()
                self.logger.info(f"Removed dependency: {plugin_name} -> {dependency_name}")
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to remove dependency {plugin_name} -> {dependency_name}: {str(e)}")
            return False

# Example migration plugin implementation
class V1ToV2Migration(MigrationPlugin):
    """
    Example migration plugin for v1 to v2 migration.
    """
    
    def __init__(self):
        super().__init__(
            name="v1_to_v2_migration",
            version="1.0.0",
            description="Migrate database schema from version 1 to 2"
        )
    
    def setup(self, connection: sqlite3.Connection) -> bool:
        """
        Setup phase - create backup and prepare for migration.
        """
        self.logger.info("Setting up v1 to v2 migration")
        
        try:
            # Create backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{connection.database}.backup_{timestamp}"
            
            import shutil
            shutil.copy2(connection.database, backup_path)
            self.logger.info(f"Backup created: {backup_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Setup failed: {str(e)}")
            return False
    
    def execute(self, connection: sqlite3.Connection) -> bool:
        """
        Execute phase - perform the actual migration.
        """
        self.logger.info("Executing v1 to v2 migration")
        
        try:
            cursor = connection.cursor()
            
            # Add new columns
            columns_to_add = [
                ("ALTER TABLE sessions ADD COLUMN processing_time REAL", "sessions", "processing_time"),
                ("ALTER TABLE sessions ADD COLUMN model_version TEXT", "sessions", "model_version"),
                ("ALTER TABLE prompts ADD COLUMN generation_time REAL", "prompts", "generation_time"),
                ("ALTER TABLE entities ADD COLUMN semantic_label TEXT", "entities", "semantic_label")
            ]
            
            for alter_sql, table, column in columns_to_add:
                try:
                    cursor.execute(alter_sql)
                except sqlite3.OperationalError:
                    # Column might already exist
                    pass
            
            # Set default values
            cursor.execute('''
                UPDATE sessions 
                SET model_version = 'v1_default', processing_time = 0.0
                WHERE model_version IS NULL OR processing_time IS NULL
            ''')
            
            connection.commit()
            self.logger.info("Migration execution completed")
            return True
            
        except sqlite3.Error as e:
            self.logger.error(f"Execution failed: {str(e)}")
            return False
    
    def verify(self, connection: sqlite3.Connection) -> bool:
        """
        Verification phase - verify the migration was successful.
        """
        self.logger.info("Verifying v1 to v2 migration")
        
        try:
            cursor = connection.cursor()
            
            # Verify new columns exist
            required_columns = {
                "sessions": ["processing_time", "model_version"],
                "prompts": ["generation_time"],
                "entities": ["semantic_label"]
            }
            
            for table, columns in required_columns.items():
                for column in columns:
                    try:
                        cursor.execute(f'SELECT {column} FROM {table} LIMIT 1')
                    except sqlite3.OperationalError:
                        self.logger.error(f"Column {column} missing from {table}")
                        return False
            
            self.logger.info("Migration verification passed")
            return True
            
        except sqlite3.Error as e:
            self.logger.error(f"Verification failed: {str(e)}")
            return False
    
    def cleanup(self, connection: sqlite3.Connection) -> bool:
        """
        Cleanup phase - perform post-migration cleanup.
        """
        self.logger.info("Cleaning up after v1 to v2 migration")
        
        # In this example, no cleanup needed
        self.logger.info("Cleanup completed")
        return True

# Example usage
if __name__ == "__main__":
    # Initialize advanced migration framework
    framework = AdvancedStorageMigrations("edi_advanced.db")
    
    # Create and register example migration plugin
    v1_to_v2_plugin = V1ToV2Migration()
    framework.register_plugin(v1_to_v2_plugin)
    
    print("Advanced Storage Migrations Framework initialized")
    print(f"Registered plugins: {list(framework.plugins.keys())}")
    
    # Add dependencies if needed
    # framework.add_dependency("dependent_plugin", "dependency_plugin")
    
    # Run migration
    success = framework.run_migration_phases("v1_to_v2_migration")
    
    if success:
        print("Migration completed successfully!")
    else:
        print("Migration failed!")
    
    # Show migration history
    history = framework.get_migration_history("v1_to_v2_migration")
    print(f"\nMigration history ({len(history)} records):")
    for record in history:
        print(f"  {record['phase']}: {record['status']} ({record['duration_seconds']:.2f}s)")
    
    # Get plugin info
    plugin_info = framework.get_plugin_info("v1_to_v2_migration")
    if plugin_info:
        print(f"\nPlugin info: {plugin_info['name']} v{plugin_info['version']}")
        print(f"  Description: {plugin_info['description']}")
        print(f"  Author: {plugin_info['author']}")
        print(f"  Created: {plugin_info['created_at']}")
```