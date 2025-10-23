# Database.save_session()

[Back to Database](../storage_database.md)

## Related User Story
"As a user, I want EDI to remember my editing sessions so I can resume later or review my history." (from PRD - implied by session persistence requirements)

## Function Signature
`save_session()`

## Parameters
- None - Uses internal session data

## Returns
- None - Saves session data to database

## Step-by-step Logic
1. Prepare session data from the current session state
2. Begin a database transaction to ensure consistency
3. Insert or update the session record in the sessions table
4. Save prompt history to the prompts table
5. Save detected entities to the entities table
6. Save validation results to the validations table
7. Save user feedback to the user_feedback table if available
8. Commit the transaction to finalize the save
9. Handle any database errors and rollback if necessary

## Data Relationships
- Session record is the primary record linking all related data
- Multiple prompt records associated with one session
- Multiple entity records associated with one session
- Multiple validation records associated with one session
- Feedback optionally linked to a session

## Transaction Management
- Uses database transactions to maintain data integrity
- Rolls back all changes if any part of the save fails
- Ensures related records are saved together consistently
- Handles concurrent access safely

## Input/Output Data Structures
### Session Record
A record in the sessions table containing:
- ID (UUID)
- Creation timestamp
- Image path
- Naive prompt
- Session status
- Final alignment score

### Related Records
- Prompts: Positive and negative prompts by iteration
- Entities: Detected objects with properties and positions
- Validations: Alignment scores and change assessments
- User feedback: Ratings and comments

## See Docs

### Python Implementation Example
Implementation of the Database.save_session() method:

```python
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import logging

class DatabaseError(Exception):
    """Custom exception for database errors."""
    pass

class Database:
    """
    Database class for session persistence in EDI.
    """
    
    def __init__(self, db_path: str = "edi_sessions.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._initialize_database()
    
    def _initialize_database(self):
        """
        Initialize the database with required tables.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        image_path TEXT NOT NULL,
                        naive_prompt TEXT NOT NULL,
                        status TEXT CHECK(status IN ('in_progress', 'completed', 'failed')),
                        final_alignment_score REAL
                    )
                ''')
                
                # Create prompts table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS prompts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT REFERENCES sessions(id),
                        iteration INT,
                        positive_prompt TEXT,
                        negative_prompt TEXT,
                        quality_score REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create entities table
                cursor.execute('''
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
                ''')
                
                # Create validations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS validations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT REFERENCES sessions(id),
                        attempt_number INT,
                        alignment_score REAL,
                        preserved_count INT,
                        modified_count INT,
                        unintended_count INT,
                        user_feedback TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create user feedback table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT REFERENCES sessions(id),
                        feedback_type TEXT CHECK(feedback_type IN ('accept', 'reject', 'partial')),
                        comments TEXT,
                        rating INT CHECK(rating BETWEEN 1 AND 5),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to initialize database: {str(e)}")
    
    def save_session(self, session_data: Dict[str, Any]) -> None:
        """
        Save session data to the database.
        
        Args:
            session_data: Dictionary containing session information to save
            
        Raises:
            DatabaseError: If saving fails
        """
        # Prepare session data from the current session state
        session_id = session_data.get("id", str(uuid.uuid4()))
        created_at = session_data.get("created_at", datetime.now().isoformat())
        image_path = session_data.get("image_path", "")
        naive_prompt = session_data.get("naive_prompt", "")
        status = session_data.get("status", "in_progress")
        final_alignment_score = session_data.get("final_alignment_score")
        
        # Validate required fields
        if not image_path or not naive_prompt:
            raise DatabaseError("Image path and naive prompt are required")
        
        # Begin a database transaction to ensure consistency
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert or update the session record in the sessions table
                cursor.execute('''
                    INSERT OR REPLACE INTO sessions 
                    (id, created_at, image_path, naive_prompt, status, final_alignment_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    session_id,
                    created_at,
                    image_path,
                    naive_prompt,
                    status,
                    final_alignment_score
                ))
                
                # Save prompt history to the prompts table
                prompts = session_data.get("prompts", [])
                for prompt_record in prompts:
                    cursor.execute('''
                        INSERT INTO prompts 
                        (session_id, iteration, positive_prompt, negative_prompt, quality_score)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        session_id,
                        prompt_record.get("iteration", 0),
                        prompt_record.get("positive_prompt", ""),
                        prompt_record.get("negative_prompt", ""),
                        prompt_record.get("quality_score")
                    ))
                
                # Save detected entities to the entities table
                entities = session_data.get("entities", [])
                for entity_record in entities:
                    # Serialize bbox to JSON
                    bbox_json = json.dumps(entity_record.get("bbox", {}))
                    
                    cursor.execute('''
                        INSERT INTO entities 
                        (session_id, entity_id, label, confidence, bbox_json, mask_path, color_hex, area_percent)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        session_id,
                        entity_record.get("entity_id", ""),
                        entity_record.get("label", ""),
                        entity_record.get("confidence", 0.0),
                        bbox_json,
                        entity_record.get("mask_path"),
                        entity_record.get("color_hex"),
                        entity_record.get("area_percent")
                    ))
                
                # Save validation results to the validations table
                validations = session_data.get("validations", [])
                for validation_record in validations:
                    cursor.execute('''
                        INSERT INTO validations 
                        (session_id, attempt_number, alignment_score, preserved_count, modified_count, unintended_count, user_feedback)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        session_id,
                        validation_record.get("attempt_number", 0),
                        validation_record.get("alignment_score", 0.0),
                        validation_record.get("preserved_count", 0),
                        validation_record.get("modified_count", 0),
                        validation_record.get("unintended_count", 0),
                        validation_record.get("user_feedback")
                    ))
                
                # Save user feedback to the user_feedback table if available
                user_feedback = session_data.get("user_feedback")
                if user_feedback:
                    cursor.execute('''
                        INSERT INTO user_feedback 
                        (session_id, feedback_type, comments, rating)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        session_id,
                        user_feedback.get("feedback_type", "partial"),
                        user_feedback.get("comments"),
                        user_feedback.get("rating")
                    ))
                
                # Commit the transaction to finalize the save
                conn.commit()
                
                self.logger.info(f"Session {session_id} saved successfully")
                
        except sqlite3.Error as e:
            # Handle any database errors and rollback if necessary
            self.logger.error(f"Failed to save session {session_id}: {str(e)}")
            raise DatabaseError(f"Failed to save session: {str(e)}")
    
    def save_session_with_batch(self, session_data: Dict[str, Any], batch_size: int = 100) -> None:
        """
        Save session data with batch processing for large datasets.
        
        Args:
            session_data: Dictionary containing session information to save
            batch_size: Number of records to insert in each batch
        """
        session_id = session_data.get("id", str(uuid.uuid4()))
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Save main session record
                cursor.execute('''
                    INSERT OR REPLACE INTO sessions 
                    (id, created_at, image_path, naive_prompt, status, final_alignment_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    session_id,
                    session_data.get("created_at", datetime.now().isoformat()),
                    session_data.get("image_path", ""),
                    session_data.get("naive_prompt", ""),
                    session_data.get("status", "in_progress"),
                    session_data.get("final_alignment_score")
                ))
                
                # Batch save prompts
                prompts = session_data.get("prompts", [])
                for i in range(0, len(prompts), batch_size):
                    batch = prompts[i:i + batch_size]
                    cursor.executemany('''
                        INSERT INTO prompts 
                        (session_id, iteration, positive_prompt, negative_prompt, quality_score)
                        VALUES (?, ?, ?, ?, ?)
                    ''', [
                        (
                            session_id,
                            prompt.get("iteration", 0),
                            prompt.get("positive_prompt", ""),
                            prompt.get("negative_prompt", ""),
                            prompt.get("quality_score")
                        ) for prompt in batch
                    ])
                
                # Batch save entities
                entities = session_data.get("entities", [])
                for i in range(0, len(entities), batch_size):
                    batch = entities[i:i + batch_size]
                    cursor.executemany('''
                        INSERT INTO entities 
                        (session_id, entity_id, label, confidence, bbox_json, mask_path, color_hex, area_percent)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', [
                        (
                            session_id,
                            entity.get("entity_id", ""),
                            entity.get("label", ""),
                            entity.get("confidence", 0.0),
                            json.dumps(entity.get("bbox", {})),
                            entity.get("mask_path"),
                            entity.get("color_hex"),
                            entity.get("area_percent")
                        ) for entity in batch
                    ])
                
                # Batch save validations
                validations = session_data.get("validations", [])
                for i in range(0, len(validations), batch_size):
                    batch = validations[i:i + batch_size]
                    cursor.executemany('''
                        INSERT INTO validations 
                        (session_id, attempt_number, alignment_score, preserved_count, modified_count, unintended_count, user_feedback)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', [
                        (
                            session_id,
                            validation.get("attempt_number", 0),
                            validation.get("alignment_score", 0.0),
                            validation.get("preserved_count", 0),
                            validation.get("modified_count", 0),
                            validation.get("unintended_count", 0),
                            validation.get("user_feedback")
                        ) for validation in batch
                    ])
                
                # Save user feedback
                user_feedback = session_data.get("user_feedback")
                if user_feedback:
                    cursor.execute('''
                        INSERT INTO user_feedback 
                        (session_id, feedback_type, comments, rating)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        session_id,
                        user_feedback.get("feedback_type", "partial"),
                        user_feedback.get("comments"),
                        user_feedback.get("rating")
                    ))
                
                conn.commit()
                self.logger.info(f"Session {session_id} saved with batch processing")
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to save session {session_id} with batch processing: {str(e)}")
            raise DatabaseError(f"Failed to save session: {str(e)}")
    
    def save_session_async(self, session_data: Dict[str, Any]) -> str:
        """
        Asynchronously save session data and return session ID for tracking.
        
        Args:
            session_data: Dictionary containing session information to save
            
        Returns:
            Session ID for tracking the save operation
        """
        import threading
        import queue
        
        # Generate session ID
        session_id = session_data.get("id", str(uuid.uuid4()))
        
        # Create a queue for result
        result_queue = queue.Queue()
        
        def save_worker():
            try:
                self.save_session(session_data)
                result_queue.put(("success", session_id, None))
            except Exception as e:
                result_queue.put(("error", session_id, str(e)))
        
        # Start background thread
        thread = threading.Thread(target=save_worker)
        thread.daemon = True
        thread.start()
        
        # Return session ID immediately
        return session_id
    
    def get_save_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get the status of an ongoing save operation.
        
        Args:
            session_id: ID of the session being saved
            
        Returns:
            Dictionary with save status information
        """
        # In a real implementation, this would check a status tracking system
        # For this example, we'll just check if the session exists in the database
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
                result = cursor.fetchone()
                
                if result:
                    return {"status": "completed", "session_id": session_id}
                else:
                    return {"status": "in_progress", "session_id": session_id}
        except sqlite3.Error:
            return {"status": "error", "session_id": session_id}

# Example usage
if __name__ == "__main__":
    # Initialize database
    db = Database()
    
    # Example session data
    example_session = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now().isoformat(),
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
            },
            {
                "iteration": 1,
                "positive_prompt": "stormy sky with lightning",
                "negative_prompt": "sunny sky, clear weather, no clouds",
                "quality_score": 0.88
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
    
    try:
        # Save session
        db.save_session(example_session)
        print(f"Session saved successfully")
        
        # Save with batch processing
        db.save_session_with_batch(example_session)
        print("Session saved with batch processing")
        
        # Async save
        session_id = db.save_session_async(example_session)
        print(f"Async save started for session: {session_id}")
        
        # Check status
        status = db.get_save_status(session_id)
        print(f"Save status: {status}")
        
    except DatabaseError as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")
```

### Advanced Database Implementation
Enhanced database operations with connection pooling and performance optimizations:

```python
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Dict, Any, List, Optional, Generator
import logging
from dataclasses import dataclass
from enum import Enum

class SessionStatus(Enum):
    """Enumeration for session statuses."""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class DatabaseConfig:
    """Configuration for database operations."""
    db_path: str = "edi_sessions.db"
    timeout: int = 30
    max_connections: int = 10
    journal_mode: str = "WAL"  # Write-Ahead Logging for better concurrency
    synchronous: str = "NORMAL"  # Balance between safety and performance

class AdvancedDatabase:
    """
    Advanced database with connection pooling and performance optimizations.
    """
    
    def __init__(self, config: DatabaseConfig = None):
        self.config = config or DatabaseConfig()
        self.logger = logging.getLogger(__name__)
        self._connections = []
        self._connection_lock = threading.Lock()
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database with optimized settings."""
        try:
            with self._get_connection() as conn:
                # Set performance optimizations
                conn.execute(f"PRAGMA journal_mode = {self.config.journal_mode}")
                conn.execute(f"PRAGMA synchronous = {self.config.synchronous}")
                conn.execute("PRAGMA cache_size = 10000")  # 10MB cache
                conn.execute("PRAGMA temp_store = MEMORY")
                
                # Create tables with indexes for performance
                self._create_tables(conn)
                
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to initialize database: {str(e)}")
    
    def _create_tables(self, conn: sqlite3.Connection):
        """Create database tables with indexes."""
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT NOT NULL,
                naive_prompt TEXT NOT NULL,
                status TEXT CHECK(status IN ('in_progress', 'completed', 'failed')),
                final_alignment_score REAL,
                INDEX idx_sessions_created_at (created_at),
                INDEX idx_sessions_status (status)
            )
        ''')
        
        # Prompts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT REFERENCES sessions(id),
                iteration INT,
                positive_prompt TEXT,
                negative_prompt TEXT,
                quality_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_prompts_session_id (session_id),
                INDEX idx_prompts_iteration (iteration)
            )
        ''')
        
        # Entities table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT REFERENCES sessions(id),
                entity_id TEXT,
                label TEXT,
                confidence REAL,
                bbox_json TEXT,
                mask_path TEXT,
                color_hex TEXT,
                area_percent REAL,
                INDEX idx_entities_session_id (session_id),
                INDEX idx_entities_label (label)
            )
        ''')
        
        # Validations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS validations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT REFERENCES sessions(id),
                attempt_number INT,
                alignment_score REAL,
                preserved_count INT,
                modified_count INT,
                unintended_count INT,
                user_feedback TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_validations_session_id (session_id),
                INDEX idx_validations_alignment_score (alignment_score)
            )
        ''')
        
        # User feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT REFERENCES sessions(id),
                feedback_type TEXT CHECK(feedback_type IN ('accept', 'reject', 'partial')),
                comments TEXT,
                rating INT CHECK(rating BETWEEN 1 AND 5),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_feedback_session_id (session_id),
                INDEX idx_feedback_rating (rating)
            )
        ''')
        
        conn.commit()
    
    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection from the pool."""
        conn = None
        try:
            # Try to get an existing connection
            with self._connection_lock:
                if self._connections:
                    conn = self._connections.pop()
            
            # If no connection available, create a new one
            if conn is None:
                conn = sqlite3.connect(
                    self.config.db_path,
                    timeout=self.config.timeout,
                    check_same_thread=False
                )
                conn.row_factory = sqlite3.Row  # Enable column access by name
            
            yield conn
            
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            raise DatabaseError(f"Database connection error: {str(e)}")
        
        finally:
            if conn:
                try:
                    conn.commit()
                except:
                    pass
                
                # Return connection to pool if pool not full
                with self._connection_lock:
                    if len(self._connections) < self.config.max_connections:
                        self._connections.append(conn)
                    else:
                        conn.close()
    
    def save_session_optimized(self, session_data: Dict[str, Any]) -> None:
        """
        Optimized session saving with prepared statements and batch operations.
        """
        session_id = session_data.get("id", str(uuid.uuid4()))
        
        try:
            with self._get_connection() as conn:
                # Disable autocommit for performance
                conn.execute("BEGIN IMMEDIATE")
                
                # Prepare statements for performance
                session_insert = '''
                    INSERT OR REPLACE INTO sessions 
                    (id, created_at, image_path, naive_prompt, status, final_alignment_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                '''
                
                prompt_insert = '''
                    INSERT INTO prompts 
                    (session_id, iteration, positive_prompt, negative_prompt, quality_score)
                    VALUES (?, ?, ?, ?, ?)
                '''
                
                entity_insert = '''
                    INSERT INTO entities 
                    (session_id, entity_id, label, confidence, bbox_json, mask_path, color_hex, area_percent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                '''
                
                validation_insert = '''
                    INSERT INTO validations 
                    (session_id, attempt_number, alignment_score, preserved_count, modified_count, unintended_count, user_feedback)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                '''
                
                feedback_insert = '''
                    INSERT INTO user_feedback 
                    (session_id, feedback_type, comments, rating)
                    VALUES (?, ?, ?, ?)
                '''
                
                # Execute session insert
                conn.execute(session_insert, (
                    session_id,
                    session_data.get("created_at", datetime.now().isoformat()),
                    session_data.get("image_path", ""),
                    session_data.get("naive_prompt", ""),
                    session_data.get("status", "in_progress"),
                    session_data.get("final_alignment_score")
                ))
                
                # Batch insert prompts
                prompts = session_data.get("prompts", [])
                if prompts:
                    conn.executemany(prompt_insert, [
                        (
                            session_id,
                            prompt.get("iteration", 0),
                            prompt.get("positive_prompt", ""),
                            prompt.get("negative_prompt", ""),
                            prompt.get("quality_score")
                        ) for prompt in prompts
                    ])
                
                # Batch insert entities
                entities = session_data.get("entities", [])
                if entities:
                    conn.executemany(entity_insert, [
                        (
                            session_id,
                            entity.get("entity_id", ""),
                            entity.get("label", ""),
                            entity.get("confidence", 0.0),
                            json.dumps(entity.get("bbox", {})),
                            entity.get("mask_path"),
                            entity.get("color_hex"),
                            entity.get("area_percent")
                        ) for entity in entities
                    ])
                
                # Batch insert validations
                validations = session_data.get("validations", [])
                if validations:
                    conn.executemany(validation_insert, [
                        (
                            session_id,
                            validation.get("attempt_number", 0),
                            validation.get("alignment_score", 0.0),
                            validation.get("preserved_count", 0),
                            validation.get("modified_count", 0),
                            validation.get("unintended_count", 0),
                            validation.get("user_feedback")
                        ) for validation in validations
                    ])
                
                # Insert user feedback
                user_feedback = session_data.get("user_feedback")
                if user_feedback:
                    conn.execute(feedback_insert, (
                        session_id,
                        user_feedback.get("feedback_type", "partial"),
                        user_feedback.get("comments"),
                        user_feedback.get("rating")
                    ))
                
                # Commit transaction
                conn.execute("COMMIT")
                
                self.logger.info(f"Session {session_id} saved successfully with optimizations")
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to save session {session_id}: {str(e)}")
            raise DatabaseError(f"Failed to save session: {str(e)}")
    
    def bulk_save_sessions(self, sessions_data: List[Dict[str, Any]]) -> List[str]:
        """
        Save multiple sessions in a single optimized operation.
        
        Args:
            sessions_data: List of session data dictionaries
            
        Returns:
            List of session IDs that were saved
        """
        session_ids = []
        
        try:
            with self._get_connection() as conn:
                conn.execute("BEGIN IMMEDIATE")
                
                for session_data in sessions_data:
                    session_id = session_data.get("id", str(uuid.uuid4()))
                    session_ids.append(session_id)
                    
                    # Save session data (same as single save but in batch)
                    self._save_single_session_internal(conn, session_data, session_id)
                
                conn.execute("COMMIT")
                self.logger.info(f"Bulk saved {len(session_ids)} sessions")
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to bulk save sessions: {str(e)}")
            raise DatabaseError(f"Failed to bulk save sessions: {str(e)}")
        
        return session_ids
    
    def _save_single_session_internal(self, 
                                     conn: sqlite3.Connection, 
                                     session_data: Dict[str, Any], 
                                     session_id: str):
        """
        Internal method to save a single session within a transaction.
        """
        # Insert session
        conn.execute('''
            INSERT OR REPLACE INTO sessions 
            (id, created_at, image_path, naive_prompt, status, final_alignment_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            session_data.get("created_at", datetime.now().isoformat()),
            session_data.get("image_path", ""),
            session_data.get("naive_prompt", ""),
            session_data.get("status", "in_progress"),
            session_data.get("final_alignment_score")
        ))
        
        # Insert related data
        # ... (similar to save_session_optimized)

# Example usage
if __name__ == "__main__":
    # Initialize advanced database
    config = DatabaseConfig(
        db_path="edi_advanced.db",
        max_connections=20
    )
    advanced_db = AdvancedDatabase(config)
    
    print("Advanced Database initialized")
    
    # Example session data would be saved similar to previous example
    print("Ready to save sessions with advanced optimizations")
```