# Database.load_session()

[Back to Database](../storage_database.md)

## Related User Story
"As a user, I want EDI to resume my editing sessions where I left off." (from PRD - implied by session persistence requirements)

## Function Signature
`load_session()`

## Parameters
- None - Uses internal session identifier

## Returns
- None - Loads session data from database into memory

## Step-by-step Logic
1. Use session ID to query the sessions table
2. Begin a database transaction for consistent reads
3. Load the main session record from the sessions table
4. Load associated prompt history from the prompts table
5. Load associated entity detections from the entities table
6. Load associated validation results from the validations table
7. Load associated user feedback from the user_feedback table if available
8. Reconstruct the complete session state in memory
9. Commit the transaction and handle any errors

## Data Relationships
- Loads the primary session record first
- Loads all associated prompt iterations
- Loads all detected entities with their properties
- Loads all validation attempts and scores
- Optionally loads user feedback if it exists

## Reconstruction Process
- Rebuilds the session state from database records
- Maintains relationships between different data types
- Validates that all required information is present
- Provides default values for any missing optional data

## Input/Output Data Structures
### Session Record
A record from the sessions table containing:
- ID (UUID)
- Creation timestamp
- Image path
- Naive prompt
- Session status
- Final alignment score

### Reconstructed Session
An in-memory object containing:
- All session metadata
- Complete prompt history with iterations
- All detected entities with properties
- All validation attempts with scores
- User feedback if available

## See Docs

### Python Implementation Example
Implementation of the Database.load_session() method:

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
    
    def load_session(self, session_id: str) -> Dict[str, Any]:
        """
        Load session data from the database.
        
        Args:
            session_id: UUID of the session to load
            
        Returns:
            Dictionary containing reconstructed session data
            
        Raises:
            DatabaseError: If loading fails or session not found
        """
        # Use session ID to query the sessions table
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Begin a database transaction for consistent reads
                conn.execute("BEGIN")
                
                # Load the main session record from the sessions table
                cursor.execute('''
                    SELECT id, created_at, image_path, naive_prompt, status, final_alignment_score
                    FROM sessions
                    WHERE id = ?
                ''', (session_id,))
                
                session_row = cursor.fetchone()
                if not session_row:
                    raise DatabaseError(f"Session {session_id} not found")
                
                # Extract session data
                session_data = {
                    "id": session_row[0],
                    "created_at": session_row[1],
                    "image_path": session_row[2],
                    "naive_prompt": session_row[3],
                    "status": session_row[4],
                    "final_alignment_score": session_row[5]
                }
                
                # Load associated prompt history from the prompts table
                cursor.execute('''
                    SELECT iteration, positive_prompt, negative_prompt, quality_score, created_at
                    FROM prompts
                    WHERE session_id = ?
                    ORDER BY iteration
                ''', (session_id,))
                
                prompt_rows = cursor.fetchall()
                session_data["prompts"] = [
                    {
                        "iteration": row[0],
                        "positive_prompt": row[1],
                        "negative_prompt": row[2],
                        "quality_score": row[3],
                        "created_at": row[4]
                    }
                    for row in prompt_rows
                ]
                
                # Load associated entity detections from the entities table
                cursor.execute('''
                    SELECT entity_id, label, confidence, bbox_json, mask_path, color_hex, area_percent
                    FROM entities
                    WHERE session_id = ?
                ''', (session_id,))
                
                entity_rows = cursor.fetchall()
                session_data["entities"] = []
                
                for row in entity_rows:
                    try:
                        bbox = json.loads(row[3]) if row[3] else {}
                    except json.JSONDecodeError:
                        bbox = {}
                    
                    session_data["entities"].append({
                        "entity_id": row[0],
                        "label": row[1],
                        "confidence": row[2],
                        "bbox": bbox,
                        "mask_path": row[4],
                        "color_hex": row[5],
                        "area_percent": row[6]
                    })
                
                # Load associated validation results from the validations table
                cursor.execute('''
                    SELECT attempt_number, alignment_score, preserved_count, modified_count, 
                           unintended_count, user_feedback, created_at
                    FROM validations
                    WHERE session_id = ?
                    ORDER BY attempt_number
                ''', (session_id,))
                
                validation_rows = cursor.fetchall()
                session_data["validations"] = [
                    {
                        "attempt_number": row[0],
                        "alignment_score": row[1],
                        "preserved_count": row[2],
                        "modified_count": row[3],
                        "unintended_count": row[4],
                        "user_feedback": row[5],
                        "created_at": row[6]
                    }
                    for row in validation_rows
                ]
                
                # Load associated user feedback from the user_feedback table if available
                cursor.execute('''
                    SELECT feedback_type, comments, rating, created_at
                    FROM user_feedback
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                ''', (session_id,))
                
                feedback_row = cursor.fetchone()
                if feedback_row:
                    session_data["user_feedback"] = {
                        "feedback_type": feedback_row[0],
                        "comments": feedback_row[1],
                        "rating": feedback_row[2],
                        "created_at": feedback_row[3]
                    }
                
                # Reconstruct the complete session state in memory
                # This is already done above by organizing data into session_data
                
                # Commit the transaction and handle any errors
                conn.execute("COMMIT")
                
                self.logger.info(f"Session {session_id} loaded successfully")
                return session_data
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to load session {session_id}: {str(e)}")
            raise DatabaseError(f"Failed to load session: {str(e)}")
    
    def load_session_partial(self, 
                            session_id: str,
                            load_prompts: bool = True,
                            load_entities: bool = True,
                            load_validations: bool = True,
                            load_feedback: bool = True) -> Dict[str, Any]:
        """
        Load session data with selective loading options.
        
        Args:
            session_id: UUID of the session to load
            load_prompts: Whether to load prompt history
            load_entities: Whether to load entity detections
            load_validations: Whether to load validation results
            load_feedback: Whether to load user feedback
            
        Returns:
            Dictionary containing reconstructed session data with selected components
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Load main session record
                cursor.execute('''
                    SELECT id, created_at, image_path, naive_prompt, status, final_alignment_score
                    FROM sessions
                    WHERE id = ?
                ''', (session_id,))
                
                session_row = cursor.fetchone()
                if not session_row:
                    raise DatabaseError(f"Session {session_id} not found")
                
                # Extract session data
                session_data = {
                    "id": session_row[0],
                    "created_at": session_row[1],
                    "image_path": session_row[2],
                    "naive_prompt": session_row[3],
                    "status": session_row[4],
                    "final_alignment_score": session_row[5]
                }
                
                # Conditionally load prompt history
                if load_prompts:
                    cursor.execute('''
                        SELECT iteration, positive_prompt, negative_prompt, quality_score, created_at
                        FROM prompts
                        WHERE session_id = ?
                        ORDER BY iteration
                    ''', (session_id,))
                    
                    prompt_rows = cursor.fetchall()
                    session_data["prompts"] = [
                        {
                            "iteration": row[0],
                            "positive_prompt": row[1],
                            "negative_prompt": row[2],
                            "quality_score": row[3],
                            "created_at": row[4]
                        }
                        for row in prompt_rows
                    ]
                
                # Conditionally load entities
                if load_entities:
                    cursor.execute('''
                        SELECT entity_id, label, confidence, bbox_json, mask_path, color_hex, area_percent
                        FROM entities
                        WHERE session_id = ?
                    ''', (session_id,))
                    
                    entity_rows = cursor.fetchall()
                    session_data["entities"] = []
                    
                    for row in entity_rows:
                        try:
                            bbox = json.loads(row[3]) if row[3] else {}
                        except json.JSONDecodeError:
                            bbox = {}
                        
                        session_data["entities"].append({
                            "entity_id": row[0],
                            "label": row[1],
                            "confidence": row[2],
                            "bbox": bbox,
                            "mask_path": row[4],
                            "color_hex": row[5],
                            "area_percent": row[6]
                        })
                
                # Conditionally load validations
                if load_validations:
                    cursor.execute('''
                        SELECT attempt_number, alignment_score, preserved_count, modified_count, 
                               unintended_count, user_feedback, created_at
                        FROM validations
                        WHERE session_id = ?
                        ORDER BY attempt_number
                    ''', (session_id,))
                    
                    validation_rows = cursor.fetchall()
                    session_data["validations"] = [
                        {
                            "attempt_number": row[0],
                            "alignment_score": row[1],
                            "preserved_count": row[2],
                            "modified_count": row[3],
                            "unintended_count": row[4],
                            "user_feedback": row[5],
                            "created_at": row[6]
                        }
                        for row in validation_rows
                    ]
                
                # Conditionally load feedback
                if load_feedback:
                    cursor.execute('''
                        SELECT feedback_type, comments, rating, created_at
                        FROM user_feedback
                        WHERE session_id = ?
                        ORDER BY created_at DESC
                        LIMIT 1
                    ''', (session_id,))
                    
                    feedback_row = cursor.fetchone()
                    if feedback_row:
                        session_data["user_feedback"] = {
                            "feedback_type": feedback_row[0],
                            "comments": feedback_row[1],
                            "rating": feedback_row[2],
                            "created_at": feedback_row[3]
                        }
                
                self.logger.info(f"Session {session_id} loaded partially")
                return session_data
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to partially load session {session_id}: {str(e)}")
            raise DatabaseError(f"Failed to load session: {str(e)}")
    
    def load_latest_session(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recently created session.
        
        Returns:
            Dictionary containing the latest session data, or None if no sessions exist
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Find the latest session
                cursor.execute('''
                    SELECT id
                    FROM sessions
                    ORDER BY created_at DESC
                    LIMIT 1
                ''')
                
                latest_row = cursor.fetchone()
                if not latest_row:
                    return None
                
                latest_session_id = latest_row[0]
                return self.load_session(latest_session_id)
                
        except Exception as e:
            self.logger.error(f"Failed to load latest session: {str(e)}")
            return None
    
    def load_sessions_by_status(self, status: str) -> List[Dict[str, Any]]:
        """
        Load all sessions with a specific status.
        
        Args:
            status: Session status to filter by
            
        Returns:
            List of session dictionaries with the specified status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Find sessions with the specified status
                cursor.execute('''
                    SELECT id
                    FROM sessions
                    WHERE status = ?
                    ORDER BY created_at DESC
                ''', (status,))
                
                session_rows = cursor.fetchall()
                sessions = []
                
                for row in session_rows:
                    session_id = row[0]
                    try:
                        session_data = self.load_session(session_id)
                        sessions.append(session_data)
                    except DatabaseError:
                        # Skip sessions that can't be loaded
                        self.logger.warning(f"Failed to load session {session_id}")
                        continue
                
                return sessions
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to load sessions by status {status}: {str(e)}")
            raise DatabaseError(f"Failed to load sessions: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Initialize database
    db = Database()
    
    # Example: Load a specific session
    try:
        # This would require an existing session ID
        # session_data = db.load_session("existing-session-uuid-here")
        # print(f"Loaded session: {session_data['id']}")
        # print(f"Image: {session_data['image_path']}")
        # print(f"Prompt: {session_data['naive_prompt']}")
        # print(f"Prompts loaded: {len(session_data.get('prompts', []))}")
        # print(f"Entities loaded: {len(session_data.get('entities', []))}")
        
        print("Database initialized for session loading")
        print("To load a session, call: db.load_session('session-uuid')")
        
    except DatabaseError as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")
```

### Advanced Session Loading Implementation
Enhanced session loading with caching and performance optimizations:

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
    cache_size: int = 100  # Number of sessions to cache

class AdvancedDatabase:
    """
    Advanced database with connection pooling, caching, and performance optimizations.
    """
    
    def __init__(self, config: DatabaseConfig = None):
        self.config = config or DatabaseConfig()
        self.logger = logging.getLogger(__name__)
        self._connections = []
        self._connection_lock = threading.Lock()
        self._session_cache = {}  # In-memory cache for loaded sessions
        self._cache_lock = threading.Lock()
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
    
    def load_session_cached(self, session_id: str, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load session with caching for improved performance.
        
        Args:
            session_id: UUID of the session to load
            force_reload: Whether to bypass cache and force reload from database
            
        Returns:
            Dictionary containing reconstructed session data
        """
        # Check cache first
        with self._cache_lock:
            if not force_reload and session_id in self._session_cache:
                cached_session = self._session_cache[session_id]
                # Check if cache is still valid (e.g., not too old)
                if time.time() - cached_session.get('_cached_at', 0) < 300:  # 5 minutes
                    self.logger.info(f"Loading session {session_id} from cache")
                    return cached_session
        
        # Load from database
        session_data = self.load_session(session_id)
        
        # Cache the result
        with self._cache_lock:
            if len(self._session_cache) >= self.config.cache_size:
                # Remove oldest entry if cache is full
                oldest_key = min(self._session_cache.keys(), 
                               key=lambda k: self._session_cache[k].get('_cached_at', 0))
                del self._session_cache[oldest_key]
            
            session_data['_cached_at'] = time.time()
            self._session_cache[session_id] = session_data
        
        return session_data
    
    def invalidate_session_cache(self, session_id: Optional[str] = None):
        """
        Invalidate session cache, either for a specific session or all sessions.
        
        Args:
            session_id: Optional UUID of specific session to invalidate, or None to clear all
        """
        with self._cache_lock:
            if session_id:
                if session_id in self._session_cache:
                    del self._session_cache[session_id]
                    self.logger.info(f"Invalidated cache for session {session_id}")
            else:
                self._session_cache.clear()
                self.logger.info("Cleared entire session cache")
    
    def load_session_optimized(self, session_id: str) -> Dict[str, Any]:
        """
        Optimized session loading with prepared statements and efficient queries.
        """
        try:
            with self._get_connection() as conn:
                # Use prepared statements for performance
                conn.execute("BEGIN")
                
                # Load session record
                session_query = '''
                    SELECT id, created_at, image_path, naive_prompt, status, final_alignment_score
                    FROM sessions
                    WHERE id = ?
                '''
                session_cursor = conn.execute(session_query, (session_id,))
                session_row = session_cursor.fetchone()
                
                if not session_row:
                    raise DatabaseError(f"Session {session_id} not found")
                
                # Extract session data
                session_data = {
                    "id": session_row[0],
                    "created_at": session_row[1],
                    "image_path": session_row[2],
                    "naive_prompt": session_row[3],
                    "status": session_row[4],
                    "final_alignment_score": session_row[5]
                }
                
                # Load all related data in a single transaction
                # Load prompts
                prompt_query = '''
                    SELECT iteration, positive_prompt, negative_prompt, quality_score, created_at
                    FROM prompts
                    WHERE session_id = ?
                    ORDER BY iteration
                '''
                prompt_cursor = conn.execute(prompt_query, (session_id,))
                prompt_rows = prompt_cursor.fetchall()
                
                session_data["prompts"] = [
                    {
                        "iteration": row[0],
                        "positive_prompt": row[1],
                        "negative_prompt": row[2],
                        "quality_score": row[3],
                        "created_at": row[4]
                    }
                    for row in prompt_rows
                ]
                
                # Load entities
                entity_query = '''
                    SELECT entity_id, label, confidence, bbox_json, mask_path, color_hex, area_percent
                    FROM entities
                    WHERE session_id = ?
                '''
                entity_cursor = conn.execute(entity_query, (session_id,))
                entity_rows = entity_cursor.fetchall()
                
                session_data["entities"] = []
                for row in entity_rows:
                    try:
                        bbox = json.loads(row[3]) if row[3] else {}
                    except json.JSONDecodeError:
                        bbox = {}
                    
                    session_data["entities"].append({
                        "entity_id": row[0],
                        "label": row[1],
                        "confidence": row[2],
                        "bbox": bbox,
                        "mask_path": row[4],
                        "color_hex": row[5],
                        "area_percent": row[6]
                    })
                
                # Load validations
                validation_query = '''
                    SELECT attempt_number, alignment_score, preserved_count, modified_count, 
                           unintended_count, user_feedback, created_at
                    FROM validations
                    WHERE session_id = ?
                    ORDER BY attempt_number
                '''
                validation_cursor = conn.execute(validation_query, (session_id,))
                validation_rows = validation_cursor.fetchall()
                
                session_data["validations"] = [
                    {
                        "attempt_number": row[0],
                        "alignment_score": row[1],
                        "preserved_count": row[2],
                        "modified_count": row[3],
                        "unintended_count": row[4],
                        "user_feedback": row[5],
                        "created_at": row[6]
                    }
                    for row in validation_rows
                ]
                
                # Load feedback
                feedback_query = '''
                    SELECT feedback_type, comments, rating, created_at
                    FROM user_feedback
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                '''
                feedback_cursor = conn.execute(feedback_query, (session_id,))
                feedback_row = feedback_cursor.fetchone()
                
                if feedback_row:
                    session_data["user_feedback"] = {
                        "feedback_type": feedback_row[0],
                        "comments": feedback_row[1],
                        "rating": feedback_row[2],
                        "created_at": feedback_row[3]
                    }
                
                conn.execute("COMMIT")
                
                self.logger.info(f"Session {session_id} loaded successfully with optimizations")
                return session_data
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to load session {session_id}: {str(e)}")
            raise DatabaseError(f"Failed to load session: {str(e)}")
    
    def bulk_load_sessions(self, session_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Load multiple sessions in a single optimized operation.
        
        Args:
            session_ids: List of session IDs to load
            
        Returns:
            Dictionary mapping session IDs to their data
        """
        sessions_data = {}
        
        try:
            with self._get_connection() as conn:
                conn.execute("BEGIN")
                
                for session_id in session_ids:
                    try:
                        # Load session data (reuse existing logic)
                        session_data = self._load_single_session_internal(conn, session_id)
                        sessions_data[session_id] = session_data
                    except DatabaseError:
                        # Continue with other sessions if one fails
                        self.logger.warning(f"Failed to load session {session_id}")
                        continue
                
                conn.execute("COMMIT")
                
                self.logger.info(f"Bulk loaded {len(sessions_data)} sessions")
                return sessions_data
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to bulk load sessions: {str(e)}")
            raise DatabaseError(f"Failed to bulk load sessions: {str(e)}")
    
    def _load_single_session_internal(self, 
                                     conn: sqlite3.Connection, 
                                     session_id: str) -> Dict[str, Any]:
        """
        Internal method to load a single session within a transaction.
        """
        # This method would contain the same logic as load_session_optimized
        # but adapted to work within an existing transaction
        # Implementation omitted for brevity but would be similar to above
        
        pass

# Example usage
if __name__ == "__main__":
    # Initialize advanced database
    config = DatabaseConfig(
        db_path="edi_advanced.db",
        max_connections=20,
        cache_size=50
    )
    advanced_db = AdvancedDatabase(config)
    
    print("Advanced Database initialized for session loading")
    
    # Example usage would be similar to previous example
    print("Ready to load sessions with advanced optimizations and caching")
```