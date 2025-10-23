# Database.query_history()

[Back to Database](../storage_database.md)

## Related User Story
"As a user, I want to review my past editing sessions." (from PRD - implied by session persistence requirements)

## Function Signature
`query_history()`

## Parameters
- None - Queries the session history with default parameters

## Returns
- List of session summary objects with basic information about past sessions

## Step-by-step Logic
1. Execute a query against the sessions table to retrieve session summaries
2. Apply default filters (e.g., exclude failed sessions, limit to recent sessions)
3. Select relevant fields for session summaries (ID, image path, prompt, timestamp, score)
4. Order results by creation date (most recent first)
5. Apply any default limits to prevent excessive result sets
6. Return the list of session summaries
7. Handle database errors gracefully and return appropriate results

## Query Parameters
- By default, retrieves recent successful sessions
- Can be extended to accept filters (date ranges, image path patterns, score thresholds)
- Limits results to prevent performance issues
- Orders by most recent sessions first

## Performance Considerations
- Uses indexing on relevant columns (timestamp, status)
- Limits result set size to prevent memory issues
- Only selects needed fields for summaries
- Efficient query structure to minimize database load

## Input/Output Data Structures
### Session Summary Object
A simplified object containing:
- Session ID (UUID)
- Image path
- Original prompt (truncated if too long)
- Creation timestamp
- Final alignment score
- Session status
- Duration (if available)

### Query Results
A list of session summary objects:
- Ordered by creation date (descending)
- Limited to prevent excessive results
- Contains essential information for session review

## See Docs

### Python Implementation Example
Implementation of the Database.query_history() method:

```python
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

class DatabaseError(Exception):
    """Custom exception for database errors."""
    pass

class Database:
    """
    Database class for session history queries in EDI.
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
                
                # Create indexes for performance
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_sessions_created_at 
                    ON sessions(created_at DESC)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_sessions_status 
                    ON sessions(status)
                ''')
                
                conn.commit()
                
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to initialize database: {str(e)}")
    
    def query_history(self, 
                      limit: int = 50,
                      offset: int = 0,
                      status_filter: Optional[str] = None,
                      date_from: Optional[datetime] = None,
                      date_to: Optional[datetime] = None,
                      min_score: Optional[float] = None,
                      max_score: Optional[float] = None,
                      search_term: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Query session history with various filtering options.
        
        Args:
            limit: Maximum number of results to return
            offset: Number of results to skip (for pagination)
            status_filter: Filter by session status ('in_progress', 'completed', 'failed')
            date_from: Filter sessions created after this date
            date_to: Filter sessions created before this date
            min_score: Filter sessions with alignment score >= this value
            max_score: Filter sessions with alignment score <= this value
            search_term: Search term to match in image path or prompt
            
        Returns:
            List of session summary objects with basic information about past sessions
        """
        # Execute a query against the sessions table to retrieve session summaries
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build query with dynamic WHERE clause
                base_query = '''
                    SELECT id, created_at, image_path, naive_prompt, status, final_alignment_score
                    FROM sessions
                '''
                
                where_conditions = []
                query_params = []
                
                # Apply default filters
                # Exclude failed sessions by default unless specifically requested
                if status_filter is None:
                    where_conditions.append("status != 'failed'")
                elif status_filter:
                    where_conditions.append("status = ?")
                    query_params.append(status_filter)
                
                # Apply date filters
                if date_from:
                    where_conditions.append("created_at >= ?")
                    query_params.append(date_from.isoformat())
                
                if date_to:
                    where_conditions.append("created_at <= ?")
                    query_params.append(date_to.isoformat())
                
                # Apply score filters
                if min_score is not None:
                    where_conditions.append("final_alignment_score >= ?")
                    query_params.append(min_score)
                
                if max_score is not None:
                    where_conditions.append("final_alignment_score <= ?")
                    query_params.append(max_score)
                
                # Apply search term filter
                if search_term:
                    search_conditions = []
                    search_params = []
                    
                    # Search in image path
                    search_conditions.append("image_path LIKE ?")
                    search_params.append(f"%{search_term}%")
                    query_params.extend(search_params)
                    
                    # Search in prompt
                    search_conditions.append("naive_prompt LIKE ?")
                    search_params.append(f"%{search_term}%")
                    query_params.extend(search_params)
                    
                    where_conditions.append(f"({' OR '.join(search_conditions)})")
                
                # Construct WHERE clause
                if where_conditions:
                    base_query += " WHERE " + " AND ".join(where_conditions)
                
                # Order results by creation date (most recent first)
                base_query += " ORDER BY created_at DESC"
                
                # Apply limit and offset for pagination
                base_query += " LIMIT ? OFFSET ?"
                query_params.extend([limit, offset])
                
                # Execute the query
                cursor.execute(base_query, query_params)
                rows = cursor.fetchall()
                
                # Process results into session summary objects
                session_summaries = []
                
                for row in rows:
                    # Select relevant fields for session summaries
                    session_summary = {
                        "id": row[0],
                        "created_at": row[1],
                        "image_path": row[2],
                        "naive_prompt": row[3][:100] + "..." if len(row[3]) > 100 else row[3],  # Truncate long prompts
                        "status": row[4],
                        "final_alignment_score": row[5]
                    }
                    
                    # Calculate duration if start/end times are available
                    # In this simple example, we'll just note that duration calculation would go here
                    # session_summary["duration"] = self._calculate_session_duration(row[0])
                    
                    session_summaries.append(session_summary)
                
                # Handle database errors gracefully and return appropriate results
                self.logger.info(f"Retrieved {len(session_summaries)} session summaries from history")
                return session_summaries
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to query session history: {str(e)}")
            # Return empty list on error rather than raising exception
            return []
    
    def _calculate_session_duration(self, session_id: str) -> Optional[str]:
        """
        Calculate the duration of a session (helper method).
        
        Args:
            session_id: UUID of the session
            
        Returns:
            Duration string or None if calculation fails
        """
        # This would require additional tables with start/end timestamps
        # For now, return None to indicate duration not available
        return None
    
    def get_recent_sessions(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent sessions (convenience method).
        
        Args:
            count: Number of recent sessions to retrieve
            
        Returns:
            List of recent session summaries
        """
        return self.query_history(limit=count)
    
    def get_successful_sessions(self, 
                               limit: int = 50,
                               min_score: Optional[float] = 0.7) -> List[Dict[str, Any]]:
        """
        Get successful sessions with optional minimum score filter.
        
        Args:
            limit: Maximum number of results to return
            min_score: Minimum alignment score for successful sessions
            
        Returns:
            List of successful session summaries
        """
        return self.query_history(
            limit=limit,
            status_filter="completed",
            min_score=min_score
        )
    
    def search_sessions(self, 
                       search_term: str,
                       limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search sessions by image path or prompt content.
        
        Args:
            search_term: Term to search for in image paths or prompts
            limit: Maximum number of results to return
            
        Returns:
            List of matching session summaries
        """
        return self.query_history(
            limit=limit,
            search_term=search_term
        )
    
    def get_sessions_by_date_range(self, 
                                  start_date: datetime,
                                  end_date: datetime,
                                  limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get sessions within a specific date range.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            limit: Maximum number of results to return
            
        Returns:
            List of session summaries within the date range
        """
        return self.query_history(
            limit=limit,
            date_from=start_date,
            date_to=end_date
        )
    
    def get_session_count(self, 
                         status_filter: Optional[str] = None,
                         date_from: Optional[datetime] = None,
                         date_to: Optional[datetime] = None) -> int:
        """
        Get the total count of sessions matching filters (for pagination).
        
        Args:
            status_filter: Filter by session status
            date_from: Filter sessions created after this date
            date_to: Filter sessions created before this date
            
        Returns:
            Count of matching sessions
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                base_query = "SELECT COUNT(*) FROM sessions"
                where_conditions = []
                query_params = []
                
                # Apply filters
                if status_filter:
                    where_conditions.append("status = ?")
                    query_params.append(status_filter)
                
                if date_from:
                    where_conditions.append("created_at >= ?")
                    query_params.append(date_from.isoformat())
                
                if date_to:
                    where_conditions.append("created_at <= ?")
                    query_params.append(date_to.isoformat())
                
                # Construct WHERE clause
                if where_conditions:
                    base_query += " WHERE " + " AND ".join(where_conditions)
                
                cursor.execute(base_query, query_params)
                count = cursor.fetchone()[0]
                
                return count
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get session count: {str(e)}")
            return 0
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics about sessions.
        
        Returns:
            Dictionary containing session statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total sessions by status
                cursor.execute('''
                    SELECT status, COUNT(*) 
                    FROM sessions 
                    GROUP BY status
                ''')
                status_counts = dict(cursor.fetchall())
                
                # Get average alignment score for completed sessions
                cursor.execute('''
                    SELECT AVG(final_alignment_score) 
                    FROM sessions 
                    WHERE status = 'completed' AND final_alignment_score IS NOT NULL
                ''')
                avg_score = cursor.fetchone()[0]
                
                # Get most recent session date
                cursor.execute('''
                    SELECT MAX(created_at) 
                    FROM sessions
                ''')
                last_session = cursor.fetchone()[0]
                
                # Get total session count
                cursor.execute('''
                    SELECT COUNT(*) 
                    FROM sessions
                ''')
                total_sessions = cursor.fetchone()[0]
                
                return {
                    "total_sessions": total_sessions,
                    "status_breakdown": status_counts,
                    "average_alignment_score": round(avg_score, 3) if avg_score else 0,
                    "last_session_date": last_session,
                    "successful_rate": round(
                        (status_counts.get("completed", 0) / total_sessions * 100) 
                        if total_sessions > 0 else 0, 
                        1
                    )
                }
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get session statistics: {str(e)}")
            return {
                "total_sessions": 0,
                "status_breakdown": {},
                "average_alignment_score": 0,
                "last_session_date": None,
                "successful_rate": 0
            }

# Example usage
if __name__ == "__main__":
    # Initialize database
    db = Database()
    
    # Example: Query recent sessions
    try:
        recent_sessions = db.get_recent_sessions(5)
        print(f"Recent sessions ({len(recent_sessions)}):")
        for session in recent_sessions:
            print(f"  - {session['created_at'][:19]}: {session['naive_prompt'][:50]}...")
        
        # Example: Query successful sessions
        successful_sessions = db.get_successful_sessions(min_score=0.8)
        print(f"\nSuccessful sessions ({len(successful_sessions)}):")
        for session in successful_sessions[:3]:  # Show first 3
            print(f"  - Score: {session['final_alignment_score']:.2f} - {session['naive_prompt'][:50]}...")
        
        # Example: Search sessions
        search_results = db.search_sessions("sky", limit=5)
        print(f"\nSearch results for 'sky' ({len(search_results)}):")
        for session in search_results:
            print(f"  - {session['image_path']} - {session['naive_prompt'][:50]}...")
        
        # Example: Get statistics
        stats = db.get_session_statistics()
        print(f"\nSession Statistics:")
        print(f"  Total sessions: {stats['total_sessions']}")
        print(f"  Successful rate: {stats['successful_rate']}%")
        print(f"  Average score: {stats['average_alignment_score']}")
        print(f"  Last session: {stats['last_session_date']}")
        
        # Example: Query with pagination
        page_1 = db.query_history(limit=10, offset=0)
        page_2 = db.query_history(limit=10, offset=10)
        print(f"\nPagination example:")
        print(f"  Page 1: {len(page_1)} sessions")
        print(f"  Page 2: {len(page_2)} sessions")
        
    except DatabaseError as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")
```

### Advanced History Query Implementation
Enhanced implementation with advanced filtering and performance optimizations:

```python
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

@dataclass
class QueryHistoryParams:
    """Parameters for querying session history."""
    limit: int = 50
    offset: int = 0
    status_filter: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    search_term: Optional[str] = None
    order_by: str = "created_at DESC"  # Default ordering
    include_duration: bool = False  # Whether to calculate session duration

@dataclass
class SessionStatistics:
    """Session statistics data structure."""
    total_sessions: int
    status_breakdown: Dict[str, int]
    average_alignment_score: float
    last_session_date: Optional[str]
    successful_rate: float
    sessions_per_day: float

class AdvancedDatabase:
    """
    Advanced database with connection pooling and performance optimizations for history queries.
    """
    
    def __init__(self, db_path: str = "edi_sessions.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._connections = []
        self._connection_lock = threading.Lock()
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database with optimized settings."""
        try:
            with self._get_connection() as conn:
                # Set performance optimizations
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
                conn.execute("PRAGMA cache_size = 10000")  # 10MB cache
                conn.execute("PRAGMA temp_store = MEMORY")
                
                # Create tables with indexes
                self._create_tables(conn)
                
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to initialize database: {str(e)}")
    
    def _create_tables(self, conn: sqlite3.Connection):
        """Create database tables with indexes for performance."""
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
                INDEX idx_sessions_created_at (created_at DESC),
                INDEX idx_sessions_status (status),
                INDEX idx_sessions_score (final_alignment_score)
            )
        ''')
        
        # Add more indexes for complex queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_sessions_path_status 
            ON sessions(image_path, status)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_sessions_prompt 
            ON sessions(naive_prompt)
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
                    self.db_path,
                    timeout=30,
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
                    if len(self._connections) < 10:  # Max pool size
                        self._connections.append(conn)
                    else:
                        conn.close()
    
    def query_history_advanced(self, params: QueryHistoryParams) -> List[Dict[str, Any]]:
        """
        Advanced session history query with optimized performance.
        
        Args:
            params: QueryHistoryParams object with filtering and pagination options
            
        Returns:
            List of session summary objects
        """
        try:
            with self._get_connection() as conn:
                # Use prepared statements for performance
                conn.execute("BEGIN")
                
                # Build dynamic query
                base_query = '''
                    SELECT id, created_at, image_path, naive_prompt, status, final_alignment_score
                    FROM sessions
                '''
                
                where_conditions = []
                query_params = []
                
                # Apply filters
                if params.status_filter:
                    where_conditions.append("status = ?")
                    query_params.append(params.status_filter)
                
                if params.date_from:
                    where_conditions.append("created_at >= ?")
                    query_params.append(params.date_from.isoformat())
                
                if params.date_to:
                    where_conditions.append("created_at <= ?")
                    query_params.append(params.date_to.isoformat())
                
                if params.min_score is not None:
                    where_conditions.append("final_alignment_score >= ?")
                    query_params.append(params.min_score)
                
                if params.max_score is not None:
                    where_conditions.append("final_alignment_score <= ?")
                    query_params.append(params.max_score)
                
                if params.search_term:
                    # Search in both image path and prompt
                    search_conditions = ["image_path LIKE ?", "naive_prompt LIKE ?"]
                    search_params = [f"%{params.search_term}%", f"%{params.search_term}%"]
                    where_conditions.append(f"({' OR '.join(search_conditions)})")
                    query_params.extend(search_params)
                
                # Construct WHERE clause
                if where_conditions:
                    base_query += " WHERE " + " AND ".join(where_conditions)
                
                # Add ordering
                base_query += f" ORDER BY {params.order_by}"
                
                # Add pagination
                base_query += " LIMIT ? OFFSET ?"
                query_params.extend([params.limit, params.offset])
                
                # Execute query
                cursor = conn.execute(base_query, query_params)
                rows = cursor.fetchall()
                
                # Process results
                session_summaries = []
                for row in rows:
                    session_summary = {
                        "id": row[0],
                        "created_at": row[1],
                        "image_path": row[2],
                        "naive_prompt": row[3][:100] + "..." if len(row[3]) > 100 else row[3],
                        "status": row[4],
                        "final_alignment_score": row[5]
                    }
                    
                    # Add duration if requested
                    if params.include_duration:
                        session_summary["duration"] = self._calculate_session_duration_optimized(conn, row[0])
                    
                    session_summaries.append(session_summary)
                
                conn.execute("COMMIT")
                
                self.logger.info(f"Retrieved {len(session_summaries)} sessions with advanced query")
                return session_summaries
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to query session history: {str(e)}")
            return []
    
    def _calculate_session_duration_optimized(self, conn: sqlite3.Connection, session_id: str) -> Optional[str]:
        """
        Optimized session duration calculation.
        """
        # This would require additional tables with timestamps
        # For now, return None
        return None
    
    def query_history_with_aggregation(self, 
                                      group_by: str = "day",
                                      date_from: Optional[datetime] = None,
                                      date_to: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Query session history with aggregation (e.g., sessions per day/week/month).
        
        Args:
            group_by: Aggregation level - "day", "week", "month"
            date_from: Start date for aggregation
            date_to: End date for aggregation
            
        Returns:
            List of aggregated session data
        """
        try:
            with self._get_connection() as conn:
                # Determine grouping function based on period
                if group_by == "day":
                    date_format = "%Y-%m-%d"
                    group_expr = "DATE(created_at)"
                elif group_by == "week":
                    date_format = "%Y-W%W"
                    group_expr = "strftime('%Y-W%W', created_at)"
                elif group_by == "month":
                    date_format = "%Y-%m"
                    group_expr = "strftime('%Y-%m', created_at)"
                else:
                    raise ValueError(f"Unsupported group_by value: {group_by}")
                
                # Build aggregation query
                query = f'''
                    SELECT 
                        {group_expr} as period,
                        COUNT(*) as session_count,
                        AVG(final_alignment_score) as avg_score,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_count,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_count
                    FROM sessions
                '''
                
                where_conditions = []
                query_params = []
                
                # Apply date filters
                if date_from:
                    where_conditions.append("created_at >= ?")
                    query_params.append(date_from.isoformat())
                
                if date_to:
                    where_conditions.append("created_at <= ?")
                    query_params.append(date_to.isoformat())
                
                # Add WHERE clause
                if where_conditions:
                    query += " WHERE " + " AND ".join(where_conditions)
                
                # Add GROUP BY and ORDER BY
                query += f" GROUP BY {group_expr} ORDER BY period"
                
                cursor = conn.execute(query, query_params)
                rows = cursor.fetchall()
                
                # Process results
                aggregated_data = []
                for row in rows:
                    aggregated_data.append({
                        "period": row[0],
                        "session_count": row[1],
                        "avg_alignment_score": round(row[2], 3) if row[2] else 0,
                        "completed_count": row[3],
                        "failed_count": row[4],
                        "success_rate": round((row[3] / row[1] * 100) if row[1] > 0 else 0, 1)
                    })
                
                return aggregated_data
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to query aggregated session history: {str(e)}")
            return []
    
    def get_advanced_session_statistics(self) -> SessionStatistics:
        """
        Get comprehensive session statistics with advanced metrics.
        
        Returns:
            SessionStatistics object with detailed metrics
        """
        try:
            with self._get_connection() as conn:
                # Get total sessions by status
                cursor = conn.execute('''
                    SELECT status, COUNT(*) 
                    FROM sessions 
                    GROUP BY status
                ''')
                status_counts = dict(cursor.fetchall())
                
                # Get average alignment score
                cursor = conn.execute('''
                    SELECT AVG(final_alignment_score) 
                    FROM sessions 
                    WHERE status = 'completed' AND final_alignment_score IS NOT NULL
                ''')
                avg_score_row = cursor.fetchone()
                avg_score = avg_score_row[0] if avg_score_row[0] is not None else 0
                
                # Get last session date
                cursor = conn.execute('''
                    SELECT MAX(created_at) 
                    FROM sessions
                ''')
                last_session = cursor.fetchone()[0]
                
                # Get total session count
                cursor = conn.execute('''
                    SELECT COUNT(*) 
                    FROM sessions
                ''')
                total_sessions = cursor.fetchone()[0]
                
                # Calculate sessions per day (last 30 days)
                thirty_days_ago = datetime.now() - timedelta(days=30)
                cursor = conn.execute('''
                    SELECT COUNT(*) 
                    FROM sessions 
                    WHERE created_at >= ?
                ''', (thirty_days_ago.isoformat(),))
                recent_sessions = cursor.fetchone()[0]
                sessions_per_day = recent_sessions / 30 if recent_sessions > 0 else 0
                
                # Calculate success rate
                successful_rate = (
                    (status_counts.get("completed", 0) / total_sessions * 100) 
                    if total_sessions > 0 else 0
                )
                
                return SessionStatistics(
                    total_sessions=total_sessions,
                    status_breakdown=status_counts,
                    average_alignment_score=round(avg_score, 3),
                    last_session_date=last_session,
                    successful_rate=round(successful_rate, 1),
                    sessions_per_day=round(sessions_per_day, 2)
                )
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get advanced session statistics: {str(e)}")
            return SessionStatistics(
                total_sessions=0,
                status_breakdown={},
                average_alignment_score=0,
                last_session_date=None,
                successful_rate=0,
                sessions_per_day=0
            )
    
    def search_sessions_advanced(self, 
                                search_term: str,
                                search_fields: List[str] = None,
                                limit: int = 20,
                                fuzzy_match: bool = False) -> List[Dict[str, Any]]:
        """
        Advanced session search with multiple field options and fuzzy matching.
        
        Args:
            search_term: Term to search for
            search_fields: Fields to search in ("image_path", "naive_prompt", etc.)
            limit: Maximum results to return
            fuzzy_match: Whether to use fuzzy matching
            
        Returns:
            List of matching session summaries
        """
        if search_fields is None:
            search_fields = ["image_path", "naive_prompt"]
        
        try:
            with self._get_connection() as conn:
                # Build search query
                base_query = '''
                    SELECT id, created_at, image_path, naive_prompt, status, final_alignment_score
                    FROM sessions
                '''
                
                where_conditions = []
                query_params = []
                
                # Add search conditions for each field
                field_conditions = []
                for field in search_fields:
                    if fuzzy_match:
                        # Use LIKE with wildcards for fuzzy matching
                        field_conditions.append(f"{field} LIKE ?")
                        query_params.append(f"%{search_term}%")
                    else:
                        # Exact match
                        field_conditions.append(f"{field} = ?")
                        query_params.append(search_term)
                
                where_conditions.append(f"({' OR '.join(field_conditions)})")
                
                # Add WHERE clause
                base_query += " WHERE " + " AND ".join(where_conditions)
                
                # Add ordering and limit
                base_query += " ORDER BY created_at DESC LIMIT ?"
                query_params.append(limit)
                
                cursor = conn.execute(base_query, query_params)
                rows = cursor.fetchall()
                
                # Process results
                search_results = []
                for row in rows:
                    search_results.append({
                        "id": row[0],
                        "created_at": row[1],
                        "image_path": row[2],
                        "naive_prompt": row[3][:100] + "..." if len(row[3]) > 100 else row[3],
                        "status": row[4],
                        "final_alignment_score": row[5]
                    })
                
                return search_results
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to search sessions: {str(e)}")
            return []

# Example usage
if __name__ == "__main__":
    # Initialize advanced database
    advanced_db = AdvancedDatabase("edi_advanced.db")
    
    print("Advanced Database initialized for session history queries")
    
    # Example usage would be similar to previous examples but with advanced features
    print("Ready to query session history with advanced options")
    
    # Example: Advanced query with parameters
    params = QueryHistoryParams(
        limit=20,
        status_filter="completed",
        min_score=0.8,
        order_by="final_alignment_score DESC"
    )
    
    results = advanced_db.query_history_advanced(params)
    print(f"Advanced query returned {len(results)} results")
    
    # Example: Aggregated data
    aggregated = advanced_db.query_history_with_aggregation(
        group_by="day",
        date_from=datetime.now() - timedelta(days=7)
    )
    print(f"Aggregated data for last 7 days: {len(aggregated)} periods")
    
    # Example: Advanced statistics
    stats = advanced_db.get_advanced_session_statistics()
    print(f"Advanced statistics:")
    print(f"  Total sessions: {stats.total_sessions}")
    print(f"  Success rate: {stats.successful_rate}%")
    print(f"  Avg score: {stats.average_alignment_score}")
    print(f"  Sessions/day: {stats.sessions_per_day}")
```