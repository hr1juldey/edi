# Storage Layer

[Back to Index](../index.md)

## Purpose

Session persistence, learning data using SQLite + JSON

## Component Design

### 5. Storage Layer

**Purpose**: Persist session data for learning and resume functionality

#### 5.1 Database Schema (SQLite)

```sql
-- sessions table
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,  -- UUID
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    image_path TEXT NOT NULL,
    naive_prompt TEXT NOT NULL,
    status TEXT CHECK(status IN ('in_progress', 'completed', 'failed')),
    final_alignment_score REAL
);

-- prompts table (stores refinement history)
CREATE TABLE prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id),
    iteration INT,
    positive_prompt TEXT,
    negative_prompt TEXT,
    quality_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- entities table (detected objects)
CREATE TABLE entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id),
    entity_id TEXT,  -- e.g., "sky_0"
    label TEXT,
    confidence REAL,
    bbox_json TEXT,  -- Serialized bounding box
    mask_path TEXT,  -- Path to saved mask file
    color_hex TEXT,
    area_percent REAL
);

-- validations table (edit assessment)
CREATE TABLE validations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id),
    attempt_number INT,
    alignment_score REAL,
    preserved_count INT,
    modified_count INT,
    unintended_count INT,
    user_feedback TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- user_feedback table (for learning)
CREATE TABLE user_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id),
    feedback_type TEXT CHECK(feedback_type IN ('accept', 'reject', 'partial')),
    comments TEXT,
    rating INT CHECK(rating BETWEEN 1 AND 5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 5.2 Session State Management

**State File** (JSON for current session):

```json
{
  "session_id": "uuid-here",
  "current_stage": "refinement",
  "image_path": "/path/to/image.jpg",
  "naive_prompt": "make sky dramatic",
  "scene_analysis": {
    "entities": [...],
    "spatial_layout": "..."
  },
  "intent": {
    "target_entities": ["sky_0"],
    "edit_type": "style",
    "confidence": 0.85
  },
  "prompts": {
    "iteration_0": {
      "positive": "...",
      "negative": "..."
    },
    "iteration_1": {...},
    "final": {...}
  },
  "edited_image_path": "/path/to/edited.jpg",
  "validation": {
    "score": 0.87,
    "delta": {...}
  }
}
```

**Auto-save**: Write state file every 5 seconds or after significant events

## Sub-modules

This component includes the following modules:

- [storage/database.py](./database/database.md)
- [storage/models.py](./models.md)
- [storage/migrations.py](./migrations/migrations.md)

## Technology Stack

- SQLite for database storage
- JSON for session state
- Pydantic for data validation

## See Docs

### Pydantic Implementation Example

Data validation models for the EDI application:

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

# Model for session data
class SessionData(BaseModel):
    session_id: str = Field(..., description="UUID for the session")
    created_at: datetime = Field(default_factory=datetime.now)
    image_path: str = Field(..., description="Path to the original image")
    naive_prompt: str = Field(..., description="Initial user prompt")
    status: str = Field(..., regex=r'^(in_progress|completed|failed))
    final_alignment_score: Optional[float] = None

# Model for prompt history
class PromptRecord(BaseModel):
    iteration: int
    positive_prompt: str
    negative_prompt: str
    quality_score: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.now)

# Model for detected entities
class EntityData(BaseModel):
    entity_id: str  # e.g., "sky_0"
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_json: Optional[str] = None  # Serialized bounding box
    mask_path: Optional[str] = None
    color_hex: Optional[str] = None
    area_percent: Optional[float] = Field(None, ge=0.0, le=100.0)

# Model for validation data
class ValidationData(BaseModel):
    attempt_number: int
    alignment_score: float = Field(ge=0.0, le=1.0)
    preserved_count: int
    modified_count: int
    unintended_count: int
    user_feedback: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

# Model for user feedback
class UserFeedback(BaseModel):
    feedback_type: str = Field(..., regex=r'^(accept|reject|partial))
    comments: Optional[str] = None
    rating: Optional[int] = Field(None, ge=1, le=5)
    created_at: datetime = Field(default_factory=datetime.now)

# Complete session state model
class SessionState(BaseModel):
    session_id: str
    current_stage: str = Field(..., regex=r'^(upload|refinement|validation|completed))
    image_path: str
    naive_prompt: str
    scene_analysis: Dict[str, Any] = {}
    intent: Dict[str, Any] = {}
    prompts: Dict[str, Any] = {}
    edited_image_path: Optional[str] = None
    validation: Dict[str, Any] = {}

# Example usage
if __name__ == "__main__":
    # Example of creating and validating session data
    session_data = SessionData(
        session_id="session-123",
        image_path="/path/to/image.jpg",
        naive_prompt="make sky dramatic",
        status="in_progress"
    )
    print("Session data validated:", session_data.model_dump())
    
    # Example of validating session state from JSON
    session_json = '''
    {
        "session_id": "session-456",
        "current_stage": "refinement",
        "image_path": "/path/to/image2.jpg",
        "naive_prompt": "enhance lighting",
        "scene_analysis": {
            "entities": [],
            "spatial_layout": "..."
        },
        "intent": {
            "target_entities": ["sky_0"],
            "edit_type": "style",
            "confidence": 0.85
        },
        "prompts": {
            "iteration_0": {
                "positive": "...",
                "negative": "..."
            }
        },
        "edited_image_path": "/path/to/edited_image2.jpg",
        "validation": {
            "score": 0.87,
            "delta": {}
        }
    }
    '''
    
    session_state = SessionState.model_validate_json(session_json)
    print("Session state loaded and validated:", session_state.model_dump())
```

### SQLite Implementation Example

Database operations for the EDI application:

```python
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
import os

class EDIDatabase:
    def __init__(self, db_path: str = "edi_sessions.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sessions table
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
        
        # Prompts table
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
                area_percent REAL
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_session(self, session_data: Dict) -> str:
        """Save a session to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sessions (id, image_path, naive_prompt, status, final_alignment_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            session_data['id'],
            session_data['image_path'], 
            session_data['naive_prompt'],
            session_data['status'],
            session_data.get('final_alignment_score')
        ))
        
        conn.commit()
        conn.close()
        return session_data['id']
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Retrieve a session from the database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM sessions WHERE id = ?', (session_id,))
        row = cursor.fetchone()
        
        if row:
            session = dict(row)
            conn.close()
            return session
        
        conn.close()
        return None
    
    def save_prompt_history(self, session_id: str, prompts: List[Dict]):
        """Save prompt history for a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for prompt in prompts:
            cursor.execute('''
                INSERT INTO prompts (session_id, iteration, positive_prompt, negative_prompt, quality_score)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                session_id,
                prompt['iteration'],
                prompt['positive_prompt'],
                prompt['negative_prompt'],
                prompt.get('quality_score')
            ))
        
        conn.commit()
        conn.close()
    
    def get_prompt_history(self, session_id: str) -> List[Dict]:
        """Retrieve prompt history for a session."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT iteration, positive_prompt, negative_prompt, quality_score, created_at
            FROM prompts
            WHERE session_id = ?
            ORDER BY iteration
        ''', (session_id,))
        
        rows = cursor.fetchall()
        prompts = [dict(row) for row in rows]
        
        conn.close()
        return prompts
    
    def save_session_state(self, session_state: Dict, filepath: str):
        """Save session state to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(session_state, f, indent=2, default=str)
    
    def load_session_state(self, filepath: str) -> Dict:
        """Load session state from a JSON file."""
        if not os.path.exists(filepath):
            return {}
        
        with open(filepath, 'r') as f:
            return json.load(f)

# Example usage
if __name__ == "__main__":
    # Initialize database
    db = EDIDatabase()
    
    # Example session data
    session_data = {
        'id': 'session-789',
        'image_path': '/path/to/image3.jpg',
        'naive_prompt': 'change background color',
        'status': 'in_progress'
    }
    
    # Save session
    session_id = db.save_session(session_data)
    print(f"Saved session with ID: {session_id}")
    
    # Retrieve session
    retrieved_session = db.get_session(session_id)
    print(f"Retrieved session: {retrieved_session}")
    
    # Example prompt history
    prompts = [
        {
            'iteration': 0,
            'positive_prompt': 'change background to blue',
            'negative_prompt': 'keep foreground unchanged',
            'quality_score': 0.85
        },
        {
            'iteration': 1,
            'positive_prompt': 'change background to bright blue',
            'negative_prompt': 'keep foreground unchanged',
            'quality_score': 0.92
        }
    ]
    
    # Save and retrieve prompt history
    db.save_prompt_history(session_id, prompts)
    retrieved_prompts = db.get_prompt_history(session_id)
    print(f"Retrieved prompts: {retrieved_prompts}")
```

### JSON Implementation Example

Session state management using JSON:

```python
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

class SessionStateManager:
    """Manages session state using JSON files."""
    
    def __init__(self, session_dir: str = "sessions"):
        self.session_dir = session_dir
        os.makedirs(session_dir, exist_ok=True)
    
    def save_session_state(self, session_id: str, state: Dict[str, Any]) -> str:
        """Save session state to a JSON file."""
        filepath = os.path.join(self.session_dir, f"{session_id}.json")
        
        # Add timestamp
        state['last_updated'] = datetime.now().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        return filepath
    
    def load_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session state from a JSON file."""
        filepath = os.path.join(self.session_dir, f"{session_id}.json")
        
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def update_session_state(self, session_id: str, updates: Dict[str, Any]) -> str:
        """Update session state with new values."""
        current_state = self.load_session_state(session_id) or {}
        current_state.update(updates)
        
        return self.save_session_state(session_id, current_state)
    
    def auto_save_state(self, session_id: str, state: Dict[str, Any], interval_seconds: int = 5):
        """Auto-save state every specified interval (simulated with a function)."""
        import threading
        import time
        
        def auto_save():
            while True:
                time.sleep(interval_seconds)
                self.save_session_state(session_id, state)
                print(f"Auto-saved session {session_id}")
        
        # Start auto-save thread (for demonstration)
        # Note: In a real application, you'd want to control this properly
        thread = threading.Thread(target=auto_save, daemon=True)
        return thread

# Example usage
if __name__ == "__main__":
    # Initialize session state manager
    state_manager = SessionStateManager()
    
    # Example session state
    session_state = {
        "session_id": "session-json-123",
        "current_stage": "refinement",
        "image_path": "/path/to/image.jpg",
        "naive_prompt": "make sky dramatic",
        "scene_analysis": {
            "entities": [
                {"id": "sky_0", "label": "sky", "confidence": 0.95},
                {"id": "mountain_1", "label": "mountain", "confidence": 0.87}
            ],
            "spatial_layout": "sky occupies top half, mountains at horizon"
        },
        "intent": {
            "target_entities": ["sky_0"],
            "edit_type": "style",
            "confidence": 0.85
        },
        "prompts": {
            "iteration_0": {
                "positive": "dramatic sky with storm clouds",
                "negative": "clear blue sky, no clouds"
            }
        },
        "edited_image_path": None,
        "validation": {}
    }
    
    # Save session state
    filepath = state_manager.save_session_state("session-json-123", session_state)
    print(f"Session state saved to: {filepath}")
    
    # Load session state
    loaded_state = state_manager.load_session_state("session-json-123")
    print(f"Session stage: {loaded_state.get('current_stage')}")
    
    # Update session state
    state_manager.update_session_state("session-json-123", {
        "edited_image_path": "/path/to/edited_image.jpg",
        "validation": {"score": 0.87}
    })
    
    print("Session state updated with edited image path")
```
