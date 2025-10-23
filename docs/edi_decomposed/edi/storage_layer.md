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