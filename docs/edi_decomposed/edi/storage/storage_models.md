# Storage: Models

[Back to Storage Layer](./storage_layer.md)

## Purpose
Database models - Contains SessionRecord, PromptRecord, EntityRecord and other data structures that map to database tables.

## Models
- `SessionRecord`: Represents a session in the database
- `PromptRecord`: Represents a prompt history record
- `EntityRecord`: Represents an entity detection record
- Other related database models

### Details
- SQLAlchemy ORM or dataclasses with SQL mapping
- Maps directly to database tables
- Provides structured access to stored data

## Technology Stack

- SQLAlchemy ORM or dataclasses
- SQL mapping
- Pydantic for validation

## See Docs

### Python Implementation Example
Storage models implementation using dataclasses and Pydantic:

```python
import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pydantic import BaseModel, Field, validator
from typing_extensions import Annotated

class SessionStatus(Enum):
    """Enumeration for session statuses."""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class SessionRecord:
    """
    Represents a session in the database.
    Maps directly to the sessions table.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    image_path: str = ""
    naive_prompt: str = ""
    status: str = SessionStatus.IN_PROGRESS.value
    final_alignment_score: Optional[float] = None
    processing_time: Optional[float] = None
    model_version: Optional[str] = None
    session_tags: Optional[str] = None
    session_metadata: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "id": self.id,
            "created_at": self.created_at,
            "image_path": self.image_path,
            "naive_prompt": self.naive_prompt,
            "status": self.status,
            "final_alignment_score": self.final_alignment_score,
            "processing_time": self.processing_time,
            "model_version": self.model_version,
            "session_tags": self.session_tags,
            "session_metadata": self.session_metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionRecord':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class PromptRecord:
    """
    Represents a prompt history record.
    Maps directly to the prompts table.
    """
    id: Optional[int] = None
    session_id: str = ""
    iteration: int = 0
    positive_prompt: str = ""
    negative_prompt: str = ""
    quality_score: Optional[float] = None
    generation_time: Optional[float] = None
    model_confidence: Optional[float] = None
    prompt_metadata: Optional[str] = None
    generation_parameters: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "iteration": self.iteration,
            "positive_prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "quality_score": self.quality_score,
            "generation_time": self.generation_time,
            "model_confidence": self.model_confidence,
            "prompt_metadata": self.prompt_metadata,
            "generation_parameters": self.generation_parameters,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptRecord':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class EntityRecord:
    """
    Represents an entity detection record.
    Maps directly to the entities table.
    """
    id: Optional[int] = None
    session_id: str = ""
    entity_id: str = ""
    label: str = ""
    confidence: float = 0.0
    bbox_json: str = "{}"
    mask_path: Optional[str] = None
    color_hex: Optional[str] = None
    area_percent: Optional[float] = None
    embedding_vector: Optional[str] = None
    semantic_label: Optional[str] = None
    entity_metadata: Optional[str] = None
    quality_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "entity_id": self.entity_id,
            "label": self.label,
            "confidence": self.confidence,
            "bbox_json": self.bbox_json,
            "mask_path": self.mask_path,
            "color_hex": self.color_hex,
            "area_percent": self.area_percent,
            "embedding_vector": self.embedding_vector,
            "semantic_label": self.semantic_label,
            "entity_metadata": self.entity_metadata,
            "quality_score": self.quality_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityRecord':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class ValidationRecord:
    """
    Represents a validation record.
    Maps directly to the validations table.
    """
    id: Optional[int] = None
    session_id: str = ""
    attempt_number: int = 0
    alignment_score: Optional[float] = None
    preserved_count: Optional[int] = None
    modified_count: Optional[int] = None
    unintended_count: Optional[int] = None
    user_feedback: Optional[str] = None
    processing_details: Optional[str] = None
    quality_metrics: Optional[str] = None
    validation_metadata: Optional[str] = None
    user_ratings: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "attempt_number": self.attempt_number,
            "alignment_score": self.alignment_score,
            "preserved_count": self.preserved_count,
            "modified_count": self.modified_count,
            "unintended_count": self.unintended_count,
            "user_feedback": self.user_feedback,
            "processing_details": self.processing_details,
            "quality_metrics": self.quality_metrics,
            "validation_metadata": self.validation_metadata,
            "user_ratings": self.user_ratings,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationRecord':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class UserFeedbackRecord:
    """
    Represents a user feedback record.
    Maps directly to the user_feedback table.
    """
    id: Optional[int] = None
    session_id: str = ""
    feedback_type: str = "partial"
    comments: Optional[str] = None
    rating: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "feedback_type": self.feedback_type,
            "comments": self.comments,
            "rating": self.rating,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserFeedbackRecord':
        """Create from dictionary."""
        return cls(**data)

# Pydantic models for validation
class SessionModel(BaseModel):
    """
    Pydantic model for session validation.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    image_path: str = Field(..., min_length=1)
    naive_prompt: str = Field(..., min_length=1)
    status: str = Field(default=SessionStatus.IN_PROGRESS.value)
    final_alignment_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    processing_time: Optional[float] = Field(None, ge=0.0)
    model_version: Optional[str] = None
    session_tags: Optional[str] = None
    session_metadata: Optional[str] = None
    
    @validator('status')
    def validate_status(cls, v):
        """Validate session status."""
        valid_statuses = [status.value for status in SessionStatus]
        if v not in valid_statuses:
            raise ValueError(f"Invalid status: {v}. Must be one of {valid_statuses}")
        return v
    
    @validator('image_path')
    def validate_image_path(cls, v):
        """Validate image path."""
        if not v:
            raise ValueError("Image path cannot be empty")
        return v

class PromptModel(BaseModel):
    """
    Pydantic model for prompt validation.
    """
    id: Optional[int] = None
    session_id: str = Field(..., min_length=1)
    iteration: int = Field(..., ge=0)
    positive_prompt: str = Field(..., min_length=1)
    negative_prompt: str = ""
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    generation_time: Optional[float] = Field(None, ge=0.0)
    model_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    prompt_metadata: Optional[str] = None
    generation_parameters: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class EntityModel(BaseModel):
    """
    Pydantic model for entity validation.
    """
    id: Optional[int] = None
    session_id: str = Field(..., min_length=1)
    entity_id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox_json: str = "{}"
    mask_path: Optional[str] = None
    color_hex: Optional[str] = None
    area_percent: Optional[float] = Field(None, ge=0.0, le=100.0)
    embedding_vector: Optional[str] = None
    semantic_label: Optional[str] = None
    entity_metadata: Optional[str] = None
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)

class ValidationModel(BaseModel):
    """
    Pydantic model for validation record validation.
    """
    id: Optional[int] = None
    session_id: str = Field(..., min_length=1)
    attempt_number: int = Field(..., ge=0)
    alignment_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    preserved_count: Optional[int] = Field(None, ge=0)
    modified_count: Optional[int] = Field(None, ge=0)
    unintended_count: Optional[int] = Field(None, ge=0)
    user_feedback: Optional[str] = None
    processing_details: Optional[str] = None
    quality_metrics: Optional[str] = None
    validation_metadata: Optional[str] = None
    user_ratings: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class UserFeedbackModel(BaseModel):
    """
    Pydantic model for user feedback validation.
    """
    id: Optional[int] = None
    session_id: str = Field(..., min_length=1)
    feedback_type: str = Field("partial", regex=r'^(accept|reject|partial))
    comments: Optional[str] = None
    rating: Optional[int] = Field(None, ge=1, le=5)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

# Database models manager
class StorageModelsManager:
    """
    Manager for storage models with database operations.
    """
    
    def __init__(self, db_path: str = "edi_sessions.db"):
        self.db_path = db_path
        self._initialize_tables()
    
    def _initialize_tables(self):
        """Initialize database tables."""
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
                        final_alignment_score REAL,
                        processing_time REAL,
                        model_version TEXT,
                        session_tags TEXT,
                        session_metadata TEXT
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
                        generation_time REAL,
                        model_confidence REAL,
                        prompt_metadata TEXT,
                        generation_parameters TEXT,
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
                        area_percent REAL,
                        embedding_vector TEXT,
                        semantic_label TEXT,
                        entity_metadata TEXT,
                        quality_score REAL
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
                        processing_details TEXT,
                        quality_metrics TEXT,
                        validation_metadata TEXT,
                        user_ratings TEXT,
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
            raise Exception(f"Failed to initialize database tables: {str(e)}")
    
    def save_session(self, session_record: SessionRecord) -> bool:
        """
        Save session record to database.
        
        Args:
            session_record: Session record to save
            
        Returns:
            Boolean - True if save successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO sessions 
                    (id, created_at, image_path, naive_prompt, status, final_alignment_score,
                     processing_time, model_version, session_tags, session_metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session_record.id,
                    session_record.created_at,
                    session_record.image_path,
                    session_record.naive_prompt,
                    session_record.status,
                    session_record.final_alignment_score,
                    session_record.processing_time,
                    session_record.model_version,
                    session_record.session_tags,
                    session_record.session_metadata
                ))
                
                conn.commit()
                return True
                
        except sqlite3.Error as e:
            print(f"Failed to save session: {str(e)}")
            return False
    
    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """
        Get session record from database.
        
        Args:
            session_id: ID of session to retrieve
            
        Returns:
            Session record or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, created_at, image_path, naive_prompt, status, final_alignment_score,
                           processing_time, model_version, session_tags, session_metadata
                    FROM sessions
                    WHERE id = ?
                ''', (session_id,))
                
                row = cursor.fetchone()
                if row:
                    return SessionRecord(
                        id=row[0],
                        created_at=row[1],
                        image_path=row[2],
                        naive_prompt=row[3],
                        status=row[4],
                        final_alignment_score=row[5],
                        processing_time=row[6],
                        model_version=row[7],
                        session_tags=row[8],
                        session_metadata=row[9]
                    )
                
                return None
                
        except sqlite3.Error as e:
            print(f"Failed to get session: {str(e)}")
            return None
    
    def save_prompt(self, prompt_record: PromptRecord) -> bool:
        """
        Save prompt record to database.
        
        Args:
            prompt_record: Prompt record to save
            
        Returns:
            Boolean - True if save successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO prompts 
                    (id, session_id, iteration, positive_prompt, negative_prompt, quality_score,
                     generation_time, model_confidence, prompt_metadata, generation_parameters, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    prompt_record.id,
                    prompt_record.session_id,
                    prompt_record.iteration,
                    prompt_record.positive_prompt,
                    prompt_record.negative_prompt,
                    prompt_record.quality_score,
                    prompt_record.generation_time,
                    prompt_record.model_confidence,
                    prompt_record.prompt_metadata,
                    prompt_record.generation_parameters,
                    prompt_record.created_at
                ))
                
                conn.commit()
                return True
                
        except sqlite3.Error as e:
            print(f"Failed to save prompt: {str(e)}")
            return False
    
    def get_prompts_by_session(self, session_id: str) -> List[PromptRecord]:
        """
        Get all prompt records for a session.
        
        Args:
            session_id: ID of session
            
        Returns:
            List of prompt records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, session_id, iteration, positive_prompt, negative_prompt, quality_score,
                           generation_time, model_confidence, prompt_metadata, generation_parameters, created_at
                    FROM prompts
                    WHERE session_id = ?
                    ORDER BY iteration
                ''', (session_id,))
                
                rows = cursor.fetchall()
                return [
                    PromptRecord(
                        id=row[0],
                        session_id=row[1],
                        iteration=row[2],
                        positive_prompt=row[3],
                        negative_prompt=row[4],
                        quality_score=row[5],
                        generation_time=row[6],
                        model_confidence=row[7],
                        prompt_metadata=row[8],
                        generation_parameters=row[9],
                        created_at=row[10]
                    )
                    for row in rows
                ]
                
        except sqlite3.Error as e:
            print(f"Failed to get prompts: {str(e)}")
            return []
    
    def save_entity(self, entity_record: EntityRecord) -> bool:
        """
        Save entity record to database.
        
        Args:
            entity_record: Entity record to save
            
        Returns:
            Boolean - True if save successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO entities 
                    (id, session_id, entity_id, label, confidence, bbox_json, mask_path,
                     color_hex, area_percent, embedding_vector, semantic_label, entity_metadata, quality_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entity_record.id,
                    entity_record.session_id,
                    entity_record.entity_id,
                    entity_record.label,
                    entity_record.confidence,
                    entity_record.bbox_json,
                    entity_record.mask_path,
                    entity_record.color_hex,
                    entity_record.area_percent,
                    entity_record.embedding_vector,
                    entity_record.semantic_label,
                    entity_record.entity_metadata,
                    entity_record.quality_score
                ))
                
                conn.commit()
                return True
                
        except sqlite3.Error as e:
            print(f"Failed to save entity: {str(e)}")
            return False
    
    def get_entities_by_session(self, session_id: str) -> List[EntityRecord]:
        """
        Get all entity records for a session.
        
        Args:
            session_id: ID of session
            
        Returns:
            List of entity records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, session_id, entity_id, label, confidence, bbox_json, mask_path,
                           color_hex, area_percent, embedding_vector, semantic_label, entity_metadata, quality_score
                    FROM entities
                    WHERE session_id = ?
                ''', (session_id,))
                
                rows = cursor.fetchall()
                return [
                    EntityRecord(
                        id=row[0],
                        session_id=row[1],
                        entity_id=row[2],
                        label=row[3],
                        confidence=row[4],
                        bbox_json=row[5],
                        mask_path=row[6],
                        color_hex=row[7],
                        area_percent=row[8],
                        embedding_vector=row[9],
                        semantic_label=row[10],
                        entity_metadata=row[11],
                        quality_score=row[12]
                    )
                    for row in rows
                ]
                
        except sqlite3.Error as e:
            print(f"Failed to get entities: {str(e)}")
            return []
    
    def save_validation(self, validation_record: ValidationRecord) -> bool:
        """
        Save validation record to database.
        
        Args:
            validation_record: Validation record to save
            
        Returns:
            Boolean - True if save successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO validations 
                    (id, session_id, attempt_number, alignment_score, preserved_count, modified_count,
                     unintended_count, user_feedback, processing_details, quality_metrics, validation_metadata, user_ratings, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    validation_record.id,
                    validation_record.session_id,
                    validation_record.attempt_number,
                    validation_record.alignment_score,
                    validation_record.preserved_count,
                    validation_record.modified_count,
                    validation_record.unintended_count,
                    validation_record.user_feedback,
                    validation_record.processing_details,
                    validation_record.quality_metrics,
                    validation_record.validation_metadata,
                    validation_record.user_ratings,
                    validation_record.created_at
                ))
                
                conn.commit()
                return True
                
        except sqlite3.Error as e:
            print(f"Failed to save validation: {str(e)}")
            return False
    
    def get_validations_by_session(self, session_id: str) -> List[ValidationRecord]:
        """
        Get all validation records for a session.
        
        Args:
            session_id: ID of session
            
        Returns:
            List of validation records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, session_id, attempt_number, alignment_score, preserved_count, modified_count,
                           unintended_count, user_feedback, processing_details, quality_metrics, validation_metadata, user_ratings, created_at
                    FROM validations
                    WHERE session_id = ?
                    ORDER BY attempt_number
                ''', (session_id,))
                
                rows = cursor.fetchall()
                return [
                    ValidationRecord(
                        id=row[0],
                        session_id=row[1],
                        attempt_number=row[2],
                        alignment_score=row[3],
                        preserved_count=row[4],
                        modified_count=row[5],
                        unintended_count=row[6],
                        user_feedback=row[7],
                        processing_details=row[8],
                        quality_metrics=row[9],
                        validation_metadata=row[10],
                        user_ratings=row[11],
                        created_at=row[12]
                    )
                    for row in rows
                ]
                
        except sqlite3.Error as e:
            print(f"Failed to get validations: {str(e)}")
            return []

# Example usage
if __name__ == "__main__":
    # Initialize storage models manager
    manager = StorageModelsManager("edi_storage_test.db")
    
    print("Storage models manager initialized")
    
    # Create and save example records
    session_record = SessionRecord(
        image_path="/path/to/image.jpg",
        naive_prompt="make the sky more dramatic",
        status=SessionStatus.COMPLETED.value,
        final_alignment_score=0.85
    )
    
    # Save session
    if manager.save_session(session_record):
        print(f"Session saved: {session_record.id}")
    else:
        print("Failed to save session")
    
    # Retrieve session
    retrieved_session = manager.get_session(session_record.id)
    if retrieved_session:
        print(f"Retrieved session: {retrieved_session.naive_prompt}")
    else:
        print("Failed to retrieve session")
    
    # Create and save prompt record
    prompt_record = PromptRecord(
        session_id=session_record.id,
        iteration=0,
        positive_prompt="dramatic sky with storm clouds",
        negative_prompt="sunny sky, clear weather",
        quality_score=0.92
    )
    
    if manager.save_prompt(prompt_record):
        print("Prompt record saved")
    else:
        print("Failed to save prompt record")
    
    # Retrieve prompts
    prompts = manager.get_prompts_by_session(session_record.id)
    print(f"Retrieved {len(prompts)} prompt records")
    
    # Create and save entity record
    entity_record = EntityRecord(
        session_id=session_record.id,
        entity_id="sky_0",
        label="sky",
        confidence=0.95,
        bbox_json='{"x1": 0, "y1": 0, "x2": 1920, "y2": 768}',
        color_hex="#87CEEB",
        area_percent=39.6
    )
    
    if manager.save_entity(entity_record):
        print("Entity record saved")
    else:
        print("Failed to save entity record")
    
    # Retrieve entities
    entities = manager.get_entities_by_session(session_record.id)
    print(f"Retrieved {len(entities)} entity records")
    
    # Create and save validation record
    validation_record = ValidationRecord(
        session_id=session_record.id,
        attempt_number=1,
        alignment_score=0.85,
        preserved_count=3,
        modified_count=1,
        unintended_count=0,
        user_feedback="Great improvement to the sky!"
    )
    
    if manager.save_validation(validation_record):
        print("Validation record saved")
    else:
        print("Failed to save validation record")
    
    # Retrieve validations
    validations = manager.get_validations_by_session(session_record.id)
    print(f"Retrieved {len(validations)} validation records")
    
    print("Storage models example completed")
```

### Advanced Storage Models Implementation
Enhanced implementation with relationship management and advanced querying:

```python
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, TypeVar, Type
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pydantic import BaseModel, Field, validator, root_validator
from contextlib import contextmanager
import logging
from abc import ABC, abstractmethod

# Type variables for generic operations
T = TypeVar('T')

class StorageModelError(Exception):
    """Custom exception for storage model errors."""
    pass

class RelationshipType(Enum):
    """Enumeration for relationship types."""
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_MANY = "many_to_many"

class AdvancedStorageModel(ABC):
    """
    Abstract base class for advanced storage models.
    """
    
    def __init__(self):
        self._relationships = {}
        self._logger = logging.getLogger(__name__)
    
    @abstractmethod
    def get_table_name(self) -> str:
        """Get the database table name for this model."""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for database storage."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdvancedStorageModel':
        """Create model instance from dictionary."""
        pass
    
    def add_relationship(self, 
                        name: str, 
                        model_class: Type['AdvancedStorageModel'], 
                        relationship_type: RelationshipType,
                        foreign_key: str = None):
        """
        Add a relationship to another model.
        
        Args:
            name: Name of the relationship
            model_class: Related model class
            relationship_type: Type of relationship
            foreign_key: Foreign key field name
        """
        self._relationships[name] = {
            "model_class": model_class,
            "type": relationship_type,
            "foreign_key": foreign_key
        }
    
    def get_relationships(self) -> Dict[str, Any]:
        """Get all relationships for this model."""
        return self._relationships.copy()

@dataclass
class AdvancedSessionRecord(AdvancedStorageModel):
    """
    Advanced session record with relationships and enhanced features.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    image_path: str = ""
    naive_prompt: str = ""
    status: str = "in_progress"
    final_alignment_score: Optional[float] = None
    processing_time: Optional[float] = None
    model_version: Optional[str] = None
    session_tags: Optional[str] = None
    session_metadata: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    
    def __post_init__(self):
        """Initialize relationships after creation."""
        super().__init__()
        # Add relationships
        self.add_relationship("prompts", AdvancedPromptRecord, RelationshipType.ONE_TO_MANY, "session_id")
        self.add_relationship("entities", AdvancedEntityRecord, RelationshipType.ONE_TO_MANY, "session_id")
        self.add_relationship("validations", AdvancedValidationRecord, RelationshipType.ONE_TO_MANY, "session_id")
        self.add_relationship("feedback", AdvancedUserFeedbackRecord, RelationshipType.ONE_TO_MANY, "session_id")
    
    def get_table_name(self) -> str:
        """Get table name."""
        return "sessions"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "image_path": self.image_path,
            "naive_prompt": self.naive_prompt,
            "status": self.status,
            "final_alignment_score": self.final_alignment_score,
            "processing_time": self.processing_time,
            "model_version": self.model_version,
            "session_tags": self.session_tags,
            "session_metadata": json.dumps(self.session_metadata) if self.session_metadata else None,
            "user_id": self.user_id,
            "project_id": self.project_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdvancedSessionRecord':
        """Create from dictionary."""
        # Handle JSON metadata
        session_metadata = data.get("session_metadata")
        if isinstance(session_metadata, str):
            try:
                session_metadata = json.loads(session_metadata)
            except json.JSONDecodeError:
                session_metadata = {}
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            image_path=data.get("image_path", ""),
            naive_prompt=data.get("naive_prompt", ""),
            status=data.get("status", "in_progress"),
            final_alignment_score=data.get("final_alignment_score"),
            processing_time=data.get("processing_time"),
            model_version=data.get("model_version"),
            session_tags=data.get("session_tags"),
            session_metadata=session_metadata,
            user_id=data.get("user_id"),
            project_id=data.get("project_id")
        )
    
    def update_timestamp(self):
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now().isoformat()
    
    def add_tag(self, tag: str):
        """Add a tag to the session."""
        if not self.session_tags:
            self.session_tags = tag
        else:
            current_tags = set(self.session_tags.split(","))
            current_tags.add(tag)
            self.session_tags = ",".join(sorted(current_tags))
    
    def remove_tag(self, tag: str):
        """Remove a tag from the session."""
        if self.session_tags:
            current_tags = set(self.session_tags.split(","))
            current_tags.discard(tag)
            self.session_tags = ",".join(sorted(current_tags)) if current_tags else None
    
    def set_metadata(self, key: str, value: Any):
        """Set a metadata key-value pair."""
        if not self.session_metadata:
            self.session_metadata = {}
        self.session_metadata[key] = value

@dataclass
class AdvancedPromptRecord(AdvancedStorageModel):
    """
    Advanced prompt record with enhanced features.
    """
    id: Optional[int] = None
    session_id: str = ""
    iteration: int = 0
    positive_prompt: str = ""
    negative_prompt: str = ""
    quality_score: Optional[float] = None
    generation_time: Optional[float] = None
    model_confidence: Optional[float] = None
    prompt_metadata: Optional[Dict[str, Any]] = None
    generation_parameters: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Initialize relationships."""
        super().__init__()
    
    def get_table_name(self) -> str:
        """Get table name."""
        return "prompts"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "iteration": self.iteration,
            "positive_prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "quality_score": self.quality_score,
            "generation_time": self.generation_time,
            "model_confidence": self.model_confidence,
            "prompt_metadata": json.dumps(self.prompt_metadata) if self.prompt_metadata else None,
            "generation_parameters": json.dumps(self.generation_parameters) if self.generation_parameters else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdvancedPromptRecord':
        """Create from dictionary."""
        # Handle JSON fields
        prompt_metadata = data.get("prompt_metadata")
        if isinstance(prompt_metadata, str):
            try:
                prompt_metadata = json.loads(prompt_metadata)
            except json.JSONDecodeError:
                prompt_metadata = {}
        
        generation_parameters = data.get("generation_parameters")
        if isinstance(generation_parameters, str):
            try:
                generation_parameters = json.loads(generation_parameters)
            except json.JSONDecodeError:
                generation_parameters = {}
        
        return cls(
            id=data.get("id"),
            session_id=data.get("session_id", ""),
            iteration=data.get("iteration", 0),
            positive_prompt=data.get("positive_prompt", ""),
            negative_prompt=data.get("negative_prompt", ""),
            quality_score=data.get("quality_score"),
            generation_time=data.get("generation_time"),
            model_confidence=data.get("model_confidence"),
            prompt_metadata=prompt_metadata,
            generation_parameters=generation_parameters,
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat())
        )
    
    def update_timestamp(self):
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now().isoformat()

@dataclass
class AdvancedEntityRecord(AdvancedStorageModel):
    """
    Advanced entity record with enhanced features.
    """
    id: Optional[int] = None
    session_id: str = ""
    entity_id: str = ""
    label: str = ""
    confidence: float = 0.0
    bbox_json: str = "{}"
    mask_path: Optional[str] = None
    color_hex: Optional[str] = None
    area_percent: Optional[float] = None
    embedding_vector: Optional[str] = None
    semantic_label: Optional[str] = None
    entity_metadata: Optional[Dict[str, Any]] = None
    quality_score: Optional[float] = None
    
    def __post_init__(self):
        """Initialize relationships."""
        super().__init__()
    
    def get_table_name(self) -> str:
        """Get table name."""
        return "entities"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "entity_id": self.entity_id,
            "label": self.label,
            "confidence": self.confidence,
            "bbox_json": self.bbox_json,
            "mask_path": self.mask_path,
            "color_hex": self.color_hex,
            "area_percent": self.area_percent,
            "embedding_vector": self.embedding_vector,
            "semantic_label": self.semantic_label,
            "entity_metadata": json.dumps(self.entity_metadata) if self.entity_metadata else None,
            "quality_score": self.quality_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdvancedEntityRecord':
        """Create from dictionary."""
        # Handle JSON metadata
        entity_metadata = data.get("entity_metadata")
        if isinstance(entity_metadata, str):
            try:
                entity_metadata = json.loads(entity_metadata)
            except json.JSONDecodeError:
                entity_metadata = {}
        
        return cls(
            id=data.get("id"),
            session_id=data.get("session_id", ""),
            entity_id=data.get("entity_id", ""),
            label=data.get("label", ""),
            confidence=data.get("confidence", 0.0),
            bbox_json=data.get("bbox_json", "{}"),
            mask_path=data.get("mask_path"),
            color_hex=data.get("color_hex"),
            area_percent=data.get("area_percent"),
            embedding_vector=data.get("embedding_vector"),
            semantic_label=data.get("semantic_label"),
            entity_metadata=entity_metadata,
            quality_score=data.get("quality_score")
        )
    
    def get_bbox(self) -> Dict[str, Any]:
        """Get bounding box as dictionary."""
        try:
            return json.loads(self.bbox_json)
        except json.JSONDecodeError:
            return {}
    
    def set_bbox(self, bbox: Dict[str, Any]):
        """Set bounding box from dictionary."""
        self.bbox_json = json.dumps(bbox)

@dataclass
class AdvancedValidationRecord(AdvancedStorageModel):
    """
    Advanced validation record with enhanced features.
    """
    id: Optional[int] = None
    session_id: str = ""
    attempt_number: int = 0
    alignment_score: Optional[float] = None
    preserved_count: Optional[int] = None
    modified_count: Optional[int] = None
    unintended_count: Optional[int] = None
    user_feedback: Optional[str] = None
    processing_details: Optional[Dict[str, Any]] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    validation_metadata: Optional[Dict[str, Any]] = None
    user_ratings: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Initialize relationships."""
        super().__init__()
    
    def get_table_name(self) -> str:
        """Get table name."""
        return "validations"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "attempt_number": self.attempt_number,
            "alignment_score": self.alignment_score,
            "preserved_count": self.preserved_count,
            "modified_count": self.modified_count,
            "unintended_count": self.unintended_count,
            "user_feedback": self.user_feedback,
            "processing_details": json.dumps(self.processing_details) if self.processing_details else None,
            "quality_metrics": json.dumps(self.quality_metrics) if self.quality_metrics else None,
            "validation_metadata": json.dumps(self.validation_metadata) if self.validation_metadata else None,
            "user_ratings": json.dumps(self.user_ratings) if self.user_ratings else None,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdvancedValidationRecord':
        """Create from dictionary."""
        # Handle JSON fields
        def parse_json_field(field_data):
            if isinstance(field_data, str):
                try:
                    return json.loads(field_data)
                except json.JSONDecodeError:
                    return {}
            return field_data or {}
        
        return cls(
            id=data.get("id"),
            session_id=data.get("session_id", ""),
            attempt_number=data.get("attempt_number", 0),
            alignment_score=data.get("alignment_score"),
            preserved_count=data.get("preserved_count"),
            modified_count=data.get("modified_count"),
            unintended_count=data.get("unintended_count"),
            user_feedback=data.get("user_feedback"),
            processing_details=parse_json_field(data.get("processing_details")),
            quality_metrics=parse_json_field(data.get("quality_metrics")),
            validation_metadata=parse_json_field(data.get("validation_metadata")),
            user_ratings=parse_json_field(data.get("user_ratings")),
            created_at=data.get("created_at", datetime.now().isoformat())
        )

@dataclass
class AdvancedUserFeedbackRecord(AdvancedStorageModel):
    """
    Advanced user feedback record with enhanced features.
    """
    id: Optional[int] = None
    session_id: str = ""
    feedback_type: str = "partial"
    comments: Optional[str] = None
    rating: Optional[int] = None
    feedback_metadata: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Initialize relationships."""
        super().__init__()
    
    def get_table_name(self) -> str:
        """Get table name."""
        return "user_feedback"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "feedback_type": self.feedback_type,
            "comments": self.comments,
            "rating": self.rating,
            "feedback_metadata": json.dumps(self.feedback_metadata) if self.feedback_metadata else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdvancedUserFeedbackRecord':
        """Create from dictionary."""
        # Handle JSON metadata
        feedback_metadata = data.get("feedback_metadata")
        if isinstance(feedback_metadata, str):
            try:
                feedback_metadata = json.loads(feedback_metadata)
            except json.JSONDecodeError:
                feedback_metadata = {}
        
        return cls(
            id=data.get("id"),
            session_id=data.get("session_id", ""),
            feedback_type=data.get("feedback_type", "partial"),
            comments=data.get("comments"),
            rating=data.get("rating"),
            feedback_metadata=feedback_metadata,
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat())
        )
    
    def update_timestamp(self):
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now().isoformat()

class AdvancedStorageModelsManager:
    """
    Advanced storage models manager with relationship handling and advanced querying.
    """
    
    def __init__(self, db_path: str = "edi_advanced.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database with all tables."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Create sessions table with enhanced schema
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        image_path TEXT NOT NULL,
                        naive_prompt TEXT NOT NULL,
                        status TEXT CHECK(status IN ('in_progress', 'completed', 'failed')),
                        final_alignment_score REAL,
                        processing_time REAL,
                        model_version TEXT,
                        session_tags TEXT,
                        session_metadata TEXT,
                        user_id TEXT,
                        project_id TEXT,
                        INDEX idx_sessions_status (status),
                        INDEX idx_sessions_user (user_id),
                        INDEX idx_sessions_project (project_id),
                        INDEX idx_sessions_created (created_at)
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
                        generation_time REAL,
                        model_confidence REAL,
                        prompt_metadata TEXT,
                        generation_parameters TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_prompts_session (session_id),
                        INDEX idx_prompts_iteration (iteration)
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
                        area_percent REAL,
                        embedding_vector TEXT,
                        semantic_label TEXT,
                        entity_metadata TEXT,
                        quality_score REAL,
                        INDEX idx_entities_session (session_id),
                        INDEX idx_entities_label (label),
                        INDEX idx_entities_quality (quality_score)
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
                        processing_details TEXT,
                        quality_metrics TEXT,
                        validation_metadata TEXT,
                        user_ratings TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_validations_session (session_id),
                        INDEX idx_validations_attempt (attempt_number),
                        INDEX idx_validations_score (alignment_score)
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
                        feedback_metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_feedback_session (session_id),
                        INDEX idx_feedback_type (feedback_type),
                        INDEX idx_feedback_rating (rating)
                    )
                ''')
                
                # Create audit log table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        table_name TEXT NOT NULL,
                        record_id TEXT,
                        operation TEXT CHECK(operation IN ('INSERT', 'UPDATE', 'DELETE')),
                        user_id TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        changes TEXT
                    )
                ''')
                
                conn.commit()
                
        except sqlite3.Error as e:
            raise StorageModelError(f"Failed to initialize database: {str(e)}")
    
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
            raise StorageModelError(f"Database connection error: {str(e)}")
        finally:
            if conn:
                try:
                    conn.commit()
                except:
                    pass
                conn.close()
    
    def save_model(self, model: AdvancedStorageModel) -> bool:
        """
        Save a model instance to the database.
        
        Args:
            model: Model instance to save
            
        Returns:
            Boolean - True if save successful, False otherwise
        """
        try:
            table_name = model.get_table_name()
            data = model.to_dict()
            
            # Build INSERT/UPDATE query
            columns = list(data.keys())
            placeholders = [f":{col}" for col in columns]
            values = data
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if record exists (for UPDATE)
                if "id" in data and data["id"]:
                    cursor.execute(f'''
                        SELECT COUNT(*) FROM {table_name} WHERE id = :id
                    ''', {"id": data["id"]})
                    
                    exists = cursor.fetchone()[0] > 0
                else:
                    exists = False
                
                if exists:
                    # UPDATE existing record
                    set_clause = ", ".join([f"{col} = :{col}" for col in columns if col != "id"])
                    query = f'''
                        UPDATE {table_name} 
                        SET {set_clause}
                        WHERE id = :id
                    '''
                else:
                    # INSERT new record
                    query = f'''
                        INSERT INTO {table_name} 
                        ({", ".join(columns)})
                        VALUES ({", ".join(placeholders)})
                    '''
                
                cursor.execute(query, values)
                conn.commit()
                
                # Log audit trail
                self._log_audit(table_name, data.get("id"), "INSERT" if not exists else "UPDATE", data)
                
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to save model {table_name}: {str(e)}")
            return False
    
    def get_model(self, model_class: Type[T], record_id: Union[str, int]) -> Optional[T]:
        """
        Get a model instance by ID.
        
        Args:
            model_class: Model class to instantiate
            record_id: ID of record to retrieve
            
        Returns:
            Model instance or None if not found
        """
        try:
            table_name = model_class().get_table_name()
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute(f'''
                    SELECT * FROM {table_name} WHERE id = ?
                ''', (record_id,))
                
                row = cursor.fetchone()
                if row:
                    # Convert row to dictionary
                    data = dict(zip([desc[0] for desc in cursor.description], row))
                    return model_class.from_dict(data)
                
                return None
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get model {table_name} with ID {record_id}: {str(e)}")
            return None
    
    def delete_model(self, model_class: Type[T], record_id: Union[str, int]) -> bool:
        """
        Delete a model instance by ID.
        
        Args:
            model_class: Model class
            record_id: ID of record to delete
            
        Returns:
            Boolean - True if delete successful, False otherwise
        """
        try:
            table_name = model_class().get_table_name()
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get record before deletion for audit logging
                cursor.execute(f'''
                    SELECT * FROM {table_name} WHERE id = ?
                ''', (record_id,))
                
                row = cursor.fetchone()
                if row:
                    # Convert row to dictionary for audit logging
                    data = dict(zip([desc[0] for desc in cursor.description], row))
                
                # Delete record
                cursor.execute(f'''
                    DELETE FROM {table_name} WHERE id = ?
                ''', (record_id,))
                
                conn.commit()
                
                # Log audit trail
                if row:
                    self._log_audit(table_name, record_id, "DELETE", data)
                
                return cursor.rowcount > 0
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to delete model {table_name} with ID {record_id}: {str(e)}")
            return False
    
    def query_models(self, 
                     model_class: Type[T], 
                     filters: Dict[str, Any] = None,
                     order_by: str = None,
                     limit: int = None,
                     offset: int = 0) -> List[T]:
        """
        Query models with filters and sorting.
        
        Args:
            model_class: Model class to query
            filters: Dictionary of field-value pairs to filter by
            order_by: Field to order by (with optional DESC suffix)
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of model instances
        """
        try:
            table_name = model_class().get_table_name()
            
            # Build query
            query = f"SELECT * FROM {table_name}"
            params = []
            
            # Add WHERE clause
            if filters:
                where_conditions = []
                for field, value in filters.items():
                    if isinstance(value, (list, tuple)):
                        # Handle IN clause
                        placeholders = ",".join(["?" for _ in value])
                        where_conditions.append(f"{field} IN ({placeholders})")
                        params.extend(value)
                    else:
                        # Handle equality
                        where_conditions.append(f"{field} = ?")
                        params.append(value)
                
                if where_conditions:
                    query += " WHERE " + " AND ".join(where_conditions)
            
            # Add ORDER BY clause
            if order_by:
                query += f" ORDER BY {order_by}"
            
            # Add LIMIT and OFFSET clauses
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            if offset:
                query += " OFFSET ?"
                params.append(offset)
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                rows = cursor.fetchall()
                results = []
                
                for row in rows:
                    # Convert row to dictionary
                    data = dict(zip([desc[0] for desc in cursor.description], row))
                    results.append(model_class.from_dict(data))
                
                return results
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to query models {table_name}: {str(e)}")
            return []
    
    def get_related_models(self, 
                          parent_model: AdvancedStorageModel,
                          relationship_name: str) -> List[AdvancedStorageModel]:
        """
        Get related models for a parent model.
        
        Args:
            parent_model: Parent model instance
            relationship_name: Name of relationship
            
        Returns:
            List of related model instances
        """
        try:
            # Get relationship information
            relationships = parent_model.get_relationships()
            if relationship_name not in relationships:
                raise StorageModelError(f"Relationship '{relationship_name}' not found")
            
            relationship = relationships[relationship_name]
            related_model_class = relationship["model_class"]
            foreign_key = relationship["foreign_key"]
            
            if not foreign_key:
                raise StorageModelError(f"No foreign key defined for relationship '{relationship_name}'")
            
            # Query related models
            filters = {foreign_key: parent_model.id}
            return self.query_models(related_model_class, filters)
            
        except Exception as e:
            self.logger.error(f"Failed to get related models: {str(e)}")
            return []
    
    def _log_audit(self, table_name: str, record_id: Union[str, int], operation: str, changes: Dict[str, Any]):
        """
        Log audit trail for database operations.
        
        Args:
            table_name: Name of table
            record_id: ID of record
            operation: Type of operation (INSERT, UPDATE, DELETE)
            changes: Changes made to record
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO audit_log 
                    (table_name, record_id, operation, changes)
                    VALUES (?, ?, ?, ?)
                ''', (
                    table_name,
                    str(record_id) if record_id else None,
                    operation,
                    json.dumps(changes) if changes else None
                ))
                
                conn.commit()
                
        except sqlite3.Error:
            # Don't fail the main operation if audit logging fails
            pass
    
    def get_audit_trail(self, 
                       table_name: str = None,
                       record_id: Union[str, int] = None,
                       operation: str = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get audit trail records.
        
        Args:
            table_name: Filter by table name
            record_id: Filter by record ID
            operation: Filter by operation type
            limit: Maximum number of records to return
            
        Returns:
            List of audit trail records
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Build query
                query = "SELECT * FROM audit_log"
                params = []
                
                # Add filters
                filters = []
                if table_name:
                    filters.append("table_name = ?")
                    params.append(table_name)
                
                if record_id:
                    filters.append("record_id = ?")
                    params.append(str(record_id))
                
                if operation:
                    filters.append("operation = ?")
                    params.append(operation)
                
                if filters:
                    query += " WHERE " + " AND ".join(filters)
                
                query += " ORDER BY timestamp DESC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [
                    dict(zip([desc[0] for desc in cursor.description], row))
                    for row in rows
                ]
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get audit trail: {str(e)}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary containing database statistics
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get table counts
                tables = ["sessions", "prompts", "entities", "validations", "user_feedback"]
                table_counts = {}
                
                for table in tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        table_counts[table] = count
                    except sqlite3.Error:
                        table_counts[table] = 0
                
                # Get recent activity
                cursor.execute('''
                    SELECT COUNT(*) 
                    FROM sessions 
                    WHERE created_at >= datetime('now', '-7 days')
                ''')
                recent_sessions = cursor.fetchone()[0]
                
                # Get session status breakdown
                cursor.execute('''
                    SELECT status, COUNT(*) 
                    FROM sessions 
                    GROUP BY status
                ''')
                status_breakdown = dict(cursor.fetchall())
                
                # Get average processing time
                cursor.execute('''
                    SELECT AVG(processing_time) 
                    FROM sessions 
                    WHERE processing_time IS NOT NULL
                ''')
                avg_processing_time = cursor.fetchone()[0]
                
                return {
                    "table_counts": table_counts,
                    "recent_sessions_7_days": recent_sessions,
                    "status_breakdown": status_breakdown,
                    "average_processing_time": round(avg_processing_time, 2) if avg_processing_time else 0,
                    "last_updated": datetime.now().isoformat()
                }
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get statistics: {str(e)}")
            return {
                "table_counts": {},
                "recent_sessions_7_days": 0,
                "status_breakdown": {},
                "average_processing_time": 0,
                "last_updated": datetime.now().isoformat()
            }

# Example usage
if __name__ == "__main__":
    # Initialize advanced storage models manager
    manager = AdvancedStorageModelsManager("edi_advanced_storage.db")
    
    print("Advanced Storage Models Manager initialized")
    
    # Create and save example session
    session = AdvancedSessionRecord(
        image_path="/path/to/image.jpg",
        naive_prompt="make the sky more dramatic",
        status="completed",
        final_alignment_score=0.85,
        processing_time=120.5,
        model_version="qwen3:8b",
        session_tags="sky,clouds,dramatic",
        session_metadata={
            "image_size": "1920x1080",
            "color_profile": "sRGB",
            "processing_steps": 3
        }
    )
    
    # Add tags
    session.add_tag("outdoor")
    session.add_tag("landscape")
    
    # Save session
    if manager.save_model(session):
        print(f"Session saved: {session.id}")
    else:
        print("Failed to save session")
    
    # Create and save prompt
    prompt = AdvancedPromptRecord(
        session_id=session.id,
        iteration=0,
        positive_prompt="dramatic sky with storm clouds",
        negative_prompt="sunny sky, clear weather",
        quality_score=0.92,
        generation_time=15.2,
        model_confidence=0.88,
        generation_parameters={
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 0.9
        }
    )
    
    if manager.save_model(prompt):
        print("Prompt saved")
    else:
        print("Failed to save prompt")
    
    # Create and save entity
    entity = AdvancedEntityRecord(
        session_id=session.id,
        entity_id="sky_0",
        label="sky",
        confidence=0.95,
        bbox_json='{"x1": 0, "y1": 0, "x2": 1920, "y2": 768}',
        color_hex="#87CEEB",
        area_percent=39.6,
        entity_metadata={
            "texture": "cloudy",
            "lighting": "dramatic",
            "movement": "dynamic"
        }
    )
    
    if manager.save_model(entity):
        print("Entity saved")
    else:
        print("Failed to save entity")
    
    # Create and save validation
    validation = AdvancedValidationRecord(
        session_id=session.id,
        attempt_number=1,
        alignment_score=0.85,
        preserved_count=3,
        modified_count=1,
        unintended_count=0,
        user_feedback="Great improvement to the sky!",
        quality_metrics={
            "sharpness": 0.88,
            "color_accuracy": 0.92,
            "composition": 0.85
        }
    )
    
    if manager.save_model(validation):
        print("Validation saved")
    else:
        print("Failed to save validation")
    
    # Query sessions
    sessions = manager.query_models(
        AdvancedSessionRecord,
        filters={"status": "completed"},
        order_by="created_at DESC",
        limit=10
    )
    print(f"Found {len(sessions)} completed sessions")
    
    # Get related models
    if sessions:
        related_prompts = manager.get_related_models(sessions[0], "prompts")
        related_entities = manager.get_related_models(sessions[0], "entities")
        print(f"Session has {len(related_prompts)} prompts and {len(related_entities)} entities")
    
    # Get audit trail
    audit_trail = manager.get_audit_trail(limit=10)
    print(f"Audit trail has {len(audit_trail)} recent entries")
    
    # Get statistics
    stats = manager.get_statistics()
    print("Database Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("Advanced storage models example completed")
```