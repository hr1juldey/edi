# Orchestration: State Manager

[Back to Orchestrator](./orchestrator.md)

## Purpose
Session state tracking - Contains the StateManager class that saves and loads session state to JSON files, with auto-save functionality.

## Class: StateManager

### Methods
- `save_state()`: Saves current session state to JSON
- `load_state()`: Loads session state from JSON
- `checkpoint()`: Creates a checkpoint of the current state

### Details
- Writes JSON to ~/.edi/sessions/<session_id>.json
- Auto-saves every 5 seconds
- Maintains session continuity across interruptions

## Functions

- [save_state()](./orchestration/save_state.md)
- [load_state()](./orchestration/load_state.md)
- [checkpoint()](./orchestration/checkpoint.md)

## Technology Stack

- JSON for serialization
- File I/O operations
- Time utilities for auto-save

## See Docs

### Python Implementation Example
Orchestration state manager implementation with auto-save functionality:

```python
import json
import os
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import tempfile
import shutil

class StateManagerError(Exception):
    """Custom exception for state manager errors."""
    pass

class StateStatus(Enum):
    """Enumeration for state statuses."""
    ACTIVE = "active"
    SAVED = "saved"
    CHECKPOINTED = "checkpointed"
    ERROR = "error"

@dataclass
class SessionState:
    """
    Represents the complete state of a session.
    """
    session_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    image_path: str = ""
    naive_prompt: str = ""
    status: str = "active"
    current_iteration: int = 0
    prompt_history: list = None
    entity_detections: list = None
    validation_results: list = None
    user_feedback: Optional[Dict[str, Any]] = None
    processing_metrics: Optional[Dict[str, Any]] = None
    model_configs: Optional[Dict[str, Any]] = None
    checkpoints: Optional[Dict[str, str]] = None
    temp_files: Optional[list] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize mutable default arguments."""
        if self.prompt_history is None:
            self.prompt_history = []
        if self.entity_detections is None:
            self.entity_detections = []
        if self.validation_results is None:
            self.validation_results = []
        if self.checkpoints is None:
            self.checkpoints = {}
        if self.temp_files is None:
            self.temp_files = []
        if self.created_at == "":
            self.created_at = datetime.now().isoformat()
        if self.updated_at == "":
            self.updated_at = datetime.now().isoformat()

class StateManager:
    """
    Session state tracking with auto-save functionality.
    """
    
    def __init__(self, 
                 base_path: str = "~/.edi/sessions",
                 auto_save_interval: float = 5.0):
        self.base_path = Path(os.path.expanduser(base_path))
        self.auto_save_interval = auto_save_interval
        self.logger = logging.getLogger(__name__)
        self._current_state: Optional[SessionState] = None
        self._auto_save_timer: Optional[threading.Timer] = None
        self._auto_save_enabled = False
        self._lock = threading.RLock()
        
        # Ensure base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def __del__(self):
        """Cleanup resources when manager is destroyed."""
        self.stop_auto_save()
    
    def save_state(self, 
                   state: SessionState,
                   session_id: Optional[str] = None,
                   create_backup: bool = True) -> bool:
        """
        Saves current session state to JSON.
        
        Args:
            state: SessionState object to save
            session_id: Optional session ID (uses state.session_id if None)
            create_backup: Whether to create backup of existing state file
            
        Returns:
            Boolean indicating success (True) or failure (False)
        """
        try:
            # Get the session ID
            actual_session_id = session_id or state.session_id
            if not actual_session_id:
                raise StateManagerError("Session ID is required to save state")
            
            # Update timestamp
            state.updated_at = datetime.now().isoformat()
            
            # Serialize state to dictionary
            state_dict = asdict(state)
            
            # Determine file paths
            state_file_path = self.base_path / f"{actual_session_id}.json"
            backup_file_path = self.base_path / f"{actual_session_id}.json.backup"
            
            # Create backup of existing state file if requested
            if create_backup and state_file_path.exists():
                shutil.copy2(state_file_path, backup_file_path)
                self.logger.debug(f"Backup created: {backup_file_path}")
            
            # Write state to temporary file first for atomic write
            temp_fd, temp_path = tempfile.mkstemp(
                suffix=".json",
                prefix=f"{actual_session_id}_",
                dir=self.base_path
            )
            
            try:
                # Write to temp file
                with os.fdopen(temp_fd, 'w') as temp_file:
                    json.dump(state_dict, temp_file, indent=2, default=str)
                
                # Atomically move temp file to final location
                shutil.move(temp_path, state_file_path)
                self.logger.info(f"State saved successfully: {state_file_path}")
                
                # Update current state reference
                with self._lock:
                    self._current_state = state
                
                return True
                
            except Exception as e:
                # Clean up temp file if something went wrong
                try:
                    os.unlink(temp_path)
                except:
                    pass
                raise e
                
        except Exception as e:
            self.logger.error(f"Failed to save state for session {actual_session_id}: {str(e)}")
            return False
    
    def load_state(self, session_id: str) -> Optional[SessionState]:
        """
        Loads session state from JSON.
        
        Args:
            session_id: ID of session to load
            
        Returns:
            SessionState object or None if not found
        """
        try:
            state_file_path = self.base_path / f"{session_id}.json"
            
            # Check if state file exists
            if not state_file_path.exists():
                self.logger.warning(f"State file not found: {state_file_path}")
                return None
            
            # Load and parse JSON
            with open(state_file_path, 'r') as f:
                state_dict = json.load(f)
            
            # Validate required fields
            required_fields = ["session_id", "created_at", "image_path", "naive_prompt"]
            for field in required_fields:
                if field not in state_dict:
                    self.logger.warning(f"Missing required field '{field}' in state file")
                    # Set default values for missing fields
                    if field == "session_id":
                        state_dict[field] = session_id
                    elif field == "created_at":
                        state_dict[field] = datetime.now().isoformat()
                    elif field in ["image_path", "naive_prompt"]:
                        state_dict[field] = ""
            
            # Convert lists to proper types if needed
            list_fields = ["prompt_history", "entity_detections", "validation_results", "temp_files"]
            for field in list_fields:
                if field not in state_dict or state_dict[field] is None:
                    state_dict[field] = []
                elif not isinstance(state_dict[field], list):
                    state_dict[field] = list(state_dict[field])
            
            # Convert dictionaries
            dict_fields = ["user_feedback", "processing_metrics", "model_configs", "checkpoints", "metadata"]
            for field in dict_fields:
                if field not in state_dict or state_dict[field] is None:
                    state_dict[field] = {}
                elif not isinstance(state_dict[field], dict):
                    # Try to convert to dict
                    try:
                        state_dict[field] = dict(state_dict[field])
                    except:
                        state_dict[field] = {}
            
            # Create SessionState object
            state = SessionState(**state_dict)
            
            # Update current state reference
            with self._lock:
                self._current_state = state
            
            self.logger.info(f"State loaded successfully: {session_id}")
            return state
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in state file for session {session_id}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load state for session {session_id}: {str(e)}")
            return None
    
    def checkpoint(self, 
                   state: Optional[SessionState] = None,
                   checkpoint_name: Optional[str] = None) -> str:
        """
        Creates a checkpoint of the current state.
        
        Args:
            state: SessionState to checkpoint (uses current state if None)
            checkpoint_name: Optional name for checkpoint (generates if None)
            
        Returns:
            Checkpoint name (timestamp or provided name)
        """
        try:
            # Get state to checkpoint
            state_to_checkpoint = state or self._current_state
            if not state_to_checkpoint:
                raise StateManagerError("No state available to checkpoint")
            
            # Generate checkpoint name if not provided
            if not checkpoint_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_name = f"checkpoint_{timestamp}"
            
            # Update state with checkpoint information
            if state_to_checkpoint.checkpoints is None:
                state_to_checkpoint.checkpoints = {}
            
            state_to_checkpoint.checkpoints[checkpoint_name] = datetime.now().isoformat()
            state_to_checkpoint.updated_at = datetime.now().isoformat()
            
            # Save checkpoint to separate file
            checkpoint_file_path = self.base_path / f"{state_to_checkpoint.session_id}_{checkpoint_name}.json"
            
            # Serialize state
            state_dict = asdict(state_to_checkpoint)
            
            # Write checkpoint
            with open(checkpoint_file_path, 'w') as f:
                json.dump(state_dict, f, indent=2, default=str)
            
            self.logger.info(f"Checkpoint created: {checkpoint_name} for session {state_to_checkpoint.session_id}")
            
            # Update current state if we're using it
            if state_to_checkpoint == self._current_state:
                with self._lock:
                    self._current_state = state_to_checkpoint
            
            return checkpoint_name
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {str(e)}")
            raise StateManagerError(f"Checkpoint failed: {str(e)}")
    
    def start_auto_save(self, state: SessionState, interval: Optional[float] = None) -> bool:
        """
        Starts auto-save functionality.
        
        Args:
            state: SessionState to auto-save
            interval: Override auto-save interval (uses default if None)
            
        Returns:
            Boolean indicating success
        """
        try:
            with self._lock:
                # Stop any existing auto-save
                self.stop_auto_save()
                
                # Set current state
                self._current_state = state
                
                # Start auto-save timer
                save_interval = interval or self.auto_save_interval
                self._schedule_auto_save(save_interval)
                self._auto_save_enabled = True
                
                self.logger.info(f"Auto-save started for session {state.session_id} (interval: {save_interval}s)")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to start auto-save: {str(e)}")
            return False
    
    def stop_auto_save(self) -> bool:
        """
        Stops auto-save functionality.
        
        Returns:
            Boolean indicating success
        """
        try:
            with self._lock:
                if self._auto_save_timer:
                    self._auto_save_timer.cancel()
                    self._auto_save_timer = None
                
                self._auto_save_enabled = False
                self.logger.info("Auto-save stopped")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to stop auto-save: {str(e)}")
            return False
    
    def _schedule_auto_save(self, interval: float):
        """
        Schedule next auto-save.
        
        Args:
            interval: Time until next save
        """
        def _auto_save_callback():
            """Callback for auto-save timer."""
            try:
                with self._lock:
                    if self._current_state and self._auto_save_enabled:
                        # Save current state
                        success = self.save_state(self._current_state)
                        
                        if success:
                            self.logger.debug(f"Auto-save successful for session {self._current_state.session_id}")
                        else:
                            self.logger.warning(f"Auto-save failed for session {self._current_state.session_id}")
                        
                        # Schedule next auto-save if still enabled
                        if self._auto_save_enabled:
                            self._schedule_auto_save(interval)
                            
            except Exception as e:
                self.logger.error(f"Auto-save callback failed: {str(e)}")
        
        # Create and start timer
        self._auto_save_timer = threading.Timer(interval, _auto_save_callback)
        self._auto_save_timer.daemon = True
        self._auto_save_timer.start()
    
    def get_session_ids(self) -> list:
        """
        Gets list of all available session IDs.
        
        Returns:
            List of session IDs
        """
        try:
            session_ids = []
            
            # Look for session files
            for file_path in self.base_path.glob("*.json"):
                if not file_path.name.endswith(".backup"):
                    # Extract session ID from filename
                    session_id = file_path.stem
                    if "_checkpoint_" in session_id:
                        # This is a checkpoint file, extract actual session ID
                        session_id = session_id.split("_checkpoint_")[0]
                    session_ids.append(session_id)
            
            # Remove duplicates and sort
            return sorted(list(set(session_ids)))
            
        except Exception as e:
            self.logger.error(f"Failed to get session IDs: {str(e)}")
            return []
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Gets information about a session without loading full state.
        
        Args:
            session_id: ID of session
            
        Returns:
            Dictionary with session information or None if not found
        """
        try:
            state_file_path = self.base_path / f"{session_id}.json"
            
            if not state_file_path.exists():
                return None
            
            # Read only the metadata portion of the file
            with open(state_file_path, 'r') as f:
                # Read first part of file to get basic info
                content = f.read(1024)  # Read first 1KB
                try:
                    # Try to parse as JSON
                    data = json.loads(content)
                except json.JSONDecodeError:
                    # If first part isn't valid JSON, read entire file
                    f.seek(0)
                    data = json.load(f)
            
            return {
                "session_id": data.get("session_id", session_id),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "image_path": data.get("image_path", ""),
                "naive_prompt": data.get("naive_prompt", ""),
                "status": data.get("status", "unknown"),
                "current_iteration": data.get("current_iteration", 0),
                "prompt_count": len(data.get("prompt_history", [])),
                "entity_count": len(data.get("entity_detections", [])),
                "validation_count": len(data.get("validation_results", [])),
                "file_size": state_file_path.stat().st_size if state_file_path.exists() else 0,
                "checkpoints": list(data.get("checkpoints", {}).keys()) if isinstance(data.get("checkpoints"), dict) else []
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get session info for {session_id}: {str(e)}")
            return None
    
    def delete_session(self, session_id: str, delete_checkpoints: bool = True) -> bool:
        """
        Deletes a session and optionally its checkpoints.
        
        Args:
            session_id: ID of session to delete
            delete_checkpoints: Whether to delete associated checkpoints
            
        Returns:
            Boolean indicating success
        """
        try:
            # Delete main session file
            state_file_path = self.base_path / f"{session_id}.json"
            if state_file_path.exists():
                state_file_path.unlink()
                self.logger.info(f"Deleted session file: {state_file_path}")
            
            # Delete backup file
            backup_file_path = self.base_path / f"{session_id}.json.backup"
            if backup_file_path.exists():
                backup_file_path.unlink()
                self.logger.info(f"Deleted backup file: {backup_file_path}")
            
            # Delete checkpoints if requested
            if delete_checkpoints:
                for checkpoint_file in self.base_path.glob(f"{session_id}_checkpoint_*.json"):
                    checkpoint_file.unlink()
                    self.logger.info(f"Deleted checkpoint: {checkpoint_file}")
            
            # Clear from current state if it's the current session
            with self._lock:
                if self._current_state and self._current_state.session_id == session_id:
                    self._current_state = None
            
            self.logger.info(f"Deleted session: {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete session {session_id}: {str(e)}")
            return False
    
    def cleanup_old_sessions(self, days_old: int = 30, keep_backup: bool = True) -> int:
        """
        Cleans up old session files.
        
        Args:
            days_old: Age in days for sessions to be considered old
            keep_backup: Whether to keep backup files
            
        Returns:
            Number of sessions deleted
        """
        try:
            cutoff_time = time.time() - (days_old * 24 * 60 * 60)  # Seconds in n days
            deleted_count = 0
            
            for session_file in self.base_path.glob("*.json"):
                # Skip backup files if keeping backups
                if keep_backup and session_file.name.endswith(".backup"):
                    continue
                
                # Skip checkpoint files if not deleting them
                if "_checkpoint_" in session_file.name:
                    continue
                
                # Check file age
                if session_file.stat().st_mtime < cutoff_time:
                    session_file.unlink()
                    self.logger.info(f"Deleted old session file: {session_file}")
                    deleted_count += 1
            
            self.logger.info(f"Cleaned up {deleted_count} old session files")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old sessions: {str(e)}")
            return 0
    
    def export_session(self, session_id: str, export_path: str) -> bool:
        """
        Exports a session to a specified location.
        
        Args:
            session_id: ID of session to export
            export_path: Destination path for export
            
        Returns:
            Boolean indicating success
        """
        try:
            state_file_path = self.base_path / f"{session_id}.json"
            
            if not state_file_path.exists():
                self.logger.error(f"Session file not found: {state_file_path}")
                return False
            
            # Copy session file
            shutil.copy2(state_file_path, export_path)
            self.logger.info(f"Session exported to: {export_path}")
            
            # Also export checkpoints if they exist
            export_dir = Path(export_path).parent
            for checkpoint_file in self.base_path.glob(f"{session_id}_checkpoint_*.json"):
                export_checkpoint_path = export_dir / checkpoint_file.name
                shutil.copy2(checkpoint_file, export_checkpoint_path)
                self.logger.info(f"Checkpoint exported to: {export_checkpoint_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export session {session_id}: {str(e)}")
            return False
    
    def import_session(self, import_path: str, new_session_id: Optional[str] = None) -> Optional[str]:
        """
        Imports a session from a specified location.
        
        Args:
            import_path: Path to session file to import
            new_session_id: Optional new session ID (generates if None)
            
        Returns:
            Imported session ID or None if failed
        """
        try:
            import_file = Path(import_path)
            
            if not import_file.exists():
                self.logger.error(f"Import file not found: {import_path}")
                return None
            
            # Generate new session ID if needed
            if new_session_id is None:
                new_session_id = f"imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
            
            # Load and validate imported file
            with open(import_file, 'r') as f:
                state_dict = json.load(f)
            
            # Update session ID
            state_dict["session_id"] = new_session_id
            state_dict["updated_at"] = datetime.now().isoformat()
            
            # Save imported session
            new_file_path = self.base_path / f"{new_session_id}.json"
            with open(new_file_path, 'w') as f:
                json.dump(state_dict, f, indent=2, default=str)
            
            self.logger.info(f"Session imported as {new_session_id}: {new_file_path}")
            return new_session_id
            
        except Exception as e:
            self.logger.error(f"Failed to import session from {import_path}: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    # Initialize state manager
    state_manager = StateManager(
        base_path="~/.edi/test_sessions",
        auto_save_interval=3.0  # 3 second auto-save for testing
    )
    
    print("State Manager initialized")
    
    # Create example session state
    session_state = SessionState(
        session_id="test_session_123",
        image_path="/path/to/test/image.jpg",
        naive_prompt="make the sky more dramatic",
        status="active",
        current_iteration=2,
        prompt_history=[
            {
                "iteration": 0,
                "positive_prompt": "dramatic sky with storm clouds",
                "negative_prompt": "sunny sky, clear weather",
                "quality_score": 0.92,
                "generated_at": datetime.now().isoformat()
            },
            {
                "iteration": 1,
                "positive_prompt": "storm clouds with lighting and dark atmosphere",
                "negative_prompt": "sunny sky, clear weather, no clouds",
                "quality_score": 0.88,
                "generated_at": datetime.now().isoformat()
            }
        ],
        entity_detections=[
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
        validation_results=[
            {
                "attempt_number": 1,
                "alignment_score": 0.85,
                "preserved_count": 3,
                "modified_count": 1,
                "unintended_count": 0,
                "user_feedback": "Great improvement to the sky!",
                "validated_at": datetime.now().isoformat()
            }
        ],
        user_feedback={
            "feedback_type": "partial",
            "comments": "Good start, but could be more dramatic",
            "rating": 4
        },
        processing_metrics={
            "total_processing_time": 120.5,
            "model_inference_time": 45.2,
            "validation_time": 15.3
        },
        model_configs={
            "vision_model": "sam2",
            "diffusion_model": "qwen3:8b",
            "refinement_iterations": 3
        }
    )
    
    # Save state
    if state_manager.save_state(session_state):
        print(f"Session state saved: {session_state.session_id}")
    else:
        print("Failed to save session state")
    
    # Load state
    loaded_state = state_manager.load_state(session_state.session_id)
    if loaded_state:
        print(f"Session state loaded: {loaded_state.naive_prompt}")
    else:
        print("Failed to load session state")
    
    # Start auto-save
    if state_manager.start_auto_save(session_state):
        print("Auto-save started")
    else:
        print("Failed to start auto-save")
    
    # Create checkpoint
    checkpoint_name = state_manager.checkpoint(session_state, "demo_checkpoint")
    print(f"Checkpoint created: {checkpoint_name}")
    
    # Update state and wait for auto-save
    session_state.current_iteration = 3
    session_state.prompt_history.append({
        "iteration": 2,
        "positive_prompt": "extremely dramatic sky with thunder and lighting",
        "negative_prompt": "sunny sky, clear weather, no clouds, subtle changes",
        "quality_score": 0.95,
        "generated_at": datetime.now().isoformat()
    })
    
    print("State updated, waiting for auto-save...")
    time.sleep(4)  # Wait for auto-save to happen
    
    # Stop auto-save
    if state_manager.stop_auto_save():
        print("Auto-save stopped")
    else:
        print("Failed to stop auto-save")
    
    # Get session info
    session_info = state_manager.get_session_info(session_state.session_id)
    if session_info:
        print(f"Session info: {session_info['prompt_count']} prompts")
    else:
        print("Failed to get session info")
    
    # Get all session IDs
    session_ids = state_manager.get_session_ids()
    print(f"Available sessions: {session_ids}")
    
    # Export session
    export_path = "/tmp/exported_session.json"
    if state_manager.export_session(session_state.session_id, export_path):
        print(f"Session exported to: {export_path}")
    else:
        print("Failed to export session")
    
    # Delete session
    if state_manager.delete_session(session_state.session_id, delete_checkpoints=True):
        print("Session deleted")
    else:
        print("Failed to delete session")
    
    print("State manager example completed")
```

### Advanced State Management Implementation
Enhanced implementation with encryption and compression support:

```python
import json
import os
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import tempfile
import shutil
import gzip
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import hashlib

class AdvancedStateManagerError(Exception):
    """Custom exception for advanced state manager errors."""
    pass

@dataclass
class AdvancedSessionState:
    """
    Advanced session state with enhanced features.
    """
    session_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    image_path: str = ""
    naive_prompt: str = ""
    status: str = "active"
    current_iteration: int = 0
    prompt_history: list = None
    entity_detections: list = None
    validation_results: list = None
    user_feedback: Optional[Dict[str, Any]] = None
    processing_metrics: Optional[Dict[str, Any]] = None
    model_configs: Optional[Dict[str, Any]] = None
    checkpoints: Optional[Dict[str, str]] = None
    temp_files: Optional[list] = None
    metadata: Optional[Dict[str, Any]] = None
    version: str = "1.0"
    checksum: str = ""
    compressed: bool = False
    encrypted: bool = False
    
    def __post_init__(self):
        """Initialize mutable default arguments."""
        if self.prompt_history is None:
            self.prompt_history = []
        if self.entity_detections is None:
            self.entity_detections = []
        if self.validation_results is None:
            self.validation_results = []
        if self.checkpoints is None:
            self.checkpoints = {}
        if self.temp_files is None:
            self.temp_files = []
        if self.created_at == "":
            self.created_at = datetime.now().isoformat()
        if self.updated_at == "":
            self.updated_at = datetime.now().isoformat()

class SecurityConfig:
    """
    Security configuration for state encryption.
    """
    
    def __init__(self,
                 encrypt_states: bool = False,
                 encryption_password: Optional[str] = None,
                 compress_states: bool = False,
                 compression_level: int = 6):
        self.encrypt_states = encrypt_states
        self.encryption_password = encryption_password
        self.compress_states = compress_states
        self.compression_level = compression_level
        
        if encrypt_states and not encryption_password:
            raise ValueError("Encryption password required when encryption is enabled")
    
    def derive_key(self, salt: bytes) -> bytes:
        """
        Derive encryption key from password.
        
        Args:
            salt: Salt for key derivation
            
        Returns:
            Derived key
        """
        if not self.encryption_password:
            raise ValueError("No encryption password set")
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.encryption_password.encode()))
        return key

class AdvancedStateManager:
    """
    Advanced state manager with security, compression, and performance features.
    """
    
    def __init__(self, 
                 base_path: str = "~/.edi/sessions",
                 auto_save_interval: float = 5.0,
                 security_config: Optional[SecurityConfig] = None,
                 max_file_size_mb: int = 100):
        self.base_path = Path(os.path.expanduser(base_path))
        self.auto_save_interval = auto_save_interval
        self.security_config = security_config or SecurityConfig()
        self.max_file_size_mb = max_file_size_mb
        self.logger = logging.getLogger(__name__)
        self._current_state: Optional[AdvancedSessionState] = None
        self._auto_save_timer: Optional[threading.Timer] = None
        self._auto_save_enabled = False
        self._lock = threading.RLock()
        
        # Ensure base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize security components
        if self.security_config.encrypt_states:
            self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize encryption components."""
        try:
            # Generate salt for key derivation
            self._salt = os.urandom(16)
            
            # Derive encryption key
            self._key = self.security_config.derive_key(self._salt)
            self._cipher_suite = Fernet(self._key)
            
        except Exception as e:
            raise AdvancedStateManagerError(f"Failed to initialize encryption: {str(e)}")
    
    def _calculate_checksum(self, data: str) -> str:
        """
        Calculate checksum for data integrity validation.
        
        Args:
            data: Data to calculate checksum for
            
        Returns:
            Checksum string
        """
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _compress_data(self, data: str) -> bytes:
        """
        Compress data using gzip.
        
        Args:
            data: Data to compress
            
        Returns:
            Compressed data
        """
        return gzip.compress(data.encode('utf-8'), 
                           compresslevel=self.security_config.compression_level)
    
    def _decompress_data(self, compressed_data: bytes) -> str:
        """
        Decompress data using gzip.
        
        Args:
            compressed_data: Data to decompress
            
        Returns:
            Decompressed data as string
        """
        return gzip.decompress(compressed_data).decode('utf-8')
    
    def _encrypt_data(self, data: str) -> bytes:
        """
        Encrypt data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data
        """
        if not self.security_config.encrypt_states:
            return data.encode('utf-8')
        
        return self._cipher_suite.encrypt(data.encode())
    
    def _decrypt_data(self, encrypted_data: bytes) -> str:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Data to decrypt
            
        Returns:
            Decrypted data as string
        """
        if not self.security_config.encrypt_states:
            return encrypted_data.decode('utf-8')
        
        return self._cipher_suite.decrypt(encrypted_data).decode()
    
    def save_state_advanced(self, 
                            state: AdvancedSessionState,
                            session_id: Optional[str] = None,
                            create_backup: bool = True,
                            validate_integrity: bool = True) -> bool:
        """
        Saves current session state with advanced features.
        
        Args:
            state: AdvancedSessionState object to save
            session_id: Optional session ID (uses state.session_id if None)
            create_backup: Whether to create backup of existing state file
            validate_integrity: Whether to validate data integrity
            
        Returns:
            Boolean indicating success (True) or failure (False)
        """
        try:
            # Get the session ID
            actual_session_id = session_id or state.session_id
            if not actual_session_id:
                raise AdvancedStateManagerError("Session ID is required to save state")
            
            # Update timestamps and version
            state.updated_at = datetime.now().isoformat()
            state.version = "1.0"
            
            # Calculate checksum if requested
            if validate_integrity:
                # Convert to dict for checksum calculation
                state_dict = asdict(state)
                # Remove checksum field for calculation
                state_dict.pop('checksum', None)
                data_for_checksum = json.dumps(state_dict, default=str, sort_keys=True)
                state.checksum = self._calculate_checksum(data_for_checksum)
            
            # Determine file paths
            state_file_path = self.base_path / f"{actual_session_id}.json"
            backup_file_path = self.base_path / f"{actual_session_id}.json.backup"
            
            # Create backup of existing state file if requested
            if create_backup and state_file_path.exists():
                shutil.copy2(state_file_path, backup_file_path)
                self.logger.debug(f"Backup created: {backup_file_path}")
            
            # Serialize state to dictionary
            state_dict = asdict(state)
            
            # Convert to JSON string
            json_data = json.dumps(state_dict, indent=2, default=str)
            
            # Apply compression if requested
            if self.security_config.compress_states:
                state.compressed = True
                compressed_data = self._compress_data(json_data)
                
                # Apply encryption if requested
                if self.security_config.encrypt_states:
                    state.encrypted = True
                    encrypted_data = self._encrypt_data(compressed_data.decode('latin1'))
                    
                    # Write encrypted data
                    with open(state_file_path, 'wb') as f:
                        f.write(encrypted_data)
                else:
                    # Write compressed data
                    with open(state_file_path, 'wb') as f:
                        f.write(compressed_data)
            else:
                # Apply encryption if requested (without compression)
                if self.security_config.encrypt_states:
                    state.encrypted = True
                    encrypted_data = self._encrypt_data(json_data)
                    
                    # Write encrypted data
                    with open(state_file_path, 'wb') as f:
                        f.write(encrypted_data)
                else:
                    # Write uncompressed JSON
                    with open(state_file_path, 'w') as f:
                        f.write(json_data)
            
            # Check file size
            file_size_mb = state_file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                self.logger.warning(f"State file size ({file_size_mb:.1f}MB) exceeds limit ({self.max_file_size_mb}MB)")
            
            self.logger.info(f"Advanced state saved successfully: {state_file_path}")
            
            # Update current state reference
            with self._lock:
                self._current_state = state
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save advanced state for session {actual_session_id}: {str(e)}")
            return False
    
    def load_state_advanced(self, session_id: str) -> Optional[AdvancedSessionState]:
        """
        Loads session state with advanced features.
        
        Args:
            session_id: ID of session to load
            
        Returns:
            AdvancedSessionState object or None if not found
        """
        try:
            state_file_path = self.base_path / f"{session_id}.json"
            
            # Check if state file exists
            if not state_file_path.exists():
                self.logger.warning(f"State file not found: {state_file_path}")
                return None
            
            # Read file data
            if self.security_config.encrypt_states:
                with open(state_file_path, 'rb') as f:
                    file_data = f.read()
                
                # Decrypt data
                decrypted_data = self._decrypt_data(file_data)
                
                if self.security_config.compress_states:
                    # Decompress decrypted data
                    json_data = self._decompress_data(decrypted_data.encode('latin1'))
                else:
                    json_data = decrypted_data
            elif self.security_config.compress_states:
                with open(state_file_path, 'rb') as f:
                    compressed_data = f.read()
                
                # Decompress data
                json_data = self._decompress_data(compressed_data)
            else:
                # Read uncompressed JSON
                with open(state_file_path, 'r') as f:
                    json_data = f.read()
            
            # Parse JSON
            state_dict = json.loads(json_data)
            
            # Validate checksum if present
            stored_checksum = state_dict.get('checksum', '')
            if stored_checksum:
                # Remove checksum field for validation
                validation_dict = state_dict.copy()
                validation_dict.pop('checksum', None)
                validation_data = json.dumps(validation_dict, sort_keys=True, default=str)
                calculated_checksum = self._calculate_checksum(validation_data)
                
                if stored_checksum != calculated_checksum:
                    self.logger.warning(f"Checksum mismatch for session {session_id}")
                else:
                    self.logger.debug(f"Checksum validated for session {session_id}")
            
            # Convert to AdvancedSessionState
            state = AdvancedSessionState(**state_dict)
            
            # Update current state reference
            with self._lock:
                self._current_state = state
            
            self.logger.info(f"Advanced state loaded successfully: {session_id}")
            return state
            
        except Exception as e:
            self.logger.error(f"Failed to load advanced state for session {session_id}: {str(e)}")
            return None
    
    def checkpoint_advanced(self, 
                            state: Optional[AdvancedSessionState] = None,
                            checkpoint_name: Optional[str] = None,
                            compress: Optional[bool] = None,
                            encrypt: Optional[bool] = None) -> str:
        """
        Creates an advanced checkpoint with optional compression and encryption.
        
        Args:
            state: AdvancedSessionState to checkpoint (uses current state if None)
            checkpoint_name: Optional name for checkpoint (generates if None)
            compress: Override compression setting (uses default if None)
            encrypt: Override encryption setting (uses default if None)
            
        Returns:
            Checkpoint name (timestamp or provided name)
        """
        try:
            # Get state to checkpoint
            state_to_checkpoint = state or self._current_state
            if not state_to_checkpoint:
                raise AdvancedStateManagerError("No state available to checkpoint")
            
            # Generate checkpoint name if not provided
            if not checkpoint_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_name = f"checkpoint_{timestamp}"
            
            # Update state with checkpoint information
            if state_to_checkpoint.checkpoints is None:
                state_to_checkpoint.checkpoints = {}
            
            state_to_checkpoint.checkpoints[checkpoint_name] = datetime.now().isoformat()
            state_to_checkpoint.updated_at = datetime.now().isoformat()
            
            # Save checkpoint to separate file
            checkpoint_file_path = self.base_path / f"{state_to_checkpoint.session_id}_{checkpoint_name}.json"
            
            # Temporarily override compression and encryption settings if specified
            original_compress = self.security_config.compress_states
            original_encrypt = self.security_config.encrypt_states
            
            if compress is not None:
                self.security_config.compress_states = compress
            
            if encrypt is not None:
                self.security_config.encrypt_states = encrypt
            
            try:
                # Save checkpoint with current settings
                success = self.save_state_advanced(
                    state_to_checkpoint,
                    create_backup=False,  # No backup for checkpoints
                    validate_integrity=True
                )
                
                if success:
                    self.logger.info(f"Advanced checkpoint created: {checkpoint_name} for session {state_to_checkpoint.session_id}")
                    return checkpoint_name
                else:
                    raise AdvancedStateManagerError("Failed to save checkpoint")
                    
            finally:
                # Restore original settings
                self.security_config.compress_states = original_compress
                self.security_config.encrypt_states = original_encrypt
                
        except Exception as e:
            self.logger.error(f"Failed to create advanced checkpoint: {str(e)}")
            raise AdvancedStateManagerError(f"Advanced checkpoint failed: {str(e)}")
    
    def start_auto_save_advanced(self, 
                                 state: AdvancedSessionState, 
                                 interval: Optional[float] = None,
                                 validate_on_save: bool = True) -> bool:
        """
        Starts advanced auto-save functionality.
        
        Args:
            state: AdvancedSessionState to auto-save
            interval: Override auto-save interval (uses default if None)
            validate_on_save: Whether to validate integrity on each save
            
        Returns:
            Boolean indicating success
        """
        try:
            with self._lock:
                # Stop any existing auto-save
                self.stop_auto_save_advanced()
                
                # Set current state
                self._current_state = state
                
                # Start auto-save timer
                save_interval = interval or self.auto_save_interval
                self._schedule_auto_save_advanced(save_interval, validate_on_save)
                self._auto_save_enabled = True
                
                self.logger.info(f"Advanced auto-save started for session {state.session_id} (interval: {save_interval}s)")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to start advanced auto-save: {str(e)}")
            return False
    
    def stop_auto_save_advanced(self) -> bool:
        """
        Stops advanced auto-save functionality.
        
        Returns:
            Boolean indicating success
        """
        try:
            with self._lock:
                if self._auto_save_timer:
                    self._auto_save_timer.cancel()
                    self._auto_save_timer = None
                
                self._auto_save_enabled = False
                self.logger.info("Advanced auto-save stopped")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to stop advanced auto-save: {str(e)}")
            return False
    
    def _schedule_auto_save_advanced(self, 
                                     interval: float,
                                     validate_on_save: bool):
        """
        Schedule next advanced auto-save.
        
        Args:
            interval: Time until next save
            validate_on_save: Whether to validate integrity on save
        """
        def _auto_save_callback():
            """Callback for advanced auto-save timer."""
            try:
                with self._lock:
                    if self._current_state and self._auto_save_enabled:
                        # Save current state
                        success = self.save_state_advanced(
                            self._current_state,
                            create_backup=False,  # No backup during auto-save for performance
                            validate_integrity=validate_on_save
                        )
                        
                        if success:
                            self.logger.debug(f"Advanced auto-save successful for session {self._current_state.session_id}")
                        else:
                            self.logger.warning(f"Advanced auto-save failed for session {self._current_state.session_id}")
                        
                        # Schedule next auto-save if still enabled
                        if self._auto_save_enabled:
                            self._schedule_auto_save_advanced(interval, validate_on_save)
                            
            except Exception as e:
                self.logger.error(f"Advanced auto-save callback failed: {str(e)}")
        
        # Create and start timer
        self._auto_save_timer = threading.Timer(interval, _auto_save_callback)
        self._auto_save_timer.daemon = True
        self._auto_save_timer.start()
    
    def get_session_info_advanced(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Gets advanced information about a session without loading full state.
        
        Args:
            session_id: ID of session
            
        Returns:
            Dictionary with session information or None if not found
        """
        try:
            state_file_path = self.base_path / f"{session_id}.json"
            
            if not state_file_path.exists():
                return None
            
            # Get file stats
            file_stat = state_file_path.stat()
            
            # Read file headers to get basic info without full load
            with open(state_file_path, 'rb') as f:
                # Read first part of file
                if self.security_config.encrypt_states:
                    # For encrypted files, we can only get file properties
                    header_data = '{"encrypted": true}'
                elif self.security_config.compress_states:
                    # For compressed files, we can only get file properties
                    header_data = '{"compressed": true}'
                else:
                    # For regular JSON, read first part
                    first_part = f.read(min(1024, file_stat.st_size)).decode('utf-8')
                    header_data = first_part
            
            # Try to parse header data
            try:
                header_dict = json.loads(header_data)
            except json.JSONDecodeError:
                header_dict = {}
            
            return {
                "session_id": header_dict.get("session_id", session_id),
                "created_at": header_dict.get("created_at"),
                "updated_at": header_dict.get("updated_at"),
                "image_path": header_dict.get("image_path", ""),
                "naive_prompt": header_dict.get("naive_prompt", ""),
                "status": header_dict.get("status", "unknown"),
                "current_iteration": header_dict.get("current_iteration", 0),
                "version": header_dict.get("version", "unknown"),
                "encrypted": header_dict.get("encrypted", False),
                "compressed": header_dict.get("compressed", False),
                "file_size_bytes": file_stat.st_size,
                "file_size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                "modified_time": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                "prompt_count": len(header_dict.get("prompt_history", [])),
                "entity_count": len(header_dict.get("entity_detections", [])),
                "validation_count": len(header_dict.get("validation_results", [])),
                "has_checkpoints": len(header_dict.get("checkpoints", {})) > 0,
                "checksum": header_dict.get("checksum", ""),
                "has_user_feedback": bool(header_dict.get("user_feedback"))
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get advanced session info for {session_id}: {str(e)}")
            return None
    
    def recover_from_backup(self, session_id: str) -> bool:
        """
        Recover session from backup file.
        
        Args:
            session_id: ID of session to recover
            
        Returns:
            Boolean indicating success
        """
        try:
            backup_file_path = self.base_path / f"{session_id}.json.backup"
            state_file_path = self.base_path / f"{session_id}.json"
            
            if not backup_file_path.exists():
                self.logger.error(f"Backup file not found: {backup_file_path}")
                return False
            
            # Restore from backup
            shutil.copy2(backup_file_path, state_file_path)
            self.logger.info(f"Session recovered from backup: {session_id}")
            
            # Reload the recovered state
            recovered_state = self.load_state_advanced(session_id)
            if recovered_state:
                with self._lock:
                    self._current_state = recovered_state
                self.logger.info(f"Recovered state loaded for session: {session_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to recover session {session_id} from backup: {str(e)}")
            return False
    
    def compact_state_file(self, session_id: str) -> bool:
        """
        Compact state file by removing redundant data and reorganizing.
        
        Args:
            session_id: ID of session to compact
            
        Returns:
            Boolean indicating success
        """
        try:
            # Load current state
            current_state = self.load_state_advanced(session_id)
            if not current_state:
                self.logger.error(f"Failed to load state for compaction: {session_id}")
                return False
            
            # Remove old prompt history (keep last 10)
            if len(current_state.prompt_history) > 10:
                current_state.prompt_history = current_state.prompt_history[-10:]
                self.logger.info(f"Compacted prompt history for {session_id}: {len(current_state.prompt_history)} entries")
            
            # Remove old validation results (keep last 5)
            if len(current_state.validation_results) > 5:
                current_state.validation_results = current_state.validation_results[-5:]
                self.logger.info(f"Compacted validation results for {session_id}: {len(current_state.validation_results)} entries")
            
            # Clean up temporary files list
            if current_state.temp_files:
                # Remove non-existent temp files
                cleaned_temp_files = []
                for temp_file in current_state.temp_files:
                    if os.path.exists(temp_file):
                        cleaned_temp_files.append(temp_file)
                    else:
                        self.logger.debug(f"Removed non-existent temp file reference: {temp_file}")
                current_state.temp_files = cleaned_temp_files
            
            # Remove old checkpoints (keep last 5)
            if current_state.checkpoints:
                checkpoint_items = list(current_state.checkpoints.items())
                if len(checkpoint_items) > 5:
                    # Keep the 5 most recent checkpoints
                    sorted_checkpoints = sorted(checkpoint_items, key=lambda x: x[1], reverse=True)
                    current_state.checkpoints = dict(sorted_checkpoints[:5])
                    self.logger.info(f"Compacted checkpoints for {session_id}: {len(current_state.checkpoints)} entries")
            
            # Save compacted state
            success = self.save_state_advanced(current_state, create_backup=False)
            if success:
                self.logger.info(f"State file compacted for session: {session_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to compact state file for {session_id}: {str(e)}")
            return False
    
    def validate_state_integrity(self, session_id: str) -> Dict[str, Any]:
        """
        Validate the integrity of a session state.
        
        Args:
            session_id: ID of session to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            state_file_path = self.base_path / f"{session_id}.json"
            
            if not state_file_path.exists():
                return {
                    "session_id": session_id,
                    "valid": False,
                    "errors": ["State file not found"],
                    "warnings": [],
                    "file_size_bytes": 0
                }
            
            # Get file stats
            file_stat = state_file_path.stat()
            
            # Load and validate state
            state = self.load_state_advanced(session_id)
            
            errors = []
            warnings = []
            
            if not state:
                errors.append("Failed to load state")
            else:
                # Check for missing required fields
                required_fields = ["session_id", "created_at", "image_path", "naive_prompt"]
                for field in required_fields:
                    if not getattr(state, field, None):
                        errors.append(f"Missing required field: {field}")
                
                # Validate checksum if present
                if hasattr(state, 'checksum') and state.checksum:
                    state_dict = asdict(state).copy()
                    state_dict.pop('checksum', None)
                    data_for_checksum = json.dumps(state_dict, sort_keys=True, default=str)
                    calculated_checksum = self._calculate_checksum(data_for_checksum)
                    
                    if state.checkpoint != calculated_checksum:
                        warnings.append("Checksum mismatch")
                
                # Check file size
                file_size_mb = file_stat.st_size / (1024 * 1024)
                if file_size_mb > self.max_file_size_mb:
                    warnings.append(f"File size ({file_size_mb:.1f}MB) exceeds recommended limit ({self.max_file_size_mb}MB)")
            
            return {
                "session_id": session_id,
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "file_size_bytes": file_stat.st_size,
                "file_size_mb": file_stat.st_size / (1024 * 1024)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to validate state integrity for {session_id}: {str(e)}")
            return {
                "session_id": session_id,
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
                "file_size_bytes": 0
            }

# Example usage
if __name__ == "__main__":
    # Initialize advanced state manager with encryption and compression
    security_config = SecurityConfig(
        encrypt_states=True,
        encryption_password="your_secure_password_here",
        compress_states=True,
        compression_level=6
    )
    
    advanced_manager = AdvancedStateManager(
        base_path="~/.edi/test_sessions_advanced",
        auto_save_interval=2.0,  # 2 second auto-save for testing
        security_config=security_config,
        max_file_size_mb=50
    )
    
    print("Advanced State Manager initialized")
    
    # Create example advanced session state
    advanced_session_state = AdvancedSessionState(
        session_id="advanced_test_session_123",
        image_path="/path/to/test/image.jpg",
        naive_prompt="make the sky more dramatic",
        status="active",
        current_iteration=2,
        prompt_history=[
            {
                "iteration": 0,
                "positive_prompt": "dramatic sky with storm clouds",
                "negative_prompt": "sunny sky, clear weather",
                "quality_score": 0.92,
                "generated_at": datetime.now().isoformat()
            },
            {
                "iteration": 1,
                "positive_prompt": "storm clouds with lighting and dark atmosphere",
                "negative_prompt": "sunny sky, clear weather, no clouds",
                "quality_score": 0.88,
                "generated_at": datetime.now().isoformat()
            }
        ],
        entity_detections=[
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
        validation_results=[
            {
                "attempt_number": 1,
                "alignment_score": 0.85,
                "preserved_count": 3,
                "modified_count": 1,
                "unintended_count": 0,
                "user_feedback": "Great improvement to the sky!",
                "validated_at": datetime.now().isoformat()
            }
        ],
        user_feedback={
            "feedback_type": "partial",
            "comments": "Good start, but could be more dramatic",
            "rating": 4
        },
        processing_metrics={
            "total_processing_time": 120.5,
            "model_inference_time": 45.2,
            "validation_time": 15.3
        },
        model_configs={
            "vision_model": "sam2",
            "diffusion_model": "qwen3:8b",
            "refinement_iterations": 3
        },
        version="1.0"
    )
    
    # Save advanced state
    if advanced_manager.save_state_advanced(advanced_session_state):
        print(f"Advanced session state saved: {advanced_session_state.session_id}")
    else:
        print("Failed to save advanced session state")
    
    # Load advanced state
    loaded_advanced_state = advanced_manager.load_state_advanced(advanced_session_state.session_id)
    if loaded_advanced_state:
        print(f"Advanced session state loaded: {loaded_advanced_state.naive_prompt}")
    else:
        print("Failed to load advanced session state")
    
    # Start advanced auto-save
    if advanced_manager.start_auto_save_advanced(advanced_session_state, validate_on_save=True):
        print("Advanced auto-save started")
    else:
        print("Failed to start advanced auto-save")
    
    # Create advanced checkpoint
    checkpoint_name = advanced_manager.checkpoint_advanced(advanced_session_state, "demo_checkpoint_advanced")
    print(f"Advanced checkpoint created: {checkpoint_name}")
    
    # Update state and wait for auto-save
    advanced_session_state.current_iteration = 3
    advanced_session_state.prompt_history.append({
        "iteration": 2,
        "positive_prompt": "extremely dramatic sky with thunder and lighting",
        "negative_prompt": "sunny sky, clear weather, no clouds, subtle changes",
        "quality_score": 0.95,
        "generated_at": datetime.now().isoformat()
    })
    
    print("Advanced state updated, waiting for auto-save...")
    time.sleep(3)  # Wait for auto-save to happen
    
    # Stop advanced auto-save
    if advanced_manager.stop_auto_save_advanced():
        print("Advanced auto-save stopped")
    else:
        print("Failed to stop advanced auto-save")
    
    # Get advanced session info
    advanced_session_info = advanced_manager.get_session_info_advanced(advanced_session_state.session_id)
    if advanced_session_info:
        print(f"Advanced session info: {advanced_session_info['prompt_count']} prompts")
        print(f"  File size: {advanced_session_info['file_size_mb']:.2f}MB")
        print(f"  Encrypted: {advanced_session_info['encrypted']}")
        print(f"  Compressed: {advanced_session_info['compressed']}")
    else:
        print("Failed to get advanced session info")
    
    # Validate state integrity
    validation_result = advanced_manager.validate_state_integrity(advanced_session_state.session_id)
    print(f"State integrity validation: {'Valid' if validation_result['valid'] else 'Invalid'}")
    if validation_result['errors']:
        print(f"  Errors: {validation_result['errors']}")
    if validation_result['warnings']:
        print(f"  Warnings: {validation_result['warnings']}")
    
    # Compact state file
    if advanced_manager.compact_state_file(advanced_session_state.session_id):
        print("State file compacted successfully")
    else:
        print("Failed to compact state file")
    
    # Get all session IDs
    session_ids = advanced_manager.get_session_ids()
    print(f"Available sessions: {session_ids}")
    
    print("Advanced State Manager example completed")
```