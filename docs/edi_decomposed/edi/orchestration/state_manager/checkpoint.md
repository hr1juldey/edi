# StateManager.checkpoint()

[Back to State Manager](../orchestration_state_manager.md)

## Related User Story
"As a user, I want EDI to remember my progress so I can resume if interrupted." (from PRD - implied by session persistence requirements)

## Function Signature
`checkpoint()`

## Parameters
- None - Creates a checkpoint of current internal state

## Returns
- None - Saves checkpoint to file system

## Step-by-step Logic
1. Capture current application state at a significant point in the workflow
2. Create a timestamped checkpoint of the session state
3. Serialize the state to JSON format with checkpoint metadata
4. Write the JSON to ~/.edi/sessions/<session_id>_checkpoint_<timestamp>.json
5. Maintain a limited number of checkpoints to manage disk usage
6. Ensure atomic file write to prevent corruption
7. Handle any file system errors gracefully

## Checkpoint Strategy
- Creates checkpoints at significant workflow transitions
- Maintains multiple recovery points in case of failure
- Limits the number of checkpoints to prevent excessive storage use
- Provides ability to roll back to previous states if needed

## Recovery Management
- Checkpoints provide alternative restoration points
- Help recover from failed operations mid-process
- Maintain metadata about when and why checkpoint was created
- Enable rollback functionality in case of errors

## Input/Output Data Structures
### Checkpoint State Object
Contains:
- Session ID (UUID)
- Checkpoint timestamp
- Checkpoint reason/metadata
- Current stage of processing
- All relevant state information for recovery
- Reference to parent session

## See Docs

```python
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class StateManager:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.state_data = {}
        self.checkpoint_dir = Path.home() / ".edi" / "sessions"
        self.checkpoint_limit = 10  # Maximum number of checkpoints to keep
        
        # Initialize the checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint(self, reason: str = "manual"):
        """
        Creates a checkpoint of the current application state at a significant point in the workflow.
        
        This method:
        1. Captures current application state at a significant point in the workflow
        2. Creates a timestamped checkpoint of the session state
        3. Serializes the state to JSON format with checkpoint metadata
        4. Writes the JSON to ~/.edi/sessions/<session_id>_checkpoint_<timestamp>.json
        5. Maintains a limited number of checkpoints to manage disk usage
        6. Ensures atomic file write to prevent corruption
        7. Handles any file system errors gracefully
        """
        try:
            # Step 1: Capture current application state at a significant point in the workflow
            checkpoint_state = {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "reason": reason,
                "stage": self.state_data.get("current_stage", "unknown"),
                "state_info": self.state_data,
                "parent_session": self.state_data.get("parent_session", None)
            }

            # Create the checkpoint filename with timestamp
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            checkpoint_filename = f"{self.session_id}_checkpoint_{timestamp_str}.json"
            checkpoint_path = self.checkpoint_dir / checkpoint_filename

            # Step 3: Serialize the state to JSON format with checkpoint metadata
            # Step 4: Write the JSON to ~/.edi/sessions/<session_id>_checkpoint_<timestamp>.json
            # Writing atomically by using a temporary file
            temp_path = checkpoint_path.with_suffix('.tmp.json')
            with temp_path.open('w', encoding='utf-8') as f:
                json.dump(checkpoint_state, f, indent=2, ensure_ascii=False)
            
            # Atomic move to actual checkpoint file
            temp_path.replace(checkpoint_path)

            # Step 5: Maintain a limited number of checkpoints to manage disk usage
            self._manage_checkpoint_count()

            print(f"Checkpoint created successfully: {checkpoint_path}")
            
        except Exception as e:
            # Step 7: Handle any file system errors gracefully
            print(f"Error creating checkpoint: {str(e)}")
            raise

    def _manage_checkpoint_count(self):
        """Maintains the number of checkpoints under the limit by removing oldest ones."""
        checkpoint_files = list(self.checkpoint_dir.glob(f"{self.session_id}_checkpoint_*.json"))
        checkpoint_files.sort(key=lambda x: x.name)  # Sort by name which contains timestamp
        
        # Remove oldest checkpoints if we exceed the limit
        while len(checkpoint_files) > self.checkpoint_limit:
            oldest_checkpoint = checkpoint_files.pop(0)
            try:
                oldest_checkpoint.unlink()  # Remove the file
                print(f"Removed old checkpoint: {oldest_checkpoint}")
            except OSError as e:
                print(f"Error removing old checkpoint {oldest_checkpoint}: {e}")

    def set_state(self, key: str, value: Any):
        """Helper method to set state data that will be included in checkpoints"""
        self.state_data[key] = value

# Example usage:
if __name__ == "__main__":
    # Initialize a state manager
    state_manager = StateManager(session_id="session_12345")
    
    # Set some state data
    state_manager.set_state("current_stage", "image_processing")
    state_manager.set_state("progress", 0.75)
    state_manager.set_state("image_path", "/path/to/working/image.jpg")
    state_manager.set_state("workflow_step", "enhancement")
    
    # Create a checkpoint at a significant transition point
    state_manager.checkpoint("workflow_transition")
    
    # Simulate more work and create another checkpoint
    state_manager.set_state("current_stage", "result_generation")
    state_manager.set_state("progress", 0.9)
    state_manager.checkpoint("completion_phase")
```