# StateManager.load_state()

[Back to State Manager](../orchestration_state_manager.md)

## Related User Story
"As a user, I want EDI to remember my progress so I can resume if interrupted." (from PRD - implied by session persistence requirements)

## Function Signature
`load_state()`

## Parameters
- None - Loads from session state file

## Returns
- None - Loads state into internal StateManager properties

## Step-by-step Logic
1. Determine the session ID to load (either from parameter or current session)
2. Read the JSON from ~/.edi/sessions/<session_id>.json
3. Validate the JSON structure and required fields
4. Deserialize the JSON into internal state properties
5. Reconstruct any complex objects from their JSON representations
6. Update the current state of the application with loaded values
7. Handle any file system or parsing errors gracefully

## Session Restoration
- Restores all aspects of the editing session
- Maintains continuity from where the session left off
- Validates that the session data is consistent and complete
- Provides feedback if session restoration fails

## Error Handling
- Checks for missing or corrupted session files
- Handles version mismatches in session data
- Provides default values for missing optional fields
- Logs warnings for any partial restoration

## Input/Output Data Structures
### Session State Object
Contains:
- Session ID (UUID)
- Current stage (refinement, execution, results, etc.)
- Image path
- Naive prompt
- Scene analysis (entities, spatial layout)
- Intent (target entities, edit type, confidence)
- Prompts (by iteration)
- Edited image path
- Validation results (score, delta)

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
        self.session_file_path = Path.home() / ".edi" / "sessions" / f"{self.session_id}.json"
        self.checkpoint_dir = Path.home() / ".edi" / "sessions"
        
        # Initialize state attributes
        self.current_stage = None
        self.image_path = None
        self.naive_prompt = None
        self.scene_analysis = {}
        self.intent = {}
        self.prompts = []
        self.edited_image_path = None
        self.validation_results = {}

    def load_state(self, session_id: Optional[str] = None):
        """
        Loads the state from a session state file into internal StateManager properties.
        
        This method:
        1. Determines the session ID to load (either from parameter or current session)
        2. Reads the JSON from ~/.edi/sessions/<session_id>.json
        3. Validates the JSON structure and required fields
        4. Deserializes the JSON into internal state properties
        5. Reconstructs any complex objects from their JSON representations
        6. Updates the current state of the application with loaded values
        7. Handles any file system or parsing errors gracefully
        """
        # Step 1: Determine the session ID to load (either from parameter or current session)
        target_session_id = session_id or self.session_id
        session_file = self.checkpoint_dir / f"{target_session_id}.json"
        
        # Step 2: Read the JSON from ~/.edi/sessions/<session_id>.json
        try:
            if not session_file.exists():
                raise FileNotFoundError(f"Session file does not exist: {session_file}")
            
            with session_file.open('r', encoding='utf-8') as f:
                session_data = json.load(f)
        
        # Step 7: Handle any file system or parsing errors gracefully
        except FileNotFoundError as e:
            print(f"Session file not found: {e}")
            # Provide default values for missing optional fields
            return False
        except json.JSONDecodeError as e:
            print(f"Error parsing session file: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error loading session: {e}")
            return False

        # Step 3: Validate the JSON structure and required fields
        required_fields = ['session_id']
        for field in required_fields:
            if field not in session_data:
                print(f"Missing required field '{field}' in session data")
                return False

        # Step 4: Deserialize the JSON into internal state properties
        self.session_id = session_data.get('session_id', self.session_id)
        
        # Update the session file path to match the loaded session
        self.session_file_path = self.checkpoint_dir / f"{self.session_id}.json"
        
        # Load other state properties
        self.current_stage = session_data.get('current_stage', self.current_stage)
        self.image_path = session_data.get('image_path', self.image_path)
        self.naive_prompt = session_data.get('naive_prompt', self.naive_prompt)
        self.scene_analysis = session_data.get('scene_analysis', self.scene_analysis)
        self.intent = session_data.get('intent', self.intent)
        self.prompts = session_data.get('prompts', self.prompts)
        self.edited_image_path = session_data.get('edited_image_path', self.edited_image_path)
        self.validation_results = session_data.get('validation_results', self.validation_results)

        # Step 5: Reconstruct any complex objects from their JSON representations
        # (In this case, most objects are basic Python types that are handled automatically by JSON)
        
        # Step 6: Update the current state of the application with loaded values
        print(f"Session state loaded successfully from {session_file}")
        return True

    def save_state(self):
        """Helper method to save the current state to the session file"""
        state_data = {
            "session_id": self.session_id,
            "current_stage": self.current_stage,
            "image_path": self.image_path,
            "naive_prompt": self.naive_prompt,
            "scene_analysis": self.scene_analysis,
            "intent": self.intent,
            "prompts": self.prompts,
            "edited_image_path": self.edited_image_path,
            "validation_results": self.validation_results
        }
        
        # Ensure the directory exists
        self.session_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the state data to the file
        with self.session_file_path.open('w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)

# Example usage:
if __name__ == "__main__":
    # Create an instance of StateManager
    state_manager = StateManager()
    
    # Attempt to load state from a session
    success = state_manager.load_state("session_12345")  # Load specific session
    
    if success:
        print(f"Session loaded. Current stage: {state_manager.current_stage}")
        print(f"Image path: {state_manager.image_path}")
        print(f"Naive prompt: {state_manager.naive_prompt}")
    else:
        print("Failed to load session state, using defaults")
```