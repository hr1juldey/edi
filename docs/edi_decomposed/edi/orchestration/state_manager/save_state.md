# StateManager.save_state()

[Back to State Manager](../orchestration_state_manager.md)

## Related User Story
"As a user, I want EDI to remember my progress so I can resume if interrupted." (from PRD - implied by session persistence requirements)

## Function Signature
`save_state()`

## Parameters
- None - Uses internal state of the StateManager

## Returns
- None - Saves state to file system

## Step-by-step Logic
1. Gather current session state including:
   - Current stage of editing process
   - Image path and naive prompt
   - Scene analysis results
   - Intent and parsed information
   - Generated prompts and refinements
   - Edited image path if available
   - Validation results if available
2. Serialize the state to JSON format
3. Write the JSON to ~/.edi/sessions/<session_id>.json
4. Ensure atomic file write to prevent corruption
5. Handle any file system errors gracefully

## Auto-save Implementation
- Automatically called every 5 seconds during active sessions
- Also called after significant events in the editing process
- Ensures data is not lost in case of interruption
- Maintains session continuity

## File Management
- Creates session files with unique identifiers
- Stores in ~/.edi/sessions directory
- Follows JSON format for easy parsing and debugging
- Handles file permissions appropriately

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
import os

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

    def save_state(self):
        """
        Saves the current session state to the file system.
        
        This method:
        1. Gathers current session state including:
           - Current stage of editing process
           - Image path and naive prompt
           - Scene analysis results
           - Intent and parsed information
           - Generated prompts and refinements
           - Edited image path if available
           - Validation results if available
        2. Serializes the state to JSON format
        3. Writes the JSON to ~/.edi/sessions/<session_id>.json
        4. Ensures atomic file write to prevent corruption
        5. Handles any file system errors gracefully
        """
        # Step 1: Gather current session state
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

        try:
            # Ensure the directory exists
            # Step 4: Ensure atomic file write to prevent corruption
            self.session_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to a temporary file first
            temp_path = self.session_file_path.with_suffix('.tmp')
            with temp_path.open('w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
            
            # Atomic move to actual file
            temp_path.replace(self.session_file_path)
            
            # Set appropriate file permissions (read/write for owner only)
            os.chmod(self.session_file_path, 0o600)
            
            print(f"State saved successfully to {self.session_file_path}")
            
        except Exception as e:
            # Step 5: Handle any file system errors gracefully
            print(f"Error saving state: {str(e)}")
            # If temp file exists and we failed, remove it
            if temp_path.exists():
                temp_path.unlink()
            raise

    def set_state(self, current_stage: str = None, image_path: str = None, 
                  naive_prompt: str = None, scene_analysis: Dict = None, 
                  intent: Dict = None, prompts: list = None, 
                  edited_image_path: str = None, validation_results: Dict = None):
        """Helper method to set state values before saving"""
        if current_stage is not None:
            self.current_stage = current_stage
        if image_path is not None:
            self.image_path = image_path
        if naive_prompt is not None:
            self.naive_prompt = naive_prompt
        if scene_analysis is not None:
            self.scene_analysis = scene_analysis
        if intent is not None:
            self.intent = intent
        if prompts is not None:
            self.prompts = prompts
        if edited_image_path is not None:
            self.edited_image_path = edited_image_path
        if validation_results is not None:
            self.validation_results = validation_results

# Example usage demonstrating auto-save:
if __name__ == "__main__":
    import time
    
    # Create an instance of StateManager
    state_manager = StateManager()
    
    # Set some state values
    state_manager.set_state(
        current_stage="refinement",
        image_path="/path/to/image.jpg",
        naive_prompt="Make the sky more blue",
        scene_analysis={"entities": ["sky", "clouds"], "spatial_layout": "wide"},
        intent={"target_entities": ["sky"], "edit_type": "color_adjustment", "confidence": 0.85},
        prompts=["Make the sky more blue", "Increase blue saturation in sky areas"],
        validation_results={"score": 0.92, "delta": 0.05}
    )
    
    # Save state
    state_manager.save_state()
    
    # Simulate auto-save functionality
    def auto_save_demo():
        """Simulate auto-save that runs every 5 seconds during active sessions"""
        print("Starting auto-save simulation...")
        
        for i in range(3):
            # Simulate some work
            state_manager.set_state(current_stage=f"stage_{i+1}")
            state_manager.save_state()
            print(f"Auto-saved at iteration {i+1}")
            
            # Sleep simulates waiting (would be 5 seconds in real implementation)
            time.sleep(0.1)  # Shortened for demo
        
        print("Auto-save simulation completed")
    
    auto_save_demo()
```