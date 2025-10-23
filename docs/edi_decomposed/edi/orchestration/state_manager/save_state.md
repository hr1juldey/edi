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