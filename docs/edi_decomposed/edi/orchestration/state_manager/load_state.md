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