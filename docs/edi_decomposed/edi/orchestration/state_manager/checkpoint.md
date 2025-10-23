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