# Database.load_session()

[Back to Database](../storage_database.md)

## Related User Story
"As a user, I want EDI to resume my editing sessions where I left off." (from PRD - implied by session persistence requirements)

## Function Signature
`load_session()`

## Parameters
- None - Uses internal session identifier

## Returns
- None - Loads session data from database into memory

## Step-by-step Logic
1. Use session ID to query the sessions table
2. Begin a database transaction for consistent reads
3. Load the main session record from the sessions table
4. Load associated prompt history from the prompts table
5. Load associated entity detections from the entities table
6. Load associated validation results from the validations table
7. Load associated user feedback from the user_feedback table if available
8. Reconstruct the complete session state in memory
9. Commit the transaction and handle any errors

## Data Relationships
- Loads the primary session record first
- Loads all associated prompt iterations
- Loads all detected entities with their properties
- Loads all validation attempts and scores
- Optionally loads user feedback if it exists

## Reconstruction Process
- Rebuilds the session state from database records
- Maintains relationships between different data types
- Validates that all required information is present
- Provides default values for any missing optional data

## Input/Output Data Structures
### Session Record
A record from the sessions table containing:
- ID (UUID)
- Creation timestamp
- Image path
- Naive prompt
- Session status
- Final alignment score

### Reconstructed Session
An in-memory object containing:
- All session metadata
- Complete prompt history with iterations
- All detected entities with properties
- All validation attempts with scores
- User feedback if available