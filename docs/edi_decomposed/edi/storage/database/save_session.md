# Database.save_session()

[Back to Database](../storage_database.md)

## Related User Story
"As a user, I want EDI to remember my editing sessions so I can resume later or review my history." (from PRD - implied by session persistence requirements)

## Function Signature
`save_session()`

## Parameters
- None - Uses internal session data

## Returns
- None - Saves session data to database

## Step-by-step Logic
1. Prepare session data from the current session state
2. Begin a database transaction to ensure consistency
3. Insert or update the session record in the sessions table
4. Save prompt history to the prompts table
5. Save detected entities to the entities table
6. Save validation results to the validations table
7. Save user feedback to the user_feedback table if available
8. Commit the transaction to finalize the save
9. Handle any database errors and rollback if necessary

## Data Relationships
- Session record is the primary record linking all related data
- Multiple prompt records associated with one session
- Multiple entity records associated with one session
- Multiple validation records associated with one session
- Feedback optionally linked to a session

## Transaction Management
- Uses database transactions to maintain data integrity
- Rolls back all changes if any part of the save fails
- Ensures related records are saved together consistently
- Handles concurrent access safely

## Input/Output Data Structures
### Session Record
A record in the sessions table containing:
- ID (UUID)
- Creation timestamp
- Image path
- Naive prompt
- Session status
- Final alignment score

### Related Records
- Prompts: Positive and negative prompts by iteration
- Entities: Detected objects with properties and positions
- Validations: Alignment scores and change assessments
- User feedback: Ratings and comments