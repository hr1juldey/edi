# Database.query_history()

[Back to Database](../storage_database.md)

## Related User Story
"As a user, I want to review my past editing sessions." (from PRD - implied by session persistence requirements)

## Function Signature
`query_history()`

## Parameters
- None - Queries the session history with default parameters

## Returns
- List of session summary objects with basic information about past sessions

## Step-by-step Logic
1. Execute a query against the sessions table to retrieve session summaries
2. Apply default filters (e.g., exclude failed sessions, limit to recent sessions)
3. Select relevant fields for session summaries (ID, image path, prompt, timestamp, score)
4. Order results by creation date (most recent first)
5. Apply any default limits to prevent excessive result sets
6. Return the list of session summaries
7. Handle database errors gracefully and return appropriate results

## Query Parameters
- By default, retrieves recent successful sessions
- Can be extended to accept filters (date ranges, image path patterns, score thresholds)
- Limits results to prevent performance issues
- Orders by most recent sessions first

## Performance Considerations
- Uses indexing on relevant columns (timestamp, status)
- Limits result set size to prevent memory issues
- Only selects needed fields for summaries
- Efficient query structure to minimize database load

## Input/Output Data Structures
### Session Summary Object
A simplified object containing:
- Session ID (UUID)
- Image path
- Original prompt (truncated if too long)
- Creation timestamp
- Final alignment score
- Session status
- Duration (if available)

### Query Results
A list of session summary objects:
- Ordered by creation date (descending)
- Limited to prevent excessive results
- Contains essential information for session review