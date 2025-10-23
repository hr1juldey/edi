# Commands: Clear

[Back to Index](./index.md)

## Purpose
Data cleanup command - Contains the clear_command async function that deletes old session files, purges database records, with user confirmation required for full cleanup.

## Functions
- `async def clear_command(sessions=False, all=False)`: Cleans up data and session files

### Details
- Deletes old session files
- Purges database records
- User confirmation required for full cleanup (--all option)
- Safely removes EDI data

## Technology Stack

- AsyncIO for asynchronous operations
- File system operations
- Database operations