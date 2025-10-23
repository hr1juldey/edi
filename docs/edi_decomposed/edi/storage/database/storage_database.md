# Storage: Database

[Back to Storage Layer](./storage_layer.md)

## Purpose
SQLite wrapper - Contains the Database class that provides methods for saving and loading sessions with transaction support.

## Class: Database

### Methods
- `save_session()`: Saves a session to the database
- `load_session()`: Loads a session from the database
- `query_history()`: Queries session history

### Details
- Initializes tables on first run
- Provides transaction support for data integrity
- Handles all database interactions for the application

## Functions

- [save_session()](./storage/save_session.md)
- [load_session()](./storage/load_session.md)
- [query_history()](./storage/query_history.md)

## Technology Stack

- SQLite for database storage
- SQL for queries
- Transaction management