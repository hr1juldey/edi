# Storage: Models

[Back to Storage Layer](./storage_layer.md)

## Purpose
Database models - Contains SessionRecord, PromptRecord, EntityRecord and other data structures that map to database tables.

## Models
- `SessionRecord`: Represents a session in the database
- `PromptRecord`: Represents a prompt history record
- `EntityRecord`: Represents an entity detection record
- Other related database models

### Details
- SQLAlchemy ORM or dataclasses with SQL mapping
- Maps directly to database tables
- Provides structured access to stored data

## Technology Stack

- SQLAlchemy ORM or dataclasses
- SQL mapping
- Pydantic for validation