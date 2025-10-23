# Storage: Migrations

[Back to Storage Layer](./storage_layer.md)

## Purpose
Schema versioning - Contains functions for handling database schema changes in a backward-compatible way.

## Functions
- `migrate_v1_to_v2()`: Migrates database schema from version 1 to 2
- Other migration functions as needed

### Details
- Handles backward-compatible schema changes
- Ensures data integrity during updates
- Manages version transitions

## Functions

- [migrate_v1_to_v2()](./storage/migrate_v1_to_v2.md)

## Technology Stack

- Database migration tools
- SQL for schema changes
- Version management