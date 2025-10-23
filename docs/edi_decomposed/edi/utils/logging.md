# Utils: Logging

[Back to Index](./index.md)

## Purpose
Logging setup - Contains functions for setting up logging with specific levels and file handlers.

## Functions
- `setup_logger(name, level)`: Sets up a logger with the given name and level
- Writes to ~/.edi/logs/edi.log with rotating file handler (10MB max, 5 backups)

### Details
- Provides consistent logging across the application
- Uses rotating file handlers to manage log size
- Configurable log levels

## Technology Stack

- Python logging module
- Rotating file handlers