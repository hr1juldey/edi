# Commands: Setup

[Back to Index](./index.md)

## Purpose
Setup command - Contains the setup_command async function that creates the ~/.edi/ directory structure, downloads models if requested, and verifies Ollama connection.

## Functions
- `async def setup_command(download_models=False)`: Sets up the EDI environment

### Details
- Creates ~/.edi/ directory structure
- Downloads default models if requested
- Verifies Ollama connection
- Prepares the system for EDI operation

## Technology Stack

- AsyncIO for asynchronous operations
- File system operations
- Model download utilities