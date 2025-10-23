# Commands: Doctor

[Back to Index](./index.md)

## Purpose
Diagnostic command - Contains the doctor_command async function that checks Python version, GPU availability, models, Ollama connection, ComfyUI connection, and outputs system diagnostics.

## Functions
- `async def doctor_command()`: Performs system diagnostics

### Details
- Checks Python version, GPU availability, models
- Tests Ollama connection, ComfyUI connection
- Outputs green checkmarks or red errors
- Provides comprehensive system health check

## Technology Stack

- System diagnostic utilities
- AsyncIO for asynchronous operations
- Hardware detection