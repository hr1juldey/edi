# Orchestration: State Manager

[Back to Orchestrator](./orchestrator.md)

## Purpose
Session state tracking - Contains the StateManager class that saves and loads session state to JSON files, with auto-save functionality.

## Class: StateManager

### Methods
- `save_state()`: Saves current session state to JSON
- `load_state()`: Loads session state from JSON
- `checkpoint()`: Creates a checkpoint of the current state

### Details
- Writes JSON to ~/.edi/sessions/<session_id>.json
- Auto-saves every 5 seconds
- Maintains session continuity across interruptions

## Functions

- [save_state()](./orchestration/save_state.md)
- [load_state()](./orchestration/load_state.md)
- [checkpoint()](./orchestration/checkpoint.md)

## Technology Stack

- JSON for serialization
- File I/O operations
- Time utilities for auto-save