# Commands: Edit

[Back to Index](./index.md)

## Purpose
Main edit command - Contains the edit_command async function that serves as the entry point for `edi edit` command, launching the Textual app or running in headless mode.

## Functions
- `async def edit_command(image_path, prompt, **kwargs)`: Main entry point for editing functionality

### Details
- Entry point for `edi edit` CLI command
- Launches Textual app or runs headless mode
- Handles command line arguments and options

## Technology Stack

- AsyncIO for asynchronous operations
- CLI argument parsing
- Textual for TUI interface