# Integration: ComfyUI Client

[Back to Integration Layer](./integration_layer.md)

## Purpose
ComfyUI API wrapper - Contains the ComfyUIClient class that handles communication with ComfyUI including submitting edits, polling status, and downloading results.

## Class: ComfyUIClient

### Methods
- `submit_edit()`: Submits an editing job to ComfyUI
- `poll_status()`: Checks the status of a submitted job
- `download_result()`: Downloads the result of a completed job

### Details
- Loads workflow templates from workflows/ directory
- Handles timeouts and retries gracefully
- Manages communication with ComfyUI API

## Functions

- [submit_edit()](./integration/submit_edit.md)
- [poll_status()](./integration/poll_status.md)
- [download_result()](./integration/download_result.md)

## Technology Stack

- Requests for HTTP communication
- JSON for API communication
- Pillow for image processing