# ComfyUIClient.poll_status()

[Back to ComfyUI Client](../integration_comfyui_client.md)

## Related User Story
"As a user, I want to see progress updates while my image is being edited." (from PRD - implied by user experience requirements)

## Function Signature
`poll_status(job_id: str) -> dict`

## Parameters
- `job_id: str` - The job ID of the edit job to check status for

## Returns
- `dict` - A dictionary containing the status information for the job

## Step-by-step Logic
1. Make an API request to ComfyUI to check the status of the job with the given ID
2. Parse the response from the ComfyUI history endpoint
3. Extract the status information (queued, processing, completed, failed)
4. Include additional metadata like progress percentage if available
5. Handle API errors and return appropriate status information
6. Return the status information as a dictionary

## Status Values
- 'queued': Job is waiting in the queue
- 'processing': Job is currently being processed
- 'completed': Job has completed successfully
- 'failed': Job has failed due to an error

## Polling Strategy
- Designed to be called periodically during long operations
- Includes appropriate delays to avoid overwhelming the server
- Provides estimated time remaining when possible
- Handles scenarios where job IDs are no longer valid

## Input/Output Data Structures
### Status Response Object
A dictionary containing:
- Status (queued, processing, completed, failed)
- Progress percentage (0-100 if available)
- Estimated time remaining (if calculable)
- Output information (image paths, etc. when completed)
- Error information (if any occurred)