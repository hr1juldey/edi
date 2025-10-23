# ComfyUIClient.download_result()

[Back to ComfyUI Client](../integration_comfyui_client.md)

## Related User Story
"As a user, I want to receive my edited image when the process is complete." (from PRD - implied by core functionality)

## Function Signature
`download_result(job_id: str, output_path: str)`

## Parameters
- `job_id: str` - The job ID of the completed edit job
- `output_path: str` - The local file path where the result should be saved

## Returns
- None - Downloads and saves the result to the specified path

## Step-by-step Logic
1. First check the job status to ensure it's completed
2. If the job is not completed, raise an error
3. Get the output image filename from the job history
4. Make a request to download the result image from ComfyUI
5. Stream the image data to the specified output path
6. Handle file writing with atomic operations to prevent corruption
7. Validate that the download was successful

## File Management
- Uses streaming to handle large image files efficiently
- Performs atomic file writes to prevent corruption
- Validates file integrity after download
- Handles file permissions appropriately

## Error Handling
- Checks that the job is actually completed before downloading
- Handles network errors during download
- Manages disk space constraints
- Validates image file format after download

## Input/Output Data Structures
### Download Result
- Input: Job ID to identify the completed job
- Output: Saved image file at the specified output path
- The result is a valid image file (format depends on workflow)