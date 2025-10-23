# ComfyUIClient.submit_edit()

[Back to ComfyUI Client](../integration_comfyui_client.md)

## Related User Story
"As a user, I want EDI to generate high-quality image edits using advanced AI models." (from PRD - implied by core functionality)

## Function Signature
`submit_edit(image_path: str, positive_prompt: str, negative_prompt: str, workflow_template: str = "img2img_default") -> str`

## Parameters
- `image_path: str` - Path to the input image for editing
- `positive_prompt: str` - The positive prompt describing desired changes
- `negative_prompt: str` - The negative prompt describing undesired elements
- `workflow_template: str` - Name of the workflow template to use (default: "img2img_default")

## Returns
- `str` - The job ID of the submitted edit job

## Step-by-step Logic
1. Load the specified workflow template from workflows/ directory
2. Inject the positive and negative prompts into the workflow
3. Inject the image path into the workflow
4. Set any additional parameters based on the workflow type
5. Submit the workflow to the ComfyUI API endpoint
6. Receive and return the job ID for tracking the progress
7. Handle connection errors and timeouts gracefully

## Workflow Integration
- Loads workflow templates from the workflows/ directory
- Supports different types of workflows (img2img, inpaint, controlnet)
- Injects parameters into the appropriate nodes of the workflow
- Validates workflow structure before submission

## Error Handling
- Handles network connection errors
- Manages timeout scenarios
- Validates response from ComfyUI
- Provides clear error messages if submission fails

## Input/Output Data Structures
### Workflow Object
A JSON object representing the ComfyUI workflow:
- Node definitions and connections
- Parameter values for each node
- Input image and output specifications