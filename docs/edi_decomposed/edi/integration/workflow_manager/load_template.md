# WorkflowManager.load_template()

[Back to Workflow Manager](../integration_workflow_manager.md)

## Related User Story
"As a user, I want EDI to use appropriate editing techniques for different types of changes." (from PRD - implied by technical requirements)

## Function Signature
`load_template(name) -> dict`

## Parameters
- `name` - The name of the workflow template to load (e.g., "img2img_default", "inpaint_masked", "controlnet_canny")

## Returns
- `dict` - A dictionary containing the workflow template JSON

## Step-by-step Logic
1. Determine the path to the workflow template file based on the name
2. Validate that the workflow file exists and is accessible
3. Read and parse the JSON from the workflow file
4. Validate the structure of the workflow to ensure it's valid for ComfyUI
5. Return the workflow template as a dictionary
6. Handle file system errors and JSON parsing errors gracefully

## Template Types
- `img2img_default.json`: Standard image-to-image workflow
- `inpaint_masked.json`: Masked inpainting for region-specific edits
- `controlnet_canny.json`: Structure-preserving edits using ControlNet
- Additional templates as needed for different editing requirements

## Validation Process
- Checks that all required nodes are present in the workflow
- Validates JSON structure conforms to ComfyUI requirements
- Ensures parameter placeholders exist where needed
- Reports validation errors if workflow is malformed

## Input/Output Data Structures
### Workflow Template Object (JSON)
A JSON object defining a ComfyUI workflow:
- Node definitions with unique IDs
- Node class types (LoadImage, PromptText, KSampler, etc.)
- Connection specifications between nodes
- Default parameter values
- Placeholder values for dynamic injection