# WorkflowManager.inject_params()

[Back to Workflow Manager](../integration_workflow_manager.md)

## Related User Story
"As a user, I want EDI to customize the editing process based on my specific request." (from PRD - implied by personalized editing requirements)

## Function Signature
`inject_params(workflow, params) -> dict`

## Parameters
- `workflow` - The workflow template dictionary to modify
- `params` - A dictionary of parameters to inject into the workflow

## Returns
- `dict` - A modified workflow dictionary with parameters injected

## Step-by-step Logic
1. Take the original workflow template and parameters dictionary as input
2. Identify the nodes in the workflow that require parameter injection
3. For each parameter in the params dictionary:
   - Find the appropriate node in the workflow
   - Update the node's parameters with the provided values
4. Validate that all required parameters have been properly injected
5. Return the modified workflow with injected parameters
6. Handle missing nodes or invalid parameter mappings gracefully

## Injection Process
- Maps parameters to appropriate nodes based on node type
- Updates positive and negative prompts in the correct nodes
- Sets image path in the input node
- Adjusts workflow-specific parameters (strength, steps, etc.)
- Preserves other workflow configuration not being modified

## Parameter Mappings
- `positive_prompt` → PromptText nodes or equivalent
- `negative_prompt` → Negative prompt nodes
- `image_path` → LoadImage nodes
- `strength` → For inpainting workflows
- `steps` → KSampler nodes
- Additional mappings as defined by specific workflow templates

## Input/Output Data Structures
### Params Object
A dictionary containing:
- positive_prompt: Text for positive prompt
- negative_prompt: Text for negative prompt
- image_path: Path to input image
- Additional workflow-specific parameters as needed

### Modified Workflow Object
A workflow dictionary with injected parameters:
- All original node definitions and connections preserved
- Parameter values updated according to params
- Valid structure for ComfyUI execution