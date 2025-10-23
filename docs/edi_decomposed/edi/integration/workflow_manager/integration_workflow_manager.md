# Integration: Workflow Manager

[Back to Integration Layer](./integration_layer.md)

## Purpose
Workflow template handler - Contains the WorkflowManager class that loads workflow templates and injects parameters, validating workflow JSON structure.

## Class: WorkflowManager

### Methods
- `load_template(name)`: Loads a workflow template by name
- `inject_params(workflow, params)`: Injects parameters into a workflow

### Details
- Validates workflow JSON structure
- Manages default parameter values
- Handles different types of ComfyUI workflows

## Functions

- [load_template(name)](./integration/load_template.md)
- [inject_params(workflow, params)](./integration/inject_params.md)

## Technology Stack

- JSON for workflow templates
- Validation utilities