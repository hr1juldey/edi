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

## See Docs

### JSON Implementation Example
Workflow manager implementation for EDI:

```python
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import os
import re

class WorkflowValidationError(Exception):
    """Exception raised for workflow validation errors."""
    pass

class WorkflowManager:
    """
    Workflow template handler that loads workflow templates and injects parameters,
    validating workflow JSON structure.
    """
    
    def __init__(self, workflows_dir: str = "workflows"):
        self.workflows_dir = Path(workflows_dir)
        self.workflows_dir.mkdir(parents=True, exist_ok=True)
        
        # Define expected workflow structure
        self.required_keys = {"class_type", "inputs", "_meta"}
        self.workflow_schema = self._get_default_schema()
    
    def _get_default_schema(self) -> Dict[str, Any]:
        """
        Returns a default schema for validating workflows.
        """
        return {
            "required_nodes": ["KSampler", "CheckpointLoaderSimple", "VAEDecode"],
            "required_inputs": {
                "KSampler": {"seed", "steps", "cfg", "sampler_name", "scheduler", "denoise", "model", "positive", "negative"},
                "CheckpointLoaderSimple": {"ckpt_name"},
                "VAEDecode": {"samples", "vae"}
            },
            "required_fields": {"class_type", "inputs"}
        }
    
    def load_template(self, name: str) -> Dict[str, Any]:
        """
        Loads a workflow template by name.
        """
        workflow_path = self.workflows_dir / f"{name}.json"
        
        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow template '{name}' not found at {workflow_path}")
        
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
        except json.JSONDecodeError as e:
            raise WorkflowValidationError(f"Invalid JSON in workflow '{name}': {str(e)}")
        
        # Validate the loaded workflow
        self.validate_workflow(workflow, name)
        
        return workflow
    
    def validate_workflow(self, workflow: Dict[str, Any], name: str = "unknown") -> bool:
        """
        Validates workflow JSON structure against expected schema.
        """
        if not isinstance(workflow, dict):
            raise WorkflowValidationError(f"Workflow '{name}' is not a valid JSON object")
        
        # Check that workflow has at least one node
        if not workflow:
            raise WorkflowValidationError(f"Workflow '{name}' is empty")
        
        # Validate each node in the workflow
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                raise WorkflowValidationError(f"Node '{node_id}' in workflow '{name}' is not a valid JSON object")
            
            # Check for required keys
            for required_key in self.required_keys:
                if required_key not in node_data:
                    raise WorkflowValidationError(f"Node '{node_id}' in workflow '{name}' missing required key: {required_key}")
        
        # Check for required node types
        class_types = {node_data["class_type"] for node_data in workflow.values() if "class_type" in node_data}
        required_nodes = set(self.workflow_schema["required_nodes"])
        missing_nodes = required_nodes - class_types
        
        if missing_nodes:
            raise WorkflowValidationError(f"Workflow '{name}' missing required nodes: {missing_nodes}")
        
        print(f"Workflow '{name}' validated successfully")
        return True
    
    def validate_workflow_complete(self, workflow: Dict[str, Any], name: str = "unknown") -> Dict[str, List[str]]:
        """
        Performs comprehensive validation of workflow structure and content.
        Returns a dictionary of validation results.
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        if not isinstance(workflow, dict):
            results["valid"] = False
            results["errors"].append(f"Workflow '{name}' is not a valid JSON object")
            return results
        
        if not workflow:
            results["valid"] = False
            results["errors"].append(f"Workflow '{name}' is empty")
            return results
        
        # Track all connection references to verify they exist
        referenced_nodes = set()
        defined_nodes = set(workflow.keys())
        
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                results["valid"] = False
                results["errors"].append(f"Node '{node_id}' in workflow '{name}' is not a valid JSON object")
                continue
            
            # Check for required keys
            for required_key in self.required_keys:
                if required_key not in node_data:
                    results["valid"] = False
                    results["errors"].append(f"Node '{node_id}' in workflow '{name}' missing required key: {required_key}")
            
            # Validate inputs
            if "inputs" in node_data and isinstance(node_data["inputs"], dict):
                for input_key, input_value in node_data["inputs"].items():
                    # Check if input value is a reference to another node
                    if isinstance(input_value, list) and len(input_value) == 2:
                        ref_node_id, ref_output_idx = input_value
                        referenced_nodes.add(str(ref_node_id))
        
        # Check if all referenced nodes exist
        missing_references = referenced_nodes - defined_nodes
        if missing_references:
            results["valid"] = False
            results["errors"].append(f"Workflow '{name}' has references to non-existent nodes: {missing_references}")
        
        # Check for required node types
        class_types = {node_data["class_type"] for node_data in workflow.values() 
                      if isinstance(node_data, dict) and "class_type" in node_data}
        required_nodes = set(self.workflow_schema["required_nodes"])
        missing_nodes = required_nodes - class_types
        
        if missing_nodes:
            results["warnings"].append(f"Workflow '{name}' missing recommended nodes: {missing_nodes}")
        
        return results
    
    def inject_params(self, workflow: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Injects parameters into a workflow.
        """
        # Make a copy of the workflow to avoid modifying the original
        modified_workflow = json.loads(json.dumps(workflow))  # Deep copy via JSON
        
        # Handle node-specific parameter injection
        for node_id, node_data in modified_workflow.items():
            if not isinstance(node_data, dict):
                continue
            
            class_type = node_data.get("class_type", "")
            
            # Handle specific node types
            if "CLIPTextEncode" in class_type:
                # Inject text prompts
                if "positive" in node_id.lower() or "6" == node_id:
                    if "positive_prompt" in params:
                        node_data["inputs"]["text"] = params["positive_prompt"]
                elif "negative" in node_id.lower() or "7" == node_id:
                    if "negative_prompt" in params:
                        node_data["inputs"]["text"] = params["negative_prompt"]
            
            elif "KSampler" in class_type:
                # Inject sampler parameters
                for param_name in ["seed", "steps", "cfg", "denoise"]:
                    if param_name in params:
                        node_data["inputs"][param_name] = params[param_name]
                
                if "sampler_name" in params:
                    node_data["inputs"]["sampler_name"] = params["sampler_name"]
            
            elif "EmptyLatentImage" in class_type:
                # Inject image dimensions
                for dim_name in ["width", "height"]:
                    if dim_name in params:
                        node_data["inputs"][dim_name] = params[dim_name]
        
        # Handle general parameter injection by searching for matching keys
        self._inject_general_params(modified_workflow, params)
        
        return modified_workflow
    
    def _inject_general_params(self, workflow: Dict[str, Any], params: Dict[str, Any]) -> None:
        """
        Injects parameters that don't match specific node types.
        """
        for param_name, param_value in params.items():
            # Look for this parameter in any node's inputs
            for node_data in workflow.values():
                if isinstance(node_data, dict) and "inputs" in node_data:
                    if param_name in node_data["inputs"]:
                        node_data["inputs"][param_name] = param_value
    
    def save_template(self, name: str, workflow: Dict[str, Any]) -> None:
        """
        Saves a workflow template to a file.
        """
        # Validate before saving
        self.validate_workflow(workflow, name)
        
        workflow_path = self.workflows_dir / f"{name}.json"
        
        with open(workflow_path, 'w', encoding='utf-8') as f:
            json.dump(workflow, f, indent=2, ensure_ascii=False)
    
    def list_templates(self) -> List[str]:
        """
        Lists all available workflow templates.
        """
        templates = []
        for file_path in self.workflows_dir.glob("*.json"):
            templates.append(file_path.stem)
        return templates
    
    def create_default_workflows(self):
        """
        Creates default workflow templates if they don't exist.
        """
        default_workflows = {
            "img2img_default": self._get_default_img2img_workflow(),
            "inpaint_masked": self._get_default_inpaint_workflow(),
            "txt2img_default": self._get_default_txt2img_workflow()
        }
        
        for name, workflow in default_workflows.items():
            workflow_path = self.workflows_dir / f"{name}.json"
            if not workflow_path.exists():
                self.save_template(name, workflow)
    
    def _get_default_img2img_workflow(self) -> Dict[str, Any]:
        """
        Returns a default image-to-image workflow.
        """
        return {
            "3": {
                "inputs": {
                    "seed": 12345,
                    "steps": 20,
                    "cfg": 7.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 0.7,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                },
                "class_type": "KSampler",
                "_meta": {"title": "KSampler"}
            },
            "4": {
                "inputs": {"ckpt_name": "model.safetensors"},
                "class_type": "CheckpointLoaderSimple",
                "_meta": {"title": "Load Checkpoint"}
            },
            "5": {
                "inputs": {
                    "width": 512,
                    "height": 512,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage",
                "_meta": {"title": "Empty Latent Image"}
            },
            "6": {
                "inputs": {"text": "", "clip": ["4", 1]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive Prompt"}
            },
            "7": {
                "inputs": {"text": "", "clip": ["4", 1]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Negative Prompt"}
            }
        }
    
    def _get_default_inpaint_workflow(self) -> Dict[str, Any]:
        """
        Returns a default inpainting workflow.
        """
        return {
            "3": {
                "inputs": {
                    "seed": 12345,
                    "steps": 20,
                    "cfg": 7.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 0.8,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["10", 0]
                },
                "class_type": "KSampler",
                "_meta": {"title": "KSampler"}
            },
            "4": {
                "inputs": {"ckpt_name": "model.safetensors"},
                "class_type": "CheckpointLoaderSimple",
                "_meta": {"title": "Load Checkpoint"}
            },
            "6": {
                "inputs": {"text": "", "clip": ["4", 1]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive Prompt"}
            },
            "7": {
                "inputs": {"text": "", "clip": ["4", 1]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Negative Prompt"}
            },
            "8": {
                "inputs": {
                    "image": "",
                    "upload": "image"
                },
                "class_type": "LoadImage",
                "_meta": {"title": "Load Image"}
            },
            "9": {
                "inputs": {
                    "image": "",
                    "upload": "image"
                },
                "class_type": "LoadImageMask",
                "_meta": {"title": "Load Mask"}
            },
            "10": {
                "inputs": {
                    "samples": ["9", 0],
                    "mask": ["9", 1]
                },
                "class_type": "SetLatentNoiseMask",
                "_meta": {"title": "Set Latent Noise Mask"}
            }
        }
    
    def _get_default_txt2img_workflow(self) -> Dict[str, Any]:
        """
        Returns a default text-to-image workflow.
        """
        return {
            "3": {
                "inputs": {
                    "seed": 12345,
                    "steps": 20,
                    "cfg": 7.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                },
                "class_type": "KSampler",
                "_meta": {"title": "KSampler"}
            },
            "4": {
                "inputs": {"ckpt_name": "model.safetensors"},
                "class_type": "CheckpointLoaderSimple",
                "_meta": {"title": "Load Checkpoint"}
            },
            "5": {
                "inputs": {
                    "width": 512,
                    "height": 512,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage",
                "_meta": {"title": "Empty Latent Image"}
            },
            "6": {
                "inputs": {"text": "", "clip": ["4", 1]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive Prompt"}
            },
            "7": {
                "inputs": {"text": "", "clip": ["4", 1]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Negative Prompt"}
            }
        }

# Example usage
if __name__ == "__main__":
    # Initialize the workflow manager
    wm = WorkflowManager()
    
    # Create default workflows if they don't exist
    wm.create_default_workflows()
    
    # List available templates
    print("Available templates:", wm.list_templates())
    
    # Load a template
    try:
        workflow = wm.load_template("img2img_default")
        print(f"Loaded workflow with {len(workflow)} nodes")
        
        # Inject parameters
        params = {
            "positive_prompt": "a beautiful landscape with mountains",
            "negative_prompt": "blurry, low quality",
            "seed": 98765,
            "steps": 25
        }
        
        modified_workflow = wm.inject_params(workflow, params)
        print("Parameters injected successfully")
        
        # Validate the modified workflow
        validation_results = wm.validate_workflow_complete(modified_workflow, "modified_img2img")
        print(f"Validation results: {validation_results}")
        
    except WorkflowValidationError as e:
        print(f"Validation error: {e}")
    except Exception as e:
        print(f"Error: {e}")
```

### Validation Utilities Implementation Example
Advanced validation utilities for workflow management:

```python
import json
from typing import Dict, Any, List, Tuple
import re
from pathlib import Path

class WorkflowValidator:
    """
    Advanced validation utilities for workflow management.
    """
    
    @staticmethod
    def validate_json_structure(data: Any, schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates JSON structure against a schema.
        """
        errors = []
        
        def validate_recursive(obj, schema_obj, path=""):
            if isinstance(schema_obj, dict):
                if not isinstance(obj, dict):
                    errors.append(f"{path}: Expected object, got {type(obj).__name__}")
                    return
                
                # Check required keys
                if "required" in schema_obj:
                    for req_key in schema_obj["required"]:
                        if req_key not in obj:
                            errors.append(f"{path}.{req_key}: Required field missing")
                
                # Validate properties
                if "properties" in schema_obj:
                    for key, prop_schema in schema_obj["properties"].items():
                        if key in obj:
                            validate_recursive(obj[key], prop_schema, f"{path}.{key}")
                        elif "default" in prop_schema:
                            # Optional field with default, which is valid
                            pass
                        elif "required" in schema_obj.get("properties", {}):
                            # If it's not in the required list, it's optional
                            pass
            
            elif isinstance(schema_obj, list):
                if not isinstance(obj, list):
                    errors.append(f"{path}: Expected array, got {type(obj).__name__}")
                    return
                
                item_schema = schema_obj[0] if schema_obj else {}
                for i, item in enumerate(obj):
                    validate_recursive(item, item_schema, f"{path}[{i}]")
        
        validate_recursive(data, schema)
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_comfyui_workflow(workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates a ComfyUI workflow against specific rules.
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "node_types": set(),
            "connection_issues": [],
            "parameter_issues": []
        }
        
        if not isinstance(workflow, dict):
            results["valid"] = False
            results["errors"].append("Workflow must be a dictionary")
            return results
        
        if not workflow:
            results["valid"] = False
            results["errors"].append("Workflow is empty")
            return results
        
        # Track nodes and connections
        node_ids = set(workflow.keys())
        connected_nodes = set()
        
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                results["valid"] = False
                results["errors"].append(f"Node '{node_id}' is not an object")
                continue
            
            # Track node types
            if "class_type" in node_data:
                results["node_types"].add(node_data["class_type"])
            
            # Validate node structure
            required_fields = ["class_type", "inputs"]
            for field in required_fields:
                if field not in node_data:
                    results["errors"].append(f"Node '{node_id}' missing required field: {field}")
            
            # Validate inputs
            if "inputs" in node_data and isinstance(node_data["inputs"], dict):
                for input_name, input_value in node_data["inputs"].items():
                    # Check if input is a connection to another node
                    if isinstance(input_value, list) and len(input_value) == 2:
                        ref_node_id, output_idx = input_value
                        connected_nodes.add(str(ref_node_id))
                        
                        # Validate that referenced node exists
                        if str(ref_node_id) not in node_ids:
                            results["connection_issues"].append(
                                f"Node '{node_id}' references non-existent node '{ref_node_id}'"
                            )
        
        # Check for orphaned nodes (not connected to the main flow)
        unused_nodes = node_ids - connected_nodes - {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}  # Common starting nodes
        if unused_nodes:
            results["warnings"].append(f"Potential orphaned nodes: {unused_nodes}")
        
        # Validate required node types for a complete workflow
        required_types = {"KSampler", "CheckpointLoaderSimple", "VAEDecode"}
        missing_types = required_types - results["node_types"]
        
        if missing_types:
            results["errors"].append(f"Missing required node types: {missing_types}")
        
        # Check for common parameter issues
        for node_id, node_data in workflow.items():
            if node_data.get("class_type") == "KSampler":
                required_inputs = {"steps", "cfg", "denoise", "model", "positive", "negative"}
                missing_inputs = required_inputs - set(node_data.get("inputs", {}).keys())
                if missing_inputs:
                    results["parameter_issues"].append(
                        f"KSampler node '{node_id}' missing inputs: {missing_inputs}"
                    )
        
        results["valid"] = len(results["errors"]) == 0
        return results
    
    @staticmethod
    def analyze_workflow_complexity(workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the complexity of a workflow.
        """
        stats = {
            "total_nodes": 0,
            "node_types": {},
            "total_connections": 0,
            "max_depth": 0,
            "estimated_execution_time": 0
        }
        
        if not isinstance(workflow, dict):
            return stats
        
        # Count nodes and types
        for node_id, node_data in workflow.items():
            if isinstance(node_data, dict) and "class_type" in node_data:
                stats["total_nodes"] += 1
                node_type = node_data["class_type"]
                stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1
        
        # Count connections
        for node_data in workflow.values():
            if isinstance(node_data, dict) and "inputs" in node_data:
                for input_value in node_data["inputs"].values():
                    if isinstance(input_value, list) and len(input_value) == 2:
                        stats["total_connections"] += 1
        
        # Estimate execution time based on node types
        node_time_weights = {
            "KSampler": 30,  # 30 seconds for sampling
            "VAEEncode": 2,  # 2 seconds for encoding
            "VAEDecode": 2,  # 2 seconds for decoding
            "LoadImage": 1,  # 1 second for loading
            "SaveImage": 1,  # 1 second for saving
        }
        
        for node_type, count in stats["node_types"].items():
            weight = node_time_weights.get(node_type, 0.5)  # Default 0.5 seconds
            stats["estimated_execution_time"] += weight * count
        
        return stats

class AdvancedWorkflowManager(WorkflowManager):
    """
    Extended workflow manager with advanced validation and analysis capabilities.
    """
    
    def __init__(self, workflows_dir: str = "workflows"):
        super().__init__(workflows_dir)
        self.validator = WorkflowValidator()
    
    def load_and_validate(self, name: str) -> Dict[str, Any]:
        """
        Loads a workflow and performs comprehensive validation.
        """
        try:
            workflow = self.load_template(name)
            
            # Perform advanced validation
            validation_results = self.validator.validate_comfyui_workflow(workflow)
            
            # Perform complexity analysis
            complexity_stats = self.validator.analyze_workflow_complexity(workflow)
            
            return {
                "workflow": workflow,
                "validation": validation_results,
                "complexity": complexity_stats,
                "name": name
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "name": name
            }
    
    def validate_and_fix_workflow(self, workflow: Dict[str, Any], auto_fix: bool = True) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validates a workflow and optionally fixes common issues.
        """
        validation_results = self.validator.validate_comfyui_workflow(workflow)
        fixes_applied = []
        
        if auto_fix and validation_results["connection_issues"]:
            # Try to fix connection issues by removing invalid references
            for issue in validation_results["connection_issues"]:
                # Parse the issue to extract node and reference info
                import re
                match = re.search(r"Node '(\d+)' references non-existent node '(\d+)'", issue)
                if match:
                    source_node = match.group(1)
                    target_node = match.group(2)
                    
                    # Remove the invalid connection from the source node
                    if source_node in workflow and "inputs" in workflow[source_node]:
                        for input_name, input_value in list(workflow[source_node]["inputs"].items()):
                            if (
                                isinstance(input_value, list) and 
                                len(input_value) == 2 and 
                                str(input_value[0]) == target_node
                            ):
                                del workflow[source_node]["inputs"][input_name]
                                fixes_applied.append(
                                    f"Removed invalid connection from node {source_node} to {target_node} in input {input_name}"
                                )
        
        return workflow, fixes_applied

# Example usage
if __name__ == "__main__":
    # Initialize the advanced workflow manager
    awm = AdvancedWorkflowManager()
    
    # Create default workflows
    awm.create_default_workflows()
    
    # Load and validate a workflow
    result = awm.load_and_validate("img2img_default")
    
    if "error" not in result:
        print(f"Workflow: {result['name']}")
        print(f"Validation valid: {result['validation']['valid']}")
        print(f"Node types: {result['validation']['node_types']}")
        print(f"Complexity: {result['complexity']['total_nodes']} nodes, estimated time: {result['complexity']['estimated_execution_time']:.1f}s")
        
        if result['validation']['errors']:
            print(f"Validation errors: {result['validation']['errors']}")
        if result['validation']['warnings']:
            print(f"Validation warnings: {result['validation']['warnings']}")
    else:
        print(f"Error loading workflow: {result['error']}")
```