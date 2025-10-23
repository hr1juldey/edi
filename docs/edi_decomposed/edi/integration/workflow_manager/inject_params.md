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

## See Docs

### Python Implementation Example
Implementation of the inject_params method for WorkflowManager:

```python
import json
from typing import Dict, Any, Union, List
from copy import deepcopy

class WorkflowManager:
    """
    Implementation of the WorkflowManager with inject_params functionality.
    """
    
    def __init__(self):
        # Define node type mappings for different parameter types
        self.node_type_mappings = {
            "positive_prompt": ["CLIPTextEncode", "Prompt", "String"],
            "negative_prompt": ["CLIPTextEncode", "Prompt", "String"],
            "image_path": ["LoadImage", "LoadImagePath"],
            "steps": ["KSampler", "Sampler", "BasicScheduler"],
            "cfg": ["KSampler", "Sampler"],
            "seed": ["KSampler", "Sampler"],
            "denoise": ["KSampler", "Sampler"],
            "strength": ["KSampler", "InpaintModelConditioning", "ControlNetApply"],
            "width": ["EmptyLatentImage", "Latent"],
            "height": ["EmptyLatentImage", "Latent"],
            "checkpoint": ["CheckpointLoaderSimple", "UNETLoader"],
        }
    
    def inject_params(self, workflow: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Injects parameters into a workflow by identifying appropriate nodes and updating their values.
        """
        # Make a deep copy to avoid modifying the original workflow
        modified_workflow = deepcopy(workflow)
        
        # Identify the nodes in the workflow that require parameter injection
        for param_name, param_value in params.items():
            # Find the appropriate node in the workflow based on parameter name
            matching_nodes = self._find_nodes_by_param_type(modified_workflow, param_name)
            
            if not matching_nodes:
                print(f"Warning: No matching nodes found for parameter '{param_name}'")
                continue
            
            # Update the node's parameters with the provided values
            for node_id, node_data in matching_nodes:
                self._update_node_parameter(node_data, param_name, param_value)
        
        # Validate that all required parameters have been properly injected
        self._validate_injection_results(modified_workflow, params)
        
        # Return the modified workflow with injected parameters
        return modified_workflow
    
    def _find_nodes_by_param_type(self, workflow: Dict[str, Any], param_type: str) -> List[tuple]:
        """
        Find nodes in the workflow that match a specific parameter type.
        """
        matching_nodes = []
        
        # Determine the node types that should handle this parameter
        target_node_types = self.node_type_mappings.get(param_type, [param_type])
        
        # Search the workflow for matching nodes
        for node_id, node_data in workflow.items():
            if isinstance(node_data, dict) and "class_type" in node_data:
                node_type = node_data["class_type"]
                
                # Check if this node type matches any of the target types
                for target_type in target_node_types:
                    if target_type.lower() in node_type.lower():
                        matching_nodes.append((node_id, node_data))
        
        return matching_nodes
    
    def _update_node_parameter(self, node_data: Dict[str, Any], param_name: str, param_value: Any) -> None:
        """
        Update a specific parameter in a node based on the parameter name.
        """
        if "inputs" not in node_data:
            node_data["inputs"] = {}
        
        # Special handling for different parameter types
        if param_name == "positive_prompt":
            # Look for positive prompt specific identifiers
            if self._is_positive_prompt_node(node_data):
                node_data["inputs"]["text"] = param_value
        
        elif param_name == "negative_prompt":
            # Look for negative prompt specific identifiers
            if self._is_negative_prompt_node(node_data):
                node_data["inputs"]["text"] = param_value
        
        elif param_name == "image_path":
            # Handle image path injection
            node_data["inputs"]["image"] = param_value
        
        elif param_name in ["steps", "cfg", "seed", "denoise", "strength", "width", "height"]:
            # Handle numeric parameters
            node_data["inputs"][param_name] = param_value
        
        elif param_name == "checkpoint":
            # Handle model checkpoint parameters
            node_data["inputs"]["ckpt_name"] = param_value
            node_data["inputs"]["config_name"] = param_value  # For some loaders
        
        else:
            # Handle custom/generic parameters
            node_data["inputs"][param_name] = param_value
    
    def _is_positive_prompt_node(self, node_data: Dict[str, Any]) -> bool:
        """
        Determine if a node is for positive prompts.
        """
        node_id = node_data.get("_meta", {}).get("title", "")
        node_type = node_data.get("class_type", "")
        
        # Check various indicators for positive prompt nodes
        positive_indicators = [
            "positive" in node_id.lower(),
            "positive" in node_type.lower(),
            "6" in node_id,  # Common default ID for positive prompt
        ]
        
        return any(positive_indicators)
    
    def _is_negative_prompt_node(self, node_data: Dict[str, Any]) -> bool:
        """
        Determine if a node is for negative prompts.
        """
        node_id = node_data.get("_meta", {}).get("title", "")
        node_type = node_data.get("class_type", "")
        
        # Check various indicators for negative prompt nodes
        negative_indicators = [
            "negative" in node_id.lower(),
            "negative" in node_type.lower(),
            "7" in node_id,  # Common default ID for negative prompt
        ]
        
        return any(negative_indicators)
    
    def _validate_injection_results(self, workflow: Dict[str, Any], params: Dict[str, Any]) -> None:
        """
        Validate that the parameter injection was successful.
        """
        # This is a basic validation - in a real implementation, you might have more specific validation
        pass
    
    def inject_params_by_node_id(self, workflow: Dict[str, Any], node_param_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Inject parameters by specifying exact node IDs and their parameters.
        """
        modified_workflow = deepcopy(workflow)
        
        for node_id, node_params in node_param_map.items():
            if node_id in modified_workflow:
                node_data = modified_workflow[node_id]
                
                if "inputs" not in node_data:
                    node_data["inputs"] = {}
                
                # Update each specified parameter
                for param_name, param_value in node_params.items():
                    node_data["inputs"][param_name] = param_value
            else:
                print(f"Warning: Node ID '{node_id}' not found in workflow")
        
        return modified_workflow

# Example usage
if __name__ == "__main__":
    # Initialize the workflow manager
    wm = WorkflowManager()
    
    # Example workflow
    sample_workflow = {
        "6": {
            "inputs": {"text": "PLACEHOLDER", "clip": ["4", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Positive Prompt"}
        },
        "7": {
            "inputs": {"text": "PLACEHOLDER", "clip": ["4", 1]},
            "class_type": "CLIPTextEncode", 
            "_meta": {"title": "Negative Prompt"}
        },
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
        }
    }
    
    # Parameters to inject
    params = {
        "positive_prompt": "a beautiful landscape with mountains and a lake",
        "negative_prompt": "blurry, low quality, distorted",
        "steps": 25,
        "cfg": 8.0,
        "seed": 98765
    }
    
    try:
        # Inject parameters
        modified_workflow = wm.inject_params(sample_workflow, params)
        
        print("Original workflow 'positive_prompt':", sample_workflow["6"]["inputs"]["text"])
        print("Modified workflow 'positive_prompt':", modified_workflow["6"]["inputs"]["text"])
        print("Modified steps:", modified_workflow["3"]["inputs"]["steps"])
        print("Modified cfg:", modified_workflow["3"]["inputs"]["cfg"])
        print("Modified seed:", modified_workflow["3"]["inputs"]["seed"])
        
    except Exception as e:
        print(f"Error during parameter injection: {e}")
```

### Advanced Parameter Injection Implementation
Enhanced parameter injection with type validation, smart matching, and comprehensive error handling:

```python
import json
from typing import Dict, Any, Union, List, Tuple, Optional
from copy import deepcopy
import re

class AdvancedWorkflowManager:
    """
    Advanced workflow manager with enhanced parameter injection capabilities.
    """
    
    def __init__(self):
        # Define complex parameter mappings with context awareness
        self.complex_param_mappings = {
            "positive_prompt": {
                "node_types": ["CLIPTextEncode"],
                "input_field": "text",
                "context": ["positive", "prompt", "6"],  # Common identifiers
                "validator": self._validate_prompt,
            },
            "negative_prompt": {
                "node_types": ["CLIPTextEncode"], 
                "input_field": "text",
                "context": ["negative", "prompt", "7"],
                "validator": self._validate_prompt,
            },
            "image_path": {
                "node_types": ["LoadImage", "LoadImagePath"],
                "input_field": "image",
                "context": ["input", "load"],
                "validator": self._validate_image_path,
            },
            "steps": {
                "node_types": ["KSampler", "SamplerCustom"],
                "input_field": "steps",
                "context": ["sampling", "step"],
                "validator": lambda x: isinstance(x, int) and 1 <= x <= 100,
            },
            "cfg": {
                "node_types": ["KSampler", "SamplerCustom"], 
                "input_field": "cfg",
                "context": ["guidance", "scale"],
                "validator": lambda x: isinstance(x, (int, float)) and 1 <= x <= 20,
            },
            "seed": {
                "node_types": ["KSampler", "SamplerCustom"],
                "input_field": "seed", 
                "context": ["random", "noise"],
                "validator": lambda x: isinstance(x, int) and x >= 0,
            },
            "denoise": {
                "node_types": ["KSampler", "SamplerCustom"],
                "input_field": "denoise",
                "context": ["strength", "noise"],
                "validator": lambda x: isinstance(x, (float, int)) and 0 <= x <= 1,
            },
            "width": {
                "node_types": ["EmptyLatentImage", "LatentFromBatch"],
                "input_field": "width",
                "context": ["dimension", "size"],
                "validator": lambda x: isinstance(x, int) and 64 <= x <= 2048 and x % 8 == 0,
            },
            "height": {
                "node_types": ["EmptyLatentImage", "LatentFromBatch"],
                "input_field": "height",
                "context": ["dimension", "size"],
                "validator": lambda x: isinstance(x, int) and 64 <= x <= 2048 and x % 8 == 0,
            },
        }
    
    def _validate_prompt(self, value: str) -> bool:
        """Validate that a value is a reasonable prompt."""
        if not isinstance(value, str):
            return False
        # Check for reasonable length and basic content
        return 1 <= len(value) <= 1000 and len(value.split()) >= 1
    
    def _validate_image_path(self, value: str) -> bool:
        """Validate that a value is a reasonable image path."""
        if not isinstance(value, str):
            return False
        # Check for common image extensions
        return re.search(r'\.(jpg|jpeg|png|bmp|tiff|webp), value.lower()) is not None
    
    def inject_params(self, 
                     workflow: Dict[str, Any], 
                     params: Dict[str, Any], 
                     strict_validation: bool = True) -> Dict[str, Any]:
        """
        Enhanced parameter injection with strict validation and context awareness.
        """
        # Make a deep copy to avoid modifying the original
        modified_workflow = deepcopy(workflow)
        
        injection_results = {
            "success": [],
            "failed": [],
            "warnings": []
        }
        
        # Process each parameter
        for param_name, param_value in params.items():
            if param_name not in self.complex_param_mappings:
                injection_results["warnings"].append(f"Unknown parameter type: {param_name}")
                continue
            
            mapping = self.complex_param_mappings[param_name]
            
            # Find matching nodes based on type and context
            matching_nodes = self._find_nodes_with_context(
                modified_workflow, 
                mapping["node_types"], 
                mapping["context"]
            )
            
            if not matching_nodes:
                error_msg = f"No matching nodes found for parameter '{param_name}' (type: {', '.join(mapping['node_types'])})"
                injection_results["failed"].append(error_msg)
                if strict_validation:
                    raise ValueError(error_msg)
                continue
            
            # Validate parameter value if validator exists
            if "validator" in mapping and mapping["validator"]:
                if not mapping["validator"](param_value):
                    error_msg = f"Invalid value for parameter '{param_name}': {param_value}"
                    injection_results["failed"].append(error_msg)
                    if strict_validation:
                        raise ValueError(error_msg)
                    continue
            
            # Update all matching nodes with the parameter
            for node_id, node_data in matching_nodes:
                try:
                    self._update_node_with_validation(
                        node_data, 
                        mapping["input_field"], 
                        param_value, 
                        mapping.get("validator")
                    )
                    
                    injection_results["success"].append({
                        "param": param_name,
                        "node_id": node_id,
                        "value": param_value
                    })
                    
                except Exception as e:
                    error_msg = f"Failed to update node {node_id} with {param_name}: {str(e)}"
                    injection_results["failed"].append(error_msg)
                    if strict_validation:
                        raise ValueError(error_msg)
        
        # Log results
        if injection_results["failed"]:
            print(f"Parameter injection completed with errors: {injection_results['failed']}")
        elif injection_results["warnings"]:
            print(f"Parameter injection completed with warnings: {injection_results['warnings']}")
        else:
            print("All parameter injections completed successfully")
        
        return modified_workflow
    
    def _find_nodes_with_context(
        self, 
        workflow: Dict[str, Any], 
        node_types: List[str], 
        context_hints: List[str]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Find nodes by both type and contextual hints.
        """
        matching_nodes = []
        
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict) or "class_type" not in node_data:
                continue
            
            node_type = node_data["class_type"]
            
            # Check if node type matches
            type_match = any(target_type.lower() in node_type.lower() for target_type in node_types)
            
            # Check context hints in node title, ID, or other properties
            context_match = False
            node_title = node_data.get("_meta", {}).get("title", "").lower()
            node_id_lower = node_id.lower()
            
            for hint in context_hints:
                if (hint.lower() in node_type.lower() or 
                    hint.lower() in node_title or 
                    hint.lower() in node_id_lower):
                    context_match = True
                    break
            
            if type_match or context_match:
                matching_nodes.append((node_id, node_data))
        
        return matching_nodes
    
    def _update_node_with_validation(
        self, 
        node_data: Dict[str, Any], 
        input_field: str, 
        value: Any, 
        validator: Optional[callable] = None
    ) -> None:
        """
        Update a node's input with validation.
        """
        if "inputs" not in node_data:
            node_data["inputs"] = {}
        
        if validator and not validator(value):
            raise ValueError(f"Validation failed for value: {value}")
        
        node_data["inputs"][input_field] = value
    
    def smart_inject_params(self, workflow: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform smart injection with automatic node detection and parameter mapping.
        """
        modified_workflow = deepcopy(workflow)
        
        # Create a map of nodes by their class type for quick lookup
        nodes_by_type = {}
        for node_id, node_data in modified_workflow.items():
            if isinstance(node_data, dict) and "class_type" in node_data:
                class_type = node_data["class_type"]
                if class_type not in nodes_by_type:
                    nodes_by_type[class_type] = []
                nodes_by_type[class_type].append((node_id, node_data))
        
        # Process each parameter with smart detection
        for param_name, param_value in params.items():
            injected = False
            
            # Try exact matching first
            if param_name in self.complex_param_mappings:
                mapping = self.complex_param_mappings[param_name]
                matching_nodes = self._find_nodes_with_context(
                    modified_workflow, 
                    mapping["node_types"], 
                    mapping["context"]
                )
                
                for node_id, node_data in matching_nodes:
                    if "inputs" not in node_data:
                        node_data["inputs"] = {}
                    node_data["inputs"][mapping["input_field"]] = param_value
                    injected = True
            
            # If not injected through mapping, try value-based detection
            if not injected:
                injected = self._try_value_based_injection(
                    modified_workflow, 
                    param_name, 
                    param_value, 
                    nodes_by_type
                )
            
            if not injected:
                print(f"Warning: Could not inject parameter '{param_name}' with value '{param_value}'")
        
        return modified_workflow
    
    def _try_value_based_injection(
        self, 
        workflow: Dict[str, Any], 
        param_name: str, 
        param_value: Any, 
        nodes_by_type: Dict[str, List[Tuple[str, Dict[str, Any]]]]
    ) -> bool:
        """
        Try to inject a parameter based on the value type and common usage patterns.
        """
        # For string values that look like prompts
        if isinstance(param_value, str) and len(param_value) > 10:
            # Look for CLIPTextEncode nodes
            if "CLIPTextEncode" in nodes_by_type:
                for node_id, node_data in nodes_by_type["CLIPTextEncode"]:
                    if "inputs" not in node_data:
                        node_data["inputs"] = {}
                    
                    # Decide whether to treat as positive or negative based on param name
                    if "positive" in param_name.lower() or param_name == "prompt":
                        node_data["inputs"]["text"] = param_value
                    elif "negative" in param_name.lower():
                        node_data["inputs"]["text"] = param_value
                    else:
                        # Default to positive if ambiguous
                        node_data["inputs"]["text"] = param_value
                    
                    return True
        
        # For numeric values that might be steps, cfg, etc.
        elif isinstance(param_value, (int, float)):
            # Look for sampler nodes for parameters like steps, cfg
            sampler_nodes = []
            for node_type in ["KSampler", "SamplerCustom"]:
                if node_type in nodes_by_type:
                    sampler_nodes.extend(nodes_by_type[node_type])
            
            for node_id, node_data in sampler_nodes:
                if "inputs" not in node_data:
                    node_data["inputs"] = {}
                
                # Map based on parameter name
                if param_name == "steps":
                    node_data["inputs"]["steps"] = int(param_value)
                    return True
                elif param_name == "cfg":
                    node_data["inputs"]["cfg"] = float(param_value)
                    return True
                elif param_name == "seed":
                    node_data["inputs"]["seed"] = int(param_value)
                    return True
                elif param_name == "denoise":
                    node_data["inputs"]["denoise"] = float(param_value)
                    return True
        
        return False

# Example usage
if __name__ == "__main__":
    # Initialize the advanced workflow manager
    awm = AdvancedWorkflowManager()
    
    # Example workflow
    sample_workflow = {
        "6": {
            "inputs": {"text": "PLACEHOLDER", "clip": ["4", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Positive Prompt"}
        },
        "7": {
            "inputs": {"text": "PLACEHOLDER", "clip": ["4", 1]},
            "class_type": "CLIPTextEncode", 
            "_meta": {"title": "Negative Prompt"}
        },
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
        }
    }
    
    # Parameters to inject
    params = {
        "positive_prompt": "a beautiful landscape with mountains and a lake at sunset",
        "negative_prompt": "blurry, low quality, distorted, extra limbs",
        "steps": 25,
        "cfg": 8.0,
        "seed": 98765
    }
    
    try:
        # Use smart injection which can match based on context
        modified_workflow = awm.smart_inject_params(sample_workflow, params)
        
        print("Smart Parameter Injection Results:")
        print("Positive prompt:", modified_workflow["6"]["inputs"]["text"])
        print("Negative prompt:", modified_workflow["7"]["inputs"]["text"])
        print("Steps:", modified_workflow["3"]["inputs"]["steps"])  
        print("CFG:", modified_workflow["3"]["inputs"]["cfg"])
        print("Seed:", modified_workflow["3"]["inputs"]["seed"])
        
    except Exception as e:
        print(f"Error during parameter injection: {e}")
```