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

## See Docs

### Python Implementation Example
Implementation of the load_template method for WorkflowManager:

```python
import json
from pathlib import Path
from typing import Dict, Any, Union
import os
import re

class WorkflowValidationError(Exception):
    """Exception raised for workflow validation errors."""
    pass

class WorkflowManager:
    """
    Implementation of the WorkflowManager with load_template functionality.
    """
    
    def __init__(self, workflows_dir: str = "workflows"):
        self.workflows_dir = Path(workflows_dir)
        self.workflows_dir.mkdir(parents=True, exist_ok=True)
        
        # Define required nodes for different template types
        self.required_nodes_by_type = {
            "img2img": {"KSampler", "CheckpointLoaderSimple", "VAEDecode"},
            "inpaint": {"KSampler", "CheckpointLoaderSimple", "VAEDecode", "SetLatentNoiseMask"},
            "txt2img": {"KSampler", "CheckpointLoaderSimple", "VAEDecode"},
            "default": {"KSampler", "CheckpointLoaderSimple", "VAEDecode"}
        }
    
    def load_template(self, name: str) -> Dict[str, Any]:
        """
        Loads a workflow template by name.
        """
        # Validate template name for security (prevent directory traversal)
        if not re.match(r'^[a-zA-Z0-9_-]+, name):
            raise ValueError(f"Invalid template name: {name}")
        
        # Determine the path to the workflow template file
        workflow_path = self.workflows_dir / f"{name}.json"
        
        # Validate that the workflow file exists and is accessible
        if not workflow_path.exists():
            available_templates = self._get_available_templates()
            raise FileNotFoundError(
                f"Workflow template '{name}' not found at {workflow_path}. "
                f"Available templates: {available_templates}"
            )
        
        if not os.access(workflow_path, os.R_OK):
            raise PermissionError(f"Cannot read workflow file: {workflow_path}")
        
        # Read and parse the JSON from the workflow file
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
        except json.JSONDecodeError as e:
            raise WorkflowValidationError(f"Invalid JSON in workflow '{name}': {str(e)}")
        except UnicodeDecodeError as e:
            raise WorkflowValidationError(f"Invalid character encoding in workflow '{name}': {str(e)}")
        
        # Validate the structure of the workflow to ensure it's valid for ComfyUI
        self._validate_workflow_structure(workflow, name)
        
        # Return the workflow template as a dictionary
        return workflow
    
    def _get_available_templates(self) -> list:
        """Get list of available template files."""
        templates = []
        for file_path in self.workflows_dir.glob("*.json"):
            templates.append(file_path.stem)
        return sorted(templates)
    
    def _validate_workflow_structure(self, workflow: Dict[str, Any], name: str) -> None:
        """
        Validates the structure of the workflow to ensure it's valid for ComfyUI.
        """
        if not isinstance(workflow, dict):
            raise WorkflowValidationError(f"Workflow '{name}' is not a valid JSON object")
        
        if not workflow:
            raise WorkflowValidationError(f"Workflow '{name}' is empty")
        
        # Check that each node has required properties
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                raise WorkflowValidationError(f"Node '{node_id}' in workflow '{name}' is not a valid JSON object")
            
            # Verify required keys exist
            required_keys = {"class_type"}
            for key in required_keys:
                if key not in node_data:
                    raise WorkflowValidationError(f"Node '{node_id}' missing required key: {key}")
            
            # Verify inputs exists (even if empty)
            if "inputs" not in node_data:
                node_data["inputs"] = {}  # Add empty inputs if missing
        
        # Determine workflow type based on name to select appropriate validation
        workflow_type = self._determine_workflow_type(name)
        required_nodes = self.required_nodes_by_type.get(workflow_type, self.required_nodes_by_type["default"])
        
        # Check that required node types exist in the workflow
        found_node_types = set()
        for node_data in workflow.values():
            if "class_type" in node_data:
                found_node_types.add(node_data["class_type"])
        
        missing_nodes = required_nodes - found_node_types
        if missing_nodes:
            raise WorkflowValidationError(
                f"Workflow '{name}' of type '{workflow_type}' is missing required node types: {missing_nodes}"
            )
    
    def _determine_workflow_type(self, name: str) -> str:
        """
        Determines the type of workflow based on the name.
        """
        if "img2img" in name.lower():
            return "img2img"
        elif "inpaint" in name.lower():
            return "inpaint"
        elif "txt2img" in name.lower():
            return "txt2img"
        else:
            return "default"
    
    def get_template_metadata(self, name: str) -> Dict[str, Any]:
        """
        Gets metadata about a template without loading the full content.
        """
        workflow_path = self.workflows_dir / f"{name}.json"
        
        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow template '{name}' not found")
        
        # Get file stats
        stat = workflow_path.stat()
        
        # Load just enough to get node count and types
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        
        node_types = {}
        for node_data in workflow.values():
            if isinstance(node_data, dict) and "class_type" in node_data:
                node_type = node_data["class_type"]
                node_types[node_type] = node_types.get(node_type, 0) + 1
        
        return {
            "name": name,
            "path": str(workflow_path),
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "node_count": len(workflow),
            "node_types": node_types,
            "workflow_type": self._determine_workflow_type(name)
        }
    
    def validate_template_exists(self, name: str) -> bool:
        """
        Checks if a template exists without loading it.
        """
        workflow_path = self.workflows_dir / f"{name}.json"
        return workflow_path.exists() and workflow_path.is_file()

# Example usage
if __name__ == "__main__":
    # Initialize the workflow manager
    wm = WorkflowManager()
    
    # Example: Create a simple workflow file to load
    sample_workflow = {
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
    
    # Save the sample workflow
    sample_path = wm.workflows_dir / "sample_template.json"
    with open(sample_path, 'w') as f:
        json.dump(sample_workflow, f, indent=2)
    
    try:
        # Load the template
        loaded_workflow = wm.load_template("sample_template")
        print(f"Successfully loaded workflow with {len(loaded_workflow)} nodes")
        
        # Get metadata
        metadata = wm.get_template_metadata("sample_template")
        print(f"Template metadata: {metadata}")
        
        # Check if template exists
        exists = wm.validate_template_exists("sample_template")
        print(f"Template exists: {exists}")
        
    except (FileNotFoundError, WorkflowValidationError, ValueError) as e:
        print(f"Error: {e}")
```

### Advanced Loading Implementation
Enhanced template loading with caching, security checks, and comprehensive validation:

```python
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Set
from datetime import datetime, timedelta
import os
import re
import time

class AdvancedWorkflowManager:
    """
    Advanced workflow manager with caching, security checks, and comprehensive validation.
    """
    
    def __init__(self, workflows_dir: str = "workflows", cache_enabled: bool = True):
        self.workflows_dir = Path(workflows_dir)
        self.workflows_dir.mkdir(parents=True, exist_ok=True)
        
        # Caching for loaded templates
        self.cache_enabled = cache_enabled
        self.template_cache = {}
        self.template_timestamps = {}
        
        # Security: Define allowed file patterns
        self.allowed_name_pattern = re.compile(r'^[a-zA-Z0-9_-]+)
        
        # Safety limits
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.max_json_depth = 100
        self.max_node_count = 1000
        
        # Define allowed node types for security
        self.allowed_node_types = {
            # Core nodes
            "KSampler", "CheckpointLoaderSimple", "VAEDecode", "VAEEncode", "EmptyLatentImage",
            # Text encoders
            "CLIPTextEncode",
            # Image loaders
            "LoadImage", "LoadImageMask",
            # Common samplers
            "SamplerCustom", "BasicScheduler",
            # Utilities
            "Reroute", "PrimitiveNode", "Note",
            # Common custom nodes
            "ControlNetApply", "ControlNetLoader", 
            "LoraLoader", "UpscaleModelLoader"
        }
    
    def load_template(self, name: str, validate_comprehensive: bool = True) -> Dict[str, Any]:
        """
        Load a workflow template with security checks and validation.
        """
        # Security check: validate template name
        if not self._is_valid_template_name(name):
            raise ValueError(f"Invalid template name: {name}. Only alphanumeric characters, hyphens, and underscores are allowed.")
        
        # Check cache first if enabled
        if self.cache_enabled and name in self.template_cache:
            cached_data, cached_time = self.template_cache[name]
            
            # Check if file has been modified since caching
            workflow_path = self.workflows_dir / f"{name}.json"
            if workflow_path.exists():
                file_modified = workflow_path.stat().st_mtime
                if file_modified <= cached_time:
                    print(f"Loading '{name}' from cache")
                    return cached_data
        
        workflow_path = self.workflows_dir / f"{name}.json"
        
        # Validate file exists and is secure to access
        if not workflow_path.exists():
            available_templates = self._get_available_templates()
            raise FileNotFoundError(
                f"Workflow template '{name}' not found. Available: {available_templates}"
            )
        
        # Security: Verify path is within allowed directory
        try:
            # This will raise ValueError if path is not within workflows_dir
            workflow_path.resolve().relative_to(self.workflows_dir.resolve())
        except ValueError:
            raise ValueError(f"Path traversal detected: {workflow_path}")
        
        # Check file size
        file_size = workflow_path.stat().st_size
        if file_size > self.max_file_size:
            raise ValueError(f"Workflow file too large: {file_size} bytes (max: {self.max_file_size})")
        
        # Read and parse JSON
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                workflow = self._safe_json_load(f.read())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in workflow '{name}': {str(e)}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid encoding in workflow '{name}': {str(e)}")
        except MemoryError:
            raise ValueError(f"Workflow '{name}' too large to load into memory")
        
        # Validate workflow structure
        if validate_comprehensive:
            self._comprehensive_validate_workflow(workflow, name)
        
        # Cache the result if caching is enabled
        if self.cache_enabled:
            current_time = time.time()
            self.template_cache[name] = (workflow, current_time)
            self.template_timestamps[name] = current_time
        
        return workflow
    
    def _is_valid_template_name(self, name: str) -> bool:
        """Validate that the template name is safe to use."""
        return bool(self.allowed_name_pattern.match(name))
    
    def _safe_json_load(self, content: str) -> Dict[str, Any]:
        """Safely load JSON with depth and size limits."""
        # Check for obvious depth issues (naive check)
        open_braces = content.count('{')
        close_braces = content.count('}')
        if abs(open_braces - close_braces) > 10:  # Allow small mismatch in comments
            raise ValueError("Potential deeply nested structure detected")
        
        # Load JSON
        data = json.loads(content)
        
        # Recursive size/depth check
        if self._get_json_depth(data) > self.max_json_depth:
            raise ValueError(f"JSON too deeply nested (max: {self.max_json_depth})")
        
        return data
    
    def _get_json_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Recursively get the maximum depth of a JSON structure."""
        if current_depth > self.max_json_depth:
            return current_depth
        
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._get_json_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._get_json_depth(v, current_depth + 1) for v in obj)
        else:
            return current_depth
    
    def _comprehensive_validate_workflow(self, workflow: Dict[str, Any], name: str) -> None:
        """Perform comprehensive validation of the workflow."""
        if not isinstance(workflow, dict):
            raise ValueError(f"Workflow '{name}' must be a dictionary")
        
        if len(workflow) > self.max_node_count:
            raise ValueError(f"Workflow '{name}' has too many nodes: {len(workflow)} (max: {self.max_node_count})")
        
        if not workflow:
            raise ValueError(f"Workflow '{name}' is empty")
        
        # Validate each node
        node_ids = set(workflow.keys())
        all_connections = set()
        
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                raise ValueError(f"Node '{node_id}' in workflow '{name}' is not a dictionary")
            
            # Validate required fields
            if "class_type" not in node_data:
                raise ValueError(f"Node '{node_id}' missing required 'class_type' field")
            
            class_type = node_data["class_type"]
            
            # Security: Check if node type is allowed
            if class_type not in self.allowed_node_types:
                raise ValueError(f"Node '{node_id}' uses disallowed class type: {class_type}")
            
            # Validate inputs structure
            if "inputs" not in node_data:
                node_data["inputs"] = {}
            elif not isinstance(node_data["inputs"], dict):
                raise ValueError(f"Node '{node_id}' inputs must be a dictionary")
            
            # Collect all connections referenced in inputs
            for input_name, input_value in node_data["inputs"].items():
                if self._is_connection_reference(input_value):
                    ref_node_id = str(input_value[0])
                    all_connections.add(ref_node_id)
        
        # Warn about unused nodes (not connected but present)
        unconnected_nodes = node_ids - all_connections
        if len(unconnected_nodes) > len(workflow) * 0.5:  # If >50% of nodes are unconnected
            print(f"Warning: Workflow '{name}' has many unconnected nodes: {unconnected_nodes}")
    
    def _is_connection_reference(self, value: Any) -> bool:
        """Check if a value is a ComfyUI connection reference [node_id, output_index]."""
        return (isinstance(value, list) and 
                len(value) == 2 and 
                isinstance(value[0], (str, int)) and 
                isinstance(value[1], int))
    
    def _get_available_templates(self) -> list:
        """Get list of available template files."""
        templates = []
        for file_path in self.workflows_dir.glob("*.json"):
            templates.append(file_path.stem)
        return sorted(templates)
    
    def get_template_info(self, name: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a template without fully loading it.
        """
        workflow_path = self.workflows_dir / f"{name}.json"
        
        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow template '{name}' not found")
        
        # Get file stats
        stat = workflow_path.stat()
        
        # Read enough to get basic info
        with open(workflow_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get size and basic JSON info without parsing deeply
        size = len(content.encode('utf-8'))
        
        # Count nodes by finding opening braces after "class_type"
        import re
        node_count = len(re.findall(r'"\d+"\s*:\s*{[^}]*"class_type"', content))
        
        return {
            "name": name,
            "path": str(workflow_path),
            "size_bytes": size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "node_count": node_count,
            "checksum": hashlib.md5(content.encode()).hexdigest()
        }
    
    def clear_cache(self) -> None:
        """Clear the template cache."""
        self.template_cache.clear()
        self.template_timestamps.clear()

# Example usage
if __name__ == "__main__":
    # Initialize the advanced workflow manager
    awm = AdvancedWorkflowManager(cache_enabled=True)
    
    # Example: Create a sample workflow file
    sample_workflow = {
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
    
    # Save the sample workflow
    sample_path = awm.workflows_dir / "advanced_sample.json"
    with open(sample_path, 'w') as f:
        json.dump(sample_workflow, f, indent=2)
    
    try:
        # Load the template
        loaded_workflow = awm.load_template("advanced_sample")
        print(f"Successfully loaded workflow with {len(loaded_workflow)} nodes")
        
        # Get template info
        info = awm.get_template_info("advanced_sample")
        print(f"Template info: {info}")
        
        # Load again (should use cache)
        loaded_workflow2 = awm.load_template("advanced_sample")
        print("Successfully loaded from cache")
        
    except Exception as e:
        print(f"Error: {e}")
```