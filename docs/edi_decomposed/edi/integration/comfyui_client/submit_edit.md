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

## See Docs

### Python Implementation Example
Implementation of the submit_edit method for ComfyUIClient:

```python
import requests
import json
import os
from typing import Dict, Any
from pathlib import Path

class ComfyUIClient:
    def __init__(self, base_url: str = "http://localhost:8188", timeout: int = 300):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout
        
        # Store workflow templates
        self.workflow_dir = Path("workflows")
        self.workflow_dir.mkdir(exist_ok=True)
    
    def submit_edit(
        self,
        image_path: str,
        positive_prompt: str,
        negative_prompt: str,
        workflow_template: str = "img2img_default"
    ) -> str:
        """
        Submits an editing job to ComfyUI.
        Returns the job ID for tracking the status.
        """
        # Validate input parameters
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not positive_prompt.strip():
            raise ValueError("Positive prompt cannot be empty")
        
        # Load workflow template
        workflow = self.load_workflow(workflow_template)
        
        # Validate workflow structure
        if not isinstance(workflow, dict):
            raise ValueError(f"Invalid workflow structure in {workflow_template}.json")
        
        # Upload image to ComfyUI (required for processing)
        image_upload_response = self.upload_image(image_path)
        image_filename = image_upload_response["name"]
        
        # Inject parameters into the workflow
        workflow = self.inject_parameters(
            workflow, 
            positive_prompt, 
            negative_prompt, 
            image_filename
        )
        
        # Submit the prompt to ComfyUI
        response = self.session.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow}
        )
        
        # Handle potential HTTP errors
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            error_details = response.json() if response.content else {}
            raise RuntimeError(f"Failed to submit job to ComfyUI: {e}, details: {error_details}")
        
        result = response.json()
        
        # Validate response structure
        if 'prompt_id' not in result:
            raise RuntimeError(f"Invalid response from ComfyUI: {result}")
        
        job_id = result['prompt_id']
        print(f"Submitted edit job with ID: {job_id}")
        return job_id
    
    def load_workflow(self, template_name: str) -> Dict[str, Any]:
        """
        Loads a workflow template from the workflows directory.
        """
        workflow_path = self.workflow_dir / f"{template_name}.json"
        
        if workflow_path.exists():
            with open(workflow_path, 'r') as f:
                return json.load(f)
        else:
            # Return a default workflow if template doesn't exist
            print(f"Warning: Workflow template '{template_name}' not found. Using default.")
            return self.get_default_workflow()
    
    def get_default_workflow(self) -> Dict[str, Any]:
        """
        Returns a default workflow structure for image-to-image editing.
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
            },
            "8": {
                "inputs": {
                    "image": "",
                    "upload": "image"
                },
                "class_type": "LoadImage",
                "_meta": {"title": "Input Image"}
            }
        }
    
    def inject_parameters(
        self,
        workflow: Dict[str, Any],
        positive_prompt: str,
        negative_prompt: str,
        image_filename: str
    ) -> Dict[str, Any]:
        """
        Injects user parameters into the workflow structure.
        """
        # Find and update prompt nodes
        for node_id, node_data in workflow.items():
            if node_data.get("class_type") == "CLIPTextEncode":
                node_title = node_data.get("_meta", {}).get("title", "").lower()
                
                # Identify positive prompt node by title or default IDs
                if "positive" in node_title or node_id == "6":
                    workflow[node_id]["inputs"]["text"] = positive_prompt
                # Identify negative prompt node by title or default IDs
                elif "negative" in node_title or node_id == "7":
                    workflow[node_id]["inputs"]["text"] = negative_prompt
            elif node_data.get("class_type") == "LoadImage":
                # Update image path in LoadImage nodes
                workflow[node_id]["inputs"]["image"] = image_filename
        
        # If no LoadImage node found, create one
        has_image_node = any(
            node.get("class_type") == "LoadImage" 
            for node in workflow.values()
        )
        
        if not has_image_node:
            # Create image loading node
            # Find a unique node ID
            new_node_id = "9"  # Common default after the basic nodes
            while new_node_id in workflow:
                new_node_id = str(int(new_node_id) + 1)
            
            workflow[new_node_id] = {
                "inputs": {"image": image_filename},
                "class_type": "LoadImage",
                "_meta": {"title": "Input Image"}
            }
        
        return workflow
    
    def upload_image(self, image_path: str) -> Dict[str, Any]:
        """
        Uploads an image to ComfyUI.
        """
        with open(image_path, 'rb') as image_file:
            files = {
                'image': (Path(image_path).name, image_file, 'image/jpeg')
            }
            response = self.session.post(
                f"{self.base_url}/upload/image",
                files=files
            )
            response.raise_for_status()
            return response.json()

# Example usage
if __name__ == "__main__":
    # Initialize the client
    client = ComfyUIClient()
    
    # Example: Submit an edit job
    try:
        job_id = client.submit_edit(
            image_path="input.jpg",
            positive_prompt="dramatic sky with storm clouds",
            negative_prompt="sunny sky, clear weather, no clouds",
            workflow_template="img2img_default"
        )
        
        print(f"Successfully submitted job with ID: {job_id}")
        
    except FileNotFoundError as e:
        print(f"File error: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to ComfyUI. Please ensure ComfyUI is running.")
    except Exception as e:
        print(f"Error during edit operation: {str(e)}")
```

### JSON Workflow Template Example
Sample workflow template for the submit_edit method:

```json
{
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
    "_meta": {
      "title": "KSampler"
    }
  },
  "4": {
    "inputs": {
      "ckpt_name": "model.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint"
    }
  },
  "5": {
    "inputs": {
      "width": 512,
      "height": 512,
      "batch_size": 1
    },
    "class_type": "EmptyLatentImage",
    "_meta": {
      "title": "Empty Latent Image"
    }
  },
  "6": {
    "inputs": {
      "text": "PROMPT_WILL_BE_INJECTED_HERE",
      "clip": ["4", 1]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "Positive Prompt"
    }
  },
  "7": {
    "inputs": {
      "text": "PROMPT_WILL_BE_INJECTED_HERE", 
      "clip": ["4", 1]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "Negative Prompt"
    }
  },
  "8": {
    "inputs": {
      "image": "IMAGE_WILL_BE_INJECTED_HERE",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Input Image"
    }
  }
}
```

### Error Handling Implementation
Proper error handling for the submit_edit method:

```python
import requests
from typing import Dict, Any
import time
import socket

class ComfyUIError(Exception):
    """Custom exception for ComfyUI-related errors."""
    pass

class ComfyUIClientWithErrors:
    def __init__(self, base_url: str = "http://localhost:8188", timeout: int = 300):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout
        
        # Store workflow templates
        self.workflow_dir = Path("workflows")
        self.workflow_dir.mkdir(exist_ok=True)
    
    def check_connection(self) -> bool:
        """Check if ComfyUI is accessible."""
        try:
            response = requests.get(f"{self.base_url}/queue", timeout=5)
            return response.status_code in [200, 404]  # 404 might be returned if endpoint doesn't exist but server is up
        except requests.exceptions.ConnectionError:
            return False
        except requests.exceptions.Timeout:
            return False
        except socket.gaierror:  # DNS resolution error
            return False
    
    def submit_edit(
        self,
        image_path: str,
        positive_prompt: str,
        negative_prompt: str,
        workflow_template: str = "img2img_default",
        max_retries: int = 3
    ) -> str:
        """
        Submits an editing job to ComfyUI with retry logic and comprehensive error handling.
        Returns the job ID for tracking the status.
        """
        # Validate inputs first
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not positive_prompt.strip():
            raise ValueError("Positive prompt cannot be empty")
        
        # Check if ComfyUI is accessible
        if not self.check_connection():
            raise ComfyUIError("Cannot connect to ComfyUI. Please ensure ComfyUI is running at the specified address.")
        
        # Load workflow template
        workflow = self.load_workflow(workflow_template)
        
        # Upload image to ComfyUI
        image_upload_response = self.upload_image_with_retry(image_path, max_retries)
        image_filename = image_upload_response["name"]
        
        # Inject parameters into the workflow
        workflow = self.inject_parameters(
            workflow, 
            positive_prompt, 
            negative_prompt, 
            image_filename
        )
        
        # Submit with retry logic
        attempt = 0
        while attempt < max_retries:
            try:
                response = self.session.post(
                    f"{self.base_url}/prompt",
                    json={"prompt": workflow},
                    timeout=self.timeout
                )
                
                # Handle potential HTTP errors
                if response.status_code == 429:  # Too many requests
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Rate limited by ComfyUI, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    attempt += 1
                    continue
                
                response.raise_for_status()
                
                result = response.json()
                
                # Validate response structure
                if 'prompt_id' not in result:
                    raise ComfyUIError(f"Invalid response from ComfyUI: {result}")
                
                job_id = result['prompt_id']
                print(f"Submitted edit job with ID: {job_id}")
                return job_id
                
            except requests.exceptions.Timeout:
                attempt += 1
                if attempt >= max_retries:
                    raise ComfyUIError(f"Timeout submitting job after {max_retries} attempts")
                print(f"Timeout on attempt {attempt}, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except requests.exceptions.ConnectionError as e:
                attempt += 1
                if attempt >= max_retries:
                    raise ComfyUIError(f"Connection error submitting job after {max_retries} attempts: {str(e)}")
                print(f"Connection error on attempt {attempt}, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except requests.exceptions.HTTPError as e:
                if response.status_code >= 500:  # Server error
                    attempt += 1
                    if attempt >= max_retries:
                        raise ComfyUIError(f"Server error submitting job after {max_retries} attempts: {str(e)}")
                    print(f"Server error on attempt {attempt}, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    # Client error, don't retry
                    error_details = response.json() if response.content else {}
                    raise ComfyUIError(f"HTTP error submitting job to ComfyUI: {e}, details: {error_details}")
        
        raise ComfyUIError(f"Failed to submit job after {max_retries} attempts")
    
    def upload_image_with_retry(self, image_path: str, max_retries: int = 3) -> Dict[str, Any]:
        """Uploads an image to ComfyUI with retry logic."""
        attempt = 0
        while attempt < max_retries:
            try:
                with open(image_path, 'rb') as image_file:
                    files = {
                        'image': (Path(image_path).name, image_file, 'image/jpeg')
                    }
                    response = self.session.post(
                        f"{self.base_url}/upload/image",
                        files=files,
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    return response.json()
            except requests.exceptions.RequestException as e:
                attempt += 1
                if attempt >= max_retries:
                    raise ComfyUIError(f"Failed to upload image after {max_retries} attempts: {str(e)}")
                print(f"Image upload failed on attempt {attempt}, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def load_workflow(self, template_name: str) -> Dict[str, Any]:
        """Loads a workflow template from the workflows directory."""
        # Implementation would be the same as previous example
        pass
    
    def inject_parameters(
        self,
        workflow: Dict[str, Any],
        positive_prompt: str,
        negative_prompt: str,
        image_filename: str
    ) -> Dict[str, Any]:
        """Injects user parameters into the workflow structure."""
        # Implementation would be the same as previous example
        pass
```