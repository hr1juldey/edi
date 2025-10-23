# Integration Layer

[Back to Index](../index.md)

## Purpose

ComfyUI API client, image I/O using requests and Pillow

## Component Design

### 6. Integration Layer

#### 6.1 ComfyUI Client

**API Wrapper**:

```python
class ComfyUIClient:
    def __init__(self, base_url="http://localhost:8188"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def submit_edit(
        self,
        image_path: str,
        positive_prompt: str,
        negative_prompt: str,
        workflow_template: str = "img2img_default"
    ) -> str:
        """
        Submit edit job to ComfyUI and return job ID.
        """
        workflow = self.load_workflow(workflow_template)
        
        # Inject parameters
        workflow['nodes']['positive_prompt']['text'] = positive_prompt
        workflow['nodes']['negative_prompt']['text'] = negative_prompt
        workflow['nodes']['input_image']['path'] = image_path
        
        response = self.session.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow}
        )
        
        return response.json()['prompt_id']
    
    def poll_status(self, job_id: str) -> dict:
        """
        Check job status (returns 'queued', 'processing', 'completed', 'failed').
        """
        response = self.session.get(
            f"{self.base_url}/history/{job_id}"
        )
        return response.json()
    
    def download_result(self, job_id: str, output_path: str):
        """
        Download completed edit to local file.
        """
        status = self.poll_status(job_id)
        if status['status'] != 'completed':
            raise RuntimeError(f"Job not completed: {status['status']}")
        
        image_url = status['outputs'][0]['images'][0]['filename']
        
        response = self.session.get(
            f"{self.base_url}/view?filename={image_url}",
            stream=True
        )
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)
```

**Workflow Templates** (stored in `workflows/` directory):

- `img2img_default.json`: Standard image-to-image with prompts
- `inpaint_masked.json`: Masked inpainting for region-specific edits
- `controlnet_canny.json`: Structure-preserving edits using ControlNet

## Sub-modules

This component includes the following modules:

- [integration/comfyui_client.py](./comfyui_client/comfyui_client.md)
- [integration/workflow_manager.py](./workflow_manager/workflow_manager.md)

## Technology Stack

- Requests for HTTP communication
- Pillow for image processing
- JSON for workflow templates

## See Docs

### Requests and Pillow Implementation Example

ComfyUI API client with image processing for the EDI application:

```python
import requests
import json
import time
from typing import Dict, Any, Optional
from PIL import Image
import io
from pathlib import Path

class ComfyUIClient:
    def __init__(self, base_url: str = "http://localhost:8188"):
        self.base_url = base_url
        self.session = requests.Session()
        # Set a reasonable timeout for API requests
        self.session.timeout = 30
    
    def submit_edit(
        self,
        image_path: str,
        positive_prompt: str,
        negative_prompt: str,
        workflow_template: str = "img2img_default"
    ) -> str:
        """
        Submit edit job to ComfyUI and return job ID.
        """
        # Load workflow template
        workflow = self.load_workflow(workflow_template)
        
        # Inject parameters into the workflow
        workflow = self.inject_parameters(workflow, positive_prompt, negative_prompt, image_path)
        
        # Submit the prompt to ComfyUI
        response = self.session.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow}
        )
        response.raise_for_status()
        
        result = response.json()
        return result['prompt_id']
    
    def load_workflow(self, template_name: str) -> Dict[str, Any]:
        """
        Load a workflow template from the workflows directory.
        """
        workflow_dir = Path("workflows")
        workflow_path = workflow_dir / f"{template_name}.json"
        
        if not workflow_path.exists():
            # Return a default workflow if template doesn't exist
            return self.get_default_workflow()
        
        with open(workflow_path, 'r') as f:
            return json.load(f)
    
    def get_default_workflow(self) -> Dict[str, Any]:
        """
        Return a default workflow structure for image-to-image editing.
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
    
    def inject_parameters(
        self, 
        workflow: Dict[str, Any], 
        positive_prompt: str, 
        negative_prompt: str, 
        image_path: str
    ) -> Dict[str, Any]:
        """
        Inject parameters into the workflow structure.
        """
        # Find nodes for positive and negative prompts
        for node_id, node_data in workflow.items():
            if node_data.get("class_type") == "CLIPTextEncode":
                # Check if this is the positive prompt node
                if "positive" in node_data.get("inputs", {}).get("text", "").lower() or \
                   node_id == "6":  # Default positive prompt node in our template
                    workflow[node_id]["inputs"]["text"] = positive_prompt
                elif "negative" in node_data.get("inputs", {}).get("text", "").lower() or \
                     node_id == "7":  # Default negative prompt node in our template
                    workflow[node_id]["inputs"]["text"] = negative_prompt
        
        # For image input, we need to upload the image first
        image_upload_response = self.upload_image(image_path)
        image_filename = image_upload_response["name"]
        
        # Add or update image input node
        image_node = {
            "inputs": {
                "image": image_filename
            },
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"}
        }
        
        # Find an appropriate node for the image (e.g., one that feeds into VAE encode or similar)
        # Add this as a node in the workflow
        workflow["image_input"] = image_node
        
        return workflow
    
    def upload_image(self, image_path: str) -> Dict[str, Any]:
        """
        Upload an image to ComfyUI to be used in workflows.
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
    
    def poll_status(self, job_id: str) -> Dict[str, Any]:
        """
        Check job status (returns 'queued', 'processing', 'completed', 'failed').
        """
        response = self.session.get(
            f"{self.base_url}/history/{job_id}"
        )
        response.raise_for_status()
        
        history = response.json()
        
        if job_id in history:
            job_info = history[job_id]
            # Determine status based on the job info
            if "outputs" in job_info and len(job_info["outputs"]) > 0:
                return {
                    "status": "completed",
                    "outputs": job_info["outputs"]
                }
            else:
                return {
                    "status": "processing",
                    "info": job_info
                }
        else:
            return {
                "status": "not_found",
                "info": "Job ID not found in history"
            }
    
    def wait_for_completion(self, job_id: str, timeout: int = 300) -> Dict[str, Any]:
        """
        Poll for job completion with timeout.
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.poll_status(job_id)
            
            if status["status"] == "completed":
                return status
            elif status["status"] == "not_found":
                raise RuntimeError(f"Job {job_id} not found")
            elif status["status"] == "failed":
                raise RuntimeError(f"Job {job_id} failed")
            
            time.sleep(5)  # Wait 5 seconds before polling again
        
        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")
    
    def download_result(self, job_id: str, output_path: str):
        """
        Download completed edit to local file.
        """
        status = self.wait_for_completion(job_id)
        
        if status['status'] != 'completed':
            raise RuntimeError(f"Job not completed: {status['status']}")
        
        # Extract the image filename from the outputs
        output_info = status['outputs']
        image_info = None
        
        # Find image output in the job result
        for node_id, node_outputs in output_info.items():
            if 'images' in node_outputs:
                image_info = node_outputs['images'][0]
                break
        
        if not image_info:
            raise RuntimeError("No image found in job outputs")
        
        image_filename = image_info['filename']
        image_subfolder = image_info.get('subfolder', '')
        image_type = image_info.get('type', 'output')
        
        # Construct the download URL
        params = {
            'filename': image_filename,
            'type': image_type
        }
        if image_subfolder:
            params['subfolder'] = image_subfolder
        
        response = self.session.get(
            f"{self.base_url}/view",
            params=params,
            stream=True
        )
        response.raise_for_status()
        
        # Save the image to the specified output path
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

class ImageProcessor:
    """
    Image processing utilities for the integration layer.
    """
    
    @staticmethod
    def validate_image(image_path: str) -> bool:
        """
        Validate that the provided file is a valid image.
        """
        try:
            with Image.open(image_path) as img:
                # Check if image can be opened and has valid format
                img.verify()
            return True
        except Exception:
            return False
    
    @staticmethod
    def resize_image(image_path: str, max_size: int = 1024) -> str:
        """
        Resize an image to fit within max_size x max_size while maintaining aspect ratio.
        Returns path to resized image.
        """
        with Image.open(image_path) as img:
            # Calculate new dimensions maintaining aspect ratio
            width, height = img.size
            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                
                # Resize the image
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save the resized image with a new name
            output_path = f"{image_path.rsplit('.', 1)[0]}_resized.{image_path.rsplit('.', 1)[1]}"
            img.save(output_path, optimize=True, quality=85)
            
        return output_path
    
    @staticmethod
    def compare_images(original_path: str, edited_path: str) -> Dict[str, float]:
        """
        Compare two images and return metrics about the changes.
        This is a simplified version - in practice, this could use more sophisticated algorithms.
        """
        with Image.open(original_path) as orig_img, Image.open(edited_path) as edit_img:
            # Convert to same mode if different
            if orig_img.mode != edit_img.mode:
                edit_img = edit_img.convert(orig_img.mode)
            
            # Convert to same size if different
            if orig_img.size != edit_img.size:
                edit_img = edit_img.resize(orig_img.size)
            
            # For now, just return basic metadata
            return {
                "original_size": orig_img.size[0] * orig_img.size[1],
                "edited_size": edit_img.size[0] * edit_img.size[1],
                "original_mode": len(orig_img.getbands()),
                "edited_mode": len(edit_img.getbands())
            }

# Example usage
if __name__ == "__main__":
    # Initialize the client
    client = ComfyUIClient()
    
    # Example image processing
    processor = ImageProcessor()
    
    # Validate and resize input image if needed
    input_image = "input.jpg"
    if not processor.validate_image(input_image):
        print("Invalid image file")
    else:
        # Resize if too large
        resized_image = processor.resize_image(input_image)
        
        # Submit edit job
        job_id = client.submit_edit(
            image_path=resized_image,
            positive_prompt="dramatic sky with storm clouds",
            negative_prompt="sunny sky, clear bright",
            workflow_template="img2img_default"
        )
        
        print(f"Submitted job with ID: {job_id}")
        
        # Wait for completion and download result
        output_path = "edited_output.jpg"
        client.download_result(job_id, output_path)
        
        print(f"Edit completed, result saved to: {output_path}")
        
        # Compare original and edited images
        comparison = processor.compare_images(input_image, output_path)
        print(f"Image comparison: {comparison}")
```

### JSON Implementation Example

Workflow templates management:

```python
import json
from pathlib import Path
from typing import Dict, Any, List
import os

class WorkflowManager:
    """
    Manages ComfyUI workflow templates using JSON files.
    """
    
    def __init__(self, workflows_dir: str = "workflows"):
        self.workflows_dir = Path(workflows_dir)
        self.workflows_dir.mkdir(exist_ok=True)
    
    def save_workflow(self, name: str, workflow: Dict[str, Any]):
        """
        Save a workflow template to a JSON file.
        """
        workflow_path = self.workflows_dir / f"{name}.json"
        
        with open(workflow_path, 'w') as f:
            json.dump(workflow, f, indent=2)
    
    def load_workflow(self, name: str) -> Dict[str, Any]:
        """
        Load a workflow template from a JSON file.
        """
        workflow_path = self.workflows_dir / f"{name}.json"
        
        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow {name} not found at {workflow_path}")
        
        with open(workflow_path, 'r') as f:
            return json.load(f)
    
    def list_workflows(self) -> List[str]:
        """
        List all available workflow templates.
        """
        workflow_files = list(self.workflows_dir.glob("*.json"))
        workflow_names = [f.stem for f in workflow_files]
        return workflow_names
    
    def create_img2img_workflow(self) -> Dict[str, Any]:
        """
        Create a standard image-to-image workflow template.
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
                "_meta": {"title": "Load Image"}
            },
            "9": {
                "inputs": {
                    "pixels": ["8", 0],
                    "vae": ["4", 2]
                },
                "class_type": "VAEEncode",
                "_meta": {"title": "VAE Encode"}
            },
            "5": {
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                },
                "class_type": "VAEDecode",
                "_meta": {"title": "VAE Decode"}
            }
        }
    
    def create_inpaint_workflow(self) -> Dict[str, Any]:
        """
        Create a masked inpainting workflow template.
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
    
    def create_controlnet_workflow(self) -> Dict[str, Any]:
        """
        Create a ControlNet-based workflow template for structure-preserving edits.
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
                "inputs": {
                    "conditioning": ["12", 0],
                    "strength": 1.0,
                    "start_percent": 0.0,
                    "end_percent": 1.0
                },
                "class_type": "ControlNetApplyAdvanced",
                "_meta": {"title": "Apply ControlNet"}
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
                    "image": ["8", 0],
                    "strength": 1.0,
                    "start_percent": 0.0,
                    "end_percent": 1.0
                },
                "class_type": "ControlNetApplyAdvanced",
                "_meta": {"title": "Apply ControlNet"}
            }
        }

# Example usage
if __name__ == "__main__":
    # Initialize workflow manager
    wm = WorkflowManager()
    
    # Create and save workflows
    img2img_workflow = wm.create_img2img_workflow()
    wm.save_workflow("img2img_default", img2img_workflow)
    
    inpaint_workflow = wm.create_inpaint_workflow()
    wm.save_workflow("inpaint_masked", inpaint_workflow)
    
    controlnet_workflow = wm.create_controlnet_workflow()
    wm.save_workflow("controlnet_canny", controlnet_workflow)
    
    # List available workflows
    print("Available workflows:", wm.list_workflows())
    
    # Load a workflow
    loaded_workflow = wm.load_workflow("img2img_default")
    print(f"Loaded workflow keys: {list(loaded_workflow.keys())}")
    
    # Save the workflow to a JSON string for direct use
    workflow_json = json.dumps(loaded_workflow, indent=2)
    print(f"Workflow JSON length: {len(workflow_json)} characters")
```
