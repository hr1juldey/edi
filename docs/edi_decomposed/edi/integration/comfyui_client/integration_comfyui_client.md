# Integration: ComfyUI Client

[Back to Integration Layer](./integration_layer.md)

## Purpose
ComfyUI API wrapper - Contains the ComfyUIClient class that handles communication with ComfyUI including submitting edits, polling status, and downloading results.

## Class: ComfyUIClient

### Methods
- `submit_edit()`: Submits an editing job to ComfyUI
- `poll_status()`: Checks the status of a submitted job
- `download_result()`: Downloads the result of a completed job

### Details
- Loads workflow templates from workflows/ directory
- Handles timeouts and retries gracefully
- Manages communication with ComfyUI API

## Functions

- [submit_edit()](./integration/submit_edit.md)
- [poll_status()](./integration/poll_status.md)
- [download_result()](./integration/download_result.md)

## Technology Stack

- Requests for HTTP communication
- JSON for API communication
- Pillow for image processing

## See Docs

### Requests Implementation Example
ComfyUI API client implementation for EDI:

```python
import requests
import json
import time
import os
from typing import Dict, Any, Optional
from pathlib import Path
from PIL import Image
import io

class ComfyUIClient:
    """ComfyUI API wrapper for EDI."""
    
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
        Injects user parameters into the workflow structure.
        """
        # Find nodes for positive and negative prompts and update them
        for node_id, node_data in workflow.items():
            if node_data.get("class_type") == "CLIPTextEncode":
                # Identify positive prompt node (usually has 'positive' in title or is node '6')
                if "positive" in node_data.get("_meta", {}).get("title", "").lower() or node_id == "6":
                    workflow[node_id]["inputs"]["text"] = positive_prompt
                # Identify negative prompt node (usually has 'negative' in title or is node '7')
                elif "negative" in node_data.get("_meta", {}).get("title", "").lower() or node_id == "7":
                    workflow[node_id]["inputs"]["text"] = negative_prompt
        
        # Handle image loading - we need to upload the image first
        image_upload_response = self.upload_image(image_path)
        image_filename = image_upload_response["name"]
        
        # Create or update image loading node
        image_node = {
            "inputs": {"image": image_filename},
            "class_type": "LoadImage",
            "_meta": {"title": "Input Image"}
        }
        
        # Find a suitable node ID that doesn't conflict
        new_node_id = "input_image"
        counter = 0
        while new_node_id in workflow or str(counter) in workflow:
            new_node_id = f"input_image_{counter}"
            counter += 1
        
        workflow[new_node_id] = image_node
        
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
    
    def poll_status(self, job_id: str) -> Dict[str, Any]:
        """
        Checks the status of a submitted job.
        Returns status and additional information.
        """
        try:
            response = self.session.get(
                f"{self.base_url}/history/{job_id}"
            )
            response.raise_for_status()
            
            history = response.json()
            
            if str(job_id) in history:
                job_info = history[str(job_id)]
                
                # Determine status based on job info
                if "outputs" in job_info and len(job_info["outputs"]) > 0:
                    return {
                        "status": "completed",
                        "job_info": job_info,
                        "outputs": job_info["outputs"]
                    }
                else:
                    return {
                        "status": "processing",
                        "job_info": job_info
                    }
            else:
                return {
                    "status": "not_found",
                    "message": "Job ID not found in history"
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Request error: {str(e)}"
            }
    
    def wait_for_completion(self, job_id: str, timeout: int = 600) -> Dict[str, Any]:
        """
        Waits for a job to complete with a timeout.
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.poll_status(job_id)
            
            if status["status"] == "completed":
                return status
            elif status["status"] == "failed" or status["status"] == "error":
                return status
            elif status["status"] == "not_found":
                # Job might not be registered yet, wait a bit more
                time.sleep(1)
                continue
            else:
                # Processing, wait and try again
                time.sleep(2)
        
        return {
            "status": "timeout",
            "message": f"Job {job_id} did not complete within {timeout} seconds"
        }
    
    def download_result(self, job_id: str, output_path: str) -> bool:
        """
        Downloads the result of a completed job to the specified path.
        """
        # First, wait for the job to complete
        completion_status = self.wait_for_completion(job_id)
        
        if completion_status["status"] != "completed":
            print(f"Job {job_id} did not complete successfully: {completion_status['message']}")
            return False
        
        # Extract image information from the completed job
        outputs = completion_status["outputs"]
        
        # Find the first image output
        image_output = None
        for node_id, node_data in outputs.items():
            if 'images' in node_data:
                image_output = node_data['images'][0]
                break
        
        if not image_output:
            print(f"No image output found for job {job_id}")
            return False
        
        # Download the image
        image_filename = image_output['filename']
        image_subfolder = image_output.get('subfolder', '')
        image_type = image_output.get('type', 'output')
        
        # Construct the download URL
        params = {
            'filename': image_filename,
            'type': image_type
        }
        if image_subfolder:
            params['subfolder'] = image_subfolder
        
        response = self.session.get(
            f"{self.base_url}/view",
            params=params
        )
        response.raise_for_status()
        
        # Save the image to the specified output path
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"Downloaded result to: {output_path}")
        return True

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
        
        # Wait for completion
        print(f"Waiting for job {job_id} to complete...")
        status = client.wait_for_completion(job_id)
        
        if status["status"] == "completed":
            print("Job completed successfully!")
            
            # Download the result
            output_path = "edited_output.jpg"
            success = client.download_result(job_id, output_path)
            
            if success:
                print(f"Result downloaded to {output_path}")
            else:
                print("Failed to download result")
        else:
            print(f"Job failed with status: {status['status']}")
            print(f"Message: {status.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"Error during edit operation: {str(e)}")
```

### JSON Implementation Example
JSON workflow handling for the ComfyUI client:

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
        Saves a workflow template to a JSON file.
        """
        workflow_path = self.workflows_dir / f"{name}.json"
        
        with open(workflow_path, 'w') as f:
            json.dump(workflow, f, indent=2)
    
    def load_workflow(self, name: str) -> Dict[str, Any]:
        """
        Loads a workflow template from a JSON file.
        """
        workflow_path = self.workflows_dir / f"{name}.json"
        
        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow {name} not found at {workflow_path}")
        
        with open(workflow_path, 'r') as f:
            return json.load(f)
    
    def list_workflows(self) -> List[str]:
        """
        Lists all available workflow templates.
        """
        workflow_files = list(self.workflows_dir.glob("*.json"))
        workflow_names = [f.stem for f in workflow_files]
        return workflow_names
    
    def create_img2img_workflow(self) -> Dict[str, Any]:
        """
        Creates a standard image-to-image workflow template.
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
            "10": {
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
        Creates a masked inpainting workflow template.
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
        Creates a ControlNet-based workflow template for structure-preserving edits.
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

### Pillow Implementation Example
Image processing utilities for the ComfyUI client:

```python
from PIL import Image, ImageOps
import io
from pathlib import Path
from typing import Tuple, Dict, Any
import os

class ImageProcessor:
    """
    Image processing utilities for the ComfyUI client.
    """
    
    @staticmethod
    def validate_image(image_path: str) -> bool:
        """
        Validates that the provided file is a valid image.
        """
        try:
            with Image.open(image_path) as img:
                img.verify()  # Verify that it's a valid image
            return True
        except Exception:
            return False
    
    @staticmethod
    def resize_image(image_path: str, max_size: int = 1024) -> str:
        """
        Resizes an image to fit within max_size x max_size while maintaining aspect ratio.
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
            
            # Preserve the original format
            if img.mode in ('RGBA', 'LA', 'P') and image_path.lower().endswith(('.jpg', '.jpeg')):
                # Convert to RGB if saving as JPEG (which doesn't support transparency)
                img = img.convert('RGB')
            
            img.save(output_path, optimize=True, quality=85)
            
        return output_path
    
    @staticmethod
    def get_image_dimensions(image_path: str) -> Tuple[int, int]:
        """
        Gets the dimensions of an image.
        """
        with Image.open(image_path) as img:
            return img.size
    
    @staticmethod
    def crop_center(image_path: str, crop_width: int, crop_height: int) -> str:
        """
        Crops the center of an image to the specified dimensions.
        """
        with Image.open(image_path) as img:
            width, height = img.size
            
            # Calculate the crop box
            left = (width - crop_width) // 2
            top = (height - crop_height) // 2
            right = left + crop_width
            bottom = top + crop_height
            
            # Ensure crop box is within image bounds
            left = max(0, left)
            top = max(0, top)
            right = min(width, right)
            bottom = min(height, bottom)
            
            # Crop the image
            cropped_img = img.crop((left, top, right, bottom))
            
            # Save the cropped image with a new name
            output_path = f"{image_path.rsplit('.', 1)[0]}_cropped.{image_path.rsplit('.', 1)[1]}"
            cropped_img.save(output_path)
            
        return output_path
    
    @staticmethod
    def enhance_image(image_path: str, brightness: float = 1.0, contrast: float = 1.0, saturation: float = 1.0) -> str:
        """
        Enhances an image with the specified parameters.
        """
        from PIL import ImageEnhance
        
        with Image.open(image_path) as img:
            # Enhance brightness
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(brightness)
            
            # Enhance contrast
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(contrast)
            
            # Enhance saturation
            if saturation != 1.0:
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(saturation)
            
            # Save the enhanced image
            output_path = f"{image_path.rsplit('.', 1)[0]}_enhanced.{image_path.rsplit('.', 1)[1]}"
            img.save(output_path)
            
        return output_path
    
    @staticmethod
    def convert_format(image_path: str, new_format: str) -> str:
        """
        Converts an image to a new format (e.g., 'png', 'jpg', 'webp').
        """
        with Image.open(image_path) as img:
            # Convert to RGB if converting to JPEG (which doesn't support transparency)
            if new_format.lower() in ['jpg', 'jpeg'] and img.mode in ('RGBA', 'LA', 'P'):
                if img.mode == 'P':
                    img = img.convert('RGBA')
                img = img.convert('RGB')
            
            # Create new path with new extension
            output_path = f"{image_path.rsplit('.', 1)[0]}.{new_format.lower()}"
            img.save(output_path, optimize=True, quality=85 if new_format.lower() in ['jpg', 'jpeg'] else None)
            
        return output_path

class ComfyUIImageHandler:
    """
    Handles image operations specific to ComfyUI workflows.
    """
    
    def __init__(self, max_size: int = 1024):
        self.max_size = max_size
        self.processor = ImageProcessor()
    
    def prepare_image_for_comfyui(self, image_path: str) -> str:
        """
        Prepares an image for use with ComfyUI by validating, resizing if needed,
        and ensuring compatibility.
        """
        # Validate the image
        if not self.processor.validate_image(image_path):
            raise ValueError(f"Invalid image file: {image_path}")
        
        # Get image dimensions
        width, height = self.processor.get_image_dimensions(image_path)
        
        # Resize if too large
        if width > self.max_size or height > self.max_size:
            print(f"Resizing image from {width}x{height} to fit within {self.max_size}x{self.max_size}")
            image_path = self.processor.resize_image(image_path, self.max_size)
        
        return image_path
    
    def create_mask_from_image(self, image_path: str, threshold: int = 128) -> str:
        """
        Creates a simple mask from an image based on a threshold.
        """
        with Image.open(image_path) as img:
            # Convert to grayscale if needed
            if img.mode != 'L':
                img = img.convert('L')
            
            # Apply threshold to create binary mask
            mask = img.point(lambda p: p > threshold and 255)
            
            # Save the mask
            output_path = f"{image_path.rsplit('.', 1)[0]}_mask.png"
            mask.save(output_path)
            
        return output_path

# Example usage
if __name__ == "__main__":
    # Example of using the image processor
    processor = ImageProcessor()
    
    # Validate image (assuming a test image exists)
    print("Image validation example would go here")
    
    # Example of using the ComfyUI image handler
    comfy_handler = ComfyUIImageHandler(max_size=768)
    
    # The handler could be used to prepare images before sending to ComfyUI
    print(f"ComfyUI image handler initialized with max size: {comfy_handler.max_size}")
    
    print("Image processing utilities example completed!")
```