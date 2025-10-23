# ComfyUIClient.download_result()

[Back to ComfyUI Client](../integration_comfyui_client.md)

## Related User Story
"As a user, I want to receive my edited image when the process is complete." (from PRD - implied by core functionality)

## Function Signature
`download_result(job_id: str, output_path: str)`

## Parameters
- `job_id: str` - The job ID of the completed edit job
- `output_path: str` - The local file path where the result should be saved

## Returns
- None - Downloads and saves the result to the specified path

## Step-by-step Logic
1. First check the job status to ensure it's completed
2. If the job is not completed, raise an error
3. Get the output image filename from the job history
4. Make a request to download the result image from ComfyUI
5. Stream the image data to the specified output path
6. Handle file writing with atomic operations to prevent corruption
7. Validate that the download was successful

## File Management
- Uses streaming to handle large image files efficiently
- Performs atomic file writes to prevent corruption
- Validates file integrity after download
- Handles file permissions appropriately

## Error Handling
- Checks that the job is actually completed before downloading
- Handles network errors during download
- Manages disk space constraints
- Validates image file format after download

## Input/Output Data Structures
### Download Result
- Input: Job ID to identify the completed job
- Output: Saved image file at the specified output path
- The result is a valid image file (format depends on workflow)

## See Docs

### Python Implementation Example
Implementation of the download_result method for ComfyUIClient:

```python
import requests
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import shutil

class ComfyUIClient:
    def __init__(self, base_url: str = "http://localhost:8188", timeout: int = 300):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout
    
    def poll_status(self, job_id: str) -> Dict[str, Any]:
        """
        Checks the status of a submitted job.
        This is a simplified version for the example.
        """
        try:
            response = self.session.get(
                f"{self.base_url}/history/{job_id}",
                timeout=self.timeout
            )
            response.raise_for_status()
            
            history = response.json()
            
            if str(job_id) in history:
                job_info = history[str(job_id)]
                
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
    
    def download_result(self, job_id: str, output_path: str) -> bool:
        """
        Downloads the result of a completed job to the specified path.
        """
        # First, check if the job is completed
        status_info = self.poll_status(job_id)
        
        if status_info["status"] != "completed":
            if status_info["status"] == "not_found":
                raise FileNotFoundError(f"Job {job_id} not found in history")
            elif status_info["status"] == "processing":
                raise RuntimeError(f"Job {job_id} is still processing, cannot download result yet")
            else:
                raise RuntimeError(f"Cannot download result for job {job_id}: {status_info.get('message', 'Unknown status')}")
        
        # Extract image information from the completed job
        outputs = status_info["outputs"]
        
        # Find the first image output
        image_output = None
        for node_id, node_data in outputs.items():
            if 'images' in node_data:
                image_output = node_data['images'][0]
                break
        
        if not image_output:
            raise RuntimeError(f"No image output found for job {job_id}")
        
        # Get image details
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
        
        # Download the image with streaming to handle large files efficiently
        try:
            response = self.session.get(
                f"{self.base_url}/view",
                params=params,
                stream=True  # Enable streaming for large files
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Use a temporary file for atomic write operation
            output_path_obj = Path(output_path)
            temp_path = output_path_obj.with_suffix(output_path_obj.suffix + '.tmp')
            
            # Ensure the directory exists
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # Stream download to temporary file
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
            
            # Verify the downloaded file is a valid image
            try:
                with Image.open(temp_path) as img:
                    img.verify()  # Verify it's a valid image file
            except Exception as e:
                temp_path.unlink()  # Remove invalid file
                raise ValueError(f"Downloaded file is not a valid image: {str(e)}")
            
            # Atomic move: rename temporary file to final path
            shutil.move(str(temp_path), str(output_path_obj))
            
            print(f"Successfully downloaded result to: {output_path}")
            return True
            
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Timeout downloading result for job {job_id}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Connection error downloading result for job {job_id}")
        except Exception as e:
            # If something goes wrong, clean up the temporary file
            temp_path = Path(output_path).with_suffix(Path(output_path).suffix + '.tmp')
            if temp_path.exists():
                temp_path.unlink()
            raise e
    
    def download_multiple_results(self, job_id: str, output_dir: str) -> list:
        """
        Downloads multiple result images from a single job if available.
        """
        # First, check if the job is completed
        status_info = self.poll_status(job_id)
        
        if status_info["status"] != "completed":
            raise RuntimeError(f"Cannot download results for job {job_id}: {status_info.get('message', 'Job not completed')}")
        
        # Extract all image information from the completed job
        outputs = status_info["outputs"]
        
        downloaded_files = []
        
        # Find all image outputs
        for node_id, node_data in outputs.items():
            if 'images' in node_data:
                for idx, image_info in enumerate(node_data['images']):
                    image_filename = image_info['filename']
                    image_subfolder = image_info.get('subfolder', '')
                    image_type = image_info.get('type', 'output')
                    
                    # Generate output filename
                    original_name = Path(image_filename).stem
                    extension = Path(image_filename).suffix
                    output_filename = f"{original_name}_{node_id}_out{idx}{extension}"
                    output_path = Path(output_dir) / output_filename
                    
                    # Construct the download URL
                    params = {
                        'filename': image_filename,
                        'type': image_type
                    }
                    if image_subfolder:
                        params['subfolder'] = image_subfolder
                    
                    # Download the image
                    try:
                        response = self.session.get(
                            f"{self.base_url}/view",
                            params=params,
                            stream=True,
                            timeout=self.timeout
                        )
                        response.raise_for_status()
                        
                        # Ensure the directory exists
                        Path(output_dir).mkdir(parents=True, exist_ok=True)
                        
                        # Stream download to file
                        with open(output_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        
                        # Verify the downloaded file is a valid image
                        with Image.open(output_path) as img:
                            img.verify()
                        
                        downloaded_files.append(str(output_path))
                        
                    except Exception as e:
                        print(f"Failed to download image {image_filename}: {str(e)}")
                        continue
        
        return downloaded_files

# Example usage
if __name__ == "__main__":
    client = ComfyUIClient()
    
    # Example job ID (this would come from a previous submit_edit call)
    example_job_id = "12345"
    
    try:
        # Download a single result
        success = client.download_result(example_job_id, "downloaded_result.png")
        if success:
            print("Result downloaded successfully!")
        
        # Or download multiple results if available
        multiple_results = client.download_multiple_results(example_job_id, "./output")
        print(f"Downloaded {len(multiple_results)} result files: {multiple_results}")
        
    except FileNotFoundError as e:
        print(f"File error: {e}")
    except RuntimeError as e:
        print(f"Runtime error: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
```

### Advanced Download Implementation
Enhanced download with retry logic, progress tracking, and validation:

```python
import requests
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from PIL import Image
import time
import hashlib
from urllib.parse import urlparse

class AdvancedComfyUIClient:
    def __init__(self, base_url: str = "http://localhost:8188", timeout: int = 300):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout
    
    def download_result(
        self, 
        job_id: str, 
        output_path: str, 
        max_retries: int = 3,
        validate_image: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> bool:
        """
        Downloads the result of a completed job to the specified path with advanced features.
        """
        # First, check if the job is completed
        from .poll_status import AdvancedComfyUIClient as PollClient  # This would be from the poll_status module
        # For this example, we'll implement the basic check
        status_info = self._check_job_completion(job_id)
        
        if not status_info["completed"]:
            raise RuntimeError(f"Job {job_id} is not completed: {status_info.get('message', 'Unknown status')}")
        
        # Extract image information from the completed job
        image_info = status_info.get("image_output")
        if not image_info:
            raise RuntimeError(f"No image output found for job {job_id}")
        
        # Attempt download with retry logic
        for attempt in range(max_retries):
            try:
                return self._download_with_progress(
                    image_info, 
                    output_path, 
                    validate_image, 
                    progress_callback
                )
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise e
                print(f"Download attempt {attempt + 1} failed: {str(e)}, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return False  # Should not reach here
    
    def _check_job_completion(self, job_id: str) -> Dict[str, Any]:
        """Check if a job has completed and return image output info."""
        try:
            response = self.session.get(
                f"{self.base_url}/history/{job_id}",
                timeout=self.timeout
            )
            response.raise_for_status()
            
            history = response.json()
            
            if str(job_id) in history:
                job_info = history[str(job_id)]
                
                if "outputs" in job_info and len(job_info["outputs"]) > 0:
                    # Find the first image output
                    for node_id, node_data in job_info["outputs"].items():
                        if 'images' in node_data:
                            return {
                                "completed": True,
                                "image_output": {
                                    "filename": node_data['images'][0]['filename'],
                                    "subfolder": node_data['images'][0].get('subfolder', ''),
                                    "type": node_data['images'][0].get('type', 'output'),
                                    "node_id": node_id
                                }
                            }
                
                return {"completed": False, "message": "Job exists but no output images found"}
            else:
                return {"completed": False, "message": "Job ID not found in history"}
                
        except requests.exceptions.RequestException as e:
            return {"completed": False, "message": f"Request error: {str(e)}"}
    
    def _download_with_progress(
        self,
        image_info: Dict[str, Any], 
        output_path: str, 
        validate_image: bool,
        progress_callback: Optional[Callable[[int, int], None]]
    ) -> bool:
        """Download image with progress tracking."""
        # Construct the download URL
        params = {
            'filename': image_info['filename'],
            'type': image_info['type']
        }
        if image_info['subfolder']:
            params['subfolder'] = image_info['subfolder']
        
        # Make the download request
        response = self.session.get(
            f"{self.base_url}/view",
            params=params,
            stream=True,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        # Get the total size if available
        total_size = int(response.headers.get('content-length', 0))
        
        # Use a temporary file for atomic write operation
        output_path_obj = Path(output_path)
        temp_path = output_path_obj.with_suffix(output_path_obj.suffix + '.tmp')
        
        # Ensure the directory exists
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress tracking
        downloaded_size = 0
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    if progress_callback and total_size > 0:
                        progress_callback(downloaded_size, total_size)
        
        # Validate the downloaded file if requested
        if validate_image:
            try:
                with Image.open(temp_path) as img:
                    img.verify()  # Verify it's a valid image file
            except Exception as e:
                temp_path.unlink()  # Remove invalid file
                raise ValueError(f"Downloaded file is not a valid image: {str(e)}")
        
        # Calculate checksum for integrity validation
        checksum = self._calculate_file_checksum(temp_path)
        
        # Atomic move: rename temporary file to final path
        shutil.move(str(temp_path), str(output_path_obj))
        
        print(f"Successfully downloaded result to: {output_path}")
        print(f"File checksum: {checksum}")
        return True
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read the file in chunks to handle large files efficiently
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def download_with_validation(self, job_id: str, output_path: str) -> Dict[str, Any]:
        """
        Download result with comprehensive validation.
        """
        try:
            success = self.download_result(job_id, output_path, validate_image=True)
            
            if success:
                # Additional validation after download
                file_path = Path(output_path)
                
                # Check file size
                file_size = file_path.stat().st_size
                
                # Check image properties
                with Image.open(file_path) as img:
                    width, height = img.size
                    mode = img.mode
                
                return {
                    "success": True,
                    "file_path": str(file_path),
                    "file_size_bytes": file_size,
                    "image_dimensions": f"{width}x{height}",
                    "image_mode": mode,
                    "checksum": self._calculate_file_checksum(file_path)
                }
            
            return {"success": False, "error": "Download failed"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}

# Progress callback example
def progress_callback(downloaded: int, total: int):
    """Example of a progress callback function."""
    if total > 0:
        percent = (downloaded / total) * 100
        print(f"\rDownload progress: {percent:.1f}% ({downloaded}/{total} bytes)", end='', flush=True)

# Example usage
if __name__ == "__main__":
    client = AdvancedComfyUIClient()
    
    # Example job ID (this would come from a previous submit_edit call)
    example_job_id = "12345"
    
    try:
        # Download with progress tracking
        print("Starting download with progress tracking...")
        result = client.download_with_validation(
            example_job_id, 
            "validated_result.png"
        )
        
        if result["success"]:
            print(f"\nDownload completed successfully!")
            print(f"File: {result['file_path']}")
            print(f"Size: {result['file_size_bytes']} bytes")
            print(f"Dimensions: {result['image_dimensions']}")
            print(f"Mode: {result['image_mode']}")
            print(f"Checksum: {result['checksum']}")
        else:
            print(f"\nDownload failed: {result['error']}")
        
    except Exception as e:
        print(f"\nError during download: {e}")
```

### Image Validation Implementation
Additional image validation and format handling:

```python
import requests
from PIL import Image, ImageChops, ImageStat
import io
from typing import Dict, Any, Tuple
from pathlib import Path

class ImageValidator:
    """Validates downloaded images for quality and correctness."""
    
    @staticmethod
    def validate_image_format(file_path: str, allowed_formats: set = None) -> Dict[str, Any]:
        """
        Validates that the downloaded file is a valid image in an allowed format.
        """
        if allowed_formats is None:
            allowed_formats = {'JPEG', 'PNG', 'BMP', 'GIF', 'TIFF', 'WEBP'}
        
        try:
            with Image.open(file_path) as img:
                # Verify it's a valid image file
                img.verify()
                
                # Reopen for format checking (verify() closes the file)
                with Image.open(file_path) as img:
                    format_name = img.format
                    
                    if format_name not in allowed_formats:
                        return {
                            "valid": False,
                            "error": f"Unsupported image format: {format_name}. Allowed: {allowed_formats}",
                            "format": format_name
                        }
                    
                    return {
                        "valid": True,
                        "format": format_name,
                        "message": f"Valid {format_name} image"
                    }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Invalid image file: {str(e)}",
                "format": "unknown"
            }
    
    @staticmethod
    def validate_image_quality(file_path: str, min_width: int = 100, min_height: int = 100) -> Dict[str, Any]:
        """
        Validates image quality and dimensions.
        """
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                mode = img.mode
                format_name = img.format
                
                # Check dimensions
                if width < min_width or height < min_height:
                    return {
                        "valid": False,
                        "error": f"Image too small: {width}x{height}, minimum required: {min_width}x{min_height}",
                        "dimensions": (width, height)
                    }
                
                # Basic quality checks
                # Check if image is completely black or white (common failure mode)
                if mode in ('RGB', 'RGBA', 'L'):
                    stat = ImageStat.Stat(img)
                    if all(mean < 5 for mean in stat.mean):  # Very dark image
                        return {
                            "valid": False,
                            "error": "Image appears to be completely black or near-black",
                            "dimensions": (width, height),
                            "mean_values": stat.mean
                        }
                    elif all(mean > 250 for mean in stat.mean):  # Very bright image
                        return {
                            "valid": False,
                            "error": "Image appears to be completely white or near-white", 
                            "dimensions": (width, height),
                            "mean_values": stat.mean
                        }
                
                return {
                    "valid": True,
                    "dimensions": (width, height),
                    "mode": mode,
                    "format": format_name,
                    "message": f"Valid image of size {width}x{height}"
                }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Error validating image quality: {str(e)}",
                "dimensions": None
            }
    
    @staticmethod
    def compare_with_original(original_path: str, result_path: str, threshold: float = 0.1) -> Dict[str, Any]:
        """
        Compares the result image with the original to ensure changes were made.
        """
        try:
            with Image.open(original_path) as orig_img, Image.open(result_path) as result_img:
                # Convert to same mode and size if different
                if orig_img.mode != result_img.mode:
                    result_img = result_img.convert(orig_img.mode)
                
                if orig_img.size != result_img.size:
                    result_img = result_img.resize(orig_img.size)
                
                # Calculate difference
                diff = ImageChops.difference(orig_img, result_img)
                
                # Calculate average difference
                stat = ImageStat.Stat(diff)
                avg_diff = sum(stat.mean) / len(stat.mean)
                
                return {
                    "similar": avg_diff < threshold,
                    "average_difference": avg_diff,
                    "threshold": threshold
                }
        except Exception as e:
            return {
                "error": f"Error comparing images: {str(e)}"
            }

# Example integration with the ComfyUIClient
class ValidatingComfyUIClient(ComfyUIClient):
    """ComfyUI Client with built-in image validation."""
    
    def download_and_validate(self, job_id: str, output_path: str, 
                             original_path: str = None) -> Dict[str, Any]:
        """
        Download and validate the result image.
        """
        try:
            # Download the result
            success = self.download_result(job_id, output_path)
            
            if not success:
                return {"success": False, "error": "Download failed"}
            
            # Validate image format
            format_validation = ImageValidator.validate_image_format(output_path)
            if not format_validation["valid"]:
                return {
                    "success": False,
                    "error": f"Format validation failed: {format_validation['error']}",
                    "validation_results": {"format_validation": format_validation}
                }
            
            # Validate image quality
            quality_validation = ImageValidator.validate_image_quality(output_path)
            if not quality_validation["valid"]:
                return {
                    "success": False,
                    "error": f"Quality validation failed: {quality_validation['error']}",
                    "validation_results": {
                        "format_validation": format_validation,
                        "quality_validation": quality_validation
                    }
                }
            
            # Compare with original if provided
            comparison_result = None
            if original_path:
                comparison_result = ImageValidator.compare_with_original(
                    original_path, output_path
                )
            
            return {
                "success": True,
                "file_path": output_path,
                "validation_results": {
                    "format_validation": format_validation,
                    "quality_validation": quality_validation,
                    "comparison_with_original": comparison_result
                }
            }
            
        except Exception as e:
            return {
                "success": False, 
                "error": f"Error during download and validation: {str(e)}"
            }

# Example usage
if __name__ == "__main__":
    client = ValidatingComfyUIClient()
    
    # Example validation
    result = client.download_and_validate(
        job_id="12345",
        output_path="validated_result.png",
        original_path="original_image.jpg"  # Optional
    )
    
    print(f"Validation result: {result}")
```