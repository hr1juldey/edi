# ComfyUIClient.poll_status()

[Back to ComfyUI Client](../integration_comfyui_client.md)

## Related User Story
"As a user, I want to see progress updates while my image is being edited." (from PRD - implied by user experience requirements)

## Function Signature
`poll_status(job_id: str) -> dict`

## Parameters
- `job_id: str` - The job ID of the edit job to check status for

## Returns
- `dict` - A dictionary containing the status information for the job

## Step-by-step Logic
1. Make an API request to ComfyUI to check the status of the job with the given ID
2. Parse the response from the ComfyUI history endpoint
3. Extract the status information (queued, processing, completed, failed)
4. Include additional metadata like progress percentage if available
5. Handle API errors and return appropriate status information
6. Return the status information as a dictionary

## Status Values
- 'queued': Job is waiting in the queue
- 'processing': Job is currently being processed
- 'completed': Job has completed successfully
- 'failed': Job has failed due to an error

## Polling Strategy
- Designed to be called periodically during long operations
- Includes appropriate delays to avoid overwhelming the server
- Provides estimated time remaining when possible
- Handles scenarios where job IDs are no longer valid

## Input/Output Data Structures
### Status Response Object
A dictionary containing:
- Status (queued, processing, completed, failed)
- Progress percentage (0-100 if available)
- Estimated time remaining (if calculable)
- Output information (image paths, etc. when completed)
- Error information (if any occurred)

## See Docs

### Python Implementation Example
Implementation of the poll_status method for ComfyUIClient:

```python
import requests
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

class ComfyUIClient:
    def __init__(self, base_url: str = "http://localhost:8188", timeout: int = 300):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout
        
        # Track job start times for time estimation
        self.job_start_times = {}
    
    def poll_status(self, job_id: str) -> Dict[str, Any]:
        """
        Checks the status of a submitted job.
        Returns status and additional information.
        """
        try:
            response = self.session.get(
                f"{self.base_url}/history/{job_id}",
                timeout=self.timeout
            )
            
            # Handle the response based on HTTP status
            if response.status_code == 404:
                return {
                    "status": "not_found",
                    "message": f"Job {job_id} not found in history",
                    "job_id": job_id,
                    "timestamp": datetime.now().isoformat()
                }
            
            response.raise_for_status()
            
            history = response.json()
            
            if str(job_id) in history:
                job_info = history[str(job_id)]
                
                # Determine status based on job info structure
                if "outputs" in job_info and any(
                    "images" in output or "gifs" in output 
                    for output in job_info.get("outputs", {}).values()
                ):
                    # Job is completed
                    return {
                        "status": "completed",
                        "job_id": job_id,
                        "outputs": job_info.get("outputs", {}),
                        "timestamp": datetime.now().isoformat(),
                        "message": "Job completed successfully"
                    }
                elif "status" in job_info and job_info["status"].get("completed", False):
                    # Job marked as completed in status
                    return {
                        "status": "completed",
                        "job_id": job_id,
                        "outputs": job_info.get("outputs", {}),
                        "timestamp": datetime.now().isoformat(),
                        "message": "Job completed successfully"
                    }
                else:
                    # Check if job is in queue or processing
                    if job_info.get("status", {}).get("status_str") == "Running":
                        # Job is currently being processed
                        progress_info = self._calculate_progress(job_id, job_info)
                        return {
                            "status": "processing",
                            "job_id": job_id,
                            "progress": progress_info,
                            "message": "Job is currently being processed",
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        # Job might be queued
                        return {
                            "status": "queued",
                            "job_id": job_id,
                            "message": "Job is queued for processing",
                            "timestamp": datetime.now().isoformat()
                        }
            else:
                # Job ID not found in history
                return {
                    "status": "not_found",
                    "message": f"Job {job_id} not found in history",
                    "job_id": job_id,
                    "timestamp": datetime.now().isoformat()
                }
                
        except requests.exceptions.Timeout:
            return {
                "status": "error",
                "message": f"Timeout while checking status for job {job_id}",
                "job_id": job_id,
                "timestamp": datetime.now().isoformat()
            }
        except requests.exceptions.ConnectionError:
            return {
                "status": "error",
                "message": f"Connection error while checking status for job {job_id}",
                "job_id": job_id,
                "timestamp": datetime.now().isoformat()
            }
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 500:
                return {
                    "status": "failed",
                    "message": f"Server error while checking status for job {job_id}",
                    "job_id": job_id,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "error", 
                    "message": f"HTTP error while checking status for job {job_id}: {str(e)}",
                    "job_id": job_id,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Unexpected error while checking status for job {job_id}: {str(e)}",
                "job_id": job_id,
                "timestamp": datetime.now().isoformat()
            }
    
    def _calculate_progress(self, job_id: str, job_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates progress information for a processing job.
        """
        progress_info = {
            "percentage": 0,
            "estimated_time_remaining": None,
            "current_step": None,
            "total_steps": None
        }
        
        # Try to extract progress from job info
        outputs = job_info.get("outputs", {})
        for node_id, node_outputs in outputs.items():
            if "progress" in node_outputs:
                progress_data = node_outputs["progress"]
                if "value" in progress_data and "max" in progress_data:
                    current = progress_data["value"]
                    total = progress_data["max"]
                    
                    if total > 0:
                        progress_info["percentage"] = int((current / total) * 100)
                        progress_info["current_step"] = current
                        progress_info["total_steps"] = total
                        
                        # Estimate time remaining if we have the start time
                        if job_id in self.job_start_times:
                            start_time = self.job_start_times[job_id]
                            elapsed = datetime.now() - start_time
                            estimated_total_time = elapsed / progress_info["percentage"] * 100 if progress_info["percentage"] > 0 else 0
                            remaining = estimated_total_time - elapsed
                            progress_info["estimated_time_remaining"] = str(remaining)
        
        return progress_info
    
    def wait_for_completion(self, job_id: str, timeout: int = 600, poll_interval: float = 5.0) -> Dict[str, Any]:
        """
        Waits for a job to complete with polling.
        """
        # Record start time to calculate progress
        self.job_start_times[job_id] = datetime.now()
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.poll_status(job_id)
            
            if status["status"] == "completed":
                # Clean up start time tracking
                if job_id in self.job_start_times:
                    del self.job_start_times[job_id]
                return status
            elif status["status"] == "failed":
                # Clean up start time tracking
                if job_id in self.job_start_times:
                    del self.job_start_times[job_id]
                return status
            elif status["status"] == "error":
                # Clean up start time tracking
                if job_id in self.job_start_times:
                    del self.job_start_times[job_id]
                return status
            else:
                # Wait before polling again
                time.sleep(poll_interval)
        
        # Clean up start time tracking
        if job_id in self.job_start_times:
            del self.job_start_times[job_id]
            
        return {
            "status": "timeout",
            "message": f"Job {job_id} did not complete within {timeout} seconds",
            "job_id": job_id,
            "timestamp": datetime.now().isoformat()
        }

# Example usage
if __name__ == "__main__":
    client = ComfyUIClient()
    
    # Example job ID (this would come from a previous submit_edit call)
    example_job_id = "12345"
    
    # Poll for status once
    status = client.poll_status(example_job_id)
    print(f"Current status: {status}")
    
    # Or wait for completion with polling
    print("Waiting for job to complete...")
    final_status = client.wait_for_completion(example_job_id, timeout=300, poll_interval=2)
    print(f"Final status: {final_status}")
```

### Advanced Polling Implementation
Enhanced polling with progress tracking and error recovery:

```python
import requests
import time
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import asyncio

class AdvancedComfyUIClient:
    def __init__(self, base_url: str = "http://localhost:8188", timeout: int = 300):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout
        
        # Track job metadata
        self.job_metadata = {}
        
        # Callbacks for status updates
        self.status_callbacks = {}
    
    def register_status_callback(self, job_id: str, callback: Callable[[Dict[str, Any]], None]):
        """Register a callback to be called when status updates occur."""
        self.status_callbacks[job_id] = callback
    
    def poll_status(self, job_id: str) -> Dict[str, Any]:
        """
        Checks the status of a submitted job with enhanced error handling.
        """
        try:
            response = self.session.get(
                f"{self.base_url}/history/{job_id}",
                timeout=self.timeout
            )
            
            # Handle different response codes
            if response.status_code == 404:
                # Job might not be registered yet, could be in queue
                return self._check_queue_status(job_id)
            elif response.status_code >= 400:
                response.raise_for_status()
            
            history = response.json()
            
            return self._parse_job_status(job_id, history)
                
        except requests.exceptions.Timeout:
            return self._create_error_response(job_id, "timeout")
        except requests.exceptions.ConnectionError:
            return self._create_error_response(job_id, "connection_error")
        except requests.exceptions.HTTPError as e:
            return self._create_error_response(job_id, "http_error", str(e))
        except Exception as e:
            return self._create_error_response(job_id, "unexpected_error", str(e))
    
    def _check_queue_status(self, job_id: str) -> Dict[str, Any]:
        """Check the queue status to see if job exists but isn't in history yet."""
        try:
            queue_response = self.session.get(f"{self.base_url}/queue", timeout=5)
            queue_response.raise_for_status()
            queue_data = queue_response.json()
            
            # Check if job is in the queue
            queue_running = queue_data.get("queue_running", [])
            queue_pending = queue_data.get("queue_pending", [])
            
            all_jobs = queue_running + queue_pending
            for queue_job in all_jobs:
                if str(queue_job[0]) == str(job_id):  # Check if job_id matches
                    return {
                        "status": "queued",
                        "job_id": job_id,
                        "position_in_queue": queue_pending.index(queue_job) if queue_job in queue_pending else 0,
                        "queue_size": len(queue_pending),
                        "message": f"Job is in queue at position {queue_pending.index(queue_job) + 1} of {len(queue_pending)}",
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Job not found anywhere
            return {
                "status": "not_found",
                "message": f"Job {job_id} not found in queue or history",
                "job_id": job_id,
                "timestamp": datetime.now().isoformat()
            }
        except:
            # If we can't check the queue, assume it's not found
            return {
                "status": "not_found",
                "message": f"Job {job_id} not found in history",
                "job_id": job_id,
                "timestamp": datetime.now().isoformat()
            }
    
    def _parse_job_status(self, job_id: str, history: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the response from ComfyUI history endpoint."""
        if str(job_id) in history:
            job_info = history[str(job_id)]
            
            # Determine status based on job info
            if "outputs" in job_info and job_info["outputs"]:
                # Check if any output contains error information
                has_error = False
                error_msg = ""
                
                for node_id, node_outputs in job_info["outputs"].items():
                    if "error" in node_outputs:
                        has_error = True
                        error_msg = node_outputs["error"]
                        break
                
                if has_error:
                    return {
                        "status": "failed",
                        "job_id": job_id,
                        "error": error_msg,
                        "outputs": job_info.get("outputs", {}),
                        "timestamp": datetime.now().isoformat(),
                        "message": f"Job failed: {error_msg}"
                    }
                else:
                    return {
                        "status": "completed",
                        "job_id": job_id,
                        "outputs": job_info.get("outputs", {}),
                        "timestamp": datetime.now().isoformat(),
                        "message": "Job completed successfully"
                    }
            else:
                # Job exists but no outputs yet, so it's processing
                return {
                    "status": "processing",
                    "job_id": job_id,
                    "message": "Job is being processed",
                    "timestamp": datetime.now().isoformat(),
                    "progress": self._extract_progress(job_id, job_info)
                }
        else:
            # Job not in history, but might be queued
            return self._check_queue_status(job_id)
    
    def _extract_progress(self, job_id: str, job_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract progress information from job info."""
        progress_info = {
            "percentage": 0,
            "current_step": None,
            "total_steps": None,
            "estimated_time_remaining": None
        }
        
        # Look for progress information in the job data
        outputs = job_info.get("outputs", {})
        for output in outputs.values():
            if "progress" in output:
                progress = output["progress"]
                if "value" in progress and "max" in progress:
                    current = progress["value"]
                    total = progress["max"] 
                    
                    if total > 0:
                        progress_info["percentage"] = min(int((current / total) * 100), 100)
                        progress_info["current_step"] = current
                        progress_info["total_steps"] = total
        
        return progress_info
    
    def _create_error_response(self, job_id: str, error_type: str, details: str = "") -> Dict[str, Any]:
        """Create a standardized error response."""
        error_messages = {
            "timeout": "Timeout while checking status",
            "connection_error": "Connection error while checking status",
            "http_error": f"HTTP error while checking status: {details}",
            "unexpected_error": f"Unexpected error while checking status: {details}"
        }
        
        return {
            "status": "error",
            "error_type": error_type,
            "message": error_messages[error_type],
            "job_id": job_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def poll_with_callback(self, job_id: str, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """
        Poll status and call a callback with the result.
        """
        status = self.poll_status(job_id)
        
        # Call registered callback if exists
        if job_id in self.status_callbacks:
            self.status_callbacks[job_id](status)
        
        # Call provided callback
        if callback:
            callback(status)
        
        return status

# Example usage of advanced client
if __name__ == "__main__":
    client = AdvancedComfyUIClient()
    
    # Define a callback function to handle status updates
    def status_update_handler(status):
        print(f"Status update for job {status['job_id']}: {status['status']}")
        if status['status'] == 'processing' and 'progress' in status:
            progress = status['progress']
            if progress['percentage'] > 0:
                print(f"  Progress: {progress['percentage']}% ({progress['current_step']}/{progress['total_steps']})")
    
    # Register the callback for a job
    example_job_id = "12345"
    client.register_status_callback(example_job_id, status_update_handler)
    
    # Poll with the callback
    status = client.poll_with_callback(example_job_id)
    print(f"Final status: {status['status']}")
```