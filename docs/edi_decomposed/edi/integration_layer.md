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