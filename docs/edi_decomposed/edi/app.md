# UI: App

[Back to TUI Layer](../tui_layer.md)

## Purpose

Main Textual App class - Contains the main application class that coordinates screen transitions, manages global state, and handles keyboard shortcuts.

## Class: App

### Methods

- Coordinates screen transitions
- Manages global session state
- Handles keyboard shortcuts (Q, H, B)

### Details

- Main entry point for the Textual TUI
- Manages the overall application flow
- Provides consistent navigation across screens

## Technology Stack

- Textual for TUI framework
- AsyncIO for non-blocking operations

## See Docs

### Textual Implementation Example

Main Textual App class for the EDI application:

```python
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual import work
import asyncio
import aiofiles
from typing import Dict, Any, Optional
import json

# EDI-specific screens
class HomeScreen(Screen):
    """Main home screen of the EDI application."""
    
    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static("Welcome to EDI: Edit with Intelligence", classes="title"),
            Static("Your AI-powered image editing assistant", classes="subtitle"),
            Container(
                Static("[1] Start new edit", id="start-new-edit", classes="menu-item"),
                Static("[2] Load recent session", id="load-session", classes="menu-item"),
                Static("[3] View examples", id="view-examples", classes="menu-item"),
                Static("[Q] Quit", id="quit", classes="menu-item"),
                classes="menu-container"
            ),
            classes="main-content"
        )
        yield Footer()

class ImageUploadScreen(Screen):
    """Screen for uploading images to edit."""
    
    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static("Upload Image", classes="screen-title"),
            Container(
                Static("Choose an image to edit:", classes="instructions"),
                Static("[U] Upload image file", id="upload-btn", classes="action-btn"),
                Static("[D] Use default image", id="default-btn", classes="action-btn"),
                Static("[B] Back", id="back-btn", classes="action-btn"),
                classes="upload-container"
            ),
            classes="main-content"
        )
        yield Footer()

class EDIMainApp(App):
    """Main Textual application class for EDI (Edit with Intelligence)."""
    
    TITLE = "EDI: Edit with Intelligence"
    SUB_TITLE = "AI-Powered Image Editing Assistant"
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    .title {
        text-align: center;
        content-align: center middle;
        height: 2;
        text-style: bold;
        color: $accent;
    }
    
    .subtitle {
        text-align: center;
        content-align: center middle;
        height: 1;
        color: $text-muted;
        margin-bottom: 1;
    }
    
    .main-content {
        height: 1fr;
        width: 1fr;
        content-align: center middle;
    }
    
    .menu-container {
        border: round $primary;
        padding: 1;
        width: 50%;
        height: auto;
    }
    
    .menu-item {
        height: 1;
        margin: 1 0;
        text-align: center;
        border-bottom: solid $surface;
    }
    
    .screen-title {
        text-align: center;
        text-style: bold;
        height: 2;
        color: $success;
    }
    
    .instructions {
        text-align: center;
        margin-bottom: 2;
    }
    
    .action-btn {
        text-align: center;
        height: 1;
        margin: 1 0;
        padding: 1;
        border: solid $primary;
        background: $panel;
    }
    """
    
    # Keyboard bindings
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("h", "show_help", "Help"),
        ("b", "go_back", "Back"),
        ("escape", "go_back", "Back"),
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Global session state
        self.session_state: Dict[str, Any] = {
            "current_image_path": None,
            "edit_history": [],
            "user_preferences": {
                "auto_save": True,
                "theme": "dark",
                "default_model": "qwen3:8b"
            }
        }
        # Available screens
        self.screen_stack = []
    
    def on_mount(self) -> None:
        """Called when the app is first mounted."""
        # Set initial screen
        self.push_screen(HomeScreen())
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        # This is handled by screens in our case, but we can have global widgets here
        pass
    
    def push_screen(self, screen: Screen) -> None:
        """Push a screen onto the stack."""
        if self.screen_stack:
            current_screen = self.screen_stack[-1]
            self.pop_screen()
        self.screen_stack.append(screen)
        super().push_screen(screen)
    
    def pop_screen(self) -> None:
        """Pop the current screen from the stack."""
        if self.screen_stack:
            self.screen_stack.pop()
        super().pop_screen()
    
    def action_quit(self) -> None:
        """Action to quit the application."""
        self.exit()
    
    def action_show_help(self) -> None:
        """Action to show help information."""
        help_text = """
        [b]EDI - Edit with Intelligence Help[/b]
        
        [u]Global Shortcuts:[/u]
        Q - Quit the application
        H - Show this help screen
        B - Go back to previous screen
        ESC - Go back to previous screen
        
        [u]Navigation:[/u]
        Arrow keys - Move cursor
        Tab/Shift+Tab - Cycle through elements
        Enter - Confirm selection
        
        [u]Editing Shortcuts:[/u]
        E - Edit prompt
        R - Retry edit
        A - Accept result
        V - View variations
        """
        
        # In a real app, we would show this in a modal dialog
        self.notify("Help information would be displayed in a modal dialog")
    
    def action_go_back(self) -> None:
        """Action to go back to previous screen."""
        if len(self.screen_stack) > 1:
            self.pop_screen()
        else:
            # If on the home screen, go back to home instead of exiting
            if not isinstance(self.screen, HomeScreen):
                self.push_screen(HomeScreen())
    
    def update_session_state(self, key: str, value: Any) -> None:
        """Update a value in the global session state."""
        self.session_state[key] = value
    
    def get_session_state(self, key: str, default: Any = None) -> Any:
        """Get a value from the global session state."""
        return self.session_state.get(key, default)
    
    @work(exclusive=True, thread=True)  # Run in a separate thread to not block UI
    def load_recent_sessions(self) -> list:
        """Load recent session data from storage."""
        # This would load from the storage layer in a real implementation
        recent_sessions = []
        try:
            # Simulate loading from a JSON file
            # In real implementation, this would use the storage layer
            session_file = "recent_sessions.json"
            if os.path.exists(session_file):
                with open(session_file, 'r') as f:
                    recent_sessions = json.load(f)
        except Exception as e:
            self.notify(f"Error loading sessions: {str(e)}", severity="error")
        
        return recent_sessions

# Example usage
if __name__ == "__main__":
    import os
    
    # Create a sample recent sessions file for testing
    sample_sessions = [
        {"id": "session-001", "image": "sunset.jpg", "date": "2023-10-20"},
        {"id": "session-002", "image": "portrait.jpg", "date": "2023-10-21"},
        {"id": "session-003", "image": "landscape.jpg", "date": "2023-10-22"}
    ]
    
    with open("recent_sessions.json", "w") as f:
        json.dump(sample_sessions, f)
    
    # Run the application
    app = EDIMainApp()
    app.run()
```

### AsyncIO Implementation Example

Asynchronous operations for the EDI application:

```python
import asyncio
import aiofiles
import aiohttp
from typing import Dict, Any, List, Optional
import json
import time
from pathlib import Path

class EDIAsyncManager:
    """Manages asynchronous operations for the EDI application."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.operation_queue = asyncio.Queue()
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def process_image_edit(self, 
                                image_path: str, 
                                positive_prompt: str, 
                                negative_prompt: str,
                                model: str = "qwen3:8b") -> Dict[str, Any]:
        """Process an image edit request asynchronously."""
        try:
            # Upload image to server
            upload_result = await self.upload_image_async(image_path)
            task_id = upload_result.get("task_id")
            
            # Submit edit job to the backend
            job_result = await self.submit_edit_job_async(
                task_id, positive_prompt, negative_prompt, model
            )
            
            job_id = job_result.get("job_id")
            
            # Poll for completion
            result = await self.wait_for_completion_async(job_id)
            
            # Download the result
            output_path = await self.download_result_async(result, image_path)
            
            return {
                "status": "success",
                "output_path": output_path,
                "job_id": job_id
            }
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e)
            }
    
    async def upload_image_async(self, image_path: str) -> Dict[str, Any]:
        """Asynchronously upload an image to the backend."""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        data = aiohttp.FormData()
        data.add_field('image', image_data, filename=Path(image_path).name)
        
        async with self.session.post("http://localhost:8188/upload/image", data=data) as response:
            return await response.json()
    
    async def submit_edit_job_async(self, 
                                   task_id: str, 
                                   positive_prompt: str, 
                                   negative_prompt: str, 
                                   model: str) -> Dict[str, Any]:
        """Asynchronously submit an edit job to the backend."""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        payload = {
            "task_id": task_id,
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "model": model
        }
        
        async with self.session.post("http://localhost:8188/submit-job", json=payload) as response:
            return await response.json()
    
    async def wait_for_completion_async(self, job_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Asynchronously wait for a job to complete."""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            async with self.session.get(f"http://localhost:8188/job-status/{job_id}") as response:
                status = await response.json()
                
                if status.get("status") == "completed":
                    return status
                elif status.get("status") == "failed":
                    raise RuntimeError(f"Job {job_id} failed")
            
            # Wait before polling again
            await asyncio.sleep(5)
        
        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")
    
    async def download_result_async(self, result: Dict[str, Any], original_path: str) -> str:
        """Asynchronously download the edited image result."""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        output_path = original_path.replace(".", "_edited.")
        
        file_url = result.get("file_url")
        if not file_url:
            raise ValueError("No file URL provided in result")
        
        async with self.session.get(file_url) as response:
            with open(output_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)
        
        return output_path
    
    async def batch_process_images(self, 
                                  image_paths: List[str], 
                                  prompt: str) -> List[Dict[str, Any]]:
        """Process multiple images in parallel."""
        tasks = [
            self.process_image_edit(img_path, prompt, f"avoid changes to background in {img_path}")
            for img_path in image_paths
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Format results
        formatted_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                formatted_results.append({
                    "image_path": image_paths[i],
                    "status": "error",
                    "error": str(result)
                })
            else:
                formatted_results.append({
                    "image_path": image_paths[i],
                    "status": result["status"],
                    "output_path": result.get("output_path", ""),
                    "job_id": result.get("job_id", "")
                })
        
        return formatted_results
    
    async def save_session_async(self, session_data: Dict[str, Any], filepath: str) -> None:
        """Asynchronously save session data to a file."""
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(json.dumps(session_data, indent=2))
    
    async def load_session_async(self, filepath: str) -> Dict[str, Any]:
        """Asynchronously load session data from a file."""
        async with aiofiles.open(filepath, 'r') as f:
            content = await f.read()
            return json.loads(content)

class EDIAsyncApp:
    """Async application manager that coordinates async operations."""
    
    def __init__(self):
        self.async_manager = EDIAsyncManager()
        self.active_jobs: Dict[str, asyncio.Task] = {}
    
    async def start_image_edit(self, 
                              image_path: str, 
                              positive_prompt: str, 
                              negative_prompt: str) -> str:
        """Start an image edit operation in the background."""
        # Create a task ID
        import uuid
        task_id = str(uuid.uuid4())
        
        # Create the async task
        task = asyncio.create_task(
            self.async_manager.process_image_edit(
                image_path, positive_prompt, negative_prompt
            )
        )
        
        # Store the task
        self.active_jobs[task_id] = task
        
        # Return the task ID so the caller can check on it later
        return task_id
    
    async def get_job_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of an active job."""
        if task_id not in self.active_jobs:
            return {"status": "unknown", "error": f"Task {task_id} not found"}
        
        task = self.active_jobs[task_id]
        
        if task.done():
            try:
                result = task.result()
                # Remove completed task
                del self.active_jobs[task_id]
                return {"status": "completed", "result": result}
            except Exception as e:
                # Remove failed task
                del self.active_jobs[task_id]
                return {"status": "failed", "error": str(e)}
        else:
            return {"status": "running", "progress": "in_progress"}
    
    async def cancel_job(self, task_id: str) -> bool:
        """Cancel an active job."""
        if task_id not in self.active_jobs:
            return False
        
        task = self.active_jobs[task_id]
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        del self.active_jobs[task_id]
        return True

# Example usage
async def main():
    # Example of using the async manager directly
    async with EDIAsyncManager() as mgr:
        result = await mgr.process_image_edit(
            "input.jpg",
            "make the sky more dramatic with storm clouds",
            "keep foreground unchanged, no cartoon effects"
        )
        print(f"Edit result: {result}")
    
    # Example of using the async app manager
    app = EDIAsyncApp()
    
    # Start a job
    task_id = await app.start_image_edit(
        "landscape.jpg",
        "enhance colors and contrast",
        "avoid oversaturation and artifacts"
    )
    print(f"Started job with ID: {task_id}")
    
    # Check status periodically
    for _ in range(10):  # Check 10 times
        status = await app.get_job_status(task_id)
        print(f"Job status: {status['status']}")
        
        if status['status'] in ['completed', 'failed']:
            break
        
        await asyncio.sleep(2)  # Wait 2 seconds before checking again
    
    # Batch processing example
    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
    batch_results = await app.async_manager.batch_process_images(
        image_paths, 
        "improve lighting and clarity"
    )
    
    print(f"Batch processing results: {len(batch_results)} items processed")

if __name__ == "__main__":
    # Run the async example
    asyncio.run(main())
```
