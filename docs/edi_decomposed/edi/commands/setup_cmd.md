# Commands: Setup

[Back to Index](./index.md)

## Purpose
Setup command - Contains the setup_command async function that creates the ~/.edi/ directory structure, downloads models if requested, and verifies Ollama connection.

## Functions
- `async def setup_command(download_models=False)`: Sets up the EDI environment

### Details
- Creates ~/.edi/ directory structure
- Downloads default models if requested
- Verifies Ollama connection
- Prepares the system for EDI operation

## Technology Stack

- AsyncIO for asynchronous operations
- File system operations
- Model download utilities

## See Docs

### AsyncIO Implementation Example
Async setup command for the EDI application:

```python
import asyncio
import aiofiles
import aiohttp
import os
import sys
from pathlib import Path
from typing import Optional
import subprocess
import json

class EDISetupManager:
    """Manages the EDI setup process asynchronously."""
    
    def __init__(self, edi_home: Optional[Path] = None):
        self.edi_home = edi_home or Path.home() / ".edi"
        self.models_dir = self.edi_home / "models"
        self.config_dir = self.edi_home / "config"
        self.cache_dir = self.edi_home / "cache"
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def create_directory_structure(self):
        """Create the required directory structure for EDI."""
        directories = [
            self.edi_home,
            self.models_dir,
            self.config_dir,
            self.cache_dir,
            self.edi_home / "sessions",
            self.edi_home / "workflows",
            self.edi_home / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
    
    async def verify_ollama_connection(self) -> bool:
        """Verify that Ollama is running and accessible."""
        try:
            async with self.session.get("http://localhost:11434/api/tags") as response:
                if response.status == 200:
                    print("✓ Ollama connection verified")
                    return True
                else:
                    print(f"✗ Ollama connection failed with status: {response.status}")
                    return False
        except aiohttp.ClientError as e:
            print(f"✗ Ollama connection failed: {str(e)}")
            return False
        except Exception as e:
            print(f"✗ Unexpected error connecting to Ollama: {str(e)}")
            return False
    
    async def download_model(self, model_name: str, progress_callback=None) -> bool:
        """Download a model asynchronously."""
        try:
            print(f"Downloading model: {model_name}")
            
            # Use subprocess to run the Ollama pull command
            process = await asyncio.create_subprocess_exec(
                'ollama', 'pull', model_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                print(f"✓ Successfully downloaded model: {model_name}")
                return True
            else:
                print(f"✗ Failed to download model {model_name}: {stderr.decode()}")
                return False
        except Exception as e:
            print(f"✗ Error downloading model {model_name}: {str(e)}")
            return False
    
    async def download_default_models(self, models_to_download: list) -> dict:
        """Download default models asynchronously."""
        results = {}
        
        for model in models_to_download:
            success = await self.download_model(model)
            results[model] = success
            
            # Add a small delay between downloads to avoid overwhelming the system
            await asyncio.sleep(1)
        
        return results
    
    async def create_default_config(self):
        """Create default configuration files."""
        config = {
            "default_model": "qwen3:8b",
            "temperature": 0.7,
            "max_tokens": 4096,
            "image_processing": {
                "max_size": 1024,
                "format": "JPEG",
                "quality": 85
            },
            "ollama": {
                "host": "localhost",
                "port": 11434,
                "timeout": 300
            },
            "comfyui": {
                "host": "localhost",
                "port": 8188,
                "timeout": 300
            }
        }
        
        config_path = self.config_dir / "config.json"
        async with aiofiles.open(config_path, 'w') as f:
            await f.write(json.dumps(config, indent=2))
        
        print(f"Created default configuration at: {config_path}")
    
    async def setup_command(self, download_models: bool = False):
        """Main setup command that orchestrates the entire setup process."""
        print("Starting EDI setup process...")
        
        # Create directory structure
        await self.create_directory_structure()
        
        # Verify Ollama connection
        ollama_ok = await self.verify_ollama_connection()
        if not ollama_ok:
            print("Warning: Ollama not accessible. Setup will continue but some features may not work.")
        
        # Download models if requested
        if download_models:
            default_models = ["qwen3:8b", "gemma3:4b", "llava:latest"]
            print("Downloading default models...")
            download_results = await self.download_default_models(default_models)
            
            success_count = sum(1 for success in download_results.values() if success)
            print(f"Successfully downloaded {success_count}/{len(default_models)} models")
        else:
            print("Skipping model download (use --download-models to download default models)")
        
        # Create default configuration
        await self.create_default_config()
        
        print("\nEDI setup completed successfully!")
        print(f"EDI home directory: {self.edi_home}")
        print("You can now run 'edi' to start the application.")
        
        return True

# Example usage
async def setup_command(download_models=False):
    """Async setup command that creates the ~/.edi/ directory structure, downloads models if requested, and verifies Ollama connection."""
    async with EDISetupManager() as setup_mgr:
        return await setup_mgr.setup_command(download_models=download_models)

if __name__ == "__main__":
    # Example of running the setup
    import argparse
    
    parser = argparse.ArgumentParser(description='EDI Setup Command')
    parser.add_argument('--download-models', action='store_true', 
                       help='Download default models during setup')
    
    args = parser.parse_args()
    
    # Run the setup command
    success = asyncio.run(setup_command(download_models=args.download_models))
    
    if success:
        print("Setup completed successfully!")
        sys.exit(0)
    else:
        print("Setup failed!")
        sys.exit(1)
```

### File System Operations Implementation Example
File system setup utilities for the EDI application:

```python
import asyncio
import aiofiles
from pathlib import Path
import os
import shutil
import stat
from typing import List, Dict, Any
import tempfile
import zipfile
import tarfile

class FileSystemManager:
    """Manages file system operations for the EDI setup process."""
    
    def __init__(self, edi_home: Path):
        self.edi_home = edi_home
        self.setup_log_path = edi_home / "logs" / "setup.log"
    
    async def ensure_directory_exists(self, path: Path) -> bool:
        """Ensure a directory exists, creating it if necessary."""
        try:
            path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"Error creating directory {path}: {e}")
            return False
    
    async def write_setup_log(self, message: str):
        """Asynchronously write a message to the setup log."""
        await self.ensure_directory_exists(self.setup_log_path.parent)
        
        async with aiofiles.open(self.setup_log_path, 'a') as f:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            await f.write(f"[{timestamp}] {message}\n")
    
    async def check_permissions(self, path: Path) -> Dict[str, bool]:
        """Check read/write/execute permissions for a path."""
        try:
            mode = path.stat().st_mode
            return {
                "readable": bool(mode & stat.S_IRUSR),
                "writable": bool(mode & stat.S_IWUSR),
                "executable": bool(mode & stat.S_IXUSR)
            }
        except Exception:
            # If path doesn't exist, return default permissions for parent
            if path.parent.exists():
                parent_mode = path.parent.stat().st_mode
                return {
                    "readable": bool(parent_mode & stat.S_IRUSR),
                    "writable": bool(parent_mode & stat.S_IWUSR),
                    "executable": bool(parent_mode & stat.S_IXUSR)
                }
            else:
                return {"readable": False, "writable": False, "executable": False}
    
    async def copy_template_files(self, source_dir: str, target_dir: Path):
        """Copy template files from source to target directory."""
        source_path = Path(source_dir)
        if not source_path.exists():
            print(f"Source directory does not exist: {source_path}")
            return
        
        await self.ensure_directory_exists(target_dir)
        
        for item in source_path.iterdir():
            target_item = target_dir / item.name
            if item.is_file():
                # Copy file asynchronously
                async with aiofiles.open(item, 'rb') as src:
                    content = await src.read()
                
                async with aiofiles.open(target_item, 'wb') as dst:
                    await dst.write(content)
                
                print(f"Copied file: {item.name}")
            elif item.is_dir():
                # Recursively copy directory
                await self.copy_template_files(str(item), target_item)
    
    async def validate_disk_space(self, required_space_mb: int) -> bool:
        """Check if there's enough disk space available."""
        # Get disk usage statistics
        try:
            total, used, free = shutil.disk_usage(self.edi_home)
            free_mb = free // (1024 * 1024)  # Convert to MB
            
            if free_mb >= required_space_mb:
                print(f"✓ Sufficient disk space: {free_mb}MB free (need {required_space_mb}MB)")
                return True
            else:
                print(f"✗ Insufficient disk space: {free_mb}MB free (need {required_space_mb}MB)")
                return False
        except Exception as e:
            print(f"Error checking disk space: {e}")
            return False
    
    async def cleanup_temp_files(self, temp_dir: Path):
        """Clean up temporary files after setup."""
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")

class ModelDownloadManager:
    """Manages model download utilities for EDI."""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
    
    async def download_model_from_url(self, url: str, filename: str) -> bool:
        """Download a model file from a URL."""
        import aiohttp
        
        await FileSystemManager(self.models_dir).ensure_directory_exists(self.models_dir)
        
        model_path = self.models_dir / filename
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        async with aiofiles.open(model_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                await f.write(chunk)
                        
                        print(f"Downloaded model: {filename}")
                        return True
                    else:
                        print(f"Failed to download {url}: HTTP {response.status}")
                        return False
        except Exception as e:
            print(f"Error downloading model: {e}")
            return False
    
    async def verify_model_integrity(self, model_path: Path, expected_hash: str = None) -> bool:
        """Verify the integrity of a downloaded model."""
        if not model_path.exists():
            return False
        
        if expected_hash:
            import hashlib
            
            async with aiofiles.open(model_path, 'rb') as f:
                content = await f.read()
                actual_hash = hashlib.sha256(content).hexdigest()
            
            if actual_hash.lower() != expected_hash.lower():
                print(f"Model integrity check failed for {model_path}")
                return False
        
        print(f"Model integrity verified: {model_path}")
        return True
    
    async def extract_model_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """Extract a model archive (zip or tar) to the specified directory."""
        try:
            if archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix.lower() in ['.tar', '.tar.gz', '.tgz']:
                with tarfile.open(archive_path, 'r') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                print(f"Unsupported archive format: {archive_path.suffix}")
                return False
            
            print(f"Extracted archive: {archive_path} to {extract_to}")
            return True
        except Exception as e:
            print(f"Error extracting archive {archive_path}: {e}")
            return False

# Example usage
async def main():
    # Example of using the file system manager
    edi_home = Path.home() / ".edi-test"
    fs_manager = FileSystemManager(edi_home)
    
    # Create directory structure
    await fs_manager.ensure_directory_exists(edi_home / "models")
    await fs_manager.ensure_directory_exists(edi_home / "config")
    await fs_manager.ensure_directory_exists(edi_home / "logs")
    
    # Check permissions
    permissions = await fs_manager.check_permissions(edi_home)
    print(f"EDI home permissions: {permissions}")
    
    # Validate disk space (require 1GB)
    has_space = await fs_manager.validate_disk_space(1024)  # 1GB in MB
    print(f"Has sufficient space: {has_space}")
    
    # Example of using the model download manager
    model_manager = ModelDownloadManager(edi_home / "models")
    
    # Write to log
    await fs_manager.write_setup_log("Setup process started")
    
    print("File system operations example completed!")

if __name__ == "__main__":
    asyncio.run(main())
```