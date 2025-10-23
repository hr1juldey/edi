# Commands: Doctor

[Back to Index](./index.md)

## Purpose

Diagnostic command - Contains the doctor_command async function that checks Python version, GPU availability, models, Ollama connection, ComfyUI connection, and outputs system diagnostics.

## Functions

- `async def doctor_command()`: Performs system diagnostics

### Details

- Checks Python version, GPU availability, models
- Tests Ollama connection, ComfyUI connection
- Outputs green checkmarks or red errors
- Provides comprehensive system health check

## Technology Stack

- System diagnostic utilities
- AsyncIO for asynchronous operations
- Hardware detection

## See Docs

### AsyncIO Implementation Example

Async doctor command for the EDI application:

```python
import asyncio
import aiohttp
import sys
import subprocess
import json
from typing import Dict, Any, List
import platform
import psutil
from pathlib import Path

class EDIDoctor:
    """Performs system diagnostics for the EDI application."""
    
    def __init__(self):
        self.diagnostics: Dict[str, Any] = {}
        self.session: aiohttp.ClientSession = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def check_python_version(self) -> Dict[str, Any]:
        """Check if the Python version meets requirements."""
        min_version = (3, 8)
        current_version = sys.version_info[:2]
        
        result = {
            "name": "Python Version",
            "current": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "required": f"{min_version[0]}.{min_version[1]}+",
            "status": "ok" if current_version >= min_version else "error",
            "message": f"Python {current_version[0]}.{current_version[1]}{'+' if current_version >= min_version else ' (min required: 3.8)'}"
        }
        
        return result
    
    async def check_gpu_availability(self) -> Dict[str, Any]:
        """Check for GPU availability and capabilities."""
        try:
            # Check for CUDA
            nvidia_smi_result = await self.run_command_async(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
            if nvidia_smi_result.returncode == 0:
                gpu_info = nvidia_smi_result.stdout.decode().strip().split('\n')[0] if nvidia_smi_result.stdout else "Unknown"
                return {
                    "name": "GPU Availability (CUDA)",
                    "current": gpu_info,
                    "required": "Optional",
                    "status": "ok",
                    "message": f"GPU available: {gpu_info}"
                }
        except Exception:
            pass
        
        try:
            # Check for OpenCL (CPU/Metal)
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                return {
                    "name": "GPU Availability (PyTorch)",
                    "current": gpu_name,
                    "required": "Optional",
                    "status": "ok",
                    "message": f"GPU available: {gpu_name}"
                }
        except ImportError:
            pass
        except Exception as e:
            pass
        
        # If no GPU found, check if CPU is available
        return {
            "name": "GPU Availability",
            "current": "CPU only",
            "required": "Optional",
            "status": "warning",  # Warning because GPU would improve performance
            "message": "GPU not available - using CPU (slower processing)"
        }
    
    async def check_system_resources(self) -> Dict[str, Any]:
        """Check system resources (CPU, RAM, disk)."""
        # Get CPU info
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get memory info
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        available_memory_gb = memory.available / (1024**3)
        
        # Get disk info for home directory
        disk_usage = psutil.disk_usage(Path.home())
        available_disk_gb = disk_usage.free / (1024**3)
        
        # Determine status based on resource availability
        status = "ok"
        if memory_percent > 90 or available_memory_gb < 1:
            status = "error"
        elif memory_percent > 80 or available_memory_gb < 2:
            status = "warning"
        
        return {
            "name": "System Resources",
            "current": f"CPU: {cpu_count} cores, RAM: {available_memory_gb:.1f}GB free, Disk: {available_disk_gb:.1f}GB free",
            "required": f"CPU: 2+ cores, RAM: 4GB+, Disk: 10GB+",
            "status": status,
            "message": f"CPU: {cpu_count} cores, {available_memory_gb:.1f}GB RAM available, {available_disk_gb:.1f}GB disk free"
        }
    
    async def check_ollama_connection(self) -> Dict[str, Any]:
        """Check if Ollama is running and accessible."""
        try:
            async with self.session.get("http://localhost:11434/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    model_count = len(data.get('models', []))
                    return {
                        "name": "Ollama Connection",
                        "current": f"Connected, {model_count} models",
                        "required": "Required",
                        "status": "ok",
                        "message": f"✓ Ollama connected with {model_count} models available"
                    }
                else:
                    return {
                        "name": "Ollama Connection",
                        "current": f"Error: HTTP {response.status}",
                        "required": "Required",
                        "status": "error",
                        "message": f"✗ Ollama connection failed with status {response.status}"
                    }
        except aiohttp.ClientError:
            return {
                "name": "Ollama Connection",
                "current": "Not accessible",
                "required": "Required",
                "status": "error",
                "message": "✗ Ollama not accessible at http://localhost:11434"
            }
        except Exception as e:
            return {
                "name": "Ollama Connection",
                "current": f"Error: {str(e)}",
                "required": "Required",
                "status": "error",
                "message": f"✗ Ollama connection error: {str(e)}"
            }
    
    async def check_comfyui_connection(self) -> Dict[str, Any]:
        """Check if ComfyUI is running and accessible."""
        try:
            async with self.session.get("http://localhost:8188") as response:
                # ComfyUI might return a redirect or a page, just check if it's accessible
                if response.status in [200, 302, 404]:  # 200=OK, 302=redirect, 404=UI exists but not root route
                    return {
                        "name": "ComfyUI Connection",
                        "current": f"Accessible (HTTP {response.status})",
                        "required": "Required",
                        "status": "ok",
                        "message": f"✓ ComfyUI accessible at http://localhost:8188"
                    }
                else:
                    return {
                        "name": "ComfyUI Connection",
                        "current": f"Error: HTTP {response.status}",
                        "required": "Required",
                        "status": "error",
                        "message": f"✗ ComfyUI connection failed with status {response.status}"
                    }
        except aiohttp.ClientError:
            return {
                "name": "ComfyUI Connection",
                "current": "Not accessible",
                "required": "Required",
                "status": "error",
                "message": "✗ ComfyUI not accessible at http://localhost:8188"
            }
        except Exception as e:
            return {
                "name": "ComfyUI Connection",
                "current": f"Error: {str(e)}",
                "required": "Required",
                "status": "error",
                "message": f"✗ ComfyUI connection error: {str(e)}"
            }
    
    async def check_models_available(self) -> Dict[str, Any]:
        """Check if required models are available."""
        try:
            async with self.session.get("http://localhost:11434/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('models', [])
                    model_names = [model.get('name', 'unknown') for model in models]
                    
                    required_models = ['qwen3:8b', 'gemma3:4b']
                    available_models = [m for m in required_models if any(req in m for m in model_names)]
                    
                    status = "ok" if len(available_models) >= 1 else "warning"  # At least one model should be available
                    message = f"✓ Found {len(available_models)}/{len(required_models)} required models: {', '.join(available_models) if available_models else 'None'}"
                    
                    return {
                        "name": "Required Models",
                        "current": f"Available: {len(available_models)}/{len(required_models)}",
                        "required": f"{len(required_models)} models",
                        "status": status,
                        "message": message
                    }
                else:
                    return {
                        "name": "Required Models",
                        "current": "Cannot check - Ollama error",
                        "required": "2+ models",
                        "status": "error",
                        "message": "✗ Cannot check models - Ollama connection failed"
                    }
        except Exception:
            return {
                "name": "Required Models",
                "current": "Cannot check",
                "required": "2+ models",
                "status": "error",
                "message": "✗ Cannot check models - Ollama not accessible"
            }
    
    async def check_dependencies(self) -> Dict[str, Any]:
        """Check if required dependencies are installed."""
        checks = [
            ("Pillow", "from PIL import Image"),
            ("Requests", "import requests"),
            ("AsyncIO", "import asyncio"),
            ("Textual", "from textual.app import App"),
        ]
        
        available = 0
        for name, import_stmt in checks:
            try:
                exec(import_stmt)
                available += 1
            except ImportError:
                pass
        
        status = "ok" if available == len(checks) else "warning"
        
        return {
            "name": "Dependencies",
            "current": f"Available: {available}/{len(checks)}",
            "required": f"{len(checks)} libraries",
            "status": status,
            "message": f"✓ {available}/{len(checks)} required dependencies available"
        }
    
    async def run_command_async(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run a command asynchronously."""
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
    
    async def perform_diagnostics(self) -> List[Dict[str, Any]]:
        """Perform all diagnostics and return results."""
        checks = [
            self.check_python_version(),
            self.check_gpu_availability(), 
            self.check_system_resources(),
            self.check_ollama_connection(),
            self.check_comfyui_connection(),
            self.check_models_available(),
            self.check_dependencies()
        ]
        
        # Run all checks concurrently
        results = await asyncio.gather(*checks, return_exceptions=True)
        
        # Filter out any exceptions and handle them
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # If a check failed with an exception, create a failed result
                check_names = [
                    "Python Version", "GPU Availability", "System Resources",
                    "Ollama Connection", "ComfyUI Connection", "Required Models", "Dependencies"
                ]
                processed_results.append({
                    "name": check_names[i],
                    "current": f"Error: {str(result)}",
                    "required": "Varies",
                    "status": "error",
                    "message": f"✗ Diagnostic check failed: {str(result)}"
                })
            else:
                processed_results.append(result)
        
        return processed_results

async def doctor_command() -> Dict[str, Any]:
    """Performs system diagnostics."""
    print("EDI Doctor: Performing system diagnostics...")
    print("=" * 50)
    
    async with EDIDoctor() as doctor:
        results = await doctor.perform_diagnostics()
        
        # Print results with appropriate styling
        statuses = {"ok": 0, "warning": 0, "error": 0}
        for result in results:
            statuses[result["status"]] += 1
            
            # Print with colored output (use terminal codes)
            if result["status"] == "ok":
                status_icon = "✓"  # Green check
            elif result["status"] == "warning":
                status_icon = "⚠"  # Yellow warning
            else:
                status_icon = "✗"  # Red error
                
            print(f"{status_icon} {result['name']}: {result['message']}")
        
        print("=" * 50)
        print(f"Summary: {statuses['ok']} OK, {statuses['warning']} Warnings, {statuses['error']} Errors")
        
        # Overall status
        overall_status = "error" if statuses["error"] > 0 else "warning" if statuses["warning"] > 0 else "ok"
        
        if overall_status == "ok":
            print("✓ All systems operational!")
        elif overall_status == "warning":
            print("⚠ System has warnings but should function")
        else:
            print("✗ System has errors that need to be addressed")
        
        return {
            "status": overall_status,
            "checks": results,
            "summary": statuses
        }

# Example usage
if __name__ == "__main__":
    # Run the doctor command
    result = asyncio.run(doctor_command())
    print(f"\nDoctor command completed with status: {result['status']}")
```

### System Diagnostic Utilities and Hardware Detection Implementation Example

System diagnostic utilities for the EDI application:

```python
import asyncio
import platform
import subprocess
import psutil
import GPUtil  # Optional: for GPU detection
import os
from typing import Dict, Any, List
from pathlib import Path
import json

class SystemDiagnosticUtils:
    """Utilities for system diagnostics and hardware detection."""
    
    @staticmethod
    async def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information."""
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_compiler": platform.python_compiler(),
        }
    
    @staticmethod
    async def get_memory_info() -> Dict[str, Any]:
        """Get memory information."""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            "total_memory_gb": memory.total / (1024**3),
            "available_memory_gb": memory.available / (1024**3),
            "used_memory_gb": memory.used / (1024**3),
            "memory_percent": memory.percent,
            "total_swap_gb": swap.total / (1024**3),
            "available_swap_gb": swap.free / (1024**3),
            "swap_percent": swap.percent,
        }
    
    @staticmethod
    async def get_cpu_info() -> Dict[str, Any]:
        """Get CPU information."""
        cpu_freq = psutil.cpu_freq()
        
        return {
            "physical_cores": psutil.cpu_count(logical=False),
            "total_cores": psutil.cpu_count(logical=True),
            "max_frequency_ghz": cpu_freq.max / 1000 if cpu_freq else 0,
            "min_frequency_ghz": cpu_freq.min / 1000 if cpu_freq else 0,
            "current_frequency_ghz": cpu_freq.current / 1000 if cpu_freq else 0,
            "cpu_percent": psutil.cpu_percent(interval=1),
            "cpu_percent_per_core": psutil.cpu_percent(interval=1, percpu=True),
        }
    
    @staticmethod
    async def get_disk_info() -> Dict[str, Any]:
        """Get disk information."""
        disk_usage = psutil.disk_usage('/')
        
        return {
            "total_disk_gb": disk_usage.total / (1024**3),
            "used_disk_gb": disk_usage.used / (1024**3),
            "free_disk_gb": disk_usage.free / (1024**3),
            "disk_percent": disk_usage.percent,
        }
    
    @staticmethod
    async def get_gpu_info() -> List[Dict[str, Any]]:
        """Get GPU information using GPUtil."""
        try:
            gpus = GPUtil.getGPUs()
            gpu_list = []
            
            for gpu in gpus:
                gpu_list.append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "load_percent": gpu.load * 100,
                    "memory_total_mb": gpu.memoryTotal,
                    "memory_used_mb": gpu.memoryUsed,
                    "memory_free_mb": gpu.memoryFree,
                    "memory_percent": gpu.memoryUtil * 100,
                    "driver": gpu.driver,
                    "temperature_celsius": gpu.temperature
                })
            
            return gpu_list
        except ImportError:
            return [{"error": "GPUtil not installed", "message": "GPU detection requires GPUtil library"}]
        except Exception as e:
            return [{"error": f"GPU detection failed: {str(e)}"}]
    
    @staticmethod
    async def get_network_info() -> Dict[str, Any]:
        """Get network information."""
        net_io = psutil.net_io_counters()
        addresses = psutil.net_if_addrs()
        
        return {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
            "network_interfaces": list(addresses.keys()),
        }
    
    @staticmethod
    async def check_ports(port_list: List[int]) -> Dict[int, bool]:
        """Check if specific ports are available."""
        import socket
            
        results = {}
        for port in port_list:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('localhost', port))
                    results[port] = True  # Port is available
                except OSError:
                    results[port] = False  # Port is in use
        
        return results
    
    @staticmethod
    async def check_process_running(process_name: str) -> bool:
        """Check if a process is running."""
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if process_name.lower() in proc.info['name'].lower():
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return False
    
    @staticmethod
    async def get_environment_info() -> Dict[str, str]:
        """Get environment variables relevant to EDI."""
        env_vars = [
            "OLLAMA_HOST", "OLLAMA_PORT", "EDI_HOME", 
            "PATH", "PYTHONPATH", "CUDA_VISIBLE_DEVICES"
        ]
        
        result = {}
        for var in env_vars:
            result[var] = os.environ.get(var, "Not set")
        
        return result

class HardwareDetector:
    """Detects hardware capabilities for EDI."""
    
    @staticmethod
    async def detect_cpu_capabilities() -> Dict[str, Any]:
        """Detect CPU-specific capabilities."""
        # Check for specific CPU features
        cpu_info = await SystemDiagnosticUtils.get_cpu_info()
        
        capabilities = {
            "has_avx": False,  # This would require platform-specific detection
            "has_avx2": False,
            "has_avx512": False,
            "architecture": cpu_info.get("machine", "unknown"),
            "is_64bit": sys.maxsize > 2**32,
        }
        
        # Check architecture for optimization recommendations
        if "arm" in capabilities["architecture"].lower():
            capabilities["architecture_specific"] = "ARM processor detected"
        elif "x86" in capabilities["architecture"].lower() or "amd64" in capabilities["architecture"].lower():
            capabilities["architecture_specific"] = "x86/x64 processor detected"
        
        return capabilities
    
    @staticmethod
    async def detect_gpu_capabilities() -> Dict[str, Any]:
        """Detect GPU-specific capabilities."""
        gpus = await SystemDiagnosticUtils.get_gpu_info()
        
        if not gpus or "error" in gpus[0]:
            return {
                "available": False,
                "message": gpus[0].get("message", "No GPU detected or error occurred") if gpus else "No GPUs found",
                "recommended_setup": "Using CPU for processing"
            }
        
        # For this example, we'll just return info about the first GPU
        primary_gpu = gpus[0]
        
        # Determine if the GPU is suitable for ML workloads
        memory_gb = primary_gpu.get("memory_total_mb", 0) / 1024
        
        return {
            "available": True,
            "name": primary_gpu.get("name", "Unknown"),
            "memory_gb": memory_gb,
            "suitable_for_ml": memory_gb >= 4,  # At least 4GB recommended for ML
            "recommended_settings": f"Use GPU with {int(memory_gb)}GB VRAM for optimal performance"
        }
    
    @staticmethod
    async def detect_system_recommendations() -> Dict[str, str]:
        """Provide system recommendations based on detected hardware."""
        cpu_info = await SystemDiagnosticUtils.get_cpu_info()
        memory_info = await SystemDiagnosticUtils.get_memory_info()
        
        recommendations = {}
        
        # CPU recommendations
        if cpu_info.get("total_cores", 0) < 4:
            recommendations["cpu"] = "Consider using a system with more CPU cores for faster processing"
        else:
            recommendations["cpu"] = "Sufficient CPU cores for processing"
        
        # Memory recommendations
        if memory_info.get("total_memory_gb", 0) < 8:
            recommendations["memory"] = "Consider using a system with more RAM (8GB+ recommended)"
        elif memory_info.get("available_memory_gb", 0) < 2:
            recommendations["memory"] = "Low available memory - close other applications"
        else:
            recommendations["memory"] = "Sufficient system memory"
        
        # GPU recommendations
        gpu_info = await HardwareDetector.detect_gpu_capabilities()
        if gpu_info.get("available") and gpu_info.get("suitable_for_ml"):
            recommendations["gpu"] = f"GPU acceleration available: {gpu_info['recommended_settings']}"
        elif gpu_info.get("available"):
            recommendations["gpu"] = f"GPU available but may have limited memory: {gpu_info.get('name', 'Unknown')}"
        else:
            recommendations["gpu"] = "No dedicated GPU detected - using CPU for processing"
        
        return recommendations

class DiagnosticRunner:
    """Runs comprehensive diagnostics for EDI."""
    
    def __init__(self):
        self.results = {}
    
    async def run_all_diagnostics(self) -> Dict[str, Any]:
        """Run all diagnostics and return comprehensive report."""
        print("Running comprehensive diagnostics...")
        
        # Gather all system information concurrently
        tasks = [
            SystemDiagnosticUtils.get_system_info(),
            SystemDiagnosticUtils.get_memory_info(),
            SystemDiagnosticUtils.get_cpu_info(),
            SystemDiagnosticUtils.get_disk_info(),
            SystemDiagnosticUtils.get_gpu_info(),
            SystemDiagnosticUtils.get_network_info(),
            SystemDiagnosticUtils.get_environment_info(),
            HardwareDetector.detect_cpu_capabilities(),
            HardwareDetector.detect_gpu_capabilities(),
            HardwareDetector.detect_system_recommendations()
        ]
        
        results = await asyncio.gather(*tasks)
        
        report = {
            "timestamp": str(asyncio.get_event_loop().time()),
            "system_info": results[0],
            "memory_info": results[1],
            "cpu_info": results[2],
            "disk_info": results[3],
            "gpu_info": results[4],
            "network_info": results[5],
            "environment_info": results[6],
            "cpu_capabilities": results[7],
            "gpu_capabilities": results[8],
            "recommendations": results[9]
        }
        
        return report
    
    async def save_diagnostics_report(self, report: Dict[str, Any], filename: str = "edi_diagnostics.json"):
        """Save diagnostics report to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Diagnostics report saved to: {filename}")
        return filename

# Example usage
async def main():
    # Example of using system diagnostic utilities
    print("System Diagnostic Utilities Example:")
    
    # Get basic system info
    sys_info = await SystemDiagnosticUtils.get_system_info()
    print(f"Platform: {sys_info['platform']}")
    print(f"Python version: {sys_info['python_version']}")
    
    # Check memory
    mem_info = await SystemDiagnosticUtils.get_memory_info()
    print(f"Memory: {mem_info['available_memory_gb']:.1f}GB available")
    
    # Check CPU
    cpu_info = await SystemDiagnosticUtils.get_cpu_info()
    print(f"CPU: {cpu_info['total_cores']} cores")
    
    # Check disk
    disk_info = await SystemDiagnosticUtils.get_disk_info()
    print(f"Disk: {disk_info['free_disk_gb']:.1f}GB free")
    
    # Check if Ollama is running (port 11434)
    port_status = await SystemDiagnosticUtils.check_ports([11434, 8188])
    print(f"Ollama port (11434) available: {port_status[11434]}")
    print(f"ComfyUI port (8188) available: {port_status[8188]}")
    
    # Run comprehensive diagnostics
    print("\nRunning comprehensive diagnostics:")
    runner = DiagnosticRunner()
    report = await runner.run_all_diagnostics()
    
    # Print some recommendations
    if "recommendations" in report:
        print("\nSystem Recommendations:")
        for category, rec in report["recommendations"].items():
            print(f"  {category.title()}: {rec}")
    
    # Save the report
    await runner.save_diagnostics_report(report)

if __name__ == "__main__":
    asyncio.run(main())
```
