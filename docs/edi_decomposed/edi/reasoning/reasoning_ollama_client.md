# Reasoning: Ollama Client

[Back to Reasoning Subsystem](./reasoning_subsystem.md)

## Purpose
Ollama API wrapper - Contains the OllamaClient class that handles communication with Ollama for LLM inference.

## Class: OllamaClient

### Methods
- `generate(prompt, model) -> str`: Sends a prompt to the specified model and returns the generated text

### Details
- Handles connection errors and retries
- Manages communication with Ollama server
- Provides a clean interface to Ollama's API

## Functions

- [generate(prompt, model)](./reasoning/generate.md)

## Technology Stack

- Requests for HTTP communication
- Ollama for LLM inference

## See Docs

### Python Implementation Example
Reasoning Ollama client implementation:

```python
import requests
import json
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import logging
from urllib.parse import urljoin

class OllamaClientError(Exception):
    """Custom exception for Ollama client errors."""
    pass

@dataclass
class OllamaConfig:
    """
    Configuration for Ollama client.
    """
    base_url: str = "http://localhost:11434"
    timeout: int = 300
    max_retries: int = 3
    retry_delay: float = 1.0
    verify_ssl: bool = True

@dataclass
class GenerationOptions:
    """
    Options for text generation.
    """
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    num_predict: int = 512
    stop: Optional[List[str]] = None
    num_ctx: int = 4096
    num_keep: Optional[int] = None
    seed: Optional[int] = None
    num_batch: Optional[int] = None
    num_gpu: Optional[int] = None
    main_gpu: Optional[int] = None
    low_vram: bool = False
    f16_kv: bool = True
    logits_all: bool = False
    vocab_only: bool = False
    use_mmap: bool = True
    use_mlock: bool = False
    embedding_only: bool = False
    rope_frequency_base: Optional[float] = None
    rope_frequency_scale: Optional[float] = None
    typical_p: Optional[float] = None
    repeat_last_n: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    mirostat: Optional[int] = None
    mirostat_tau: Optional[float] = None
    mirostat_eta: Optional[float] = None
    penalize_newline: bool = True
    perpend_images: Optional[str] = None
    images: Optional[List[str]] = None

class OllamaClient:
    """
    Ollama API wrapper that handles communication with Ollama for LLM inference.
    """
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.timeout = self.config.timeout
        self.session.verify = self.config.verify_ssl
    
    def generate(self, 
                  prompt: str, 
                  model: str,
                  options: Optional[GenerationOptions] = None,
                  system_prompt: Optional[str] = None,
                  template: Optional[str] = None,
                  context: Optional[List[int]] = None,
                  stream: bool = False,
                  raw: bool = False,
                  format: Optional[str] = None,
                  keep_alive: Optional[str] = None) -> Union[str, Dict[str, Any]]:
        """
        Sends a prompt to the specified model and returns the generated text.
        
        Args:
            prompt: Text prompt to send to the model
            model: Name of the model to use for generation
            options: Generation options (temperature, top_p, etc.)
            system_prompt: System prompt to prepend to the conversation
            template: Template to use for generation
            context: Context from previous conversation
            stream: Whether to stream the response
            raw: Whether to return raw response
            format: Format of the response (e.g., "json")
            keep_alive: Keep model alive for specified duration
            
        Returns:
            Generated text or dictionary with full response if raw=True
        """
        # Prepare API request with prompt and model parameters
        url = urljoin(self.config.base_url, "/api/generate")
        
        # Build request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        # Add optional parameters
        if options:
            payload["options"] = self._options_to_dict(options)
        
        if system_prompt:
            payload["system"] = system_prompt
        
        if template:
            payload["template"] = template
        
        if context:
            payload["context"] = context
        
        if format:
            payload["format"] = format
        
        if keep_alive:
            payload["keep_alive"] = keep_alive
        
        # Handle connection errors and retries gracefully
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Send request to Ollama server
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=self.config.timeout
                )
                
                # Handle HTTP errors
                if response.status_code != 200:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", f"HTTP {response.status_code}")
                    except json.JSONDecodeError:
                        error_msg = f"HTTP {response.status_code}: {response.text}"
                    
                    raise OllamaClientError(f"Ollama API error: {error_msg}")
                
                # Process streaming responses if applicable
                if stream:
                    return self._process_streaming_response(response)
                
                # Parse response
                try:
                    result = response.json()
                except json.JSONDecodeError:
                    raise OllamaClientError("Invalid JSON response from Ollama API")
                
                # Validate response structure
                if "response" not in result:
                    raise OllamaClientError(f"Unexpected response format: {result}")
                
                # Return generated text
                if raw:
                    return result
                else:
                    return result["response"]
                
            except requests.exceptions.ConnectionError as e:
                last_error = OllamaClientError(f"Connection error: {str(e)}")
                self.logger.warning(f"Connection error (attempt {attempt + 1}): {str(e)}")
                
            except requests.exceptions.Timeout as e:
                last_error = OllamaClientError(f"Timeout error: {str(e)}")
                self.logger.warning(f"Timeout error (attempt {attempt + 1}): {str(e)}")
                
            except requests.exceptions.RequestException as e:
                last_error = OllamaClientError(f"Request error: {str(e)}")
                self.logger.warning(f"Request error (attempt {attempt + 1}): {str(e)}")
                
            except Exception as e:
                last_error = OllamaClientError(f"Unexpected error: {str(e)}")
                self.logger.warning(f"Unexpected error (attempt {attempt + 1}): {str(e)}")
            
            # Retry if not the last attempt
            if attempt < self.config.max_retries:
                time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
        
        # If we get here, all retries failed
        raise last_error
    
    def _options_to_dict(self, options: GenerationOptions) -> Dict[str, Any]:
        """
        Convert GenerationOptions to dictionary.
        
        Args:
            options: GenerationOptions object
            
        Returns:
            Dictionary of options
        """
        options_dict = {}
        
        # Get all attributes that don't start with underscore
        for attr_name in dir(options):
            if not attr_name.startswith('_') and not callable(getattr(options, attr_name)):
                attr_value = getattr(options, attr_name)
                if attr_value is not None:
                    options_dict[attr_name] = attr_value
        
        return options_dict
    
    def _process_streaming_response(self, response: requests.Response) -> str:
        """
        Process streaming response from Ollama.
        
        Args:
            response: HTTP response object
            
        Returns:
            Complete generated text
        """
        full_response = ""
        
        try:
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            full_response += chunk["response"]
                        
                        # Handle completion
                        if chunk.get("done", False):
                            break
                            
                    except json.JSONDecodeError:
                        self.logger.warning(f"Invalid JSON in streaming response: {line}")
                        
        except Exception as e:
            self.logger.error(f"Error processing streaming response: {str(e)}")
            raise OllamaClientError(f"Streaming error: {str(e)}")
        
        return full_response
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             model: str,
             options: Optional[GenerationOptions] = None,
             stream: bool = False,
             format: Optional[str] = None,
             keep_alive: Optional[str] = None) -> Union[str, Dict[str, Any]]:
        """
        Chat with the model using a conversation history.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Name of the model to use for chat
            options: Generation options
            stream: Whether to stream the response
            format: Format of the response
            keep_alive: Keep model alive for specified duration
            
        Returns:
            Generated response or dictionary with full response if raw=True
        """
        # Prepare chat API request
        url = urljoin(self.config.base_url, "/api/chat")
        
        # Build request payload
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        # Add optional parameters
        if options:
            payload["options"] = self._options_to_dict(options)
        
        if format:
            payload["format"] = format
        
        if keep_alive:
            payload["keep_alive"] = keep_alive
        
        # Handle connection errors and retries gracefully
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Send request to Ollama server
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=self.config.timeout
                )
                
                # Handle HTTP errors
                if response.status_code != 200:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", f"HTTP {response.status_code}")
                    except json.JSONDecodeError:
                        error_msg = f"HTTP {response.status_code}: {response.text}"
                    
                    raise OllamaClientError(f"Ollama API error: {error_msg}")
                
                # Process streaming responses if applicable
                if stream:
                    return self._process_chat_streaming_response(response)
                
                # Parse response
                try:
                    result = response.json()
                except json.JSONDecodeError:
                    raise OllamaClientError("Invalid JSON response from Ollama API")
                
                # Validate response structure
                if "message" not in result:
                    raise OllamaClientError(f"Unexpected response format: {result}")
                
                # Return generated text
                message = result["message"]
                if isinstance(message, dict) and "content" in message:
                    return message["content"]
                else:
                    return str(message)
                
            except requests.exceptions.ConnectionError as e:
                last_error = OllamaClientError(f"Connection error: {str(e)}")
                self.logger.warning(f"Connection error (attempt {attempt + 1}): {str(e)}")
                
            except requests.exceptions.Timeout as e:
                last_error = OllamaClientError(f"Timeout error: {str(e)}")
                self.logger.warning(f"Timeout error (attempt {attempt + 1}): {str(e)}")
                
            except requests.exceptions.RequestException as e:
                last_error = OllamaClientError(f"Request error: {str(e)}")
                self.logger.warning(f"Request error (attempt {attempt + 1}): {str(e)}")
                
            except Exception as e:
                last_error = OllamaClientError(f"Unexpected error: {str(e)}")
                self.logger.warning(f"Unexpected error (attempt {attempt + 1}): {str(e)}")
            
            # Retry if not the last attempt
            if attempt < self.config.max_retries:
                time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
        
        # If we get here, all retries failed
        raise last_error
    
    def _process_chat_streaming_response(self, response: requests.Response) -> str:
        """
        Process streaming chat response from Ollama.
        
        Args:
            response: HTTP response object
            
        Returns:
            Complete generated text
        """
        full_response = ""
        
        try:
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if "message" in chunk and isinstance(chunk["message"], dict):
                            message = chunk["message"]
                            if "content" in message:
                                full_response += message["content"]
                        
                        # Handle completion
                        if chunk.get("done", False):
                            break
                            
                    except json.JSONDecodeError:
                        self.logger.warning(f"Invalid JSON in chat streaming response: {line}")
                        
        except Exception as e:
            self.logger.error(f"Error processing chat streaming response: {str(e)}")
            raise OllamaClientError(f"Chat streaming error: {str(e)}")
        
        return full_response
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models on the Ollama server.
        
        Returns:
            List of model information dictionaries
        """
        try:
            url = urljoin(self.config.base_url, "/api/tags")
            response = self.session.get(url, timeout=self.config.timeout)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("models", [])
            else:
                raise OllamaClientError(f"Failed to list models: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            raise OllamaClientError(f"Request error listing models: {str(e)}")
        except Exception as e:
            raise OllamaClientError(f"Unexpected error listing models: {str(e)}")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model to get information for
            
        Returns:
            Dictionary with model information
        """
        try:
            url = urljoin(self.config.base_url, "/api/show")
            payload = {"name": model_name}
            response = self.session.post(url, json=payload, timeout=self.config.timeout)
            
            if response.status_code == 200:
                return response.json()
            else:
                raise OllamaClientError(f"Failed to get model info: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            raise OllamaClientError(f"Request error getting model info: {str(e)}")
        except Exception as e:
            raise OllamaClientError(f"Unexpected error getting model info: {str(e)}")
    
    def pull_model(self, model_name: str, insecure: bool = False) -> bool:
        """
        Pull a model from the Ollama library.
        
        Args:
            model_name: Name of the model to pull
            insecure: Whether to skip SSL verification
            
        Returns:
            Boolean indicating success
        """
        try:
            url = urljoin(self.config.base_url, "/api/pull")
            payload = {
                "name": model_name,
                "insecure": insecure
            }
            response = self.session.post(url, json=payload, timeout=self.config.timeout)
            
            return response.status_code == 200
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error pulling model {model_name}: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error pulling model {model_name}: {str(e)}")
            return False
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model from the Ollama server.
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            Boolean indicating success
        """
        try:
            url = urljoin(self.config.base_url, "/api/delete")
            payload = {"name": model_name}
            response = self.session.delete(url, json=payload, timeout=self.config.timeout)
            
            return response.status_code == 200
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error deleting model {model_name}: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error deleting model {model_name}: {str(e)}")
            return False
    
    def generate_embeddings(self, 
                            input_text: str, 
                            model: str = "nomic-embed-text") -> List[float]:
        """
        Generate embeddings for input text.
        
        Args:
            input_text: Text to generate embeddings for
            model: Model to use for embedding generation
            
        Returns:
            List of embedding values
        """
        try:
            url = urljoin(self.config.base_url, "/api/embeddings")
            payload = {
                "model": model,
                "prompt": input_text
            }
            response = self.session.post(url, json=payload, timeout=self.config.timeout)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("embedding", [])
            else:
                raise OllamaClientError(f"Failed to generate embeddings: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            raise OllamaClientError(f"Request error generating embeddings: {str(e)}")
        except Exception as e:
            raise OllamaClientError(f"Unexpected error generating embeddings: {str(e)}")
    
    def is_model_available(self, model_name: str) -> bool:
        """
        Check if a model is available on the Ollama server.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            Boolean indicating if model is available
        """
        try:
            models = self.list_models()
            return any(model.get("name") == model_name for model in models)
        except:
            return False
    
    def get_server_status(self) -> Dict[str, Any]:
        """
        Get the status of the Ollama server.
        
        Returns:
            Dictionary with server status information
        """
        try:
            url = urljoin(self.config.base_url, "/api/version")
            response = self.session.get(url, timeout=5)  # Short timeout for status check
            
            if response.status_code == 200:
                version_info = response.json()
                return {
                    "status": "online",
                    "version": version_info.get("version", "unknown"),
                    "base_url": self.config.base_url
                }
            else:
                return {
                    "status": "offline",
                    "base_url": self.config.base_url,
                    "error": f"HTTP {response.status_code}"
                }
                
        except requests.exceptions.RequestException:
            return {
                "status": "offline",
                "base_url": self.config.base_url,
                "error": "Connection failed"
            }
        except Exception as e:
            return {
                "status": "offline",
                "base_url": self.config.base_url,
                "error": str(e)
            }

# Example usage
if __name__ == "__main__":
    # Initialize Ollama client
    config = OllamaConfig(
        base_url="http://localhost:11434",
        timeout=300,
        max_retries=3,
        retry_delay=1.0
    )
    client = OllamaClient(config)
    
    print("Ollama Client initialized")
    
    # Check server status
    status = client.get_server_status()
    print(f"Server status: {status['status']}")
    if status['status'] == 'online':
        print(f"  Version: {status['version']}")
        print(f"  Base URL: {status['base_url']}")
    
    # List available models
    try:
        models = client.list_models()
        print(f"Available models: {len(models)}")
        for model in models[:5]:  # Show first 5 models
            print(f"  - {model.get('name', 'unknown')}")
    except OllamaClientError as e:
        print(f"Failed to list models: {e}")
    
    # Check if a specific model is available
    model_name = "qwen3:8b"
    if client.is_model_available(model_name):
        print(f"Model {model_name} is available")
    else:
        print(f"Model {model_name} is not available")
    
    # Example generation options
    options = GenerationOptions(
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        num_predict=512,
        num_ctx=4096
    )
    
    # Example prompt
    prompt = "Explain how to implement a simple image editing system using AI."
    
    try:
        # Generate text with retries and error handling
        print(f"Generating response for prompt: {prompt[:50]}...")
        response = client.generate(
            prompt=prompt,
            model=model_name,
            options=options,
            stream=False
        )
        print(f"Generated response: {response[:200]}...")
        
        # Example chat conversation
        messages = [
            {"role": "user", "content": "What are the benefits of using AI for image editing?"},
            {"role": "assistant", "content": "AI image editing offers several benefits including automation, consistency, and creative enhancement capabilities."},
            {"role": "user", "content": "Can you elaborate on the creative enhancement capabilities?"}
        ]
        
        print("Chatting with model...")
        chat_response = client.chat(
            messages=messages,
            model=model_name,
            options=options,
            stream=False
        )
        print(f"Chat response: {chat_response[:200]}...")
        
        # Generate embeddings
        embedding_text = "Artificial intelligence is revolutionizing image processing."
        embeddings = client.generate_embeddings(embedding_text)
        print(f"Generated embeddings: {len(embeddings)} dimensions")
        print(f"  First 5 values: {embeddings[:5]}")
        
    except OllamaClientError as e:
        print(f"Ollama client error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    # Get model information
    try:
        if client.is_model_available(model_name):
            model_info = client.get_model_info(model_name)
            print(f"Model info for {model_name}:")
            print(f"  Family: {model_info.get('family', 'unknown')}")
            print(f"  Format: {model_info.get('format', 'unknown')}")
            print(f"  Size: {model_info.get('size', 0) / (1024**3):.2f} GB")
    except OllamaClientError as e:
        print(f"Failed to get model info: {e}")
    
    print("Ollama client example completed")
```

### Advanced Ollama Client Implementation
Enhanced implementation with async support and advanced features:

```python
import requests
import json
import time
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List, Union, AsyncGenerator
from dataclasses import dataclass
import logging
from urllib.parse import urljoin
import base64
from pathlib import Path

class AdvancedOllamaClientError(Exception):
    """Custom exception for advanced Ollama client errors."""
    pass

@dataclass
class AdvancedGenerationOptions:
    """
    Advanced options for text generation.
    """
    # Basic generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    num_predict: int = 512
    stop: Optional[List[str]] = None
    num_ctx: int = 4096
    
    # Advanced parameters
    seed: Optional[int] = None
    num_batch: Optional[int] = None
    num_gpu: Optional[int] = None
    main_gpu: Optional[int] = None
    low_vram: bool = False
    f16_kv: bool = True
    logits_all: bool = False
    vocab_only: bool = False
    use_mmap: bool = True
    use_mlock: bool = False
    embedding_only: bool = False
    rope_frequency_base: Optional[float] = None
    rope_frequency_scale: Optional[float] = None
    typical_p: Optional[float] = None
    repeat_last_n: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    mirostat: Optional[int] = None
    mirostat_tau: Optional[float] = None
    mirostat_eta: Optional[float] = None
    penalize_newline: bool = True
    perpend_images: Optional[str] = None
    images: Optional[List[str]] = None  # Base64 encoded images

class AdvancedOllamaClient:
    """
    Advanced Ollama client with async support and enhanced features.
    """
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 timeout: int = 300,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 use_async: bool = True):
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_async = use_async
        self.logger = logging.getLogger(__name__)
        
        # Initialize session for sync requests
        self.session = requests.Session()
        self.session.timeout = self.timeout
        
        # Initialize async session if needed
        self.async_session = None
        if use_async:
            self._initialize_async_session()
    
    def _initialize_async_session(self):
        """Initialize async HTTP session."""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.async_session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        if not self.async_session:
            self._initialize_async_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.async_session:
            await self.async_session.close()
    
    def __del__(self):
        """Cleanup resources."""
        if self.async_session:
            try:
                # This is not ideal in __del__, but needed for proper cleanup
                if hasattr(asyncio, 'get_event_loop'):
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Schedule close for later
                            loop.create_task(self.async_session.close())
                        else:
                            # Close immediately
                            asyncio.run(self.async_session.close())
                    except:
                        pass
                else:
                    # Close immediately
                    asyncio.run(self.async_session.close())
            except:
                pass
    
    async def generate_async(self, 
                             prompt: str, 
                             model: str,
                             options: Optional[AdvancedGenerationOptions] = None,
                             system_prompt: Optional[str] = None,
                             template: Optional[str] = None,
                             context: Optional[List[int]] = None,
                             stream: bool = False,
                             raw: bool = False,
                             format: Optional[str] = None,
                             keep_alive: Optional[str] = None,
                             images: Optional[List[str]] = None) -> Union[str, Dict[str, Any], AsyncGenerator[str, None]]:
        """
        Asynchronously sends a prompt to the specified model and returns the generated text.
        
        Args:
            prompt: Text prompt to send to the model
            model: Name of the model to use for generation
            options: Advanced generation options
            system_prompt: System prompt to prepend to the conversation
            template: Template to use for generation
            context: Context from previous conversation
            stream: Whether to stream the response
            raw: Whether to return raw response
            format: Format of the response (e.g., "json")
            keep_alive: Keep model alive for specified duration
            images: Base64 encoded images to include in prompt
            
        Returns:
            Generated text, dictionary with full response if raw=True, or async generator if streaming
        """
        if not self.async_session:
            raise AdvancedOllamaClientError("Async session not initialized")
        
        # Prepare API request with prompt and model parameters
        url = urljoin(self.base_url, "/api/generate")
        
        # Build request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        # Add optional parameters
        if options:
            payload["options"] = self._options_to_dict(options)
        
        if system_prompt:
            payload["system"] = system_prompt
        
        if template:
            payload["template"] = template
        
        if context:
            payload["context"] = context
        
        if format:
            payload["format"] = format
        
        if keep_alive:
            payload["keep_alive"] = keep_alive
        
        if images:
            payload["images"] = images
        
        # Handle connection errors and retries gracefully
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Send request to Ollama server
                async with self.async_session.post(url, json=payload) as response:
                    # Handle HTTP errors
                    if response.status != 200:
                        try:
                            error_data = await response.json()
                            error_msg = error_data.get("error", f"HTTP {response.status}")
                        except:
                            error_msg = f"HTTP {response.status}: {response.reason}"
                        
                        raise AdvancedOllamaClientError(f"Ollama API error: {error_msg}")
                    
                    # Process streaming responses if applicable
                    if stream:
                        return self._process_async_streaming_response(response)
                    
                    # Parse response
                    try:
                        result = await response.json()
                    except:
                        raise AdvancedOllamaClientError("Invalid JSON response from Ollama API")
                    
                    # Validate response structure
                    if "response" not in result:
                        raise AdvancedOllamaClientError(f"Unexpected response format: {result}")
                    
                    # Return generated text
                    if raw:
                        return result
                    else:
                        return result["response"]
                
            except aiohttp.ClientError as e:
                last_error = AdvancedOllamaClientError(f"Client error: {str(e)}")
                self.logger.warning(f"Client error (attempt {attempt + 1}): {str(e)}")
                
            except asyncio.TimeoutError as e:
                last_error = AdvancedOllamaClientError(f"Timeout error: {str(e)}")
                self.logger.warning(f"Timeout error (attempt {attempt + 1}): {str(e)}")
                
            except Exception as e:
                last_error = AdvancedOllamaClientError(f"Unexpected error: {str(e)}")
                self.logger.warning(f"Unexpected error (attempt {attempt + 1}): {str(e)}")
            
            # Retry if not the last attempt
            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
        
        # If we get here, all retries failed
        raise last_error
    
    def _options_to_dict(self, options: AdvancedGenerationOptions) -> Dict[str, Any]:
        """
        Convert AdvancedGenerationOptions to dictionary.
        
        Args:
            options: AdvancedGenerationOptions object
            
        Returns:
            Dictionary of options
        """
        options_dict = {}
        
        # Get all attributes that don't start with underscore
        for attr_name in dir(options):
            if not attr_name.startswith('_') and not callable(getattr(options, attr_name)):
                attr_value = getattr(options, attr_name)
                if attr_value is not None:
                    options_dict[attr_name] = attr_value
        
        return options_dict
    
    async def _process_async_streaming_response(self, response: aiohttp.ClientResponse) -> AsyncGenerator[str, None]:
        """
        Process async streaming response from Ollama.
        
        Args:
            response: HTTP response object
            
        Yields:
            Chunks of generated text
        """
        try:
            async for line in response.content:
                if line:
                    try:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            yield chunk["response"]
                        
                        # Handle completion
                        if chunk.get("done", False):
                            break
                            
                    except json.JSONDecodeError:
                        self.logger.warning(f"Invalid JSON in streaming response: {line}")
                        
        except Exception as e:
            self.logger.error(f"Error processing async streaming response: {str(e)}")
            raise AdvancedOllamaClientError(f"Async streaming error: {str(e)}")
    
    async def chat_async(self, 
                         messages: List[Dict[str, str]], 
                         model: str,
                         options: Optional[AdvancedGenerationOptions] = None,
                         stream: bool = False,
                         format: Optional[str] = None,
                         keep_alive: Optional[str] = None) -> Union[str, Dict[str, Any], AsyncGenerator[str, None]]:
        """
        Asynchronously chat with the model using a conversation history.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Name of the model to use for chat
            options: Generation options
            stream: Whether to stream the response
            format: Format of the response
            keep_alive: Keep model alive for specified duration
            
        Returns:
            Generated response, dictionary with full response if raw=True, or async generator if streaming
        """
        if not self.async_session:
            raise AdvancedOllamaClientError("Async session not initialized")
        
        # Prepare chat API request
        url = urljoin(self.base_url, "/api/chat")
        
        # Build request payload
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        # Add optional parameters
        if options:
            payload["options"] = self._options_to_dict(options)
        
        if format:
            payload["format"] = format
        
        if keep_alive:
            payload["keep_alive"] = keep_alive
        
        # Handle connection errors and retries gracefully
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Send request to Ollama server
                async with self.async_session.post(url, json=payload) as response:
                    # Handle HTTP errors
                    if response.status != 200:
                        try:
                            error_data = await response.json()
                            error_msg = error_data.get("error", f"HTTP {response.status}")
                        except:
                            error_msg = f"HTTP {response.status}: {response.reason}"
                        
                        raise AdvancedOllamaClientError(f"Ollama API error: {error_msg}")
                    
                    # Process streaming responses if applicable
                    if stream:
                        return self._process_async_chat_streaming_response(response)
                    
                    # Parse response
                    try:
                        result = await response.json()
                    except:
                        raise AdvancedOllamaClientError("Invalid JSON response from Ollama API")
                    
                    # Validate response structure
                    if "message" not in result:
                        raise AdvancedOllamaClientError(f"Unexpected response format: {result}")
                    
                    # Return generated text
                    message = result["message"]
                    if isinstance(message, dict) and "content" in message:
                        if raw:
                            return result
                        else:
                            return message["content"]
                    else:
                        return str(message)
                
            except aiohttp.ClientError as e:
                last_error = AdvancedOllamaClientError(f"Client error: {str(e)}")
                self.logger.warning(f"Client error (attempt {attempt + 1}): {str(e)}")
                
            except asyncio.TimeoutError as e:
                last_error = AdvancedOllamaClientError(f"Timeout error: {str(e)}")
                self.logger.warning(f"Timeout error (attempt {attempt + 1}): {str(e)}")
                
            except Exception as e:
                last_error = AdvancedOllamaClientError(f"Unexpected error: {str(e)}")
                self.logger.warning(f"Unexpected error (attempt {attempt + 1}): {str(e)}")
            
            # Retry if not the last attempt
            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
        
        # If we get here, all retries failed
        raise last_error
    
    async def _process_async_chat_streaming_response(self, response: aiohttp.ClientResponse) -> AsyncGenerator[str, None]:
        """
        Process async streaming chat response from Ollama.
        
        Args:
            response: HTTP response object
            
        Yields:
            Chunks of generated text
        """
        try:
            async for line in response.content:
                if line:
                    try:
                        chunk = json.loads(line)
                        if "message" in chunk and isinstance(chunk["message"], dict):
                            message = chunk["message"]
                            if "content" in message:
                                yield message["content"]
                        
                        # Handle completion
                        if chunk.get("done", False):
                            break
                            
                    except json.JSONDecodeError:
                        self.logger.warning(f"Invalid JSON in chat streaming response: {line}")
                        
        except Exception as e:
            self.logger.error(f"Error processing async chat streaming response: {str(e)}")
            raise AdvancedOllamaClientError(f"Async chat streaming error: {str(e)}")
    
    async def list_models_async(self) -> List[Dict[str, Any]]:
        """
        Asynchronously list available models on the Ollama server.
        
        Returns:
            List of model information dictionaries
        """
        if not self.async_session:
            raise AdvancedOllamaClientError("Async session not initialized")
        
        try:
            url = urljoin(self.base_url, "/api/tags")
            async with self.async_session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("models", [])
                else:
                    raise AdvancedOllamaClientError(f"Failed to list models: HTTP {response.status}")
                    
        except aiohttp.ClientError as e:
            raise AdvancedOllamaClientError(f"Client error listing models: {str(e)}")
        except Exception as e:
            raise AdvancedOllamaClientError(f"Unexpected error listing models: {str(e)}")
    
    async def generate_embeddings_async(self, 
                                       input_text: str, 
                                       model: str = "nomic-embed-text") -> List[float]:
        """
        Asynchronously generate embeddings for input text.
        
        Args:
            input_text: Text to generate embeddings for
            model: Model to use for embedding generation
            
        Returns:
            List of embedding values
        """
        if not self.async_session:
            raise AdvancedOllamaClientError("Async session not initialized")
        
        try:
            url = urljoin(self.base_url, "/api/embeddings")
            payload = {
                "model": model,
                "prompt": input_text
            }
            
            async with self.async_session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("embedding", [])
                else:
                    raise AdvancedOllamaClientError(f"Failed to generate embeddings: HTTP {response.status}")
                    
        except aiohttp.ClientError as e:
            raise AdvancedOllamaClientError(f"Client error generating embeddings: {str(e)}")
        except Exception as e:
            raise AdvancedOllamaClientError(f"Unexpected error generating embeddings: {str(e)}")
    
    async def batch_generate_async(self, 
                                   prompts: List[str], 
                                   model: str,
                                   options: Optional[AdvancedGenerationOptions] = None) -> List[str]:
        """
        Asynchronously generate text for multiple prompts in batch.
        
        Args:
            prompts: List of prompts to generate text for
            model: Name of the model to use for generation
            options: Generation options
            
        Returns:
            List of generated texts
        """
        # Create tasks for all prompts
        tasks = [
            self.generate_async(
                prompt=prompt,
                model=model,
                options=options,
                stream=False
            )
            for prompt in prompts
        ]
        
        # Execute all tasks concurrently
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to generate response for prompt {i}: {str(result)}")
                    processed_results.append("")
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Batch generation failed: {str(e)}")
            raise AdvancedOllamaClientError(f"Batch generation failed: {str(e)}")
    
    async def is_model_available_async(self, model_name: str) -> bool:
        """
        Asynchronously check if a model is available on the Ollama server.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            Boolean indicating if model is available
        """
        try:
            models = await self.list_models_async()
            return any(model.get("name") == model_name for model in models)
        except:
            return False
    
    async def get_server_status_async(self) -> Dict[str, Any]:
        """
        Asynchronously get the status of the Ollama server.
        
        Returns:
            Dictionary with server status information
        """
        try:
            url = urljoin(self.base_url, "/api/version")
            async with self.async_session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    version_info = await response.json()
                    return {
                        "status": "online",
                        "version": version_info.get("version", "unknown"),
                        "base_url": self.base_url
                    }
                else:
                    return {
                        "status": "offline",
                        "base_url": self.base_url,
                        "error": f"HTTP {response.status}"
                    }
                    
        except aiohttp.ClientError:
            return {
                "status": "offline",
                "base_url": self.base_url,
                "error": "Connection failed"
            }
        except Exception as e:
            return {
                "status": "offline",
                "base_url": self.base_url,
                "error": str(e)
            }
    
    async def pull_model_async(self, model_name: str, insecure: bool = False) -> bool:
        """
        Asynchronously pull a model from the Ollama library.
        
        Args:
            model_name: Name of the model to pull
            insecure: Whether to skip SSL verification
            
        Returns:
            Boolean indicating success
        """
        if not self.async_session:
            raise AdvancedOllamaClientError("Async session not initialized")
        
        try:
            url = urljoin(self.base_url, "/api/pull")
            payload = {
                "name": model_name,
                "insecure": insecure
            }
            
            async with self.async_session.post(url, json=payload) as response:
                return response.status == 200
                
        except aiohttp.ClientError as e:
            self.logger.error(f"Client error pulling model {model_name}: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error pulling model {model_name}: {str(e)}")
            return False
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode an image file to base64 string.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            self.logger.error(f"Failed to encode image {image_path}: {str(e)}")
            raise AdvancedOllamaClientError(f"Failed to encode image: {str(e)}")
    
    async def generate_with_images_async(self, 
                                         prompt: str,
                                         image_paths: List[str],
                                         model: str,
                                         options: Optional[AdvancedGenerationOptions] = None) -> str:
        """
        Asynchronously generate text with images.
        
        Args:
            prompt: Text prompt to send to the model
            image_paths: List of paths to image files
            model: Name of the model to use for generation
            options: Generation options
            
        Returns:
            Generated text
        """
        # Encode images to base64
        encoded_images = []
        for image_path in image_paths:
            encoded_image = self.encode_image_to_base64(image_path)
            encoded_images.append(encoded_image)
        
        # Add images to options
        if options:
            options.images = encoded_images
        else:
            options = AdvancedGenerationOptions(images=encoded_images)
        
        # Generate with images
        return await self.generate_async(
            prompt=prompt,
            model=model,
            options=options,
            stream=False
        )

# Example usage
if __name__ == "__main__":
    # Initialize advanced Ollama client
    client = AdvancedOllamaClient(
        base_url="http://localhost:11434",
        timeout=300,
        max_retries=3,
        retry_delay=1.0,
        use_async=True
    )
    
    print("Advanced Ollama Client initialized")
    
    async def run_examples():
        # Check server status
        status = await client.get_server_status_async()
        print(f"Server status: {status['status']}")
        if status['status'] == 'online':
            print(f"  Version: {status['version']}")
            print(f"  Base URL: {status['base_url']}")
        
        # List available models
        try:
            models = await client.list_models_async()
            print(f"Available models: {len(models)}")
            for model in models[:5]:  # Show first 5 models
                print(f"  - {model.get('name', 'unknown')}")
        except AdvancedOllamaClientError as e:
            print(f"Failed to list models: {e}")
        
        # Check if a specific model is available
        model_name = "qwen3:8b"
        if await client.is_model_available_async(model_name):
            print(f"Model {model_name} is available")
        else:
            print(f"Model {model_name} is not available")
        
        # Example generation options
        options = AdvancedGenerationOptions(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            num_predict=512,
            num_ctx=4096
        )
        
        # Example prompt
        prompt = "Explain how to implement a simple image editing system using AI."
        
        try:
            # Generate text with retries and error handling
            print(f"Generating response for prompt: {prompt[:50]}...")
            response = await client.generate_async(
                prompt=prompt,
                model=model_name,
                options=options,
                stream=False
            )
            print(f"Generated response: {response[:200]}...")
            
            # Example chat conversation
            messages = [
                {"role": "user", "content": "What are the benefits of using AI for image editing?"},
                {"role": "assistant", "content": "AI image editing offers several benefits including automation, consistency, and creative enhancement capabilities."},
                {"role": "user", "content": "Can you elaborate on the creative enhancement capabilities?"}
            ]
            
            print("Chatting with model...")
            chat_response = await client.chat_async(
                messages=messages,
                model=model_name,
                options=options,
                stream=False
            )
            print(f"Chat response: {chat_response[:200]}...")
            
            # Generate embeddings
            embedding_text = "Artificial intelligence is revolutionizing image processing."
            embeddings = await client.generate_embeddings_async(embedding_text)
            print(f"Generated embeddings: {len(embeddings)} dimensions")
            print(f"  First 5 values: {embeddings[:5]}")
            
            # Batch generation
            batch_prompts = [
                "Write a short summary of AI image editing.",
                "Explain the difference between traditional and AI editing.",
                "List three advantages of AI-based image editing."
            ]
            
            print("Batch generating responses...")
            batch_responses = await client.batch_generate_async(
                prompts=batch_prompts,
                model=model_name,
                options=options
            )
            print(f"Batch generation completed: {len(batch_responses)} responses")
            for i, response in enumerate(batch_responses):
                print(f"  Response {i+1}: {response[:100]}...")
            
        except AdvancedOllamaClientError as e:
            print(f"Advanced Ollama client error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        
        # Get model information
        try:
            if await client.is_model_available_async(model_name):
                # Note: The /api/show endpoint might require a different implementation
                print(f"Model {model_name} is available")
        except AdvancedOllamaClientError as e:
            print(f"Failed to get model info: {e}")
    
    # Run async examples
    asyncio.run(run_examples())
    
    print("Advanced Ollama client example completed")
```