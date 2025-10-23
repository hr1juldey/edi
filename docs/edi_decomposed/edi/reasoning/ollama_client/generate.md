# OllamaClient.generate()

[Back to Ollama Client](../reasoning_ollama_client.md)

## Related User Story
"As a user, I want EDI to understand my intent and generate appropriate editing instructions." (from PRD - core functionality)

## Function Signature
`generate(prompt, model) -> str`

## Parameters
- `prompt` - The input prompt to send to the LLM
- `model` - The identifier for the model to use (e.g., "qwen3:8b", "gemma3:4b")

## Returns
- `str` - The generated text response from the LLM

## Step-by-step Logic
1. Prepare the API request with the given prompt and model parameters
2. Send the request to the Ollama server endpoint
3. Handle the response from the Ollama API
4. Process any streaming responses if applicable
5. Handle connection errors, timeouts, and retry if needed
6. Return the generated text response
7. Log the interaction for debugging and learning purposes

## Error Handling
- Handles network connection issues with retries
- Manages timeout scenarios gracefully
- Validates response from Ollama server
- Provides fallback mechanisms if primary model fails
- Reports clear error messages to higher-level functions

## Performance Optimizations
- Uses streaming API to show partial results
- Sets appropriate context size (num_ctx=4096) for faster processing
- Maintains connection to Ollama server to avoid cold starts
- Implements efficient request/response processing

## Input/Output Data Structures
### Input
- Prompt: Text string containing the instruction for the LLM
- Model: String identifier for the specific model to use

### Output
- Generated text response from the LLM as a string
- May contain structured data (JSON) depending on the prompt

## See Docs

### Python Implementation Example
Implementation of the OllamaClient.generate() method:

```python
import requests
import json
import time
from typing import Dict, Any, Optional, Generator
import logging
from urllib.parse import urljoin

class OllamaError(Exception):
    """Custom exception for Ollama API errors."""
    pass

class OllamaClient:
    """
    Client for interacting with Ollama API to generate text responses.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 300):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        
        # Connection pooling and reuse
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    def generate(self, prompt: str, model: str, **kwargs) -> str:
        """
        Generate text response from Ollama API.
        
        Args:
            prompt: Text string containing the instruction for the LLM
            model: String identifier for the specific model to use
            **kwargs: Additional options to pass to the API
            
        Returns:
            Generated text response from the LLM as a string
        """
        # Prepare API request with parameters
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,  # We want to return a complete response
            "options": {
                "num_ctx": kwargs.get("num_ctx", 4096),  # Context size (performance optimization)
                "temperature": kwargs.get("temperature", 0.7),  # Creativity controls
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", 40),
                "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
                "num_predict": kwargs.get("num_predict", 512)  # Max tokens to generate
            }
        }
        
        # Add any additional options
        for key, value in kwargs.items():
            if key not in data["options"]:
                data[key] = value
        
        # Set system prompt if provided
        if "system" in kwargs:
            data["system"] = kwargs["system"]
        
        # Send request to Ollama server endpoint
        url = urljoin(self.base_url, "/api/generate")
        
        try:
            response = self.session.post(url, json=data, timeout=self.timeout)
        except requests.exceptions.ConnectionError:
            raise OllamaError(f"Failed to connect to Ollama server at {self.base_url}")
        except requests.exceptions.Timeout:
            raise OllamaError(f"Request to Ollama server timed out after {self.timeout} seconds")
        except requests.exceptions.RequestException as e:
            raise OllamaError(f"Request error: {str(e)}")
        
        # Handle the response from the Ollama API
        if response.status_code != 200:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", f"HTTP {response.status_code}")
            except json.JSONDecodeError:
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            raise OllamaError(f"Ollama API error: {error_msg}")
        
        try:
            result = response.json()
        except json.JSONDecodeError:
            raise OllamaError("Invalid JSON response from Ollama API")
        
        # Process any streaming responses if applicable
        if "response" not in result:
            raise OllamaError(f"Unexpected response format: {result}")
        
        generated_text = result["response"]
        
        # Log the interaction for debugging and learning purposes
        self.logger.info(
            f"Ollama request completed: model={model}, "
            f"prompt_len={len(prompt)}, response_len={len(generated_text)}"
        )
        
        return generated_text
    
    def generate_streaming(self, prompt: str, model: str, **kwargs) -> Generator[str, None, None]:
        """
        Generate text response with streaming (for showing partial results).
        
        Args:
            prompt: Text string containing the instruction for the LLM
            model: String identifier for the specific model to use
            **kwargs: Additional options to pass to the API
            
        Yields:
            Chunks of generated text as they become available
        """
        data = {
            "model": model,
            "prompt": prompt,
            "stream": True,  # Enable streaming
            "options": {
                "num_ctx": kwargs.get("num_ctx", 4096),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", 40),
                "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
                "num_predict": kwargs.get("num_predict", 512)
            }
        }
        
        for key, value in kwargs.items():
            if key not in data["options"]:
                data[key] = value
        
        if "system" in kwargs:
            data["system"] = kwargs["system"]
        
        url = urljoin(self.base_url, "/api/generate")
        
        try:
            response = self.session.post(url, json=data, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            yield chunk["response"]
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
        except requests.exceptions.RequestException as e:
            raise OllamaError(f"Streaming request error: {str(e)}")
    
    def generate_with_retry(self, prompt: str, model: str, max_retries: int = 3, **kwargs) -> str:
        """
        Generate with automatic retry on failures.
        
        Args:
            prompt: Text string containing the instruction for the LLM
            model: String identifier for the specific model to use
            max_retries: Maximum number of retry attempts
            **kwargs: Additional options to pass to the API
            
        Returns:
            Generated text response from the LLM as a string
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return self.generate(prompt, model, **kwargs)
            except OllamaError as e:
                last_error = e
                if attempt < max_retries:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"All {max_retries + 1} attempts failed")
        
        # If we get here, all retries failed
        raise last_error
    
    def check_model_available(self, model_name: str) -> bool:
        """
        Check if a specific model is available on the Ollama server.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if the model is available, False otherwise
        """
        try:
            response = self.session.get(urljoin(self.base_url, "/api/tags"))
            response.raise_for_status()
            
            data = response.json()
            available_models = [model["name"] for model in data.get("models", [])]
            
            # Check if the requested model is in the list (with possible variations like :latest)
            return any(
                model_name == available_name or 
                f"{model_name}:" in available_name or
                available_name.startswith(f"{model_name}:")
                for available_name in available_models
            )
        except Exception:
            return False
    
    def get_available_models(self) -> list:
        """
        Get a list of all available models on the Ollama server.
        
        Returns:
            List of available model names
        """
        try:
            response = self.session.get(urljoin(self.base_url, "/api/tags"))
            response.raise_for_status()
            
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            self.logger.error(f"Error getting available models: {e}")
            return []

# Example usage
if __name__ == "__main__":
    client = OllamaClient()
    
    # Check if models are available
    available_models = client.get_available_models()
    print(f"Available models: {available_models[:5]}...")  # Show first 5
    
    # Example prompts
    test_prompt = "Describe how to change the sky in an image to make it more dramatic."
    
    try:
        # Basic generation
        result = client.generate(
            prompt=test_prompt,
            model="qwen3:8b",  # Use an actual model name if available
            temperature=0.7
        )
        print(f"Generated response: {result[:200]}...")
        
        # Generate with retry
        result = client.generate_with_retry(
            prompt="Convert this image to black and white with high contrast",
            model="qwen3:8b",  # Use an actual model name if available
            max_retries=2
        )
        print(f"Generated with retry: {result[:200]}...")
        
        # Streaming example (commented out to avoid real streaming in example)
        # print("Streaming response:")
        # for chunk in client.generate_streaming(
        #     prompt="Write a short poem about image editing",
        #     model="qwen3:8b"
        # ):
        #     print(chunk, end="", flush=True)
        # print()
        
    except OllamaError as e:
        print(f"Ollama error: {e}")
    except Exception as e:
        print(f"Other error: {e}")
```

### Advanced Ollama Client Implementation
Enhanced Ollama client with additional features:

```python
import requests
import json
import time
import asyncio
import aiohttp
from typing import Dict, Any, Optional, AsyncGenerator, Union
import logging
from urllib.parse import urljoin
import hashlib
import tempfile
import os

class AdvancedOllamaClient:
    """
    Advanced Ollama client with async support, caching, and enhanced error handling.
    """
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434", 
                 timeout: int = 300,
                 cache_enabled: bool = True,
                 cache_size_mb: int = 100):
        self.base_url = base_url
        self.timeout = timeout
        self.cache_enabled = cache_enabled
        self.cache_size_mb = cache_size_mb
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache
        self.cache = {}
        self.cache_size = 0
        
        # Session for sync requests
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    def _get_cache_key(self, prompt: str, model: str, **kwargs) -> str:
        """Generate a cache key for the given parameters."""
        cache_input = f"{prompt}_{model}_{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[str]:
        """Check if response is in cache."""
        if self.cache_enabled and cache_key in self.cache:
            self.logger.info(f"Cache hit for key: {cache_key[:8]}")
            return self.cache[cache_key]
        return None
    
    def _save_to_cache(self, cache_key: str, response: str):
        """Save response to cache with size management."""
        if not self.cache_enabled:
            return
        
        # Simple size management: don't cache if it would exceed size
        response_size = len(response.encode('utf-8'))
        if (self.cache_size + response_size) > (self.cache_size_mb * 1024 * 1024):
            # Clear cache if too large (simple approach)
            self.cache = {}
            self.cache_size = 0
            self.logger.info("Cache cleared due to size limit")
        
        self.cache[cache_key] = response
        self.cache_size += response_size
    
    def generate(self, 
                 prompt: str, 
                 model: str, 
                 use_cache: bool = True,
                 **kwargs) -> str:
        """
        Generate text response with caching and advanced options.
        """
        # Create cache key
        cache_key = self._get_cache_key(prompt, model, **kwargs)
        
        # Check cache first
        if use_cache:
            cached_response = self._check_cache(cache_key)
            if cached_response is not None:
                return cached_response
        
        # Prepare request data
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": kwargs.get("num_ctx", 4096),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", 40),
                "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
                "num_predict": kwargs.get("num_predict", 512),
                "stop": kwargs.get("stop", [])  # Stop sequences
            }
        }
        
        # Add additional options
        for key, value in kwargs.items():
            if key not in data["options"]:
                data[key] = value
        
        if "system" in kwargs:
            data["system"] = kwargs["system"]
        
        # Make request
        url = urljoin(self.base_url, "/api/generate")
        
        start_time = time.time()
        try:
            response = self.session.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            
            if "response" not in result:
                raise OllamaError(f"Unexpected response format: {result}")
            
            generated_text = result["response"]
            
            # Cache the response
            if use_cache:
                self._save_to_cache(cache_key, generated_text)
            
            self.logger.info(
                f"Generated {len(generated_text)} chars in {time.time() - start_time:.2f}s "
                f"using model {model}"
            )
            
            return generated_text
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise OllamaError(f"Request error: {str(e)}")
        except json.JSONDecodeError:
            raise OllamaError("Invalid JSON response from Ollama API")
    
    async def generate_async(self, 
                           prompt: str, 
                           model: str, 
                           **kwargs) -> str:
        """
        Asynchronously generate text response.
        """
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": kwargs.get("num_ctx", 4096),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", 40),
                "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
                "num_predict": kwargs.get("num_predict", 512)
            }
        }
        
        for key, value in kwargs.items():
            if key not in data["options"]:
                data[key] = value
        
        if "system" in kwargs:
            data["system"] = kwargs["system"]
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            url = urljoin(self.base_url, "/api/generate")
            
            try:
                async with session.post(url, json=data) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise OllamaError(f"Ollama API error: HTTP {response.status} - {text}")
                    
                    result = await response.json()
                    
                    if "response" not in result:
                        raise OllamaError(f"Unexpected response format: {result}")
                    
                    return result["response"]
                    
            except asyncio.TimeoutError:
                raise OllamaError(f"Request timed out after {self.timeout} seconds")
            except aiohttp.ClientError as e:
                raise OllamaError(f"Client error: {str(e)}")
    
    def generate_structured(self, 
                           prompt: str, 
                           model: str, 
                           expected_format: str = "json",
                           **kwargs) -> Union[Dict[str, Any], str]:
        """
        Generate structured output like JSON.
        
        Args:
            prompt: Instruction for the LLM
            model: Model to use
            expected_format: Expected format (json, yaml, etc.)
            **kwargs: Additional options
            
        Returns:
            Parsed structured data or error string
        """
        # Enhance prompt to request specific format
        if expected_format.lower() == "json":
            formatted_prompt = f"{prompt}\n\nRespond in valid JSON format only, with no other text."
        else:
            formatted_prompt = f"{prompt}\n\nRespond in {expected_format} format only."
        
        try:
            response = self.generate(formatted_prompt, model, **kwargs)
            
            if expected_format.lower() == "json":
                # Try to extract JSON from the response if it has extra text
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    return json.loads(json_str)
                else:
                    # If no JSON found, try to parse the whole response
                    return json.loads(response)
            
            return response
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed: {e}")
            return {"error": f"Failed to parse JSON response: {str(e)}", "raw_response": response}
        except Exception as e:
            self.logger.error(f"Structure generation failed: {e}")
            return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Initialize advanced client
    advanced_client = AdvancedOllamaClient()
    
    # Example: Generate structured response
    try:
        structured_result = advanced_client.generate_structured(
            prompt="Generate an image editing instruction with target entity, change type, and parameters",
            model="qwen3:8b",  # Use an actual model name if available
            expected_format="json"
        )
        print(f"Structured result: {structured_result}")
        
        # Example: Async generation
        async def async_example():
            result = await advanced_client.generate_async(
                prompt="Write a short explanation of how to enhance sky colors in an image",
                model="qwen3:8b",  # Use an actual model name if available
                temperature=0.5
            )
            print(f"Async result: {result[:200]}...")
        
        asyncio.run(async_example())
        
    except Exception as e:
        print(f"Error in advanced example: {e}")
```