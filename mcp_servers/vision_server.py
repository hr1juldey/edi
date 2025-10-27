#!/usr/bin/env python3
"""
Local Vision MCP Server for Claude Code

Provides image vision capabilities using local Ollama vision models (qwen2.5vl:7b),
avoiding cloud token usage. Completely local processing with structured responses.

Usage:
    # Run with default (INFO) logging
    python vision_server.py

    # Run with specific log level
    LOG_LEVEL=DEBUG python vision_server.py
    LOG_LEVEL=TRACE python vision_server.py

    # Or use fastmcp CLI
    fastmcp run vision_server.py
    fastmcp dev vision_server.py  # With auto-reload
"""

import base64
import json
import os
from pathlib import Path
from typing import Any

import requests
from fastmcp import Context, FastMCP


# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:7b")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Create FastMCP server
mcp = FastMCP(
    name="local-vision",
    instructions=(
        "Local vision analysis using Ollama vision models. "
        "Provides image understanding without using cloud tokens. "
        f"Currently using model: {VISION_MODEL}"
    )
)


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image file to base64 string for Ollama API.

    Args:
        image_path: Path to image file

    Returns:
        Base64-encoded image string

    Raises:
        FileNotFoundError: If image file doesn't exist
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@mcp.tool
async def see_image(image_path: str, prompt: str, ctx: Context) -> dict[str, Any]:
    """
    Analyze an image using local Ollama vision model (qwen2.5vl:7b).

    Uses NO cloud tokens - completely local processing. Provide an image path
    and a question/request about the image. Returns a detailed description or
    answer based on the prompt.

    Use cases:
    - Describe image contents and composition
    - Identify objects, colors, and spatial relationships
    - Detect entities for mask generation
    - Validate if edits match user intent
    - Answer specific questions about images

    Example prompts:
    - "Describe this image in detail"
    - "What objects are in this image? Return as JSON list"
    - "List all entities with their locations and colors"
    - "Does this image contain a blue roof?"
    - "Describe the sky in this image"

    Args:
        image_path: Absolute or relative path to image file (JPG, PNG, JPEG, WEBP, etc.)
        prompt: Question or request about the image. Be specific about what you want.
                Can request JSON output for structured data.
        ctx: FastMCP context for logging and progress reporting

    Returns:
        Dictionary containing:
        - success: Boolean indicating if analysis succeeded
        - response: The model's textual description/answer (if success=True)
        - model: Model name used for analysis
        - image_path: Absolute path to analyzed image (if success=True)
        - error: Error message (if success=False)
        - troubleshooting: Helpful tips (if success=False)

    Examples:
        >>> result = await see_image("./test.jpg", "What colors are in this image?")
        >>> print(result["response"])
    """
    await ctx.info(f"🔍 Vision analysis starting for: {image_path}")
    await ctx.debug(f"Prompt: {prompt}")
    await ctx.debug(f"Using model: {VISION_MODEL}")

    try:
        # Validate image path
        await ctx.debug("Validating image path...")
        img_path = Path(image_path).expanduser().resolve()

        if not img_path.exists():
            error_msg = f"Image file not found: {image_path}"
            await ctx.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "model": VISION_MODEL,
                "troubleshooting": [
                    "Check if the file path is correct",
                    "Use absolute paths for clarity",
                    f"Attempted to find: {img_path}"
                ]
            }

        if not img_path.is_file():
            error_msg = f"Path exists but is not a file: {image_path}"
            await ctx.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "model": VISION_MODEL,
                "troubleshooting": ["Ensure the path points to an image file, not a directory"]
            }

        await ctx.info(f"✓ Image found: {img_path.name}")
        await ctx.report_progress(1, 3, "Encoding image...")

        # Encode image to base64
        try:
            await ctx.debug(f"Encoding image: {img_path}")
            image_base64 = encode_image_to_base64(str(img_path))
            await ctx.debug(f"Image encoded: {len(image_base64)} bytes")
        except Exception as e:
            error_msg = f"Failed to encode image: {str(e)}"
            await ctx.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "model": VISION_MODEL,
                "troubleshooting": [
                    "Ensure the file is a valid image format",
                    "Supported formats: JPG, PNG, JPEG, WEBP, GIF",
                    "Check file permissions"
                ]
            }

        await ctx.report_progress(2, 3, "Analyzing with vision model...")

        # Prepare request to Ollama
        payload = {
            "model": VISION_MODEL,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False
        }

        await ctx.debug(f"Sending request to Ollama at {OLLAMA_BASE_URL}")

        # Call Ollama API
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=120  # Vision models can take longer
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Cannot connect to Ollama: {str(e)}"
            await ctx.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "model": VISION_MODEL,
                "troubleshooting": [
                    "Ensure Ollama is running: ollama serve",
                    f"Check if Ollama is accessible at: {OLLAMA_BASE_URL}",
                    "Verify Ollama is not blocked by firewall"
                ]
            }
        except requests.exceptions.Timeout as e:
            error_msg = f"Ollama request timed out: {str(e)}"
            await ctx.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "model": VISION_MODEL,
                "troubleshooting": [
                    "Vision model processing can take time for large images",
                    "Try reducing image size if very large",
                    "Check Ollama server logs for issues"
                ]
            }
        except requests.exceptions.RequestException as e:
            error_msg = f"Ollama API error: {str(e)}"
            await ctx.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "model": VISION_MODEL,
                "troubleshooting": [
                    f"Check if {VISION_MODEL} is installed: ollama list",
                    f"Install if needed: ollama pull {VISION_MODEL}",
                    "Check Ollama server logs for details"
                ]
            }

        result = response.json()
        await ctx.debug(f"Ollama response received: {len(result.get('response', ''))} chars")

        await ctx.report_progress(3, 3, "Analysis complete!")

        model_response = result.get("response", "")
        if not model_response:
            await ctx.warning("Model returned empty response")
            model_response = "[Empty response from model]"

        await ctx.info(f"✓ Vision analysis complete: {len(model_response)} characters")

        return {
            "success": True,
            "response": model_response,
            "model": VISION_MODEL,
            "image_path": str(img_path),
            "image_size_bytes": img_path.stat().st_size,
            "image_name": img_path.name
        }

    except Exception as e:
        error_msg = f"Unexpected error during vision analysis: {str(e)}"
        await ctx.error(error_msg)
        await ctx.debug(f"Exception type: {type(e).__name__}")
        return {
            "success": False,
            "error": error_msg,
            "model": VISION_MODEL,
            "troubleshooting": [
                "Check server logs for detailed error information",
                "Ensure all dependencies are installed",
                "Verify image file is not corrupted"
            ]
        }


# Health check resource
@mcp.resource("vision://health")
async def health_check() -> str:
    """Check if Ollama vision service is available."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = [m["name"] for m in models]

        vision_available = VISION_MODEL in model_names

        return json.dumps({
            "status": "healthy" if vision_available else "degraded",
            "ollama_url": OLLAMA_BASE_URL,
            "vision_model": VISION_MODEL,
            "vision_model_available": vision_available,
            "available_models": model_names
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "unhealthy",
            "ollama_url": OLLAMA_BASE_URL,
            "error": str(e),
            "troubleshooting": [
                "Start Ollama: ollama serve",
                f"Install vision model: ollama pull {VISION_MODEL}"
            ]
        }, indent=2)


if __name__ == "__main__":
    # Run the server
    print(f"🚀 Starting Local Vision MCP Server")
    print(f"   Model: {VISION_MODEL}")
    print(f"   Ollama: {OLLAMA_BASE_URL}")
    print(f"   Log Level: {LOG_LEVEL}")
    print(f"   Debug Mode: {LOG_LEVEL in ['DEBUG', 'TRACE']}")
    print()

    mcp.run(
        transport="stdio",
        log_level=LOG_LEVEL,
        debug=(LOG_LEVEL in ["DEBUG", "TRACE"])
    )
