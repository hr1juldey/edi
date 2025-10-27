#!/usr/bin/env python3
"""
Test script for the local vision MCP server.

This script tests the vision server functionality without requiring
Claude Code to be running.

Usage:
    python test_vision.py <image_path> [prompt]

Example:
    python test_vision.py ../images/IP.jpeg "Describe this image in detail"
    python test_vision.py test.jpg "What colors are present?"
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path to import vision_server
sys.path.insert(0, str(Path(__file__).parent))

from fastmcp import Context, FastMCP
import requests


async def test_health():
    """Test the health check endpoint."""
    print("🏥 Testing health check...")
    print("-" * 60)

    # Directly test Ollama connection
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = [m["name"] for m in models]

        vision_available = "qwen2.5vl:7b" in model_names

        health_data = {
            "status": "healthy" if vision_available else "degraded",
            "ollama_url": "http://localhost:11434",
            "vision_model": "qwen2.5vl:7b",
            "vision_model_available": vision_available,
            "available_models": model_names
        }
    except Exception as e:
        health_data = {
            "status": "unhealthy",
            "ollama_url": "http://localhost:11434",
            "error": str(e),
            "troubleshooting": [
                "Start Ollama: ollama serve",
                "Install vision model: ollama pull qwen2.5vl:7b"
            ]
        }

    print(json.dumps(health_data, indent=2))
    print("-" * 60)

    if health_data.get("status") == "healthy":
        print("✅ Health check: PASSED")
        return True
    else:
        print("❌ Health check: FAILED")
        print()
        print("Troubleshooting:")
        for tip in health_data.get("troubleshooting", []):
            print(f"  • {tip}")
        return False


async def test_vision(image_path: str, prompt: str):
    """Test the vision analysis with a real image."""
    print()
    print("🔍 Testing vision analysis...")
    print("-" * 60)
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")
    print("-" * 60)
    print()

    # Import vision analysis logic directly
    from vision_server import encode_image_to_base64, OLLAMA_BASE_URL, VISION_MODEL
    import requests

    try:
        # Validate and encode image
        img_path = Path(image_path).expanduser().resolve()

        if not img_path.exists():
            return {
                "success": False,
                "error": f"Image file not found: {image_path}"
            }

        print(f"✓ Image found: {img_path.name}")
        print(f"  Encoding image...")

        image_base64 = encode_image_to_base64(str(img_path))

        print(f"  Image encoded: {len(image_base64)} bytes")
        print(f"  Sending to Ollama ({VISION_MODEL})...")

        # Call Ollama
        payload = {
            "model": VISION_MODEL,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False
        }

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()

        result_data = response.json()
        model_response = result_data.get("response", "")

        result = {
            "success": True,
            "response": model_response,
            "model": VISION_MODEL,
            "image_path": str(img_path),
            "image_size_bytes": img_path.stat().st_size,
            "image_name": img_path.name
        }

    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "model": VISION_MODEL
        }

    print("Result:")
    print("=" * 60)

    if result["success"]:
        print("✅ Status: SUCCESS")
        print()
        print(f"Model: {result['model']}")
        print(f"Image: {result['image_name']} ({result['image_size_bytes']:,} bytes)")
        print()
        print("Response:")
        print("-" * 60)
        print(result["response"])
        print("-" * 60)
        return True
    else:
        print("❌ Status: FAILED")
        print()
        print(f"Error: {result['error']}")
        print()
        if "troubleshooting" in result:
            print("Troubleshooting:")
            for tip in result["troubleshooting"]:
                print(f"  • {tip}")
        return False


async def main():
    """Main test function."""
    print()
    print("🧪 Local Vision MCP Server Test Suite")
    print("=" * 60)
    print()

    # Test 1: Health check
    health_ok = await test_health()

    if not health_ok:
        print()
        print("⚠️  Skipping vision test due to failed health check")
        return 1

    # Test 2: Vision analysis (if image path provided)
    if len(sys.argv) < 2:
        print()
        print("ℹ️  No image path provided, skipping vision test")
        print()
        print("Usage:")
        print(f"  python {Path(__file__).name} <image_path> [prompt]")
        print()
        print("Example:")
        print(f"  python {Path(__file__).name} ../images/IP.jpeg \"Describe this image\"")
        return 0

    image_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Describe this image in detail"

    vision_ok = await test_vision(image_path, prompt)

    print()
    print("=" * 60)
    if health_ok and vision_ok:
        print("✅ All tests PASSED")
        print()
        print("The vision server is ready to use!")
        print()
        print("Next steps:")
        print("1. Run setup: python setup_claude_config.py")
        print("2. Restart Claude Code")
        print("3. Use the 'see_image' tool in Claude Code")
        return 0
    else:
        print("❌ Some tests FAILED")
        print()
        print("Please fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
