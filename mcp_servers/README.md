# Local Vision MCP Server

Local vision analysis using Ollama's vision models (qwen2.5vl:7b) without consuming cloud tokens.

**Universal Compatibility**: Works with Claude Code, Qwen Code CLI, and Gemini CLI through the standard Model Context Protocol (MCP).

## Features

- 🔍 **Image analysis** using local Ollama vision models
- 🚫 **Zero cloud tokens** - completely local processing
- 📊 **Structured responses** with detailed error handling
- 📝 **Multiple log levels** (TRACE, DEBUG, INFO, WARNING, ERROR)
- ✅ **Health check** resource to verify Ollama availability
- 🎯 **Context-aware logging** for debugging
- 🌐 **Universal MCP compatibility** - works with multiple AI CLIs

## Compatibility Matrix

| AI CLI | MCP Support | Status | Config File | Verification Command |
|--------|-------------|--------|-------------|---------------------|
| **Claude Code** | ✅ Native | ✅ Tested | `~/.config/claude/claude_desktop_config.json` | Built-in |
| **Qwen Code CLI** | ✅ Native | ✅ Tested | `~/.qwen/settings.json` | `/mcp` |
| **Gemini CLI** | ✅ Native | ✅ Tested | `~/.gemini/settings.json` | `gemini mcp list` |

**Recommended Use Cases:**
- **Claude Code**: Complex reasoning tasks, interactive development
- **Qwen Code CLI**: Batch processing, large codebases, simpler tasks with high volume
- **Gemini CLI**: Multi-modal tasks, Google ecosystem integration

## Quick Start

Choose your AI CLI and follow the setup:

### Universal Setup (All CLIs at once)

```bash
# Configure all detected AI CLIs
python setup_universal.py --log-level INFO

# Configure specific CLIs only
python setup_universal.py --cli claude,qwen
```

### Individual CLI Setup

```bash
# Claude Code
python setup_claude_config.py --log-level INFO

# Qwen Code CLI
python setup_qwen_config.py --log-level INFO

# Gemini CLI
python setup_gemini_config.py --log-level INFO
```

### Test the Server

```bash
# Test with a sample image
python test_vision.py ../images/IP.jpeg "Describe this image"
```

## Prerequisites

1. **Ollama running locally**:
   ```bash
   ollama serve
   ```

2. **Vision model installed**:
   ```bash
   ollama pull qwen2.5vl:7b
   ```

3. **Verify installation**:
   ```bash
   ollama list | grep qwen2.5vl
   ```

## Usage

### Manual Testing

Test the server manually before connecting to Claude:

```bash
# Run with default (INFO) logging
python vision_server.py

# Run with DEBUG logging
LOG_LEVEL=DEBUG python vision_server.py

# Run with TRACE logging (most verbose)
LOG_LEVEL=TRACE python vision_server.py

# Using fastmcp CLI
fastmcp run vision_server.py

# With auto-reload (development)
fastmcp dev vision_server.py
```

### Testing the Tool

Once running, you can test with the MCP inspector or by calling from Claude Code:

```python
# Example tool call
result = await see_image(
    image_path="./test.jpg",
    prompt="Describe this image in detail"
)

# Request structured JSON
result = await see_image(
    image_path="./test.jpg",
    prompt="List all objects in this image as a JSON array with their colors and locations"
)
```

### Check Health

```python
# Check if vision service is healthy
health = await read_resource("vision://health")
```

## Configuration

### Automatic Configuration (Recommended)

Use the setup scripts for automatic configuration:

```bash
# Configure all CLIs at once
python setup_universal.py

# Or configure individual CLIs
python setup_claude_config.py   # For Claude Code
python setup_qwen_config.py     # For Qwen Code CLI
python setup_gemini_config.py   # For Gemini CLI
```

### Manual Configuration

If you prefer manual configuration, add the following to your CLI's config file:

#### Claude Code

**Linux/Mac**: `~/.config/claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "local-vision": {
      "command": "python",
      "args": ["/absolute/path/to/vision_server.py"],
      "env": {
        "LOG_LEVEL": "INFO",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "VISION_MODEL": "qwen2.5vl:7b"
      }
    }
  }
}
```

#### Qwen Code CLI

**Config file**: `~/.qwen/settings.json` (user-level) or `.qwen/settings.json` (project-level)

```json
{
  "mcpServers": {
    "local-vision": {
      "command": "python",
      "args": ["/absolute/path/to/vision_server.py"],
      "env": {
        "LOG_LEVEL": "INFO",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "VISION_MODEL": "qwen2.5vl:7b"
      },
      "timeout": 120000,
      "trust": false
    }
  }
}
```

**Verification**: Type `/mcp` in Qwen CLI to see loaded servers

#### Gemini CLI

**Config file**: `~/.gemini/settings.json` (user-level) or `.gemini/settings.json` (project-level)

```json
{
  "mcpServers": {
    "local-vision": {
      "command": "python",
      "args": ["/absolute/path/to/vision_server.py"],
      "env": {
        "LOG_LEVEL": "INFO",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "VISION_MODEL": "qwen2.5vl:7b"
      },
      "timeout": 120000,
      "trust": false
    }
  }
}
```

**Verification**: Run `gemini mcp list` to see configured servers

### Log Level Options

Set `LOG_LEVEL` environment variable to control verbosity:

- `ERROR` - Only errors
- `WARNING` - Warnings and errors
- `INFO` - General information (default, recommended)
- `DEBUG` - Detailed debugging information
- `TRACE` - Maximum verbosity (all operations logged)

## Tool: `see_image`

Analyze images using local vision model.

### Parameters

- `image_path` (string, required): Path to image file (JPG, PNG, JPEG, WEBP, etc.)
- `prompt` (string, required): Question or request about the image

### Returns

Structured dictionary with:

**On Success**:
```json
{
  "success": true,
  "response": "Detailed description from vision model...",
  "model": "qwen2.5vl:7b",
  "image_path": "/absolute/path/to/image.jpg",
  "image_size_bytes": 1234567,
  "image_name": "image.jpg"
}
```

**On Failure**:
```json
{
  "success": false,
  "error": "Detailed error message",
  "model": "qwen2.5vl:7b",
  "troubleshooting": [
    "Helpful tip 1",
    "Helpful tip 2"
  ]
}
```

### Example Prompts

**General description**:
```
"Describe this image in detail"
```

**Structured JSON output**:
```
"List all objects in this image as a JSON array with properties: name, color, position, size"
```

**Entity detection for mask generation**:
```
"Identify all distinct regions in this image. For each region, provide: label, dominant color (hex), approximate percentage of image area, spatial location (top/center/bottom, left/center/right)"
```

**Validation queries**:
```
"Does this image contain a blue roof? Answer yes or no and explain."
"Compare the sky in this image. Is it dramatic? Describe the cloud formation."
```

**Color analysis**:
```
"What are the dominant colors in this image? Return as JSON array with hex codes."
```

### CLI-Specific Usage

#### Using with Claude Code
```
# Claude automatically discovers MCP tools
"Use the see_image tool to analyze test.jpg and tell me what objects are in it"
```

#### Using with Qwen Code CLI
```bash
# Verify MCP server is loaded
/mcp

# Use the tool
"Use the see_image tool to describe input.jpg in detail"

# For batch processing (Qwen's strength)
"Use see_image to analyze all images in the folder and create a summary"
```

#### Using with Gemini CLI
```bash
# Verify configuration
gemini mcp list

# Use the tool
"Use the see_image tool to analyze photo.jpg"

# With structured output
"Use see_image on diagram.jpg to extract all text and labels as JSON"
```

## Resource: `vision://health`

Health check endpoint to verify Ollama and vision model availability.

### Returns

```json
{
  "status": "healthy",
  "ollama_url": "http://localhost:11434",
  "vision_model": "qwen2.5vl:7b",
  "vision_model_available": true,
  "available_models": ["qwen2.5vl:7b", "qwen3:8b", "gemma3:4b", ...]
}
```

## Integration with EDI Project

This vision tool is designed to support EDI's vision subsystem:

1. **Stage 1: Initial Analysis** - Use `see_image` to get high-level understanding of image contents before running SAM/CLIP
2. **Stage 4: Validation** - Verify generated masks match user intent by asking vision model
3. **Intent Parsing** - Help DSpy understand what entities user is referring to
4. **Change Detection** - Validate if edits match expected changes

### Example EDI Usage

```python
# Before running expensive SAM/CLIP pipeline, get quick overview
result = await see_image(
    image_path="input.jpg",
    prompt="""List all major objects/regions in this image.
    For each, provide: name, approximate location, dominant color.
    Return as JSON array."""
)

# Parse result to guide SAM/CLIP processing
entities_preview = json.loads(result["response"])

# After generating mask, validate it
validation = await see_image(
    image_path="mask_overlay.jpg",
    prompt=f"""This image shows a red overlay mask.
    Does the mask correctly cover: {user_target_entity}?
    Answer yes or no and explain what the mask is covering."""
)
```

## Troubleshooting

### "Cannot connect to Ollama"

```bash
# Start Ollama server
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

### "Vision model not available"

```bash
# Check installed models
ollama list

# Install vision model if missing
ollama pull qwen2.5vl:7b
```

### "Image file not found"

- Use absolute paths: `/home/user/image.jpg` instead of `~/image.jpg`
- Verify file exists: `ls -la /path/to/image.jpg`
- Check permissions: Image file must be readable

### High latency

- Vision models take 5-15 seconds per image (normal)
- Reduce image size if very large (>4096px)
- Ensure GPU is available for faster processing

## Performance

**Typical response times** (RTX 3060 12GB):
- Small images (512x512): 3-5 seconds
- Medium images (1024x1024): 5-10 seconds
- Large images (2048x2048): 10-15 seconds

**Memory usage**:
- Vision model (qwen2.5vl:7b): ~6 GB VRAM
- Per request: ~500 MB RAM

## Development

### Running Tests

```bash
# Test with sample image
python -c "
import asyncio
from vision_server import see_image
from fastmcp import Context, FastMCP

async def test():
    mcp = FastMCP('test')
    ctx = Context(mcp)
    result = await see_image('./test.jpg', 'Describe this image', ctx)
    print(result)

asyncio.run(test())
"
```

### Debugging

Enable DEBUG or TRACE logging to see detailed operation logs:

```bash
LOG_LEVEL=DEBUG python vision_server.py
```

Look for:
- Image encoding steps
- Ollama request/response
- Error details
- Progress reporting

## License

Part of the EDI (Edit with Intelligence) project.
