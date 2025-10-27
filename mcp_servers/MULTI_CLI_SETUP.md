# Multi-CLI Vision Server Setup Guide

Your local vision MCP server is now compatible with **three major AI CLIs**: Claude Code, Qwen Code CLI, and Gemini CLI!

## 🎯 Why Multiple CLIs?

Different AI CLIs excel at different tasks:

- **Claude Code**: Best for complex reasoning, architecture decisions, and interactive development
- **Qwen Code CLI**: Excellent for batch processing, handling large codebases, and repetitive tasks
- **Gemini CLI**: Great for multi-modal tasks and Google ecosystem integration

**Use case**: Use Qwen for batch image analysis when you need to process many images quickly and the task is straightforward, then use Claude for complex decision-making about the results.

## 🚀 Quick Setup

### Option 1: Configure All CLIs at Once (Recommended)

```bash
cd mcp_servers
python setup_universal.py --log-level INFO
```

This will:
- Auto-detect all installed AI CLIs
- Configure the vision server for each one
- Create backups of existing configs
- Show you exactly what was configured

### Option 2: Configure Specific CLIs

```bash
# Just Qwen (for your batch processing workflows)
python setup_qwen_config.py --log-level INFO

# Just Gemini
python setup_gemini_config.py --log-level INFO

# Claude (already working, but you can reconfigure)
python setup_claude_config.py --log-level INFO
```

### Option 3: Selective Configuration

```bash
# Only configure Claude and Qwen
python setup_universal.py --cli claude,qwen

# Only Qwen at project level (per-project config)
python setup_qwen_config.py --project
```

## 📋 Setup Scripts Overview

### `setup_universal.py` - One Command, All CLIs

**Best for**: First-time setup or when you want all CLIs configured

**Features**:
- Auto-detects which CLIs are installed
- Configures all of them in one go
- Selective configuration with `--cli` flag
- Shows summary of what was configured

**Usage**:
```bash
python setup_universal.py                    # Configure all
python setup_universal.py --cli qwen,gemini  # Configure specific CLIs
python setup_universal.py --dry-run          # Preview without changes
```

### `setup_qwen_config.py` - Qwen Code CLI

**Best for**: When you want to configure just Qwen (your go-to for batch tasks)

**Features**:
- User-level config (`~/.qwen/settings.json`) by default
- Project-level config with `--project` flag
- Includes `trust: false` for safety (tool confirmation required)

**Usage**:
```bash
python setup_qwen_config.py                  # User-level config
python setup_qwen_config.py --project        # Project-level config
python setup_qwen_config.py --log-level DEBUG # Enable debug logging
```

### `setup_gemini_config.py` - Gemini CLI

**Best for**: When you want to configure just Gemini

**Features**:
- Same structure as Qwen setup
- User and project-level support
- Compatible with Gemini's MCP management commands

**Usage**:
```bash
python setup_gemini_config.py                # User-level config
python setup_gemini_config.py --project      # Project-level config
```

### `setup_claude_config.py` - Claude Code (Existing)

**Best for**: Reconfiguring Claude or changing log levels

**Usage**:
```bash
python setup_claude_config.py --log-level INFO
```

## 🔧 Configuration Locations

Each CLI stores its config in a different location:

| CLI | User-Level Config | Project-Level Config |
|-----|-------------------|---------------------|
| **Claude Code** | `~/.config/claude/claude_desktop_config.json` | N/A (global only) |
| **Qwen Code CLI** | `~/.qwen/settings.json` | `.qwen/settings.json` |
| **Gemini CLI** | `~/.gemini/settings.json` | `.gemini/settings.json` |

**User-level**: Configuration applies to all projects
**Project-level**: Configuration only for current directory (Qwen/Gemini only)

## ✅ Verification

After setup, verify each CLI can access the vision server:

### Claude Code
```
# In Claude Code CLI
"Is the local-vision MCP server available?"
# Claude will confirm if it can see the tool
```

### Qwen Code CLI
```bash
# In Qwen CLI
/mcp
# Should show "local-vision" in the list
```

### Gemini CLI
```bash
# In terminal
gemini mcp list
# Should show "local-vision" with server details
```

## 🎨 Usage Examples

### Example 1: Batch Processing with Qwen

**Scenario**: You have 50 images to analyze for a dataset

```bash
# In Qwen Code CLI
"Use see_image to analyze all images in ./dataset/ and create a CSV with:
- filename
- detected objects
- dominant colors
- image dimensions"
```

### Example 2: Complex Analysis with Claude

**Scenario**: Architectural decision about image processing

```bash
# In Claude Code
"Use see_image to analyze this image, then suggest the best approach
for implementing a similar effect in our vision pipeline. Consider
performance, accuracy, and maintainability."
```

### Example 3: Multi-Modal Tasks with Gemini

**Scenario**: Combining vision with other data sources

```bash
# In Gemini CLI
"Use see_image on diagram.jpg to extract the architecture,
then cross-reference with our GCP documentation to verify
the configuration is correct."
```

## 🛠️ Advanced Configuration

### Changing Log Levels

```bash
# More verbose logging for debugging
python setup_universal.py --log-level DEBUG

# Minimal logging for production
python setup_universal.py --log-level WARNING
```

### Trust Settings

By default, `trust: false` means CLIs will ask for confirmation before calling the vision tool.

**To auto-approve** (skip confirmation):
1. Run setup to create config
2. Edit the config file manually
3. Change `"trust": false` to `"trust": true`

**Security note**: Only enable auto-trust if you're the only user of the CLI.

### Custom Ollama URL

```bash
# If Ollama is running on different host/port
python setup_universal.py --ollama-url http://192.168.1.100:11434
```

### Different Vision Models

```bash
# Use a different vision model
python setup_universal.py --vision-model llava:13b
```

## 🧪 Testing

### Test the Server Directly

```bash
# Make sure Ollama is running
ollama serve

# Test the vision server
cd mcp_servers
python test_vision.py ../images/IP.jpeg "Describe this image"
```

### Dry Run Before Making Changes

```bash
# See what would be configured without making changes
python setup_universal.py --dry-run
```

## 🔄 Updating Configuration

Need to change settings? Just run the setup script again:

```bash
# Change log level for all CLIs
python setup_universal.py --log-level DEBUG

# Change just Qwen's config
python setup_qwen_config.py --log-level TRACE
```

Existing configs are automatically backed up to `*.json.backup` before being modified.

## 📊 Compatibility Matrix

| Feature | Claude Code | Qwen Code CLI | Gemini CLI |
|---------|-------------|---------------|------------|
| MCP Support | ✅ Native | ✅ Native | ✅ Native |
| FastMCP | ✅ Yes | ✅ Yes | ✅ Yes (v2.12.3+) |
| Tool Confirmation | Auto | Optional (`trust` flag) | Optional (`trust` flag) |
| Config Levels | User only | User + Project | User + Project |
| Verification | Built-in | `/mcp` command | `gemini mcp list` |

## 🐛 Troubleshooting

### "Cannot connect to Ollama"

```bash
# Start Ollama
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

### "Vision model not available"

```bash
ollama pull qwen2.5vl:7b
```

### "MCP server not showing up in CLI"

1. Verify the config file was created/updated
2. Restart your AI CLI completely
3. Check the config file path matches your CLI's expected location

### "Tool calls timing out"

- Vision processing takes 5-15 seconds per image (normal)
- Increase timeout in config: `"timeout": 180000` (3 minutes)

## 📚 Further Reading

- **README.md**: Complete vision server documentation
- **MCP Specification**: https://modelcontextprotocol.io/
- **Qwen Code CLI Docs**: https://qwenlm.github.io/qwen-code-docs/
- **Gemini CLI Docs**: https://google-gemini.github.io/gemini-cli/

## 🎉 You're All Set!

Your vision server is now universally accessible across multiple AI CLIs. Use the right tool for each task:

- **Complex reasoning** → Claude Code
- **Batch processing** → Qwen Code CLI
- **Multi-modal tasks** → Gemini CLI

All powered by the same local vision model with zero cloud tokens!
