#!/usr/bin/env python3
"""
Setup script to configure Gemini CLI to use the local vision MCP server.

This script automatically adds the vision server to Gemini's configuration file.

Usage:
    python setup_gemini_config.py [--log-level LEVEL] [--user | --project]

Options:
    --log-level: Set logging level (ERROR, WARNING, INFO, DEBUG, TRACE)
                 Default: INFO
    --user:      Configure at user level (~/.gemini/settings.json) [default]
    --project:   Configure at project level (.gemini/settings.json)
"""

import argparse
import json
import sys
from pathlib import Path


def get_gemini_config_path(project_level: bool = False) -> Path:
    """Get the path to Gemini's configuration file."""
    if project_level:
        # Project-level configuration
        config_dir = Path.cwd() / ".gemini"
    else:
        # User-level configuration
        config_dir = Path.home() / ".gemini"

    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "settings.json"


def get_vision_server_path() -> Path:
    """Get absolute path to vision_server.py"""
    return Path(__file__).parent / "vision_server.py"


def load_gemini_config(config_path: Path) -> dict:
    """Load existing Gemini configuration or create new one."""
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def save_gemini_config(config_path: Path, config: dict):
    """Save Gemini configuration to file."""
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✅ Configuration saved to: {config_path}")


def add_vision_server_to_config(
    config: dict,
    log_level: str = "INFO",
    ollama_url: str = "http://localhost:11434",
    vision_model: str = "qwen2.5vl:7b"
) -> dict:
    """Add vision server configuration to Gemini config."""

    vision_server_path = get_vision_server_path()

    # Ensure mcpServers key exists
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Add or update local-vision server
    config["mcpServers"]["local-vision"] = {
        "command": sys.executable,  # Use current Python interpreter
        "args": [str(vision_server_path)],
        "env": {
            "LOG_LEVEL": log_level.upper(),
            "OLLAMA_BASE_URL": ollama_url,
            "VISION_MODEL": vision_model
        },
        "timeout": 120000,  # 2 minutes (vision processing can take time)
        "trust": False  # Require confirmation for tool calls (safe default)
    }

    return config


def verify_prerequisites() -> list[str]:
    """Check if prerequisites are met."""
    warnings = []

    # Check if vision_server.py exists
    vision_server_path = get_vision_server_path()
    if not vision_server_path.exists():
        warnings.append(f"❌ vision_server.py not found at: {vision_server_path}")

    # Check if fastmcp is installed
    try:
        import fastmcp  # noqa: F401
    except ImportError:
        warnings.append("❌ fastmcp not installed. Run: pip install fastmcp")

    # Check if requests is installed
    try:
        import requests
    except ImportError:
        warnings.append("❌ requests not installed. Run: pip install requests")
        return warnings  # Can't check Ollama without requests

    # Try to check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code != 200:
            warnings.append("⚠️  Ollama may not be running. Start with: ollama serve")
    except Exception:
        warnings.append("⚠️  Cannot connect to Ollama. Start with: ollama serve")

    # Check if vision model is available
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            if "qwen2.5vl:7b" not in model_names:
                warnings.append("⚠️  Vision model not installed. Run: ollama pull qwen2.5vl:7b")
    except Exception:
        pass

    return warnings


def main():
    parser = argparse.ArgumentParser(
        description="Configure Gemini CLI to use local vision MCP server"
    )
    parser.add_argument(
        "--log-level",
        choices=["ERROR", "WARNING", "INFO", "DEBUG", "TRACE"],
        default="INFO",
        help="Logging level for the vision server (default: INFO)"
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--vision-model",
        default="qwen2.5vl:7b",
        help="Vision model to use (default: qwen2.5vl:7b)"
    )
    parser.add_argument(
        "--project",
        action="store_true",
        help="Configure at project level (.gemini/settings.json) instead of user level"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    print("🔧 Gemini CLI Vision Server Configuration Setup")
    print("=" * 60)
    print()

    # Check prerequisites
    print("Checking prerequisites...")
    warnings = verify_prerequisites()

    if warnings:
        print()
        for warning in warnings:
            print(warning)
        print()

        if any("❌" in w for w in warnings):
            print("❌ Critical issues found. Please resolve them before proceeding.")
            return 1

    print("✅ Prerequisites check complete")
    print()

    # Get configuration path
    config_path = get_gemini_config_path(project_level=args.project)
    config_scope = "project" if args.project else "user"
    print(f"📄 Gemini config location ({config_scope}-level): {config_path}")

    # Load existing config
    config = load_gemini_config(config_path)

    # Add vision server
    config = add_vision_server_to_config(
        config,
        log_level=args.log_level,
        ollama_url=args.ollama_url,
        vision_model=args.vision_model
    )

    # Display configuration
    print()
    print("📋 Configuration to be added/updated:")
    print("-" * 60)
    print(json.dumps(config["mcpServers"]["local-vision"], indent=2))
    print("-" * 60)
    print()

    if args.dry_run:
        print("🔍 Dry run mode - no changes made")
        return 0

    # Backup existing config
    if config_path.exists():
        backup_path = config_path.with_suffix(".json.backup")
        with open(config_path, "r") as f:
            backup_content = f.read()
        with open(backup_path, "w") as f:
            f.write(backup_content)
        print(f"💾 Backup created: {backup_path}")

    # Save configuration
    save_gemini_config(config_path, config)

    print()
    print("🎉 Setup complete!")
    print()
    print("Next steps:")
    print("1. Ensure Ollama is running: ollama serve")
    print(f"2. Ensure vision model is installed: ollama pull {args.vision_model}")
    print("3. Restart Gemini CLI to load the new configuration")
    print()
    print("To verify the server is available in Gemini CLI:")
    print("  Type 'gemini mcp list' to see configured MCP servers")
    print()
    print("To test the server manually:")
    print(f"  python {get_vision_server_path()}")
    print()
    print("Usage in Gemini CLI:")
    print("  - The 'see_image' tool will be available automatically")
    print("  - Example: 'Use the see_image tool to analyze test.jpg'")
    print()
    print(f"Note: trust=false means Gemini will ask for confirmation before calling the tool.")
    print(f"      To auto-approve, edit {config_path} and set 'trust': true")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
