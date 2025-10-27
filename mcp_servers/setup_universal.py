#!/usr/bin/env python3
"""
Universal setup script to configure multiple AI CLIs to use the local vision MCP server.

This script automatically detects installed CLIs and configures them all at once.
Supports: Claude Code, Qwen Code CLI, Gemini CLI

Usage:
    python setup_universal.py [--log-level LEVEL] [--cli CLI1,CLI2,...]

Options:
    --log-level: Set logging level (ERROR, WARNING, INFO, DEBUG, TRACE)
                 Default: INFO
    --cli:       Comma-separated list of CLIs to configure
                 Options: claude, qwen, gemini, all
                 Default: all (auto-detect and configure all found CLIs)
    --project:   For Qwen/Gemini, use project-level config instead of user-level
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def get_vision_server_path() -> Path:
    """Get absolute path to vision_server.py"""
    return Path(__file__).parent / "vision_server.py"


def get_claude_config_path() -> Tuple[Path, bool]:
    """Get Claude config path and whether it exists."""
    if sys.platform == "win32":
        config_dir = Path.home() / "AppData" / "Roaming" / "Claude"
    elif sys.platform == "darwin":
        config_dir = Path.home() / "Library" / "Application Support" / "Claude"
    else:
        config_dir = Path.home() / ".config" / "claude"

    config_path = config_dir / "claude_desktop_config.json"
    return config_path, config_path.exists()


def get_qwen_config_path(project_level: bool = False) -> Tuple[Path, bool]:
    """Get Qwen config path and whether it exists."""
    if project_level:
        config_dir = Path.cwd() / ".qwen"
    else:
        config_dir = Path.home() / ".qwen"

    config_path = config_dir / "settings.json"
    return config_path, config_path.exists()


def get_gemini_config_path(project_level: bool = False) -> Tuple[Path, bool]:
    """Get Gemini config path and whether it exists."""
    if project_level:
        config_dir = Path.cwd() / ".gemini"
    else:
        config_dir = Path.home() / ".gemini"

    config_path = config_dir / "settings.json"
    return config_path, config_path.exists()


def detect_installed_clis(project_level: bool = False) -> Dict[str, Tuple[Path, bool]]:
    """Detect which CLIs are installed/configured."""
    return {
        "claude": get_claude_config_path(),
        "qwen": get_qwen_config_path(project_level),
        "gemini": get_gemini_config_path(project_level)
    }


def load_config(config_path: Path) -> dict:
    """Load existing configuration or return empty dict."""
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def save_config(config_path: Path, config: dict):
    """Save configuration to file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def create_vision_server_config(
    log_level: str = "INFO",
    ollama_url: str = "http://localhost:11434",
    vision_model: str = "qwen2.5vl:7b",
    timeout: int = 120000,
    trust: bool = False
) -> dict:
    """Create vision server configuration dictionary."""
    vision_server_path = get_vision_server_path()

    return {
        "command": sys.executable,
        "args": [str(vision_server_path)],
        "env": {
            "LOG_LEVEL": log_level.upper(),
            "OLLAMA_BASE_URL": ollama_url,
            "VISION_MODEL": vision_model
        },
        "timeout": timeout,
        "trust": trust
    }


def configure_cli(
    cli_name: str,
    config_path: Path,
    log_level: str,
    dry_run: bool = False
) -> bool:
    """Configure a specific CLI."""
    try:
        # Load existing config
        config = load_config(config_path)

        # Ensure mcpServers key exists
        if "mcpServers" not in config:
            config["mcpServers"] = {}

        # Add/update vision server
        config["mcpServers"]["local-vision"] = create_vision_server_config(log_level=log_level)

        if not dry_run:
            # Backup existing config
            if config_path.exists():
                backup_path = config_path.with_suffix(".json.backup")
                with open(config_path, "r") as f:
                    backup_content = f.read()
                with open(backup_path, "w") as f:
                    f.write(backup_content)
                print(f"   💾 Backup: {backup_path}")

            # Save configuration
            save_config(config_path, config)
            print(f"   ✅ Configured: {config_path}")
        else:
            print(f"   🔍 Would configure: {config_path}")

        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def verify_prerequisites() -> List[str]:
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
        description="Universal setup for vision MCP server across multiple AI CLIs"
    )
    parser.add_argument(
        "--log-level",
        choices=["ERROR", "WARNING", "INFO", "DEBUG", "TRACE"],
        default="INFO",
        help="Logging level for the vision server (default: INFO)"
    )
    parser.add_argument(
        "--cli",
        default="all",
        help="Comma-separated list of CLIs to configure: claude,qwen,gemini,all (default: all)"
    )
    parser.add_argument(
        "--project",
        action="store_true",
        help="For Qwen/Gemini, use project-level config instead of user-level"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    print("🔧 Universal Vision MCP Server Configuration Setup")
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

    # Parse CLI selection
    if args.cli.lower() == "all":
        selected_clis = ["claude", "qwen", "gemini"]
    else:
        selected_clis = [cli.strip().lower() for cli in args.cli.split(",")]

    # Detect installed CLIs
    print("Detecting installed CLIs...")
    detected = detect_installed_clis(project_level=args.project)

    print()
    for cli_name in ["claude", "qwen", "gemini"]:
        config_path, exists = detected[cli_name]
        status = "✅ Found" if exists else "➕ Will create"
        if cli_name in selected_clis:
            print(f"  {status}: {cli_name.capitalize():7} → {config_path}")
        else:
            print(f"  ⏭️  Skip:   {cli_name.capitalize():7} (not selected)")

    print()

    # Configure selected CLIs
    print("Configuring CLIs...")
    print("-" * 60)

    success_count = 0
    for cli_name in selected_clis:
        config_path, _ = detected[cli_name]
        print(f"\n📋 {cli_name.capitalize()}:")

        if configure_cli(cli_name, config_path, args.log_level, args.dry_run):
            success_count += 1

    print()
    print("=" * 60)

    if args.dry_run:
        print("🔍 Dry run mode - no changes made")
        print()
    else:
        print(f"🎉 Setup complete! Configured {success_count}/{len(selected_clis)} CLIs")
        print()

    # Print next steps
    print("Next steps:")
    print("1. Ensure Ollama is running: ollama serve")
    print("2. Ensure vision model is installed: ollama pull qwen2.5vl:7b")
    print("3. Restart your AI CLI(s) to load the new configuration")
    print()

    if "claude" in selected_clis:
        print("Claude Code:")
        print("  - The 'see_image' tool will be available via MCP")
        print()

    if "qwen" in selected_clis:
        print("Qwen Code CLI:")
        print("  - Type /mcp to verify the server is loaded")
        print("  - Example: 'Use see_image to analyze test.jpg'")
        print()

    if "gemini" in selected_clis:
        print("Gemini CLI:")
        print("  - Type 'gemini mcp list' to verify configuration")
        print("  - Example: 'Use see_image to analyze test.jpg'")
        print()

    print("To test the server manually:")
    print(f"  python {get_vision_server_path()}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
