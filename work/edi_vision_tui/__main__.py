#!/usr/bin/env python3
"""
Main entry point for EDI Vision Subsystem TUI Application.
This file demonstrates all the functionality in a single executable script.
"""

import sys
import os
from pathlib import Path


def main():
    """Main entry point - demonstrates the functionality"""
    print("EDI Vision Subsystem TUI Application")
    print("=" * 50)
    print()
    
    # Show available functionality
    print("Available modules in this implementation:")
    print("1. Mask Generation (using SAM-inspired approach)")
    print("2. Mock Editing (simulates image editing without external dependencies)")
    print("3. Change Detection (compares images inside/outside masks)")
    print("4. System Testing (verifies detection of wrong outputs)")
    print("5. Statistical Testing (performance, consistency, stability, utility)")
    print()
    
    print("Usage examples:")
    print()
    
    # Explain how to run the main application
    print("To run the main application:")
    print("  python runnable_app.py --image @images/IP.jpeg --prompt \"edit the blue tin sheds to green\"")
    print()
    
    # Explain how to run statistical tests
    print("To run statistical tests:")
    print("  python statistical_test.py --image @images/IP.jpeg")
    print()
    
    print("The system includes:")
    print("- Keyword extraction from natural language prompts")
    print("- Mask generation around identified entities")
    print("- Mock editing that simulates the editing process")
    print("- Change detection to verify edit quality")
    print("- System testing to verify detection of wrong outputs")
    print("- Statistical validation of performance and consistency")
    print()
    
    print("Files created:")
    print("- app.py: Textual TUI application")
    print("- mask_generator.py: Mask generation functionality") 
    print("- mock_edit.py: Mock editing functionality")
    print("- change_detector.py: Change detection functionality")
    print("- system_test.py: System testing functionality")
    print("- runnable_app.py: One-click runnable application")
    print("- statistical_test.py: Statistical testing module")
    print()
    
    print("To test with the provided images:")
    print("$ cd /path/to/work/edi_vision_tui")
    print("$ python runnable_app.py --image ../../../images/IP.jpeg --prompt \"edit the blue tin sheds to green\"")
    print()
    
    print("For statistical testing:")
    print("$ python statistical_test.py --image ../../../images/IP.jpeg --output results.json")


if __name__ == "__main__":
    main()