# EDI: Edit with Intelligence - Documentation Overview

## Executive Summary

**Product Name**: EDI (Edit with Intelligence)  
**Version**: 1.0 POC  
**Target Release**: 24-hour development sprint  
**Development Model**: Solo developer, local-first, open-source stack

EDI is a conversational AI image editing assistant that bridges the gap between human intention and technical image manipulation. Unlike traditional diffusion-based editors that require prompt engineering expertise, EDI acts as an intelligent intermediary that SEEs, THINKS, ASKS, and LISTENS before generating optimal editing instructions for downstream tools like ComfyUI.

## Architecture Overview

### System Context

```
┌─────────────────────────────────────────────────────────────┐
│                         USER                                │
│                    (Terminal Session)                       │
└───────────────────────────┬─────────────────────────────────┘
                            │
                    ┌───────▼────────┐
                    │   EDI (TUI)    │
                    │   Textual App  │
                    └───────┬────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌───────▼────────┐   ┌──────▼──────┐
│ Vision         │  │ Reasoning      │   │ Orchestrator│
│ Subsystem      │  │ Subsystem      │   │  (DSpy)     │
│ SAM+CLIP       │  │ Ollama LLMs    │   │             │
└───────┬────────┘  └───────┬────────┘   └──────┬──────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                    ┌───────▼────────┐
                    │   ComfyUI      │
                    │  (External)    │
                    └────────────────┘
```

## System Components

This documentation is organized into the following main components:

- [TUI Layer](./edi/tui_layer.md)
- [Vision Subsystem](./edi/vision/vision_subsystem.md)
- [Reasoning Subsystem](./edi/reasoning_subsystem.md)
- [Orchestrator](./edi/orchestrator.md)
- [Storage Layer](./edi/storage_layer.md)
- [Integration Layer](./edi/integration_layer.md)

## Core Innovation: Conversational Image Understanding

EDI implements a **See → Think → Ask → Listen → Execute** loop:

1. **SEE**: Analyze image structure using SAM 2.1 + CLIP (deterministic object detection)
2. **THINK**: Reason about user intent using local LLMs via DSpy orchestration
3. **ASK**: Engage in clarifying dialogue when ambiguity detected
4. **LISTEN**: Incorporate user feedback into refined understanding
5. **EXECUTE**: Generate optimal positive/negative prompts for ComfyUI

### Key Differentiator

**Not an image editor** - EDI is a **software operator** that:

- Understands image composition like a graphic designer
- Translates casual language to technical specifications
- Self-corrects mistakes through validation loops
- Explains decisions and solicits feedback

### Technology Foundation

**Local-First Architecture**:

- No cloud dependencies (all inference on RTX 3060 12GB)
- Models: Ollama-served LLMs (qwen3:8b for reasoning, gemma3:4b for vision)
- Image Analysis: SAM 2.1 + OpenCLIP (ViT-B/32)
- UI: Textual TUI (terminal-native, keyboard-driven)
- Orchestration: DSpy for structured LLM interactions