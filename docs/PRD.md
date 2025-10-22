# PRD

## Executive Summary

**Product Name**: EDI (Edit with Intelligence)  
**Version**: 1.0 POC  
**Target Release**: 24-hour development sprint  
**Development Model**: Solo developer, local-first, open-source stack

EDI is a conversational AI image editing assistant that bridges the gap between human intention and technical image manipulation. Unlike traditional diffusion-based editors that require prompt engineering expertise, EDI acts as an intelligent intermediary that SEEs, THINKS, ASKS, and LISTENS before generating optimal editing instructions for downstream tools like ComfyUI.

## Problem Statement

### Current State of AI Image Editing

**Critical Pain Points**:

- Users must learn prompt engineering to get consistent results
- Black-box systems provide no visibility into decision-making
- No mechanism for iterative refinement through conversation
- Tools assume users know what they want upfront
- Results fail silently without explanation

**Example Failure Scenario**:

```bash
User: "Make the background more dramatic"
Traditional Tool: [generates random dramatic effect]
User: "No, I meant darker clouds, not a sunset"
Traditional Tool: [starts from scratch, ignores context]
```

### Target User

**Primary Persona**: Creative professionals who understand visual concepts but lack prompt engineering skills

- **Current Workflow**: Trial-and-error with AI tools (30+ attempts common)
- **Pain Point**: Cannot articulate technical requirements for desired outcome
- **Need**: Conversational partner that helps refine intent before execution

## Solution Overview

### Core Innovation: Conversational Image Understanding

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

## Core Features (POC Scope)

### Feature 1: Intelligent Image Analysis

**User Story**: As a user, I want EDI to understand my image's composition so it knows what can be safely edited.

**Functionality**:

- Upload image via file path
- SAM 2.1 generates candidate object masks
- CLIP ranks masks by semantic relevance to common edit targets
- Display structured breakdown: "I see 3 main regions: sky (top 40%), building (center), grass (bottom)"

**Acceptance Criteria**:

- Analysis completes in <5 seconds on test hardware
- Detects 85%+ of salient objects (human validation on 20 test images)
- Groups related masks (e.g., "person" mask includes face, torso, limbs)

### Feature 2: Conversational Intent Clarification

**User Story**: As a user, I want EDI to ask questions when my request is ambiguous rather than guessing.

**Functionality**:

- Parse naive prompt for ambiguity markers ("dramatic", "better", "more")
- Use DSpy ChainOfThought to generate clarifying questions
- Present questions in Textual UI with keyboard shortcuts (1-5 for options)
- Incorporate answers into refined understanding

**Example Interaction**:

```bash
User: "Make it more dramatic"
EDI: "I notice your image has a sky (40%) and a building (60%). 
      Which should I focus on?
      [1] Darken the sky with storm clouds
      [2] Add contrast to the building
      [3] Both"
User: [presses 1]
EDI: "Got it. Darkening sky while preserving building."
```

**Acceptance Criteria**:

- Ambiguous prompts trigger questions 90%+ of the time
- Questions are answerable via single keypress
- System remembers choices within session

### Feature 3: Multi-Iteration Prompt Refinement

**User Story**: As a user, I want EDI to refine its understanding through multiple passes before committing to an edit.

**Functionality**:

- Generate initial positive/negative prompts using DSpy
- Run 3-iteration refinement loop (DSpy Refine module)
- Show progressive improvements in TUI: "Refinement 1/3... 2/3... 3/3"
- Display final prompts for user review before sending to ComfyUI

**Refinement Strategy**:

```bash
Iteration 0 (Initial):
  Positive: "dramatic sky, dark clouds"
  Negative: "bright sky, clear weather"

Iteration 1 (Context-aware):
  Positive: "stormy cumulus clouds, volumetric lighting, overcast sky, 
            moody atmosphere, preserve building detail"
  Negative: "sunny, blue sky, lens flare, building modifications"

Iteration 2 (Technical):
  Positive: "photorealistic storm clouds, cumulonimbus formation, 
            diffuse natural lighting, 8k, preserve foreground subjects"
  Negative: "cartoon clouds, oversaturation, building color changes, 
            artifacts, blurry"
```

**Acceptance Criteria**:

- Each iteration increases token diversity by 20%+
- Final prompts include preservation constraints (e.g., "preserve building")
- User can approve or request additional refinement

### Feature 4: Validation Loop with User Preference Learning

**User Story**: As a user, I want EDI to check if the edit matches my intent and learn from my corrections.

**Functionality**:

- After ComfyUI returns edited image, re-analyze with SAM + CLIP
- Compare before/after entity structures (delta analysis)
- Ask user: "Did this match your vision? [Y]es / [N]o / [P]artial"
- If No/Partial, use feedback to refine prompts and retry (max 3 attempts)

**Validation Metrics**:

```bash
Alignment Score = (
  0.4 × Entities Preserved Correctly +
  0.4 × Intended Changes Applied +
  0.2 × (1 - Unintended Changes)
)
```

**Acceptance Criteria**:

- Validation completes in <8 seconds
- Score >0.8 auto-accepts, 0.6-0.8 asks user, <0.6 auto-retries
- User feedback incorporated into next attempt

### Feature 5: Multi-Image Comparison & Best-of-N Selection

**User Story**: As a user, I want to see multiple variations and pick the best parts from each.

**Functionality**:

- Generate 3 prompt variations using DSpy BestOfN
- Submit all to ComfyUI in parallel (or sequential if GPU limited)
- Display results in grid view (Textual DataTable widget)
- Allow selection: "Which sky looks best? [1/2/3/None]"
- Composite selected regions into final output

**Technical Approach**:

- Use SAM masks to identify region boundaries
- Blend selected regions with Poisson blending (seamless transitions)
- Show preview before finalizing

**Acceptance Criteria**:

- Generates 3 distinct variations (prompts differ by >30% tokens)
- Grid view fits in 80×24 terminal minimum
- Compositing preserves image quality (no compression artifacts)

## User Flows

### Primary Flow: Single Image Edit

```bash
1. Launch: $ edi edit photo.jpg "make sky dramatic"
   
2. Analysis Phase:
   [EDI] Analyzing image... (2.3s)
   [EDI] I see: Sky (40%, #87CEEB blue), Building (55%, #8B7355 brown), 
                Grass (5%, #228B22 green)

3. Intent Clarification:
   [EDI] Your request "dramatic sky" could mean:
         [1] Darker storm clouds
         [2] Sunset/sunrise colors  
         [3] HDR enhancement
         Which matches your vision?
   
   [User] 1
   [EDI] Got it. Focusing on storm clouds in sky region.

4. Refinement:
   [EDI] Refining prompts... 
         [████████████████████████] 100% (3 iterations)
         
   [EDI] Final prompts:
         Positive: "dramatic storm clouds, cumulonimbus, dark gray sky, 
                    volumetric lighting, moody atmosphere, photorealistic,
                    preserve building and grass"
         Negative: "blue sky, sunny, bright, lens flare, building changes"
         
         Proceed? [Y]es / [E]dit prompts / [C]ancel

5. Execution:
   [User] Y
   [EDI] Sending to ComfyUI... (45s)
   [EDI] Edit complete! Opening preview...

6. Validation:
   [Side-by-side comparison in TUI]
   [EDI] Alignment score: 0.87 (87% match)
         - Sky: Changed as intended ✓
         - Building: Preserved ✓  
         - Grass: Preserved ✓
         
         Accept this result? [Y]es / [R]etry / [T]weak prompts

7. Completion:
   [User] Y
   [EDI] Saved to: photo_edited_001.jpg
         Session saved for learning.
```

### Alternative Flow: Multi-Variation Selection

```bash
1. Launch: $ edi edit photo.jpg "improve lighting" --variations 3

2. Analysis: [same as above]

3. Intent Clarification:
   [EDI] "Improve lighting" is broad. Generating 3 interpretations:
         - Variation A: Increase brightness
         - Variation B: Add golden hour warmth
         - Variation C: Enhance contrast
         
         Generating all 3... (120s)

4. Selection:
   [Grid view in Textual]
   ┌─────────┬─────────┬─────────┐
   │    A    │    B    │    C    │
   │ Bright  │  Warm   │Contrast │
   │[Preview]│[Preview]│[Preview]│
   └─────────┴─────────┴─────────┘
   
   [EDI] Which looks best? [A/B/C/None]
   [User] B
   
   [EDI] Would you like to combine elements?
         [1] Just B
         [2] B's sky + C's building
         [3] Custom blend
   
   [User] 2
   [EDI] Blending B's sky with C's building... (8s)
   
5. Completion: [same as above]
```

## Non-Functional Requirements

### Performance

**Response Time Targets** (RTX 3060 12GB, 32GB RAM):

- Image analysis (SAM + CLIP): <5 seconds
- LLM reasoning (qwen3:8b): <2 seconds per prompt
- Prompt refinement (3 iterations): <6 seconds total
- Validation (re-analysis): <8 seconds

**Throughput**:

- Support 1 edit session at a time (single GPU limitation)
- Queue additional requests with estimated wait time

### Reliability

**Error Handling**:

- Model loading failures: Fallback to smaller models (gemma3:4b)
- Out of memory: Auto-reduce image resolution to 1024px max dimension
- ComfyUI unavailable: Provide prompts for manual use
- Corrupted images: Detect and reject with helpful message

**Data Integrity**:

- Auto-save conversation history to JSON
- Preserve original images (never overwrite)
- Atomic file writes (no partial outputs)

### Usability

**Terminal Compatibility**:

- Works in 80×24 minimum terminal size
- Graceful degradation on smaller screens (vertical scrolling)
- No Unicode emojis (ASCII-safe for SSH sessions)

**Keyboard Navigation**:

- All actions accessible via keyboard (no mouse required)
- Vim-style shortcuts where appropriate (j/k for navigation)
- Quick-access keys for common actions (Q=quit, R=retry, H=help)

**Feedback**:

- Progress indicators for long operations (>2s)
- Color-coded messages: info (cyan), warning (yellow), error (red), success (green)
- Explain-as-you-go narration of internal reasoning

### Maintainability

**Code Quality**:

- Type hints for all functions (Python 3.10+)
- Docstrings following Google style
- <200 lines per module (enforced by linter)

**Testing**:

- Unit tests for core logic (pytest)
- Integration tests for DSpy pipelines
- Snapshot tests for TUI layouts

**Documentation**:

- README with 5-minute quickstart
- ARCHITECTURE.md explaining system design
- TROUBLESHOOTING.md for common issues

---
