# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**EDI (Edit with Intelligence)** is a conversational AI image editing assistant that bridges human intention and technical image manipulation. Unlike traditional diffusion-based editors requiring prompt engineering expertise, EDI acts as an intelligent intermediary that SEEs, THINKS, ASKS, and LISTENS before generating optimal editing instructions for downstream tools like ComfyUI.

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

### Architecture Layers

The system follows a **5-layer architecture**:

1. **Vision Subsystem** (SAM 2.1 + OpenCLIP) - Image analysis, object detection, change detection
2. **Reasoning Subsystem** (Ollama + DSpy) - Intent understanding, prompt generation, validation
3. **Orchestrator** (DSpy) - Workflow coordination, DSpy pipelines, state management
4. **TUI Layer** (Textual) - User interaction, display, navigation
5. **Integration Layer** - ComfyUI API client, image I/O

## Development Workflow: Progressive Refinement Strategy

This project uses a **multi-stage development workflow** designed to reduce errors through isolation, experimentation, and progressive integration:

### Stage 1: `example_code/` - Technology Exploration

**Purpose**: Working reference implementations for new, uncommon, or critical technologies

**Contents**:
- `Image_analysis/` - SAM 2.1 + CLIP + YOLO integration examples
  - `advanced_mask_generator.py` - Production-quality mask generation
  - `comparison_analyzer.py` - Before/after image comparison
  - `adaptive_edit_validator.py` - Edit quality validation
  - `batch_analyzer.py` - Batch processing utilities
- `VLM_based_image_analysis/` - Vision-Language Model entity extraction
  - `entity_extractor.py` - Structured entity extraction from VLM
  - `entity_structurer.py` - Converting VLM responses to structured data
  - `stable_entity_extraction.py` - Robust extraction with retries
- `dspy_toys/` - DSpy pattern demonstrations
  - `dspy_text_RPG_game.py` - ChainOfThought and ReAct patterns
  - `dspy_finance_analyst.py` - Multi-signature orchestration
  - `dspy_code_generation.py` - Code generation with validation
  - `dspy_image.py` - DSpy with vision models
- `textual/` - Textual TUI examples
  - `calculator.py` - Widget composition and event handling
  - `code_browser.py` - File navigation and display
  - `dictionary.py` - API integration in TUI

**When to use**: When learning a new technology or validating a technical approach before integration. These are **standalone, fully functional examples** that demonstrate individual capabilities.

**Development approach**:
- Each example is self-contained and runnable
- Focus on demonstrating specific patterns or integrations
- No dependencies on other parts of EDI
- Well-commented to explain non-obvious decisions

### Stage 2: `work/` - Sandboxed Experimentation

**Purpose**: Agents work autonomously to build micro-features and gain practical understanding without affecting the main codebase

**Current contents**:
- `edi_vision_tui/` - Vision pipeline prototype
  - `app.py` - Adaptive mask generator (fully functional)
  - `pipeline/` - 5-stage mask generation system:
    - `stage1_clip/` - CLIP entity detection
    - `stage2_yolo/` - YOLO refinement
    - `stage3_sam/` - SAM integration
    - `stage4_validation/` - VLM validation
    - `stage5_feedback/` - Feedback processing
  - `change_detector.py` - Change detection implementation
  - `dspy_*.py` - DSpy experimentation files

**Agent autonomy in work/**:
- Agents can create new directories for feature experiments
- No supervision required for implementation iterations
- Focus on **making it work** rather than perfect architecture
- Rapid prototyping with immediate feedback
- Can break things without affecting other features

**When to use**:
- Implementing a feature for the first time to understand requirements
- Testing integration between 2-3 technologies
- Building practical understanding through hands-on experimentation
- Validating architectural decisions with working prototypes

**Example workflow**:
```bash
# Agent creates new experiment
mkdir work/dspy_intent_parsing
cd work/dspy_intent_parsing

# Build standalone feature
# - Test DSpy signatures
# - Validate with real prompts
# - Iterate on refinement strategy
# - Document lessons learned

# When working, graduate to builds/
```

### Stage 3: `builds/` - Integration & Testing

**Purpose**: Features from `work/` are integrated into the whole project architecture with comprehensive testing

**Structure** (to be created):
```
builds/
├── vision/           # Vision subsystem integration
├── reasoning/        # Reasoning subsystem integration
├── orchestration/    # Orchestration integration
├── ui/              # TUI integration
├── integration/     # ComfyUI integration
└── tests/
    ├── unit/        # Unit tests for each component
    ├── integration/ # Integration tests between subsystems
    └── e2e/         # End-to-end workflow tests
```

**Development approach**:
- Follow architecture defined in `docs/HLD.md` and `docs/LLD.md`
- Implement proper module structure and imports
- Add comprehensive unit tests (pytest)
- Integration tests between subsystems
- E2E tests for complete workflows
- Performance benchmarking against targets

**Testing requirements**:
- Vision subsystem: 90%+ coverage (critical path)
- Reasoning subsystem: 85%+ coverage
- TUI widgets: 70%+ coverage (snapshot tests)
- All acceptance criteria from `docs/PRD.md` must pass

**When to use**: After a feature works in `work/`, integrate it here with proper architecture and testing before final release.

### Stage 4: `src/` - Production Release

**Purpose**: Final product directory for alpha build after big chunks are complete and ready for user testing

**Entry criteria**:
- All core features implemented in `builds/`
- Unit tests passing (>85% coverage)
- Integration tests passing
- E2E tests demonstrating complete workflows
- Performance meets targets from `docs/PRD.md`
- Documentation complete

**Structure** (from `docs/LLD.md`):
```
src/edi/
├── __init__.py
├── __main__.py
├── cli.py
├── config.py
├── app.py
├── vision/          # SAM, CLIP, scene analysis, change detection
├── reasoning/       # Intent parser, prompt generator, validator
├── orchestration/   # Pipeline, variations, compositor, state
├── integration/     # ComfyUI client, workflow manager
├── storage/         # Database, migrations
├── ui/              # Screens, widgets, styles
├── utils/           # Image ops, logging, validators
└── commands/        # CLI commands (edit, setup, doctor, clear)
```

**Migration process**:
1. Copy tested code from `builds/` to `src/`
2. Verify all tests still pass in new location
3. Update imports and dependencies
4. Run full test suite
5. Perform user acceptance testing
6. Tag release version

**Current status**: Empty - implementation has not reached this stage yet.

### Development Progression Summary

```
example_code/          work/                builds/               src/
   [Learn]      →   [Experiment]    →    [Integrate]     →   [Release]

- Reference code   - Rapid prototyping  - Proper architecture - Production code
- Isolated demos   - Agent autonomy     - Comprehensive tests  - User testing
- Technology proof - Practical learning - Performance tuning   - Version control
- No dependencies  - Can break things   - Quality assurance    - Deployment ready
```

## Local Vision Tool (MCP Server)

### Overview

A custom MCP server that provides **local vision analysis** using Ollama's vision models (`qwen2.5vl:7b`), allowing Claude Code to "SEE" images **without consuming cloud tokens**.

**Location**: `mcp_servers/vision_server.py`

### Features

- 🔍 **Image analysis** using local Ollama vision models
- 🚫 **Zero cloud tokens** - completely local processing
- 📊 **Structured responses** with detailed error handling
- 📝 **Multiple log levels** (TRACE, DEBUG, INFO, WARNING, ERROR)
- ✅ **Health check** resource to verify Ollama availability
- 🎯 **Context-aware logging** for debugging

### Quick Start

```bash
# 1. Test the vision server
cd mcp_servers
python test_vision.py ../images/IP.jpeg "Describe this image"

# 2. Configure Claude Code to use it
python setup_claude_config.py --log-level INFO

# 3. Restart Claude Code
# The vision tool will now be available as 'see_image'
```

### Tool: `see_image`

Analyze images using local vision model without cloud tokens.

**Parameters**:
- `image_path` (string, required): Path to image file (JPG, PNG, JPEG, WEBP, etc.)
- `prompt` (string, required): Question or request about the image

**Returns**: Structured dictionary with:
```json
{
  "success": true,
  "response": "Detailed description from vision model...",
  "model": "qwen2.5vl:7b",
  "image_path": "/absolute/path/to/image.jpg",
  "image_size_bytes": 113670,
  "image_name": "image.jpg"
}
```

**Example Usage in Claude Code**:

```python
# General description
result = await see_image(
    image_path="./test.jpg",
    prompt="Describe this image in detail"
)

# Structured entity detection (for EDI vision pipeline)
result = await see_image(
    image_path="./input.jpg",
    prompt="""List all distinct regions in this image. For each region, provide:
    - label (object name)
    - dominant color (hex code)
    - approximate percentage of image area
    - spatial location (top/center/bottom, left/center/right)
    Return as JSON array."""
)

# Validation (for EDI edit validation)
result = await see_image(
    image_path="./mask_overlay.jpg",
    prompt="Does the red mask correctly cover the blue roof? Answer yes or no and explain."
)
```

### Integration with EDI Vision Pipeline

The vision tool supports EDI's multi-stage pipeline:

1. **Pre-Analysis** (before expensive SAM/CLIP):
   - Quick overview of image contents
   - Identify entities mentioned in user prompt
   - Guide which regions to focus SAM processing on

2. **Validation** (Stage 4):
   - Verify generated masks match user intent
   - Check if correct entities are covered
   - Provide feedback for iterative refinement

3. **Intent Parsing** (Reasoning Subsystem):
   - Help DSpy understand which entities user refers to
   - Disambiguate when multiple similar objects exist
   - Provide context for clarifying questions

4. **Change Detection** (Post-Edit):
   - Validate if edits match expected changes
   - Describe differences between before/after
   - Calculate semantic alignment beyond pixel differences

### Configuration

**Claude Config**: `~/.config/claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "local-vision": {
      "command": "/path/to/.venv/bin/python",
      "args": ["/path/to/mcp_servers/vision_server.py"],
      "env": {
        "LOG_LEVEL": "INFO",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "VISION_MODEL": "qwen2.5vl:7b"
      }
    }
  }
}
```

**Log Levels**:
- `ERROR` - Only errors
- `WARNING` - Warnings and errors
- `INFO` - General information (default, recommended)
- `DEBUG` - Detailed debugging information
- `TRACE` - Maximum verbosity

Change log level: `python setup_claude_config.py --log-level DEBUG`

### Performance

**Typical response times** (RTX 3060 12GB):
- Small images (512x512): 3-5 seconds
- Medium images (1024x1024): 5-10 seconds
- Large images (2048x2048): 10-15 seconds

**Memory usage**:
- Vision model (qwen2.5vl:7b): ~6 GB VRAM
- Per request: ~500 MB RAM

### Troubleshooting

See `mcp_servers/README.md` for detailed troubleshooting guide.

**Common issues**:
- "Cannot connect to Ollama": Run `ollama serve`
- "Vision model not available": Run `ollama pull qwen2.5vl:7b`
- High latency: Reduce image size if very large (>4096px)

## Development Commands

### Environment Setup
```bash
# Using uv (recommended, managed by lock file)
uv venv
source .venv/bin/activate  # or .venv/Scripts/activate on Windows
uv pip install -e .

# Or using pip
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Running Current Implementations

```bash
# Main entry point (minimal placeholder)
python main.py

# Work: Vision subsystem prototype
python -m work.edi_vision_tui

# Work: Adaptive mask generator (fully functional)
cd work/edi_vision_tui
python app.py --image ../../images/IP.jpeg --prompt "change orange roofs to blue" --output mask.png --apply-to-image --verbose

# Example: Advanced mask generator
cd example_code/Image_analysis
python advanced_mask_generator.py --image ../../images/IP.jpeg --prompt "edit the blue tin sheds to green"

# Example: DSpy RPG game (demonstrates ChainOfThought + ReAct)
cd example_code/dspy_toys
python dspy_text_RPG_game.py

# Example: Textual TUI demos
cd example_code/textual
python calculator.py
python code_browser.py
```

### Testing (when implemented in builds/)
```bash
# Run all tests
pytest builds/tests/

# Run with coverage
pytest builds/tests/ --cov=builds --cov-report=html

# Run specific subsystem
pytest builds/tests/unit/test_vision_subsystem.py -v

# Run integration tests
pytest builds/tests/integration/ -v

# Run E2E tests
pytest builds/tests/e2e/ -v
```

## Critical Architecture Concepts

### DSpy Integration Pattern (MOST IMPORTANT)

**DSpy is used for ALL mission-critical LLM reasoning tasks**. This is the foundational architectural decision.

#### When to Use DSpy vs Raw Ollama

- **Use DSpy** (dspy.ChainOfThought, dspy.Refine, dspy.BestOfN) for:
  - Intent parsing from ambiguous user prompts
  - Multi-iteration prompt refinement (3 stages)
  - Validation and quality assessment
  - Generating multiple prompt variations
  - Any task requiring **guided, deterministic, structured LLM behavior**
  - Information extraction where reliability > latency

- **Use Ollama directly** for:
  - Simple VLM validation calls
  - Quick one-off queries without structured outputs
  - Basic tasks where DSpy overhead isn't justified

- **Use Ultralytics (SAM/YOLO) and OpenCLIP** for:
  - All vision tasks (not LLM tasks)
  - Object detection, segmentation, semantic labeling

#### Core DSpy Modules to Implement

From `docs/HLD.md` and `docs/edi_decomposed/`:

1. **IntentParser** (`dspy.ChainOfThought`)
   ```python
   class ParseIntent(dspy.Signature):
       """Extract structured intent from casual user prompt."""
       naive_prompt = dspy.InputField(desc="User's conversational edit request")
       scene_analysis = dspy.InputField(desc="JSON of detected entities and layout")

       target_entities = dspy.OutputField(desc="Comma-separated list of entity IDs to edit")
       edit_type = dspy.OutputField(desc="One of: color, style, add, remove, transform")
       confidence = dspy.OutputField(desc="Float 0-1 indicating clarity of intent")
       clarifying_questions = dspy.OutputField(desc="JSON array of questions if confidence <0.7")
   ```
   - Detects ambiguity (confidence < 0.7) and generates clarification options
   - Example: "make it dramatic" → asks user to choose between storm clouds, sunset, or HDR

2. **PromptGenerator** (`dspy.ChainOfThought` + `dspy.Refine`)
   ```python
   class GenerateBasePrompt(dspy.Signature):
       naive_prompt = dspy.InputField()
       scene_analysis = dspy.InputField()
       target_entities = dspy.InputField()
       edit_type = dspy.InputField()

       positive_prompt = dspy.OutputField(desc="Technical prompt for desired changes")
       negative_prompt = dspy.OutputField(desc="Technical prompt for things to avoid")

   class RefinePrompt(dspy.Signature):
       naive_prompt = dspy.InputField()
       previous_positive = dspy.InputField()
       previous_negative = dspy.InputField()
       refinement_goal = dspy.InputField(
           desc="E.g., 'add technical quality terms', 'strengthen preservation constraints'"
       )

       refined_positive = dspy.OutputField()
       refined_negative = dspy.OutputField()
       improvement_explanation = dspy.OutputField()
   ```
   - 3-iteration refinement process:
     - Iteration 1: Add preservation constraints ("preserve building", "maintain foreground")
     - Iteration 2: Increase technical specificity ("photorealistic", "volumetric lighting", "8k")
     - Iteration 3: Add quality/style modifiers ("cumulonimbus formation", "diffuse lighting")
   - Each iteration increases token diversity by 20%+

3. **EditingPipeline** (`dspy.Module`)
   ```python
   class EditingPipeline(dspy.Module):
       def __init__(self):
           self.analyzer = VisionSubsystem()
           self.intent_parser = dspy.ChainOfThought(ParseIntent)
           self.prompt_generator = dspy.ChainOfThought(GenerateBasePrompt)
           self.prompt_refiner = dspy.Refine(RefinePrompt, N=3, reward_fn=prompt_quality_score)
           self.validator = dspy.ChainOfThought(ValidateEdit)

       def forward(self, image_path: str, naive_prompt: str):
           # Stage 1: Analyze image
           scene = self.analyzer.analyze(image_path)

           # Stage 2: Parse intent
           intent = self.intent_parser(naive_prompt=naive_prompt, scene_analysis=scene.to_json())

           # Stage 3: Clarify if ambiguous (confidence < 0.7)
           if intent.confidence < 0.7:
               user_input = self.ask_clarifying_questions(intent.clarifying_questions)
               intent = self.intent_parser(
                   naive_prompt=f"{naive_prompt}. User clarified: {user_input}",
                   scene_analysis=scene.to_json()
               )

           # Stage 4-5: Generate and refine prompts
           base_prompts = self.prompt_generator(...)
           final_prompts = self.prompt_refiner(...)

           return final_prompts
   ```
   - Orchestrates entire workflow: analyze → parse → clarify → generate → execute → validate
   - Handles retry logic (max 3 attempts)

4. **VariationGenerator** (`dspy.BestOfN`)
   - Generates 3 distinct prompt variations using different rollout IDs
   - Allows user to select best or blend regions from multiple results
   - Prompts must differ by >30% tokens to ensure diversity

### Vision Processing Pipeline

From `work/edi_vision_tui/pipeline/` and `docs/HLD.md`:

**Multi-stage approach for precise mask generation**:

1. **Stage 1: CLIP Entity Detection** (`stage1_clip/clip_entity_detector.py`)
   - Semantic matching of target entities from user prompt
   - Encodes image regions and text descriptions with CLIP ViT-B/32
   - Returns candidate regions with confidence scores

2. **Stage 2: YOLO Refinement** (`stage2_yolo/yolo_refiner.py`)
   - Object detection for precise bounding boxes
   - Filters CLIP candidates using YOLO object classes
   - Improves spatial accuracy

3. **Stage 3: SAM Integration** (`stage3_sam/sam_integration.py`)
   - Pixel-perfect segmentation masks using SAM 2.1
   - Takes YOLO bounding boxes as prompts
   - Generates binary masks for editing

4. **Stage 4: VLM Validation** (`stage4_validation/validation_system.py`)
   - Uses Vision-Language Model to verify mask matches intent
   - Calls Ollama VLM with image + mask overlay
   - Returns validation score and feedback

5. **Stage 5: Feedback Processing** (`stage5_feedback/feedback_processor.py`)
   - Iterative refinement based on VLM feedback
   - Max 3 attempts with progressively refined parameters
   - Applies morphological operations to clean up masks

**Implementation in `work/edi_vision_tui/app.py`**:
- Fully functional adaptive mask generator
- Command: `python app.py --image input.jpg --prompt "change orange roofs to blue" --output mask.png --apply-to-image --verbose`
- Demonstrates complete pipeline with debug logging

### GPU Memory Management (CRITICAL)

**Hardware**: RTX 3060 12GB VRAM

**Peak VRAM Usage**:
```
SAM 2.1 Base (FP16):      ~3.5 GB
CLIP ViT-B/32:            ~0.5 GB
qwen3:8b (Ollama):        ~5.0 GB
ComfyUI SD model:         ~4.0 GB (external process)
─────────────────────────────────
Total:                    ~13 GB (EXCEEDS 12GB LIMIT!)
```

**Solution: Sequential model loading with explicit cleanup**:

```python
# 1. Load SAM, analyze image, unload SAM
sam_model = load_sam()
masks = sam_model(image)
del sam_model
torch.cuda.empty_cache()

# 2. Load CLIP, label masks, unload CLIP
clip_model = load_clip()
entities = clip_model(image, masks)
del clip_model
torch.cuda.empty_cache()

# 3. LLM inference (Ollama manages its own memory)
prompts = ollama_generate(intent, entities)

# 4. ComfyUI runs in separate process (dedicated GPU)
```

**Additional optimizations**:
- Use half precision (FP16): `model.half()`
- Pre-resize images >2048px to 2048px max dimension
- Cache models in memory between sessions (configurable)
- Skip fine-grained segmentation if mask <2% of image area

### Validation & Retry Loop

From `docs/PRD.md` Feature 4 and `docs/HLD.md` Section 3.2:

**All edits go through quality assessment**:

1. **Re-analyze edited image** with SAM+CLIP
2. **Compute delta** (preserved/modified/removed/added entities)
   - Match entities by spatial overlap (IoU > 0.5)
   - Compare color (ΔE2000 < 10), position (center shift < 5%), shape (IoU > 0.85)
3. **Calculate alignment score**:
   ```python
   alignment_score = (
       0.4 × (entities_preserved_correctly / total_to_preserve) +
       0.4 × (intended_changes_applied / total_intended) +
       0.2 × (1 - unintended_changes / total_entities)
   )
   ```
4. **Determine action**:
   - Score ≥ 0.8: Auto-accept ✓
   - Score 0.6-0.8: Ask user (Review) ⚠
   - Score < 0.6: Auto-retry with hints (max 3 attempts) ✗

**Retry hints generation**:
- If preserved entities changed: Strengthen negative prompt with entity names
- If intended changes weak: Boost positive prompt weight
- If unintended changes high: Add spatial constraints to prompts

### State Management & Persistence

From `docs/HLD.md` Section 5 and `docs/LLD.md`:

**Dual storage strategy**:

1. **SQLite Database** (`~/.edi/sessions.db`)
   ```sql
   -- Core tables
   sessions       -- session_id, image_path, naive_prompt, status, alignment_score
   prompts        -- session_id, iteration, positive_prompt, negative_prompt, quality_score
   entities       -- session_id, entity_id, label, confidence, bbox, mask_path, color, area
   validations    -- session_id, attempt_number, alignment_score, preserved/modified/unintended counts
   user_feedback  -- session_id, feedback_type, comments, rating
   ```
   - For querying history and learning from user preferences
   - Enables analytics on prompt effectiveness
   - Powers recommendation system (future feature)

2. **JSON State Files** (`~/.edi/sessions/<session_id>.json`)
   ```json
   {
     "session_id": "uuid",
     "current_stage": "refinement",
     "image_path": "/path/to/image.jpg",
     "naive_prompt": "make sky dramatic",
     "scene_analysis": { "entities": [...], "spatial_layout": "..." },
     "intent": { "target_entities": ["sky_0"], "edit_type": "style", "confidence": 0.85 },
     "prompts": {
       "iteration_0": { "positive": "...", "negative": "..." },
       "iteration_1": { "positive": "...", "negative": "..." },
       "final": { "positive": "...", "negative": "..." }
     },
     "edited_image_path": "/path/to/edited.jpg",
     "validation": { "score": 0.87, "delta": {...} }
   }
   ```
   - Auto-saves every 5 seconds
   - Enables resume functionality after crashes
   - Contains full conversation context for DSpy
   - Atomic writes (temp file + rename) to prevent corruption

### TUI Navigation Structure

From `docs/HLD.md` Section 4 and `docs/LLD.md`:

**Screen flow**:
```
HomeScreen
  → ImageUploadScreen (file selection, preview)
  → PromptInputScreen (user describes edit, entity selection)
  → AnalysisScreen (SAM+CLIP processing with progress bar)
  → ClarificationScreen (if confidence < 0.7, radio button questions)
  → RefinementScreen (shows 3 DSpy iterations with diff viewer)
  → ExecutionScreen (submits to ComfyUI, polls status)
  → ResultsScreen (side-by-side comparison, validation metrics, accept/retry/tweak)
  → [Retry loop or completion]

Alternative: MultiVariationScreen (grid of 3 variations, selection + optional blending)
```

**Key widgets** (see `example_code/textual/` for patterns):
- `ImageComparisonPane` - Side-by-side before/after with ANSI art
- `PromptDiffViewer` - Shows prompt evolution (green additions, red removals)
- `EntitySelectorList` - Checkbox list of detected entities
- `ValidationMetricsTable` - Color-coded alignment scores
- `ProgressSpinner` - Animated status for long operations

**Keyboard shortcuts**:
- Global: `Q` quit, `H` help, `B` back, `Ctrl+C` cancel
- Navigation: Arrow keys, Tab/Shift+Tab, Numbers (1-9) for quick selection
- Actions: `Enter` confirm, `E` edit, `R` retry, `A` accept, `V` view variations

## Performance Targets

From `docs/PRD.md` Section "Non-Functional Requirements":

**Response Time Targets** (RTX 3060 12GB, 32GB RAM):
- Image analysis (SAM + CLIP): **<5 seconds**
- LLM reasoning (qwen3:8b): **<2 seconds per prompt**
- Prompt refinement (3 iterations): **<6 seconds total**
- Validation (re-analysis): **<8 seconds**
- **Total pre-ComfyUI time: <15 seconds**

**Acceptance Criteria**:
- SAM detects 85%+ of salient objects (validation on 20 test images)
- CLIP labels match human annotations 80%+ of the time
- Intent parser detects ambiguity in 90%+ of vague prompts
- Prompt refinement improves token diversity by 20%+ per iteration
- Alignment score >0.8 for 80%+ of successful edits

## Configuration Management

From `docs/HLD.md` Section "Deployment Considerations":

**User config**: `~/.edi/config.yaml`

Key sections:
```yaml
models:
  reasoning_llm: "qwen3:8b"        # Options: qwen3:8b, mistral:7b, gemma3:4b
  vision_llm: "gemma3:4b"          # For VLM fallback
  sam_checkpoint: "sam2.1_b.pt"   # Options: sam2.1_t.pt (fast), sam2.1_b.pt, sam2.1_h.pt (slow)
  clip_model: "ViT-B/32"           # Options: ViT-B/32, ViT-L/14

performance:
  max_image_size: 2048             # Larger images downscaled
  use_half_precision: true         # FP16 for SAM/CLIP
  enable_model_caching: true       # Keep models in memory

prompts:
  refinement_iterations: 3         # DSpy refinement passes (1-5)
  quality_keywords:                # Auto-added to positive prompts
    - "high quality"
    - "8k"
    - "detailed"
  default_negative:                # Always in negative prompts
    - "low quality"
    - "blurry"
    - "artifacts"

comfyui:
  base_url: "http://localhost:8188"
  default_workflow: "img2img_default"
  timeout_seconds: 180
  poll_interval_seconds: 5

validation:
  alignment_threshold_accept: 0.8  # Auto-accept if ≥ this
  alignment_threshold_review: 0.6  # Ask user if in this range
  max_retry_attempts: 3            # Max retries for low scores

ui:
  theme: "dark"                    # or "light"
  animation_speed: "normal"        # slow, normal, fast
  show_debug_info: false           # Show timing info, model details
```

## External Dependencies & Services

**Required external services**:
- **Ollama** at `http://localhost:11434` with models:
  - `qwen3:8b` - Primary reasoning model
  - `gemma3:4b` - Fallback and vision tasks
  - Download: `ollama pull qwen3:8b && ollama pull gemma3:4b`
- **ComfyUI** at `http://localhost:8188` - For actual image editing
  - Optional for development (can test prompt generation without it)
  - See `docs/HLD.md` Section 6.1 for workflow templates

**Key Python dependencies** (from `pyproject.toml`):
```toml
dspy = ">=3.0.0"              # LLM orchestration (CRITICAL)
ultralytics = ">=8.3.220"     # SAM 2.1 and YOLO (CRITICAL)
open-clip-torch = ">=3.2.0"   # CLIP (CRITICAL)
textual = ">=6.4.0"           # TUI framework (CRITICAL)
torch = ">=2.0.0"             # PyTorch with CUDA 12.1
pydantic = ">=2.12.3"         # Data validation
pillow = ">=12.0.0"           # Image I/O
numpy = ">=2.3.4"             # Numerical operations
scikit-image = ">=0.25.2"     # Image metrics (ΔE2000, etc.)
```

## Reference Documentation Priority

When implementing a new component, **read in this order**:

1. **`docs/PRD.md`** - Product Requirements Document
   - Executive summary and problem statement
   - User stories and acceptance criteria for each feature
   - Non-functional requirements (performance, reliability, usability)
   - User flows (primary and alternative)

2. **`docs/HLD.md`** - High-Level Design
   - System architecture and component responsibilities
   - Data flow diagrams and processing pipelines
   - DSpy module specifications with example code
   - Technology stack details and performance optimization strategies

3. **`docs/LLD.md`** - Low-Level Design
   - Complete file structure with module descriptions
   - Function signatures and responsibilities
   - Module dependency graph
   - Development roadmap and testing checklist

4. **`docs/edi_decomposed/`** - Function-level specifications
   - Navigate from `index.md` to specific components
   - Each function has dedicated markdown with:
     - Related user story from PRD
     - Function signature and parameter descriptions
     - Step-by-step logic in plain English
     - Input/output data structures
     - Example implementation code
   - Use this as implementation guide when coding

5. **`example_code/`** - Working reference implementations
   - See specific technology in action
   - Copy patterns for similar functionality
   - Understand integration points

## Important Design Patterns

### Iterative Refinement Strategy

From `docs/PRD.md` Feature 3 and `docs/HLD.md` Section 2.2:

**DSpy prompt refinement follows a specific 3-iteration pattern**:

```
Iteration 0 (Initial):
  Positive: "dramatic sky, dark clouds"
  Negative: "bright sky, clear weather"

Iteration 1 (Context-aware + Preservation):
  Goal: "Add preservation constraints"
  Positive: "stormy cumulus clouds, volumetric lighting, overcast sky,
            moody atmosphere, preserve building detail"
  Negative: "sunny, blue sky, lens flare, building modifications"

Iteration 2 (Technical Specificity):
  Goal: "Increase technical quality terms"
  Positive: "photorealistic storm clouds, cumulonimbus formation,
            diffuse natural lighting, 8k, preserve foreground subjects"
  Negative: "cartoon clouds, oversaturation, building color changes,
            artifacts, blurry"

Iteration 3 (Quality/Style Modifiers):
  Goal: "Add quality and style modifiers"
  Positive: "professional photography, cinematic storm clouds,
            volumetric god rays, atmospheric perspective,
            photorealistic rendering, ultra detailed 8k"
  Negative: "amateur, low quality, compression artifacts,
            unrealistic lighting, overprocessed HDR"
```

Each iteration must increase token diversity by **20%+** while maintaining intent.

### Error Handling Philosophy

From `docs/HLD.md` and `docs/PRD.md`:

**Graceful degradation at every level**:

- **Model loading failures**: Fallback chain
  ```python
  try:
      model = load_model("qwen3:8b")
  except OOM:
      model = load_model("gemma3:4b")  # Smaller fallback
  except ConnectionError:
      raise RuntimeError("Ollama not running. Start with: ollama serve")
  ```

- **Out of memory**: Auto-reduce image resolution
  ```python
  if torch.cuda.OutOfMemoryError:
      logger.warning("OOM detected, reducing image resolution")
      image = resize_image(image, max_size=1024)  # Was 2048
      retry_analysis()
  ```

- **ComfyUI unavailable**: Provide prompts for manual use
  ```python
  if not comfyui_client.is_available():
      print("ComfyUI not available. Use these prompts manually:")
      print(f"Positive: {positive_prompt}")
      print(f"Negative: {negative_prompt}")
      save_prompts_to_file("prompts.txt")
  ```

- **Corrupted images**: Detect early and provide actionable feedback
  ```python
  try:
      image = load_image(path)
      validate_image_format(image)
  except FileNotFoundError:
      raise ValueError(f"Image not found: {path}")
  except UnidentifiedImageError:
      raise ValueError(f"Corrupted or unsupported format: {path}. Use JPG/PNG.")
  ```

- **All file writes are atomic**: Prevent partial outputs
  ```python
  temp_path = f"{output_path}.tmp"
  with open(temp_path, 'wb') as f:
      f.write(data)
  os.rename(temp_path, output_path)  # Atomic on Unix
  ```

### Local-First Privacy Philosophy

From `docs/PRD.md` and `docs/HLD.md` Section "Security & Privacy":

**Critical principle: No cloud dependencies. All processing runs locally.**

- Models downloaded once to `~/.edi/models/`
- SHA256 checksum verification for model integrity
- Images never transmitted outside localhost
- Session data stored locally in `~/.edi/sessions/`
- No telemetry or analytics sent to external servers
- User can purge all data: `edi clear --all` (requires confirmation)
- Sandboxed execution: No arbitrary code execution from prompts
- Config stored locally: `~/.edi/config.yaml`

## Directory Structure Summary

```
edi/
├── docs/                          # Complete specifications
│   ├── PRD.md                     # Product requirements
│   ├── HLD.md                     # High-level design
│   ├── LLD.md                     # Low-level design
│   ├── Instruction.md             # Decomposition instructions
│   └── edi_decomposed/            # Function-level specs
│
├── example_code/                  # Technology reference implementations
│   ├── Image_analysis/            # SAM + CLIP + YOLO working code
│   ├── VLM_based_image_analysis/  # VLM entity extraction
│   ├── dspy_toys/                 # DSpy pattern demonstrations
│   └── textual/                   # TUI widget examples
│
├── work/                          # Sandboxed experimentation (agent autonomy)
│   └── edi_vision_tui/            # Vision pipeline prototype (functional)
│       ├── app.py                 # Adaptive mask generator
│       └── pipeline/              # 5-stage mask generation
│
├── builds/                        # Integration + testing (to be created)
│   ├── vision/
│   ├── reasoning/
│   ├── orchestration/
│   ├── ui/
│   └── tests/
│
├── src/                           # Production release (empty, future)
│   └── edi/                       # Final package structure
│
├── images/                        # Test images
├── pyproject.toml                 # Dependencies
├── uv.lock                        # Dependency lock file
└── main.py                        # Entry point (placeholder)
```

## Development Guidelines for Agents

### When Working in `work/`

**You have full autonomy**:
- Create new directories for experiments
- Implement features without supervision
- Iterate rapidly based on results
- Can break things without affecting main codebase
- Focus on making it work, not perfect architecture
- Document lessons learned in README.md within experiment directory

**Example experiment structure**:
```
work/
└── feature_name/
    ├── README.md          # What you learned, what works, what doesn't
    ├── prototype.py       # Working implementation
    ├── tests.py           # Quick validation tests
    └── notes.md           # Implementation decisions and gotchas
```

### When Moving to `builds/`

**Follow architecture strictly**:
- Match file structure from `docs/LLD.md`
- Implement proper module imports
- Add type hints (Python 3.10+)
- Write comprehensive tests (pytest)
- Follow Google-style docstrings
- Keep modules <200 lines (enforced by linter)
- Use Pydantic models for data validation

### When Migrating to `src/`

**Production quality required**:
- All tests passing (>85% coverage)
- Performance meets targets
- Error handling comprehensive
- Documentation complete
- Config validated
- User acceptance testing passed

## Quick Reference: Key Files

**Currently working prototypes**:
- `work/edi_vision_tui/app.py` - Adaptive mask generator (fully functional)
- `example_code/Image_analysis/advanced_mask_generator.py` - Production-quality mask generation
- `example_code/dspy_toys/dspy_text_RPG_game.py` - DSpy ChainOfThought + ReAct demo
- `example_code/textual/code_browser.py` - Textual TUI navigation pattern

**Comprehensive specifications**:
- `docs/PRD.md` - What to build and why (user stories, acceptance criteria)
- `docs/HLD.md` - How to build it (architecture, algorithms, data flow)
- `docs/LLD.md` - Where to build it (file structure, function signatures)
- `docs/edi_decomposed/index.md` - Navigate to function-level implementation guides

**Configuration**:
- `pyproject.toml` - Python dependencies
- `~/.edi/config.yaml` - User configuration (created by `edi setup`)

## Notes on Current Implementation Status

This is a **specification-first project**. The `docs/` directory contains complete specifications down to the function level, but **most of the actual implementation does not exist yet**.

**What exists**:
- ✅ Complete documentation (PRD, HLD, LLD, function-level decomposition)
- ✅ Working vision pipeline prototype (`work/edi_vision_tui/`)
- ✅ Technology reference examples (`example_code/`)
- ✅ Environment setup and dependencies

**What needs implementation** (follow work/ → builds/ → src/ progression):
- ❌ DSpy reasoning modules (IntentParser, PromptGenerator, Validator)
- ❌ TUI screens and widgets (HomeScreen, RefinementScreen, etc.)
- ❌ Storage layer (Database, StateManager, migrations)
- ❌ ComfyUI integration (ComfyUIClient, WorkflowManager)
- ❌ CLI commands (edit, setup, doctor, clear)
- ❌ Test suite (unit, integration, E2E)

**Development priority** (from `docs/LLD.md` roadmap):
1. Core vision system (use `work/edi_vision_tui/` as reference)
2. DSpy reasoning modules (use `example_code/dspy_toys/` as reference)
3. Orchestration pipeline
4. Basic TUI (use `example_code/textual/` as reference)
5. Integration layer
6. Comprehensive testing

When in doubt about implementation details, consult `docs/edi_decomposed/` for function-level guidance with example code.
