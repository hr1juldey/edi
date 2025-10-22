# File Structure & Module Descriptions (LLD or low level design)

```bash
edi/
├── README.md                           # Quick start guide
├── ARCHITECTURE.md                     # This document (HLD)
├── LICENSE                             # MIT License
├── pyproject.toml                      # Package metadata & dependencies
├── setup.py                            # Installation script
│
├── edi/                                # Main package
│   ├── __init__.py                     # Package initialization
│   ├── __main__.py                     # Entry point: `python -m edi`
│   ├── cli.py                          # CLI argument parsing
│   │   # Parses: edi edit <image> <prompt> [--variations N] [--model X]
│   │   # Routes to appropriate command handler
│   │
│   ├── config.py                       # Configuration management
│   │   # Loads ~/.edi/config.yaml
│   │   # Provides Config dataclass with defaults
│   │   # Validates model availability via Ollama
│   │
│   ├── app.py                          # Main Textual App class
│   │   # Coordinates screen transitions
│   │   # Manages global state (current session)
│   │   # Handles keyboard shortcuts (Q, H, B)
│   │
│   ├── vision/                         # Vision Subsystem
│   │   ├── __init__.py
│   │   ├── sam_analyzer.py             # SAM 2.1 wrapper
│   │   │   # Class: SAMAnalyzer
│   │   │   # Methods: analyze(image_path) -> List[Mask]
│   │   │   # Caches model in memory
│   │   │   # Handles OOM by downscaling
│   │   │
│   │   ├── clip_labeler.py             # CLIP-based entity labeling
│   │   │   # Class: CLIPLabeler
│   │   │   # Methods: label_masks(image, masks) -> List[Entity]
│   │   │   # Compares mask regions to text labels via CLIP
│   │   │   # Returns confidence scores
│   │   │
│   │   ├── scene_builder.py            # Assembles SceneAnalysis
│   │   │   # Class: SceneBuilder
│   │   │   # Methods: build(masks, labels) -> SceneAnalysis
│   │   │   # Clusters related entities
│   │   │   # Computes spatial layout description
│   │   │
│   │   ├── change_detector.py          # Before/after comparison
│   │   │   # Class: ChangeDetector
│   │   │   # Methods: compute_delta(before, after) -> EditDelta
│   │   │   # Matches entities by IoU
│   │   │   # Calculates alignment score
│   │   │
│   │   └── models.py                   # Pydantic models
│   │       # SceneAnalysis, Entity, EditDelta, Mask
│   │       # Type-safe data structures
│   │
│   ├── reasoning/                      # Reasoning Subsystem
│   │   ├── __init__.py
│   │   ├── ollama_client.py            # Ollama API wrapper
│   │   │   # Class: OllamaClient
│   │   │   # Methods: generate(prompt, model) -> str
│   │   │   # Handles connection errors, retries
│   │   │
│   │   ├── intent_parser.py            # DSpy intent extraction
│   │   │   # Class: IntentParser(dspy.Module)
│   │   │   # forward(naive_prompt, scene) -> Intent
│   │   │   # Detects ambiguity, generates questions
│   │   │
│   │   ├── prompt_generator.py         # DSpy prompt creation
│   │   │   # Class: PromptGenerator(dspy.Module)
│   │   │   # forward(intent, scene) -> Prompts
│   │   │   # Base generation + 3 refinement iterations
│   │   │
│   │   ├── validator.py                # Edit quality assessment
│   │   │   # Class: Validator
│   │   │   # Methods: validate(delta, intent) -> ValidationResult
│   │   │   # Calculates alignment score
│   │   │   # Generates retry hints if score low
│   │   │
│   │   └── models.py                   # Pydantic models
│   │       # Intent, Prompts, ValidationResult
│   │       # Type-safe reasoning outputs
│   │
│   ├── orchestration/                  # Workflow Coordination
│   │   ├── __init__.py
│   │   ├── pipeline.py                 # Main editing pipeline
│   │   │   # Class: EditingPipeline(dspy.Module)
│   │   │   # forward(image_path, naive_prompt) -> EditResult
│   │   │   # Orchestrates: analyze → parse → generate → execute → validate
│   │   │   # Handles retry logic (max 3 attempts)
│   │   │
│   │   ├── variation_generator.py      # Multi-variation support
│   │   │   # Class: VariationGenerator
│   │   │   # Methods: generate_variations(intent, N=3) -> List[Prompts]
│   │   │   # Uses DSpy BestOfN with different rollout IDs
│   │   │
│   │   ├── compositor.py               # Region blending
│   │   │   # Class: RegionCompositor
│   │   │   # Methods: blend(images, regions, masks) -> Image
│   │   │   # Poisson blending for seamless transitions
│   │   │   # Handles mask feathering
│   │   │
│   │   └── state_manager.py            # Session state tracking
│   │       # Class: StateManager
│   │       # Methods: save_state(), load_state(), checkpoint()
│   │       # Writes JSON to ~/.edi/sessions/<session_id>.json
│   │       # Auto-saves every 5 seconds
│   │
│   ├── integration/                    # External Services
│   │   ├── __init__.py
│   │   ├── comfyui_client.py           # ComfyUI API wrapper
│   │   │   # Class: ComfyUIClient
│   │   │   # Methods: submit_edit(), poll_status(), download_result()
│   │   │   # Loads workflow templates from workflows/ directory
│   │   │   # Handles timeouts and retries
│   │   │
│   │   └── workflow_manager.py         # Workflow template handler
│   │       # Class: WorkflowManager
│   │       # Methods: load_template(name), inject_params(workflow, params)
│   │       # Validates workflow JSON structure
│   │       # Manages default parameter values
│   │
│   ├── storage/                        # Data Persistence
│   │   ├── __init__.py
│   │   ├── database.py                 # SQLite wrapper
│   │   │   # Class: Database
│   │   │   # Methods: save_session(), load_session(), query_history()
│   │   │   # Initializes tables on first run
│   │   │   # Provides transaction support
│   │   │
│   │   ├── models.py                   # Database models
│   │   │   # SessionRecord, PromptRecord, EntityRecord, etc.
│   │   │   # SQLAlchemy ORM or dataclasses with SQL mapping
│   │   │
│   │   └── migrations.py               # Schema versioning
│   │       # Functions: migrate_v1_to_v2(), etc.
│   │       # Handles backward-compatible schema changes
│   │
│   ├── ui/                             # Textual TUI
│   │   ├── __init__.py
│   │   ├── screens/                    # Screen definitions
│   │   │   ├── __init__.py
│   │   │   ├── home.py                 # HomeScreen
│   │   │   │   # Welcome screen with main menu
│   │   │   │   # Options: New edit, Resume, Examples, Help
│   │   │   │   # Keyboard: 1-4 for options, Q to quit
│   │   │   │
│   │   │   ├── upload.py               # ImageUploadScreen
│   │   │   │   # File selection via input field
│   │   │   │   # Image preview (ASCII art representation)
│   │   │   │   # Validates file exists and is image format
│   │   │   │
│   │   │   ├── prompt_input.py         # PromptInputScreen
│   │   │   │   # TextArea for naive prompt (multi-line supported)
│   │   │   │   # EntitySelectorList for targeting specific objects
│   │   │   │   # Submit button (Enter key)
│   │   │   │
│   │   │   ├── analysis.py             # AnalysisScreen
│   │   │   │   # Shows progress bar during SAM+CLIP analysis
│   │   │   │   # Displays detected entities as they're found
│   │   │   │   # Final summary: "Found 5 entities in 3.2s"
│   │   │   │
│   │   │   ├── clarification.py        # ClarificationScreen
│   │   │   │   # Displays questions generated by intent parser
│   │   │   │   # Radio buttons or number selection (1-5)
│   │   │   │   # Conditional: only shown if confidence < 0.7
│   │   │   │
│   │   │   ├── refinement.py           # RefinementScreen
│   │   │   │   # Shows prompt evolution across 3 iterations
│   │   │   │   # PromptDiffViewer highlights changes (green/red)
│   │   │   │   # Progress bar: "Refining... 2/3"
│   │   │   │   # Final approval: [A]pprove or [E]dit manually
│   │   │   │
│   │   │   ├── execution.py            # ExecutionScreen
│   │   │   │   # "Sending to ComfyUI..." with spinner
│   │   │   │   # Polls job status every 2 seconds
│   │   │   │   # Shows estimated time remaining
│   │   │   │
│   │   │   ├── results.py              # ResultsScreen
│   │   │   │   # Side-by-side ImageComparisonPane
│   │   │   │   # Validation metrics table (alignment score, etc.)
│   │   │   │   # Actions: [A]ccept, [R]etry, [T]weak prompts
│   │   │   │
│   │   │   ├── variations.py           # MultiVariationScreen
│   │   │   │   # 3-column grid layout for A/B/C
│   │   │   │   # Keyboard: 1/2/3 to select, B to blend
│   │   │   │   # Shows generation time for each
│   │   │   │
│   │   │   └── feedback.py             # FeedbackScreen
│   │   │       # Optional user rating (1-5 stars)
│   │   │       # Comment text area
│   │   │       # Thank you message after submission
│   │   │
│   │   ├── widgets/                    # Custom Widgets
│   │   │   ├── __init__.py
│   │   │   ├── image_comparison.py     # ImageComparisonPane
│   │   │   │   # Class: ImageComparisonPane(Widget)
│   │   │   │   # render_images(before_path, after_path)
│   │   │   │   # Uses Rich's image-to-ANSI conversion
│   │   │   │   # Supports zoom with +/- keys
│   │   │   │
│   │   │   ├── prompt_diff.py          # PromptDiffViewer
│   │   │   │   # Class: PromptDiffViewer(Widget)
│   │   │   │   # compute_diff(old, new) -> colored markup
│   │   │   │   # Green for additions, red for removals
│   │   │   │
│   │   │   ├── entity_list.py          # EntitySelectorList
│   │   │   │   # Class: EntitySelectorList(ListView)
│   │   │   │   # Checkbox list of detected entities
│   │   │   │   # Space to toggle, Enter to confirm selection
│   │   │   │
│   │   │   ├── metrics_table.py        # ValidationMetricsTable
│   │   │   │   # Class: ValidationMetricsTable(DataTable)
│   │   │   │   # Rows: Preserved, Modified, Unintended, Score
│   │   │   │   # Color-coded: green >0.8, yellow 0.6-0.8, red <0.6
│   │   │   │
│   │   │   └── progress_spinner.py     # ProgressSpinner
│   │   │       # Class: ProgressSpinner(Widget)
│   │   │       # Animated spinner with status text
│   │   │       # Auto-updates via Textual reactive
│   │   │
│   │   ├── styles/                     # CSS styling
│   │   │   ├── dark_theme.tcss         # Dark mode styles
│   │   │   └── light_theme.tcss        # Light mode styles
│   │   │
│   │   └── utils.py                    # UI utilities
│   │       # image_to_ansi_art(path, max_width) -> str
│   │       # format_duration(seconds) -> "1m 23s"
│   │       # color_code_score(score) -> Rich markup
│   │
│   ├── utils/                          # General Utilities
│   │   ├── __init__.py
│   │   ├── image_ops.py                # Image manipulation
│   │   │   # resize_image(image, max_size)
│   │   │   # validate_image(path) -> bool
│   │   │   # compute_image_hash(path) -> str
│   │   │
│   │   ├── logging.py                  # Logging setup
│   │   │   # setup_logger(name, level)
│   │   │   # Writes to ~/.edi/logs/edi.log
│   │   │   # Rotating file handler (10MB max, 5 backups)
│   │   │
│   │   └── validators.py               # Input validation
│   │       # validate_prompt(text) -> bool
│   │       # validate_model_name(name) -> bool
│   │       # sanitize_filename(name) -> str
│   │
│   └── commands/                       # CLI Command Handlers
│       ├── __init__.py
│       ├── edit.py                     # Main edit command
│       │   # async def edit_command(image_path, prompt, **kwargs)
│       │   # Entry point for `edi edit`
│       │   # Launches Textual app or runs headless mode
│       │
│       ├── setup.py                    # Setup command
│       │   # async def setup_command(download_models=False)
│       │   # Creates ~/.edi/ directory structure
│       │   # Downloads default models if requested
│       │   # Verifies Ollama connection
│       │
│       ├── doctor.py                   # Diagnostic command
│       │   # async def doctor_command()
│       │   # Checks: Python version, GPU availability, models
│       │   # Tests: Ollama connection, ComfyUI connection
│       │   # Outputs: Green checkmarks or red errors
│       │
│       └── clear.py                    # Data cleanup command
│           # async def clear_command(sessions=False, all=False)
│           # Deletes old session files
│           # Purges database records
│           # User confirmation required for --all
│
├── workflows/                          # ComfyUI Workflow Templates
│   ├── img2img_default.json            # Standard image-to-image
│   │   # Nodes: LoadImage, PromptText, KSampler, SaveImage
│   │   # Parameters: positive_prompt, negative_prompt, steps=30
│   │
│   ├── inpaint_masked.json             # Region-specific editing
│   │   # Nodes: LoadImage, LoadMask, Inpaint, SaveImage
│   │   # Parameters: mask_path, prompts, strength=0.8
│   │
│   └── controlnet_canny.json           # Structure-preserving edits
│       # Nodes: LoadImage, CannyEdge, ControlNet, SaveImage
│       # Parameters: prompts, canny_threshold, controlnet_strength
│
├── tests/                              # Test Suite
│   ├── __init__.py
│   ├── conftest.py                     # Pytest fixtures
│   │   # Fixtures: sample_images, mock_ollama, mock_comfyui
│   │   # Setup/teardown for test database
│   │
│   ├── unit/                           # Unit Tests
│   │   ├── test_vision_subsystem.py    # SAM + CLIP tests
│   │   │   # test_sam_analysis(), test_clip_labeling()
│   │   │   # test_scene_builder(), test_change_detection()
│   │   │
│   │   ├── test_reasoning.py           # DSpy module tests
│   │   │   # test_intent_parser(), test_prompt_generator()
│   │   │   # test_refinement_improves_quality()
│   │   │
│   │   ├── test_orchestration.py       # Pipeline tests
│   │   │   # test_full_pipeline(), test_retry_logic()
│   │   │   # test_variation_generation()
│   │   │
│   │   ├── test_storage.py             # Database tests
│   │   │   # test_save_session(), test_query_history()
│   │   │   # test_migration()
│   │   │
│   │   └── test_utils.py               # Utility function tests
│   │       # test_image_validation(), test_prompt_sanitization()
│   │
│   ├── integration/                    # Integration Tests
│   │   ├── test_editing_pipeline.py    # End-to-end editing
│   │   │   # test_simple_edit(), test_ambiguous_prompt()
│   │   │   # test_validation_loop()
│   │   │
│   │   ├── test_comfyui_integration.py # ComfyUI client tests
│   │   │   # test_submit_job(), test_poll_status()
│   │   │   # test_download_result()
│   │   │
│   │   └── test_tui_navigation.py      # UI flow tests
│   │       # test_home_to_edit_flow(), test_keyboard_shortcuts()
│   │       # Uses Textual's pilot for automated testing
│   │
│   └── fixtures/                       # Test Data
│       ├── sample_images/              # Test images
│       │   ├── portrait.jpg            # Simple portrait
│       │   ├── landscape.jpg           # Outdoor scene
│       │   └── complex_scene.jpg       # 10+ entities
│       │
│       ├── mock_models.py              # Mock model responses
│       │   # mock_sam_output(), mock_clip_labels()
│       │   # mock_ollama_response()
│       │
│       └── expected_outputs/           # Ground truth data
│           ├── portrait_entities.json  # Expected entities
│           └── prompt_examples.json    # Expected prompts
│
├── docs/                               # Documentation
│   ├── getting_started.md              # Installation & first edit
│   ├── user_guide.md                   # Detailed usage instructions
│   ├── prompt_tips.md                  # How to write good prompts
│   ├── troubleshooting.md              # Common issues & solutions
│   ├── architecture.md                 # System design (this doc)
│   ├── api_reference.md                # Function/class documentation
│   └── contributing.md                 # Development guidelines
│
├── scripts/                            # Development Scripts
│   ├── download_models.sh              # Download SAM, CLIP weights
│   ├── setup_comfyui.sh                # Install & configure ComfyUI
│   ├── run_tests.sh                    # Run full test suite
│   ├── benchmark.py                    # Performance benchmarking
│   └── generate_docs.py                # Auto-generate API docs
│
└── .github/                            # GitHub Specific
    └── workflows/
        ├── tests.yml                   # CI: Run tests on push
        └── release.yml                 # CD: Build & publish releases
```

---

## Module Dependencies Graph

```bash
┌─────────────────────────────────────────────────────────────┐
│                         cli.py                              │
│                    (Entry Point)                            │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    ┌────────┐     ┌─────────┐    ┌──────────┐
    │edit.py │     │setup.py │    │doctor.py │
    │command │     │command  │    │command   │
    └────┬───┘     └────┬────┘    └─────┬────┘
         │              │               │
         │              │               │
         ▼              ▼               ▼
    ┌────────────────────────────────────────┐
    │            app.py                      │
    │      (Main Textual App)                │
    └────┬───────────────────────────────────┘
         │
         ├──────────────┬──────────────┬─────────────────┐
         │              │              │                 │
         ▼              ▼              ▼                 ▼
    ┌─────────┐   ┌──────────┐   ┌────────┐      ┌──────────┐
    │ui/      │   │pipeline  │   │storage/│      │config.py │
    │screens/ │   │.py       │   │database│      │          │
    └────┬────┘   └────┬─────┘   └────────┘      └──────────┘
         │             │
         │             ├─────────────┬─────────────┐
         │             │             │             │
         │             ▼             ▼             ▼
         │      ┌──────────┐  ┌──────────┐  ┌────────────┐
         │      │vision/   │  │reasoning/│  │integration/│
         │      │          │  │          │  │comfyui     │
         │      └──────────┘  └────┬─────┘  └────────────┘
         │                          │
         │                          ▼
         │                    ┌──────────┐
         │                    │ollama    │
         │                    │client.py │
         │                    └──────────┘
         │
         └──────────► ui/widgets/ (ImageComparisonPane, etc.)
```

---

## Critical Paths & Bottlenecks

### Performance-Critical Components

**1. SAM 2.1 Inference** (~3-5 seconds on RTX 3060)

- **Optimization**:
  - Load model once, keep in VRAM
  - Use `sam2.1_b.pt` (base) not `sam2.1_h.pt` (huge)
  - Pre-resize images >2048px to 2048px max dimension
  - Use FP16 precision: `model.half()`

**2. CLIP Encoding** (~0.5-1 second per mask)

- **Optimization**:
  - Batch process masks (encode all crops in single forward pass)
  - Cache text embeddings for common labels ("sky", "building", etc.)
  - Skip masks <2% of image area (likely noise)

**3. LLM Inference via Ollama** (~1-2 seconds per call)

- **Optimization**:
  - Use `qwen3:8b` (not 30b) for speed
  - Keep Ollama server running (avoid cold start)
  - Use streaming API to show partial results
  - Set `num_ctx=4096` (shorter context = faster)

**4. ComfyUI Generation** (~30-60 seconds)

- **Out of scope**: User expectation, can't optimize
- **Mitigation**: Show progress updates every 5 seconds

### Memory Management

**Peak VRAM Usage**:

```bash
SAM 2.1 Base (FP16):      ~3.5 GB
CLIP ViT-B/32:            ~0.5 GB
qwen3:8b (Ollama):        ~5.0 GB
ComfyUI SD model:         ~4.0 GB (external process)
─────────────────────────────────
Total:                    ~13 GB (exceeds 12GB limit!)
```

**Solution**: Sequential model loading with explicit unloading

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

**System RAM Usage**:

```bash
Python process:           ~500 MB
Textual TUI:             ~50 MB
Image buffers (2048²):   ~50 MB per image × 3 = 150 MB
SQLite database:         ~10 MB
─────────────────────────────────
Total:                   ~700 MB (well within 32GB)
```

---

## Development Roadmap (24-Hour Sprint)

### Hour 0-2: Project Setup

- [ ] Initialize Git repository
- [ ] Create directory structure
- [ ] Setup `pyproject.toml` with dependencies
- [ ] Create virtual environment, install packages
- [ ] Download SAM 2.1 base model (~375 MB)
- [ ] Test Ollama connection with `qwen3:8b`

### Hour 2-6: Core Vision System

- [ ] Implement `vision/sam_analyzer.py`
  - Test on 3 sample images
  - Verify mask generation quality
- [ ] Implement `vision/clip_labeler.py`
  - Test entity labeling accuracy
  - Create label taxonomy (20 common labels)
- [ ] Implement `vision/scene_builder.py`
  - Test spatial layout generation
- [ ] Write unit tests for vision subsystem (80% coverage)

### Hour 6-10: Reasoning System

- [ ] Implement `reasoning/ollama_client.py`
  - Test connection handling, retries
- [ ] Implement `reasoning/intent_parser.py` (DSpy)
  - Create test prompts (ambiguous vs clear)
  - Verify question generation
- [ ] Implement `reasoning/prompt_generator.py` (DSpy)
  - Test refinement loop (3 iterations)
  - Validate prompt quality improvements
- [ ] Write unit tests for reasoning subsystem

### Hour 10-14: Orchestration

- [ ] Implement `orchestration/pipeline.py`
  - Connect vision + reasoning
  - Test end-to-end flow (image → prompts)
- [ ] Implement `orchestration/state_manager.py`
  - Test session save/load
- [ ] Implement `storage/database.py`
  - Create SQLite schema
  - Test CRUD operations

### Hour 14-18: TUI Development

- [ ] Implement core screens:
  - `ui/screens/home.py` (1 hour)
  - `ui/screens/upload.py` (1 hour)
  - `ui/screens/prompt_input.py` (1 hour)
  - `ui/screens/results.py` (1 hour)
- [ ] Implement key widgets:
  - `ui/widgets/image_comparison.py`
  - `ui/widgets/prompt_diff.py`
- [ ] Test navigation flow with keyboard shortcuts

### Hour 18-21: Integration

- [ ] Implement `integration/comfyui_client.py`
  - Test API calls (submit, poll, download)
  - Handle timeouts gracefully
- [ ] Implement `integration/workflow_manager.py`
  - Load `img2img_default.json` template
  - Test parameter injection
- [ ] Connect TUI → Pipeline → ComfyUI
- [ ] End-to-end integration test

### Hour 21-23: Polish & Testing

- [ ] Implement `commands/edit.py` (CLI entry point)
- [ ] Implement `commands/doctor.py` (diagnostics)
- [ ] Write integration tests
- [ ] Fix critical bugs from testing
- [ ] Add error messages for common failures
- [ ] Create `README.md` with quickstart

### Hour 23-24: Documentation

- [ ] Write `docs/getting_started.md`
- [ ] Write `docs/troubleshooting.md`
- [ ] Record 3-minute demo video
- [ ] Tag v0.1.0 release

---

## Testing Checklist

### Functional Tests

**Vision Subsystem**:

- [ ] SAM detects 85%+ of salient objects on 20 test images
- [ ] CLIP labels match human annotations 80%+ of the time
- [ ] Change detector correctly identifies preserved/modified entities
- [ ] Handles edge cases: blank images, solid colors, text-heavy images

**Reasoning Subsystem**:

- [ ] Intent parser detects ambiguity in 90%+ of vague prompts
- [ ] Clarifying questions are answerable via single choice
- [ ] Prompt refinement improves token diversity by 20%+ per iteration
- [ ] Final prompts include preservation constraints

**Orchestration**:

- [ ] Pipeline completes simple edits in <90 seconds (excluding ComfyUI)
- [ ] Retry logic triggers correctly on low alignment scores (<0.6)
- [ ] State manager saves/loads session without data loss
- [ ] Handles crashes gracefully (resumable sessions)

**TUI**:

- [ ] All screens navigable via keyboard (no mouse required)
- [ ] Works in 80×24 terminal (minimum size)
- [ ] Progress bars update correctly
- [ ] Image comparison visible in terminal (ASCII art)

**Integration**:

- [ ] ComfyUI client submits jobs successfully
- [ ] Polls status without blocking UI
- [ ] Downloads results to correct location
- [ ] Handles ComfyUI offline scenario

### Non-Functional Tests

**Performance**:

- [ ] SAM analysis completes in <5 seconds
- [ ] CLIP labeling completes in <2 seconds
- [ ] Prompt generation completes in <6 seconds
- [ ] Total pre-ComfyUI time <15 seconds

**Reliability**:

- [ ] Handles OOM by reducing image resolution
- [ ] Recovers from Ollama connection loss
- [ ] Saves session state before crashes
- [ ] No data corruption in database

**Usability**:

- [ ] First-time user completes edit in <5 minutes
- [ ] Error messages are actionable (not technical jargon)
- [ ] Help text accessible via [H] key on all screens
- [ ] Color-coding is consistent (green=good, red=bad, yellow=warning)

---

## Risk Mitigation

### Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **GPU OOM during SAM inference** | High | Medium | Auto-downscale images >2048px, use FP16, unload models sequentially |
| **Ollama server crashes** | Medium | Low | Auto-restart via subprocess, fallback to smaller model (gemma3:4b) |
| **ComfyUI unavailable** | High | Low | Detect early, provide prompts for manual use, clear error message |
| **CLIP labeling inaccurate** | Medium | Medium | Use confidence thresholds (0.7), allow manual entity selection |
| **DSpy prompt quality low** | Medium | Medium | Implement quality scoring, use BestOfN for variations, allow manual editing |
| **Textual TUI rendering issues** | Low | Low | Test on Linux/macOS/Windows terminals, provide ASCII-only mode |

### Schedule Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Vision system takes >6 hours** | High | Use existing code from `advanced_mask_generator.py`, reduce test coverage to 70% |
| **DSpy learning curve steep** | Medium | Study `Notes.md` examples first, use simple ChainOfThought initially |
| **TUI development slower than expected** | Medium | Start with minimal screens (Home, Results only), iterate later |
| **Integration debugging takes too long** | High | Mock ComfyUI for testing, use sample images with known outputs |

### User Experience Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Users don't understand TUI navigation** | Medium | Add persistent help bar at bottom, tutorial on first launch |
| **Prompts are still too technical** | High | Add examples library, prompt suggestions based on detected entities |
| **Alignment score not intuitive** | Medium | Show visual indicators (checkmarks/X marks per entity), explain in plain language |
| **Waiting for ComfyUI frustrating** | Low | Show estimated time, allow queueing multiple edits |

---

## Configuration & Extensibility

### User Configuration (`~/.edi/config.yaml`)

```yaml
# Model Selection
models:
  reasoning_llm: "qwen3:8b"  # Options: qwen3:8b, mistral:7b, gemma3:4b
  vision_llm: "gemma3:4b"    # For optional VLM fallback
  sam_checkpoint: "sam2.1_b.pt"  # Options: sam2.1_t.pt (faster), sam2.1_b.pt, sam2.1_h.pt (slower)
  clip_model: "ViT-B/32"     # Options: ViT-B/32, ViT-L/14
  clip_pretrained: "openai"  # Options: openai, laion2b_s34b_b79k

# Performance Tuning
performance:
  max_image_size: 2048        # Max dimension, larger images downscaled
  use_half_precision: true    # FP16 for SAM/CLIP
  enable_model_caching: true  # Keep models in memory between sessions
  sam_batch_size: 4           # Number of masks to process in parallel

# Prompt Generation
prompts:
  refinement_iterations: 3    # Number of DSpy refinement passes (1-5)
  quality_keywords:           # Auto-added to positive prompts
    - "high quality"
    - "8k"
    - "detailed"
  default_negative:           # Always included in negative prompts
    - "low quality"
    - "blurry"
    - "artifacts"

# ComfyUI Integration
comfyui:
  base_url: "http://localhost:8188"
  default_workflow: "img2img_default"
  timeout_seconds: 180
  poll_interval_seconds: 5
  auto_open_results: true     # Open results in external viewer

# Validation
validation:
  alignment_threshold_accept: 0.8   # Auto-accept if score ≥ this
  alignment_threshold_review: 0.6   # Ask user if score in this range
  max_retry_attempts: 3             # Max times to retry low-scoring edits

# UI Preferences
ui:
  theme: "dark"               # Options: dark, light
  animation_speed: "normal"   # Options: slow, normal, fast
  show_debug_info: false      # Show timing info, model details
  terminal_size_warning: true # Warn if terminal too small

# Storage
storage:
  database_path: "~/.edi/sessions.db"
  max_session_history: 100    # Oldest sessions auto-deleted
  auto_cleanup_days: 30       # Delete sessions older than this
  save_edited_images: true    # Keep edited images or delete after session

# Logging
logging:
  level: "INFO"               # Options: DEBUG, INFO, WARNING, ERROR
  file_path: "~/.edi/logs/edi.log"
  max_file_size_mb: 10
  backup_count: 5
```

### Plugin System (Future Extension)

**Custom Analyzers**:

```python
# ~/.edi/plugins/face_detector.py
from edi.plugins import AnalyzerPlugin
from edi.vision.models import Entity
import face_recognition

class FaceDetectionAnalyzer(AnalyzerPlugin):
    name = "face_detector"
    priority = 10  # Run after SAM, before CLIP
    
    def analyze(self, image: Image) -> List[Entity]:
        """Detect faces and add as entities."""
        locations = face_recognition.face_locations(np.array(image))
        entities = []
        for i, (top, right, bottom, left) in enumerate(locations):
            entities.append(Entity(
                id=f"face_{i}",
                label="face",
                confidence=0.95,
                bbox=(left, top, right, bottom)
            ))
        return entities
    
    def should_run(self, config) -> bool:
        """Only run if faces expected."""
        return config.enable_face_detection
```

**Custom Prompt Templates**:

```python
# ~/.edi/plugins/cinematic_template.py
from edi.plugins import PromptTemplate

class CinematicTemplate(PromptTemplate):
    name = "cinematic"
    description = "Hollywood-style color grading and composition"
    
    def generate(self, intent, scene) -> tuple[str, str]:
        positive = (
            f"{intent.edit_type}, cinematic color grading, "
            "anamorphic lens, film grain, 2.39:1 aspect, "
            "professional cinematography, preserve: {entities}"
        ).format(entities=", ".join(intent.target_entities))
        
        negative = (
            "amateur, snapshot, oversaturated, digital artifacts, "
            "instagram filter, HDR, smartphone photo"
        )
        
        return positive, negative
```

**Loading Plugins**:

```python
# In config.yaml
plugins:
  enabled:
    - face_detector
    - cinematic_template
  search_paths:
    - "~/.edi/plugins"
    - "/usr/share/edi/plugins"
```

---

## Deployment & Distribution

### Installation Methods

**1. PyPI Package** (Primary):

```bash
pip install edi-image-editor
edi setup --download-models
edi doctor
```

**2. Git Clone** (Development):

```bash
git clone https://github.com/user/edi.git
cd edi
pip install -e .
python -m edi.setup
```

**3. Binary Distribution** (Future):

```bash
# Single executable with bundled Python
curl -sSL https://edi.dev/install.sh | sh
```

### Pre-requisites Check (`edi doctor` output)

```bash
EDI System Diagnostics
══════════════════════════════════════════════════════════

Python Environment:
  ✓ Python version: 3.10.12
  ✓ Virtual environment: /home/user/venv
  ✓ Required packages: All installed

GPU Configuration:
  ✓ CUDA available: Yes (12.1)
  ✓ GPU: NVIDIA GeForce RTX 3060 (12GB)
  ✓ Driver version: 535.104.05

Models:
  ✓ SAM 2.1 Base: ~/.edi/models/sam2.1_b.pt (375 MB)
  ✓ CLIP ViT-B/32: Cached in Hugging Face hub
  ✓ Ollama: Running on localhost:11434
    - qwen3:8b: Available
    - gemma3:4b: Available

External Services:
  ✗ ComfyUI: Not reachable at http://localhost:8188
    → Start ComfyUI: cd ComfyUI && python main.py
  ✓ Disk space: 42 GB available

Configuration:
  ✓ Config file: ~/.edi/config.yaml
  ✓ Database: ~/.edi/sessions.db (15 sessions)

Performance Test:
  ✓ SAM inference: 3.2s (acceptable)
  ✓ CLIP encoding: 0.8s (acceptable)
  ✓ Ollama qwen3:8b: 1.4s (acceptable)

════════════════════════════════════════════════════════════
Status: Ready (1 warning)
Run 'edi edit test.jpg "test prompt"' to verify full system.
```

---

## Success Metrics (POC Validation)

### Quantitative Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Analysis Speed** | <5s | Time from image load to SceneAnalysis |
| **Prompt Quality** | 80%+ alignment | Average score across 20 test images |
| **Retry Rate** | <30% | % of edits requiring validation retry |
| **User Completion** | >80% | % of sessions reaching "Accept" state |
| **Crash Rate** | <5% | % of sessions ending in error |

### Qualitative Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Intent Clarity** | User understands questions | 3 beta testers successfully answer clarifications |
| **Prompt Readability** | Non-technical users understand | 3 beta testers can explain what positive/negative prompts do |
| **UI Intuitiveness** | No manual needed | 3 first-time users complete edit in <5 min without help |
| **Error Messages** | Actionable guidance | 3 testers can resolve errors without developer support |

### User Feedback Questions

1. "On a scale of 1-5, how well did EDI understand your editing intent?"
2. "Did the clarifying questions help refine your request? (Yes/No)"
3. "How satisfied are you with the final result? (1-5 stars)"
4. "Would you use EDI for your next image edit? (Yes/No)"
5. "What was most confusing about the process?"

**Target**: Average 4.0+ on questions 1 & 3, 80%+ "Yes" on questions 2 & 4

---
