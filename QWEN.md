## Qwen Added Memories
- EDI (Edit with Intelligence) Project - Complete Specifications:
- Product: Conversational AI image editing assistant that bridges human intention and technical image manipulation
- Core Flow: SEE (analyze image with SAM+CLIP) → THINK (reason about intent with DSPy) → ASK (clarifying questions) → LISTEN (incorporate feedback) → EXECUTE (generate prompts for ComfyUI)
- Architecture: TUI Layer (Textual), Vision Subsystem (SAM 2.1 + CLIP), Reasoning Subsystem (Ollama LLMs), Orchestrator (DSPy), Storage Layer (SQLite), Integration Layer (ComfyUI)
- Key Components:
  * Vision: SAMAnalyzer (segmentation), CLIPLabeler (labeling), SceneBuilder (analysis), ChangeDetector (validation)
  * Reasoning: IntentParser (DSPy module), PromptGenerator (DSPy module), Validator (quality assessment), OllamaClient (LLM interface)
  * Orchestration: EditingPipeline (DSPy module), VariationGenerator (DSPy BestOfN), RegionCompositor (blending), StateManager (persistence)
  * Integration: ComfyUIClient (API wrapper), WorkflowManager (templates)
  * Storage: Database (SQLite), Migration tools
  * TUI: Multiple screens (Home, Upload, Prompt, Analysis, Clarification, Refinement, Results, Variations), custom widgets
- DSPy Usage: For mission-critical reasoning tasks requiring guided, deterministic LLM behavior - Intent parsing, prompt generation/refinement, validation, multi-variation generation using ChainOfThought, Refine, and BestOfN patterns
- Textual TUI: Terminal-based interface with reactive widgets, keyboard navigation, ANSI image rendering, screen-based navigation
- Vision Processing: Uses SAM 2.1 for segmentation, CLIP for entity labeling, with Pydantic models for structured data
- File Structure: CLI handlers, config management, main app, vision/analysis, reasoning/intent, orchestration/pipelines, integration/ComfyUI, storage/persistence, UI/screens/widgets, utilities
- DSPy Implementation Guidelines: Use for complex reasoning, information extraction, guided LLM interactions where reliability is more important than latency; for simple tasks, use Ollama directly; for vision tasks, use Ultralytics (SAM/YOLO) and OpenCLIP
- Configuration: YAML-based, supports different models (qwen3:8b default), performance tuning, ComfyUI settings, UI preferences
- Validation: Alignment scoring using formula (0.4 × Entities Preserved + 0.4 × Intended Changes + 0.2 × (1 - Unintended Changes)), retry logic with max 3 attempts
- DSPy Usage Patterns in EDI Project:
- Used for mission-critical reasoning tasks that require guided, deterministic LLM behavior
- Applied when system cannot be allowed to fail regardless of latency requirements
- Key DSPy Components: 
  1. IntentParser (dspy.ChainOfThought) - Parses user intent from casual prompts, identifies target entities, edit types, and confidence levels
  2. PromptGenerator (dspy.ChainOfThought) - Generates initial prompts with 3-iteration refinement using dspy.Refine
  3. EditingPipeline (dspy.Module) - Orchestrates entire workflow with DSPy modules
  4. VariationGenerator (dspy.BestOfN) - Creates multiple prompt variations using different rollout IDs for diversity
- DSPy Patterns Used:
  * ChainOfThought - For complex reasoning tasks like intent parsing and prompt generation
  * Refine - For iterative prompt improvement (3 iterations with specific goals)
  * BestOfN - For generating diverse prompt variations
  * ReAct - For tool-based interactions (when needed)
- Configuration: Uses Ollama LM with proper API configuration (dspy.LM('ollama_chat/qwen3:8b', api_base='http://localhost:11434', api_key=''))
- When to use DSPy vs Ollama directly: Use DSPy for complex reasoning, information extraction, guided LLM interactions; use Ollama directly for simple 2+2=4 type answers
- DSPy Signatures: Define InputFields and OutputFields with detailed descriptions for structured LLM interactions
- Pydantic models work alongside DSPy for strict data validation and modeling
- Textual TUI Patterns in EDI Project:
- Used for building terminal-based UI similar to Qwen CLI inside terminal
- Architecture: Screen-based navigation with reactive widgets
- Screen Hierarchy: HomeScreen → ImageUploadScreen → PromptInputScreen → AnalysisScreen → ClarificationScreen → RefinementScreen → ExecutionScreen → ResultsScreen/VariationsScreen → FeedbackScreen
- Key Widgets:
  * ImageComparisonPane - Side-by-side image viewer with ANSI art conversion
  * PromptDiffViewer - Shows prompt evolution with highlighted changes
  * EntitySelectorList - Checkbox list for entity selection
  * ValidationMetricsTable - DataTable for validation metrics
  * ProgressSpinner - Animated spinner with status text
- Styling: Uses TCSS (Textual CSS) files for styling with dark/light themes
- UI Utilities: image_to_ansi_art (converts images to terminal display), format_duration (time formatting), color_code_score (score-based color coding)
- Navigation: Keyboard-driven with shortcuts (Q=quit, H=help, B=back, number keys for selections)
- Reactive Programming: Uses reactive variables and watchers for dynamic updates
- Async Operations: Uses @work decorator for long-running operations without blocking UI
- Layout: Uses containers, grids, and dock system for responsive layouts
- Custom Components: Builds domain-specific widgets for image editing workflows
- Coding Principle for EDI Project: Avoid creating hardcoded keyword libraries, dictionaries, or fixed filters in code. Instead of using fixed CLIP labels or hardcoded examples as deterministic filters that reject LLM inferences when they don't match, rely on DSPy pipelines and LLMs to reason through the content naturally. Examples can be provided in code but should not be used as rigid filters that invalidate inferences not matching the examples. The system should allow for deterministic output through LLM reasoning and DSPy pipelines rather than through hardcoded value matching. This ensures flexibility while maintaining reliability through structured DSPy approaches rather than rigid keyword matching.
