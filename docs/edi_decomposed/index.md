# EDI Documentation Index

This is the comprehensive index of all documentation files for the EDI (Edit with Intelligence) project, organized by directory structure.

## Main Documentation

- [EDI Project Overview](./index.md) - EDI: Edit with Intelligence - Documentation Overview
- [DSPy Implementation Review Summary](./edi/dspy_review_summary.md) - DSPy Implementation Review Summary

## Commands

- [Commands: Clear](./edi/commands/clear.md) - Data cleanup command - Contains the clear_command async function that deletes old session files, purges database records, with user confirmation required for full cleanup
- [Commands: Doctor](./edi/commands/doctor.md) - Diagnostic command - Contains the doctor_command async function that checks Python version, GPU availability, models, Ollama connection, ComfyUI connection, and outputs system diagnostics
- [Commands: Edit](./edi/commands/edit.md) - Main edit command - Contains the edit_command async function that serves as the entry point for `edi edit` command, launching the Textual app or running in headless mode
- [Commands: Setup](./edi/commands/setup_cmd.md) - Setup command - Contains the setup_command async function that creates the ~/.edi/ directory structure, downloads models if requested, and verifies Ollama connection

## Integration Layer

### ComfyUI Client

- [Integration: ComfyUI Client](./edi/integration/comfyui_client/integration_comfyui_client.md) - ComfyUI API wrapper - Contains the ComfyUIClient class that handles communication with ComfyUI including submitting edits, polling status, and downloading results
- [ComfyUIClient.download_result()](./edi/integration/comfyui_client/download_result.md) - As a user, I want to receive my edited image when the process is complete
- [ComfyUIClient.poll_status()](./edi/integration/comfyui_client/poll_status.md) - As a user, I want to see progress updates while my image is being edited
- [ComfyUIClient.submit_edit()](./edi/integration/comfyui_client/submit_edit.md) - As a user, I want EDI to generate high-quality image edits using advanced AI models

### Workflow Manager

- [Integration: Workflow Manager](./edi/integration/workflow_manager/integration_workflow_manager.md) - Workflow template handler - Contains the WorkflowManager class that loads workflow templates and injects parameters, validating workflow JSON structure
- [WorkflowManager.inject_params()](./edi/integration/workflow_manager/inject_params.md) - As a user, I want EDI to customize the editing process based on my specific request
- [WorkflowManager.load_template()](./edi/integration/workflow_manager/load_template.md) - As a user, I want EDI to use appropriate editing techniques for different types of changes

- [Integration Layer](./edi/integration_layer.md) - ComfyUI API client, image I/O using requests and Pillow

## Orchestrator

### Pipeline

- [Orchestration: Pipeline](./edi/orchestration/pipeline/orchestration_pipeline.md) - Main editing pipeline - Contains the EditingPipeline class (a dspy.Module) that orchestrates the entire editing process from analysis to validation, with retry logic
- [EditingPipeline.forward()](./edi/orchestration/pipeline/forward_pipeline.md) - As a user, I want EDI to understand my image's composition so it knows what can be safely edited

### Variation Generator

- [Orchestration: Variation Generator](./edi/orchestration/variation_generator/orchestration_variation_generator.md) - Multi-variation support - Contains the VariationGenerator class that creates multiple prompt variations using DSpy BestOfN with different rollout IDs
- [VariationGenerator.generate_variations()](./edi/orchestration/variation_generator/generate_variations.md) - As a user, I want to see multiple variations and pick the best parts from each

### Compositor

- [Orchestration: Compositor](./edi/orchestration/compositor/orchestration_compositor.md) - Region blending - Contains the RegionCompositor class that blends images using Poisson blending for seamless transitions and handles mask feathering
- [RegionCompositor.blend()](./edi/orchestration/compositor/blend.md) - As a user, I want to see multiple variations and pick the best parts from each

### State Manager

- [Orchestration: State Manager](./edi/orchestration/state_manager/orchestration_state_manager.md) - Session state tracking - Contains the StateManager class that saves and loads session state to JSON files, with auto-save functionality
- [StateManager.checkpoint()](./edi/orchestration/state_manager/checkpoint.md) - As a user, I want EDI to remember my progress so I can resume if interrupted
- [StateManager.load_state()](./edi/orchestration/state_manager/load_state.md) - As a user, I want EDI to remember my progress so I can resume if interrupted
- [StateManager.save_state()](./edi/orchestration/state_manager/save_state.md) - As a user, I want EDI to remember my progress so I can resume if interrupted

- [Orchestrator](./edi/orchestrator.md) - Workflow coordination, DSpy pipelines, state management using DSpy 2.6+

## Reasoning Subsystem

### Models

- [Reasoning: Models](./edi/reasoning/models.md) - Pydantic models for reasoning subsystem - Contains Intent, Prompts, and ValidationResult data structures for type-safe operations

### Intent Parser

- [Reasoning: Intent Parser](./edi/reasoning/reasoning_intent_parser.md) - DSpy intent extraction - Contains the IntentParser class (a dspy.Module) that processes naive prompts and scene analysis to extract structured intent
- [IntentParser.forward()](./edi/reasoning/intent_parser/forward_intent.md) - As a user, I want EDI to ask questions when my request is ambiguous rather than guessing

### Ollama Client

- [Reasoning: Ollama Client](./edi/reasoning/reasoning_ollama_client.md) - Ollama API wrapper - Contains the OllamaClient class that handles communication with Ollama for LLM inference
- [OllamaClient.generate()](./edi/reasoning/ollama_client/generate.md) - As a user, I want EDI to understand my intent and generate appropriate editing instructions

### Prompt Generator

- [Reasoning: Prompt Generator](./edi/reasoning/reasoning_prompt_generator.md) - DSpy prompt creation - Contains the PromptGenerator class (a dspy.Module) that creates positive and negative prompts based on intent and scene, with 3 refinement iterations
- [PromptGenerator.forward()](./edi/reasoning/prompt_generator/forward_prompts.md) - As a user, I want EDI to refine its understanding through multiple passes before committing to an edit

### Validator

- [Reasoning: Validator](./edi/reasoning/reasoning_validator.md) - Edit quality assessment - Contains the Validator class that calculates alignment scores and generates retry hints if scores are low
- [Validator.validate()](./edi/reasoning/validator/validate.md) - As a user, I want EDI to check if the edit matches my intent and learn from my corrections

- [Reasoning Subsystem](./edi/reasoning_subsystem.md) - Intent understanding, prompt generation, validation using Ollama (qwen3:8b, gemma3:4b)

## Storage Layer

### Database

- [Storage: Database](./edi/storage/database/storage_database.md) - SQLite wrapper - Contains the Database class that provides methods for saving and loading sessions with transaction support
- [Database.load_session()](./edi/storage/database/load_session.md) - As a user, I want EDI to resume my editing sessions where I left off
- [Database.query_history()](./edi/storage/database/query_history.md) - As a user, I want to review my past editing sessions
- [Database.save_session()](./edi/storage/database/save_session.md) - As a user, I want EDI to remember my editing sessions so I can resume later or review my history

### Migrations

- [Storage: Migrations](./edi/storage/migrations/storage_migrations.md) - Schema versioning - Contains functions for handling database schema changes in a backward-compatible way
- [migrate_v1_to_v2()](./edi/storage/migrations/migrate_v1_to_v2.md) - As a user, I want EDI to continue working properly after software updates

- [Storage: Models](./edi/storage/storage_models.md) - Database models - Contains SessionRecord, PromptRecord, EntityRecord and other data structures that map to database tables
- [Storage Layer](./edi/storage_layer.md) - Session persistence, learning data using SQLite + JSON

## TUI Layer

- [UI: App](./edi/app.md) - Main Textual App class - Contains the main application class that coordinates screen transitions, manages global state, and handles keyboard shortcuts
- [UI: Screens](./edi/ui/screens/screens.md) - Screen definitions - Contains all the screen classes for the Textual TUI including HomeScreen, ImageUploadScreen, PromptInputScreen, etc.
- [UI: Styles](./edi/ui/styles.md) - CSS styling for the Textual TUI - Contains theme files for dark and light mode styling
- [UI: Utils](./edi/ui/utils/utils.md) - UI utilities - Contains helper functions for UI operations like image to ANSI art conversion, duration formatting, etc.
- [color_code_score()](./edi/ui/utils/color_code_score/color_code_score.md) - As a user, I want clear visual feedback about the quality of my edits
- [format_duration()](./edi/ui/utils/format_duration/format_duration.md) - As a user, I want to see clear time estimates for long operations
- [image_to_ansi_art()](./edi/ui/utils/image_to_ansi_art/image_to_ansi_art.md) - As a user, I want to see my images displayed in the terminal interface
- [UI: Widgets](./edi/ui/widgets/widgets.md) - Custom Widgets - Contains specialized UI components like ImageComparisonPane, PromptDiffViewer, EntitySelectorList, etc.
- [TUI Layer](./edi/tui_layer.md) - User interaction, display, navigation using Textual 0.87+

## Vision Subsystem

### Vision Models

- [Vision: Models](./edi/vision/models.md) - Pydantic models for vision subsystem - Contains SceneAnalysis, Entity, EditDelta, and Mask data structures for type-safe operations

### Change Detector

- [Vision: Change Detector](./edi/vision/change_detector/vision_change_detector.md) - Before/after comparison - Contains the ChangeDetector class that matches entities by IoU and calculates alignment scores
- [ChangeDetector.compute_delta()](./edi/vision/change_detector/change_compute_delta.md) - As a user, I want EDI to check if the edit matches my intent and learn from my corrections

### CLIP Labeler

- [Vision: CLIP Labeler](./edi/vision/clip_labeler/vision_clip_labeler.md) - CLIP-based entity labeling - Contains the CLIPLabeler class that compares mask regions to text labels via CLIP and returns confidence scores
- [CLIPLabeler.label_masks()](./edi/vision/clip_labeler/clip_label_masks.md) - As a user, I want EDI to understand my image's composition so it knows what can be safely edited

### SAM Analyzer

- [Vision: SAM Analyzer](./edi/vision/sam_analyzer/vision_sam_analyzer.md) - SAM 2.1 wrapper - Contains the SAMAnalyzer class with analyze method that takes an image path and returns a list of masks. Caches model in memory and handles out-of-memory situations by downscaling
- [SAMAnalyzer.analyze()](./edi/vision/sam_analyzer/sam_analyze.md) - As a user, I want EDI to understand my image's composition so it knows what can be safely edited

### Scene Builder

- [Vision: Scene Builder](./edi/vision/scene_builder/vision_scene_builder.md) - Assembles SceneAnalysis - Contains the SceneBuilder class that clusters related entities and computes spatial layout description
- [SceneBuilder.build()](./edi/vision/scene_builder/scene_build.md) - As a user, I want EDI to understand my image's composition so it knows what can be safely edited

- [Vision Subsystem](./edi/vision/vision_subsystem.md) - Image analysis, object detection, change detection using SAM 2.1 and OpenCLIP

## Utils

### Image Ops

- [Utils: Image Ops](./edi/utils/image_ops/image_ops.md) - Image manipulation utilities - Contains functions for resizing images, validating image files, computing image hashes, etc.
- [resize_image()](./edi/utils/image_ops/resize_image/resize_image.md) - As a user, I want EDI to process my images efficiently without running out of memory

### Validators

- [Utils: Logging](./edi/utils/logging.md) - Logging setup - Contains functions for setting up logging with specific levels and file handlers
- [Utils: Validators](./edi/utils/validators/validators.md) - Input validation - Contains functions for validating prompts, model names, and sanitizing filenames
- [validate_prompt()](./edi/utils/validators/validate_prompt/validate_prompt.md) - As a user, I want EDI to handle my inputs safely and appropriately

## Development Documentation

- [EDI Development Sequence Plan - Revised Iteration](./edit_test/dev_seq.md) - This document outlines the development sequence for implementing the EDI (Edit with Intelligence) CLI tool using Test-Driven Development (TDD) principles
- [EDI Development Approach Using TDD](./edit_test/tdd_approach.md) - This document outlines the comprehensive Test-Driven Development (TDD) approach for implementing the EDI (Edit with Intelligence) CLI tool
