# EDI Development Sequence Plan - Revised Iteration

## Overview
This document outlines the development sequence for implementing the EDI (Edit with Intelligence) CLI tool using Test-Driven Development (TDD) principles. The plan prioritizes core functionality while minimizing wasted effort through strategic test placement and incremental implementation.

## Revised Development Phases

### Phase 1: Core Infrastructure Setup
**Objective**: Establish foundational components required by all other modules

#### Priority Modules:
1. **Utils Layer**:
   - Logging setup (`setup_logger`)
   - Basic validators (`validate_prompt`, `validate_model_name`, `sanitize_filename`)
   - Image operations (`validate_image`, `compute_image_hash`)

#### Implementation Order:
1. Logging utilities (needed for debugging all other components)
2. Basic validators (used throughout the application)
3. Image operations (foundation for vision subsystem)

#### Test Strategy:
- Unit tests for all utility functions with pytest parametrized tests
- Mock-based tests where external dependencies exist
- Simple pytest fixtures for predictable test data
- Test edge cases (invalid inputs, boundary conditions)

### Phase 2: Data Models and Storage Layer
**Objective**: Create data structures and persistence mechanism for state management

#### Components:
1. **Vision Models** (`edi/vision/models.py`):
   - `SceneAnalysis`, `Entity`, `EditDelta`, `Mask` Pydantic models
2. **Reasoning Models** (`edi/reasoning/models.py`):
   - `Intent`, `Prompts`, `ValidationResult` Pydantic models
3. **Storage Models** (`edi/storage/models.py`):
   - Database record models
4. **Database Layer** (`edi/storage/database.py`):
   - SQLite schema implementation
   - Basic CRUD operations for sessions
   - Migration system (`edi/storage/migrations.py`)

#### Implementation Order:
1. Pydantic models (used by all other components)
2. Database schema and basic operations
3. Migration system

#### Test Strategy:
- Unit tests for all Pydantic models with validation testing
- In-memory SQLite for fast database unit tests
- Test schema integrity and migrations
- Test CRUD operations with pytest fixtures

### Phase 3: Vision Subsystem Foundation
**Objective**: Implement core image analysis capabilities

#### Components (in order):
1. **SAM Analyzer** (`edi/vision/sam_analyzer.py`):
   - Basic image segmentation capability with mock SAM model
   - Model caching mechanism
2. **CLIP Labeler** (`edi/vision/clip_labeler.py`):
   - Entity labeling functionality with mock CLIP model
3. **Scene Builder** (`edi/vision/scene_builder.py`):
   - Scene analysis composition from masks and labels
4. **Change Detector** (`edi/vision/change_detector.py`):
   - Before/after comparison for validation

#### Implementation Order:
1. SAM Analyzer (foundational for all vision processing)
2. CLIP Labeler (adds semantic understanding)
3. Scene Builder (combines segmentation and labeling)
4. Change Detector (enables validation)

#### Test Strategy:
- Unit tests with mocked SAM/CLIP models for deterministic behavior
- Integration tests with sample images using real models (later)
- Test performance optimizations (caching, downscaling)
- Test edge cases (small images, complex scenes)

### Phase 4: Reasoning Subsystem Implementation
**Objective**: Enable intent understanding and prompt generation

#### Components (in order):
1. **Ollama Client** (`edi/reasoning/ollama_client.py`):
   - Basic LLM communication with mock responses
2. **Intent Parser** (`edi/reasoning/intent_parser.py`):
   - Simple intent extraction with DSpy mock
3. **Prompt Generator** (`edi/reasoning/prompt_generator.py`):
   - Base prompt creation with refinement loops
4. **Validator** (`edi/reasoning/validator.py`):
   - Edit quality assessment

#### Implementation Order:
1. Ollama Client (enables LLM communication)
2. Intent Parser (translates user requests)
3. Prompt Generator (creates editing instructions)
4. Validator (assesses edit quality)

#### Test Strategy:
- Extensive mock responses for deterministic LLM testing
- Test intent parsing accuracy with various prompt types
- Validate prompt generation quality with example templates
- Test validation logic with predefined scenarios

### Phase 5: Integration Layer Implementation
**Objective**: Enable communication with external services (ComfyUI)

#### Components:
1. **ComfyUI Client** (`edi/integration/comfyui_client.py`):
   - Job submission, status polling, result download
2. **Workflow Manager** (`edi/integration/workflow_manager.py`):
   - Template loading and parameter injection

#### Implementation Order:
1. Workflow Manager (handles templates)
2. ComfyUI Client (communicates with external service)

#### Test Strategy:
- Mock ComfyUI API endpoints for deterministic testing
- Test workflow template handling with various scenarios
- Validate parameter injection with complex templates
- Test error handling (timeouts, connection failures)

### Phase 6: Orchestration and State Management
**Objective**: Connect all components and manage workflow state

#### Components:
1. **State Manager** (`edi/orchestration/state_manager.py`):
   - Session state tracking with auto-save
2. **Editing Pipeline** (`edi/orchestration/pipeline.py`):
   - Main editing workflow coordination
3. **Variation Generator** (`edi/orchestration/variation_generator.py`):
   - Multi-variation support
4. **Compositor** (`edi/orchestration/compositor.py`):
   - Region blending capabilities

#### Implementation Order:
1. State Manager (needed for session persistence)
2. Editing Pipeline (core workflow orchestration)
3. Variation Generator (enhanced capabilities)
4. Compositor (advanced features)

#### Test Strategy:
- Test state serialization/deserialization
- Mock subsystems for pipeline testing
- Test workflow branching logic
- Validate error handling and retry mechanisms

### Phase 7: CLI Commands Implementation
**Objective**: Provide user interface for all functionality

#### Components:
1. **Edit Command** (`edi/commands/edit.py`):
   - Main editing workflow
2. **Setup Command** (`edi/commands/setup_cmd.py`):
   - Environment setup and configuration
3. **Doctor Command** (`edi/commands/doctor.py`):
   - System diagnostics
4. **Clear Command** (`edi/commands/clear.py`):
   - Data cleanup

#### Implementation Order:
1. Setup Command (prerequisite for other commands)
2. Doctor Command (system validation)
3. Edit Command (core functionality)
4. Clear Command (maintenance)

#### Test Strategy:
- CLI argument parsing tests
- End-to-end workflow tests with mocked subsystems
- Test various command combinations and flags
- Validate error messages and help text

### Phase 8: Advanced Features and UI Components
**Objective**: Implement advanced capabilities and user interface enhancements

#### Components:
1. **TUI Layer** (`edi/ui/`):
   - Textual-based user interface
2. **Advanced UI Utilities** (`edi/ui/utils/`):
   - Enhanced terminal visualization
3. **Specialized Widgets** (`edi/ui/widgets/`):
   - Custom UI components

#### Implementation Order:
1. Basic UI components
2. Advanced visualization utilities
3. Full TUI implementation

#### Test Strategy:
- UI component unit tests
- Terminal rendering tests
- User interaction simulation
- Accessibility and usability validation

## TDD Approach Guidelines

### Unit Tests (Priority: Highest)
1. **Scope**: Individual functions and class methods
2. **Strategy**: Mock all external dependencies
3. **Tools**: pytest, unittest.mock, pytest fixtures
4. **Coverage Goal**: 90%+ for core logic

### Integration Tests (Priority: High)
1. **Scope**: Component interfaces and subsystem interactions
2. **Strategy**: Test with simplified mock implementations
3. **Tools**: pytest with custom fixtures
4. **Coverage Goal**: 80%+ for component boundaries

### End-to-End Tests (Priority: Medium)
1. **Scope**: Core user workflows
2. **Strategy**: Minimal mocks; test actual functionality where possible
3. **Tools**: pytest with temporary directories/files
4. **Coverage Goal**: 70%+ for main user journeys

## Risk Mitigation Strategies

### LLM Dependency:
- Create extensive mock responses for deterministic testing
- Implement fallback mechanisms in code design
- Separate LLM-dependent logic for easier mocking
- Design test suites that can run without actual LLMs

### External Service Dependencies:
- Develop clients with configurable base URLs
- Design interfaces that can be easily mocked
- Implement timeout and retry logic gracefully
- Create comprehensive mock services for testing

### Performance Concerns:
- Profile critical paths during development
- Implement caching strategies early
- Design scalable architecture from the start
- Create performance benchmarks for regression testing

### Complex AI Model Integration:
- Use abstract interfaces for model interactions
- Implement adapter patterns for different model types
- Create comprehensive mock models for testing
- Design fallback mechanisms for model failures

## Testing Efficiency Principles

1. **Fast Feedback Loops**:
   - Prioritize unit tests over integration tests
   - Use in-memory databases where possible
   - Mock expensive operations (LLM calls, image processing)
   - Parallelize test execution where safe

2. **Deterministic Tests**:
   - Avoid randomness in test data
   - Use fixed seeds for any pseudo-random operations
   - Mock time-dependent functionality
   - Create reproducible test environments

3. **Focused Test Scopes**:
   - One logical assertion per test
   - Isolate test concerns with proper setup/teardown
   - Use parametrized tests for similar scenarios
   - Separate happy path from error condition tests

4. **Resource Management**:
   - Clean up temporary files and directories automatically
   - Use pytest fixtures for consistent setup/teardown
   - Limit concurrent resource usage in tests
   - Monitor test execution time and optimize slow tests

5. **Test Data Management**:
   - Use factory functions for complex test data
   - Share test data fixtures appropriately
   - Version test data with the code it tests
   - Minimize test data duplication

## Dependency Management Strategy

### Internal Dependencies:
1. Implement components in dependency order
2. Use dependency injection for loose coupling
3. Create abstract interfaces for major subsystems
4. Develop with mocking in mind from the start

### External Dependencies:
1. Wrap external libraries in adapter classes
2. Create mock implementations for testing
3. Handle version compatibility gracefully
4. Document external dependency requirements clearly

## Next Steps Implementation Plan

### Immediate Actions (Next 2 Days):
1. Implement Utils Layer with comprehensive unit tests
2. Create Pydantic models for all subsystems
3. Set up pytest configuration and CI pipeline
4. Begin Storage Layer implementation

### Short-term Goals (Week 1):
1. Complete core infrastructure (Utils, Models, Storage)
2. Implement Vision subsystem foundation with mocks
3. Create comprehensive test suite for existing components
4. Establish code quality standards and linting

### Mid-term Goals (Week 2-3):
1. Implement Reasoning subsystem with LLM mocks
2. Complete Integration layer with ComfyUI mocks
3. Develop Orchestration components
4. Expand test coverage and performance benchmarks

### Long-term Goals (Week 4+):
1. Implement CLI commands with full workflow testing
2. Develop TUI components and advanced UI features
3. Integrate with real AI models and services
4. Conduct performance optimization and user testing

## Success Metrics

### Code Quality:
- Test coverage >85%
- Code linting with no errors/warnings
- Consistent code style enforcement
- Low cyclomatic complexity in core functions

### Performance:
- Image analysis <5 seconds (large images)
- Prompt generation <6 seconds (3 iterations)
- Validation <8 seconds (re-analysis)
- Memory usage within system limits

### Reliability:
- Error handling for all external dependencies
- Graceful degradation under failure conditions
- Data integrity guarantees for session persistence
- Recovery mechanisms for interrupted operations

### User Experience:
- Clear error messages and guidance
- Predictable command behavior
- Helpful documentation and examples
- Responsive feedback during long operations