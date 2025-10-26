# EDI (Edit with Intelligence) Project - Complete Specifications

## Overview

**Product**: Conversational AI image editing assistant that bridges human intention and technical image manipulation

**Core Flow**: SEE (analyze image with SAM+CLIP) → THINK (reason about intent with DSPy) → ASK (clarifying questions) → LISTEN (incorporate feedback) → EXECUTE (generate prompts for ComfyUI)

**Architecture**: TUI Layer (Textual), Vision Subsystem (SAM 2.1 + CLIP), Reasoning Subsystem (Ollama LLMs), Orchestrator (DSPy), Storage Layer (SQLite), Integration Layer (ComfyUI)

## System Components

### Vision Components
- **SAMAnalyzer**: Segmentation
- **CLIPLabeler**: Labeling
- **SceneBuilder**: Analysis
- **ChangeDetector**: Validation

### Reasoning Components
- **IntentParser**: DSPy module for intent parsing
- **PromptGenerator**: DSPy module for prompt generation
- **Validator**: Quality assessment
- **OllamaClient**: LLM interface

### Orchestration Components
- **EditingPipeline**: DSPy module for workflow orchestration
- **VariationGenerator**: DSPy BestOfN for variations
- **RegionCompositor**: Blending
- **StateManager**: Persistence

### Integration Components
- **ComfyUIClient**: API wrapper
- **WorkflowManager**: Templates

### Storage Components
- **Database**: SQLite
- **Migration tools**

### TUI Components
- Multiple screens (Home, Upload, Prompt, Analysis, Clarification, Refinement, Results, Variations)
- Custom widgets

## DSPy Usage and Implementation

### Core DSPy Concepts
- Used for mission-critical reasoning tasks requiring guided, deterministic LLM behavior
- Applied for Intent parsing, prompt generation/refinement, validation, multi-variation generation
- Uses ChainOfThought, Refine, and BestOfN patterns

### Key DSPy Components
1. **IntentParser (dspy.ChainOfThought)**: Parses user intent from casual prompts, identifies target entities, edit types, and confidence levels
2. **PromptGenerator (dspy.ChainOfThought)**: Generates initial prompts with 3-iteration refinement using dspy.Refine
3. **EditingPipeline (dspy.Module)**: Orchestrates entire workflow with DSPy modules
4. **VariationGenerator (dspy.BestOfN)**: Creates multiple prompt variations using different rollout IDs for diversity

### DSPy Patterns Used
- **ChainOfThought**: For complex reasoning tasks like intent parsing and prompt generation
- **Refine**: For iterative prompt improvement (3 iterations with specific goals)
- **BestOfN**: For generating diverse prompt variations
- **ReAct**: For tool-based interactions (when needed)

### When to Use DSPy vs Ollama Directly
- **Use DSPy** for complex reasoning, information extraction, guided LLM interactions where reliability is more important than latency
- **Use Ollama directly** for simple tasks like basic calculations
- **Use Ultralytics (SAM/YOLO) and OpenCLIP** for vision tasks

### DSPy Signatures
- Define InputFields and OutputFields with detailed descriptions for structured LLM interactions
- Pydantic models work alongside DSPy for strict data validation and modeling

## Detailed DSPy Usage in the EDI System

### 1. Intent Parsing and Ambiguity Resolution
- When a user provides an ambiguous prompt like "make it more dramatic", DSPy's ChainOfThought reasoning analyzes the naive prompt and scene analysis to:
  - Extract structured intent with target entities (specific image regions to edit)
  - Classify edit type (color, style, add, remove, transform)
  - Calculate confidence score (0-1) indicating clarity of intent
  - Generate clarifying questions if confidence < 0.7
- Example: "I notice your image has a sky (40%) and a building (60%). Which should I focus on? [1] Darken the sky with storm clouds [2] Add contrast to the building [3] Both"

### 2. Prompt Generation and Refinement
- DSPy generates technical positive/negative prompts from casual user requests through a 3-stage refinement process:
  - Stage 1: Initial generation based on intent and scene analysis
  - Stage 2-4: Iterative refinement with increasing technical specificity
  - Each iteration improves token diversity by 20%+
- Example progression:
  - Iteration 0: "dramatic sky, dark clouds"
  - Iteration 1: "stormy cumulus clouds, volumetric lighting, overcast sky, moody atmosphere, preserve building detail"
  - Iteration 2: "photorealistic storm clouds, cumulonimbus formation, diffuse natural lighting, 8k detail, preserve foreground subjects"
  - Iteration 3: Adds quality/style modifiers and final preservation constraints

### 3. Validation and Quality Assessment
- After ComfyUI returns an edited image, DSPy validates the edit quality:
  - Re-analyzes edited image with SAM+CLIP
  - Computes delta vs original to identify preserved/modified/removed entities
  - Calculates alignment score using formula: (0.4 × Entities Preserved + 0.4 × Intended Changes + 0.2 × (1 - Unintended Changes))
  - Determines action based on score: Accept (>0.8), Review/User Decision (0.6-0.8), Retry (<0.6)
  - Generates retry hints to improve next attempt if needed

### 4. Multi-Variation Generation
- For complex edits, DSPy generates multiple prompt variations using BestOfN approach:
  - Creates 3+ distinct interpretations of the same intent
  - Uses different rollout IDs for diversity
  - Ensures variations differ by >30% tokens
  - Submits all to ComfyUI in parallel for user selection

### 5. Workflow Orchestration
- Coordinating the complete editing pipeline with branching logic:
  - Analyze image → Parse intent → Generate prompts → Execute → Validate
  - Handling retry logic (max 3 attempts)
  - Managing conditional flows based on confidence scores
  - Composing region blending when combining multiple variations

## DSPy Implementation Guidelines

### Core Principles
- Used for mission-critical reasoning tasks that require guided, deterministic LLM behavior
- Applied when system cannot be allowed to fail regardless of latency requirements

### Implementation Approach
- Use for complex reasoning, information extraction, guided LLM interactions where reliability is more important than latency
- For simple tasks, use Ollama directly
- For vision tasks, use Ultralytics (SAM/YOLO) and OpenCLIP

### Configuration
- Uses Ollama LM with proper API configuration: dspy.LM('ollama_chat/qwen3:8b', api_base='http://localhost:11434', api_key='')

## Textual TUI Patterns in EDI Project

### General Architecture
- Used for building terminal-based UI similar to Qwen CLI inside terminal
- Architecture: Screen-based navigation with reactive widgets

### Screen Hierarchy
HomeScreen → ImageUploadScreen → PromptInputScreen → AnalysisScreen → ClarificationScreen → RefinementScreen → ExecutionScreen → ResultsScreen/VariationsScreen → FeedbackScreen

### Key Widgets
- **ImageComparisonPane**: Side-by-side image viewer with ANSI art conversion
- **PromptDiffViewer**: Shows prompt evolution with highlighted changes
- **EntitySelectorList**: Checkbox list for entity selection
- **ValidationMetricsTable**: DataTable for validation metrics
- **ProgressSpinner**: Animated spinner with status text

### Styling and UI Utilities
- Uses TCSS (Textual CSS) files for styling with dark/light themes
- **UI Utilities**: 
  - image_to_ansi_art (converts images to terminal display)
  - format_duration (time formatting)
  - color_code_score (score-based color coding)

### Navigation and Reactive Programming
- Keyboard-driven with shortcuts (Q=quit, H=help, B=back, number keys for selections)
- Uses reactive variables and watchers for dynamic updates
- Uses @work decorator for long-running operations without blocking UI
- Uses containers, grids, and dock system for responsive layouts
- Builds domain-specific widgets for image editing workflows

## Critical Development Principles: Avoiding Hardcoded Libraries and Deterministic Filters

When coding solutions for the EDI system, it's essential to avoid creating hardcoded keyword libraries, dictionaries, or fixed filters in the code. This principle applies specifically to:

1. **No Fixed CLIP Label Libraries**: Don't create predefined lists of CLIP labels that act as deterministic filters. The system should not reject LLM inferences just because they don't match predefined categories.

2. **No Hardcoded Entity Recognition**: Don't build fixed dictionaries of recognizable entities. Instead, use DSPy pipelines and LLM reasoning to dynamically determine appropriate labels and interpretations.

3. **No Deterministic Filtering**: Don't implement filters that categorically reject outputs that don't match hardcoded examples. Novel interpretations should be allowed as long as they're logically sound.

4. **LLM-Guided Reasoning Over Fixed Rules**: Use DSPy's ChainOfThought and ReAct patterns for uncertainty resolution rather than hardcoded conditional logic.

5. **Dynamic vs Static Classification**: Replace static lookup tables with dynamic DSPy modules that can adapt to new contexts and inputs.

6. **Examples as Guidance, Not Constraints**: Provide examples in code for training/reference purposes but don't use them as strict filters that block valid but novel inferences.

The key insight is that while the system needs deterministic outputs for reliability, this determinism should come from structured DSPy reasoning rather than rigid predefined libraries. The system should use LLMs and DSPy pipelines to reason about what's appropriate for each specific context rather than trying to fit everything into hardcoded categories.

## Inspirations and Similarities to Other DSPy Applications

### EDI's Question-Answering Framework and RPG Games
EDI's question-answering framework is inspired by the RPG game's approach to contextual dialogue and clarification. Key similarities include:
1. Dynamic clarification generation based on confidence - RPG's DialogueGenerator for NPC responses vs EDI's ParseIntent generating clarifying questions when confidence < 0.7
2. Context-aware question generation - RPG considers player stats, location, game progress vs EDI uses scene analysis with detected entities and spatial layout
3. Interactive loop for understanding refinement - RPG generates responses leading to quests/information vs EDI re-parses with user responses to refine intent
4. DSPy Signatures for structured reasoning - RPG's DialogueGenerator signature vs EDI's ParseIntent signature
5. Adaptive interaction flow - RPG adapts based on NPC personality/choices vs EDI adapts based on ambiguity in user prompt
6. User input integration - RPG integrates player choices into game state vs EDI integrates user responses to refine understanding
7. Dedicated UI for clarifications - EDI has a ClarificationScreen with QuestionLabel, OptionsRadioSet, and ConfirmButton widgets specifically for displaying questions, mirroring RPG's interactive choice presentation

Both systems use DSPy to generate intelligent, context-aware responses rather than fixed decision trees, creating adaptive and personalized interaction flows.

### DSPy Email Extractor Similarities
The DSPy email extractor is a system for reliable information extraction from arbitrary noisy text using multiple DSPy ChainOfThought modules in sequence:
1. ClassifyEmail: Uses ChainOfThought to classify email type and urgency with reasoning
2. ExtractEntities: Uses ChainOfThought to extract structured entities with confidence scores
3. GenerateActionItems: Uses ChainOfThought to determine required actions and priorities
4. SummarizeEmail: Uses ChainOfThought to create concise summaries

Key similarities with EDI system:
1. ChainOfThought reasoning for complex information processing - both use ChainOfThought for reliable processing of noisy/complex input
2. Multi-stage information refinement pipeline - both process through multiple stages where each feeds context to the next
3. Confidence-based decision making - both assign confidence scores and make decisions based on confidence thresholds (EDI: intent parser confidence <0.7 triggers questions; Email extractor: confidence scores for entities)
4. Structured output generation with reasoning - both provide reasoning for their classifications and decisions
5. Noisy input processing - both transform unstructured, messy input into clean, structured data
6. Sequential ChainOfThought processing - both use multiple ChainOfThought modules in sequence where each builds on the previous output

Why ChainOfThought is critical in both systems:
- Complex reasoning required: both need to understand complex input and extract meaningful structured information
- Reliability over speed: both prioritize getting the right answer consistently
- Context dependencies: each step requires understanding of previous step results
- Ambiguity handling: when input is ambiguous, the system needs to reason through possible interpretations before making decisions

The core insight is that both systems use ChainOfThought to transform unstructured, potentially noisy input into structured, reliable output through systematic reasoning rather than direct pattern matching.

## Relationship to DSPy Code Generation and Finance Analyst Examples

### Abstract Pattern of DSPy Code Generation Applied to EDI
Instead of "library documentation URLs → code examples", EDI uses "image analysis + user intent → editing instructions":

**Abstract Mapping**:
- Documentation URLs → Images (visual content with scene analysis)
- Library concepts/patterns → Image entities/layout (sky, building, person, spatial relationships)
- Code examples → User natural language prompts ("make it dramatic", "enhance colors")
- Generated code → Technical editing prompts (positive/negative prompt pairs for ComfyUI)
- Library API understanding → Understanding of what can be safely edited without breaking composition

**The Core Transformation**:
Code Generation Pattern:
```python
documentation_urls → LibraryAnalyzer → library_concepts → CodeGenerator → working_code
```

EDI Pattern:
```python
image + naive_prompt → VisionSubsystem → scene_analysis → IntentParser → structured_intent → PromptGenerator → technical_prompts
```

**Where the Learning Happens**:
In code generation, the system learns programming patterns from documentation. In EDI, the system learns:
1. Visual composition patterns from scene analysis (what elements make up an image)
2. Editing patterns from user intent (what changes people commonly want)
3. Technical translation patterns (how to convert casual language to precise editing instructions)

**The Refinement Process**:
Both systems use iterative refinement:
- Code generation: Basic code → Refine with best practices → Optimize for performance
- EDI: Basic prompts → Add preservation constraints → Increase technical specificity → Add quality modifiers

**Knowledge Synthesis**:
- Code generation synthesizes library usage patterns into working code
- EDI synthesizes visual understanding and user intent into precise editing instructions

The key insight is that both are "intelligent translation systems" - they take a high-level, ambiguous input (documentation text or casual image edit request) and translate it into precise, executable output (code or technical prompts) through structured reasoning and pattern recognition.

### DSPy-inspired Pattern for Taking User Intent and Converting to Structured Actions in EDI

**Abstract Pattern**: 
User Intent → Structured Analysis → Precise Technical Actions

**In Code Generation Context**:
- Input: Library documentation URLs
- Analysis: LibraryAnalyzer extracts core concepts, patterns, methods
- Output: CodeGenerator creates working code examples

**In EDI Context**:
- Input: User natural language prompt + image analysis
- Analysis: IntentParser extracts structured intent with target entities, edit type, confidence
- Output: PromptGenerator creates technical positive/negative prompt pairs

**Key Components Mapping**:

1. **DocumentationFetcher** → **Vision Subsystem** (SAM+CLIP)
   - Code Gen: Fetches and processes library documentation
   - EDI: Analyzes images to understand visual composition and editable elements

2. **LibraryAnalyzer** → **IntentParser** (dspy.ChainOfThought)
   - Code Gen: Extracts library concepts, patterns, installation info
   - EDI: Extracts target entities, edit type, confidence, clarifying questions

3. **CodeGenerator** → **PromptGenerator** (dspy.ChainOfThought)
   - Code Gen: Generates working code with imports, explanations, best practices
   - EDI: Generates technical prompts with preservation constraints, quality modifiers

4. **DocumentationLearningAgent** → **EditingPipeline** (dspy.Module)
   - Code Gen: Orchestrates learning from docs and generating code examples
   - EDI: Orchestrates image analysis → intent parsing → prompt generation → execution → validation

**Refinement Process**:
- Code Gen: Basic code → Add best practices → Optimize for requirements
- EDI: Casual prompt → Add preservation constraints → Increase technical specificity → Add quality/style modifiers

**Quality Assurance**:
- Code Gen: Code examples validated by running and testing
- EDI: Edit quality assessed by re-analyzing edited image and comparing to intent

The core insight is that both systems use DSPy to transform ambiguous human inputs (documentation text or casual edit requests) into precise, executable outputs (code or technical prompts) through structured reasoning rather than direct pattern matching.

### DSPy Finance Analyst Pattern
The finance analyst example shows another application of DSPy pattern:

**Pattern**: 
Financial Query → Tool-Based Analysis → Structured Financial Response

**Components**:
1. **Tools Integration**: 
   - Yahoo Finance News Tool (LangChain converted to DSPy Tool)
   - Custom stock price functions (get_stock_price, compare_stocks)
   
2. **ReAct Agent**: 
   - FinancialAnalysisAgent(dspy.Module) with dspy.ReAct
   - Uses signature "financial_query -> analysis_response"
   - Max 6 iterations for complex financial reasoning

3. **Tool-Based Reasoning**:
   - Agent can call external tools (Yahoo Finance News)
   - Can execute custom functions (stock price lookup)
   - Combines multiple information sources

**Application to EDI**:
This pattern maps to EDI as:

1. **Tools Integration** → **Vision + Reasoning Tools**
   - Finance: Yahoo Finance API, stock price functions
   - EDI: SAM analyzer, CLIP labeler, Ollama LLM

2. **ReAct Agent** → **Editing Pipeline**
   - Finance: FinancialAnalysisAgent with ReAct loop
   - EDI: EditingPipeline with ChainOfThought modules

3. **Multi-tool Reasoning** → **Multi-stage Processing**
   - Finance: News lookup + Price lookup + Comparison
   - EDI: Image analysis + Intent parsing + Prompt generation + Validation

The key insight is that DSPy enables creating domain-specialized intelligent agents that can reason through complex tasks by combining multiple tools and knowledge sources in a structured way.

## DSPy Implementation Areas in EDI

Based on the LLD documentation, DSPy will be used in the following primary areas:

1. **`edi/reasoning/intent_parser.py`** - ChainOfThought module to extract structured intent from casual prompts
2. **`edi/reasoning/prompt_generator.py`** - ChainOfThought module with 3-stage refinement for technical prompt creation
3. **`edi/reasoning/validator.py`** - Quality assessment of edits with alignment scoring
4. **`edi/orchestration/pipeline.py`** - Main EditingPipeline coordinating the complete workflow
5. **`edi/orchestration/variation_generator.py`** - BestOfN for generating diverse prompt interpretations

## Key Benefits of DSPy Approach for EDI

- **Reliability over Speed**: Guided LLM reasoning ensures consistent, correct outputs
- **Structured Uncertainty Handling**: Confidence scoring (<0.7) triggers clarifying questions
- **Iterative Refinement**: 3-stage prompt improvement (casual → constrained → technical)
- **Multi-tool Orchestration**: Combines vision analysis, LLM reasoning, and external tools
- **Quality Assurance**: Validation loops with retry logic ensure edit quality

This approach directly addresses the PRD's core problem: "Users must learn prompt engineering to get consistent results" by creating an intelligent intermediary that translates casual human intent into precise technical specifications.

## Additional DSPy Patterns and Considerations

### Core Insight: Intelligent Translation Systems
All three DSPy examples (`dspy_code_generation.py`, `dspy_finance_analyst.py`, and EDI's planned implementation) follow the same fundamental pattern:

**Intelligent Translation Systems**:
```
Ambiguous Human Input → Structured DSPy Analysis → Precise Technical Output
```

### Pattern Applications
1. **Code Generation**: Documentation URLs → Library Understanding → Working Code
2. **Finance Analysis**: Financial Queries → Multi-tool Research → Investment Advice  
3. **EDI**: Natural Language Edits + Image Analysis → Structured Intent → Technical Prompts

### DSPy Code Generation Pattern Applied to Avoiding Hardcoded Libraries

The dspy_code_generation.py example demonstrates an important principle for EDI development - avoiding hardcoded libraries and deterministic filters:

1. **Dynamic Learning Approach**: Instead of using fixed keyword dictionaries or hardcoded libraries, the system dynamically learns from documentation URLs:
   - DocumentationFetcher fetches content from live URLs
   - LibraryAnalyzer extracts concepts, patterns, and methods dynamically
   - No predefined lists of "valid" keywords or concepts

2. **Adaptive Code Generation**: The CodeGenerator creates code examples based on learned library information rather than fixed templates:
   - Takes learned library_info as context (concepts, patterns, methods)
   - Generates code for specific use cases dynamically
   - No hardcoded code patterns or fixed function signatures

3. **Flexible Output Without Filters**: The system doesn't filter outputs against predefined libraries:
   - Generated code isn't validated against hardcoded "correct" patterns
   - Novel implementations are allowed as long as they meet the use case requirements
   - Quality is determined through explanation and best practices rather than matching templates

4. **LLM-Guided Reasoning Over Fixed Rules**: The approach uses LLM reasoning for all decisions:
   - Library analysis through ChainOfThought reasoning
   - Code generation with contextual understanding
   - Quality assessment via explanations and best practices
   - No rule-based filtering against predefined categories

This pattern should be applied to EDI's vision system:
- DON'T create fixed CLIP label dictionaries that act as filters
- DON'T reject LLM inferences that don't match predefined categories
- DO use DSPy pipelines to dynamically determine appropriate labels and interpretations
- DO allow novel interpretations as long as they're logically sound
- DO use ChainOfThought reasoning for uncertainty resolution rather than hardcoded rules

The key insight is that reliability comes from structured LLM reasoning (DSPy) rather than rigid predefined libraries.

## Additional DSPy Considerations and Examples

### DSPy Usage Patterns in EDI Project
- Used for mission-critical reasoning tasks that require guided, deterministic LLM behavior
- Applied when system cannot be allowed to fail regardless of latency requirements
- Key DSPy Components: 
  1. IntentParser (dspy.ChainOfThought) - Parses user intent from casual prompts, identifies target entities, edit types, and confidence levels
  2. PromptGenerator (dspy.ChainOfThought) - Generates initial prompts with 3-iteration refinement using dspy.Refine
  3. EditingPipeline (dspy.Module) - Orchestrates entire workflow with DSPy modules
  4. VariationGenerator (dspy.BestOfN) - Creates multiple prompt variations using different rollout IDs for diversity

### DSPy Code Generation Technology in the EDI System
In the EDI system, DSPy code generation technology would be used in several key areas:

1. **Orchestrator/EditingPipeline Module**: The main editing pipeline (class EditingPipeline(dspy.Module)) uses multiple DSPy ChainOfThought modules for sequential processing:
   - IntentParser (dspy.ChainOfThought) for understanding user requests
   - PromptGenerator (dspy.ChainOfThought) for initial prompt creation
   - PromptRefiner (dspy.Refine) for iterative prompt improvement
   - Validator (dspy.ChainOfThought) for edit quality assessment

2. **Reasoning Subsystem Components**: Several core reasoning modules use DSPy:
   - IntentParser.forward() - parses naive prompts and scene analysis to extract structured intent with target entities, edit type, confidence, and clarifying questions
   - PromptGenerator.forward() - generates base positive/negative prompts with 3-stage refinement
   - Validator.validate() - evaluates edit quality and generates retry hints

3. **Orchestration Logic**: The system uses DSPy for workflow coordination:
   - Handling ambiguity when confidence < 0.7 by asking clarifying questions
   - Multi-iteration refinement loops for prompt improvement
   - Validation loops with retry logic (max 3 attempts)
   - BestOfN generation for creating multiple prompt variations

4. **Dynamic Signature Generation**: DSPy signatures like ParseIntent, GenerateBasePrompt, RefinePrompt, and ValidateEdit define structured input/output patterns that guide LLMs to generate specific types of content based on context

The key insight is that DSPy is used throughout EDI for complex reasoning tasks that require guided, deterministic LLM behavior where reliability is more important than raw speed - specifically for translating ambiguous user intent into structured technical specifications for image editing.

## Vision Processing and File Structure

### Vision Processing
- Uses SAM 2.1 for segmentation
- Uses CLIP for entity labeling
- Uses Pydantic models for structured data

### File Structure
- CLI handlers
- Config management
- Main app
- Vision/analysis
- Reasoning/intent
- Orchestration/pipelines
- Integration/ComfyUI
- Storage/persistence
- UI/screens/widgets
- Utilities

## Configuration and Validation

### Configuration
- YAML-based, supports different models (qwen3:8b default)
- Performance tuning options
- ComfyUI settings
- UI preferences

### Validation
- Alignment scoring using formula: (0.4 × Entities Preserved + 0.4 × Intended Changes + 0.2 × (1 - Unintended Changes))
- Retry logic with maximum 3 attempts

## Project Dependencies and Setup

Based on pyproject.toml:

- Python requirement: >=3.12
- Core dependencies include:
  - dspy>=3.0.0 (DSPy framework for LLM orchestration)
  - fastapi>=0.119.1 (web framework)
  - fastmcp>=2.12.5 (MCP protocol support)
  - matplotlib>=3.10.7, numpy>=2.3.4, pandas>=2.3.3, scipy>=1.15.1, seaborn>=0.13.2 (data science tools)
  - open-clip-torch>=3.2.0 (CLIP model support)
  - pillow>=12.0.0, scikit-image>=0.25.2 (image processing)
  - textual>=6.4.0, textual-dev>=1.8.0 (TUI framework)
  - ultralytics>=8.3.220 (YOLO/SAM models)
  - pydantic>=2.12.3 (data validation)
  - tqdm>=4.66.5 (progress bars)
  - pytest>=8.4.2 (testing)

This confirms that the project is set up to use modern Python 3.12+ with all the necessary dependencies for implementing the EDI system as described in the documentation, including DSPy 3.0+, Textual for the TUI, and the required computer vision libraries.

## DSPy Design Patterns and Implementation Details

### Key Understanding: DSPy in EDI System

#### 1. Core Principle: Reasoning Over Hardcoding
The fundamental approach in EDI is to replace hardcoded libraries and deterministic filters with LLM-guided reasoning:
- Don't: Create fixed CLIP label dictionaries that act as filters
- Don't: Reject LLM inferences that don't match predefined categories
- Do: Use DSPy pipelines to dynamically determine appropriate interpretations
- Do: Allow novel outputs as long as they're logically sound

#### 2. DSPy Usage Areas

**A. Intent Parsing & Ambiguity Resolution**

# Instead of: if prompt.contains("dramatic"): apply_dramatic_template()
# Use DSPy:
class ParseIntent(dspy.Signature):
    naive_prompt = dspy.InputField(desc="User's conversational edit request")
    scene_analysis = dspy.InputField(desc="JSON of detected entities and layout")

    target_entities = dspy.OutputField(desc="Comma-separated list of entity IDs to edit")
    edit_type = dspy.OutputField(desc="One of: color, style, add, remove, transform")
    confidence = dspy.OutputField(desc="Float 0-1 indicating clarity of intent")
    clarifying_questions = dspy.OutputField(desc="JSON array of questions if confidence <0.7")

**B. Prompt Generation & Refinement**

# Instead of: prompt_templates = {"dramatic": "dramatic sky, ..."}
# Use DSPy with iterative refinement:
class GenerateBasePrompt(dspy.Signature):
    # ... inputs ...
    positive_prompt = dspy.OutputField(desc="Technical prompt for desired changes")
    negative_prompt = dspy.OutputField(desc="Technical prompt for things to avoid")

# 3-stage refinement using dspy.Refine

**C. Validation & Quality Assessment**

# Instead of: if delta.mean() > 6.0: accept else: reject
# Use DSPy with statistical reasoning:
class ValidateEdit(dspy.Signature):
    delta = dspy.InputField(desc="EditDelta object with changes")
    intent = dspy.InputField(desc="Original structured intent")

    verdict = dspy.OutputField(desc="One of: ACCEPT, REVIEW, RETRY")
    score = dspy.OutputField(desc="Float 0-1 alignment score")
    message = dspy.OutputField(desc="Human-readable explanation")
    retry_hints = dspy.OutputField(desc="JSON array of hints if RETRY")

#### 3. Key DSPy Patterns in EDI

1. ChainOfThought for Complex Reasoning:
   - IntentParser → Extracts structured intent from casual prompts
   - PromptGenerator → Creates technical prompts from structured intent
   - Validator → Assesses edit quality with explanations

2. Refine for Iterative Improvement:
   - 3-stage prompt refinement (add constraints → increase specificity → add modifiers)
   - Each iteration improves token diversity by 20%+

3. BestOfN for Variation Generation:
   - Creates 3+ prompt variations using different rollout IDs
   - Ensures diversity (>30% token difference between variations)

4. Module Composition for Workflow Orchestration:

   class EditingPipeline(dspy.Module):
       def __init__(self):
           self.analyzer = VisionSubsystem()
           self.intent_parser = dspy.ChainOfThought(ParseIntent)
           self.prompt_generator = dspy.ChainOfThought(GenerateBasePrompt)
           self.prompt_refiner = dspy.Refine(RefinePrompt, N=3)
           self.validator = dspy.ChainOfThought(ValidateEdit)

#### 4. Why This Approach Works for EDI

1. Reliability Over Speed: Guided LLM reasoning ensures consistent outputs even if slower
2. Adaptability: No hardcoded filters means the system can handle novel requests
3. Transparency: ChainOfThought provides explanations for all decisions
4. Quality Assurance: Multi-stage validation with statistical scoring
5. User Experience: Clarifying questions when uncertain, not silent failures

The core insight is that EDI uses structured DSPy reasoning to replace brittle hardcoded logic, ensuring the system remains flexible, reliable, and adaptive while still providing deterministic behavior through guided LLM interactions.

## Development Principles Summary

### Comprehensive DSPy Usage in EDI System - Final Summary

#### 1. Core Principle: Reasoning Over Hardcoding
- Replace hardcoded libraries and deterministic filters with LLM-guided reasoning
- Don't create fixed CLIP label dictionaries that act as filters
- Don't reject LLM inferences that don't match predefined categories
- Do use DSPy pipelines to dynamically determine appropriate interpretations
- Do allow novel outputs as long as they're logically sound

#### 2. DSPy Usage Areas

**A. Intent Parsing & Ambiguity Resolution**:
   - Use ParseIntent(dspy.Signature) instead of hardcoded pattern matching
   - Extract structured intent with target_entities, edit_type, confidence, clarifying_questions
   - Handle ambiguity with confidence-based branching logic (<0.7 triggers questions)

**B. Prompt Generation & Refinement**:
   - Use GenerateBasePrompt(dspy.Signature) instead of prompt_templates dictionary
   - Implement 3-stage refinement with dspy.Refine for iterative improvement
   - Each iteration increases token diversity by 20%+

**C. Validation & Quality Assessment**:
   - Use ValidateEdit(dspy.Signature) instead of fixed threshold checking
   - Calculate alignment scores with statistical formulas
   - Implement retry logic with max 3 attempts

#### 3. Key DSPy Patterns:
   - ChainOfThought for complex reasoning tasks (IntentParser, PromptGenerator, Validator)
   - Refine for iterative improvement (3-stage prompt refinement)
   - BestOfN for variation generation (different rollout IDs for diversity)
   - Module composition for workflow orchestration (EditingPipeline)

#### 4. Why This Approach Works:
   - Reliability over speed through guided LLM reasoning
   - Adaptability to novel requests without hardcoded filters
   - Transparency via ChainOfThought explanations
   - Quality assurance through multi-stage validation
   - Better UX with clarifying questions vs silent failures

The core insight is that EDI uses structured DSPy reasoning to replace brittle hardcoded logic, ensuring flexibility and reliability through guided LLM interactions rather than rigid predefined libraries.