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

# EDI: AI Image Editor - High-Level Design (HLD)

## Architecture Overview

### System Context

```bash
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
┌───────▼────────┐  ┌───────▼────────┐  ┌──────▼──────┐
│ Vision         │  │ Reasoning      │  │ Orchestrator│
│ Subsystem      │  │ Subsystem      │  │  (DSpy)     │
│ SAM+CLIP       │  │ Ollama LLMs    │  │             │
└───────┬────────┘  └───────┬────────┘  └──────┬──────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                    ┌───────▼────────┐
                    │   ComfyUI      │
                    │  (External)    │
                    └────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Technology |
|-----------|---------------|------------|
| **TUI Layer** | User interaction, display, navigation | Textual 0.87+ |
| **Vision Subsystem** | Image analysis, object detection, change detection | SAM 2.1, OpenCLIP |
| **Reasoning Subsystem** | Intent understanding, prompt generation, validation | Ollama (qwen3:8b, gemma3:4b) |
| **Orchestrator** | Workflow coordination, DSpy pipelines, state management | DSpy 2.6+ |
| **Storage Layer** | Session persistence, learning data | SQLite + JSON |
| **Integration Layer** | ComfyUI API client, image I/O | requests, Pillow |

## Component Design

### 1. Vision Subsystem

**Purpose**: Transform images into structured scene understanding

#### 1.1 Object Detection Module

**Inputs**:

- Image file path
- Optional region-of-interest hints from user

**Processing**:

```bash
1. Load image → PIL Image
2. SAM 2.1 automatic segmentation → List[Mask]
3. For each mask:
   a. Extract bounding box
   b. Crop region
   c. CLIP encode → embedding vector
4. Cluster masks by semantic similarity
5. Label clusters using CLIP text similarity
   (compare to predefined labels: "sky", "building", "person", etc.)
```

**Outputs**:

```python
SceneAnalysis(
    entities=[
        Entity(
            id="sky_0",
            label="sky",
            confidence=0.94,
            bbox=(0, 0, 1920, 760),  # XYXY format
            mask=ndarray,  # Binary mask
            color_dominant="#87CEEB",
            area_percent=39.6
        ),
        Entity(id="building_0", ...),
        ...
    ],
    spatial_layout="sky (top 40%), building (center 55%), grass (bottom 5%)"
)
```

**Performance Optimization**:

- Cache SAM model in memory (load once per session)
- Resize images >2048px to reduce processing time
- Skip fine-grained segmentation if <5% area (noise filtering)

#### 1.2 Change Detection Module

**Purpose**: Compare before/after images to validate edits

**Algorithm**:

```python
def compute_delta(before: SceneAnalysis, after: SceneAnalysis) -> EditDelta:
    # Match entities by spatial overlap (IoU > 0.5)
    matches = match_entities(before.entities, after.entities)
    
    preserved = []
    modified = []
    removed = []
    added = []
    
    for before_entity, after_entity in matches:
        if after_entity is None:
            removed.append(before_entity)
        elif entities_similar(before_entity, after_entity):
            preserved.append((before_entity, after_entity))
        else:
            modified.append((before_entity, after_entity))
    
    for entity in after.entities:
        if entity not in [m[1] for m in matches]:
            added.append(entity)
    
    return EditDelta(
        preserved=preserved,
        modified=modified,
        removed=removed,
        added=added,
        alignment_score=calculate_alignment(...)
    )
```

**Similarity Metrics**:

- Color: ΔE2000 < 10 (perceptually similar)
- Position: Center shift < 5% of image dimension
- Shape: Mask IoU > 0.85

### 2. Reasoning Subsystem

**Purpose**: Translate user intent to technical specifications

#### 2.1 Intent Parser

**DSpy Module**:

```python
class ParseIntent(dspy.Signature):
    """
    Extract structured intent from casual user prompt.
    """
    naive_prompt = dspy.InputField(
        desc="User's conversational edit request"
    )
    scene_analysis = dspy.InputField(
        desc="JSON of detected entities and layout"
    )
    
    target_entities = dspy.OutputField(
        desc="Comma-separated list of entity IDs to edit"
    )
    edit_type = dspy.OutputField(
        desc="One of: color, style, add, remove, transform"
    )
    confidence = dspy.OutputField(
        desc="Float 0-1 indicating clarity of intent"
    )
    clarifying_questions = dspy.OutputField(
        desc="JSON array of questions if confidence <0.7"
    )
```

**Usage**:

```python
parser = dspy.ChainOfThought(ParseIntent)
result = parser(
    naive_prompt="make the sky more dramatic",
    scene_analysis=json.dumps(analysis)
)

if result.confidence < 0.7:
    # Show clarifying questions to user
    display_questions(result.clarifying_questions)
```

#### 2.2 Prompt Generator

**DSpy Pipeline** (3-stage refinement):

##### **Stage 1: Initial Generation**

```python
class GenerateBasePrompt(dspy.Signature):
    naive_prompt = dspy.InputField()
    scene_analysis = dspy.InputField()
    target_entities = dspy.InputField()
    edit_type = dspy.InputField()
    
    positive_prompt = dspy.OutputField(
        desc="Technical prompt for desired changes"
    )
    negative_prompt = dspy.OutputField(
        desc="Technical prompt for things to avoid"
    )
```

##### **Stage 2-4: Iterative Refinement**

```python
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

**Refinement Strategy**:

```bash
Iteration 1: Add preservation constraints
Iteration 2: Increase technical specificity
Iteration 3: Add quality/style modifiers
```

**Prompt Template** (for positive prompt):

```bash
{edit_description}, {technical_terms}, {quality_modifiers},
preserve: {entities_to_keep}, maintain composition
```

Example output:

```bash
Positive: "dramatic storm clouds, cumulonimbus formation, 
          dark gray nimbus, volumetric lighting, overcast mood,
          photorealistic, 8k detail, preserve building structure,
          maintain foreground subjects"
          
Negative: "sunny sky, blue sky, bright lighting, lens flare,
          building color changes, grass modifications,
          cartoon style, oversaturated, artifacts"
```

### 3. Orchestrator (DSpy Pipeline Manager)

**Purpose**: Coordinate multi-step workflows with branching logic

#### 3.1 Main Editing Pipeline

```python
class EditingPipeline(dspy.Module):
    def __init__(self):
        self.analyzer = VisionSubsystem()
        self.intent_parser = dspy.ChainOfThought(ParseIntent)
        self.prompt_generator = dspy.ChainOfThought(GenerateBasePrompt)
        self.prompt_refiner = dspy.Refine(
            RefinePrompt,
            N=3,  # 3 refinement iterations
            reward_fn=prompt_quality_score
        )
        self.validator = dspy.ChainOfThought(ValidateEdit)
        
    def forward(self, image_path: str, naive_prompt: str):
        # Stage 1: Analyze image
        scene = self.analyzer.analyze(image_path)
        
        # Stage 2: Parse intent
        intent = self.intent_parser(
            naive_prompt=naive_prompt,
            scene_analysis=scene.to_json()
        )
        
        # Stage 3: Handle ambiguity
        if intent.confidence < 0.7:
            user_input = self.ask_clarifying_questions(
                intent.clarifying_questions
            )
            # Re-parse with additional context
            intent = self.intent_parser(
                naive_prompt=f"{naive_prompt}. User clarified: {user_input}",
                scene_analysis=scene.to_json()
            )
        
        # Stage 4: Generate prompts
        base_prompts = self.prompt_generator(
            naive_prompt=naive_prompt,
            scene_analysis=scene.to_json(),
            target_entities=intent.target_entities,
            edit_type=intent.edit_type
        )
        
        # Stage 5: Refine prompts
        final_prompts = self.prompt_refiner(
            naive_prompt=naive_prompt,
            previous_positive=base_prompts.positive_prompt,
            previous_negative=base_prompts.negative_prompt,
            refinement_goal="maximize technical quality and preservation"
        )
        
        return final_prompts
```

#### 3.2 Validation Loop

```python
class ValidationLoop:
    def execute(self, original_image, edited_image, expected_changes):
        # Re-analyze edited image
        before_scene = self.analyzer.analyze(original_image)
        after_scene = self.analyzer.analyze(edited_image)
        
        # Compute delta
        delta = compute_delta(before_scene, after_scene)
        
        # Calculate alignment score
        score = self.calculate_alignment(delta, expected_changes)
        
        if score >= 0.8:
            return ValidationResult(
                status="ACCEPT",
                score=score,
                message="Edit matches intent"
            )
        elif score >= 0.6:
            return ValidationResult(
                status="REVIEW",
                score=score,
                message="Partial match - user decision required"
            )
        else:
            return ValidationResult(
                status="RETRY",
                score=score,
                message="Poor match - regenerating prompts",
                retry_hints=self.generate_retry_hints(delta)
            )
```

### 4. TUI Layer (Textual Application)

**Architecture**: Screen-based navigation with reactive widgets

#### 4.1 Screen Hierarchy

```bash
HomeScreen
├── ImageUploadScreen
│   ├── FileInput widget
│   ├── PreviewPane widget
│   └── AnalysisProgressBar widget
├── PromptInputScreen
│   ├── TextArea widget (for naive prompt)
│   ├── EntitySelectorList widget (checkboxes)
│   └── SubmitButton widget
├── ClarificationScreen
│   ├── QuestionLabel widget
│   ├── OptionsRadioSet widget
│   └── ConfirmButton widget
├── RefinementScreen
│   ├── IterationProgressBar widget
│   ├── PromptDiffViewer widget (shows evolution)
│   └── ApproveRejectButtons widget
├── ResultsScreen
│   ├── ImageComparisonPane (side-by-side)
│   ├── ValidationMetricsTable widget
│   ├── AcceptRetryButtons widget
│   └── FeedbackTextArea widget
└── MultiVariationScreen
    ├── GridLayout (3 columns)
    ├── VariationCards (A/B/C)
    └── SelectionControls widget
```

#### 4.2 Key Widgets

**ImageComparisonPane**:

```python
class ImageComparisonPane(Widget):
    """
    Side-by-side image viewer with overlay support.
    """
    def compose(self):
        yield Container(
            Container(id="before-pane"),
            Container(id="after-pane"),
            id="comparison-container"
        )
    
    def render_images(self, before_path, after_path):
        # Convert images to ANSI art using Rich
        before_art = image_to_ansi_art(before_path, max_width=40)
        after_art = image_to_ansi_art(after_path, max_width=40)
        
        self.query_one("#before-pane").update(before_art)
        self.query_one("#after-pane").update(after_art)
```

**PromptDiffViewer**:

```python
class PromptDiffViewer(Widget):
    """
    Shows prompt evolution across refinement iterations.
    """
    def display_refinement(self, iteration, positive, negative):
        # Highlight added tokens in green, removed in red
        diff_positive = self.compute_diff(
            previous=self.prompts[iteration-1].positive,
            current=positive
        )
        
        self.render_diff(diff_positive, panel_title=f"Positive (v{iteration})")
```

#### 4.3 Navigation Flow

```bash
┌──────────────────────────────────────────────────────────┐
│  [1] Upload Image       [2] Recent Sessions   [Q] Quit   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│                   EDI: Edit with Intelligence            │
│                                                          │
│  Welcome! Let's edit your image together.                │
│                                                          │
│  > [1] Start new edit                                    │
│    [2] Resume session                                    │
│    [3] View examples                                     │
│    [H] Help                                              │
│                                                          │
│                                                          │
│  Navigation: Arrow keys / Numbers                        │
│  Quick actions: [Q]uit  [H]elp  [B]ack                   │
└──────────────────────────────────────────────────────────┘
```

**Keyboard Shortcuts**:

- Global: `Q` quit, `H` help, `B` back, `Ctrl+C` cancel operation
- Navigation: Arrow keys, Tab/Shift+Tab
- Actions: Numbers (1-9) for quick selection, Enter to confirm
- Editing: `E` edit prompt, `R` retry, `A` accept, `V` view variations

### 5. Storage Layer

**Purpose**: Persist session data for learning and resume functionality

#### 5.1 Database Schema (SQLite)

```sql
-- sessions table
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,  -- UUID
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    image_path TEXT NOT NULL,
    naive_prompt TEXT NOT NULL,
    status TEXT CHECK(status IN ('in_progress', 'completed', 'failed')),
    final_alignment_score REAL
);

-- prompts table (stores refinement history)
CREATE TABLE prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id),
    iteration INT,
    positive_prompt TEXT,
    negative_prompt TEXT,
    quality_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- entities table (detected objects)
CREATE TABLE entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id),
    entity_id TEXT,  -- e.g., "sky_0"
    label TEXT,
    confidence REAL,
    bbox_json TEXT,  -- Serialized bounding box
    mask_path TEXT,  -- Path to saved mask file
    color_hex TEXT,
    area_percent REAL
);

-- validations table (edit assessment)
CREATE TABLE validations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id),
    attempt_number INT,
    alignment_score REAL,
    preserved_count INT,
    modified_count INT,
    unintended_count INT,
    user_feedback TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- user_feedback table (for learning)
CREATE TABLE user_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id),
    feedback_type TEXT CHECK(feedback_type IN ('accept', 'reject', 'partial')),
    comments TEXT,
    rating INT CHECK(rating BETWEEN 1 AND 5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 5.2 Session State Management

**State File** (JSON for current session):

```json
{
  "session_id": "uuid-here",
  "current_stage": "refinement",
  "image_path": "/path/to/image.jpg",
  "naive_prompt": "make sky dramatic",
  "scene_analysis": {
    "entities": [...],
    "spatial_layout": "..."
  },
  "intent": {
    "target_entities": ["sky_0"],
    "edit_type": "style",
    "confidence": 0.85
  },
  "prompts": {
    "iteration_0": {
      "positive": "...",
      "negative": "..."
    },
    "iteration_1": {...},
    "final": {...}
  },
  "edited_image_path": "/path/to/edited.jpg",
  "validation": {
    "score": 0.87,
    "delta": {...}
  }
}
```

**Auto-save**: Write state file every 5 seconds or after significant events

### 6. Integration Layer

#### 6.1 ComfyUI Client

**API Wrapper**:

```python
class ComfyUIClient:
    def __init__(self, base_url="http://localhost:8188"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def submit_edit(
        self,
        image_path: str,
        positive_prompt: str,
        negative_prompt: str,
        workflow_template: str = "img2img_default"
    ) -> str:
        """
        Submit edit job to ComfyUI and return job ID.
        """
        workflow = self.load_workflow(workflow_template)
        
        # Inject parameters
        workflow['nodes']['positive_prompt']['text'] = positive_prompt
        workflow['nodes']['negative_prompt']['text'] = negative_prompt
        workflow['nodes']['input_image']['path'] = image_path
        
        response = self.session.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow}
        )
        
        return response.json()['prompt_id']
    
    def poll_status(self, job_id: str) -> dict:
        """
        Check job status (returns 'queued', 'processing', 'completed', 'failed').
        """
        response = self.session.get(
            f"{self.base_url}/history/{job_id}"
        )
        return response.json()
    
    def download_result(self, job_id: str, output_path: str):
        """
        Download completed edit to local file.
        """
        status = self.poll_status(job_id)
        if status['status'] != 'completed':
            raise RuntimeError(f"Job not completed: {status['status']}")
        
        image_url = status['outputs'][0]['images'][0]['filename']
        
        response = self.session.get(
            f"{self.base_url}/view?filename={image_url}",
            stream=True
        )
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)
```

**Workflow Templates** (stored in `workflows/` directory):

- `img2img_default.json`: Standard image-to-image with prompts
- `inpaint_masked.json`: Masked inpainting for region-specific edits
- `controlnet_canny.json`: Structure-preserving edits using ControlNet

## Data Flow Diagrams

### Primary Edit Flow

```bash
┌─────────┐
│  User   │
│ Input   │
└────┬────┘
     │
     │ 1. image_path + naive_prompt
     ▼
┌─────────────────────────────────┐
│   Vision Subsystem              │
│   ├─ Load & preprocess image    │
│   ├─ SAM 2.1 segmentation       │
│   ├─ CLIP entity labeling       │
│   └─ Generate SceneAnalysis     │
└────┬────────────────────────────┘
     │
     │ 2. SceneAnalysis
     ▼
┌─────────────────────────────────┐
│   Intent Parser (DSpy)          │
│   ├─ Analyze naive_prompt       │
│   ├─ Map to entities            │
│   ├─ Detect ambiguity           │
│   └─ Generate questions if <70% │
└────┬────────────────────────────┘
     │
     │ 3. Intent + Questions
     ▼
┌─────────────────────────────────┐
│   TUI Clarification Screen      │
│   [IF confidence < 0.7]          │
│   ├─ Display questions           │
│   ├─ Collect user responses      │
│   └─ Update intent context       │
└────┬────────────────────────────┘
     │
     │ 4. Refined Intent
     ▼
┌─────────────────────────────────┐
│   Prompt Generator (DSpy)       │
│   ├─ Generate base prompts      │
│   ├─ Refine iteration 1          │
│   ├─ Refine iteration 2          │
│   └─ Refine iteration 3          │
└────┬────────────────────────────┘
     │
     │ 5. Final Prompts
     ▼
┌─────────────────────────────────┐
│   TUI Review Screen             │
│   ├─ Display prompts             │
│   ├─ Show refinement evolution   │
│   └─ [User Approval Required]    │
└────┬────────────────────────────┘
     │
     │ 6. Approved Prompts
     ▼
┌─────────────────────────────────┐
│   ComfyUI Client                │
│   ├─ Submit edit job             │
│   ├─ Poll status (async)         │
│   └─ Download result             │
└────┬────────────────────────────┘
     │
     │ 7. edited_image_path
     ▼
┌─────────────────────────────────┐
│   Validation Loop               │
│   ├─ Re-analyze edited image     │
│   ├─ Compute delta vs original   │
│   ├─ Calculate alignment score   │
│   └─ Determine action             │
└────┬────────────────────────────┘
     │
     ├─ IF score >= 0.8 → Accept
     │
     ├─ IF 0.6 <= score < 0.8 → Ask User
     │
     └─ IF score < 0.6 → Retry (max 3x)
                         │
                         └─────► Return to step 4 with retry hints
```

### Multi-Variation Flow

```bash
[Same steps 1-4 as above]
     │
     │ 5. Base Prompts
     ▼
┌─────────────────────────────────┐
│   BestOfN Generator (DSpy)      │
│   ├─ Generate variation A        │
│   ├─ Generate variation B        │
│   └─ Generate variation C        │
│       (using different rollout   │
│        IDs for diversity)        │
└────┬────────────────────────────┘
     │
     │ 6. Prompts A, B, C
     ▼
┌─────────────────────────────────┐
│   ComfyUI Parallel Submission   │
│   ├─ Submit job A                │
│   ├─ Submit job B                │
│   └─ Submit job C                │
│       (wait for all completions) │
└────┬────────────────────────────┘
     │
     │ 7. edited_A, edited_B, edited_C
     ▼
┌─────────────────────────────────┐
│   TUI Variation Grid            │
│   ├─ Display all 3 side-by-side │
│   ├─ User selects best           │
│   └─ Option: blend regions       │
└────┬────────────────────────────┘
     │
     │ 8. Selected variation OR blend instructions
     ▼
[IF blend requested]
┌─────────────────────────────────┐
│   Region Compositor             │
│   ├─ Extract masks for regions   │
│   ├─ Poisson blend transition    │
│   └─ Generate composite image    │
└────┬────────────────────────────┘
     │
     ▼
[Final validation as in primary flow]
```

## Technology Stack Details

### Development Environment

```yaml
Python Version: "3.10+"
Virtual Environment: venv or conda

Core Dependencies:
  textual: ">=0.87.0"  # TUI framework
  dspy: ">=2.6.0"      # LLM orchestration
  ultralytics: ">=8.3.0"  # SAM 2.1
  open-clip-torch: ">=3.2.0"  # CLIP
  torch: ">=2.0.0"     # PyTorch (with CUDA 12.1)
  torchvision: ">=0.15.0"
  Pillow: ">=12.0.0"   # Image processing
  numpy: ">=2.0.0"
  opencv-python: ">=4.11.0"  # Computer vision ops
  scikit-image: ">=0.25.0"   # Image metrics
  requests: ">=2.32.0"  # HTTP client
  pydantic: ">=2.0.0"  # Data validation
  
Development Tools:
  pytest: ">=8.0.0"
  pytest-asyncio: ">=0.23.0"
  black: ">=24.0.0"    # Code formatter
  mypy: ">=1.8.0"      # Type checker
  ruff: ">=0.1.0"      # Linter
```

### Model Selection Strategy

**Reasoning LLM** (user-configurable):

- Default: `qwen3:8b` (best balance of quality/speed)
- Fallback: `gemma3:4b` (if OOM)
- Alternative: `mistral:7b` (if user prefers)

**Vision LLM** (for optional VLM fallback):

- Default: `gemma3:4b` (vision capabilities)
- Alternative: `qwen2.5vl:7b` (higher quality, slower)

**Segmentation**:

- SAM 2.1 (Base model): `sam2.1_b.pt`
- Alternative: SAM 2.1 Tiny (`sam2.1_t.pt`) for speed

**CLIP**:

- Model: `ViT-B/32`
- Pretrained: `openai` (or fallback to `laion2b_s34b_b79k`)

### Performance Tuning

**GPU Memory Management**:

```python
# Load models sequentially, clear cache between
torch.cuda.empty_cache()

# Use half precision where possible
model = model.half()  # FP16 instead of FP32

# Tile large images for SAM
if image.size > (2048, 2048):
    tiles = split_into_tiles(image, overlap=128)
    masks = [sam(tile) for tile in tiles]
    combined = merge_tiles(masks)
```

**Async Operations**:

```python
# Use Textual's worker threads for long operations
@work(exclusive=True)  # Ensures single operation at a time
async def analyze_image(self, path):
    result = await asyncio.to_thread(
        vision_subsystem.analyze, path
    )
    self.post_message(AnalysisComplete(result))
```

## Deployment Considerations

### Installation (User Perspective)

```bash
# 1. Clone repository
git clone https://github.com/user/edi.git
cd edi

# 2. Install dependencies
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .

# 3. Download models
edi setup --download-models

# 4. Verify installation
edi doctor

# 5. Run
edi edit photo.jpg "make sky dramatic"
```

### Configuration Files

**`~/.edi/config.yaml`**:

```yaml
models:
  reasoning_llm: "qwen3:8b"
  vision_llm: "gemma3:4b"
  sam_checkpoint: "sam2.1_b.pt"
  clip_model: "ViT-B/32"

performance:
  max_image_size: 2048
  use_half_precision: true
  enable_model_caching: true

comfyui:
  base_url: "http://localhost:8188"
  default_workflow: "img2img_default"
  timeout_seconds: 180

ui:
  theme: "dark"  # or "light"
  animation_speed: "normal"  # or "slow", "fast"
  show_debug_info: false

storage:
  database_path: "~/.edi/sessions.db"
  max_session_history: 100
  auto_cleanup_days: 30
```

### Testing Strategy

**Unit Tests** (pytest):

```bash
tests/
├── unit/
│   ├── test_vision_subsystem.py
│   ├── test_intent_parser.py
│   ├── test_prompt_generator.py
│   ├── test_validation.py
│   └── test_storage.py
├── integration/
│   ├── test_editing_pipeline.py
│   ├── test_comfyui_client.py
│   └── test_tui_navigation.py
└── fixtures/
    ├── sample_images/
    └── mock_models.py
```

**Test Coverage Targets**:

- Vision subsystem: 90%+ (critical path)
- Reasoning subsystem: 85%+
- TUI widgets: 70%+ (snapshot tests)

**Continuous Testing**:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=edi --cov-report=html

# Run specific subsystem
pytest tests/unit/test_vision_subsystem.py -v
```

## Security & Privacy

**Data Handling**:

- All processing local (no cloud uploads)
- Images never transmitted outside localhost
- Session data sanitized (no PII extraction)
- User can purge all data: `edi clear --all`

**Model Safety**:

- Models downloaded from trusted sources only
- SHA256 checksum verification
- Sandboxed execution (no arbitrary code execution)

## Extensibility

### Plugin System (Future)

**Custom Analyzers**:

```python
from edi.plugins import AnalyzerPlugin

class FaceDetectionAnalyzer(AnalyzerPlugin):
    def analyze(self, image: Image) -> List[Entity]:
        # Custom face detection logic
        faces = detect_faces(image)
        return [Entity(label="face", bbox=f.bbox) for f in faces]

# Register plugin
edi.register_analyzer(FaceDetectionAnalyzer())
```

**Custom Prompt Templates**:

```python
from edi.plugins import PromptTemplate

class PhotographyTemplate(PromptTemplate):
    def generate(self, intent) -> tuple[str, str]:
        positive = f"professional photography, {intent.edit_type}, ..."
        negative = "amateur, snapshot, ..."
        return positive, negative
```

---

# File Structure & Module Descriptions

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

This completes the Product Requirements Document and High-Level Design for EDI. The file structure and module descriptions provide a clear implementation roadmap for a solo developer to build a working POC in 24 hours, with all components grounded in proven technologies (SAM 2.1, CLIP, DSpy, Textual) and no ambiguous "AI magic" features.
