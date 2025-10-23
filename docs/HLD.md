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
│   [IF confidence < 0.7]         │
│   ├─ Display questions          │
│   ├─ Collect user responses     │
│   └─ Update intent context      │
└────┬────────────────────────────┘
     │
     │ 4. Refined Intent
     ▼
┌─────────────────────────────────┐
│   Prompt Generator (DSpy)       │
│   ├─ Generate base prompts      │
│   ├─ Refine iteration 1         │
│   ├─ Refine iteration 2         │
│   └─ Refine iteration 3         │
└────┬────────────────────────────┘
     │
     │ 5. Final Prompts
     ▼
┌─────────────────────────────────┐
│   TUI Review Screen             │
│   ├─ Display prompts            │
│   ├─ Show refinement evolution  │
│   └─ [User Approval Required]   │
└────┬────────────────────────────┘
     │
     │ 6. Approved Prompts
     ▼
┌─────────────────────────────────┐
│   ComfyUI Client                │
│   ├─ Submit edit job            │
│   ├─ Poll status (async)        │
│   └─ Download result            │
└────┬────────────────────────────┘
     │
     │ 7. edited_image_path
     ▼
┌─────────────────────────────────┐
│   Validation Loop               │
│   ├─ Re-analyze edited image    │
│   ├─ Compute delta vs original  │
│   ├─ Calculate alignment score  │
│   └─ Determine action           │
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
│   ├─ Generate variation A       │
│   ├─ Generate variation B       │
│   └─ Generate variation C       │
│       (using different rollout  │
│        IDs for diversity)       │
└────┬────────────────────────────┘
     │
     │ 6. Prompts A, B, C
     ▼
┌─────────────────────────────────┐
│   ComfyUI Parallel Submission   │
│   ├─ Submit job A               │
│   ├─ Submit job B               │
│   └─ Submit job C               │
│       (wait for all completions)│
└────┬────────────────────────────┘
     │
     │ 7. edited_A, edited_B, edited_C
     ▼
┌─────────────────────────────────┐
│   TUI Variation Grid            │
│   ├─ Display all 3 side-by-side │
│   ├─ User selects best          │
│   └─ Option: blend regions      │
└────┬────────────────────────────┘
     │
     │ 8. Selected variation OR blend instructions
     ▼
[IF blend requested]
┌─────────────────────────────────┐
│   Region Compositor             │
│   ├─ Extract masks for regions  │
│   ├─ Poisson blend transition   │
│   └─ Generate composite image   │
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
