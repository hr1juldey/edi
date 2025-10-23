# Orchestrator

[Back to Index](../index.md)

## Purpose

Workflow coordination, DSpy pipelines, state management using DSpy 2.6+

## Component Design

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

## AI Action Required: External Library Investigation

This module uses the **DSpy** library

1. **Request for Information:** Provide a link to the official documentation for `DSpy`

2. **Search and Test:** I will search for examples of `dspy.Signature` and `dspy.ChainOfThought` to verify my understanding before implementation

## Sub-modules

This component includes the following modules:

- [orchestration/pipeline.py](./pipeline/pipeline.md)
- [orchestration/variation_generator.py](./variation_generator/variation_generator.md)
- [orchestration/compositor.py](./compositor/compositor.md)
- [orchestration/state_manager.py](./state_manager/state_manager.md)

## Technology Stack

- DSpy 2.6+ for LLM orchestration
- Pydantic for data validation

## See Docs

### DSpy Implementation Example

Orchestrator pipeline for the EDI application:

```python
import dspy
import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from dataclasses import dataclass

# Define DSPy signatures for our orchestrator
class ValidateEdit(dspy.Signature):
    """Validate if an edit matches user intent."""
    original_image_description: str = dspy.InputField()
    edited_image_description: str = dspy.InputField()
    user_intent: str = dspy.InputField()
    
    alignment_score: float = dspy.OutputField(desc="0.0 to 1.0, higher is better")
    preserved_elements: List[str] = dspy.OutputField(desc="Elements that should be preserved")
    modified_elements: List[str] = dspy.OutputField(desc="Elements that were modified")
    unintended_changes: List[str] = dspy.OutputField(desc="Unintended changes that occurred")

class GenerateRetryHints(dspy.Signature):
    """Generate hints for retrying an edit that didn't meet expectations."""
    original_image_description: str = dspy.InputField()
    edited_image_description: str = dspy.InputField()
    user_intent: str = dspy.InputField()
    alignment_score: float = dspy.InputField(desc="Current alignment score")
    
    retry_hints: List[str] = dspy.OutputField(desc="Suggestions for improving the edit")

class ValidationResult(BaseModel):
    status: str = Field(..., regex=r'^(ACCEPT|REVIEW|RETRY)
)
    score: float = Field(ge=0.0, le=1.0)
    message: str
    retry_hints: List[str] = []

# Pydantic models for structured data
class SceneAnalysis(BaseModel):
    entities: List[Dict[str, Any]]
    spatial_layout: str
    
    def to_json(self) -> str:
        return self.model_dump_json()

class PromptPair(BaseModel):
    positive_prompt: str
    negative_prompt: str

# Mock vision subsystem for demonstration
class VisionSubsystem:
    def analyze(self, image_path: str) -> SceneAnalysis:
        # In a real implementation, this would use actual vision analysis
        # For demo purposes, return a mock analysis
        return SceneAnalysis(
            entities=[
                {"id": "sky_0", "label": "sky", "confidence": 0.95},
                {"id": "mountain_1", "label": "mountain", "confidence": 0.87}
            ],
            spatial_layout="sky occupies top half, mountains at horizon"
        )

# DSPy modules for the orchestrator pipeline
class EDIEditingPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyzer = VisionSubsystem()
        self.intent_parser = dspy.ChainOfThought("naive_prompt, scene_analysis -> target_entities, edit_type, confidence, clarifying_questions")
        self.prompt_generator = dspy.ChainOfThought("naive_prompt, scene_analysis, target_entities, edit_type -> positive_prompt, negative_prompt")
        self.prompt_refiner = dspy.ChainOfThought("naive_prompt, previous_positive, previous_negative, refinement_goal -> refined_positive, refined_negative, improvement_explanation")
        self.validator = dspy.ChainOfThought(ValidateEdit)
        
    def forward(self, image_path: str, naive_prompt: str):
        # Stage 1: Analyze image
        scene = self.analyzer.analyze(image_path)
        
        # Stage 2: Parse intent
        intent_result = self.intent_parser(
            naive_prompt=naive_prompt,
            scene_analysis=scene.to_json()
        )
        
        # Stage 3: Handle ambiguity
        confidence = float(intent_result.confidence) if intent_result.confidence.replace('.', '', 1).isdigit() else 0.0
        
        if confidence < 0.7:
            # For demo purposes, we'll simulate asking clarifying questions
            # In a real system, this would involve user interaction
            user_input = "Focus on the sky and make it stormy with dramatic clouds"
            # Re-parse with additional context
            intent_result = self.intent_parser(
                naive_prompt=f"{naive_prompt}. User clarified: {user_input}",
                scene_analysis=scene.to_json()
            )
        
        # Stage 4: Generate prompts
        target_entities = intent_result.target_entities
        edit_type = intent_result.edit_type
        base_prompts = self.prompt_generator(
            naive_prompt=naive_prompt,
            scene_analysis=scene.to_json(),
            target_entities=target_entities,
            edit_type=edit_type
        )
        
        # Stage 5: Refine prompts
        refinement_goals = [
            "add preservation constraints",
            "increase technical specificity",
            "add quality/style modifiers"
        ]
        
        positive_prompt = base_prompts.positive_prompt
        negative_prompt = base_prompts.negative_prompt
        
        for goal in refinement_goals:
            refine_result = self.prompt_refiner(
                naive_prompt=naive_prompt,
                previous_positive=positive_prompt,
                previous_negative=negative_prompt,
                refinement_goal=goal
            )
            positive_prompt = refine_result.refined_positive
            negative_prompt = refine_result.refined_negative
        
        return PromptPair(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt
        )

class EDIValidationLoop:
    def __init__(self):
        self.analyzer = VisionSubsystem()
        self.validator = dspy.ChainOfThought(ValidateEdit)
        self.retry_generator = dspy.ChainOfThought(GenerateRetryHints)
    
    def execute(self, 
                original_image_path: str, 
                edited_image_path: str, 
                expected_changes: str,
                original_prompt: str) -> ValidationResult:
        # Re-analyze both images
        original_scene = self.analyzer.analyze(original_image_path)
        edited_scene = self.analyzer.analyze(edited_image_path)
        
        # Use DSPy to validate the edit
        validation_result = self.validator(
            original_image_description=original_scene.to_json(),
            edited_image_description=edited_scene.to_json(),
            user_intent=expected_changes
        )
        
        # Calculate alignment score (in real scenario, this comes from DSPy model)
        try:
            score = float(validation_result.alignment_score)
        except ValueError:
            score = 0.5  # default value if conversion fails
        
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
            # Generate retry hints for low-scoring edits
            retry_hints_result = self.retry_generator(
                original_image_description=original_scene.to_json(),
                edited_image_description=edited_scene.to_json(),
                user_intent=expected_changes,
                alignment_score=str(score)
            )
            
            return ValidationResult(
                status="RETRY",
                score=score,
                message="Poor match - regenerating prompts",
                retry_hints=retry_hints_result.retry_hints
            )

# Complete orchestrator example
class EDIOrchestrator:
    def __init__(self):
        self.editing_pipeline = EDIEditingPipeline()
        self.validation_loop = EDIValidationLoop()
    
    def process_edit_request(self, image_path: str, naive_prompt: str):
        """Process an edit request from start to finish."""
        print(f"Processing edit request: '{naive_prompt}' for image {image_path}")
        
        # Generate prompts using the editing pipeline
        prompts = self.editing_pipeline.forward(image_path, naive_prompt)
        print(f"Generated prompts:")
        print(f"  Positive: {prompts.positive_prompt}")
        print(f"  Negative: {prompts.negative_prompt}")
        
        # In a real system, we would now use these prompts to generate an edited image
        # For this example, we'll simulate having an edited image
        edited_image_path = image_path.replace('.jpg', '_edited.jpg')
        
        # Validate the results
        result = self.validation_loop.execute(
            original_image_path=image_path,
            edited_image_path=edited_image_path,
            expected_changes=naive_prompt,
            original_prompt=naive_prompt
        )
        
        print(f"Validation result: {result.status} (score: {result.score})")
        if result.retry_hints:
            print(f"Retry hints: {result.retry_hints}")
        
        return result

# Example usage
if __name__ == "__main__":
    # Initialize the orchestrator
    orchestrator = EDIOrchestrator()
    
    # Example usage
    result = orchestrator.process_edit_request(
        image_path="landscape.jpg",
        naive_prompt="make the sky more dramatic with storm clouds"
    )
```

### Pydantic Implementation Example

Data validation for the orchestrator:

```python
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from enum import Enum
import json

class StatusEnum(str, Enum):
    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    RETRY = "RETRY"

class ValidationResult(BaseModel):
    status: StatusEnum
    score: float = Field(ge=0.0, le=1.0)
    message: str
    retry_hints: List[str] = []

class SceneEntity(BaseModel):
    id: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: Optional[List[float]] = None  # [x, y, width, height]
    mask_path: Optional[str] = None

class SceneAnalysis(BaseModel):
    entities: List[SceneEntity]
    spatial_layout: str
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)

class EditRequest(BaseModel):
    image_path: str
    naive_prompt: str
    session_id: Optional[str] = None
    target_entities: Optional[List[str]] = []
    
    @validator('image_path')
    def validate_image_path(cls, v):
        if not v.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            raise ValueError('image_path must be a valid image file')
        return v

class PromptPair(BaseModel):
    positive_prompt: str
    negative_prompt: str
    iteration: int = 0

class EditHistory(BaseModel):
    request: EditRequest
    scene_analysis: SceneAnalysis
    prompts: List[PromptPair]
    validation_result: Optional[ValidationResult] = None
    created_at: str

class OrchestratorState(BaseModel):
    session_id: str
    current_stage: str = Field(..., regex=r'^(upload|analysis|intent|generation|validation|completed)
)
    edit_history: List[EditHistory] = []
    current_iteration: int = 0
    
    def add_history_entry(self, entry: EditHistory):
        self.edit_history.append(entry)
        self.current_iteration += 1

# Example usage
if __name__ == "__main__":
    # Example scene entity
    entity = SceneEntity(
        id="sky_0",
        label="sky",
        confidence=0.95,
        bbox=[0, 0, 100, 50]
    )
    print(f"Scene entity: {entity}")
    
    # Example scene analysis
    scene = SceneAnalysis(
        entities=[entity],
        spatial_layout="sky occupies top half of image",
        quality_score=0.85
    )
    print(f"Scene analysis: {scene}")
    
    # Example edit request
    request = EditRequest(
        image_path="skyline.jpg",
        naive_prompt="make the sky more dramatic",
        target_entities=["sky_0"]
    )
    print(f"Edit request: {request}")
    
    # Example validation result
    validation = ValidationResult(
        status=StatusEnum.REVIEW,
        score=0.75,
        message="Partial match - user decision required"
    )
    print(f"Validation result: {validation}")
    
    # Example edit history entry
    history_entry = EditHistory(
        request=request,
        scene_analysis=scene,
        prompts=[
            PromptPair(
                positive_prompt="dramatic storm clouds, cumulonimbus formation",
                negative_prompt="sunny sky, blue sky, bright lighting",
                iteration=0
            )
        ],
        validation_result=validation,
        created_at="2023-10-23T10:00:00Z"
    )
    
    # Example orchestrator state
    state = OrchestratorState(
        session_id="session-123",
        current_stage="validation"
    )
    state.add_history_entry(history_entry)
    
    print(f"Orchestrator state: {state}")
    
    # Save and load example
    state_json = state.model_dump_json(indent=2)
    print("Serialized state:")
    print(state_json)
    
    # Load from JSON
    loaded_state = OrchestratorState.model_validate_json(state_json)
    print(f"Loaded state session ID: {loaded_state.session_id}")
```
