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
