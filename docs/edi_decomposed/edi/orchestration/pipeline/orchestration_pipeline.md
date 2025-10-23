# Orchestration: Pipeline

[Back to Orchestrator](./orchestrator.md)

## Purpose
Main editing pipeline - Contains the EditingPipeline class (a dspy.Module) that orchestrates the entire editing process from analysis to validation, with retry logic.

## Class: EditingPipeline(dspy.Module)

### Methods
- `forward(image_path, naive_prompt) -> EditResult`: Coordinates the entire editing process

### Details
- Orchestrates: analyze → parse → generate → execute → validate
- Handles retry logic (max 3 attempts)
- Uses DSpy framework for structured LLM interactions

## AI Action Required: External Library Investigation

This module uses the **DSpy** library

1.  **Request for Information:** Provide a link to the official documentation for `DSpy`

2.  **Search and Test:** I will search for examples of `dspy.Signature` and `dspy.ChainOfThought` to verify my understanding before implementation

## Functions

- [forward(image_path, naive_prompt)](./orchestration/forward_pipeline.md)

## Technology Stack

- DSpy for LLM orchestration
- Pydantic for data validation

## See Docs

```python
import dspy
import time
from typing import Dict, Any, Optional, List
import json

class EditingPipeline(dspy.Module):
    """
    Main editing pipeline - Orchestrates the entire editing process 
    from analysis to validation, with retry logic.
    """
    
    def __init__(self, analyzer=None, intent_parser=None, prompt_generator=None, 
                 prompt_refiner=None, validator=None, comfyui_client=None):
        """
        Initializes the editing pipeline with required components.
        """
        super().__init__()
        self.analyzer = analyzer
        self.intent_parser = intent_parser
        self.prompt_generator = prompt_generator
        self.prompt_refiner = prompt_refiner
        self.validator = validator
        self.comfyui_client = comfyui_client
        self.max_retries = 3
        self.ambiguity_threshold = 0.7

    def forward(self, image_path: str, naive_prompt: str) -> 'EditResult':
        """
        Coordinates the entire editing process.
        
        This method:
        - Orchestrates: analyze → parse → generate → execute → validate
        - Handles retry logic (max 3 attempts)
        - Uses DSpy framework for structured LLM interactions
        """
        start_time = time.time()
        
        # Stage 1: Analyze image - Call the vision subsystem to analyze the image
        try:
            scene = self.analyzer.analyze(image_path)
        except Exception as e:
            return EditResult(
                error=f"Failed to analyze image: {str(e)}",
                processing_time=time.time() - start_time
            )
        
        # Stage 2: Parse intent - Use the intent parser to extract structured intent
        try:
            intent = self.intent_parser(naive_prompt=naive_prompt, scene_analysis=scene.to_json())
        except Exception as e:
            return EditResult(
                error=f"Failed to parse intent: {str(e)}",
                processing_time=time.time() - start_time
            )
        
        # Stage 3: Handle ambiguity - If confidence < 0.7, ask clarifying questions
        if intent.get('confidence', 1.0) < self.ambiguity_threshold:
            # In a real implementation, this would involve user interaction
            # For this example, we'll proceed with the intent as is
            pass
        
        # Stage 4: Generate prompts - Create base positive/negative prompts
        try:
            base_prompts = self.prompt_generator(
                scene_analysis=scene.to_json(),
                intent=intent
            )
        except Exception as e:
            return EditResult(
                error=f"Failed to generate prompts: {str(e)}",
                processing_time=time.time() - start_time
            )
        
        # Stage 5: Refine prompts - Run 3-iteration refinement loop
        try:
            final_prompts = self.prompt_refiner(
                base_prompts=base_prompts,
                scene_analysis=scene.to_json(),
                intent=intent,
                iterations=3
            )
        except Exception as e:
            return EditResult(
                error=f"Failed to refine prompts: {str(e)}",
                processing_time=time.time() - start_time
            )
        
        # Stage 6: Execute edit - Send prompts to ComfyUI (with retry logic)
        retry_count = 0
        last_error = None
        
        while retry_count < self.max_retries:
            try:
                edited_image_path = self.comfyui_client.execute_edit(
                    image_path=image_path,
                    positive_prompt=final_prompts.get('positive', ''),
                    negative_prompt=final_prompts.get('negative', '')
                )
                
                # Stage 7: Validate result - Check if edit matches intent
                validation_result = self.validator.validate(
                    original_image_path=image_path,
                    edited_image_path=edited_image_path,
                    intent=intent
                )
                
                # If validation passes, return the result
                if validation_result.get('score', 0) >= self.ambiguity_threshold:
                    return EditResult(
                        edited_image_path=edited_image_path,
                        prompts=final_prompts,
                        validation_score=validation_result.get('score', 0),
                        processing_time=time.time() - start_time,
                        validation_details=validation_result
                    )
                else:
                    # Validation failed, but we'll try again if retries remain
                    retry_count += 1
                    if retry_count >= self.max_retries:
                        return EditResult(
                            edited_image_path=edited_image_path,
                            prompts=final_prompts,
                            validation_score=validation_result.get('score', 0),
                            processing_time=time.time() - start_time,
                            error="Validation failed after all retry attempts",
                            validation_details=validation_result
                        )
                    else:
                        # Retry with slightly modified parameters
                        continue
                        
            except Exception as e:
                retry_count += 1
                last_error = str(e)
                if retry_count >= self.max_retries:
                    return EditResult(
                        error=f"Edit execution failed after {self.max_retries} attempts: {last_error}",
                        processing_time=time.time() - start_time
                    )
        
        # This shouldn't be reached, but included for completeness
        return EditResult(
            error="Unexpected error in processing pipeline",
            processing_time=time.time() - start_time
        )

class EditResult:
    def __init__(self, 
                 edited_image_path: Optional[str] = None,
                 prompts: Optional[Dict[str, str]] = None,
                 validation_score: float = 0.0,
                 processing_time: float = 0.0,
                 error: Optional[str] = None,
                 validation_details: Optional[Dict] = None):
        """
        Represents the result of an editing operation.
        """
        self.edited_image_path = edited_image_path
        self.prompts = prompts or {}
        self.validation_score = validation_score
        self.processing_time = processing_time
        self.error = error
        self.validation_details = validation_details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Converts the EditResult to a dictionary representation."""
        return {
            'edited_image_path': self.edited_image_path,
            'prompts': self.prompts,
            'validation_score': self.validation_score,
            'processing_time': self.processing_time,
            'error': self.error,
            'validation_details': self.validation_details
        }

    def __repr__(self) -> str:
        """String representation of the EditResult."""
        return f"EditResult(path={self.edited_image_path}, score={self.validation_score}, error={self.error})"

# Example usage and mock classes for demonstration:
class MockAnalyzer:
    def analyze(self, image_path: str):
        class MockScene:
            def to_json(self):
                return {
                    "entities": ["sky", "trees", "mountains"],
                    "spatial_layout": "mountains in background, trees in middle ground, sky above"
                }
        return MockScene()

class MockIntentParser:
    def __call__(self, naive_prompt: str, scene_analysis: str):
        return {
            "target_entities": ["sky"],
            "edit_type": "color adjustment",
            "description": "make more dramatic",
            "confidence": 0.85
        }

class MockPromptGenerator:
    def __call__(self, scene_analysis: str, intent: Dict[str, Any]):
        return {
            "positive": f"Make the {intent['target_entities'][0]} more dramatic with vibrant colors",
            "negative": "avoid over-saturation, keep natural lighting"
        }

class MockPromptRefiner:
    def __call__(self, base_prompts: Dict[str, str], scene_analysis: str, 
                 intent: Dict[str, Any], iterations: int):
        # Simulate refinement process
        refined = base_prompts.copy()
        for i in range(iterations):
            refined['positive'] += f", iteration {i+1} refinement"
        return refined

class MockValidator:
    def validate(self, original_image_path: str, edited_image_path: str, intent: Dict[str, Any]):
        return {
            "score": 0.87,
            "reasoning": "Edit successfully modified target entities as specified"
        }

class MockComfyUIClient:
    def execute_edit(self, image_path: str, positive_prompt: str, negative_prompt: str):
        # Simulate creating an edited image
        import os
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        edited_path = f"edited_{base_name}.jpg"
        # In a real implementation, this would actually call ComfyUI
        # For this example, we'll just return a mock path
        return edited_path

if __name__ == "__main__":
    # Create mock components
    analyzer = MockAnalyzer()
    intent_parser = MockIntentParser()
    prompt_generator = MockPromptGenerator()
    prompt_refiner = MockPromptRefiner()
    validator = MockValidator()
    comfyui_client = MockComfyUIClient()
    
    # Create the pipeline
    pipeline = EditingPipeline(
        analyzer=analyzer,
        intent_parser=intent_parser,
        prompt_generator=prompt_generator,
        prompt_refiner=prompt_refiner,
        validator=validator,
        comfyui_client=comfyui_client
    )
    
    # Run the pipeline
    result = pipeline.forward(
        image_path="/path/to/image.jpg",
        naive_prompt="make the sky more dramatic"
    )
    
    print("Pipeline Result:")
    print(result.to_dict())
```