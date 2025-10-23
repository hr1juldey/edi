# EditingPipeline.forward()

[Back to Pipeline](../orchestration_pipeline.md)

## Related User Story
"As a user, I want EDI to understand my image's composition so it knows what can be safely edited." (from PRD)

## Function Signature
`forward(image_path: str, naive_prompt: str) -> EditResult`

## Parameters
- `image_path: str` - The file path to the image that needs to be edited
- `naive_prompt: str` - The user's casual edit request (e.g., "make the sky more dramatic")

## Returns
- `EditResult` - An object containing the result of the editing process

## Step-by-step Logic
1. **Stage 1: Analyze image** - Call the vision subsystem to analyze the image
   - scene = self.analyzer.analyze(image_path)
2. **Stage 2: Parse intent** - Use the intent parser to extract structured intent
   - intent = self.intent_parser(naive_prompt=naive_prompt, scene_analysis=scene.to_json())
3. **Stage 3: Handle ambiguity** - If confidence < 0.7, ask clarifying questions
   - Get user input and re-parse with additional context
4. **Stage 4: Generate prompts** - Create base positive/negative prompts
   - base_prompts = self.prompt_generator(...)
5. **Stage 5: Refine prompts** - Run 3-iteration refinement loop
   - final_prompts = self.prompt_refiner(...)
6. **Stage 6: Execute edit** - Send prompts to ComfyUI
7. **Stage 7: Validate result** - Check if edit matches intent
8. **Stage 8: Handle retries** - Up to 3 retry attempts if validation fails

## Orchestration Logic
- Coordinates multiple subsystems in the correct sequence
- Implements retry logic with max 3 attempts
- Handles ambiguity through clarifying questions
- Manages state throughout the process

## DSpy Integration
- Uses dspy.Module as base class for pipeline
- Integrates multiple DSpy modules (intent_parser, prompt_generator, etc.)
- Implements dspy.Refine for prompt improvement

## Input/Output Data Structures
### EditResult Object
An EditResult object contains:
- Final edited image path
- Generated prompts (positive and negative)
- Validation score
- Processing timeline
- Any error messages or warnings

## See Docs

```python
import dspy
import time
from typing import Dict, Any, Optional, List
import json

class EditingPipeline(dspy.Module):
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
        Main forward function that processes an image editing request through all stages.
        
        This method:
        1. Analyzes the image to understand its composition
        2. Parses the user's naive prompt into structured intent
        3. Handles ambiguity if confidence is low
        4. Generates and refines prompts 
        5. Executes the edit using ComfyUI
        6. Validates the result
        7. Implements retry logic if validation fails
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
        
        An EditResult object contains:
        - Final edited image path
        - Generated prompts (positive and negative)
        - Validation score
        - Processing timeline
        - Any error messages or warnings
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