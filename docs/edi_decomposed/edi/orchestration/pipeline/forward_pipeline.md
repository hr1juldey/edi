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