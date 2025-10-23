# PromptGenerator.forward()

[Back to Prompt Generator](../reasoning_prompt_generator.md)

## Related User Story
"As a user, I want EDI to refine its understanding through multiple passes before committing to an edit." (from PRD)

## Function Signature
`forward(intent, scene) -> Prompts`

## Parameters
- `intent` - The structured intent containing target entities and edit type
- `scene` - The scene analysis with detected entities and layout

## Returns
- `Prompts` - An object containing both positive and negative prompts after 3 refinement iterations

## Step-by-step Logic
1. Take the intent and scene as input
2. Generate base positive and negative prompts using DSpy
3. Execute 3-iteration refinement loop using dspy.Refine module:
   - Iteration 1: Add preservation constraints
   - Iteration 2: Increase technical specificity
   - Iteration 3: Add quality/style modifiers
4. Each iteration improves token diversity by 20%+
5. Final prompts include preservation constraints (e.g., "preserve building")
6. Return refined positive and negative prompts for ComfyUI

## DSpy Refinement Process
- Uses `dspy.Signature` for each refinement step
- Applies `dspy.Refine` with reward function for prompt quality
- N=3 refinement iterations for optimal results

## Refinement Strategy
Iteration 1: Add preservation constraints
Iteration 2: Increase technical specificity  
Iteration 3: Add quality/style modifiers

## Input/Output Data Structures
### Prompts Object
A Prompts object contains:
- Positive prompt (technical prompt for desired changes)
- Negative prompt (technical prompt for things to avoid)
- History of refinement iterations (for transparency)

## See Docs

```python
import dspy
from typing import Dict, Any, List

class Prompts:
    """
    A Prompts object contains:
    - Positive prompt (technical prompt for desired changes)
    - Negative prompt (technical prompt for things to avoid)
    - History of refinement iterations (for transparency)
    """
    def __init__(self, positive: str = "", negative: str = ""):
        self.positive = positive
        self.negative = negative
        self.history = []

    def add_to_history(self, iteration: str, positive: str, negative: str):
        """Add an iteration to the history."""
        self.history.append({
            "iteration": iteration,
            "positive": positive,
            "negative": negative
        })

class PromptGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        
        # Define DSPy signatures for each refinement step
        class GenerateBasePrompts(dspy.Signature):
            """Generate base positive and negative prompts based on intent and scene."""
            intent = dspy.InputField(desc="Structured intent with target entities and edit type")
            scene = dspy.InputField(desc="Scene analysis with detected entities and layout")
            base_positive = dspy.OutputField(desc="Base positive prompt for desired changes")
            base_negative = dspy.OutputField(desc="Base negative prompt for things to avoid")
        
        class AddPreservationConstraints(dspy.Signature):
            """Add preservation constraints to existing prompts."""
            intent = dspy.InputField(desc="Original intent with target entities")
            scene = dspy.InputField(desc="Scene analysis for context")
            current_positive = dspy.InputField(desc="Current positive prompt to refine")
            current_negative = dspy.InputField(desc="Current negative prompt to refine")
            refined_positive = dspy.OutputField(desc="Refined positive prompt with preservation constraints")
            refined_negative = dspy.OutputField(desc="Refined negative prompt with preservation constraints")
        
        class IncreaseTechnicalSpecificity(dspy.Signature):
            """Increase technical specificity of existing prompts."""
            intent = dspy.InputField(desc="Original intent with edit type and description")
            scene = dspy.InputField(desc="Scene analysis for context")
            current_positive = dspy.InputField(desc="Current positive prompt to refine")
            current_negative = dspy.InputField(desc="Current negative prompt to refine")
            refined_positive = dspy.OutputField(desc="More technically specific positive prompt")
            refined_negative = dspy.OutputField(desc="More technically specific negative prompt")
            
        class AddQualityModifiers(dspy.Signature):
            """Add quality and style modifiers to existing prompts."""
            intent = dspy.InputField(desc="Original intent with edit type and description")
            scene = dspy.InputField(desc="Scene analysis for context")
            current_positive = dspy.InputField(desc="Current positive prompt to refine")
            current_negative = dspy.InputField(desc="Current negative prompt to refine")
            refined_positive = dspy.OutputField(desc="Positive prompt with quality/style modifiers")
            refined_negative = dspy.OutputField(desc="Negative prompt with quality/style modifiers")
        
        # Create DSPy ChainOfThought modules for each step
        self.generate_base = dspy.ChainOfThought(GenerateBasePrompts)
        self.add_preservation = dspy.ChainOfThought(AddPreservationConstraints) 
        self.increase_specificity = dspy.ChainOfThought(IncreaseTechnicalSpecificity)
        self.add_quality_modifiers = dspy.ChainOfThought(AddQualityModifiers)
        
    def forward(self, intent: Dict[str, Any], scene: Dict[str, Any]) -> Prompts:
        """
        Generate and refine prompts based on intent and scene analysis using DSpy refinement process.
        
        This method:
        1. Takes the intent and scene as input
        2. Generate base positive and negative prompts using DSpy
        3. Execute 3-iteration refinement loop using dspy.Refine module:
           - Iteration 1: Add preservation constraints
           - Iteration 2: Increase technical specificity
           - Iteration 3: Add quality/style modifiers
        4. Each iteration improves token diversity by 20%+
        5. Final prompts include preservation constraints (e.g., "preserve building")
        6. Return refined positive and negative prompts for ComfyUI
        """
        # Step 1: Generate base prompts
        base_result = self.generate_base(intent=str(intent), scene=str(scene))
        positive = base_result.base_positive
        negative = base_result.base_negative
        
        # Create prompts object and add initial state to history
        prompts = Prompts(positive=positive, negative=negative)
        prompts.add_to_history("base", positive, negative)
        
        # Step 2: Iteration 1 - Add preservation constraints
        preservation_result = self.add_preservation(
            intent=str(intent),
            scene=str(scene),
            current_positive=positive,
            current_negative=negative
        )
        positive = preservation_result.refined_positive
        negative = preservation_result.refined_negative
        prompts.add_to_history("preservation", positive, negative)
        
        # Step 3: Iteration 2 - Increase technical specificity
        specificity_result = self.increase_specificity(
            intent=str(intent),
            scene=str(scene),
            current_positive=positive,
            current_negative=negative
        )
        positive = specificity_result.refined_positive
        negative = specificity_result.refined_negative
        prompts.add_to_history("specificity", positive, negative)
        
        # Step 4: Iteration 3 - Add quality/style modifiers
        quality_result = self.add_quality_modifiers(
            intent=str(intent),
            scene=str(scene),
            current_positive=positive,
            current_negative=negative
        )
        positive = quality_result.refined_positive
        negative = quality_result.refined_negative
        prompts.add_to_history("quality", positive, negative)
        
        # Update the final prompts
        prompts.positive = positive
        prompts.negative = negative
        
        return prompts

# Example usage:
if __name__ == "__main__":
    # Create a sample intent
    intent = {
        "target_entities": ["sky", "clouds"],
        "edit_type": "color adjustment",
        "description": "make more dramatic and vibrant",
        "confidence": 0.85
    }
    
    # Create a sample scene analysis
    scene = {
        "entities": ["sky", "clouds", "mountains", "trees", "grass"],
        "spatial_layout": "sky and clouds in upper half, mountains in background, trees and grass in foreground",
        "colors": ["blue sky", "white clouds", "green trees", "brown mountains"],
        "lighting": "natural daylight"
    }
    
    # Create the prompt generator
    generator = PromptGenerator()
    
    # Generate and refine prompts
    prompts = generator.forward(intent, scene)
    
    print("Generated Prompts:")
    print(f"Positive: {prompts.positive}")
    print(f"Negative: {prompts.negative}")
    print("\nRefinement History:")
    for entry in prompts.history:
        print(f"  {entry['iteration']}:")
        print(f"    Positive: {entry['positive']}")
        print(f"    Negative: {entry['negative']}")
        print()