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