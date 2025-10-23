# IntentParser.forward()

[Back to Intent Parser](../reasoning_intent_parser.md)

## Related User Story
"As a user, I want EDI to ask questions when my request is ambiguous rather than guessing." (from PRD)

## Function Signature
`forward(naive_prompt, scene) -> Intent`

## Parameters
- `naive_prompt` - The user's casual edit request (e.g., "make the sky more dramatic")
- `scene` - The scene analysis containing detected entities and layout

## Returns
- `Intent` - A structured intent object with target entities, edit type, confidence, and clarifying questions if needed

## Step-by-step Logic
1. Parse the naive prompt for ambiguity markers ("dramatic", "better", "more")
2. Apply DSpy ChainOfThought reasoning to extract structured intent
3. Identify target entities to edit based on the prompt and scene analysis
4. Determine edit type (one of: color, style, add, remove, transform)
5. Calculate confidence score (0-1) indicating clarity of intent
6. If confidence < 0.7, generate clarifying questions using DSpy
7. Return structured intent with target entities, edit type, confidence, and questions

## DSpy Components Used
- dspy.Signature for structured input/output
- dspy.ChainOfThought for multi-step reasoning
- Confidence scoring for ambiguity detection

## Ambiguity Handling
- Detects vague terms that could have multiple interpretations
- Generates specific, answerable questions (1-5 options)
- Maintains context for refined understanding

## Input/Output Data Structures
### Intent Object
An Intent object contains:
- Target entities (comma-separated list of entity IDs to edit)
- Edit type (color, style, add, remove, transform)
- Confidence score (0-1)
- Clarifying questions (JSON array if confidence < 0.7)