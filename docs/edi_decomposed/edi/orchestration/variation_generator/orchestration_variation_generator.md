# Orchestration: Variation Generator

[Back to Orchestrator](./orchestrator.md)

## Purpose
Multi-variation support - Contains the VariationGenerator class that creates multiple prompt variations using DSpy BestOfN with different rollout IDs.

## Class: VariationGenerator

### Methods
- `generate_variations(intent, N=3) -> List[Prompts]`: Generates N different prompt variations based on the intent

### Details
- Uses DSpy BestOfN with different rollout IDs for diversity
- Creates multiple interpretations of the same intent
- Supports Best-of-N selection for better results

## AI Action Required: External Library Investigation

This module uses the **DSpy** library

1.  **Request for Information:** Provide a link to the official documentation for `DSpy`

2.  **Search and Test:** I will search for examples of `dspy.Signature` and `dspy.ChainOfThought` to verify my understanding before implementation

## Functions

- [generate_variations(intent, N=3)](./orchestration/generate_variations.md)

## Technology Stack

- DSpy for LLM orchestration
- Pydantic for data validation