# Reasoning: Prompt Generator

[Back to Reasoning Subsystem](./reasoning_subsystem.md)

## Purpose
DSpy prompt creation - Contains the PromptGenerator class (a dspy.Module) that creates positive and negative prompts based on intent and scene, with 3 refinement iterations.

## Class: PromptGenerator(dspy.Module)

### Methods
- `forward(intent, scene) -> Prompts`: Generates initial prompts based on intent and scene analysis

### Details
- Base generation followed by 3 refinement iterations
- Creates both positive and negative prompts
- Uses DSpy framework for structured LLM interactions

## AI Action Required: External Library Investigation

This module uses the **DSpy** library

1.  **Request for Information:** Provide a link to the official documentation for `DSpy`

2.  **Search and Test:** I will search for examples of `dspy.Signature` and `dspy.ChainOfThought` to verify my understanding before implementation

## Functions

- [forward(intent, scene)](./reasoning/forward_prompts.md)

## Technology Stack

- DSpy for LLM orchestration
- Pydantic for data validation