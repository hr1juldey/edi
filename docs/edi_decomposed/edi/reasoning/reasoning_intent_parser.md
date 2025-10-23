# Reasoning: Intent Parser

[Back to Reasoning Subsystem](./reasoning_subsystem.md)

## Purpose
DSpy intent extraction - Contains the IntentParser class (a dspy.Module) that processes naive prompts and scene analysis to extract structured intent.

## Class: IntentParser(dspy.Module)

### Methods
- `forward(naive_prompt, scene) -> Intent`: Takes a naive prompt and scene analysis and returns structured intent

### Details
- Detects ambiguity in user requests
- Generates clarifying questions when confidence is low
- Uses DSpy framework for structured LLM interactions

## AI Action Required: External Library Investigation

This module uses the **DSpy** library

1.  **Request for Information:** Provide a link to the official documentation for `DSpy`

2.  **Search and Test:** I will search for examples of `dspy.Signature` and `dspy.ChainOfThought` to verify my understanding before implementation

## Functions

- [forward(naive_prompt, scene)](./reasoning/forward_intent.md)

## Technology Stack

- DSpy for LLM orchestration
- Pydantic for data validation