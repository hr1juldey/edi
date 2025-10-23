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