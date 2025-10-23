# Reasoning: Validator

[Back to Reasoning Subsystem](./reasoning_subsystem.md)

## Purpose
Edit quality assessment - Contains the Validator class that calculates alignment scores and generates retry hints if scores are low.

## Class: Validator

### Methods
- `validate(delta, intent) -> ValidationResult`: Evaluates the quality of an edit based on the delta and original intent

### Details
- Calculates alignment scores for validation
- Generates retry hints if score is low
- Helps determine if an edit matches user intent

## Functions

- [validate(delta, intent)](./reasoning/validate.md)

## Technology Stack

- Pydantic for data validation
- NumPy for calculations