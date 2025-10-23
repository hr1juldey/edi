# Reasoning: Models

[Back to Reasoning Subsystem](./reasoning_subsystem.md)

## Purpose
Pydantic models for reasoning subsystem - Contains Intent, Prompts, and ValidationResult data structures for type-safe operations.

## Models
- `Intent`: Represents structured user intent with target entities and edit type
- `Prompts`: Contains positive and negative prompts for image generation
- `ValidationResult`: Represents the result of edit validation

### Details
- Type-safe data structures
- Pydantic validation
- JSON serialization support

## Technology Stack

- Pydantic for data validation
- Type hints for type safety

## See Docs

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class EditType(str, Enum):
    """Enumeration of possible edit types."""
    COLOR = "color"
    STYLE = "style"
    ADD = "add"
    REMOVE = "remove"
    TRANSFORM = "transform"

class Intent(BaseModel):
    """
    Represents structured user intent with target entities and edit type.
    """
    target_entities: List[str] = Field(
        ..., 
        description="List of entity IDs to edit (e.g., ['sky', 'trees'])",
        min_items=1
    )
    edit_type: EditType = Field(
        ..., 
        description="Type of edit to perform"
    )
    description: str = Field(
        ..., 
        description="Human-readable description of the intended edit"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Confidence score (0-1) indicating clarity of intent"
    )
    clarifying_questions: Optional[List[str]] = Field(
        default_factory=list,
        description="List of clarifying questions if confidence is low"
    )
    
    class Config:
        use_enum_values = True  # Use string values instead of enum objects for JSON serialization

class Prompts(BaseModel):
    """
    Contains positive and negative prompts for image generation.
    """
    positive: str = Field(
        ..., 
        min_length=1,
        description="Technical prompt for desired changes"
    )
    negative: str = Field(
        default="",
        description="Technical prompt for things to avoid"
    )
    history: Optional[List[dict]] = Field(
        default_factory=list,
        description="History of refinement iterations (for transparency)"
    )

class ValidationResult(BaseModel):
    """
    Represents the result of edit validation.
    """
    status: str = Field(
        ..., 
        regex=r"^(ACCEPT|REVIEW|RETRY)$",
        description="Validation status: ACCEPT, REVIEW, or RETRY"
    )
    score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Alignment score (0-1) between edit and intent"
    )
    message: str = Field(
        ..., 
        description="Human-readable explanation of result"
    )
    retry_hints: Optional[List[str]] = Field(
        default_factory=list,
        description="Hints for improving the next attempt (if status is RETRY)"
    )

# Example usage
if __name__ == "__main__":
    # Example of creating an Intent
    intent = Intent(
        target_entities=["sky", "clouds"],
        edit_type=EditType.COLOR,
        description="make the sky more dramatic and vibrant",
        confidence=0.85
    )
    print("Intent:", intent.json(indent=2))
    
    # Example of creating Prompts
    prompts = Prompts(
        positive="dramatic blue sky with fluffy white clouds, high detail, professional lighting",
        negative="blurry, pixelated, over-saturated, unnatural colors",
        history=[
            {"iteration": "base", "positive": "blue sky", "negative": "blurry"},
            {"iteration": "refined", "positive": "dramatic blue sky", "negative": "blurry, pixelated"}
        ]
    )
    print("\\nPrompts:", prompts.json(indent=2))
    
    # Example of creating a ValidationResult
    validation_result = ValidationResult(
        status="ACCEPT",
        score=0.87,
        message="Edit matches intent well with dramatic sky improvement"
    )
    print("\\nValidation Result:", validation_result.json(indent=2))
    
    # Example with low confidence intent that needs clarification
    low_confidence_intent = Intent(
        target_entities=["object"],
        edit_type=EditType.TRANSFORM,
        description="make it better",
        confidence=0.3,
        clarifying_questions=[
            "Which specific object would you like to modify?",
            "What kind of transformation are you looking for?",
            "Are there any parts that should remain unchanged?"
        ]
    )
    print("\\nLow Confidence Intent:", low_confidence_intent.json(indent=2))
```