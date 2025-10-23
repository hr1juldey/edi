# Validator.validate()

[Back to Validator](../reasoning_validator.md)

## Related User Story
"As a user, I want EDI to check if the edit matches my intent and learn from my corrections." (from PRD)

## Function Signature
`validate(delta, intent) -> ValidationResult`

## Parameters
- `delta` - The EditDelta object containing changes between before/after images
- `intent` - The original structured intent containing the user's edit request

## Returns
- `ValidationResult` - An object containing status (ACCEPT/REVIEW/RETRY), score, and message

## Step-by-step Logic
1. Take the EditDelta and original intent as input
2. Calculate alignment score using the formula:
   - Alignment Score = (0.4 × Entities Preserved Correctly + 0.4 × Intended Changes Applied + 0.2 × (1 - Unintended Changes))
3. Based on the score:
   - If score >= 0.8 → return ACCEPT status
   - If score >= 0.6 → return REVIEW status (user decision required)
   - If score < 0.6 → return RETRY status (regenerating prompts)
4. If RETRY status, generate retry hints to improve next attempt
5. Return ValidationResult with status, score, and appropriate message

## Validation Criteria
- High score (>0.8): Edit matches intent well
- Medium score (0.6-0.8): Partial match, user decision needed
- Low score (<0.6): Poor match, regeneration needed

## Retry Logic
- When score is low, generate hints for better prompts
- Incorporates feedback from delta analysis
- Helps improve subsequent attempts

## Input/Output Data Structures
### ValidationResult Object
A ValidationResult object contains:
- Status (ACCEPT, REVIEW, or RETRY)
- Score (0-1 alignment score)
- Message (explanation of result)
- Retry hints (if status is RETRY)

## See Docs

```python
from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass

# Define type aliases for clarity
ValidationStatus = Literal["ACCEPT", "REVIEW", "RETRY"]

@dataclass
class ValidationResult:
    """
    A ValidationResult object contains:
    - Status (ACCEPT, REVIEW, or RETRY)
    - Score (0-1 alignment score)
    - Message (explanation of result)
    - Retry hints (if status is RETRY)
    """
    status: ValidationStatus
    score: float
    message: str
    retry_hints: Optional[list] = None

class EditDelta:
    """
    Represents the changes between before/after images.
    Contains information about what changed in the image.
    """
    def __init__(self, 
                 entities_preserved_correctly: float,
                 intended_changes_applied: float, 
                 unintended_changes: float,
                 change_details: Optional[Dict[str, Any]] = None):
        self.entities_preserved_correctly = entities_preserved_correctly
        self.intended_changes_applied = intended_changes_applied
        self.unintended_changes = unintended_changes
        self.change_details = change_details or {}

class Validator:
    def __init__(self, accept_threshold: float = 0.8, review_threshold: float = 0.6):
        self.accept_threshold = accept_threshold
        self.review_threshold = review_threshold

    def validate(self, delta: EditDelta, intent: Dict[str, Any]) -> ValidationResult:
        """
        Validates whether the edit matches the original intent.
        
        This method:
        1. Takes the EditDelta and original intent as input
        2. Calculates alignment score using the formula:
           - Alignment Score = (0.4 × Entities Preserved Correctly + 0.4 × Intended Changes Applied + 0.2 × (1 - Unintended Changes))
        3. Based on the score:
           - If score >= 0.8 → return ACCEPT status
           - If score >= 0.6 → return REVIEW status (user decision required)
           - If score < 0.6 → return RETRY status (regenerating prompts)
        4. If RETRY status, generate retry hints to improve next attempt
        5. Return ValidationResult with status, score, and appropriate message
        """
        # Calculate alignment score using the specified formula
        # Alignment Score = (0.4 × Entities Preserved Correctly + 0.4 × Intended Changes Applied + 0.2 × (1 - Unintended Changes))
        alignment_score = (
            0.4 * delta.entities_preserved_correctly +
            0.4 * delta.intended_changes_applied + 
            0.2 * (1 - delta.unintended_changes)
        )
        
        # Ensure the score is within bounds [0, 1]
        alignment_score = max(0.0, min(1.0, alignment_score))
        
        # Determine status based on score
        if alignment_score >= self.accept_threshold:
            status = "ACCEPT"
            message = f"Edit matches intent well (score: {alignment_score:.2f}). The changes align with your request."
        elif alignment_score >= self.review_threshold:
            status = "REVIEW"
            message = f"Edit partially matches intent (score: {alignment_score:.2f}). Please review the result and decide if it meets your requirements."
        else:
            status = "RETRY"
            message = f"Edit does not match intent well (score: {alignment_score:.2f}). Regenerating prompts for a better result."
        
        # Generate retry hints if needed
        retry_hints = None
        if status == "RETRY":
            retry_hints = self._generate_retry_hints(delta, intent)
        
        return ValidationResult(
            status=status,
            score=alignment_score,
            message=message,
            retry_hints=retry_hints
        )
    
    def _generate_retry_hints(self, delta: EditDelta, intent: Dict[str, Any]) -> list:
        """
        Generate hints to improve the next attempt when validation fails.
        """
        hints = []
        
        # Check if entities were not preserved correctly
        if delta.entities_preserved_correctly < 0.5:
            target_entities = intent.get('target_entities', [])
            if target_entities:
                hints.append(f"Preserve non-target entities like {', '.join([e for e in ['background', 'foreground'] if e not in target_entities])} better")
        
        # Check if intended changes were not applied
        if delta.intended_changes_applied < 0.7:
            edit_type = intent.get('edit_type', 'edit')
            edit_description = intent.get('description', 'changes')
            hints.append(f"Focus more on applying the {edit_type}: {edit_description}")
        
        # Check for unintended changes
        if delta.unintended_changes > 0.3:
            hints.append("Reduce unintended changes outside the target area")
        
        # Add a general hint if no specific ones apply
        if not hints:
            hints.append("Try a different approach or more specific prompt")
        
        return hints

# Example usage:
if __name__ == "__main__":
    # Create a sample intent
    intent = {
        "target_entities": ["sky"],
        "edit_type": "color adjustment", 
        "description": "make more dramatic",
        "confidence": 0.85
    }
    
    # Create a sample EditDelta (simulating changes between before/after images)
    delta = EditDelta(
        entities_preserved_correctly=0.8,  # 80% of non-target entities preserved
        intended_changes_applied=0.7,      # 70% of intended changes applied
        unintended_changes=0.2             # 20% unintended changes occurred
    )
    
    # Create validator and run validation
    validator = Validator()
    result = validator.validate(delta, intent)
    
    print(f"Validation Result:")
    print(f"  Status: {result.status}")
    print(f"  Score: {result.score:.2f}")
    print(f"  Message: {result.message}")
    if result.retry_hints:
        print(f"  Retry Hints: {result.retry_hints}")
    
    # Example with low score (should trigger RETRY)
    print("\n" + "="*50)
    print("Example with low score (should trigger RETRY):")
    
    low_delta = EditDelta(
        entities_preserved_correctly=0.3,
        intended_changes_applied=0.4,
        unintended_changes=0.7
    )
    
    low_result = validator.validate(low_delta, intent)
    print(f"  Status: {low_result.status}")
    print(f"  Score: {low_result.score:.2f}")
    print(f"  Message: {low_result.message}")
    if low_result.retry_hints:
        print(f"  Retry Hints: {low_result.retry_hints}")