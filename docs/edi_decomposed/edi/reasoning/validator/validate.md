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