# validate_prompt()

[Back to Validators](../validators.md)

## Related User Story
"As a user, I want EDI to handle my inputs safely and appropriately." (from PRD - implied by reliability and security requirements)

## Function Signature
`validate_prompt(text) -> bool`

## Parameters
- `text` - The prompt text to validate

## Returns
- `bool` - True if the prompt is valid, False otherwise

## Step-by-step Logic
1. Check if the input text is not empty or just whitespace
2. Validate that the text contains valid characters and doesn't include problematic sequences
3. Check for potential security issues like code injection patterns
4. Verify that the prompt length is within acceptable limits
5. Apply any custom validation rules specific to image editing prompts
6. Return True if all validations pass, False otherwise

## Validation Criteria
- Non-empty text with meaningful content
- No potentially harmful characters or sequences
- Length within configured limits (not too short or too long)
- Appropriate content for image editing context
- Proper encoding without issues

## Security Considerations
- Prevents code injection through prompts
- Blocks potentially harmful sequences
- Validates character encoding properly
- Ensures safe processing by downstream systems

## Input/Output Data Structures
### Input
- text: String containing the prompt to validate

### Output
- Boolean indicating validity (True for valid, False for invalid)