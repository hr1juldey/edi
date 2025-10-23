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

## See Docs

### Python Implementation Example
Implementation of the validate_prompt function:

```python
import re
from typing import List, Tuple
import unicodedata

def validate_prompt(text: str) -> bool:
    """
    Validates a prompt text for safety and appropriateness.
    
    Args:
        text: String containing the prompt to validate
    
    Returns:
        Boolean indicating validity (True for valid, False for invalid)
    """
    # Check if the input text is not empty or just whitespace
    if not text or not text.strip():
        return False
    
    # Validate that the text contains valid characters and doesn't include problematic sequences
    if not _has_valid_characters(text):
        return False
    
    # Check for potential security issues like code injection patterns
    if _has_security_issues(text):
        return False
    
    # Verify that the prompt length is within acceptable limits
    if not _is_length_valid(text):
        return False
    
    # Apply custom validation rules specific to image editing prompts
    if not _is_appropriate_for_image_editing(text):
        return False
    
    # All validations passed
    return True

def _has_valid_characters(text: str) -> bool:
    """
    Checks if the text contains only valid characters.
    """
    # Check for control characters (except common whitespace)
    control_chars = [char for char in text if ord(char) < 32 and char not in "\t\n\r"]
    if control_chars:
        return False
    
    # Check if the text contains valid Unicode (not just ASCII)
    try:
        text.encode('utf-8').decode('utf-8')
    except UnicodeDecodeError:
        return False
    
    return True

def _has_security_issues(text: str) -> bool:
    """
    Checks for potential security issue patterns in the text.
    """
    # Common patterns that could indicate code injection attempts
    dangerous_patterns = [
        r'<script',  # HTML script tags
        r'javascript:',  # JavaScript URLs
        r'vbscript:',  # VBScript URLs
        r'on\w+\s*=',  # HTML event handlers
        r'\{\{',  # Template injection (e.g., Jinja2)
        r'%\{',  # Another template pattern
        r'exec\(',  # Code execution functions
        r'eval\(',  # Code evaluation functions
        r'import\s+',  # Import statements
        r'__import__',  # Magic import function
        r'os\.',  # OS module access
        r'subprocess.',  # Subprocess module access
        r'file\(',  # File operations
        r'open\([^)]*w',  # Writing files
        r'\|.*\|',  # Potential shell command pipes
    ]
    
    text_lower = text.lower()
    for pattern in dangerous_patterns:
        if re.search(pattern, text_lower):
            return True  # Has security issue
    
    return False

def _is_length_valid(text: str, min_length: int = 1, max_length: int = 2000) -> bool:
    """
    Checks if the text length is within acceptable limits.
    """
    length = len(text.strip())
    return min_length <= length <= max_length

def _is_appropriate_for_image_editing(text: str) -> bool:
    """
    Applies custom validation rules specific to image editing prompts.
    """
    # Check for inappropriate or meaningless content
    text_lower = text.lower().strip()
    
    # Common patterns that are not meaningful for image editing
    meaningless_patterns = [
        r'^\s*[xX\-\.\_]+\s*,  # Just symbols like "xxx", "---", "..."
        r'^(the\s+)+\w*,  # Just "the the the..."
        r'^(a\s+)+\w*,  # Just "a a a..."
        r'^[A-Z][a-z]*\s+[A-Z][a-z]*,  # Just two capitalized words
    ]
    
    for pattern in meaningless_patterns:
        if re.match(pattern, text_lower):
            return False
    
    # Additional checks for image editing context
    # For example, check that the prompt is not just a single character repeated
    if len(set(text_lower)) == 1 and len(text) > 5:
        return False
    
    # Check for excessive repetition of words or phrases
    words = text_lower.split()
    if len(words) > 10:  # Only check longer texts
        unique_words = set(words)
        if len(unique_words) / len(words) < 0.3:  # Less than 30% unique words
            return False
    
    return True

def validate_prompt_detailed(text: str) -> Tuple[bool, List[str]]:
    """
    Validates a prompt and returns detailed feedback about validation failures.
    
    Args:
        text: String containing the prompt to validate
    
    Returns:
        A tuple containing (is_valid: bool, failure_reasons: List[str])
    """
    failure_reasons = []
    
    # Check if the input text is not empty or just whitespace
    if not text or not text.strip():
        failure_reasons.append("Prompt is empty or contains only whitespace")
        return False, failure_reasons
    
    # Validate that the text contains valid characters
    if not _has_valid_characters(text):
        failure_reasons.append("Prompt contains invalid characters")
    
    # Check for potential security issues
    if _has_security_issues(text):
        failure_reasons.append("Prompt contains potentially unsafe patterns")
    
    # Verify that the prompt length is within acceptable limits
    if not _is_length_valid(text):
        failure_reasons.append(f"Prompt length must be between {1} and {2000} characters")
    
    # Apply custom validation rules
    if not _is_appropriate_for_image_editing(text):
        failure_reasons.append("Prompt is not appropriate for image editing context")
    
    is_valid = len(failure_reasons) == 0
    return is_valid, failure_reasons

def sanitize_prompt(text: str) -> str:
    """
    Sanitize a prompt by normalizing and cleaning it, while preserving its meaning.
    """
    if not text:
        return ""
    
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    # Remove excessive whitespace while preserving intent
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Remove potential harmful characters that passed initial validation
    # but are still not appropriate
    text = re.sub(r'[<>"\']', '', text)
    
    return text

def validate_and_sanitize_prompt(text: str) -> Tuple[bool, str]:
    """
    Validates and sanitizes a prompt in one operation.
    
    Args:
        text: String containing the prompt to validate and sanitize
    
    Returns:
        A tuple containing (is_valid: bool, sanitized_text: str)
    """
    # First sanitize
    sanitized = sanitize_prompt(text)
    
    # Then validate the sanitized version
    is_valid = validate_prompt(sanitized)
    
    return is_valid, sanitized

# Example usage
if __name__ == "__main__":
    test_prompts = [
        "A beautiful landscape with mountains",  # Valid
        "",  # Invalid: empty
        "   ",  # Invalid: whitespace only
        "Make <script>alert('xss')</script> the sky blue",  # Invalid: security issue
        "This is a very long prompt that might be too long for the system to handle properly, so we need to make sure it's within the allowed length limits which are typically 2000 characters or less",  # May be invalid: too long
        "Beautiful sunset with mountains",  # Valid
        "xxx",  # Invalid: meaningless
        "add 1+1 and then do math",  # Invalid: contains code-like pattern
    ]
    
    print("Prompt Validation Results:")
    for prompt in test_prompts:
        is_valid, reasons = validate_prompt_detailed(prompt)
        print(f"Prompt: '{prompt[:30]}{'...' if len(prompt) > 30 else ''}'")
        print(f"  Valid: {is_valid}")
        if not is_valid:
            print(f"  Reasons: {reasons}")
        print()
    
    # Test sanitization
    print("Sanitization Examples:")
    to_sanitize = ' A   lot   of    spaces  <script>bad</script> '
    is_valid, sanitized = validate_and_sanitize_prompt(to_sanitize)
    print(f"Original: '{to_sanitize}'")
    print(f"Sanitized: '{sanitized}'")
    print(f"Valid after sanitization: {is_valid}")
```

### Advanced Prompt Validation Implementation
Enhanced validation with more sophisticated checks:

```python
import re
import unicodedata
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

@dataclass
class ValidationResult:
    """Represents the result of validation."""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    code: str

class AdvancedPromptValidator:
    """
    Advanced prompt validator with configurable rules and detailed reporting.
    """
    
    def __init__(self):
        # Configuration for validation rules
        self.config = {
            "min_length": 3,
            "max_length": 2000,
            "max_consecutive_chars": 5,
            "max_consecutive_words": 3,
            "forbidden_patterns": [
                r"eval\(",
                r"exec\(",
                r"import\s+\w+",
                r"__import__",
                r"open\([^)]*w",
                r"subprocess",
                r"os\.",
                r"sys\.",
                r"shutil",
                r"socket",
                r"pickle",
                r"execfile",
                r"compile",
                r"file\(",
            ],
            "suspicious_patterns": [
                r"system\(",
                r"input\(",  # In some contexts could be risky
                r"raw_input\(",
                r"globals\(\)",
                r"locals\(\)",
                r"__.*__",  # Magic methods/attributes
            ],
            "forbidden_words": [
                "password", "username", "login", "secret", "token", "api_key",
                "key", "credential", "authentication", "auth_token", "sessionid"
            ],
            "whitelist_chars": set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?;:-_()[]{}'\"/&@#"),
            "encoding": "utf-8"
        }
    
    def validate(self, text: str) -> Tuple[bool, List[ValidationResult]]:
        """
        Performs comprehensive validation on the prompt text.
        """
        results = []
        
        # Check basic requirements
        results.extend(self._check_basic_requirements(text))
        
        # Check for security issues
        results.extend(self._check_security_issues(text))
        
        # Check for content appropriateness
        results.extend(self._check_content_appropriateness(text))
        
        # Check for quality issues
        results.extend(self._check_quality_issues(text))
        
        # Overall validity is True only if no ERROR-level issues exist
        is_valid = not any(r.severity == ValidationSeverity.ERROR for r in results)
        
        return is_valid, results
    
    def _check_basic_requirements(self, text: str) -> List[ValidationResult]:
        """Check basic requirements like non-empty and length."""
        results = []
        
        # Check if empty or whitespace only
        if not text or not text.strip():
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Prompt cannot be empty or contain only whitespace",
                code="EMPTY_PROMPT"
            ))
            return results  # Early return since empty prompt fails everything
        
        # Check length
        length = len(text.strip())
        if length < self.config["min_length"]:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Prompt too short: {length} characters (minimum: {self.config['min_length']})",
                code="TOO_SHORT"
            ))
        elif length > self.config["max_length"]:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Prompt too long: {length} characters (maximum: {self.config['max_length']})",
                code="TOO_LONG"
            ))
        
        return results
    
    def _check_security_issues(self, text: str) -> List[ValidationResult]:
        """Check for potential security issues like code injection."""
        results = []
        text_lower = text.lower()
        
        # Check forbidden patterns
        for pattern in self.config["forbidden_patterns"]:
            if re.search(pattern, text_lower):
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Prompt contains potentially dangerous pattern: {pattern}",
                    code="SECURITY_ISSUE"
                ))
                # Don't check further patterns if a forbidden one is found
        
        # Check suspicious patterns with warning level
        for pattern in self.config["suspicious_patterns"]:
            if re.search(pattern, text_lower):
                results.append(ValidationResult(
                    is_valid=True,  # Still valid but with warning
                    severity=ValidationSeverity.WARNING,
                    message=f"Prompt contains potentially suspicious pattern: {pattern}",
                    code="SUSPICIOUS"
                ))
        
        # Check for forbidden words
        for word in self.config["forbidden_words"]:
            if word.lower() in text_lower:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Prompt contains forbidden word: {word}",
                    code="FORBIDDEN_WORD"
                ))
        
        # Check for control characters that might be used maliciously
        for i, char in enumerate(text):
            if ord(char) < 32 and char not in "\t\n\r":
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Prompt contains control character at position {i}",
                    code="CONTROL_CHAR"
                ))
        
        return results
    
    def _check_content_appropriateness(self, text: str) -> List[ValidationResult]:
        """Check if content is appropriate for image editing context."""
        results = []
        
        # Check for excessive repetition
        words = text.lower().split()
        if len(words) > 1:
            # Check for consecutive repeating words
            consecutive_count = 1
            max_consecutive = 1
            for i in range(1, len(words)):
                if words[i] == words[i-1]:
                    consecutive_count += 1
                    max_consecutive = max(max_consecutive, consecutive_count)
                else:
                    consecutive_count = 1
            
            if max_consecutive > self.config["max_consecutive_words"]:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Prompt contains too many consecutive repeating words: {max_consecutive}",
                    code="REPEATING_WORDS"
                ))
        
        # Check for character-level repetition
        char_count = 1
        max_char_count = 1
        for i in range(1, len(text)):
            if text[i] == text[i-1]:
                char_count += 1
                max_char_count = max(max_char_count, char_count)
            else:
                char_count = 1
        
        if max_char_count > self.config["max_consecutive_chars"]:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Prompt contains too many consecutive repeating characters: {max_char_count}",
                code="REPEATING_CHARS"
            ))
        
        return results
    
    def _check_quality_issues(self, text: str) -> List[ValidationResult]:
        """Check for quality issues in the prompt."""
        results = []
        
        # Check if text is mostly non-alphanumeric (indicates low quality)
        alphanumeric_chars = sum(1 for c in text if c.isalnum())
        if len(text) > 0 and (alphanumeric_chars / len(text)) < 0.2:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Prompt contains too many special characters relative to alphanumeric content",
                code="LOW_QUALITY"
            ))
        
        # Check if the prompt has meaningful variety in words
        words = text.lower().split()
        if len(words) > 10:  # Only check for longer prompts
            unique_words = set(words)
            variety_ratio = len(unique_words) / len(words)
            if variety_ratio < 0.3:  # Less than 30% unique words
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"Prompt has low lexical variety: only {variety_ratio:.2%} unique words",
                    code="LOW_VARIETY"
                ))
        
        return results

def validate_prompt_advanced(text: str) -> Dict[str, Any]:
    """
    Advanced validation that provides detailed information about the validation process.
    
    Args:
        text: String containing the prompt to validate
    
    Returns:
        A dictionary containing validation details
    """
    validator = AdvancedPromptValidator()
    is_valid, results = validator.validate(text)
    
    # Categorize results by severity
    by_severity = {
        "errors": [r for r in results if r.severity == ValidationSeverity.ERROR],
        "warnings": [r for r in results if r.severity == ValidationSeverity.WARNING],
        "info": [r for r in results if r.severity == ValidationSeverity.INFO]
    }
    
    return {
        "is_valid": is_valid,
        "results": results,
        "by_severity": by_severity,
        "message_count": {
            "errors": len(by_severity["errors"]),
            "warnings": len(by_severity["warnings"]),
            "info": len(by_severity["info"])
        }
    }

# Example usage
if __name__ == "__main__":
    validator = AdvancedPromptValidator()
    
    test_cases = [
        "A beautiful landscape with mountains and a blue sky",  # Valid
        "",  # Empty - should fail
        "xxx",  # Too short - should fail
        "Make me a beautiful sunset with mountains and a lake that looks amazing and wonderful and fantastic",  # Valid
        "Beautiful sky <script>alert('xss')</script>",  # Security issue - should fail
        "eval(this is bad)",  # Contains forbidden pattern - should fail
        "This is a valid prompt but with a lot of repeating words repeating words repeating words",  # Repeating - should fail
        "A@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",  # Too many special chars - should fail
    ]
    
    print("Advanced Prompt Validation Results:")
    for i, test_prompt in enumerate(test_cases):
        print(f"\n{i+1}. Testing: '{test_prompt[:50]}{'...' if len(test_prompt) > 50 else ''}'")
        result = validate_prompt_advanced(test_prompt)
        
        print(f"   Valid: {result['is_valid']}")
        print(f"   Errors: {result['message_count']['errors']}")
        print(f"   Warnings: {result['message_count']['warnings']}")
        
        if result['message_count']['errors'] > 0:
            print("   Error details:")
            for error in result['by_severity']['errors']:
                print(f"     - {error.message} (Code: {error.code})")
        
        if result['message_count']['warnings'] > 0:
            print("   Warning details:")
            for warning in result['by_severity']['warnings']:
                print(f"     - {warning.message} (Code: {warning.code})")
```