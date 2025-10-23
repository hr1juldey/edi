# IntentParser.forward()

[Back to Intent Parser](../reasoning_intent_parser.md)

## Related User Story
"As a user, I want EDI to ask questions when my request is ambiguous rather than guessing." (from PRD)

## Function Signature
`forward(naive_prompt, scene) -> Intent`

## Parameters
- `naive_prompt` - The user's casual edit request (e.g., "make the sky more dramatic")
- `scene` - The scene analysis containing detected entities and layout

## Returns
- `Intent` - A structured intent object with target entities, edit type, confidence, and clarifying questions if needed

## Step-by-step Logic
1. Parse the naive prompt for ambiguity markers ("dramatic", "better", "more")
2. Apply DSpy ChainOfThought reasoning to extract structured intent
3. Identify target entities to edit based on the prompt and scene analysis
4. Determine edit type (one of: color, style, add, remove, transform)
5. Calculate confidence score (0-1) indicating clarity of intent
6. If confidence < 0.7, generate clarifying questions using DSpy
7. Return structured intent with target entities, edit type, confidence, and questions

## DSpy Components Used
- dspy.Signature for structured input/output
- dspy.ChainOfThought for multi-step reasoning
- Confidence scoring for ambiguity detection

## Ambiguity Handling
- Detects vague terms that could have multiple interpretations
- Generates specific, answerable questions (1-5 options)
- Maintains context for refined understanding

## Input/Output Data Structures
### Intent Object
An Intent object contains:
- Target entities (comma-separated list of entity IDs to edit)
- Edit type (color, style, add, remove, transform)
- Confidence score (0-1)
- Clarifying questions (JSON array if confidence < 0.7)

## See Docs

```python
import dspy
import re
from typing import List, Dict, Any, Optional

class Intent:
    """
    An Intent object contains:
    - Target entities (comma-separated list of entity IDs to edit)
    - Edit type (color, style, add, remove, transform)
    - Confidence score (0-1)
    - Clarifying questions (JSON array if confidence < 0.7)
    """
    def __init__(self, 
                 target_entities: List[str] = None, 
                 edit_type: str = None,
                 confidence: float = 0.0, 
                 clarifying_questions: List[str] = None):
        self.target_entities = target_entities or []
        self.edit_type = edit_type
        self.confidence = confidence
        self.clarifying_questions = clarifying_questions or []

    def to_json(self) -> Dict[str, Any]:
        """Convert the intent to a JSON-serializable dictionary."""
        return {
            "target_entities": self.target_entities,
            "edit_type": self.edit_type,
            "confidence": self.confidence,
            "clarifying_questions": self.clarifying_questions
        }

class IntentParser(dspy.Module):
    def __init__(self, ambiguity_threshold: float = 0.7):
        super().__init__()
        self.ambiguity_threshold = ambiguity_threshold
        
        # Define DSPy signatures for different parts of the process
        class ExtractStructuredIntent(dspy.Signature):
            """Extract structured intent from a naive prompt and scene analysis."""
            naive_prompt = dspy.InputField(desc="User's casual edit request (e.g., 'make the sky more dramatic')")
            scene = dspy.InputField(desc="Scene analysis with detected entities and layout")
            target_entities = dspy.OutputField(desc="Comma-separated list of entity IDs to edit")
            edit_type = dspy.OutputField(desc="One of: color, style, add, remove, transform")
            confidence = dspy.OutputField(desc="Confidence score between 0-1 indicating clarity of intent")
        
        class GenerateClarifyingQuestions(dspy.Signature):
            """Generate clarifying questions when intent confidence is low."""
            naive_prompt = dspy.InputField(desc="Original naive prompt from user")
            scene = dspy.InputField(desc="Scene analysis for context")
            target_entities = dspy.InputField(desc="Target entities identified so far")
            edit_type = dspy.InputField(desc="Edit type identified so far")
            clarifying_questions = dspy.OutputField(desc="JSON array of 1-5 specific, answerable questions")
        
        # Create DSPy ChainOfThought modules
        self.extract_intent = dspy.ChainOfThought(ExtractStructuredIntent)
        self.generate_questions = dspy.ChainOfThought(GenerateClarifyingQuestions)
    
    def forward(self, naive_prompt: str, scene: Dict[str, Any]) -> Intent:
        """
        Parse the user's naive prompt into a structured intent using DSPy.
        
        This method:
        1. Parses the naive prompt for ambiguity markers ("dramatic", "better", "more")
        2. Applies DSpy ChainOfThought reasoning to extract structured intent
        3. Identifies target entities to edit based on the prompt and scene analysis
        4. Determines edit type (one of: color, style, add, remove, transform)
        5. Calculates confidence score (0-1) indicating clarity of intent
        6. If confidence < 0.7, generates clarifying questions using DSpy
        7. Returns structured intent with target entities, edit type, confidence, and questions
        """
        # Run the extract intent module
        result = self.extract_intent(naive_prompt=naive_prompt, scene=str(scene))
        
        # Parse the target entities from the comma-separated string
        target_entities_list = [entity.strip() for entity in result.target_entities.split(",") if entity.strip()]
        
        # Convert confidence to float
        confidence = float(result.confidence) if result.confidence.replace('.', '').isdigit() else 0.5
        
        # Ensure confidence is between 0 and 1
        confidence = max(0.0, min(1.0, confidence))
        
        # Check if we need to generate clarifying questions
        clarifying_questions = []
        if confidence < self.ambiguity_threshold:
            # Generate clarifying questions when confidence is low
            questions_result = self.generate_questions(
                naive_prompt=naive_prompt,
                scene=str(scene),
                target_entities=result.target_entities,
                edit_type=result.edit_type
            )
            
            try:
                # Try to parse the questions from the output
                questions_str = questions_result.clarifying_questions.strip()
                
                # If it looks like a JSON array, parse it
                if questions_str.startswith("[") and questions_str.endswith("]"):
                    import json
                    clarifying_questions = json.loads(questions_str)
                else:
                    # Split by newlines or commas, if needed
                    clarifying_questions = [q.strip() for q in questions_str.split('\\n') if q.strip()]
            except:
                # If parsing fails, just use the raw string as a single question
                clarifying_questions = [questions_str] if questions_str.strip() else []
        
        return Intent(
            target_entities=target_entities_list,
            edit_type=result.edit_type,
            confidence=confidence,
            clarifying_questions=clarifying_questions
        )
    
    def _parse_ambiguity_markers(self, naive_prompt: str) -> List[str]:
        """
        Parse the naive prompt for common ambiguity markers.
        """
        ambiguity_patterns = [
            r'\bmore\b',
            r'\bbetter\b',
            r'\bdramatic\b',
            r'\bimprove\b',
            r'\bfix\b',
            r'\bchange\b',
            r'\badjust\b'
        ]
        
        found_markers = []
        for pattern in ambiguity_patterns:
            if re.search(pattern, naive_prompt, re.IGNORECASE):
                found_markers.append(re.search(pattern, naive_prompt, re.IGNORECASE).group())
        
        return found_markers

# Example usage:
if __name__ == "__main__":
    # Create a sample scene analysis
    scene = {
        "entities": ["sky", "trees", "mountains", "grass", "water"],
        "spatial_layout": "sky in background, mountains mid-ground, trees and grass in foreground, water in foreground",
        "colors": ["blue sky", "green trees", "brown mountains", "green grass", "blue water"],
        "lighting": "natural daylight"
    }
    
    # Create the intent parser
    parser = IntentParser()
    
    # Test with an ambiguous prompt
    ambiguous_prompt = "make the sky more dramatic"
    intent = parser.forward(ambiguous_prompt, scene)
    
    print("Intent Analysis:")
    print(f"  Prompt: {ambiguous_prompt}")
    print(f"  Target Entities: {intent.target_entities}")
    print(f"  Edit Type: {intent.edit_type}")
    print(f"  Confidence: {intent.confidence}")
    print(f"  Clarifying Questions: {intent.clarifying_questions}")
    print()
    
    # Test with a clearer prompt
    clear_prompt = "increase the blue saturation in the sky"
    intent = parser.forward(clear_prompt, scene)
    
    print("Clear Intent Analysis:")
    print(f"  Prompt: {clear_prompt}")
    print(f"  Target Entities: {intent.target_entities}")
    print(f"  Edit Type: {intent.edit_type}")
    print(f"  Confidence: {intent.confidence}")
    print(f"  Clarifying Questions: {intent.clarifying_questions}")