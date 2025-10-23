# Reasoning Subsystem

[Back to Index](../index.md)

## Purpose

Intent understanding, prompt generation, validation using Ollama (qwen3:8b, gemma3:4b)

## Component Design

### 2. Reasoning Subsystem

**Purpose**: Translate user intent to technical specifications

#### 2.1 Intent Parser

**DSpy Module**:

```python
class ParseIntent(dspy.Signature):
    """
    Extract structured intent from casual user prompt.
    """
    naive_prompt = dspy.InputField(
        desc="User's conversational edit request"
    )
    scene_analysis = dspy.InputField(
        desc="JSON of detected entities and layout"
    )
    
    target_entities = dspy.OutputField(
        desc="Comma-separated list of entity IDs to edit"
    )
    edit_type = dspy.OutputField(
        desc="One of: color, style, add, remove, transform"
    )
    confidence = dspy.OutputField(
        desc="Float 0-1 indicating clarity of intent"
    )
    clarifying_questions = dspy.OutputField(
        desc="JSON array of questions if confidence <0.7"
    )
```

**Usage**:

```python
parser = dspy.ChainOfThought(ParseIntent)
result = parser(
    naive_prompt="make the sky more dramatic",
    scene_analysis=json.dumps(analysis)
)

if result.confidence < 0.7:
    # Show clarifying questions to user
    display_questions(result.clarifying_questions)
```

#### 2.2 Prompt Generator

**DSpy Pipeline** (3-stage refinement):

##### **Stage 1: Initial Generation**

```python
class GenerateBasePrompt(dspy.Signature):
    naive_prompt = dspy.InputField()
    scene_analysis = dspy.InputField()
    target_entities = dspy.InputField()
    edit_type = dspy.InputField()
    
    positive_prompt = dspy.OutputField(
        desc="Technical prompt for desired changes"
    )
    negative_prompt = dspy.OutputField(
        desc="Technical prompt for things to avoid"
    )
```

##### **Stage 2-4: Iterative Refinement**

```python
class RefinePrompt(dspy.Signature):
    naive_prompt = dspy.InputField()
    previous_positive = dspy.InputField()
    previous_negative = dspy.InputField()
    refinement_goal = dspy.InputField(
        desc="E.g., 'add technical quality terms', 'strengthen preservation constraints'"
    )
    
    refined_positive = dspy.OutputField()
    refined_negative = dspy.OutputField()
    improvement_explanation = dspy.OutputField()
```

**Refinement Strategy**:

```bash
Iteration 1: Add preservation constraints
Iteration 2: Increase technical specificity
Iteration 3: Add quality/style modifiers
```

**Prompt Template** (for positive prompt):

```bash
{edit_description}, {technical_terms}, {quality_modifiers},
preserve: {entities_to_keep}, maintain composition
```

Example output:

```bash
Positive: "dramatic storm clouds, cumulonimbus formation, 
          dark gray nimbus, volumetric lighting, overcast mood,
          photorealistic, 8k detail, preserve building structure,
          maintain foreground subjects"
          
Negative: "sunny sky, blue sky, bright lighting, lens flare,
          building color changes, grass modifications,
          cartoon style, oversaturated, artifacts"
```

## Sub-modules

This component includes the following modules:

- [reasoning/ollama_client.py](./ollama_client/ollama_client.md)
- [reasoning/intent_parser.py](./intent_parser/intent_parser.md)
- [reasoning/prompt_generator.py](./prompt_generator/prompt_generator.md)
- [reasoning/validator.py](./validator/validator.md)
- [reasoning/models.py](./models.md)

## Technology Stack

- Ollama for LLM inference
- DSpy for structured LLM interactions
- Pydantic for data validation
- Requests for API communication

## See Docs

### DSpy Implementation Example

Structured LLM interactions for the EDI reasoning subsystem:

```python
import dspy
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Define the signatures as per the documentation
class ParseIntent(dspy.Signature):
    """Extract structured intent from casual user prompt."""
    naive_prompt: str = dspy.InputField(desc="User's conversational edit request")
    scene_analysis: str = dspy.InputField(desc="JSON of detected entities and layout")
    
    target_entities: str = dspy.OutputField(desc="Comma-separated list of entity IDs to edit")
    edit_type: str = dspy.OutputField(desc="One of: color, style, add, remove, transform")
    confidence: str = dspy.OutputField(desc="Float 0-1 indicating clarity of intent")
    clarifying_questions: str = dspy.OutputField(desc="JSON array of questions if confidence <0.7")

class GenerateBasePrompt(dspy.Signature):
    """Generate initial positive and negative prompts based on intent."""
    naive_prompt: str = dspy.InputField()
    scene_analysis: str = dspy.InputField()
    target_entities: str = dspy.InputField()
    edit_type: str = dspy.InputField()
    
    positive_prompt: str = dspy.OutputField(desc="Technical prompt for desired changes")
    negative_prompt: str = dspy.OutputField(desc="Technical prompt for things to avoid")

class RefinePrompt(dspy.Signature):
    """Refine existing prompts based on a specific goal."""
    naive_prompt: str = dspy.InputField()
    previous_positive: str = dspy.InputField()
    previous_negative: str = dspy.InputField()
    refinement_goal: str = dspy.InputField(desc="E.g., 'add technical quality terms', 'strengthen preservation constraints'")
    
    refined_positive: str = dspy.OutputField()
    refined_negative: str = dspy.OutputField()
    improvement_explanation: str = dspy.OutputField()

# Pydantic models for structured data
class IntentResult(BaseModel):
    target_entities: List[str]
    edit_type: str
    confidence: float
    clarifying_questions: Optional[List[str]] = None

class PromptPair(BaseModel):
    positive_prompt: str
    negative_prompt: str

class RefinedPromptPair(PromptPair):
    improvement_explanation: str

# DSPy modules for the reasoning pipeline
class EDIIntentParser(dspy.Module):
    def __init__(self):
        super().__init__()
        self.parser = dspy.ChainOfThought(ParseIntent)
    
    def forward(self, naive_prompt: str, scene_analysis: Dict[str, Any]) -> IntentResult:
        result = self.parser(
            naive_prompt=naive_prompt,
            scene_analysis=json.dumps(scene_analysis)
        )
        
        confidence = float(result.confidence) if result.confidence.replace('.', '', 1).isdigit() else 0.0
        
        return IntentResult(
            target_entities=result.target_entities.split(', '),
            edit_type=result.edit_type,
            confidence=confidence,
            clarifying_questions=json.loads(result.clarifying_questions) if result.clarifying_questions else None
        )

class EDIPromptGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.Predict(GenerateBasePrompt)
        self.refiner = dspy.ChainOfThought(RefinePrompt)
    
    def forward(self, 
                naive_prompt: str, 
                scene_analysis: Dict[str, Any],
                target_entities: List[str],
                edit_type: str) -> PromptPair:
        
        # Initial generation
        result = self.generator(
            naive_prompt=naive_prompt,
            scene_analysis=json.dumps(scene_analysis),
            target_entities=', '.join(target_entities),
            edit_type=edit_type
        )
        
        # Refinement iterations
        positive = result.positive_prompt
        negative = result.negative_prompt
        
        # Refinement strategy
        refinement_goals = [
            'add preservation constraints',
            'increase technical specificity',
            'add quality/style modifiers'
        ]
        
        for goal in refinement_goals:
            refine_result = self.refiner(
                naive_prompt=naive_prompt,
                previous_positive=positive,
                previous_negative=negative,
                refinement_goal=goal
            )
            positive = refine_result.refined_positive
            negative = refine_result.refined_negative
        
        return PromptPair(
            positive_prompt=positive,
            negative_prompt=negative
        )

# Example usage
if __name__ == "__main__":
    # Configure DSPy with a local model (for example, using Ollama)
    # Using the correct DSPy LM API with Ollama
    ollama_lm = dspy.LM('ollama_chat/qwen3:8b', api_base='http://localhost:11434', api_key='')
    dspy.configure(lm=ollama_lm)
    
    # Alternative: Using multiple Ollama models with context manager
    main_ollama = dspy.LM('ollama_chat/qwen3:8b', api_base='http://localhost:11434', api_key='')
    alt_ollama = dspy.LM('ollama_chat/gemma3:4b', api_base='http://localhost:11434', api_key='')
    
    dspy.configure(lm=main_ollama)  # Set main model globally
    
    # Use alternative model only in a specific context
    with dspy.context(lm=alt_ollama):
        # This code block uses the alt_ollama model
        pass
    
    # Example scene analysis (as would be generated by the vision subsystem)
    scene_analysis = {
        "entities": [
            {"id": "sky_0", "label": "sky", "confidence": 0.95, "bbox": [0, 0, 100, 50]},
            {"id": "mountain_1", "label": "mountain", "confidence": 0.87, "bbox": [20, 50, 80, 100]},
            {"id": "tree_2", "label": "tree", "confidence": 0.92, "bbox": [10, 70, 15, 85]}
        ],
        "spatial_layout": "sky occupies top half, mountains at horizon, trees in foreground"
    }
    
    # Example usage of intent parser
    intent_parser = EDIIntentParser()
    intent_result = intent_parser.forward(
        naive_prompt="make the sky more dramatic",
        scene_analysis=scene_analysis
    )
    
    print("Parsed Intent:")
    print(f"Target entities: {intent_result.target_entities}")
    print(f"Edit type: {intent_result.edit_type}")
    print(f"Confidence: {intent_result.confidence}")
    
    if intent_result.clarifying_questions:
        print("Clarifying questions:", intent_result.clarifying_questions)
    
    # Example usage of prompt generator
    if intent_result.confidence >= 0.7:  # Only generate prompts if confidence is sufficient
        prompt_generator = EDIPromptGenerator()
        prompt_pair = prompt_generator.forward(
            naive_prompt="make the sky more dramatic",
            scene_analysis=scene_analysis,
            target_entities=intent_result.target_entities,
            edit_type=intent_result.edit_type
        )
        
        print("\nGenerated Prompts:")
        print(f"Positive: {prompt_pair.positive_prompt}")
        print(f"Negative: {prompt_pair.negative_prompt}")
```

### Requests Implementation Example

API communication for the EDI reasoning subsystem:

```python
import requests
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel
import time

class OllamaResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool
    total_duration: int
    load_duration: int
    prompt_eval_count: int
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def generate(self, 
                 model: str, 
                 prompt: str, 
                 system: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Generate text using an Ollama model."""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        if system:
            payload["system"] = system
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def chat(self,
             model: str,
             messages: list,
             temperature: float = 0.7) -> Dict[str, Any]:
        """Chat completion using an Ollama model."""
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def list_models(self) -> Dict[str, Any]:
        """List available Ollama models."""
        url = f"{self.base_url}/api/tags"
        
        response = self.session.get(url)
        response.raise_for_status()
        
        return response.json()

class EDIReasoningAPI:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama = OllamaClient(ollama_url)
        self.default_model = "qwen3:8b"
    
    def parse_intent(self, 
                     naive_prompt: str, 
                     scene_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Parse user intent using the reasoning model."""
        system_prompt = """
        You are an expert at understanding user edit requests for images. 
        Extract the following information from the user's request:
        - Target entities to edit (comma-separated list of entity IDs)
        - Edit type (one of: color, style, add, remove, transform)
        - Confidence (a float between 0 and 1 indicating how clear the intent is)
        - Clarifying questions (if confidence is less than 0.7)
        """
        
        prompt = f"""
        Scene Analysis: {json.dumps(scene_analysis)}
        
        User Request: {naive_prompt}
        
        Extract structured intent information as requested.
        Return the information in this JSON format:
        {{
            "target_entities": "...",
            "edit_type": "...",
            "confidence": "...",
            "clarifying_questions": [...]
        }}
        """
        
        response = self.ollama.generate(
            model=self.default_model,
            prompt=prompt,
            system=system_prompt
        )
        
        # Parse the response to extract structured data
        try:
            # Extract JSON from the response
            response_text = response['response']
            
            # Find JSON within the response text
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                
                # Process the result to match our expected format
                result["target_entities"] = result["target_entities"].split(", ")
                result["confidence"] = float(result["confidence"]) if result["confidence"].replace('.', '', 1).isdigit() else 0.0
                
                return result
            else:
                # If JSON parsing fails, return a default response
                return {
                    "target_entities": [],
                    "edit_type": "transform",
                    "confidence": 0.3,
                    "clarifying_questions": ["Could you clarify which specific parts of the image you want to edit?"]
                }
        except (json.JSONDecodeError, KeyError):
            return {
                "target_entities": [],
                "edit_type": "transform",
                "confidence": 0.3,
                "clarifying_questions": ["Could you provide more details about the edit you want?"]
            }
    
    def generate_prompt(self,
                       naive_prompt: str,
                       scene_analysis: Dict[str, Any],
                       target_entities: list,
                       edit_type: str) -> Dict[str, str]:
        """Generate positive and negative prompts for image editing."""
        system_prompt = """
        You are an expert at creating detailed prompts for image editing models.
        Given a user request and scene analysis, create both positive and negative prompts.
        The positive prompt should describe what should be in the edited image.
        The negative prompt should describe what should be avoided or preserved.
        """
        
        prompt = f"""
        Scene Analysis: {json.dumps(scene_analysis)}
        
        User Request: {naive_prompt}
        Target Entities: {', '.join(target_entities)}
        Edit Type: {edit_type}
        
        Generate a positive and negative prompt for an image editing model.
        The positive prompt should describe the desired changes.
        The negative prompt should describe preservation constraints.
        
        Return in this format:
        Positive: [positive prompt here]
        Negative: [negative prompt here]
        """
        
        response = self.ollama.generate(
            model=self.default_model,
            prompt=prompt,
            system=system_prompt
        )
        
        # Parse the response to extract positive and negative prompts
        try:
            response_text = response['response']
            
            # Extract positive and negative prompts
            positive_start = response_text.find('Positive:')
            negative_start = response_text.find('Negative:')
            
            if positive_start != -1 and negative_start != -1:
                positive_prompt = response_text[positive_start + 9:negative_start].strip()
                negative_prompt = response_text[negative_start + 9:].strip()
                
                return {
                    "positive_prompt": positive_prompt,
                    "negative_prompt": negative_prompt
                }
            else:
                # Fallback if format isn't as expected
                return {
                    "positive_prompt": f"Make the {', '.join(target_entities)} {edit_type} to match '{naive_prompt}'",
                    "negative_prompt": "Do not change anything else in the image"
                }
        except Exception:
            return {
                "positive_prompt": f"Make the {', '.join(target_entities)} {edit_type} to match '{naive_prompt}'",
                "negative_prompt": "Do not change anything else in the image"
            }

# Example usage
if __name__ == "__main__":
    # Example scene analysis
    scene_analysis = {
        "entities": [
            {"id": "sky_0", "label": "sky", "confidence": 0.95, "bbox": [0, 0, 100, 50]},
            {"id": "mountain_1", "label": "mountain", "confidence": 0.87, "bbox": [20, 50, 80, 100]},
            {"id": "tree_2", "label": "tree", "confidence": 0.92, "bbox": [10, 70, 15, 85]}
        ],
        "spatial_layout": "sky occupies top half, mountains at horizon, trees in foreground"
    }
    
    # Initialize the reasoning API
    reasoning_api = EDIReasoningAPI()
    
    # Example: Parse intent
    intent_result = reasoning_api.parse_intent(
        naive_prompt="make the sky more dramatic",
        scene_analysis=scene_analysis
    )
    
    print("Parsed Intent:")
    for key, value in intent_result.items():
        print(f"  {key}: {value}")
    
    # Example: Generate prompts (only if confidence is high enough)
    if intent_result["confidence"] >= 0.7:
        prompt_result = reasoning_api.generate_prompt(
            naive_prompt="make the sky more dramatic",
            scene_analysis=scene_analysis,
            target_entities=intent_result["target_entities"],
            edit_type=intent_result["edit_type"]
        )
        
        print("\nGenerated Prompts:")
        print(f"  Positive: {prompt_result['positive_prompt']}")
        print(f"  Negative: {prompt_result['negative_prompt']}")
    else:
        print(f"\nConfidence too low ({intent_result['confidence']}), need clarification:")
        for q in intent_result['clarifying_questions']:
            print(f"  - {q}")
```
