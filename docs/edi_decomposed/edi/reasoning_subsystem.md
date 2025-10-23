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

```
Iteration 1: Add preservation constraints
Iteration 2: Increase technical specificity
Iteration 3: Add quality/style modifiers
```

**Prompt Template** (for positive prompt):

```
{edit_description}, {technical_terms}, {quality_modifiers},
preserve: {entities_to_keep}, maintain composition
```

Example output:

```
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