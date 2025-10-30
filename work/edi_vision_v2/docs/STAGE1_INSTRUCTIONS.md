# Stage 1: DSpy Entity Extraction - Implementation Instructions

**File to create**: `pipeline/stage1_entity_extraction.py`

**Reference**: See `/home/riju279/Documents/Code/Zonko/EDI/edi/docs/VISION_PIPELINE_RESEARCH.md` Section "Stage 1: DSpy Entity Extraction"

---

## Requirements

### 1. Define Pydantic Models

```python
from pydantic import BaseModel
from enum import Enum
from typing import Optional, List

class EditType(str, Enum):
    RECOLOR = "recolor"
    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"
    STYLE_TRANSFER = "style_transfer"

class EntityDescription(BaseModel):
    label: str  # e.g., "tin roof"
    color: Optional[str] = None  # e.g., "blue"
    texture: Optional[str] = None  # e.g., "tin", "metallic"
    size_descriptor: Optional[str] = None  # e.g., "large", "small", "all"
```

### 2. Create DSpy Signature

```python
import dspy

class ExtractEditIntent(dspy.Signature):
    """Extract structured editing intent from natural language prompt."""

    user_prompt: str = dspy.InputField(
        desc="User's edit request in natural language"
    )

    # Outputs
    target_entities: List[EntityDescription] = dspy.OutputField(
        desc="List of entities to be edited with their attributes"
    )
    edit_type: EditType = dspy.OutputField(
        desc="Type of edit operation"
    )
    new_value: str = dspy.OutputField(
        desc="Target value for the edit (e.g., 'green' for recolor)"
    )
    quantity: str = dspy.OutputField(
        desc="One of: 'all', 'first', 'largest', 'specific'"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence in parsing (0.0-1.0)"
    )
```

### 3. Implement IntentParser Module

```python
class IntentParser(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(ExtractEditIntent)

    def forward(self, user_prompt: str):
        result = self.extractor(user_prompt=user_prompt)
        return result
```

### 4. Add Convenience Functions

Create helper functions:
- `parse_intent(prompt: str, llm_model: str = "ollama/qwen3:8b") -> dict`
  - Configures DSpy with specified LLM
  - Calls IntentParser
  - Returns dict representation

### 5. Testing

Create `tests/test_stage1.py` with these test cases:

**Test Case 1**:
- Input: `"turn the blue tin roofs of all those buildings to green"`
- Expected:
  ```python
  {
    "target_entities": [{"label": "tin roof", "color": "blue", ...}],
    "edit_type": "recolor",
    "new_value": "green",
    "quantity": "all",
    "confidence": > 0.8
  }
  ```

**Test Case 2**:
- Input: `"make the sky more dramatic"`
- Expected:
  ```python
  {
    "target_entities": [{"label": "sky", "color": None, ...}],
    "edit_type": "style_transfer",
    "new_value": "dramatic",
    "quantity": "all"
  }
  ```

**Test Case 3**:
- Input: `"remove the red car"`
- Expected:
  ```python
  {
    "target_entities": [{"label": "car", "color": "red", ...}],
    "edit_type": "remove",
    "quantity": "all"
  }
  ```

**Test Case 4**:
- Input: `"change the large green door to blue"`
- Expected:
  ```python
  {
    "target_entities": [{"label": "door", "color": "green", "size_descriptor": "large"}],
    "edit_type": "recolor",
    "new_value": "blue"
  }
  ```

**Test Case 5**:
- Input: `"add clouds to the sky"`
- Expected:
  ```python
  {
    "target_entities": [{"label": "sky", ...}],
    "edit_type": "add",
    "new_value": "clouds"
  }
  ```

---

## Implementation Checklist

- [ ] Import all required libraries (dspy, pydantic, enum, typing)
- [ ] Define `EditType` enum
- [ ] Define `EntityDescription` Pydantic model
- [ ] Create `ExtractEditIntent` DSpy signature
- [ ] Implement `IntentParser` class
- [ ] Add `parse_intent()` helper function
- [ ] Configure DSpy with Ollama (qwen3:8b model)
- [ ] Add logging statements
- [ ] Add type hints everywhere
- [ ] Add docstrings (Google style)
- [ ] Create `tests/test_stage1.py`
- [ ] Run all 5 test cases
- [ ] Verify outputs are deterministic (run each test 3 times)

---

## Code Quality Requirements

1. **Imports at top**: Standard library, then third-party, then local
2. **Type hints**: Every function parameter and return type
3. **Docstrings**: Every class and function
4. **Error handling**: Graceful fallbacks if DSpy fails
5. **Logging**: Use `logging.info()` for key steps
6. **No hardcoded values**: Make LLM model configurable

---

## Expected Output Format

After running:
```python
result = parse_intent("turn the blue tin roofs of all those buildings to green")
print(result)
```

Should print:
```python
{
  'target_entities': [
    {
      'label': 'tin roof',
      'color': 'blue',
      'texture': 'tin',
      'size_descriptor': 'all'
    }
  ],
  'edit_type': 'recolor',
  'new_value': 'green',
  'quantity': 'all',
  'confidence': 0.95
}
```

---

## Validation Steps

1. **Run pytest**: `pytest tests/test_stage1.py -v`
2. **Check determinism**: Run same prompt 3 times, verify identical output
3. **Check confidence**: All test cases should have confidence >0.7
4. **Manual test**: Try 3 new prompts not in test suite

---

## Report Format

After completion, report:

```
STAGE 1: DSpy Entity Extraction - [COMPLETE/FAILED]

Implementation:
- File: pipeline/stage1_entity_extraction.py
- Lines of code: XXX
- Functions: parse_intent(), IntentParser class

Test Results:
- Test Case 1: [PASS/FAIL]
- Test Case 2: [PASS/FAIL]
- Test Case 3: [PASS/FAIL]
- Test Case 4: [PASS/FAIL]
- Test Case 5: [PASS/FAIL]

Determinism Check:
- Same prompt run 3x: [IDENTICAL/VARIED]

Performance:
- Average execution time: XX ms

Issues: [None / List any problems]

Next: Awaiting approval to proceed to Stage 2
```

---

## Notes

- DSpy may require Ollama to be running: `ollama serve`
- Use model `qwen3:8b` for best results with local setup
- If DSpy fails, catch exception and return default low-confidence output
- Save example outputs to `logs/stage1_examples.json` for reference

---

**Begin implementation now!**
