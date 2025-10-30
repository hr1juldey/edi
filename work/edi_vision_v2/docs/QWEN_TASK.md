# Task for Qwen CLI - Stage 1

## Instructions

You are implementing **Stage 1: DSpy Entity Extraction** for the EDI Vision V2 pipeline.

**IMPORTANT**: Read these documents in order:
1. `CRITICAL_REQUIREMENT.md` - Critical: Masks must stay SEPARATE
2. `STAGE1_INSTRUCTIONS.md` - Detailed implementation guide for Stage 1

## Your Task

Implement Stage 1 exactly as specified in `STAGE1_INSTRUCTIONS.md`:

1. Create file: `pipeline/stage1_entity_extraction.py`
2. Define Pydantic models: `EditType`, `EntityDescription`
3. Create DSpy signature: `ExtractEditIntent`
4. Implement `IntentParser` module
5. Add helper function: `parse_intent()`
6. Create test file: `tests/test_stage1.py` with 5 test cases
7. Run tests and verify determinism

## Test Cases (from STAGE1_INSTRUCTIONS.md)

Test these 5 prompts:
1. "turn the blue tin roofs of all those buildings to green"
2. "make the sky more dramatic"
3. "remove the red car"
4. "change the large green door to blue"
5. "add clouds to the sky"

## Requirements

- Use Python 3.10+ type hints
- Add Google-style docstrings
- Configure DSpy with Ollama (qwen3:8b)
- Ensure deterministic output (same input → same output)
- All tests must pass

## When Complete

Report in this format:

```
STAGE 1: [COMPLETE/FAILED]

Implementation:
- File: pipeline/stage1_entity_extraction.py
- Lines of code: XXX

Test Results:
- Test Case 1: PASS ✅
- Test Case 2: PASS ✅
- Test Case 3: PASS ✅
- Test Case 4: PASS ✅
- Test Case 5: PASS ✅

Determinism Check:
- Same prompt run 3x: [IDENTICAL/VARIED]

Performance:
- Average execution time: XX ms

Issues: [None / List any problems]

Status: Awaiting supervisor approval
```

## Begin Implementation Now!

Start by creating `pipeline/stage1_entity_extraction.py`.
