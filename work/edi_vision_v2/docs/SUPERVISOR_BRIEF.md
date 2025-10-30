# Supervisor Brief: Claude Manages Qwen

**Date**: 2025-10-28
**Project**: EDI Vision V2 - Multi-Entity Detection Fix
**Supervisor**: Claude Code
**Worker**: Qwen CLI
**Status**: ✅ Ready to deploy Qwen

---

## Mission

Fix the critical bug where vision pipeline only masks **1 out of 20** entities.

**Solution**: Complete rewrite with 6-stage pipeline (Color→SAM→CLIP→VLM) that eliminates YOLO dependency and processes ALL matching entities.

---

## Workspace Setup ✅

```
work/edi_vision_v2/
├── pipeline/                   # Empty - Qwen will populate
├── tests/                      # Empty - Qwen will populate
├── logs/                       # Empty - Debug outputs go here
├── test_image.jpeg             # ✅ Copied (20 blue roofs)
├── IMPLEMENTATION_PLAN.md      # ✅ Master checklist (9 stages)
├── STAGE1_INSTRUCTIONS.md      # ✅ Detailed task for Qwen
├── README.md                   # ✅ Project overview
└── SUPERVISOR_BRIEF.md         # This file
```

**Architecture Reference**: `docs/VISION_PIPELINE_RESEARCH.md` (comprehensive research & design)

---

## Prerequisites Check ✅

- [x] Ollama running at localhost:11434
- [x] Models available:
  - qwen3:8b (for DSpy reasoning)
  - qwen2.5vl:7b (for vision validation)
  - gemma3:4b (fallback)
- [x] Python environment with all deps from req.txt
- [x] Test image copied to workspace
- [x] Implementation plan written
- [x] Stage 1 instructions detailed

---

## How to Deploy Qwen

### Option 1: Direct Command
```bash
cd work/edi_vision_v2

# Tell Qwen to start
qwen "Read STAGE1_INSTRUCTIONS.md and implement Stage 1: DSpy Entity Extraction. Follow the instructions EXACTLY. When done, run the tests and report results."
```

### Option 2: Interactive Session
```bash
qwen

# Then in Qwen CLI:
> cd /home/riju279/Documents/Code/Zonko/EDI/edi/work/edi_vision_v2
> Read STAGE1_INSTRUCTIONS.md
> Implement the code as specified
> Run tests when done
> Report results
```

---

## Supervision Protocol

### After Each Stage, Claude Will:

1. **Review Code Quality**
   ```bash
   # Check what Qwen created
   ls -la pipeline/
   cat pipeline/stage1_entity_extraction.py
   ```

2. **Run Tests**
   ```bash
   cd work/edi_vision_v2
   pytest tests/test_stage1.py -v
   ```

3. **Visual Validation** (for stages with image output)
   ```python
   # Use Claude's vision via MCP tool
   await see_image("logs/stage2_output.png", "Does this mask cover ALL blue roofs?")
   ```

4. **Performance Check**
   ```bash
   # Check execution time
   python -m cProfile pipeline/stage1_entity_extraction.py
   ```

5. **Approve or Revise**
   - ✅ If all checks pass: Approve, tell Qwen to proceed to next stage
   - ⚠️ If minor issues: Request revision with specific guidance
   - ❌ If critical failure: Stop, debug, provide corrective instructions

---

## Quality Gates (Each Stage Must Pass)

- [ ] **Functionality**: Does it work as specified?
- [ ] **Tests**: All unit tests passing?
- [ ] **Code Quality**: Type hints, docstrings, error handling?
- [ ] **Performance**: Meets time/memory targets?
- [ ] **Visual Validation**: Produces correct output? (vision check)
- [ ] **Integration**: Compatible with other stages?

---

## Vision Validation Strategy

### Qwen's Tools (During Development)
- **Direct Ollama API calls** to qwen2.5vl:7b
- For: Quick checks of intermediate masks
- Example:
  ```python
  import requests, base64
  # ... Qwen can call Ollama directly
  ```

### Claude's Tools (Supervisor Checks)
- **see_image MCP tool** via local vision server
- For: Final validation before approving stage
- Example:
  ```python
  # Claude checks Qwen's output
  result = await see_image(
      "work/edi_vision_v2/logs/stage2_color_mask.png",
      "Does this binary mask cover ALL blue regions in the image? Count them."
  )
  ```

**Why Both?**
- Qwen can't switch between qwen-coder and qwen-vision automatically
- Claude has native vision via MCP tool
- Double validation ensures quality

---

## Stage-by-Stage Checklist

### Stage 1: DSpy Entity Extraction ⏳
- [ ] Qwen implements code
- [ ] Qwen runs tests (5 test cases)
- [ ] Claude reviews code quality
- [ ] Claude validates determinism (run same prompt 3x)
- [ ] Claude approves → proceed to Stage 2

### Stage 2: Color Pre-Filter ⏸️
- [ ] Qwen implements HSV filtering
- [ ] Qwen tests on test_image.jpeg
- [ ] Claude uses see_image to verify ALL blue roofs detected
- [ ] Claude checks performance (<100ms)
- [ ] Claude approves → proceed to Stage 3

### Stage 3: SAM Segmentation ⏸️
- [ ] Qwen implements SAM with point prompts
- [ ] Qwen processes ALL colored regions
- [ ] Claude verifies 20 individual masks created
- [ ] Claude checks mask quality (pixel-perfect boundaries)
- [ ] Claude approves → proceed to Stage 4

### Stage 4: CLIP Filtering ⏸️
- [ ] Qwen implements threshold-based filtering
- [ ] Qwen removes top-k limit
- [ ] Claude verifies semantic filtering works
- [ ] Claude tests: "tin roof" filters out "blue flag"
- [ ] Claude approves → proceed to Stage 5

### Stage 5: Mask Aggregation ⏸️
- [ ] Qwen implements mask combination
- [ ] Claude verifies union of all masks
- [ ] Claude checks coverage matches sum of inputs
- [ ] Claude approves → proceed to Stage 6

### Stage 6: VLM Validation ⏸️
- [ ] Qwen implements VLM validation
- [ ] Claude verifies JSON structured output
- [ ] Claude tests validation accuracy
- [ ] Claude approves → proceed to Stage 7

### Stage 7: Orchestrator ⏸️
- [ ] Qwen chains all stages together
- [ ] Claude runs end-to-end test
- [ ] Claude validates ALL 20 roofs detected
- [ ] Claude checks total time <20s
- [ ] Claude approves → proceed to Stage 8

### Stage 8: Testing ⏸️
- [ ] Qwen creates comprehensive test suite
- [ ] Claude runs all tests
- [ ] Claude validates edge cases
- [ ] Claude approves → proceed to Stage 9

### Stage 9: Main App ⏸️
- [ ] Qwen creates CLI entry point
- [ ] Claude tests user workflow
- [ ] Claude validates final output
- [ ] Claude marks project COMPLETE ✅

---

## Emergency Procedures

### If Qwen Hallucinates
- Stop execution immediately
- Review logs to find where it deviated
- Provide corrective instruction referencing EXACT section of STAGE*_INSTRUCTIONS.md
- Restart from last good checkpoint

### If Performance Degrades
- Profile execution
- Identify bottleneck
- Guide Qwen to optimize specific function
- Re-benchmark

### If Tests Fail
- Examine test output
- Use see_image to diagnose visual failures
- Guide Qwen to fix root cause
- Verify fix with additional tests

### If Out of Memory
- Check GPU memory usage: `nvidia-smi`
- Guide Qwen to add model cleanup: `torch.cuda.empty_cache()`
- Reduce batch sizes
- Process regions sequentially if needed

---

## Success Metrics

**Project Complete When**:
- ✅ All 9 stages implemented
- ✅ All tests passing (>80% coverage)
- ✅ test_image.jpeg correctly processed (20/20 roofs)
- ✅ VLM validation confidence >0.85
- ✅ Total execution time <20 seconds
- ✅ Zero YOLO dependency
- ✅ Claude final vision check approves

---

## Communication Template

**Qwen Reports**:
```
STAGE X: [COMPLETE/FAILED]

Files Created:
- pipeline/stageX_*.py (XXX lines)
- tests/test_stageX.py (YYY lines)

Test Results:
- Test Case 1: PASS ✅
- Test Case 2: PASS ✅
- Test Case 3: FAIL ❌ (reason)

Performance:
- Execution time: XX ms
- Memory usage: YY MB

Issues:
- [List any problems or deviations]

Visual Validation:
- Saved to: logs/stageX_output.png
- Description: [what it shows]

Status: Awaiting supervisor approval
```

**Claude Response**:
```
REVIEW: Stage X

Code Quality: ✅/⚠️/❌
- Type hints: [comments]
- Docstrings: [comments]
- Error handling: [comments]

Tests: ✅/⚠️/❌
- All passing: [yes/no]
- Coverage: [percentage]

Performance: ✅/⚠️/❌
- Time: [within/exceeds target]
- Memory: [within/exceeds budget]

Visual Check: ✅/⚠️/❌
- [Claude's see_image validation]

DECISION: ✅ APPROVED | ⚠️ REVISE | ❌ BLOCKED
- [Specific feedback]
- [Next steps]
```

---

## Resource Monitoring

Claude will monitor:
- CPU/GPU usage (`nvidia-smi`, `htop`)
- Disk space in logs/ directory
- Ollama service health
- Python environment integrity

Alert if:
- GPU memory >90% utilized
- logs/ directory >1GB
- Ollama becomes unresponsive
- Dependencies missing/corrupted

---

## Expected Timeline

**Total Estimated Time**: 2-3 hours

| Stage | Time Estimate | Status |
|-------|---------------|--------|
| Stage 1 | 15-20 min | ⏳ Ready |
| Stage 2 | 10-15 min | ⏸️ Pending |
| Stage 3 | 20-25 min | ⏸️ Pending |
| Stage 4 | 15-20 min | ⏸️ Pending |
| Stage 5 | 5-10 min | ⏸️ Pending |
| Stage 6 | 15-20 min | ⏸️ Pending |
| Stage 7 | 20-25 min | ⏸️ Pending |
| Stage 8 | 15-20 min | ⏸️ Pending |
| Stage 9 | 10-15 min | ⏸️ Pending |

---

## Final Checklist Before Deployment

- [x] Workspace created
- [x] Instructions written
- [x] Test image available
- [x] Ollama running
- [x] Models downloaded
- [x] Dependencies installed
- [x] Supervisor (Claude) ready
- [ ] Qwen deployed ← **NEXT STEP**

---

**Status: ✅ READY TO DEPLOY QWEN**

**First Command to Qwen**:
```
Read /home/riju279/Documents/Code/Zonko/EDI/edi/work/edi_vision_v2/STAGE1_INSTRUCTIONS.md and implement Stage 1 exactly as specified. When complete, run the tests and report results in the format shown in SUPERVISOR_BRIEF.md.
```

---

**Claude is standing by to supervise. Let's begin!** 🚀
