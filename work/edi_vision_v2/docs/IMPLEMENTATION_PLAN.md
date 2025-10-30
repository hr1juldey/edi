# EDI Vision V2 - Implementation Plan for Qwen CLI

**Supervisor**: Claude Code
**Worker**: Qwen CLI
**Architecture Reference**: `/home/riju279/Documents/Code/Zonko/EDI/edi/docs/VISION_PIPELINE_RESEARCH.md`

---

## Project Structure

```
work/edi_vision_v2/
├── pipeline/
│   ├── stage1_entity_extraction.py    # DSpy intent parser
│   ├── stage2_color_filter.py         # HSV pre-filtering
│   ├── stage3_sam_segmentation.py     # SAM with point prompts
│   ├── stage4_clip_filter.py          # Semantic filtering
│   ├── stage5_organization.py         # Mask organization (KEEP SEPARATE!)
│   ├── stage6_validation.py           # VLM validation
│   └── orchestrator.py                # Full pipeline coordinator
├── tests/
│   ├── test_stage1.py
│   ├── test_stage2.py
│   ├── test_stage3.py
│   ├── test_stage4.py
│   ├── test_stage5.py
│   ├── test_stage6.py
│   └── test_integration.py
├── logs/                              # Debug outputs, intermediate images
├── app.py                             # Main entry point
├── IMPLEMENTATION_PLAN.md             # This file
└── README.md                          # Usage instructions
```

---

## Implementation Checklist

### Stage 1: DSpy Entity Extraction ⏳
- [ ] Create `pipeline/stage1_entity_extraction.py`
- [ ] Define Pydantic models: `EditType`, `EntityDescription`
- [ ] Create DSpy signature: `ExtractEditIntent`
- [ ] Implement `IntentParser` module
- [ ] Test with 5+ different prompts
- [ ] Validate outputs are deterministic

**Acceptance Criteria**:
- Input: "turn the blue tin roofs of all those buildings to green"
- Output: Structured dict with entities=[{label:"tin roof", color:"blue"}], edit_type="recolor", new_value="green", quantity="all"

---

### Stage 2: Color Pre-Filter ⏳
- [ ] Create `pipeline/stage2_color_filter.py`
- [ ] Implement `color_prefilter()` function with HSV ranges
- [ ] Support colors: blue, green, red, yellow, orange
- [ ] Add morphological cleanup (MORPH_CLOSE, MORPH_OPEN)
- [ ] Test on test_image.jpeg (should detect ALL blue roofs)
- [ ] Validate with see_image tool

**Acceptance Criteria**:
- Input: test_image.jpeg + color="blue"
- Output: Binary mask covering ALL ~20 blue roof regions
- Performance: <100ms execution time

---

### Stage 3: SAM Segmentation ⏳
- [ ] Create `pipeline/stage3_sam_segmentation.py`
- [ ] Implement `sam_segment_colored_regions()` function
- [ ] Use cv2.connectedComponentsWithStats for region detection
- [ ] Use SAM with point prompts (centroids)
- [ ] Process ALL regions, not top-k
- [ ] Return list of individual masks

**Acceptance Criteria**:
- Input: image + color_mask from Stage 2
- Output: List of ~20 precise masks (one per roof)
- Each mask should be pixel-perfect around roof boundaries

---

### Stage 4: CLIP Filtering ⏳
- [ ] Create `pipeline/stage4_clip_filter.py`
- [ ] Implement `clip_filter_masks()` function
- [ ] Use threshold-based filtering (NOT top-k)
- [ ] Default threshold: 0.15
- [ ] Return ALL masks above threshold
- [ ] Sort by similarity score

**Acceptance Criteria**:
- Input: List of 25 masks (20 roofs + 5 blue flags/objects)
- Query: "tin roof"
- Output: 20 masks (filtered out non-roof objects)

---

### Stage 5: Mask Organization & Labeling ⏳
- [ ] Create `pipeline/stage5_organization.py`
- [ ] Implement `EntityMask` class with metadata
- [ ] Implement `organize_masks()` function
- [ ] Keep masks SEPARATE - DO NOT merge!
- [ ] Add entity_id, bbox, centroid, area to each mask
- [ ] Optional: `get_combined_mask_for_visualization()` for validation only

**Acceptance Criteria**:
- Input: List of 20 individual roof masks
- Output: List of 20 EntityMask objects (SEPARATE, NOT MERGED!)
- Each EntityMask has: mask, entity_id, bbox, centroid, similarity_score
- Masks can be edited individually later

---

### Stage 6: VLM Validation ⏳
- [ ] Create `pipeline/stage6_validation.py`
- [ ] Implement `validate_with_vlm()` function
- [ ] Use Ollama API directly (not MCP - for Qwen's use)
- [ ] Create red overlay visualization
- [ ] Parse JSON response from VLM
- [ ] Return validation metrics

**Acceptance Criteria**:
- Input: image + combined_mask + user_request
- Output: {covers_all_targets: true, confidence: 0.9, feedback: "..."}
- Uses qwen2.5vl:7b model locally

---

### Stage 7: Orchestrator ⏳
- [ ] Create `pipeline/orchestrator.py`
- [ ] Implement `VisionPipeline` class
- [ ] Chain all 6 stages together
- [ ] Add error handling at each stage
- [ ] Add debug logging with timing
- [ ] Save intermediate outputs to logs/

**Acceptance Criteria**:
- Input: image_path + prompt
- Output: Final mask + validation report
- All stages execute in sequence
- Total time: <20 seconds

---

### Stage 8: Testing ⏳
- [ ] Create unit tests for each stage
- [ ] Create integration test with test_image.jpeg
- [ ] Test edge cases (no color, ambiguous prompts)
- [ ] Validate against expected outputs
- [ ] Performance benchmarks

---

### Stage 9: Main Application Interfaces ⏳

**Dual Interface Implementation**: CLI for automation, TUI for human users

#### Part A: CLI Interface (app.py)
- [ ] Create `app.py` with argparse
- [ ] Add CLI arguments (image, prompt, output, verbose, debug, save-steps, no-validation, config, thresholds)
- [ ] Implement input validation (image exists, format valid, prompt non-empty)
- [ ] Implement logging setup (verbose/debug modes)
- [ ] Implement config file loading (YAML)
- [ ] Implement 2x2 grid visualization output
- [ ] Implement console summary output
- [ ] Implement error handling with user-friendly messages
- [ ] Create default config.yaml template
- [ ] Test all CLI functionality

#### Part B: TUI Interface (tui.py)
- [ ] Create `tui.py` with Textual App structure
- [ ] Implement WelcomeScreen (introduction + start button)
- [ ] Implement ImageSelectionScreen (file browser + preview)
- [ ] Implement PromptInputScreen (text input + settings)
- [ ] Implement ProcessingScreen (live progress bars + logs)
- [ ] Implement ResultsScreen (summary + entity table + actions)
- [ ] Create `tui.tcss` stylesheet
- [ ] Implement keyboard shortcuts (Q, H, ESC, arrows)
- [ ] Implement ANSI art image preview
- [ ] Add error handling with modal dialogs
- [ ] Test full navigation flow
- [ ] Test with real pipeline execution

**Acceptance Criteria**:
- CLI: Non-interactive batch processing with proper exit codes
- TUI: Interactive guided workflow with real-time progress
- Both: Process images using the same orchestrator.py backend
- Both: Produce equivalent results for same inputs
- Documentation: Clear usage examples for both interfaces

---

## Development Guidelines for Qwen

### Code Quality Standards
1. **Type hints**: Use Python 3.10+ type hints everywhere
2. **Docstrings**: Google-style docstrings for all functions
3. **Error handling**: Try-except with specific exceptions
4. **Logging**: Use logging module, not print statements
5. **No hardcoded paths**: Use Path objects, make paths configurable

### Dependencies (Already in requirements)
- opencv-python (cv2)
- numpy
- torch
- open-clip-torch
- ultralytics (SAM)
- dspy
- pydantic
- requests (for Ollama API)
- Pillow

### Testing Protocol
After implementing each stage:
1. Run unit test
2. Test with test_image.jpeg
3. Use see_image tool to validate visually
4. Report results to supervisor (Claude)
5. Wait for approval before proceeding

### Debugging
- Save all intermediate images to `logs/` with timestamps
- Log execution time for each stage
- Log mask statistics (area, bbox, count)
- On failure, save debug visualization

---

## Execution Plan

**Implementation Order**:
1. Stage 1 (Entity Extraction) - Foundation
2. Stage 2 (Color Filter) - Quick win
3. Stage 3 (SAM) - Core segmentation
4. Stage 4 (CLIP) - Semantic filtering
5. Stage 5 (Aggregation) - Simple combination
6. Stage 6 (Validation) - Quality check
7. Stage 7 (Orchestrator) - Integration
8. Stage 8 (Testing) - Validation
9. Stage 9A (CLI App) - Non-interactive interface
10. Stage 9B (TUI App) - Interactive interface

**Estimated Time**: 3-4 hours total
- Stages 1-8: ~2.5 hours (15-20 min per stage)
- Stage 9A (CLI): ~20 min
- Stage 9B (TUI): ~30-40 min

---

## Success Criteria

### Final Test Case
- **Input Image**: `work/edi_vision_tui/test_image.jpeg`
- **Prompt**: "turn the blue tin roofs of all those buildings to green"
- **Expected Output**:
  - Mask covering ALL ~20 blue roofs
  - No false positives (flags, sky, etc.)
  - Pixel-perfect boundaries
  - VLM validation confidence >0.85
  - Execution time <20 seconds

### Comparison to Old System
| Metric | Old (v1) | New (v2) |
|--------|----------|----------|
| Roofs detected | 1/20 ❌ | 20/20 ✅ |
| Uses YOLO | Yes | No ✅ |
| Multi-entity | No ❌ | Yes ✅ |
| Deterministic parsing | No ❌ | Yes (DSpy) ✅ |

---

## Communication Protocol

**Qwen reports to Claude after each stage**:
```
Stage X: [COMPLETE/FAILED]
- Implementation: [file path]
- Test result: [PASS/FAIL]
- Metrics: [execution time, accuracy, etc.]
- Issues: [any problems encountered]
- Next: [awaiting approval to proceed]
```

**Claude responds**:
- ✅ APPROVED - Proceed to next stage
- ⚠️ REVISE - Fix issues then resubmit
- ❌ BLOCKED - Critical issue, stop and debug

---

## Notes

- This is a REWRITE, not a refactor of v1
- v1 code is reference only, do not copy blindly
- Follow the architecture in VISION_PIPELINE_RESEARCH.md EXACTLY
- When in doubt, ask supervisor (Claude)
- Use see_image tool liberally for visual validation

---

**Ready to begin? Start with Stage 1!** 🚀
