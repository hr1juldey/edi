# EDI Vision V3 - Dual-Path Multi-Entity Detection

**Status**: 🚧 **IN DEVELOPMENT**
**Supervisor**: Claude Code
**Implementer**: Qwen CLI
**Target Success Rate**: 85%+ (7.5/9 wildcard tests)

---

## 🎯 Project Overview

Complete redesign of the EDI vision pipeline to achieve **production-grade robustness** with dual-path detection architecture.

### Problem Statement
- **v2.0 Reality**: 11% true success rate (1/9 wildcard tests), 22% false positive rate
- **v3.0 Target**: 85%+ success rate, 0% false positive rate

### Root Cause Analysis
v2.0 failed due to THREE fundamental architectural flaws:
1. **Color-first assumption**: 60% of queries crashed (no semantic-only path)
2. **Toxic fallback behavior**: `np.ones()` mask created false positives
3. **SAM black box failures**: No diagnostics for 50% of image failures

### v3.0 Solution: Dual-Path Architecture
```
User Prompt → Intent Parser (routing strategy) →

├─→ COLOR-GUIDED PATH (fast, precise)
│   └─→ Dynamic Color Mapper → SAM on regions → CLIP high threshold
│
├─→ SEMANTIC-ONLY PATH (comprehensive)
│   └─→ SAM full image → CLIP low threshold → Post-filters (size/position)
│
└─→ HYBRID PATH (best of both)
    └─→ Run both paths → Merge results → Deduplicate by IoU
```

---

## 📁 Project Structure

```
work/edi_vision_v3/
├── docs/                                    # 📚 Comprehensive documentation
│   ├── CRITICAL_ARCHITECTURE_FLAWS.md       # Root cause analysis from v2
│   ├── V2_TO_V3_MIGRATION_ANALYSIS.md       # What to copy vs redesign
│   ├── QWEN_SUPERVISION_PLAN.md             # Implementation supervision plan
│   ├── EDIT_VALIDATION_RESEARCH.md          # Auto-validation system design
│   ├── NEXT_STEPS_VALIDATION.md             # Integration roadmap
│   └── CRITICAL_REQUIREMENT.md              # Separate masks requirement
│
├── pipeline/                                # 🔧 Core pipeline components
│   ├── stage1_intent_parser_v3.py           # Enhanced intent + routing strategy
│   ├── stage2_dynamic_color_mapper.py       # LLM-based HSV mapping (no static dict)
│   ├── stage3_sam_segmentation.py           # SAM with diagnostics + fallback
│   ├── stage4_clip_filter.py                # CLIP with adaptive thresholds
│   ├── stage4_5_post_filters.py             # Size/position/shape filtering
│   ├── stage5_mask_organization.py          # Entity mask organization
│   ├── stage6_vlm_validation.py             # VLM validation with structured feedback
│   ├── orchestrator_v3.py                   # Dual-path orchestrator
│   └── result_merger.py                     # IoU-based deduplication
│
├── validation/                              # ✅ Edit validation system
│   ├── vision_delta_analysis.py             # Before/after comparison
│   ├── quality_scoring.py                   # Alignment score calculation
│   └── feedback_generator.py                # Correction hints generation
│
├── tests/                                   # 🧪 Test suite
│   ├── unit/                                # Unit tests for each module
│   ├── integration/                         # Integration tests
│   ├── wildcard/                            # Wildcard robustness tests
│   └── ground_truth/                        # 50+ annotated images
│
├── logs/                                    # 📊 Pipeline outputs
│   ├── orchestrator/                        # Intermediate visualizations
│   ├── wildcard/                            # Wildcard test results
│   └── benchmarks/                          # Performance benchmarks
│
├── images/                                  # 🖼️  Test images
├── app.py                                   # 🖥️  CLI interface
├── tui.py                                   # 🎨 TUI interface
├── config.yaml                              # ⚙️  Configuration
├── sam2.1_b.pt                              # 🤖 SAM model weights
└── README.md                                # 📖 This file
```

---

## 🏗️ Architecture Improvements Over V2

### 1. Dynamic Color Handling (No Static Dictionary)

**v2.0 Problem**:
```python
# Static dictionary (8 colors only)
color_ranges = {"red": [...], "blue": [...], "green": [...]}

# Toxic fallback
if color not in color_ranges:
    return np.ones()  # ❌ 100% mask = garbage results
```

**v3.0 Solution**:
```python
# Dynamic LLM-based HSV mapping
class DynamicColorMapper:
    def get_hsv_ranges(self, color_description: str):
        # Pre-populated cache (50 common colors)
        if color in COMMON_COLOR_CACHE:
            return cached_ranges

        # Query LLM for uncommon colors
        result = llm_query(color_description)
        if result.is_valid_color:
            return result.hsv_ranges
        else:
            return None  # ✅ Clear failure signal
```

**Benefits**:
- ✅ Handles ANY color description ("burgundy", "sky blue", "ochre")
- ✅ No static dictionary maintenance
- ✅ Returns `None` for non-colors (no toxic fallback)
- ✅ Caches results for fast repeated use

---

### 2. Intent-Based Routing Strategy

**v2.0 Problem**:
```python
# Always runs color-first pipeline
def process(prompt):
    color = extract_color(prompt)
    color_mask = color_filter(image, color)  # Crashes if no color
    sam_masks = sam(image, color_mask)
```

**v3.0 Solution**:
```python
# Route based on query type
class EnhancedIntentParser:
    def forward(self, prompt):
        return IntentV3(
            entities=["auto-rickshaws"],
            color="yellow" or None,
            detection_strategy="color_guided" | "semantic_only" | "hybrid"
        )

# Orchestrator routes to appropriate path
if intent.detection_strategy == "color_guided":
    return color_guided_path(image, intent)
elif intent.detection_strategy == "semantic_only":
    return semantic_only_path(image, intent)
else:
    return hybrid_path(image, intent)
```

**Query Type Coverage**:
- **Color-guided** (40%): "red vehicles", "blue roofs"
- **Semantic-only** (30%): "auto-rickshaws", "birds", "faces"
- **Hybrid** (30%): "blue sky", "brown roofs", "yellow buildings"

---

### 3. Comprehensive Diagnostics & Fallbacks

**v2.0 Problem**:
```python
# Black box failure
if len(sam_masks) == 0:
    raise ValueError("SAM failed")  # No diagnostics, no fallback
```

**v3.0 Solution**:
```python
# Detailed diagnostics
logging.info(f"SAM input: image_shape={image.shape}, color_coverage={mask.mean():.2%}")

# Fallback with relaxed parameters
if len(sam_masks) == 0:
    logging.warning("SAM 0 masks, trying relaxed parameters")
    sam_masks = sam_fallback(image, pred_iou_thresh=0.70, stability_score=0.85)

# Detailed failure report
if len(sam_masks) == 0:
    logging.error(f"SAM failed. Image stats: dtype={image.dtype}, "
                 f"min={image.min()}, max={image.max()}")
    return []  # Graceful degradation
```

---

### 4. Post-Filtering for Semantic Queries

**New in v3.0**: Additional filtering by size, position, shape

```python
# Example: "small birds" query
intent = IntentV3(entities=["birds"], size_hint="small", detection_strategy="semantic_only")

# After CLIP filtering
filtered = post_filters.apply(
    entity_masks,
    size_hint="small",  # Keep only entities < 2% of image
    position_hint=None,
    image_shape=(H, W)
)
```

**Filtering Capabilities**:
- **Size**: small (< 2%), large (> 5%)
- **Position**: top (< 33%), center (33-67%), bottom (> 67%)
- **Aspect ratio**: Remove extreme elongations (< 0.2 or > 5.0)

---

### 5. Hybrid Path with Result Merging

**New in v3.0**: Combine color-guided and semantic-only results

```python
# Run both paths
color_results = color_guided_path(image, intent)
semantic_results = semantic_only_path(image, intent)

# Merge and deduplicate by IoU
merged = merge_results(color_results, semantic_results, iou_threshold=0.5)
```

**Benefits**:
- ✅ Best of both worlds (precision + recall)
- ✅ Handles ambiguous queries ("blue sky" = position + color)
- ✅ Removes duplicate detections

---

## 🎯 Expected Performance

| Metric | v2.0 Actual | v3.0 Target | Method |
|--------|-------------|-------------|--------|
| **Success Rate** | 11% (1/9) | 85% (7.5/9) | Wildcard tests |
| **False Positive Rate** | 22% (2/9) | 0% | No toxic fallback |
| **Query Coverage** | 40% (color only) | 100% (all types) | Dual-path routing |
| **Color Coverage** | 40% (8 colors) | 100% (dynamic) | LLM-based mapping |
| **Processing Time (color)** | 10s | <5s | Optimized pipeline |
| **Processing Time (semantic)** | N/A (crashed) | <8s | Full SAM acceptable |

---

## 📊 Validation Strategy

### Ground Truth Dataset
- 50+ manually annotated images
- Diverse scenes (urban, rural, indoor, edge cases)
- Precision/recall metrics for each query type

### Benchmark Suite
- 9 wildcard tests (from v2.0)
- 20 expanded semantic tests
- 10 hybrid query tests
- 5 edge case tests

### Performance Targets
- **Precision**: ≥ 85% (correct detections / total detections)
- **Recall**: ≥ 80% (correct detections / ground truth entities)
- **F1 Score**: ≥ 0.82
- **False Positive Rate**: < 5%

---

## 🔄 Implementation Status

### Phase 1: Foundation (Week 1) - 🚧 IN PROGRESS
- [ ] Workspace setup
- [ ] Copy working components (Stage 3, 4, 5, 6)
- [ ] Implement dynamic color mapper
- [ ] Enhance intent parser with routing
- [ ] Implement post-filters
- [ ] Unit tests for all modules

### Phase 2: Integration (Week 2) - ⏳ PENDING
- [ ] Implement dual-path orchestrator
- [ ] Implement result merger
- [ ] Integration tests
- [ ] Wildcard tests (target: 5-6/9 pass)
- [ ] Performance benchmarks

### Phase 3: Validation System (Week 3) - ⏳ PENDING
- [ ] Vision delta analysis
- [ ] Quality scoring
- [ ] Auto-retry loop
- [ ] Structured feedback integration

### Phase 4: Production (Week 4) - ⏳ PENDING
- [ ] Ground truth dataset (50+ images)
- [ ] Precision/recall metrics
- [ ] Performance optimization
- [ ] Documentation
- [ ] User testing

---

## 🚀 Quick Start (After Implementation)

### CLI Interface
```bash
# Auto-detect strategy
python app.py --image img.jpg --prompt "red vehicles" --output result.png

# Force specific strategy
python app.py --image img.jpg --prompt "birds" --output result.png --strategy semantic-only

# Hybrid mode
python app.py --image img.jpg --prompt "blue sky" --output result.png --strategy hybrid

# With validation
python app.py --image img.jpg --prompt "brown roofs" --output result.png --validate
```

### TUI Interface
```bash
# Interactive mode
python tui.py
```

---

## 📚 Key Documentation

1. **CRITICAL_ARCHITECTURE_FLAWS.md**: Complete root cause analysis of v2.0 failures
2. **V2_TO_V3_MIGRATION_ANALYSIS.md**: Component-by-component migration plan
3. **QWEN_SUPERVISION_PLAN.md**: Detailed implementation specifications for Qwen
4. **EDIT_VALIDATION_RESEARCH.md**: Auto-validation system architecture
5. **NEXT_STEPS_VALIDATION.md**: Integration roadmap

---

## 🎓 Lessons from V2

### What Worked ✅
- SAM 2.1 segmentation (when given good input)
- CLIP semantic filtering
- VLM validation feedback
- 6-stage pipeline structure
- CLI/TUI interfaces

### What Failed ❌
- Color-first assumption (60% of queries crashed)
- Static color dictionary (limited coverage)
- Toxic fallback behavior (false positives)
- No semantic-only path
- Black box SAM failures

### Core Principles for V3 ✨
1. **No silent failures**: Return `None` or raise clear errors
2. **Query-aware routing**: Different paths for different query types
3. **Graceful degradation**: Fallbacks with diagnostics
4. **Comprehensive diagnostics**: Log everything
5. **User-centric**: Clear error messages with suggestions

---

## 🤝 Development Team

**Claude Code (Supervisor)**:
- Architecture design
- Quality assurance
- Performance validation
- Edge case analysis
- Documentation review

**Qwen CLI (Implementer)**:
- Code implementation
- Unit testing
- Integration testing
- Debugging
- Documentation

---

## 📈 Success Criteria

**Minimum Viable Product (MVP)**:
- [ ] 7/9 wildcard tests pass (78% success rate)
- [ ] 0 false positives (0% false positive rate)
- [ ] All query types supported (color, semantic, hybrid)
- [ ] Processing time < 8 seconds average
- [ ] No crashes on valid inputs

**Production Ready**:
- [ ] 8/9 wildcard tests pass (89% success rate)
- [ ] Precision ≥ 85%, Recall ≥ 80%
- [ ] 50+ ground truth images validated
- [ ] User testing with 5+ alpha users
- [ ] Documentation complete

---

## 📞 Support & Feedback

**Issues**: Document in logs/ directory with:
- Input image path
- User prompt
- Expected vs actual results
- Error logs
- Performance metrics

**Questions**: Refer to QWEN_SUPERVISION_PLAN.md for detailed specifications

---

**v3.0 Mission**: Build a vision pipeline that "just works" for 85%+ of real-world editing queries.

*"In v2.0, we learned what doesn't work. In v3.0, we build what does."*
