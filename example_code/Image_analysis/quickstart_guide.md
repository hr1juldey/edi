# Quick Start Guide: Adaptive Image Edit Validation

## 🚀 Get Started in 5 Minutes

### Prerequisites
```bash
pip install torch opencv-python numpy pillow open_clip_torch \
            ultralytics scikit-image scipy matplotlib seaborn pandas tqdm
```

---

## 📋 Usage Scenarios

### Scenario 1: Single Image Validation

**Use Case**: You have one before/after pair to validate quickly.

```bash
python adaptive_validator.py \
  --before IP.jpeg \
  --after OP.jpeg \
  --prompt "blue tin roof and clouds over mountain peak" \
  --output result.png
```

**Output**:
```
============================================================
ADAPTIVE VALIDATION RESULTS (Statistical)
============================================================
Verdict       : FAIL
Score         : 50.0% (weighted)
Confidence    : 67.3%

Statistical Tests:
  Cohen's d        : 4.6342 (large)
  Mann-Whitney p   : 0.000000 (significant)

Inside Mask:
  Mean ΔE     : 17.9715
  Median ΔE   : 15.2341
  % Changed   : 54.82%

Outside Mask:
  Mean ΔE     : 3.8801
  Median ΔE   : 2.1234
  % Changed   : 12.0238%
  Max Allowed : 3.00%    ← VIOLATED

Signal Metrics:
  SNR (median) : 7.1543

Criteria Breakdown (Weighted):
  ✓ PASS: Sufficient Change             (weight: 1.5)
  ✗ FAIL: Low Outside Change            (weight: 2.0)  ← FAILED
  ✓ PASS: Good Effect Size               (weight: 1.5)
  ✓ PASS: Statistically Significant      (weight: 1.0)
  ✓ PASS: Good SNR                       (weight: 1.0)
  ✗ FAIL: IQR Separation                 (weight: 0.5)

Weighted Score: 5.50 / 8.00
============================================================
```

**Interpretation**:
- ✓ Edit happened (large Cohen's d)
- ✓ Statistically significant
- ✗ **12% of outside pixels changed** (allowed: 3%)
- **Verdict: FAIL** - Too much unintended change

---

### Scenario 2: Compare Methods

**Use Case**: Understand why statistical method differs from fixed thresholds.

```bash
python comparison_analyzer.py \
  --before IP.jpeg \
  --after OP.jpeg \
  --prompt "blue tin roof and clouds over mountain peak" \
  --output comparison.png
```

**Output**: 
- Side-by-side comparison visualization
- Distribution plots (inside vs outside changes)
- Box plots with statistical annotations
- Comparison table showing both methods

**Key Insight**: Shows how adaptive thresholds adjust based on edit magnitude.

---

### Scenario 3: Batch Processing

**Use Case**: Validate 10-100+ image edits, identify patterns.

#### Step 1: Create Config
```bash
# Option A: From directory
python config_generator.py \
  --input-dir ./my_edits \
  --output config.json

# Option B: From CSV
python config_generator.py \
  --csv edits.csv \
  --output config.json

# Option C: Start with template
python config_generator.py \
  --template \
  --output config.json \
  --num-examples 5
```

**Expected directory structure** (Option A):
```
my_edits/
├── before_roof_edit.jpg
├── after_roof_edit.jpg
├── prompt_roof_edit.txt        # Contains: "change roof to blue"
├── before_door_edit.jpg
├── after_door_edit.jpg
├── prompt_door_edit.txt         # Contains: "add green door"
└── ...
```

**CSV format** (Option B):
```csv
name,before,after,prompt
roof_edit,./images/before_1.jpg,./images/after_1.jpg,"change roof to blue"
door_edit,./images/before_2.jpg,./images/after_2.jpg,"add green door"
```

#### Step 2: Validate Config
```bash
python config_generator.py --validate config.json
```

#### Step 3: Run Batch Analysis
```bash
python batch_analyzer.py \
  --config config.json \
  --output-dir batch_results/
```

**Outputs**:
```
batch_results/
├── batch_summary.png          # 6-panel visualization
├── results.csv                # Detailed metrics per edit
└── insights.txt               # Actionable recommendations
```

**Example insights.txt**:
```
======================================================================
ACTIONABLE INSIGHTS
======================================================================

1. OVERALL QUALITY
   Pass rate: 65.0%
   ⚡ MODERATE: Majority pass but significant failures remain.
   Action: Investigate failed cases for patterns.

2. FAILURE ANALYSIS (7 failures)
   Most common failure: low_outside (42.9% pass rate)
   Issue: Excessive unintended changes outside edit region.
   Average outside change in failures: 9.23%
   Action: Improve masking or use more controlled editing methods.

3. EDIT STRENGTH ANALYSIS
   Average Cohen's d: 2.345
   Status: Large effect sizes (strong edits)
   Note: System applies stricter validation for obvious edits.

4. OUTLIER DETECTION
   3 edits with high outside change (>90th percentile):
   - mountain_sky: 15.23% outside changed
   - beach_scene: 13.45% outside changed
   - urban_edit: 11.87% outside changed
   Action: Manual review recommended for these cases.

5. RECOMMENDATIONS
   • Only 78.0% of edits are statistically significant (p<0.01)
     Consider: Stronger edits or better targeting
   • Average 7.23% outside change (target: <5%)
     Consider: Better mask generation or attention mechanisms
======================================================================
```

---

## 🎯 Decision Tree: Which Tool to Use?

```
Do you have one image pair?
├─ Yes → Use adaptive_validator.py
└─ No → Do you have many pairs?
    ├─ Yes → Use batch_analyzer.py
    └─ No → Are results confusing?
        └─ Yes → Use comparison_analyzer.py
```

---

## 📊 Understanding the Metrics

### Cohen's d (Effect Size)
- **< 0.2**: Negligible edit
- **0.2 - 0.5**: Small edit (system more lenient, allows 15% outside)
- **0.5 - 0.8**: Medium edit (allows 8% outside)
- **> 0.8**: Large edit (strict, allows 3% outside)

### Mann-Whitney p-value
- **< 0.01**: Highly significant (inside ≠ outside, statistically proven)
- **0.01 - 0.05**: Significant
- **> 0.05**: Not significant (edit may be too subtle)

### Score (Weighted)
- **≥ 70%**: PASS
- **50-70%**: Borderline (consider LMM review)
- **< 50%**: FAIL

### Confidence
- **> 80%**: High confidence in verdict
- **60-80%**: Moderate confidence
- **< 60%**: Low confidence (may need human review)

---

## 🔧 Troubleshooting

### Issue: "SAM returned no masks"
**Solution**: Image quality issue or SAM checkpoint missing
```bash
# Download SAM checkpoint if needed
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O sam2.1_b.pt
```

### Issue: "Cannot load CLIP"
**Solution**: Install open_clip_torch
```bash
pip install open_clip_torch
```

### Issue: All edits failing with low effect size
**Solution**: Edits may be too subtle. Check:
1. Are the before/after images actually different?
2. Is the prompt matching the actual edit region?
3. Try visualizing delta map to see where changes occur

### Issue: High outside change but edit looks clean
**Possible causes**:
1. Mask is too small (doesn't cover full edit region)
2. Global lighting changes (acceptable but flagged)
3. JPEG compression artifacts

**Solution**: Use comparison tool to visualize:
```bash
python comparison_analyzer.py --before B.jpg --after A.jpg --prompt "your prompt"
# Check the red/blue overlay - is red (inside change) in right place?
```

---

## 🎨 Customizing Thresholds

If you want to adjust the adaptive behavior:

**Edit `adaptive_validator.py` line ~90**:
```python
# Current adaptive logic
if cohens_d > 1.0:  # large effect - stringent
    max_outside_ratio = 0.03  # 3%
elif cohens_d > 0.5:  # medium effect
    max_outside_ratio = 0.08  # 8%
else:  # small effect - lenient
    max_outside_ratio = 0.15  # 15%

# Make it more/less strict
if cohens_d > 1.0:
    max_outside_ratio = 0.05  # More lenient: 5% instead of 3%
elif cohens_d > 0.5:
    max_outside_ratio = 0.10  # More lenient: 10% instead of 8%
else:
    max_outside_ratio = 0.20  # More lenient: 20% instead of 15%
```

**⚠️ Warning**: Loosening thresholds may hide real quality issues!

---

## 🧪 Example Workflows

### Workflow 1: Development Iteration
```bash
# 1. Test single edit
python adaptive_validator.py --before B.jpg --after A.jpg --prompt "blue roof"

# 2. Failed? Debug with comparison
python comparison_analyzer.py --before B.jpg --after A.jpg --prompt "blue roof"

# 3. Adjust editing pipeline (improve masking, etc.)

# 4. Re-test
python adaptive_validator.py --before B.jpg --after A_v2.jpg --prompt "blue roof"
```

### Workflow 2: Quality Audit
```bash
# 1. Generate config from production edits
python config_generator.py --input-dir ./prod_edits --output audit.json

# 2. Run batch analysis
python batch_analyzer.py --config audit.json --output-dir audit_results/

# 3. Review insights
cat audit_results/insights.txt

# 4. Manually review outliers listed in insights
# 5. Update editing pipeline based on patterns
```

### Workflow 3: CI/CD Integration
```bash
#!/bin/bash
# In your CI pipeline

# Run validation
python adaptive_validator.py --before $BEFORE --after $AFTER --prompt "$PROMPT"
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Edit quality passed"
    # Proceed with deployment
else
    echo "✗ Edit quality failed"
    # Block deployment, notify team
    exit 1
fi
```

---

## 📈 Next Steps

### After First Run
1. **Check your pass rate** (from batch analysis)
   - < 50%: Major issues, review editing pipeline
   - 50-80%: Some issues, optimize masking/prompts
   - > 80%: Good quality, minor tweaks only

2. **Identify failure patterns** (from insights.txt)
   - Consistent failures on one criterion → targeted fix
   - Random failures → may be acceptable variance

3. **Calibrate expectations**
   - Some outside change is inevitable with diffusion models
   - Goal is < 5% outside change for production quality

### Advanced Usage
- Integrate LMM evaluation (see research_recommendations.md)
- Add perceptual metrics (LPIPS)
- Implement attention-based masking
- Build feedback loop for continuous improvement

---

## 📚 File Reference

| File | Purpose | When to Use |
|------|---------|-------------|
| `adaptive_validator.py` | Single image validation | Quick checks, development |
| `comparison_analyzer.py` | Method comparison | Understanding why results differ |
| `batch_analyzer.py` | Bulk validation | Production audits, QA |
| `config_generator.py` | Create batch configs | Setup for batch processing |
| `research_recommendations.md` | In-depth guide | Understanding theory, next steps |

---

## 💬 Support

### Common Questions

**Q: Why did my edit fail when it looks good?**
A: Check the overlay visualization. Sometimes global lighting changes flag as "outside change" even if they're acceptable.

**Q: Can I use this for video frames?**
A: Yes, but consider adding temporal consistency checks (see research_recommendations.md).

**Q: How do I know if my thresholds are right?**
A: Run batch analysis on 20-50 diverse edits. If pass rate is 50-80%, thresholds are reasonable.

**Q: What's the runtime?**
A: ~2-5 seconds per image (SAM + CLIP + statistical analysis) on GPU.

---

## 🎓 Learning Path

1. **Day 1**: Run adaptive_validator.py on 3-5 test cases
2. **Day 2**: Run comparison_analyzer.py to understand differences
3. **Day 3**: Set up batch processing for your dataset
4. **Week 2**: Analyze patterns, adjust editing pipeline
5. **Month 1**: Integrate into CI/CD, iterate based on insights

Good luck! 🚀
