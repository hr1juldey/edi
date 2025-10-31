# 🚨 CRITICAL CORRECTION INSTRUCTIONS FOR QWEN

**Date**: 2025-10-31
**Issue**: Incorrect test file placement

---

## ❌ PROBLEM IDENTIFIED

### Test Files in Wrong Location

**Current (INCORRECT)**:
```
work/edi_vision_v3/
├── test_comprehensive_sam.py        ❌ WRONG - Root directory
├── test_sam_fallback.py              ❌ WRONG - Root directory
├── test_stage3_enhanced.py           ❌ WRONG - Root directory
└── tests/
    └── __init__.py                   (empty folder)
```

**Expected (CORRECT)**:
```
work/edi_vision_v3/
└── tests/
    ├── __init__.py
    ├── test_comprehensive_sam.py     ✅ CORRECT location
    ├── test_sam_fallback.py          ✅ CORRECT location
    └── test_stage3_enhanced.py       ✅ CORRECT location
```

---

## ✅ IMMEDIATE CORRECTIVE ACTION

**Execute these commands NOW**:

```bash
cd /home/riju279/Documents/Code/Zonko/EDI/edi/work/edi_vision_v3

# Move all test files to tests/ folder
mv test_comprehensive_sam.py tests/
mv test_sam_fallback.py tests/
mv test_stage3_enhanced.py tests/

# Verify tests/ folder
ls -la tests/

# Verify root is clean (no .py files)
ls -la *.py 2>/dev/null
```

**After moving files, report**:
```
CORRECTIVE ACTION COMPLETE:
- Moved 3 test files to tests/: YES/NO
- tests/ folder contains 4 files total: YES/NO
- Root directory has no .py files: YES/NO
```

---

## 🚨 MANDATORY RULES - FOLLOW EXACTLY

### RULE 1: Test File Location
**ALL test files MUST be created in `tests/` folder. NEVER in root.**

**CORRECT**:
```bash
# Always create tests here
vim tests/test_something.py
python tests/test_something.py
```

**INCORRECT**:
```bash
# NEVER create tests here
vim test_something.py        ❌ WRONG
python test_something.py     ❌ WRONG
```

### RULE 2: Project Structure
**Only these locations are allowed for different file types**:

```
work/edi_vision_v3/
├── docs/              → Documentation (.md files only)
├── pipeline/          → Pipeline modules (stage*.py)
├── tests/             → ALL test files (test_*.py)
├── validation/        → Validation system (future)
├── logs/              → Log outputs (auto-generated)
├── images/            → Test images
├── README.md          → Project readme
└── sam2.1_b.pt        → Model weights
```

**NO Python files (.py) should exist in root directory.**

### RULE 3: Wait for Validation
After completing each task:
1. Submit completion report
2. **STOP - Do not proceed**
3. Wait for Claude's APPROVED/NEEDS REVISION
4. Only proceed when APPROVED

**Do NOT start next task before previous is approved.**

---

## 📋 CORRECTED TASK 1.2 COMPLETION REPORT

After moving files, submit this:

```markdown
=== TASK 1.2 COMPLETION REPORT (CORRECTED) ===

Task: Copy Stage 3 (SAM) with Enhancements
Status: COMPLETE

CORRECTIVE ACTION:
- Moved all test files from root to tests/: YES
- Verified tests/ folder structure: YES
- Verified root has no .py files: YES

Files Created/Modified:
- pipeline/stage3_sam_segmentation.py (8778 bytes)
- tests/test_comprehensive_sam.py
- tests/test_sam_fallback.py
- tests/test_stage3_enhanced.py

Enhancements Implemented:
- [X] Enhancement 1: Input diagnostics
- [X] Enhancement 2: Fallback SAM with relaxed parameters
- [X] Enhancement 3: Detailed failure diagnostics
- [X] Added scipy import

File Organization:
- All files in correct locations: YES
- Project structure follows rules: YES

Ready for Validation: YES

=== END REPORT ===
```

---

## 🎯 NEXT STEPS

1. **Execute corrective action** (move files)
2. **Submit corrected report** (above format)
3. **STOP and WAIT** for Claude validation
4. **Do NOT start Task 1.3** until approved

---

**This is a strict instruction. Follow exactly.**
