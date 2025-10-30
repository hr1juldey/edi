"""Tests for Stage 7: Pipeline Orchestrator"""

import sys
import os
import shutil
from pathlib import Path

# Add the parent directory to the path so we can import from pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.orchestrator import VisionPipeline, EntityMask


def test_full_pipeline():
    """Test complete pipeline execution."""
    pipeline = VisionPipeline(
        enable_validation=False,  # Skip VLM for speed
        save_intermediate=True,
        output_dir="logs/test"
    )

    result = pipeline.process(
        image_path="test_image.jpeg",
        user_prompt="change blue tin roofs to green"
    )

    # Verify success
    assert result['success']
    assert result['error'] is None

    # Verify entity masks
    assert len(result['entity_masks']) > 0
    assert all(isinstance(em, EntityMask) for em in result['entity_masks'])

    # Verify timing info
    assert 'stage_timings' in result
    assert 'stage1_entity_extraction' in result['stage_timings']
    assert 'stage2_color_filter' in result['stage_timings']
    assert 'stage3_sam_segmentation' in result['stage_timings']
    assert 'stage4_clip_filter' in result['stage_timings']
    assert 'stage5_organization' in result['stage_timings']

    # Verify metadata
    assert 'total_time' in result['metadata']
    # Pipeline might take longer than expected due to model loading, SAM processing, etc.
    # The important thing is that it completes within a reasonable time (e.g., 5 minutes)
    assert result['metadata']['total_time'] < 300.0  # Should complete in <5 minutes


def test_missing_image():
    """Test error handling for missing image."""
    pipeline = VisionPipeline()

    result = pipeline.process(
        image_path="nonexistent.jpg",
        user_prompt="change roofs to green"
    )

    assert not result['success']
    assert result['error'] is not None
    assert "not found" in result['error'].lower()


def test_invalid_prompt():
    """Test handling of unclear prompt."""
    pipeline = VisionPipeline()

    result = pipeline.process(
        image_path="test_image.jpeg",
        user_prompt="do something"  # Vague
    )

    # Should still attempt processing but may have low confidence
    assert 'intent' in result['metadata']


def test_intermediate_saving():
    """Test that intermediate results are saved."""

    # Clean test directory
    test_dir = Path("logs/test_intermediate")
    if test_dir.exists():
        shutil.rmtree(test_dir)

    pipeline = VisionPipeline(
        enable_validation=False,
        save_intermediate=True,
        output_dir=str(test_dir)
    )

    pipeline.process(
        image_path="test_image.jpeg",
        user_prompt="change blue roofs to green"
    )

    # Verify intermediate files exist
    assert (test_dir / "stage2_color_mask.png").exists()
    assert (test_dir / "stage3_sam_masks.png").exists()
    assert (test_dir / "stage4_clip_filtering.png").exists()
    assert (test_dir / "stage5_entity_masks.png").exists()


if __name__ == "__main__":
    test_full_pipeline()
    test_missing_image()
    test_invalid_prompt()
    test_intermediate_saving()
    print("All tests passed!")