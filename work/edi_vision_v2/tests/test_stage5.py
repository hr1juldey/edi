"""Unit tests for Stage 5: Mask Organization

This module contains unit tests for the mask organization functionality.
"""

import pytest
import numpy as np
from pipeline.stage5_mask_organization import organize_masks, EntityMask


def test_organize_masks():
    """Test that masks are organized into EntityMask objects."""
    # Create synthetic image
    image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

    # Create 3 test masks
    mask1 = np.zeros((200, 200), dtype=np.uint8)
    mask1[50:100, 50:100] = 1

    mask2 = np.zeros((200, 200), dtype=np.uint8)
    mask2[120:150, 120:150] = 1

    mask3 = np.zeros((200, 200), dtype=np.uint8)
    mask3[10:30, 10:30] = 1

    filtered_masks = [(mask1, 0.8), (mask2, 0.6), (mask3, 0.9)]

    entity_masks = organize_masks(image, filtered_masks)

    # Should get 3 separate EntityMask objects
    assert len(entity_masks) == 3

    # Each should be an EntityMask
    for entity_mask in entity_masks:
        assert isinstance(entity_mask, EntityMask)
        assert isinstance(entity_mask.mask, np.ndarray)
        assert isinstance(entity_mask.entity_id, int)
        assert isinstance(entity_mask.similarity_score, float)
        assert isinstance(entity_mask.bbox, tuple)
        assert len(entity_mask.bbox) == 4
        assert isinstance(entity_mask.centroid, tuple)
        assert len(entity_mask.centroid) == 2
        assert isinstance(entity_mask.area, int)
        assert entity_mask.area > 0
        assert isinstance(entity_mask.dominant_color, tuple)
        assert len(entity_mask.dominant_color) == 3


def test_metadata_correctness():
    """Test that metadata is calculated correctly."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[20:40, 20:40] = [100, 150, 200]  # Blue region

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 20:40] = 1

    entity_masks = organize_masks(image, [(mask, 0.75)])

    assert len(entity_masks) == 1
    entity = entity_masks[0]

    # Verify bbox
    assert entity.bbox == (20, 20, 39, 39)

    # Verify centroid (center of 20:40 is 29.5)
    assert abs(entity.centroid[0] - 29.5) < 1.0
    assert abs(entity.centroid[1] - 29.5) < 1.0

    # Verify area
    assert entity.area == 20 * 20  # 400 pixels

    # Verify similarity score
    assert entity.similarity_score == 0.75

    # Verify dominant color (should be close to [100, 150, 200])
    assert abs(entity.dominant_color[0] - 100) < 5
    assert abs(entity.dominant_color[1] - 150) < 5
    assert abs(entity.dominant_color[2] - 200) < 5


def test_sorted_by_area():
    """Test that EntityMasks are sorted by area (largest first)."""
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    # Small mask (400 pixels)
    mask_small = np.zeros((200, 200), dtype=np.uint8)
    mask_small[10:30, 10:30] = 1

    # Large mask (2500 pixels)
    mask_large = np.zeros((200, 200), dtype=np.uint8)
    mask_large[50:100, 50:100] = 1

    # Medium mask (900 pixels)
    mask_medium = np.zeros((200, 200), dtype=np.uint8)
    mask_medium[120:150, 120:150] = 1

    filtered_masks = [(mask_small, 0.5), (mask_large, 0.6), (mask_medium, 0.7)]

    entity_masks = organize_masks(image, filtered_masks)

    # Should be sorted by area: large, medium, small
    assert entity_masks[0].area == 2500
    assert entity_masks[1].area == 900
    assert entity_masks[2].area == 400

    # Entity IDs should be reassigned: 0, 1, 2 after sorting
    assert entity_masks[0].entity_id == 0
    assert entity_masks[1].entity_id == 1
    assert entity_masks[2].entity_id == 2


def test_empty_masks_list():
    """Test behavior with empty masks list."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    entity_masks = organize_masks(image, [])

    assert entity_masks == []


def test_masks_stay_separate():
    """CRITICAL TEST: Verify masks are NOT merged."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Create 5 separate masks
    masks = []
    for i in range(5):
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Place each mask in a different location
        start_row = i * 15
        end_row = (i + 1) * 15
        start_col = i * 15
        end_col = (i + 1) * 15
        
        # Make sure we don't exceed image bounds
        if end_row > 100:
            start_row = 100 - 15
            end_row = 100
        if end_col > 100:
            start_col = 100 - 15
            end_col = 100
            
        mask[start_row:end_row, start_col:end_col] = 1
        masks.append((mask, 0.5 + i*0.05))

    entity_masks = organize_masks(image, masks)

    # CRITICAL: Should get 5 separate EntityMask objects
    assert len(entity_masks) == 5, f"Expected 5 separate masks, got {len(entity_masks)}"

    # Each EntityMask should have a unique mask array
    for i in range(len(entity_masks)):
        for j in range(i+1, len(entity_masks)):
            # Masks should NOT be identical
            assert not np.array_equal(entity_masks[i].mask, entity_masks[j].mask), \
                f"Masks {i} and {j} are identical - they were merged!"
    
    # Verify that each mask has the same number of pixels (should be 15x15=225 for all)
    expected_area = 15 * 15  # pixels
    for entity in entity_masks:
        assert entity.area == expected_area, f"Area {entity.area} not equal to expected {expected_area}"


if __name__ == "__main__":
    pytest.main([__file__])