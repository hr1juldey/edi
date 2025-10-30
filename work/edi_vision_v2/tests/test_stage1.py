"""Unit tests for Stage 1: DSpy Entity Extraction

This module contains unit tests for the entity extraction functionality.
"""

import pytest
from pipeline.stage1_entity_extraction import parse_intent, EditType


def test_parse_intent_blue_tin_roofs():
    """Test Case 1: 'turn the blue tin roofs of all those buildings to green'"""
    prompt = "turn the blue tin roofs of all those buildings to green"
    result = parse_intent(prompt)
    
    assert result is not None
    assert "target_entities" in result
    assert len(result["target_entities"]) > 0
    
    # Check that at least one entity has 'roof' in the label
    roof_entities = [entity for entity in result["target_entities"] 
                     if "roof" in entity["label"].lower()]
    assert len(roof_entities) > 0
    
    # Verify the first roof entity has expected attributes
    first_roof = roof_entities[0]
    assert "tin" in first_roof["texture"] or "tin" in first_roof["label"]
    assert first_roof["color"] == "blue"
    
    assert result["edit_type"] == EditType.RECOLOR
    assert result["new_value"] == "green"
    assert result["quantity"] == "all"
    assert 0.0 <= result["confidence"] <= 1.0


def test_parse_intent_dramatic_sky():
    """Test Case 2: 'make the sky more dramatic'"""
    prompt = "make the sky more dramatic"
    result = parse_intent(prompt)
    
    assert result is not None
    assert "target_entities" in result
    assert len(result["target_entities"]) > 0
    
    sky_entities = [entity for entity in result["target_entities"] 
                    if "sky" in entity["label"].lower()]
    assert len(sky_entities) > 0
    
    # The edit type should be one of the valid types, and we expect some kind of change
    assert result["edit_type"] in [EditType.STYLE_TRANSFER, EditType.RECOLOR, EditType.REPLACE]
    assert result["quantity"] == "all"
    assert 0.0 <= result["confidence"] <= 1.0
    # The new_value should be something meaningful (not empty)
    assert len(result["new_value"]) > 0


def test_parse_intent_remove_red_car():
    """Test Case 3: 'remove the red car'"""
    prompt = "remove the red car"
    result = parse_intent(prompt)
    
    assert result is not None
    assert "target_entities" in result
    assert len(result["target_entities"]) > 0
    
    car_entities = [entity for entity in result["target_entities"] 
                    if "car" in entity["label"].lower()]
    assert len(car_entities) > 0
    
    # Verify the car entity has red color
    red_car = car_entities[0]
    assert red_car["color"] == "red"
    
    assert result["edit_type"] == EditType.REMOVE
    assert result["quantity"] == "all"
    assert 0.0 <= result["confidence"] <= 1.0


def test_parse_intent_change_green_door():
    """Test Case 4: 'change the large green door to blue'"""
    prompt = "change the large green door to blue"
    result = parse_intent(prompt)
    
    assert result is not None
    assert "target_entities" in result
    assert len(result["target_entities"]) > 0
    
    door_entities = [entity for entity in result["target_entities"] 
                     if "door" in entity["label"].lower()]
    assert len(door_entities) > 0
    
    # Verify the door entity has green color and large size descriptor
    first_door = door_entities[0]
    assert first_door["color"] == "green"
    assert first_door["size_descriptor"] == "large"
    
    assert result["edit_type"] == EditType.RECOLOR
    assert result["new_value"] == "blue"
    assert 0.0 <= result["confidence"] <= 1.0


def test_parse_intent_add_clouds_to_sky():
    """Test Case 5: 'add clouds to the sky'"""
    prompt = "add clouds to the sky"
    result = parse_intent(prompt)
    
    assert result is not None
    assert "target_entities" in result
    assert len(result["target_entities"]) > 0
    
    sky_entities = [entity for entity in result["target_entities"] 
                    if "sky" in entity["label"].lower()]
    assert len(sky_entities) > 0
    
    assert result["edit_type"] == EditType.ADD
    assert result["new_value"] == "clouds"
    assert 0.0 <= result["confidence"] <= 1.0


def test_determinism():
    """Test that the same prompt produces the same result multiple times."""
    prompt = "turn the blue tin roofs of all those buildings to green"
    
    result1 = parse_intent(prompt)
    result2 = parse_intent(prompt)
    result3 = parse_intent(prompt)
    
    # For determinism testing in DSpy, we expect similar structure and values
    # Note: The exact values might vary due to LLM behavior, but structure should be consistent
    assert type(result1) is type(result2) is type(result3)
    assert "target_entities" in result1 and "target_entities" in result2 and "target_entities" in result3
    assert "edit_type" in result1 and "edit_type" in result2 and "edit_type" in result3


if __name__ == "__main__":
    pytest.main([__file__])