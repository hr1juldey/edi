"""Stage 1: DSpy Entity Extraction

This module implements the entity extraction stage of the EDI Vision V2 pipeline.
It uses DSpy to extract structured editing intents from natural language prompts.
"""

import logging
from enum import Enum
from typing import List, Optional
import dspy
from pydantic import BaseModel


class EditType(str, Enum):
    """Enumeration of different types of edit operations."""
    RECOLOR = "recolor"
    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"
    STYLE_TRANSFER = "style_transfer"


class EntityDescription(BaseModel):
    """Model describing a target entity to be edited.

    Attributes:
        label: The general label of the entity (e.g., "tin roof")
        color: The color of the entity (e.g., "blue")
        texture: The texture of the entity (e.g., "tin", "metallic")
        size_descriptor: Size descriptor of the entity (e.g., "large", "small", "all")
    """
    label: str  # e.g., "tin roof"
    color: Optional[str] = None  # e.g., "blue"
    texture: Optional[str] = None  # e.g., "tin", "metallic"
    size_descriptor: Optional[str] = None  # e.g., "large", "small", "all"


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


class IntentParser(dspy.Module):
    """Module to parse user editing intent using DSpy ChainOfThought."""

    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(ExtractEditIntent)

    def forward(self, user_prompt: str):
        """Parse the user prompt to extract structured editing intent.

        Args:
            user_prompt: Natural language edit request from user

        Returns:
            Parsed intent with target entities, edit type, etc.
        """
        result = self.extractor(user_prompt=user_prompt)
        return result
        
    def __call__(self, user_prompt: str):
        """Call method to support the recommended DSpy usage pattern."""
        return self.forward(user_prompt)


def parse_intent(prompt: str, llm_model: str = "ollama/qwen3:8b") -> dict:
    """Convenience function to parse user intent using DSpy.

    Args:
        prompt: Natural language edit request from user
        llm_model: Name of the LLM model to use (default: ollama/qwen3:8b)

    Returns:
        Dictionary representation of the parsed intent
    """
    try:
        # Configure DSpy with the specified LLM
        logging.info(f"Configuring DSpy with model: {llm_model}")
        llm = dspy.LM(llm_model)
        dspy.configure(lm=llm)
        
        # Create an instance of the IntentParser
        parser = IntentParser()
        
        # Parse the intent
        result = parser(user_prompt=prompt)
        
        # Convert result to dictionary format
        target_entities_dict = [
            {
                "label": entity.label,
                "color": entity.color,
                "texture": entity.texture,
                "size_descriptor": entity.size_descriptor
            } 
            for entity in result.target_entities
        ]
        
        intent_dict = {
            "target_entities": target_entities_dict,
            "edit_type": result.edit_type,
            "new_value": result.new_value,
            "quantity": result.quantity,
            "confidence": result.confidence
        }
        
        logging.info(f"Parsed intent: {intent_dict}")
        return intent_dict
        
    except Exception as e:
        logging.error(f"Error parsing intent: {e}")
        # Return a default low-confidence output if DSpy fails
        return {
            "target_entities": [],
            "edit_type": "recolor",
            "new_value": "",
            "quantity": "all",
            "confidence": 0.1
        }