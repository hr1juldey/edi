# Reasoning: Intent Parser

[Back to Reasoning Subsystem](./reasoning_subsystem.md)

## Purpose
DSpy intent extraction - Contains the IntentParser class (a dspy.Module) that processes naive prompts and scene analysis to extract structured intent.

## Class: IntentParser(dspy.Module)

### Methods
- `forward(naive_prompt, scene) -> Intent`: Takes a naive prompt and scene analysis and returns structured intent

### Details
- Detects ambiguity in user requests
- Generates clarifying questions when confidence is low
- Uses DSpy framework for structured LLM interactions

## AI Action Required: External Library Investigation

This module uses the **DSpy** library

1.  **Request for Information:** Provide a link to the official documentation for `DSpy`

2.  **Search and Test:** I will search for examples of `dspy.Signature` and `dspy.ChainOfThought` to verify my understanding before implementation

## Functions

- [forward(naive_prompt, scene)](./reasoning/forward_intent.md)

## Technology Stack

- DSpy for LLM orchestration
- Pydantic for data validation

## See Docs

### Python Implementation Example
Reasoning intent parser implementation using DSpy:

```python
import dspy
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum
import logging

class IntentError(Exception):
    """Custom exception for intent parsing errors."""
    pass

class EditType(Enum):
    """Enumeration for edit types."""
    COLOR = "color"
    STYLE = "style"
    ADD = "add"
    REMOVE = "remove"
    TRANSFORM = "transform"

class Intent(BaseModel):
    """
    Structured intent extracted from a naive prompt.
    """
    naive_prompt: str = Field(..., description="Original user prompt")
    target_entities: List[str] = Field([], description="List of entity IDs to edit")
    edit_type: EditType = Field(EditType.TRANSFORM, description="Type of edit to perform")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in intent parsing")
    clarifying_questions: List[str] = Field([], description="Questions to clarify ambiguous requests")
    scene_analysis: Optional[Dict[str, Any]] = Field(None, description="Scene analysis data")

class SceneAnalysis(BaseModel):
    """
    Scene analysis data structure.
    """
    entities: List[Dict[str, Any]] = Field([], description="Detected entities in the scene")
    spatial_layout: str = Field("", description="Description of spatial relationships")
    image_description: str = Field("", description="Overall image description")

class ParseIntentSignature(dspy.Signature):
    """
    DSpy signature for parsing user intent from naive prompts and scene analysis.
    """
    naive_prompt = dspy.InputField(
        desc="User's conversational edit request"
    )
    scene_analysis = dspy.InputField(
        desc="JSON of detected entities and layout"
    )
    
    target_entities = dspy.OutputField(
        desc="Comma-separated list of entity IDs to edit"
    )
    edit_type = dspy.OutputField(
        desc="One of: color, style, add, remove, transform"
    )
    confidence = dspy.OutputField(
        desc="Float 0-1 indicating clarity of intent"
    )
    clarifying_questions = dspy.OutputField(
        desc="JSON array of questions if confidence <0.7"
    )

class IntentParser(dspy.Module):
    """
    DSpy module for parsing user intent from naive prompts and scene analysis.
    """
    
    def __init__(self):
        super().__init__()
        self.parse_intent = dspy.ChainOfThought(ParseIntentSignature)
        self.logger = logging.getLogger(__name__)
    
    def forward(self, naive_prompt: str, scene: Dict[str, Any]) -> Intent:
        """
        Takes a naive prompt and scene analysis and returns structured intent.
        
        Args:
            naive_prompt: User's conversational edit request
            scene: Scene analysis data
            
        Returns:
            Structured intent with target entities, edit type, confidence, and clarifying questions
        """
        # Validate inputs
        if not naive_prompt or not naive_prompt.strip():
            raise IntentError("Naive prompt cannot be empty")
        
        if not scene:
            raise IntentError("Scene analysis cannot be empty")
        
        # Detect ambiguity in user requests
        # This would be handled by the DSpy model
        
        try:
            # Process naive prompt and scene analysis to extract structured intent
            result = self.parse_intent(
                naive_prompt=naive_prompt,
                scene_analysis=json.dumps(scene, default=str)
            )
            
            # Parse results
            target_entities = result.target_entities.split(", ") if result.target_entities else []
            edit_type = EditType(result.edit_type) if result.edit_type in EditType._value2member_map_ else EditType.TRANSFORM
            confidence = float(result.confidence) if self._is_valid_confidence(result.confidence) else 0.0
            
            # Parse clarifying questions
            try:
                clarifying_questions = json.loads(result.clarifying_questions) if result.clarifying_questions else []
            except json.JSONDecodeError:
                clarifying_questions = []
            
            # Generate clarifying questions when confidence is low
            if confidence < 0.7 and not clarifying_questions:
                clarifying_questions = self._generate_clarifying_questions(naive_prompt, scene)
            
            # Create structured intent
            intent = Intent(
                naive_prompt=naive_prompt,
                target_entities=target_entities,
                edit_type=edit_type,
                confidence=confidence,
                clarifying_questions=clarifying_questions,
                scene_analysis=scene
            )
            
            self.logger.info(f"Parsed intent: {intent.edit_type.value} with confidence {intent.confidence}")
            return intent
            
        except Exception as e:
            self.logger.error(f"Failed to parse intent: {str(e)}")
            raise IntentError(f"Intent parsing failed: {str(e)}")
    
    def _is_valid_confidence(self, confidence_str: str) -> bool:
        """
        Check if confidence string is valid.
        
        Args:
            confidence_str: Confidence value as string
            
        Returns:
            Boolean indicating if valid
        """
        try:
            confidence = float(confidence_str)
            return 0.0 <= confidence <= 1.0
        except (ValueError, TypeError):
            return False
    
    def _generate_clarifying_questions(self, naive_prompt: str, scene: Dict[str, Any]) -> List[str]:
        """
        Generate clarifying questions when confidence is low.
        
        Args:
            naive_prompt: User's conversational edit request
            scene: Scene analysis data
            
        Returns:
            List of clarifying questions
        """
        # In a real implementation, this would use another DSpy module or LLM call
        # For this example, we'll generate some placeholder questions
        
        questions = []
        
        # Check for common ambiguous terms
        ambiguous_terms = ["better", "improve", "enhance", "good", "nice"]
        if any(term in naive_prompt.lower() for term in ambiguous_terms):
            questions.append("What specific aspect would you like to improve?")
        
        # Check for vague references to scene elements
        entities = scene.get("entities", [])
        if not entities:
            questions.append("Could you specify which part of the image you want to modify?")
        else:
            # Check if prompt references entities not clearly identified
            entity_labels = [entity.get("label", "").lower() for entity in entities]
            prompt_entities = [word.lower() for word in naive_prompt.split() if len(word) > 2]
            
            unmatched_entities = [
                entity for entity in prompt_entities 
                if not any(entity in label for label in entity_labels)
            ]
            
            if unmatched_entities:
                questions.append(f"Did you mean to refer to any of these entities: {', '.join(entity_labels)}?")
        
        # General clarifying questions
        questions.extend([
            "Can you be more specific about what changes you want?",
            "Which areas of the image should be affected by this edit?"
        ])
        
        return questions[:3]  # Limit to 3 questions
    
    def validate_intent(self, intent: Intent) -> bool:
        """
        Validate the parsed intent.
        
        Args:
            intent: Parsed intent to validate
            
        Returns:
            Boolean indicating if intent is valid
        """
        # Validate intent structure
        if not intent.naive_prompt or not intent.naive_prompt.strip():
            self.logger.error("Intent validation failed: Empty naive prompt")
            return False
        
        # Validate edit type
        if not isinstance(intent.edit_type, EditType):
            self.logger.error("Intent validation failed: Invalid edit type")
            return False
        
        # Validate confidence
        if not (0.0 <= intent.confidence <= 1.0):
            self.logger.error("Intent validation failed: Invalid confidence value")
            return False
        
        # Validate target entities if present
        if intent.target_entities and not isinstance(intent.target_entities, list):
            self.logger.error("Intent validation failed: Invalid target entities format")
            return False
        
        # Validate clarifying questions if present
        if intent.clarifying_questions and not isinstance(intent.clarifying_questions, list):
            self.logger.error("Intent validation failed: Invalid clarifying questions format")
            return False
        
        self.logger.info("Intent validation passed")
        return True
    
    def enhance_intent_with_context(self, intent: Intent, context: Dict[str, Any]) -> Intent:
        """
        Enhance intent with additional context information.
        
        Args:
            intent: Parsed intent to enhance
            context: Additional context information
            
        Returns:
            Enhanced intent
        """
        # Add context to scene analysis
        if intent.scene_analysis:
            intent.scene_analysis.update(context)
        else:
            intent.scene_analysis = context
        
        # Adjust confidence based on context
        if context.get("user_expertise") == "expert":
            intent.confidence = min(1.0, intent.confidence * 1.2)  # Boost confidence for experts
        
        # Add context-based clarifying questions
        if context.get("previous_edits"):
            intent.clarifying_questions.append("How should this edit relate to your previous edits?")
        
        self.logger.info("Intent enhanced with context")
        return intent

# Example usage
if __name__ == "__main__":
    # Initialize intent parser
    parser = IntentParser()
    
    print("Intent Parser initialized")
    
    # Example naive prompt and scene analysis
    naive_prompt = "make the sky more dramatic"
    scene_analysis = {
        "entities": [
            {"id": "sky_0", "label": "sky", "confidence": 0.95, "bbox": [0, 0, 1920, 768]},
            {"id": "mountain_1", "label": "mountain", "confidence": 0.87, "bbox": [20, 768, 1900, 1080]},
            {"id": "tree_2", "label": "tree", "confidence": 0.92, "bbox": [100, 800, 200, 1000]}
        ],
        "spatial_layout": "sky occupies top portion, mountains at bottom, trees in foreground",
        "image_description": "landscape with sky, mountains, and trees"
    }
    
    # Parse intent
    try:
        intent = parser.forward(naive_prompt, scene_analysis)
        print(f"Parsed intent:")
        print(f"  Naive prompt: {intent.naive_prompt}")
        print(f"  Target entities: {intent.target_entities}")
        print(f"  Edit type: {intent.edit_type.value}")
        print(f"  Confidence: {intent.confidence}")
        print(f"  Clarifying questions: {intent.clarifying_questions}")
        
        # Validate intent
        is_valid = parser.validate_intent(intent)
        print(f"Intent validation: {'Passed' if is_valid else 'Failed'}")
        
        # Enhance intent with context
        context = {
            "user_expertise": "beginner",
            "previous_edits": ["enhanced_colors", "adjusted_brightness"],
            "preferred_style": "photorealistic"
        }
        
        enhanced_intent = parser.enhance_intent_with_context(intent, context)
        print(f"Enhanced intent confidence: {enhanced_intent.confidence}")
        print(f"Enhanced clarifying questions: {enhanced_intent.clarifying_questions}")
        
    except IntentError as e:
        print(f"Intent parsing failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    # Example with low confidence
    print("\nExample with low confidence prompt:")
    ambiguous_prompt = "make it better"
    ambiguous_scene = {
        "entities": [
            {"id": "sky_0", "label": "sky", "confidence": 0.95},
            {"id": "mountain_1", "label": "mountain", "confidence": 0.87}
        ],
        "spatial_layout": "sky at top, mountains below",
        "image_description": "landscape scene"
    }
    
    try:
        ambiguous_intent = parser.forward(ambiguous_prompt, ambiguous_scene)
        print(f"Ambiguous intent:")
        print(f"  Naive prompt: {ambiguous_intent.naive_prompt}")
        print(f"  Confidence: {ambiguous_intent.confidence}")
        print(f"  Clarifying questions: {ambiguous_intent.clarifying_questions}")
        
    except IntentError as e:
        print(f"Ambiguous intent parsing failed: {e}")
    
    print("Intent parser example completed")
```

### Advanced Intent Parser Implementation
Enhanced implementation with multiple parsing strategies and context awareness:

```python
import dspy
import json
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
import logging
import asyncio
from dataclasses import dataclass
from datetime import datetime
import re

class AdvancedIntentError(Exception):
    """Custom exception for advanced intent parsing errors."""
    pass

class AdvancedEditType(Enum):
    """Enhanced enumeration for edit types."""
    COLOR = "color"
    STYLE = "style"
    ADD = "add"
    REMOVE = "remove"
    TRANSFORM = "transform"
    ENHANCE = "enhance"
    RESTORE = "restore"
    COMPOSE = "compose"

class AdvancedIntent(BaseModel):
    """
    Enhanced structured intent with additional metadata.
    """
    naive_prompt: str = Field(..., description="Original user prompt")
    target_entities: List[str] = Field([], description="List of entity IDs to edit")
    edit_type: AdvancedEditType = Field(AdvancedEditType.TRANSFORM, description="Type of edit to perform")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in intent parsing")
    clarifying_questions: List[str] = Field([], description="Questions to clarify ambiguous requests")
    scene_analysis: Optional[Dict[str, Any]] = Field(None, description="Scene analysis data")
    context: Optional[Dict[str, Any]] = Field({}, description="Additional context information")
    metadata: Optional[Dict[str, Any]] = Field({}, description="Metadata about the parsing process")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp of parsing")
    parsing_strategy: str = Field("default", description="Strategy used for parsing")
    alternative_intents: List[Dict[str, Any]] = Field([], description="Alternative interpretations of intent")

class AdvancedSceneAnalysis(BaseModel):
    """
    Enhanced scene analysis with detailed metadata.
    """
    entities: List[Dict[str, Any]] = Field([], description="Detected entities in the scene")
    spatial_layout: str = Field("", description="Description of spatial relationships")
    image_description: str = Field("", description="Overall image description")
    metadata: Optional[Dict[str, Any]] = Field({}, description="Metadata about the analysis")
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Quality score of analysis")
    processing_time: Optional[float] = Field(None, ge=0.0, description="Time taken for analysis")

class MultiStrategyIntentParser(dspy.Module):
    """
    Advanced intent parser with multiple parsing strategies.
    """
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Initialize multiple parsing strategies
        self.strategies = {
            "default": dspy.ChainOfThought(self._get_parse_intent_signature()),
            "cot": dspy.ChainOfThought(self._get_parse_intent_signature()),
            "react": dspy.ReAct(self._get_parse_intent_signature()),
            "programmed": dspy.ProgramOfThought(self._get_parse_intent_signature()),
            "multi_shot": dspy.MultiChainComparison(self._get_parse_intent_signature())
        }
    
    def _get_parse_intent_signature(self) -> dspy.Signature:
        """
        Get the parse intent signature for DSpy.
        
        Returns:
            ParseIntentSignature class
        """
        class ParseIntentSignature(dspy.Signature):
            """
            DSpy signature for parsing user intent from naive prompts and scene analysis.
            """
            naive_prompt = dspy.InputField(
                desc="User's conversational edit request"
            )
            scene_analysis = dspy.InputField(
                desc="JSON of detected entities and layout"
            )
            context = dspy.InputField(
                desc="Additional context information (optional)"
            )
            
            target_entities = dspy.OutputField(
                desc="Comma-separated list of entity IDs to edit"
            )
            edit_type = dspy.OutputField(
                desc="One of: color, style, add, remove, transform, enhance, restore, compose"
            )
            confidence = dspy.OutputField(
                desc="Float 0-1 indicating clarity of intent"
            )
            clarifying_questions = dspy.OutputField(
                desc="JSON array of questions if confidence <0.7"
            )
            alternative_intents = dspy.OutputField(
                desc="JSON array of alternative interpretations with confidence scores"
            )
        
        return ParseIntentSignature
    
    def forward(self, 
                naive_prompt: str, 
                scene: Dict[str, Any],
                context: Optional[Dict[str, Any]] = None,
                strategy: str = "default",
                use_multiple_strategies: bool = False) -> AdvancedIntent:
        """
        Takes a naive prompt and scene analysis and returns structured intent using multiple strategies.
        
        Args:
            naive_prompt: User's conversational edit request
            scene: Scene analysis data
            context: Additional context information
            strategy: Parsing strategy to use
            use_multiple_strategies: Whether to use multiple strategies and compare results
            
        Returns:
            Enhanced structured intent with confidence and alternatives
        """
        # Validate inputs
        if not naive_prompt or not naive_prompt.strip():
            raise AdvancedIntentError("Naive prompt cannot be empty")
        
        if not scene:
            raise AdvancedIntentError("Scene analysis cannot be empty")
        
        # Ensure context is a dictionary
        if context is None:
            context = {}
        
        self.logger.info(f"Parsing intent using strategy: {strategy}")
        
        try:
            if use_multiple_strategies:
                # Use multiple strategies and compare results
                intent = self._parse_with_multiple_strategies(naive_prompt, scene, context)
            else:
                # Use single strategy
                intent = self._parse_with_single_strategy(naive_prompt, scene, context, strategy)
            
            self.logger.info(f"Parsed intent: {intent.edit_type.value} with confidence {intent.confidence}")
            return intent
            
        except Exception as e:
            self.logger.error(f"Failed to parse intent: {str(e)}")
            raise AdvancedIntentError(f"Intent parsing failed: {str(e)}")
    
    def _parse_with_single_strategy(self, 
                                   naive_prompt: str, 
                                   scene: Dict[str, Any],
                                   context: Dict[str, Any],
                                   strategy: str) -> AdvancedIntent:
        """
        Parse intent using a single strategy.
        
        Args:
            naive_prompt: User's conversational edit request
            scene: Scene analysis data
            context: Additional context information
            strategy: Parsing strategy to use
            
        Returns:
            Enhanced structured intent
        """
        # Get the appropriate parser
        if strategy not in self.strategies:
            self.logger.warning(f"Unknown strategy '{strategy}', using default")
            strategy = "default"
        
        parser = self.strategies[strategy]
        
        # Process naive prompt and scene analysis to extract structured intent
        result = parser(
            naive_prompt=naive_prompt,
            scene_analysis=json.dumps(scene, default=str),
            context=json.dumps(context, default=str) if context else "{}"
        )
        
        # Parse results
        return self._parse_dspy_result(result, naive_prompt, scene, context, strategy)
    
    def _parse_with_multiple_strategies(self, 
                                      naive_prompt: str, 
                                      scene: Dict[str, Any],
                                      context: Dict[str, Any]) -> AdvancedIntent:
        """
        Parse intent using multiple strategies and compare results.
        
        Args:
            naive_prompt: User's conversational edit request
            scene: Scene analysis data
            context: Additional context information
            
        Returns:
            Enhanced structured intent with best confidence
        """
        # Run all strategies
        results = {}
        for strategy_name, parser in self.strategies.items():
            try:
                result = parser(
                    naive_prompt=naive_prompt,
                    scene_analysis=json.dumps(scene, default=str),
                    context=json.dumps(context, default=str) if context else "{}"
                )
                results[strategy_name] = result
            except Exception as e:
                self.logger.warning(f"Strategy {strategy_name} failed: {str(e)}")
                continue
        
        # Compare results and select best
        if not results:
            raise AdvancedIntentError("All parsing strategies failed")
        
        # For simplicity, we'll use the first successful result
        # In a real implementation, you'd compare confidences and aggregate results
        best_strategy = next(iter(results.keys()))
        best_result = results[best_strategy]
        
        return self._parse_dspy_result(best_result, naive_prompt, scene, context, best_strategy, results)
    
    def _parse_dspy_result(self, 
                          result: Any, 
                          naive_prompt: str, 
                          scene: Dict[str, Any],
                          context: Dict[str, Any],
                          strategy: str,
                          all_results: Optional[Dict[str, Any]] = None) -> AdvancedIntent:
        """
        Parse DSpy result into AdvancedIntent.
        
        Args:
            result: DSpy result object
            naive_prompt: User's conversational edit request
            scene: Scene analysis data
            context: Additional context information
            strategy: Strategy used for parsing
            all_results: All results from multiple strategies (optional)
            
        Returns:
            AdvancedIntent object
        """
        # Parse target entities
        target_entities = []
        if hasattr(result, 'target_entities') and result.target_entities:
            target_entities = result.target_entities.split(", ") if isinstance(result.target_entities, str) else []
        
        # Parse edit type
        edit_type = AdvancedEditType.TRANSFORM
        if hasattr(result, 'edit_type') and result.edit_type:
            try:
                edit_type = AdvancedEditType(result.edit_type)
            except ValueError:
                # Default to transform if unknown type
                edit_type = AdvancedEditType.TRANSFORM
        
        # Parse confidence
        confidence = 0.0
        if hasattr(result, 'confidence') and result.confidence:
            try:
                confidence = float(result.confidence)
                confidence = max(0.0, min(1.0, confidence))  # Clamp to 0-1 range
            except (ValueError, TypeError):
                confidence = 0.0
        
        # Parse clarifying questions
        clarifying_questions = []
        if hasattr(result, 'clarifying_questions') and result.clarifying_questions:
            try:
                clarifying_questions = json.loads(result.clarifying_questions) if isinstance(result.clarifying_questions, str) else []
            except json.JSONDecodeError:
                clarifying_questions = []
        
        # Parse alternative intents
        alternative_intents = []
        if hasattr(result, 'alternative_intents') and result.alternative_intents:
            try:
                alternative_intents = json.loads(result.alternative_intents) if isinstance(result.alternative_intents, str) else []
            except json.JSONDecodeError:
                alternative_intents = []
        
        # Generate clarifying questions when confidence is low
        if confidence < 0.7 and not clarifying_questions:
            clarifying_questions = self._generate_clarifying_questions(naive_prompt, scene)
        
        # Create metadata about the parsing process
        metadata = {
            "parsing_timestamp": datetime.now().isoformat(),
            "strategy_used": strategy,
            "total_strategies_tried": len(all_results) if all_results else 1,
            "result_consistency": self._assess_result_consistency(all_results) if all_results else "N/A"
        }
        
        # Create enhanced structured intent
        intent = AdvancedIntent(
            naive_prompt=naive_prompt,
            target_entities=target_entities,
            edit_type=edit_type,
            confidence=confidence,
            clarifying_questions=clarifying_questions,
            scene_analysis=scene,
            context=context,
            metadata=metadata,
            parsing_strategy=strategy,
            alternative_intents=alternative_intents
        )
        
        return intent
    
    def _generate_clarifying_questions(self, naive_prompt: str, scene: Dict[str, Any]) -> List[str]:
        """
        Generate clarifying questions when confidence is low.
        
        Args:
            naive_prompt: User's conversational edit request
            scene: Scene analysis data
            
        Returns:
            List of clarifying questions
        """
        # Enhanced question generation with pattern matching
        questions = []
        
        # Check for common ambiguous terms using regex
        ambiguous_patterns = [
            r"\bbetter\b", r"\bimprove\b", r"\benhance\b", r"\bgood\b", r"\bnice\b",
            r"\bmore\b", r"\bless\b", r"\bdifferent\b", r"\bchange\b"
        ]
        
        if any(re.search(pattern, naive_prompt.lower()) for pattern in ambiguous_patterns):
            questions.append("What specific aspect would you like to improve?")
        
        # Check for vague references to scene elements
        entities = scene.get("entities", [])
        if not entities:
            questions.append("Could you specify which part of the image you want to modify?")
        else:
            # Extract entity labels
            entity_labels = [entity.get("label", "").lower() for entity in entities]
            
            # Check if prompt references entities not clearly identified
            prompt_words = re.findall(r'\b\w+\b', naive_prompt.lower())
            unmatched_entities = [
                word for word in prompt_words 
                if len(word) > 2 and not any(word in label for label in entity_labels)
            ]
            
            if unmatched_entities:
                questions.append(f"Did you mean to refer to any of these entities: {', '.join(entity_labels)}?")
        
        # Check for missing spatial references
        spatial_keywords = ["above", "below", "left", "right", "top", "bottom", "center"]
        if not any(keyword in naive_prompt.lower() for keyword in spatial_keywords):
            questions.append("Where in the image should this change be applied?")
        
        # General clarifying questions
        questions.extend([
            "Can you be more specific about what changes you want?",
            "Are there any elements you want to preserve unchanged?",
            "What is the desired outcome of this edit?"
        ])
        
        return questions[:5]  # Limit to 5 questions
    
    def _assess_result_consistency(self, results: Dict[str, Any]) -> str:
        """
        Assess consistency between different parsing strategies.
        
        Args:
            results: Dictionary of results from different strategies
            
        Returns:
            Consistency assessment string
        """
        if len(results) < 2:
            return "Single strategy used"
        
        # Extract key attributes from each result
        attributes = {}
        for strategy, result in results.items():
            attrs = {}
            if hasattr(result, 'target_entities'):
                attrs['target_entities'] = result.target_entities
            if hasattr(result, 'edit_type'):
                attrs['edit_type'] = result.edit_type
            if hasattr(result, 'confidence'):
                try:
                    attrs['confidence'] = float(result.confidence)
                except (ValueError, TypeError):
                    attrs['confidence'] = 0.0
            attributes[strategy] = attrs
        
        # Simple consistency check
        # In a real implementation, you'd have more sophisticated comparison
        return f"Compared {len(attributes)} strategies"
    
    def validate_intent(self, intent: AdvancedIntent) -> bool:
        """
        Validate the parsed intent.
        
        Args:
            intent: Parsed intent to validate
            
        Returns:
            Boolean indicating if intent is valid
        """
        # Validate intent structure
        if not intent.naive_prompt or not intent.naive_prompt.strip():
            self.logger.error("Intent validation failed: Empty naive prompt")
            return False
        
        # Validate edit type
        if not isinstance(intent.edit_type, AdvancedEditType):
            self.logger.error("Intent validation failed: Invalid edit type")
            return False
        
        # Validate confidence
        if not (0.0 <= intent.confidence <= 1.0):
            self.logger.error("Intent validation failed: Invalid confidence value")
            return False
        
        # Validate target entities if present
        if intent.target_entities and not isinstance(intent.target_entities, list):
            self.logger.error("Intent validation failed: Invalid target entities format")
            return False
        
        # Validate clarifying questions if present
        if intent.clarifying_questions and not isinstance(intent.clarifying_questions, list):
            self.logger.error("Intent validation failed: Invalid clarifying questions format")
            return False
        
        # Validate alternative intents if present
        if intent.alternative_intents and not isinstance(intent.alternative_intents, list):
            self.logger.error("Intent validation failed: Invalid alternative intents format")
            return False
        
        # Validate context if present
        if intent.context and not isinstance(intent.context, dict):
            self.logger.error("Intent validation failed: Invalid context format")
            return False
        
        # Validate metadata if present
        if intent.metadata and not isinstance(intent.metadata, dict):
            self.logger.error("Intent validation failed: Invalid metadata format")
            return False
        
        self.logger.info("Intent validation passed")
        return True
    
    def enhance_intent_with_context(self, intent: AdvancedIntent, context: Dict[str, Any]) -> AdvancedIntent:
        """
        Enhance intent with additional context information.
        
        Args:
            intent: Parsed intent to enhance
            context: Additional context information
            
        Returns:
            Enhanced intent
        """
        # Update context
        intent.context.update(context)
        
        # Adjust confidence based on context
        if context.get("user_expertise") == "expert":
            intent.confidence = min(1.0, intent.confidence * 1.2)  # Boost confidence for experts
        elif context.get("user_expertise") == "beginner":
            # Reduce confidence slightly for beginners as they might not be precise
            intent.confidence = max(0.0, intent.confidence * 0.9)
        
        # Add context-based clarifying questions
        if context.get("previous_edits"):
            intent.clarifying_questions.append("How should this edit relate to your previous edits?")
        
        # Add metadata about context enhancement
        if "context_enhancements" not in intent.metadata:
            intent.metadata["context_enhancements"] = []
        intent.metadata["context_enhancements"].append({
            "enhanced_at": datetime.now().isoformat(),
            "enhanced_by": "context_enhancement",
            "context_keys": list(context.keys())
        })
        
        self.logger.info("Intent enhanced with context")
        return intent
    
    def get_intent_confidence_breakdown(self, intent: AdvancedIntent) -> Dict[str, Any]:
        """
        Get a detailed breakdown of intent confidence factors.
        
        Args:
            intent: Parsed intent
            
        Returns:
            Dictionary with confidence breakdown
        """
        breakdown = {
            "overall_confidence": intent.confidence,
            "factors": {},
            "recommendations": []
        }
        
        # Analyze confidence factors
        factors = {}
        
        # Factor 1: Prompt specificity
        prompt_words = len(intent.naive_prompt.split())
        factors["prompt_specificity"] = {
            "score": min(1.0, prompt_words / 10.0),  # Normalize to 0-1
            "weight": 0.3,
            "description": f"Prompt has {prompt_words} words"
        }
        
        # Factor 2: Entity identification
        entities_identified = len(intent.target_entities)
        factors["entity_identification"] = {
            "score": min(1.0, entities_identified / 3.0),  # Normalize to 0-1
            "weight": 0.4,
            "description": f"Identified {entities_identified} entities"
        }
        
        # Factor 3: Context utilization
        context_provided = len(intent.context) if intent.context else 0
        factors["context_utilization"] = {
            "score": min(1.0, context_provided / 5.0),  # Normalize to 0-1
            "weight": 0.2,
            "description": f"Context provided with {context_provided} keys"
        }
        
        # Factor 4: Clarifying questions
        questions_needed = len(intent.clarifying_questions)
        factors["clarification_needed"] = {
            "score": 1.0 - min(1.0, questions_needed / 3.0),  # Inverse relationship
            "weight": 0.1,
            "description": f"{questions_needed} clarifying questions needed"
        }
        
        breakdown["factors"] = factors
        
        # Calculate weighted confidence
        weighted_confidence = 0.0
        total_weight = 0.0
        for factor_name, factor_data in factors.items():
            weighted_confidence += factor_data["score"] * factor_data["weight"]
            total_weight += factor_data["weight"]
        
        if total_weight > 0:
            calculated_confidence = weighted_confidence / total_weight
            breakdown["calculated_confidence"] = calculated_confidence
        
        # Add recommendations
        if intent.confidence < 0.5:
            breakdown["recommendations"].append("Consider providing more specific details about the desired edit")
        elif intent.confidence < 0.7:
            breakdown["recommendations"].append("Clarify which elements should be modified")
        elif intent.confidence < 0.9:
            breakdown["recommendations"].append("Specify the desired outcome more precisely")
        
        if questions_needed > 0:
            breakdown["recommendations"].extend(intent.clarifying_questions)
        
        return breakdown

# Example usage
if __name__ == "__main__":
    # Initialize advanced intent parser
    parser = MultiStrategyIntentParser()
    
    print("Advanced Intent Parser initialized")
    
    # Example naive prompt and scene analysis
    naive_prompt = "make the sky more dramatic with storm clouds"
    scene_analysis = {
        "entities": [
            {"id": "sky_0", "label": "sky", "confidence": 0.95, "bbox": [0, 0, 1920, 768]},
            {"id": "mountain_1", "label": "mountain", "confidence": 0.87, "bbox": [20, 768, 1900, 1080]},
            {"id": "tree_2", "label": "tree", "confidence": 0.92, "bbox": [100, 800, 200, 1000]}
        ],
        "spatial_layout": "sky occupies top portion, mountains at bottom, trees in foreground",
        "image_description": "landscape with sky, mountains, and trees",
        "metadata": {
            "analysis_quality": 0.92,
            "processing_time": 2.3
        }
    }
    
    # Parse intent with default strategy
    try:
        intent = parser.forward(naive_prompt, scene_analysis)
        print(f"Parsed intent:")
        print(f"  Naive prompt: {intent.naive_prompt}")
        print(f"  Target entities: {intent.target_entities}")
        print(f"  Edit type: {intent.edit_type.value}")
        print(f"  Confidence: {intent.confidence}")
        print(f"  Clarifying questions: {intent.clarifying_questions}")
        print(f"  Strategy used: {intent.parsing_strategy}")
        print(f"  Timestamp: {intent.timestamp}")
        
        # Validate intent
        is_valid = parser.validate_intent(intent)
        print(f"\nIntent validation: {'Passed' if is_valid else 'Failed'}")
        
        # Get confidence breakdown
        confidence_breakdown = parser.get_intent_confidence_breakdown(intent)
        print(f"\nConfidence breakdown:")
        print(f"  Overall: {confidence_breakdown['overall_confidence']}")
        print(f"  Calculated: {confidence_breakdown.get('calculated_confidence', 'N/A')}")
        print(f"  Factors:")
        for factor_name, factor_data in confidence_breakdown['factors'].items():
            print(f"    {factor_name}: {factor_data['score']:.2f} (weight: {factor_data['weight']}) - {factor_data['description']}")
        print(f"  Recommendations: {confidence_breakdown['recommendations']}")
        
        # Enhance intent with context
        context = {
            "user_expertise": "intermediate",
            "previous_edits": ["enhanced_colors", "adjusted_brightness"],
            "preferred_style": "photorealistic",
            "user_preferences": {
                "avoid_over_saturation": True,
                "preserve_natural_look": True
            }
        }
        
        enhanced_intent = parser.enhance_intent_with_context(intent, context)
        print(f"\nEnhanced intent:")
        print(f"  Confidence: {enhanced_intent.confidence}")
        print(f"  Clarifying questions: {enhanced_intent.clarifying_questions}")
        print(f"  Context keys: {list(enhanced_intent.context.keys())}")
        
    except AdvancedIntentError as e:
        print(f"Intent parsing failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    # Example with low confidence
    print("\nExample with low confidence prompt:")
    ambiguous_prompt = "make it better"
    ambiguous_scene = {
        "entities": [
            {"id": "sky_0", "label": "sky", "confidence": 0.95},
            {"id": "mountain_1", "label": "mountain", "confidence": 0.87}
        ],
        "spatial_layout": "sky at top, mountains below",
        "image_description": "landscape scene"
    }
    
    try:
        ambiguous_intent = parser.forward(ambiguous_prompt, ambiguous_scene)
        print(f"Ambiguous intent:")
        print(f"  Naive prompt: {ambiguous_intent.naive_prompt}")
        print(f"  Confidence: {ambiguous_intent.confidence}")
        print(f"  Clarifying questions: {ambiguous_intent.clarifying_questions}")
        print(f"  Alternative intents: {ambiguous_intent.alternative_intents}")
        
    except AdvancedIntentError as e:
        print(f"Ambiguous intent parsing failed: {e}")
    
    # Example with multiple strategies
    print("\nExample with multiple strategies:")
    try:
        multi_strategy_intent = parser.forward(
            naive_prompt, 
            scene_analysis, 
            strategy="default",
            use_multiple_strategies=True
        )
        print(f"Multi-strategy intent:")
        print(f"  Strategy used: {multi_strategy_intent.parsing_strategy}")
        print(f"  Confidence: {multi_strategy_intent.confidence}")
        print(f"  Metadata: {multi_strategy_intent.metadata}")
        
    except AdvancedIntentError as e:
        print(f"Multi-strategy parsing failed: {e}")
    
    print("Advanced intent parser example completed")
```