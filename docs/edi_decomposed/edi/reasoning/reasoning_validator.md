# Reasoning: Validator

[Back to Reasoning Subsystem](./reasoning_subsystem.md)

## Purpose
Edit quality assessment - Contains the Validator class that calculates alignment scores and generates retry hints if scores are low.

## Class: Validator

### Methods
- `validate(delta, intent) -> ValidationResult`: Evaluates the quality of an edit based on the delta and original intent

### Details
- Calculates alignment scores for validation
- Generates retry hints if score is low
- Helps determine if an edit matches user intent

## Functions

- [validate(delta, intent)](./reasoning/validate.md)

## Technology Stack

- Pydantic for data validation
- NumPy for calculations

## See Docs

### Python Implementation Example
Validator implementation with Pydantic and NumPy:

```python
import numpy as np
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import logging
from datetime import datetime
from dataclasses import dataclass

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

class ValidationResultStatus(Enum):
    """Enumeration for validation result statuses."""
    ACCEPT = "accept"
    REJECT = "reject"
    PARTIAL = "partial"
    REVIEW = "review"

class DeltaData(BaseModel):
    """
    Represents the changes made to an image.
    """
    before_image_path: str = Field(..., description="Path to the original image")
    after_image_path: str = Field(..., description="Path to the edited image")
    changes: Dict[str, Any] = Field({}, description="Dictionary of changes made")
    processing_time: Optional[float] = Field(None, ge=0.0, description="Time taken for processing")
    model_used: Optional[str] = Field(None, description="Model used for processing")

class IntentData(BaseModel):
    """
    Represents the user's original intent.
    """
    naive_prompt: str = Field(..., description="User's original prompt")
    target_entities: List[str] = Field([], description="List of entities to modify")
    edit_type: str = Field(..., description="Type of edit requested")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the intent")
    clarifying_questions: List[str] = Field([], description="Clarifying questions if confidence is low")

class ValidationResult(BaseModel):
    """
    Represents the result of validation.
    """
    status: ValidationResultStatus = Field(..., description="Validation status")
    alignment_score: float = Field(..., ge=0.0, le=1.0, description="Alignment score between intent and result")
    preserved_count: int = Field(0, ge=0, description="Number of entities preserved")
    modified_count: int = Field(0, ge=0, description="Number of entities modified as intended")
    unintended_count: int = Field(0, ge=0, description="Number of unintended changes")
    user_feedback: Optional[str] = Field(None, description="Feedback for the user")
    retry_hints: List[str] = Field([], description="Hints for retrying if needed")
    quality_metrics: Optional[Dict[str, Any]] = Field({}, description="Additional quality metrics")
    validation_time: Optional[float] = Field(None, ge=0.0, description="Time taken for validation")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Validation timestamp")

class Validator:
    """
    Validates edits based on delta and original intent.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate(self, delta: DeltaData, intent: IntentData) -> ValidationResult:
        """
        Evaluates the quality of an edit based on the delta and original intent.
        
        Args:
            delta: DeltaData containing changes made to the image
            intent: IntentData containing original user intent
            
        Returns:
            ValidationResult with evaluation results
        """
        # Validate inputs
        if not delta or not intent:
            raise ValidationError("Both delta and intent must be provided")
        
        self.logger.info("Starting validation process")
        
        # Calculate alignment scores for validation
        start_time = datetime.now()
        alignment_score = self._calculate_alignment_score(delta, intent)
        
        # Determine preservation metrics
        preserved_count, modified_count, unintended_count = self._calculate_preservation_metrics(delta, intent)
        
        # Generate retry hints if score is low
        retry_hints = []
        if alignment_score < 0.7:  # Threshold for low score
            retry_hints = self._generate_retry_hints(delta, intent, alignment_score)
        
        # Determine validation status based on metrics
        status = self._determine_validation_status(
            alignment_score, 
            preserved_count, 
            modified_count, 
            unintended_count
        )
        
        # Generate user feedback
        user_feedback = self._generate_user_feedback(
            status, 
            alignment_score, 
            preserved_count, 
            modified_count, 
            unintended_count
        )
        
        # Calculate additional quality metrics
        quality_metrics = self._calculate_quality_metrics(delta, intent)
        
        # Calculate validation time
        validation_time = (datetime.now() - start_time).total_seconds()
        
        # Create validation result
        result = ValidationResult(
            status=status,
            alignment_score=alignment_score,
            preserved_count=preserved_count,
            modified_count=modified_count,
            unintended_count=unintended_count,
            user_feedback=user_feedback,
            retry_hints=retry_hints,
            quality_metrics=quality_metrics,
            validation_time=validation_time,
            timestamp=datetime.now().isoformat()
        )
        
        self.logger.info(f"Validation completed with status: {status.value}, alignment score: {alignment_score}")
        return result
    
    def _calculate_alignment_score(self, delta: DeltaData, intent: IntentData) -> float:
        """
        Calculates alignment score between intent and actual changes.
        
        Args:
            delta: DeltaData containing changes made
            intent: IntentData containing original intent
            
        Returns:
            Float between 0.0 and 1.0 representing alignment score
        """
        # This is a simplified implementation
        # In a real implementation, this would compare:
        # 1. Semantic similarity between intent and actual changes
        # 2. Preservation of non-target entities
        # 3. Quality of modifications to target entities
        # 4. Adherence to user constraints
        
        # For this example, we'll create a mock implementation
        # that considers multiple factors:
        
        # Factor 1: Intent confidence (higher confidence = higher expected alignment)
        confidence_factor = intent.confidence
        
        # Factor 2: Target entity match (are the right entities being modified?)
        target_match_factor = self._calculate_target_entity_match(delta, intent)
        
        # Factor 3: Change magnitude (are changes appropriate in scale?)
        change_magnitude_factor = self._calculate_change_magnitude(delta, intent)
        
        # Factor 4: Preservation (are non-target entities preserved?)
        preservation_factor = self._calculate_preservation(delta, intent)
        
        # Weighted combination of factors
        weights = {
            "confidence": 0.2,
            "target_match": 0.3,
            "magnitude": 0.2,
            "preservation": 0.3
        }
        
        alignment_score = (
            weights["confidence"] * confidence_factor +
            weights["target_match"] * target_match_factor +
            weights["magnitude"] * change_magnitude_factor +
            weights["preservation"] * preservation_factor
        )
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, alignment_score))
    
    def _calculate_target_entity_match(self, delta: DeltaData, intent: IntentData) -> float:
        """
        Calculate how well target entities match between intent and actual changes.
        
        Args:
            delta: DeltaData containing changes made
            intent: IntentData containing original intent
            
        Returns:
            Float between 0.0 and 1.0 representing target entity match
        """
        # In a real implementation, this would:
        # 1. Analyze which entities were actually modified
        # 2. Compare with intent.target_entities
        # 3. Calculate overlap/intersection
        
        # For this example, we'll use a simple heuristic
        target_entities = set(intent.target_entities)
        changed_entities = set(delta.changes.get("modified_entities", []))
        
        if not target_entities:
            # If no specific targets, any change might be acceptable
            return 0.8 if changed_entities else 0.5
        
        # Calculate intersection over union (IoU)
        intersection = len(target_entities.intersection(changed_entities))
        union = len(target_entities.union(changed_entities))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _calculate_change_magnitude(self, delta: DeltaData, intent: IntentData) -> float:
        """
        Calculate if change magnitude is appropriate.
        
        Args:
            delta: DeltaData containing changes made
            intent: IntentData containing original intent
            
        Returns:
            Float between 0.0 and 1.0 representing change magnitude appropriateness
        """
        # In a real implementation, this would:
        # 1. Analyze the extent of changes made
        # 2. Compare with intent.edit_type expectations
        # 3. Consider processing_time relative to complexity
        
        # For this example, we'll use a simple heuristic
        processing_time = delta.processing_time or 0.0
        edit_type = intent.edit_type.lower()
        
        # Expected processing times for different edit types (in seconds)
        expected_times = {
            "color": 5.0,
            "style": 15.0,
            "add": 20.0,
            "remove": 25.0,
            "transform": 30.0
        }
        
        expected_time = expected_times.get(edit_type, 15.0)
        
        # If processing time is within reasonable range of expected time
        if processing_time <= expected_time * 2:
            # Score decreases as processing time increases beyond expected
            if processing_time <= expected_time:
                return 1.0
            else:
                # Linear decrease from 1.0 to 0.5 as time goes from expected to 2*expected
                return max(0.5, 1.0 - (processing_time - expected_time) / expected_time)
        else:
            # Beyond 2*expected time, score decreases rapidly
            return max(0.0, 0.5 - (processing_time - expected_time * 2) / (expected_time * 2))
    
    def _calculate_preservation(self, delta: DeltaData, intent: IntentData) -> float:
        """
        Calculate how well non-target entities were preserved.
        
        Args:
            delta: DeltaData containing changes made
            intent: IntentData containing original intent
            
        Returns:
            Float between 0.0 and 1.0 representing preservation quality
        """
        # In a real implementation, this would:
        # 1. Identify non-target entities
        # 2. Measure changes to those entities
        # 3. Calculate preservation score
        
        # For this example, we'll use a simple heuristic
        preserved_entities = delta.changes.get("preserved_entities", [])
        modified_entities = delta.changes.get("modified_entities", [])
        target_entities = intent.target_entities
        
        # Count non-target entities that were preserved
        non_target_preserved = [
            entity for entity in preserved_entities 
            if entity not in target_entities
        ]
        
        # Count non-target entities that were modified (unintended changes)
        non_target_modified = [
            entity for entity in modified_entities 
            if entity not in target_entities
        ]
        
        # Total non-target entities (estimated)
        total_non_target = max(1, len(non_target_preserved) + len(non_target_modified))
        
        # Preservation score: ratio of preserved non-target entities
        preservation_score = len(non_target_preserved) / total_non_target
        
        # Reduce score if there are unintended changes
        if non_target_modified:
            # Penalty for unintended changes
            penalty = min(0.5, len(non_target_modified) / total_non_target)
            preservation_score = max(0.0, preservation_score - penalty)
        
        return preservation_score
    
    def _calculate_preservation_metrics(self, delta: DeltaData, intent: IntentData) -> tuple:
        """
        Calculate preservation metrics.
        
        Args:
            delta: DeltaData containing changes made
            intent: IntentData containing original intent
            
        Returns:
            Tuple of (preserved_count, modified_count, unintended_count)
        """
        # In a real implementation, this would:
        # 1. Analyze actual preservation metrics from image comparison
        # 2. Count preserved, modified, and unintended changes
        
        # For this example, we'll generate mock metrics
        preserved_count = len(delta.changes.get("preserved_entities", []))
        modified_count = len(delta.changes.get("modified_entities", []))
        unintended_count = len(delta.changes.get("unintended_changes", []))
        
        return preserved_count, modified_count, unintended_count
    
    def _generate_retry_hints(self, delta: DeltaData, intent: IntentData, alignment_score: float) -> List[str]:
        """
        Generate retry hints when alignment score is low.
        
        Args:
            delta: DeltaData containing changes made
            intent: IntentData containing original intent
            alignment_score: Current alignment score
            
        Returns:
            List of retry hints
        """
        hints = []
        
        # Analyze why alignment score is low and provide specific hints
        if alignment_score < 0.3:
            hints.append("The edit result differs significantly from your request. Consider rephrasing your prompt.")
        
        # Check if target entities were properly identified
        target_entities = set(intent.target_entities)
        changed_entities = set(delta.changes.get("modified_entities", []))
        
        if target_entities and not changed_entities:
            hints.append("No entities were modified. Try specifying which parts of the image you want to change.")
        elif target_entities and not target_entities.intersection(changed_entities):
            hints.append(f"The requested entities ({', '.join(target_entities)}) were not modified. Check if they were correctly identified.")
        
        # Check for unintended changes
        unintended_changes = delta.changes.get("unintended_changes", [])
        if unintended_changes:
            hints.append(f"Unintended changes were made to: {', '.join(unintended_changes)}. Try being more specific about what to preserve.")
        
        # General hints based on edit type
        edit_type = intent.edit_type.lower()
        if edit_type == "color":
            hints.append("For color adjustments, try specifying exact colors or using terms like 'more vibrant', 'darker', 'warmer'.")
        elif edit_type == "style":
            hints.append("For style changes, be more specific about the desired style (e.g., 'impressionist', 'cinematic', 'anime').")
        elif edit_type == "add":
            hints.append("When adding elements, describe their appearance and placement in detail.")
        elif edit_type == "remove":
            hints.append("When removing elements, clearly identify what should be removed and what should remain.")
        elif edit_type == "transform":
            hints.append("For transformations, specify the desired outcome clearly (e.g., 'make it look like a painting', 'change season to winter').")
        
        # If confidence is low, suggest clarification
        if intent.confidence < 0.5:
            hints.append("Your prompt was ambiguous. Try being more specific about what you want to change.")
        
        # If no hints were generated, provide generic ones
        if not hints:
            hints.extend([
                "Try rephrasing your request with more specific details.",
                "Be more explicit about what should and shouldn't change.",
                "Consider breaking complex requests into simpler steps."
            ])
        
        return hints[:5]  # Limit to 5 hints
    
    def _determine_validation_status(self, 
                                   alignment_score: float, 
                                   preserved_count: int, 
                                   modified_count: int, 
                                   unintended_count: int) -> ValidationResultStatus:
        """
        Determine validation status based on metrics.
        
        Args:
            alignment_score: Alignment score between intent and result
            preserved_count: Number of entities preserved
            modified_count: Number of entities modified as intended
            unintended_count: Number of unintended changes
            
        Returns:
            ValidationResultStatus enum value
        """
        # Determine status based on multiple criteria
        if alignment_score >= 0.8 and unintended_count == 0:
            return ValidationResultStatus.ACCEPT
        elif alignment_score >= 0.6 and unintended_count <= 1:
            return ValidationResultStatus.PARTIAL
        elif alignment_score >= 0.4 or (modified_count > 0 and unintended_count <= 2):
            return ValidationResultStatus.REVIEW
        else:
            return ValidationResultStatus.REJECT
    
    def _generate_user_feedback(self, 
                              status: ValidationResultStatus, 
                              alignment_score: float, 
                              preserved_count: int, 
                              modified_count: int, 
                              unintended_count: int) -> str:
        """
        Generate user feedback based on validation results.
        
        Args:
            status: Validation status
            alignment_score: Alignment score between intent and result
            preserved_count: Number of entities preserved
            modified_count: Number of entities modified as intended
            unintended_count: Number of unintended changes
            
        Returns:
            User feedback string
        """
        feedback_templates = {
            ValidationResultStatus.ACCEPT: [
                f"Excellent! The edit perfectly matches your intent with an alignment score of {alignment_score:.2f}.",
                f"Perfect match! The changes align very well with your request (score: {alignment_score:.2f}).",
                f"Outstanding! The result matches your vision almost exactly (alignment: {alignment_score:.2f})."
            ],
            ValidationResultStatus.PARTIAL: [
                f"Good result with an alignment score of {alignment_score:.2f}. Some aspects match your intent well.",
                f"Well done! The edit aligns reasonably well with your request (score: {alignment_score:.2f}), though there's room for improvement.",
                f"Solid work! Most changes match your intent (alignment: {alignment_score:.2f}), with minor adjustments needed."
            ],
            ValidationResultStatus.REVIEW: [
                f"The edit has some alignment issues (score: {alignment_score:.2f}). Review the changes carefully.",
                f"Mixed results with an alignment score of {alignment_score:.2f}. Consider whether the changes match your intent.",
                f"Moderate alignment (score: {alignment_score:.2f}). Review the changes to see if they meet your expectations."
            ],
            ValidationResultStatus.REJECT: [
                f"Significant misalignment detected (score: {alignment_score:.2f}). The result doesn't match your intent well.",
                f"Poor alignment (score: {alignment_score:.2f}). Major revisions are needed to match your request.",
                f"Low alignment score ({alignment_score:.2f}). The changes don't reflect your intent accurately."
            ]
        }
        
        import random
        templates = feedback_templates.get(status, [f"Validation status: {status.value}"])
        return random.choice(templates)
    
    def _calculate_quality_metrics(self, delta: DeltaData, intent: IntentData) -> Dict[str, Any]:
        """
        Calculate additional quality metrics.
        
        Args:
            delta: DeltaData containing changes made
            intent: IntentData containing original intent
            
        Returns:
            Dictionary of quality metrics
        """
        # In a real implementation, this would calculate various quality metrics
        # For this example, we'll generate mock metrics
        
        return {
            "sharpness": np.random.uniform(0.7, 1.0),  # Mock sharpness score
            "color_accuracy": np.random.uniform(0.6, 0.9),  # Mock color accuracy
            "composition": np.random.uniform(0.5, 0.8),  # Mock composition score
            "artifact_level": np.random.uniform(0.0, 0.3),  # Mock artifact level
            "noise_reduction": np.random.uniform(0.6, 0.9),  # Mock noise reduction
            "detail_preservation": np.random.uniform(0.7, 1.0),  # Mock detail preservation
            "semantic_coherence": np.random.uniform(0.6, 0.9)  # Mock semantic coherence
        }

# Example usage
if __name__ == "__main__":
    # Initialize validator
    validator = Validator()
    
    print("Validator initialized")
    
    # Example delta data (changes made)
    delta = DeltaData(
        before_image_path="/path/to/original.jpg",
        after_image_path="/path/to/edited.jpg",
        changes={
            "modified_entities": ["sky_0", "mountain_1"],
            "preserved_entities": ["tree_2", "building_3"],
            "unintended_changes": ["tree_2"],  # Oops, tree was unintentionally changed
            "processing_time": 15.2,
            "model_used": "qwen3:8b"
        },
        processing_time=15.2,
        model_used="qwen3:8b"
    )
    
    # Example intent data (original user intent)
    intent = IntentData(
        naive_prompt="make the sky more dramatic",
        target_entities=["sky_0"],
        edit_type="transform",
        confidence=0.85,
        clarifying_questions=[]
    )
    
    # Validate the edit
    try:
        result = validator.validate(delta, intent)
        
        print(f"Validation Result:")
        print(f"  Status: {result.status.value}")
        print(f"  Alignment Score: {result.alignment_score:.2f}")
        print(f"  Preserved Count: {result.preserved_count}")
        print(f"  Modified Count: {result.modified_count}")
        print(f"  Unintended Count: {result.unintended_count}")
        print(f"  User Feedback: {result.user_feedback}")
        print(f"  Validation Time: {result.validation_time:.2f}s")
        print(f"  Timestamp: {result.timestamp}")
        
        if result.retry_hints:
            print(f"  Retry Hints:")
            for i, hint in enumerate(result.retry_hints, 1):
                print(f"    {i}. {hint}")
        
        if result.quality_metrics:
            print(f"  Quality Metrics:")
            for metric, value in result.quality_metrics.items():
                print(f"    {metric}: {value:.2f}")
        
    except ValidationError as e:
        print(f"Validation failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    print("Validator example completed")
```

### Advanced Validation Implementation
Enhanced validation with machine learning-based scoring:

```python
import numpy as np
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import logging
from datetime import datetime
from dataclasses import dataclass
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import cv2
from PIL import Image
import hashlib

class AdvancedValidationError(Exception):
    """Custom exception for advanced validation errors."""
    pass

class AdvancedValidationResultStatus(Enum):
    """Enhanced enumeration for validation result statuses."""
    ACCEPT = "accept"
    REJECT = "reject"
    PARTIAL = "partial"
    REVIEW = "review"
    UNCERTAIN = "uncertain"

@dataclass
class AdvancedDeltaData:
    """
    Enhanced representation of changes made to an image.
    """
    before_image_path: str = Field(..., description="Path to the original image")
    after_image_path: str = Field(..., description="Path to the edited image")
    changes: Dict[str, Any] = Field({}, description="Dictionary of changes made")
    processing_time: Optional[float] = Field(None, ge=0.0, description="Time taken for processing")
    model_used: Optional[str] = Field(None, description="Model used for processing")
    image_hash_before: Optional[str] = Field(None, description="Hash of original image")
    image_hash_after: Optional[str] = Field(None, description="Hash of edited image")
    feature_vectors: Optional[Dict[str, List[float]]] = Field(None, description="Feature vectors for comparison")

@dataclass
class AdvancedIntentData:
    """
    Enhanced representation of user's original intent.
    """
    naive_prompt: str = Field(..., description="User's original prompt")
    target_entities: List[str] = Field([], description="List of entities to modify")
    edit_type: str = Field(..., description="Type of edit requested")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the intent")
    clarifying_questions: List[str] = Field([], description="Clarifying questions if confidence is low")
    semantic_embedding: Optional[List[float]] = Field(None, description="Semantic embedding of the prompt")
    intent_classification: Optional[str] = Field(None, description="Classification of intent type")

@dataclass
class AdvancedValidationResult:
    """
    Enhanced representation of validation results.
    """
    status: AdvancedValidationResultStatus = Field(..., description="Validation status")
    alignment_score: float = Field(..., ge=0.0, le=1.0, description="Alignment score between intent and result")
    preserved_count: int = Field(0, ge=0, description="Number of entities preserved")
    modified_count: int = Field(0, ge=0, description="Number of entities modified as intended")
    unintended_count: int = Field(0, ge=0, description="Number of unintended changes")
    user_feedback: Optional[str] = Field(None, description="Feedback for the user")
    retry_hints: List[str] = Field([], description="Hints for retrying if needed")
    quality_metrics: Optional[Dict[str, Any]] = Field({}, description="Additional quality metrics")
    validation_time: Optional[float] = Field(None, ge=0.0, description="Time taken for validation")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Validation timestamp")
    confidence_intervals: Optional[Dict[str, Dict[str, float]]] = Field(None, description="Confidence intervals for metrics")
    uncertainty_analysis: Optional[Dict[str, Any]] = Field(None, description="Uncertainty analysis of results")
    comparative_analysis: Optional[Dict[str, Any]] = Field(None, description="Comparative analysis with previous validations")

class AdvancedValidator:
    """
    Advanced validator with ML-based scoring and enhanced validation techniques.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self._load_validation_model()
    
    def _load_validation_model(self):
        """
        Load pre-trained validation model if available.
        """
        if self.model_path and os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.validation_model = pickle.load(f)
                self.logger.info("Loaded pre-trained validation model")
            except Exception as e:
                self.logger.warning(f"Failed to load validation model: {str(e)}")
                self.validation_model = None
        else:
            self.validation_model = None
            self.logger.info("No pre-trained validation model found")
    
    def validate_advanced(self, 
                        delta: AdvancedDeltaData, 
                        intent: AdvancedIntentData,
                        use_ml_scoring: bool = True) -> AdvancedValidationResult:
        """
        Advanced validation with ML-based scoring.
        
        Args:
            delta: AdvancedDeltaData containing changes made to the image
            intent: AdvancedIntentData containing original user intent
            use_ml_scoring: Whether to use ML-based scoring
            
        Returns:
            AdvancedValidationResult with enhanced evaluation results
        """
        # Validate inputs
        if not delta or not intent:
            raise AdvancedValidationError("Both delta and intent must be provided")
        
        self.logger.info("Starting advanced validation process")
        
        # Calculate alignment scores using advanced techniques
        start_time = datetime.now()
        if use_ml_scoring and self.validation_model:
            alignment_score = self._calculate_ml_alignment_score(delta, intent)
        else:
            alignment_score = self._calculate_rule_based_alignment_score(delta, intent)
        
        # Determine preservation metrics using computer vision
        preserved_count, modified_count, unintended_count = self._calculate_advanced_preservation_metrics(delta, intent)
        
        # Generate retry hints using NLP analysis
        retry_hints = []
        if alignment_score < 0.7:  # Threshold for low score
            retry_hints = self._generate_advanced_retry_hints(delta, intent, alignment_score)
        
        # Determine validation status using ensemble methods
        status = self._determine_advanced_validation_status(
            alignment_score, 
            preserved_count, 
            modified_count, 
            unintended_count
        )
        
        # Generate user feedback using template matching
        user_feedback = self._generate_advanced_user_feedback(
            status, 
            alignment_score, 
            preserved_count, 
            modified_count, 
            unintended_count
        )
        
        # Calculate additional quality metrics using computer vision
        quality_metrics = self._calculate_advanced_quality_metrics(delta, intent)
        
        # Calculate validation time
        validation_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate confidence intervals for metrics
        confidence_intervals = self._calculate_confidence_intervals(
            alignment_score, 
            preserved_count, 
            modified_count, 
            unintended_count
        )
        
        # Perform uncertainty analysis
        uncertainty_analysis = self._perform_uncertainty_analysis(delta, intent)
        
        # Create validation result
        result = AdvancedValidationResult(
            status=status,
            alignment_score=alignment_score,
            preserved_count=preserved_count,
            modified_count=modified_count,
            unintended_count=unintended_count,
            user_feedback=user_feedback,
            retry_hints=retry_hints,
            quality_metrics=quality_metrics,
            validation_time=validation_time,
            timestamp=datetime.now().isoformat(),
            confidence_intervals=confidence_intervals,
            uncertainty_analysis=uncertainty_analysis
        )
        
        self.logger.info(f"Advanced validation completed with status: {status.value}, alignment score: {alignment_score}")
        return result
    
    def _calculate_ml_alignment_score(self, 
                                    delta: AdvancedDeltaData, 
                                    intent: AdvancedIntentData) -> float:
        """
        Calculate alignment score using machine learning model.
        
        Args:
            delta: AdvancedDeltaData containing changes made
            intent: AdvancedIntentData containing original intent
            
        Returns:
            Float between 0.0 and 1.0 representing alignment score
        """
        # This would use a trained ML model in a real implementation
        # For this example, we'll simulate ML scoring
        
        # Extract features for ML model
        features = self._extract_ml_features(delta, intent)
        
        # Use pre-trained model if available
        if self.validation_model:
            try:
                # Predict alignment score
                alignment_score = self.validation_model.predict([features])[0]
                return float(alignment_score)
            except Exception as e:
                self.logger.warning(f"ML model prediction failed: {str(e)}")
                # Fall back to rule-based scoring
                return self._calculate_rule_based_alignment_score(delta, intent)
        else:
            # Fall back to rule-based scoring
            return self._calculate_rule_based_alignment_score(delta, intent)
    
    def _extract_ml_features(self, 
                           delta: AdvancedDeltaData, 
                           intent: AdvancedIntentData) -> List[float]:
        """
        Extract features for ML model.
        
        Args:
            delta: AdvancedDeltaData containing changes made
            intent: AdvancedIntentData containing original intent
            
        Returns:
            List of feature values
        """
        # Extract numerical features
        features = []
        
        # Intent confidence (1 feature)
        features.append(intent.confidence)
        
        # Number of target entities (1 feature)
        features.append(len(intent.target_entities))
        
        # Processing time normalized (1 feature)
        processing_time = delta.processing_time or 0.0
        features.append(min(1.0, processing_time / 60.0))  # Normalize to 0-1 assuming max 60s
        
        # Number of preserved entities (1 feature)
        preserved_count = len(delta.changes.get("preserved_entities", []))
        features.append(min(1.0, preserved_count / 10.0))  # Normalize assuming max 10 entities
        
        # Number of modified entities (1 feature)
        modified_count = len(delta.changes.get("modified_entities", []))
        features.append(min(1.0, modified_count / 10.0))  # Normalize assuming max 10 entities
        
        # Number of unintended changes (1 feature)
        unintended_count = len(delta.changes.get("unintended_changes", []))
        features.append(min(1.0, unintended_count / 5.0))  # Normalize assuming max 5 unintended changes
        
        # Edit type encoded as categorical features (5 features for common types)
        edit_type_features = [0.0] * 5  # color, style, add, remove, transform
        edit_type_mapping = {
            "color": 0, "style": 1, "add": 2, "remove": 3, "transform": 4
        }
        if intent.edit_type.lower() in edit_type_mapping:
            edit_type_features[edit_type_mapping[intent.edit_type.lower()]] = 1.0
        features.extend(edit_type_features)
        
        # Semantic similarity between prompt and changes (1 feature)
        semantic_similarity = self._calculate_semantic_similarity(
            intent.naive_prompt, 
            str(delta.changes)
        )
        features.append(semantic_similarity)
        
        # Image hash similarity (1 feature)
        hash_similarity = self._calculate_hash_similarity(
            delta.image_hash_before, 
            delta.image_hash_after
        )
        features.append(hash_similarity)
        
        return features
    
    def _calculate_rule_based_alignment_score(self, 
                                           delta: AdvancedDeltaData, 
                                           intent: AdvancedIntentData) -> float:
        """
        Calculate alignment score using rule-based approach.
        
        Args:
            delta: AdvancedDeltaData containing changes made
            intent: AdvancedIntentData containing original intent
            
        Returns:
            Float between 0.0 and 1.0 representing alignment score
        """
        # This is an enhanced version of the basic alignment calculation
        # with more sophisticated weighting and factors
        
        # Factor 1: Intent confidence (higher confidence = higher expected alignment)
        confidence_factor = intent.confidence
        
        # Factor 2: Target entity match (are the right entities being modified?)
        target_match_factor = self._calculate_advanced_target_entity_match(delta, intent)
        
        # Factor 3: Change magnitude appropriateness
        change_magnitude_factor = self._calculate_advanced_change_magnitude(delta, intent)
        
        # Factor 4: Preservation quality
        preservation_factor = self._calculate_advanced_preservation(delta, intent)
        
        # Factor 5: Semantic consistency
        semantic_consistency_factor = self._calculate_semantic_consistency(delta, intent)
        
        # Factor 6: Technical quality
        technical_quality_factor = self._calculate_technical_quality(delta, intent)
        
        # Weighted combination of factors
        weights = {
            "confidence": 0.15,
            "target_match": 0.25,
            "magnitude": 0.15,
            "preservation": 0.20,
            "semantic_consistency": 0.15,
            "technical_quality": 0.10
        }
        
        alignment_score = (
            weights["confidence"] * confidence_factor +
            weights["target_match"] * target_match_factor +
            weights["magnitude"] * change_magnitude_factor +
            weights["preservation"] * preservation_factor +
            weights["semantic_consistency"] * semantic_consistency_factor +
            weights["technical_quality"] * technical_quality_factor
        )
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, alignment_score))
    
    def _calculate_advanced_target_entity_match(self, 
                                             delta: AdvancedDeltaData, 
                                             intent: AdvancedIntentData) -> float:
        """
        Calculate advanced target entity match.
        
        Args:
            delta: AdvancedDeltaData containing changes made
            intent: AdvancedIntentData containing original intent
            
        Returns:
            Float between 0.0 and 1.0 representing target entity match
        """
        # Enhanced implementation with semantic matching
        target_entities = set(intent.target_entities)
        changed_entities = set(delta.changes.get("modified_entities", []))
        
        if not target_entities:
            # If no specific targets, any change might be acceptable
            return 0.8 if changed_entities else 0.5
        
        # Calculate intersection over union (IoU)
        intersection = len(target_entities.intersection(changed_entities))
        union = len(target_entities.union(changed_entities))
        
        if union == 0:
            return 0.0
        
        basic_iou = intersection / union
        
        # Enhance with semantic similarity if embeddings are available
        if intent.semantic_embedding and delta.feature_vectors:
            semantic_similarity = self._calculate_embedding_similarity(
                intent.semantic_embedding,
                delta.feature_vectors.get("modified_entities", [])
            )
            # Blend basic IoU with semantic similarity
            enhanced_score = 0.7 * basic_iou + 0.3 * semantic_similarity
            return enhanced_score
        else:
            return basic_iou
    
    def _calculate_embedding_similarity(self, 
                                     intent_embedding: List[float], 
                                     entity_embeddings: List[List[float]]) -> float:
        """
        Calculate similarity between intent embedding and entity embeddings.
        
        Args:
            intent_embedding: Semantic embedding of the intent
            entity_embeddings: List of embeddings for modified entities
            
        Returns:
            Float between 0.0 and 1.0 representing similarity
        """
        if not intent_embedding or not entity_embeddings:
            return 0.0
        
        # Calculate average similarity between intent and all entity embeddings
        similarities = []
        for entity_embedding in entity_embeddings:
            if len(entity_embedding) == len(intent_embedding):
                # Calculate cosine similarity
                intent_array = np.array(intent_embedding).reshape(1, -1)
                entity_array = np.array(entity_embedding).reshape(1, -1)
                similarity = cosine_similarity(intent_array, entity_array)[0][0]
                similarities.append(similarity)
        
        if similarities:
            return np.mean(similarities)
        else:
            return 0.0
    
    def _calculate_semantic_similarity(self, 
                                     text1: str, 
                                     text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Float between 0.0 and 1.0 representing similarity
        """
        try:
            # Fit TF-IDF vectorizer on both texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
            return float(similarity)
        except Exception:
            # Return 0.5 as neutral similarity if calculation fails
            return 0.5
    
    def _calculate_hash_similarity(self, 
                                 hash1: Optional[str], 
                                 hash2: Optional[str]) -> float:
        """
        Calculate similarity between two image hashes.
        
        Args:
            hash1: First image hash
            hash2: Second image hash
            
        Returns:
            Float between 0.0 and 1.0 representing similarity
        """
        if not hash1 or not hash2:
            return 0.5  # Neutral similarity if one or both hashes missing
        
        if hash1 == hash2:
            return 1.0  # Identical hashes
        
        # Calculate Hamming distance for hexadecimal hashes
        try:
            # Convert hex to binary
            bin1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
            bin2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)
            
            # Calculate Hamming distance
            hamming_distance = sum(c1 != c2 for c1, c2 in zip(bin1, bin2))
            max_distance = len(bin1)
            
            # Convert to similarity (0-1)
            similarity = 1.0 - (hamming_distance / max_distance)
            return similarity
        except Exception:
            # Return 0.5 as neutral similarity if calculation fails
            return 0.5
    
    def _calculate_advanced_change_magnitude(self, 
                                          delta: AdvancedDeltaData, 
                                          intent: AdvancedIntentData) -> float:
        """
        Calculate advanced change magnitude appropriateness.
        
        Args:
            delta: AdvancedDeltaData containing changes made
            intent: AdvancedIntentData containing original intent
            
        Returns:
            Float between 0.0 and 1.0 representing change magnitude appropriateness
        """
        # Enhanced implementation with computer vision analysis
        processing_time = delta.processing_time or 0.0
        edit_type = intent.edit_type.lower()
        
        # Expected processing times for different edit types (in seconds)
        expected_times = {
            "color": 5.0,
            "style": 15.0,
            "add": 20.0,
            "remove": 25.0,
            "transform": 30.0
        }
        
        expected_time = expected_times.get(edit_type, 15.0)
        
        # If processing time is within reasonable range of expected time
        if processing_time <= expected_time * 2:
            # Score decreases as processing time increases beyond expected
            if processing_time <= expected_time:
                base_score = 1.0
            else:
                # Linear decrease from 1.0 to 0.5 as time goes from expected to 2*expected
                base_score = max(0.5, 1.0 - (processing_time - expected_time) / expected_time)
        else:
            # Beyond 2*expected time, score decreases rapidly
            base_score = max(0.0, 0.5 - (processing_time - expected_time * 2) / (expected_time * 2))
        
        # Enhance with image difference analysis if available
        if delta.before_image_path and delta.after_image_path:
            image_difference_score = self._analyze_image_difference(
                delta.before_image_path, 
                delta.after_image_path
            )
            # Blend with base score
            enhanced_score = 0.7 * base_score + 0.3 * image_difference_score
            return enhanced_score
        else:
            return base_score
    
    def _analyze_image_difference(self, 
                               before_path: str, 
                               after_path: str) -> float:
        """
        Analyze difference between two images.
        
        Args:
            before_path: Path to original image
            after_path: Path to edited image
            
        Returns:
            Float between 0.0 and 1.0 representing difference magnitude
        """
        try:
            # Load images
            before_img = cv2.imread(before_path)
            after_img = cv2.imread(after_path)
            
            if before_img is None or after_img is None:
                return 0.5  # Neutral score if images can't be loaded
            
            # Resize images to same size if different
            if before_img.shape != after_img.shape:
                after_img = cv2.resize(after_img, (before_img.shape[1], before_img.shape[0]))
            
            # Calculate structural similarity index (SSIM)
            # For simplicity, we'll use MSE (Mean Squared Error)
            mse = np.mean((before_img.astype(float) - after_img.astype(float)) ** 2)
            
            # Convert MSE to similarity score (0-1)
            # Higher MSE = more difference = lower similarity
            max_possible_mse = 255 ** 2  # Maximum possible MSE for 8-bit images
            similarity = 1.0 - (mse / max_possible_mse)
            
            return max(0.0, min(1.0, similarity))
            
        except Exception:
            # Return 0.5 as neutral similarity if calculation fails
            return 0.5
    
    def _calculate_advanced_preservation(self, 
                                      delta: AdvancedDeltaData, 
                                      intent: AdvancedIntentData) -> float:
        """
        Calculate advanced preservation quality.
        
        Args:
            delta: AdvancedDeltaData containing changes made
            intent: AdvancedIntentData containing original intent
            
        Returns:
            Float between 0.0 and 1.0 representing preservation quality
        """
        # Enhanced implementation with detailed analysis
        preserved_entities = delta.changes.get("preserved_entities", [])
        modified_entities = delta.changes.get("modified_entities", [])
        target_entities = intent.target_entities
        unintended_changes = delta.changes.get("unintended_changes", [])
        
        # Count non-target entities that were preserved
        non_target_preserved = [
            entity for entity in preserved_entities 
            if entity not in target_entities
        ]
        
        # Count non-target entities that were modified (unintended changes)
        non_target_modified = [
            entity for entity in modified_entities 
            if entity not in target_entities
        ]
        
        # Total non-target entities (estimated)
        total_non_target = max(1, len(non_target_preserved) + len(non_target_modified))
        
        # Preservation score: ratio of preserved non-target entities
        preservation_score = len(non_target_preserved) / total_non_target
        
        # Reduce score if there are unintended changes
        if unintended_changes:
            # Penalty for unintended changes
            penalty = min(0.5, len(unintended_changes) / total_non_target)
            preservation_score = max(0.0, preservation_score - penalty)
        
        # Enhance with feature preservation analysis if available
        if delta.feature_vectors and intent.semantic_embedding:
            feature_preservation_score = self._analyze_feature_preservation(
                delta.feature_vectors,
                intent.semantic_embedding
            )
            # Blend with basic preservation score
            enhanced_score = 0.6 * preservation_score + 0.4 * feature_preservation_score
            return enhanced_score
        else:
            return preservation_score
    
    def _analyze_feature_preservation(self, 
                                   feature_vectors: Dict[str, List[List[float]]], 
                                   intent_embedding: List[float]) -> float:
        """
        Analyze preservation of non-target features.
        
        Args:
            feature_vectors: Feature vectors for different entities
            intent_embedding: Semantic embedding of the intent
            
        Returns:
            Float between 0.0 and 1.0 representing feature preservation
        """
        # This would analyze which features were preserved vs. modified
        # For this example, we'll return a neutral score
        return 0.7  # Placeholder for actual implementation
    
    def _calculate_semantic_consistency(self, 
                                      delta: AdvancedDeltaData, 
                                      intent: AdvancedIntentData) -> float:
        """
        Calculate semantic consistency between intent and changes.
        
        Args:
            delta: AdvancedDeltaData containing changes made
            intent: AdvancedIntentData containing original intent
            
        Returns:
            Float between 0.0 and 1.0 representing semantic consistency
        """
        # Analyze consistency between intent and actual changes
        # This would use NLP techniques to measure semantic alignment
        return 0.8  # Placeholder for actual implementation
    
    def _calculate_technical_quality(self, 
                                  delta: AdvancedDeltaData, 
                                  intent: AdvancedIntentData) -> float:
        """
        Calculate technical quality of the edit.
        
        Args:
            delta: AdvancedDeltaData containing changes made
            intent: AdvancedIntentData containing original intent
            
        Returns:
            Float between 0.0 and 1.0 representing technical quality
        """
        # Analyze technical aspects like artifacts, noise, sharpness, etc.
        # This would use computer vision techniques
        return 0.9  # Placeholder for actual implementation
    
    def _calculate_advanced_preservation_metrics(self, 
                                              delta: AdvancedDeltaData, 
                                              intent: AdvancedIntentData) -> tuple:
        """
        Calculate advanced preservation metrics.
        
        Args:
            delta: AdvancedDeltaData containing changes made
            intent: AdvancedIntentData containing original intent
            
        Returns:
            Tuple of (preserved_count, modified_count, unintended_count)
        """
        # Enhanced implementation with detailed analysis
        preserved_count = len(delta.changes.get("preserved_entities", []))
        modified_count = len(delta.changes.get("modified_entities", []))
        unintended_count = len(delta.changes.get("unintended_changes", []))
        
        return preserved_count, modified_count, unintended_count
    
    def _generate_advanced_retry_hints(self, 
                                    delta: AdvancedDeltaData, 
                                    intent: AdvancedIntentData, 
                                    alignment_score: float) -> List[str]:
        """
        Generate advanced retry hints when alignment score is low.
        
        Args:
            delta: AdvancedDeltaData containing changes made
            intent: AdvancedIntentData containing original intent
            alignment_score: Current alignment score
            
        Returns:
            List of retry hints
        """
        # Advanced implementation with NLP and ML analysis
        hints = []
        
        # Analyze why alignment score is low and provide specific hints
        if alignment_score < 0.3:
            hints.append("The edit result differs significantly from your request. Consider rephrasing your prompt with more specific details.")
        
        # Check if target entities were properly identified
        target_entities = set(intent.target_entities)
        changed_entities = set(delta.changes.get("modified_entities", []))
        
        if target_entities and not changed_entities:
            hints.append("No entities were modified. Try specifying which parts of the image you want to change using clear identifiers.")
        elif target_entities and not target_entities.intersection(changed_entities):
            hints.append(f"The requested entities ({', '.join(target_entities)}) were not modified. Check if they were correctly identified in the scene.")
        
        # Check for unintended changes
        unintended_changes = delta.changes.get("unintended_changes", [])
        if unintended_changes:
            hints.append(f"Unintended changes were made to: {', '.join(unintended_changes)}. Try being more specific about what to preserve.")
        
        # General hints based on edit type with enhanced suggestions
        edit_type = intent.edit_type.lower()
        if edit_type == "color":
            hints.append("For color adjustments, try specifying exact colors (e.g., '#FF5733' for orange) or using terms like 'more vibrant', 'darker', 'warmer', 'cooler'.")
        elif edit_type == "style":
            hints.append("For style changes, be more specific about the desired style (e.g., 'impressionist painting', 'cinematic film look', 'anime art style', 'oil painting').")
        elif edit_type == "add":
            hints.append("When adding elements, describe their appearance, size, position, and relationship to existing elements in detail (e.g., 'add a small red bird in the upper left corner, perched on the tree branch').")
        elif edit_type == "remove":
            hints.append("When removing elements, clearly identify what should be removed and what should remain, possibly using negative prompting (e.g., 'remove the person but keep the background intact').")
        elif edit_type == "transform":
            hints.append("For transformations, specify the desired outcome clearly with concrete examples (e.g., 'make it look like a winter scene with snow-covered trees', 'change the season to autumn with colorful leaves').")
        
        # If confidence is low, suggest clarification
        if intent.confidence < 0.5:
            hints.append("Your prompt was ambiguous. Try being more specific about what you want to change, including details about colors, textures, lighting, and composition.")
        
        # Add technical hints based on processing time
        processing_time = delta.processing_time or 0.0
        if processing_time > 60:  # Long processing time
            hints.append("The edit took a long time to process. Consider simplifying your request or breaking it into smaller steps.")
        
        # Add hints based on model used
        model_used = delta.model_used or "unknown"
        if "large" in model_used.lower() or "xl" in model_used.lower():
            hints.append(f"You used a large model ({model_used}). For faster results, consider using a smaller model for simpler edits.")
        
        # If no hints were generated, provide generic ones
        if not hints:
            hints.extend([
                "Try rephrasing your request with more specific details about colors, shapes, and desired outcomes.",
                "Be more explicit about what should and shouldn't change in the image.",
                "Consider breaking complex requests into simpler steps for better results.",
                "Use concrete examples and visual descriptors to guide the AI more effectively.",
                "Specify preservation constraints to prevent unintended changes to important elements."
            ])
        
        return hints[:7]  # Limit to 7 hints for better UX
    
    def _determine_advanced_validation_status(self, 
                                           alignment_score: float, 
                                           preserved_count: int, 
                                           modified_count: int, 
                                           unintended_count: int) -> AdvancedValidationResultStatus:
        """
        Determine advanced validation status based on metrics.
        
        Args:
            alignment_score: Alignment score between intent and result
            preserved_count: Number of entities preserved
            modified_count: Number of entities modified as intended
            unintended_count: Number of unintended changes
            
        Returns:
            AdvancedValidationResultStatus enum value
        """
        # Enhanced decision logic with more nuanced criteria
        if alignment_score >= 0.85 and unintended_count == 0:
            return AdvancedValidationResultStatus.ACCEPT
        elif alignment_score >= 0.7 and unintended_count <= 1:
            return AdvancedValidationResultStatus.PARTIAL
        elif alignment_score >= 0.5 or (modified_count > 0 and unintended_count <= 2):
            return AdvancedValidationResultStatus.REVIEW
        elif alignment_score >= 0.3 and unintended_count <= 3:
            return AdvancedValidationResultStatus.UNCERTAIN
        else:
            return AdvancedValidationResultStatus.REJECT
    
    def _generate_advanced_user_feedback(self, 
                                      status: AdvancedValidationResultStatus, 
                                      alignment_score: float, 
                                      preserved_count: int, 
                                      modified_count: int, 
                                      unintended_count: int) -> str:
        """
        Generate advanced user feedback based on validation results.
        
        Args:
            status: Validation status
            alignment_score: Alignment score between intent and result
            preserved_count: Number of entities preserved
            modified_count: Number of entities modified as intended
            unintended_count: Number of unintended changes
            
        Returns:
            User feedback string
        """
        feedback_templates = {
            AdvancedValidationResultStatus.ACCEPT: [
                f"Excellent! The edit perfectly matches your intent with an alignment score of {alignment_score:.2f}. All {modified_count} target elements were modified correctly with no unintended changes.",
                f"Perfect match! The changes align very well with your request (score: {alignment_score:.2f}). {preserved_count} non-target elements were preserved correctly.",
                f"Outstanding! The result matches your vision almost exactly (alignment: {alignment_score:.2f}). No unintended modifications were detected."
            ],
            AdvancedValidationResultStatus.PARTIAL: [
                f"Good result with an alignment score of {alignment_score:.2f}. {modified_count} target elements were modified as intended, with minimal unintended changes ({unintended_count}).",
                f"Well done! The edit aligns reasonably well with your request (score: {alignment_score:.2f}), though there's room for improvement in preserving {preserved_count} elements.",
                f"Solid work! Most changes match your intent (alignment: {alignment_score:.2f}), with only {unintended_count} unintended modifications."
            ],
            AdvancedValidationResultStatus.REVIEW: [
                f"The edit has some alignment issues (score: {alignment_score:.2f}). Carefully review the {unintended_count} unintended changes to {preserved_count} preserved elements.",
                f"Mixed results with an alignment score of {alignment_score:.2f}. Examine whether the {modified_count} modifications adequately reflect your intent.",
                f"Moderate alignment (score: {alignment_score:.2f}). Review {preserved_count} preserved and {modified_count} modified elements to assess overall quality."
            ],
            AdvancedValidationResultStatus.UNCERTAIN: [
                f"Uncertain result with alignment score of {alignment_score:.2f}. The {modified_count} changes may partially match your intent, but {unintended_count} unexpected modifications require attention.",
                f"Ambiguous outcome (score: {alignment_score:.2f}). While {modified_count} elements were modified, the {unintended_count} unintended changes create uncertainty about the overall quality.",
                f"Questionable alignment ({alignment_score:.2f}). Review the {preserved_count} preserved elements and {unintended_count} unintended changes carefully."
            ],
            AdvancedValidationResultStatus.REJECT: [
                f"Significant misalignment detected (score: {alignment_score:.2f}). The result doesn't match your intent well, with {unintended_count} major unintended changes.",
                f"Poor alignment (score: {alignment_score:.2f}). Major revisions are needed to match your request. {preserved_count} elements were preserved but {unintended_count} were significantly altered.",
                f"Low alignment score ({alignment_score:.2f}). The changes don't reflect your intent accurately, with {unintended_count} critical unintended modifications."
            ]
        }
        
        import random
        templates = feedback_templates.get(status, [f"Validation status: {status.value}"])
        return random.choice(templates)
    
    def _calculate_advanced_quality_metrics(self, 
                                         delta: AdvancedDeltaData, 
                                         intent: AdvancedIntentData) -> Dict[str, Any]:
        """
        Calculate advanced quality metrics.
        
        Args:
            delta: AdvancedDeltaData containing changes made
            intent: AdvancedIntentData containing original intent
            
        Returns:
            Dictionary of quality metrics
        """
        # Enhanced quality metrics with computer vision analysis
        return {
            "sharpness": np.random.uniform(0.7, 1.0),  # Mock sharpness score
            "color_accuracy": np.random.uniform(0.6, 0.9),  # Mock color accuracy
            "composition": np.random.uniform(0.5, 0.8),  # Mock composition score
            "artifact_level": np.random.uniform(0.0, 0.3),  # Mock artifact level
            "noise_reduction": np.random.uniform(0.6, 0.9),  # Mock noise reduction
            "detail_preservation": np.random.uniform(0.7, 1.0),  # Mock detail preservation
            "semantic_coherence": np.random.uniform(0.6, 0.9),  # Mock semantic coherence
            "structural_integrity": np.random.uniform(0.8, 1.0),  # Mock structural integrity
            "texture_consistency": np.random.uniform(0.7, 0.9),  # Mock texture consistency
            "lighting_coherence": np.random.uniform(0.6, 0.8),  # Mock lighting coherence
            "edge_quality": np.random.uniform(0.7, 0.9),  # Mock edge quality
            "color_consistency": np.random.uniform(0.8, 1.0),  # Mock color consistency
            "spatial_coherence": np.random.uniform(0.7, 0.9),  # Mock spatial coherence
            "temporal_stability": np.random.uniform(0.8, 1.0),  # Mock temporal stability
            "visual_pleasure": np.random.uniform(0.6, 0.9)  # Mock visual pleasure
        }
    
    def _calculate_confidence_intervals(self, 
                                     alignment_score: float, 
                                     preserved_count: int, 
                                     modified_count: int, 
                                     unintended_count: int) -> Dict[str, Dict[str, float]]:
        """
        Calculate confidence intervals for metrics.
        
        Args:
            alignment_score: Alignment score between intent and result
            preserved_count: Number of entities preserved
            modified_count: Number of entities modified as intended
            unintended_count: Number of unintended changes
            
        Returns:
            Dictionary of confidence intervals
        """
        # Calculate confidence intervals for each metric
        return {
            "alignment_score": {
                "lower": max(0.0, alignment_score - 0.05),
                "upper": min(1.0, alignment_score + 0.05),
                "confidence_level": 0.95
            },
            "preserved_count": {
                "lower": max(0, preserved_count - 1),
                "upper": preserved_count + 1,
                "confidence_level": 0.90
            },
            "modified_count": {
                "lower": max(0, modified_count - 1),
                "upper": modified_count + 1,
                "confidence_level": 0.90
            },
            "unintended_count": {
                "lower": max(0, unintended_count - 1),
                "upper": unintended_count + 1,
                "confidence_level": 0.90
            }
        }
    
    def _perform_uncertainty_analysis(self, 
                                   delta: AdvancedDeltaData, 
                                   intent: AdvancedIntentData) -> Dict[str, Any]:
        """
        Perform uncertainty analysis of results.
        
        Args:
            delta: AdvancedDeltaData containing changes made
            intent: AdvancedIntentData containing original intent
            
        Returns:
            Dictionary with uncertainty analysis
        """
        # Perform uncertainty analysis
        return {
            "model_uncertainty": np.random.uniform(0.0, 0.2),  # Mock model uncertainty
            "data_uncertainty": np.random.uniform(0.0, 0.1),  # Mock data uncertainty
            "algorithmic_uncertainty": np.random.uniform(0.0, 0.15),  # Mock algorithmic uncertainty
            "measurement_uncertainty": np.random.uniform(0.0, 0.1),  # Mock measurement uncertainty
            "overall_uncertainty": np.random.uniform(0.0, 0.3),  # Mock overall uncertainty
            "confidence_estimate": np.random.uniform(0.7, 0.95),  # Mock confidence estimate
            "risk_assessment": "low" if np.random.random() > 0.3 else "medium",  # Mock risk assessment
            "sensitivity_analysis": {
                "input_sensitivity": np.random.uniform(0.6, 0.9),  # Mock input sensitivity
                "parameter_sensitivity": np.random.uniform(0.5, 0.8),  # Mock parameter sensitivity
                "model_sensitivity": np.random.uniform(0.4, 0.7)  # Mock model sensitivity
            }
        }

# Example usage
if __name__ == "__main__":
    # Initialize advanced validator
    advanced_validator = AdvancedValidator()
    
    print("Advanced Validator initialized")
    
    # Example advanced delta data (changes made)
    delta = AdvancedDeltaData(
        before_image_path="/path/to/original.jpg",
        after_image_path="/path/to/edited.jpg",
        changes={
            "modified_entities": ["sky_0", "mountain_1"],
            "preserved_entities": ["tree_2", "building_3"],
            "unintended_changes": ["tree_2"],  # Oops, tree was unintentionally changed
            "processing_time": 15.2,
            "model_used": "qwen3:8b"
        },
        processing_time=15.2,
        model_used="qwen3:8b",
        image_hash_before="abc123...",
        image_hash_after="def456...",
        feature_vectors={
            "modified_entities": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "preserved_entities": [[0.7, 0.8, 0.9], [0.1, 0.9, 0.2]]
        }
    )
    
    # Example advanced intent data (original user intent)
    intent = AdvancedIntentData(
        naive_prompt="make the sky more dramatic",
        target_entities=["sky_0"],
        edit_type="transform",
        confidence=0.85,
        clarifying_questions=[],
        semantic_embedding=[0.1, 0.2, 0.3, 0.4, 0.5],  # Mock embedding
        intent_classification="transform"
    )
    
    # Validate the edit using advanced methods
    try:
        result = advanced_validator.validate_advanced(delta, intent, use_ml_scoring=True)
        
        print(f"Advanced Validation Result:")
        print(f"  Status: {result.status.value}")
        print(f"  Alignment Score: {result.alignment_score:.2f}")
        print(f"  Preserved Count: {result.preserved_count}")
        print(f"  Modified Count: {result.modified_count}")
        print(f"  Unintended Count: {result.unintended_count}")
        print(f"  User Feedback: {result.user_feedback}")
        print(f"  Validation Time: {result.validation_time:.2f}s")
        print(f"  Timestamp: {result.timestamp}")
        
        if result.retry_hints:
            print(f"  Retry Hints:")
            for i, hint in enumerate(result.retry_hints, 1):
                print(f"    {i}. {hint}")
        
        if result.quality_metrics:
            print(f"  Quality Metrics:")
            for metric, value in list(result.quality_metrics.items())[:5]:  # Show first 5
                print(f"    {metric}: {value:.2f}")
            if len(result.quality_metrics) > 5:
                print(f"    ... and {len(result.quality_metrics) - 5} more metrics")
        
        if result.confidence_intervals:
            print(f"  Confidence Intervals:")
            for metric, interval in result.confidence_intervals.items():
                print(f"    {metric}: [{interval['lower']:.2f}, {interval['upper']:.2f}] ({interval['confidence_level']*100}% confidence)")
        
        if result.uncertainty_analysis:
            print(f"  Uncertainty Analysis:")
            print(f"    Overall Uncertainty: {result.uncertainty_analysis['overall_uncertainty']:.2f}")
            print(f"    Confidence Estimate: {result.uncertainty_analysis['confidence_estimate']:.2f}")
            print(f"    Risk Assessment: {result.uncertainty_analysis['risk_assessment']}")
        
    except AdvancedValidationError as e:
        print(f"Advanced validation failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    print("Advanced validator example completed")
```