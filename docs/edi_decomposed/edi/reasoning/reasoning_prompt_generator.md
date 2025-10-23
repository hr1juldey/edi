# Reasoning: Prompt Generator

[Back to Reasoning Subsystem](./reasoning_subsystem.md)

## Purpose
DSpy prompt creation - Contains the PromptGenerator class (a dspy.Module) that creates positive and negative prompts based on intent and scene, with 3 refinement iterations.

## Class: PromptGenerator(dspy.Module)

### Methods
- `forward(intent, scene) -> Prompts`: Generates initial prompts based on intent and scene analysis

### Details
- Base generation followed by 3 refinement iterations
- Creates both positive and negative prompts
- Uses DSpy framework for structured LLM interactions

## AI Action Required: External Library Investigation

This module uses the **DSpy** library

1.  **Request for Information:** Provide a link to the official documentation for `DSpy`

2.  **Search and Test:** I will search for examples of `dspy.Signature` and `dspy.ChainOfThought` to verify my understanding before implementation

## Functions

- [forward(intent, scene)](./reasoning/forward_prompts.md)

## Technology Stack

- DSpy for LLM orchestration
- Pydantic for data validation

## See Docs

### Python Implementation Example
Reasoning prompt generator implementation using DSpy:

```python
import dspy
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
import logging
from dataclasses import dataclass

class PromptGenerationError(Exception):
    """Custom exception for prompt generation errors."""
    pass

class PromptType(Enum):
    """Enumeration for prompt types."""
    POSITIVE = "positive"
    NEGATIVE = "negative"

@dataclass
class GeneratedPrompts:
    """
    Data structure for generated prompts.
    """
    positive_prompt: str = Field(..., description="Positive prompt for desired changes")
    negative_prompt: str = Field(..., description="Negative prompt for things to avoid")
    iteration: int = Field(0, description="Iteration number")
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Quality score of the prompts")
    generation_time: Optional[float] = Field(None, ge=0.0, description="Time taken to generate prompts")
    metadata: Optional[Dict[str, Any]] = Field({}, description="Additional metadata about the generation")

class GeneratePromptsSignature(dspy.Signature):
    """
    DSpy signature for generating prompts based on intent and scene analysis.
    """
    intent = dspy.InputField(
        desc="JSON of parsed user intent with target entities and edit type"
    )
    scene_analysis = dspy.InputField(
        desc="JSON of detected entities and layout"
    )
    
    positive_prompt = dspy.OutputField(
        desc="Positive prompt describing desired changes"
    )
    negative_prompt = dspy.OutputField(
        desc="Negative prompt describing things to avoid"
    )
    quality_score = dspy.OutputField(
        desc="Float 0-1 indicating quality of generated prompts"
    )

class RefinePromptsSignature(dspy.Signature):
    """
    DSpy signature for refining prompts based on a specific goal.
    """
    naive_prompt = dspy.InputField(
        desc="Original user prompt"
    )
    previous_positive = dspy.InputField(
        desc="Previous positive prompt to refine"
    )
    previous_negative = dspy.InputField(
        desc="Previous negative prompt to refine"
    )
    refinement_goal = dspy.InputField(
        desc="Specific goal for refinement (e.g., 'add preservation constraints', 'increase technical specificity')"
    )
    
    refined_positive = dspy.OutputField(
        desc="Refined positive prompt"
    )
    refined_negative = dspy.OutputField(
        desc="Refined negative prompt"
    )
    improvement_explanation = dspy.OutputField(
        desc="Explanation of improvements made"
    )

class PromptGenerator(dspy.Module):
    """
    DSpy module for generating positive and negative prompts based on intent and scene,
    with 3 refinement iterations.
    """
    
    def __init__(self):
        super().__init__()
        self.generate_prompts = dspy.ChainOfThought(GeneratePromptsSignature)
        self.refine_prompts = dspy.ChainOfThought(RefinePromptsSignature)
        self.logger = logging.getLogger(__name__)
    
    def forward(self, intent: Dict[str, Any], scene: Dict[str, Any]) -> List[GeneratedPrompts]:
        """
        Generates initial prompts based on intent and scene analysis.
        
        Args:
            intent: Parsed user intent with target entities and edit type
            scene: Scene analysis data with detected entities and layout
            
        Returns:
            List of GeneratedPrompts objects with refinement iterations
        """
        # Validate inputs
        if not intent:
            raise PromptGenerationError("Intent cannot be empty")
        
        if not scene:
            raise PromptGenerationError("Scene analysis cannot be empty")
        
        self.logger.info("Starting prompt generation process")
        
        # Generate base prompts based on intent and scene analysis
        base_prompts = self._generate_base_prompts(intent, scene)
        
        # Perform 3 refinement iterations
        refined_prompts = self._refine_prompts_iteratively(
            base_prompts, 
            intent.get("naive_prompt", ""), 
            3
        )
        
        self.logger.info(f"Generated {len(refined_prompts)} prompt iterations")
        return refined_prompts
    
    def _generate_base_prompts(self, intent: Dict[str, Any], scene: Dict[str, Any]) -> GeneratedPrompts:
        """
        Generate base prompts based on intent and scene analysis.
        
        Args:
            intent: Parsed user intent with target entities and edit type
            scene: Scene analysis data with detected entities and layout
            
        Returns:
            GeneratedPrompts object with base prompts
        """
        try:
            # Process intent and scene analysis to create base prompts
            result = self.generate_prompts(
                intent=json.dumps(intent, default=str),
                scene_analysis=json.dumps(scene, default=str)
            )
            
            # Parse results
            positive_prompt = result.positive_prompt
            negative_prompt = result.negative_prompt
            
            # Parse quality score
            try:
                quality_score = float(result.quality_score) if result.quality_score else None
            except (ValueError, TypeError):
                quality_score = None
            
            # Create base prompts object
            base_prompts = GeneratedPrompts(
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                iteration=0,
                quality_score=quality_score,
                metadata={
                    "generation_method": "base",
                    "intent_used": intent,
                    "scene_used": scene
                }
            )
            
            self.logger.info("Base prompts generated successfully")
            return base_prompts
            
        except Exception as e:
            self.logger.error(f"Failed to generate base prompts: {str(e)}")
            raise PromptGenerationError(f"Base prompt generation failed: {str(e)}")
    
    def _refine_prompts_iteratively(self, 
                                  base_prompts: GeneratedPrompts, 
                                  naive_prompt: str,
                                  iterations: int = 3) -> List[GeneratedPrompts]:
        """
        Perform refinement iterations on base prompts.
        
        Args:
            base_prompts: Base prompts to refine
            naive_prompt: Original user prompt
            iterations: Number of refinement iterations to perform
            
        Returns:
            List of GeneratedPrompts objects with refinement iterations
        """
        # Base generation followed by 3 refinement iterations
        all_prompts = [base_prompts]
        
        # Define refinement goals for each iteration
        refinement_goals = [
            "add preservation constraints",
            "increase technical specificity", 
            "add quality/style modifiers"
        ]
        
        # Create both positive and negative prompts
        current_positive = base_prompts.positive_prompt
        current_negative = base_prompts.negative_prompt
        current_quality_score = base_prompts.quality_score
        
        # Iterate through refinement steps
        for i in range(min(iterations, len(refinement_goals))):
            try:
                # Refine prompts based on specific goals
                result = self.refine_prompts(
                    naive_prompt=naive_prompt,
                    previous_positive=current_positive,
                    previous_negative=current_negative,
                    refinement_goal=refinement_goals[i]
                )
                
                # Parse refined results
                refined_positive = result.refined_positive
                refined_negative = result.refined_negative
                improvement_explanation = result.improvement_explanation
                
                # Create refined prompts object
                refined_prompts = GeneratedPrompts(
                    positive_prompt=refined_positive,
                    negative_prompt=refined_negative,
                    iteration=i + 1,
                    quality_score=current_quality_score,  # Would be updated in a real implementation
                    metadata={
                        "generation_method": "refined",
                        "refinement_iteration": i + 1,
                        "refinement_goal": refinement_goals[i],
                        "improvement_explanation": improvement_explanation
                    }
                )
                
                # Update current prompts for next iteration
                current_positive = refined_positive
                current_negative = refined_negative
                
                # Add to collection
                all_prompts.append(refined_prompts)
                
                self.logger.info(f"Refinement iteration {i + 1} completed")
                
            except Exception as e:
                self.logger.error(f"Failed to refine prompts in iteration {i + 1}: {str(e)}")
                # Continue with next iteration if possible
                continue
        
        return all_prompts
    
    def validate_prompts(self, prompts: GeneratedPrompts) -> bool:
        """
        Validate generated prompts.
        
        Args:
            prompts: Generated prompts to validate
            
        Returns:
            Boolean indicating if prompts are valid
        """
        # Validate prompt structure
        if not prompts.positive_prompt or not prompts.positive_prompt.strip():
            self.logger.error("Prompt validation failed: Empty positive prompt")
            return False
        
        if not prompts.negative_prompt or not prompts.negative_prompt.strip():
            self.logger.error("Prompt validation failed: Empty negative prompt")
            return False
        
        # Validate iteration number
        if not isinstance(prompts.iteration, int) or prompts.iteration < 0:
            self.logger.error("Prompt validation failed: Invalid iteration number")
            return False
        
        # Validate quality score if present
        if prompts.quality_score is not None:
            if not (0.0 <= prompts.quality_score <= 1.0):
                self.logger.error("Prompt validation failed: Invalid quality score")
                return False
        
        # Validate generation time if present
        if prompts.generation_time is not None:
            if prompts.generation_time < 0:
                self.logger.error("Prompt validation failed: Invalid generation time")
                return False
        
        # Validate metadata if present
        if prompts.metadata and not isinstance(prompts.metadata, dict):
            self.logger.error("Prompt validation failed: Invalid metadata format")
            return False
        
        self.logger.info("Prompt validation passed")
        return True
    
    def compare_prompt_iterations(self, prompt_iterations: List[GeneratedPrompts]) -> Dict[str, Any]:
        """
        Compare different prompt iterations and provide analysis.
        
        Args:
            prompt_iterations: List of prompt iterations to compare
            
        Returns:
            Dictionary with comparison analysis
        """
        if not prompt_iterations:
            return {"error": "No prompt iterations to compare"}
        
        analysis = {
            "total_iterations": len(prompt_iterations),
            "iterations_with_quality_scores": 0,
            "average_quality_score": 0.0,
            "quality_improvement": False,
            "latest_iteration": prompt_iterations[-1] if prompt_iterations else None,
            "comparison_details": []
        }
        
        # Calculate statistics
        quality_scores = []
        for i, prompts in enumerate(prompt_iterations):
            iteration_detail = {
                "iteration": i,
                "has_quality_score": prompts.quality_score is not None,
                "quality_score": prompts.quality_score,
                "positive_prompt_length": len(prompts.positive_prompt),
                "negative_prompt_length": len(prompts.negative_prompt)
            }
            
            if prompts.quality_score is not None:
                quality_scores.append(prompts.quality_score)
                analysis["iterations_with_quality_scores"] += 1
            
            analysis["comparison_details"].append(iteration_detail)
        
        # Calculate average quality score
        if quality_scores:
            analysis["average_quality_score"] = sum(quality_scores) / len(quality_scores)
        
        # Check for quality improvement
        if len(quality_scores) >= 2:
            analysis["quality_improvement"] = quality_scores[-1] > quality_scores[0]
        
        return analysis

# Example usage
if __name__ == "__main__":
    # Initialize prompt generator
    generator = PromptGenerator()
    
    print("Prompt Generator initialized")
    
    # Example intent and scene analysis
    intent = {
        "naive_prompt": "make the sky more dramatic",
        "target_entities": ["sky_0"],
        "edit_type": "transform",
        "confidence": 0.85,
        "clarifying_questions": [],
        "scene_analysis": {
            "entities": [
                {"id": "sky_0", "label": "sky", "confidence": 0.95},
                {"id": "mountain_1", "label": "mountain", "confidence": 0.87}
            ]
        }
    }
    
    scene_analysis = {
        "entities": [
            {"id": "sky_0", "label": "sky", "confidence": 0.95, "bbox": [0, 0, 1920, 768]},
            {"id": "mountain_1", "label": "mountain", "confidence": 0.87, "bbox": [20, 768, 1900, 1080]},
            {"id": "tree_2", "label": "tree", "confidence": 0.92, "bbox": [100, 800, 200, 1000]}
        ],
        "spatial_layout": "sky occupies top portion, mountains at bottom, trees in foreground",
        "image_description": "landscape with sky, mountains, and trees"
    }
    
    # Generate prompts
    try:
        prompt_iterations = generator.forward(intent, scene_analysis)
        print(f"Generated {len(prompt_iterations)} prompt iterations")
        
        # Display each iteration
        for i, prompts in enumerate(prompt_iterations):
            print(f"\nIteration {i}:")
            print(f"  Positive: {prompts.positive_prompt}")
            print(f"  Negative: {prompts.negative_prompt}")
            print(f"  Quality Score: {prompts.quality_score}")
            print(f"  Metadata: {prompts.metadata}")
        
        # Validate prompts
        for i, prompts in enumerate(prompt_iterations):
            is_valid = generator.validate_prompts(prompts)
            print(f"Iteration {i} validation: {'Passed' if is_valid else 'Failed'}")
        
        # Compare iterations
        comparison = generator.compare_prompt_iterations(prompt_iterations)
        print(f"\nComparison analysis:")
        print(f"  Total iterations: {comparison['total_iterations']}")
        print(f"  Iterations with quality scores: {comparison['iterations_with_quality_scores']}")
        print(f"  Average quality score: {comparison['average_quality_score']:.2f}")
        print(f"  Quality improvement: {comparison['quality_improvement']}")
        
        # Display latest iteration
        if comparison['latest_iteration']:
            latest = comparison['latest_iteration']
            print(f"\nLatest iteration:")
            print(f"  Positive: {latest.positive_prompt}")
            print(f"  Negative: {latest.negative_prompt}")
            print(f"  Quality Score: {latest.quality_score}")
        
    except PromptGenerationError as e:
        print(f"Prompt generation failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    print("Prompt generator example completed")
```

### Advanced Prompt Generation Implementation
Enhanced implementation with multiple strategies and quality assessment:

```python
import dspy
import json
from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel, Field, validator
from enum import Enum
import logging
from dataclasses import dataclass
import asyncio
import time
from datetime import datetime

class AdvancedPromptGenerationError(Exception):
    """Custom exception for advanced prompt generation errors."""
    pass

class AdvancedPromptType(Enum):
    """Enhanced enumeration for prompt types."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class AdvancedPromptStrategy(Enum):
    """Enumeration for different prompt generation strategies."""
    BASELINE = "baseline"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    BALANCED = "balanced"
    CONTEXTUAL = "contextual"

@dataclass
class AdvancedGeneratedPrompts:
    """
    Enhanced data structure for generated prompts with quality metrics.
    """
    positive_prompt: str = Field(..., description="Positive prompt for desired changes")
    negative_prompt: str = Field(..., description="Negative prompt for things to avoid")
    neutral_prompt: Optional[str] = Field(None, description="Neutral prompt for balanced guidance")
    iteration: int = Field(0, description="Iteration number")
    strategy: AdvancedPromptStrategy = Field(AdvancedPromptStrategy.BASELINE, description="Generation strategy used")
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Quality score of the prompts")
    generation_time: Optional[float] = Field(None, ge=0.0, description="Time taken to generate prompts")
    metadata: Optional[Dict[str, Any]] = Field({}, description="Additional metadata about the generation")
    quality_metrics: Optional[Dict[str, Any]] = Field({}, description="Detailed quality metrics")
    refinement_history: Optional[List[Dict[str, Any]]] = Field([], description="History of refinements")
    alternative_prompts: Optional[List[Dict[str, str]]] = Field([], description="Alternative prompt variations")

class MultiStrategyPromptGenerator(dspy.Module):
    """
    Advanced DSpy module for generating prompts using multiple strategies.
    """
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Initialize multiple generation strategies
        self.strategies = {
            "baseline": dspy.ChainOfThought(self._get_baseline_signature()),
            "technical": dspy.ChainOfThought(self._get_technical_signature()),
            "creative": dspy.ChainOfThought(self._get_creative_signature()),
            "balanced": dspy.ChainOfThought(self._get_balanced_signature()),
            "contextual": dspy.ChainOfThought(self._get_contextual_signature())
        }
        
        # Initialize refinement strategies
        self.refinement_strategies = {
            "preservation": dspy.ChainOfThought(self._get_preservation_refinement_signature()),
            "specificity": dspy.ChainOfThought(self._get_specificity_refinement_signature()),
            "quality": dspy.ChainOfThought(self._get_quality_refinement_signature()),
            "style": dspy.ChainOfThought(self._get_style_refinement_signature())
        }
    
    def _get_baseline_signature(self) -> dspy.Signature:
        """Get baseline prompt generation signature."""
        class BaselinePromptSignature(dspy.Signature):
            intent = dspy.InputField(
                desc="JSON of parsed user intent with target entities and edit type"
            )
            scene_analysis = dspy.InputField(
                desc="JSON of detected entities and layout"
            )
            
            positive_prompt = dspy.OutputField(
                desc="Positive prompt describing desired changes"
            )
            negative_prompt = dspy.OutputField(
                desc="Negative prompt describing things to avoid"
            )
            quality_score = dspy.OutputField(
                desc="Float 0-1 indicating quality of generated prompts"
            )
        
        return BaselinePromptSignature
    
    def _get_technical_signature(self) -> dspy.Signature:
        """Get technical prompt generation signature."""
        class TechnicalPromptSignature(dspy.Signature):
            intent = dspy.InputField(
                desc="JSON of parsed user intent with target entities and edit type"
            )
            scene_analysis = dspy.InputField(
                desc="JSON of detected entities and layout"
            )
            technical_requirements = dspy.InputField(
                desc="Technical requirements for image processing"
            )
            
            positive_prompt = dspy.OutputField(
                desc="Technical positive prompt with precise parameters"
            )
            negative_prompt = dspy.OutputField(
                desc="Technical negative prompt with avoidance parameters"
            )
            quality_score = dspy.OutputField(
                desc="Float 0-1 indicating technical quality of generated prompts"
            )
            technical_specifications = dspy.OutputField(
                desc="JSON of technical specifications used in prompts"
            )
        
        return TechnicalPromptSignature
    
    def _get_creative_signature(self) -> dspy.Signature:
        """Get creative prompt generation signature."""
        class CreativePromptSignature(dspy.Signature):
            intent = dspy.InputField(
                desc="JSON of parsed user intent with target entities and edit type"
            )
            scene_analysis = dspy.InputField(
                desc="JSON of detected entities and layout"
            )
            creative_direction = dspy.InputField(
                desc="Creative direction for artistic enhancement"
            )
            
            positive_prompt = dspy.OutputField(
                desc="Creative positive prompt with artistic elements"
            )
            negative_prompt = dspy.OutputField(
                desc="Creative negative prompt with artistic constraints"
            )
            quality_score = dspy.OutputField(
                desc="Float 0-1 indicating creative quality of generated prompts"
            )
            artistic_elements = dspy.OutputField(
                desc="JSON of artistic elements incorporated"
            )
        
        return CreativePromptSignature
    
    def _get_balanced_signature(self) -> dspy.Signature:
        """Get balanced prompt generation signature."""
        class BalancedPromptSignature(dspy.Signature):
            intent = dspy.InputField(
                desc="JSON of parsed user intent with target entities and edit type"
            )
            scene_analysis = dspy.InputField(
                desc="JSON of detected entities and layout"
            )
            user_preferences = dspy.InputField(
                desc="User preferences for balanced editing"
            )
            
            positive_prompt = dspy.OutputField(
                desc="Balanced positive prompt combining technical and creative elements"
            )
            negative_prompt = dspy.OutputField(
                desc="Balanced negative prompt avoiding extremes"
            )
            neutral_prompt = dspy.OutputField(
                desc="Neutral prompt for guiding balanced outcomes"
            )
            quality_score = dspy.OutputField(
                desc="Float 0-1 indicating balance quality of generated prompts"
            )
            balance_factors = dspy.OutputField(
                desc="JSON of balance factors considered"
            )
        
        return BalancedPromptSignature
    
    def _get_contextual_signature(self) -> dspy.Signature:
        """Get contextual prompt generation signature."""
        class ContextualPromptSignature(dspy.Signature):
            intent = dspy.InputField(
                desc="JSON of parsed user intent with target entities and edit type"
            )
            scene_analysis = dspy.InputField(
                desc="JSON of detected entities and layout"
            )
            context_information = dspy.InputField(
                desc="Contextual information about the image and user"
            )
            
            positive_prompt = dspy.OutputField(
                desc="Context-aware positive prompt considering situational factors"
            )
            negative_prompt = dspy.OutputField(
                desc="Context-aware negative prompt avoiding contextual conflicts"
            )
            quality_score = dspy.OutputField(
                desc="Float 0-1 indicating contextual relevance of generated prompts"
            )
            context_factors = dspy.OutputField(
                desc="JSON of contextual factors considered"
            )
        
        return ContextualPromptSignature
    
    def _get_preservation_refinement_signature(self) -> dspy.Signature:
        """Get preservation refinement signature."""
        class PreservationRefinementSignature(dspy.Signature):
            naive_prompt = dspy.InputField(
                desc="Original user prompt"
            )
            previous_positive = dspy.InputField(
                desc="Previous positive prompt to refine"
            )
            previous_negative = dspy.InputField(
                desc="Previous negative prompt to refine"
            )
            scene_analysis = dspy.InputField(
                desc="JSON of detected entities and layout for preservation guidance"
            )
            
            refined_positive = dspy.OutputField(
                desc="Refined positive prompt with preservation constraints"
            )
            refined_negative = dspy.OutputField(
                desc="Refined negative prompt with preservation guidance"
            )
            improvement_explanation = dspy.OutputField(
                desc="Explanation of preservation improvements made"
            )
            preservation_score = dspy.OutputField(
                desc="Float 0-1 indicating effectiveness of preservation constraints"
            )
        
        return PreservationRefinementSignature
    
    def _get_specificity_refinement_signature(self) -> dspy.Signature:
        """Get specificity refinement signature."""
        class SpecificityRefinementSignature(dspy.Signature):
            naive_prompt = dspy.InputField(
                desc="Original user prompt"
            )
            previous_positive = dspy.InputField(
                desc="Previous positive prompt to refine"
            )
            previous_negative = dspy.InputField(
                desc="Previous negative prompt to refine"
            )
            technical_requirements = dspy.InputField(
                desc="Technical requirements for increased specificity"
            )
            
            refined_positive = dspy.OutputField(
                desc="Refined positive prompt with increased technical specificity"
            )
            refined_negative = dspy.OutputField(
                desc="Refined negative prompt with technical avoidance parameters"
            )
            improvement_explanation = dspy.OutputField(
                desc="Explanation of specificity improvements made"
            )
            specificity_score = dspy.OutputField(
                desc="Float 0-1 indicating level of technical specificity"
            )
        
        return SpecificityRefinementSignature
    
    def _get_quality_refinement_signature(self) -> dspy.Signature:
        """Get quality refinement signature."""
        class QualityRefinementSignature(dspy.Signature):
            naive_prompt = dspy.InputField(
                desc="Original user prompt"
            )
            previous_positive = dspy.InputField(
                desc="Previous positive prompt to refine"
            )
            previous_negative = dspy.InputField(
                desc="Previous negative prompt to refine"
            )
            quality_guidelines = dspy.InputField(
                desc="Quality guidelines for enhancement"
            )
            
            refined_positive = dspy.OutputField(
                desc="Refined positive prompt with quality/style modifiers"
            )
            refined_negative = dspy.OutputField(
                desc="Refined negative prompt with quality constraints"
            )
            improvement_explanation = dspy.OutputField(
                desc="Explanation of quality improvements made"
            )
            quality_score = dspy.OutputField(
                desc="Float 0-1 indicating overall quality enhancement"
            )
        
        return QualityRefinementSignature
    
    def _get_style_refinement_signature(self) -> dspy.Signature:
        """Get style refinement signature."""
        class StyleRefinementSignature(dspy.Signature):
            naive_prompt = dspy.InputField(
                desc="Original user prompt"
            )
            previous_positive = dspy.InputField(
                desc="Previous positive prompt to refine"
            )
            previous_negative = dspy.InputField(
                desc="Previous negative prompt to refine"
            )
            style_preferences = dspy.InputField(
                desc="Style preferences for artistic enhancement"
            )
            
            refined_positive = dspy.OutputField(
                desc="Refined positive prompt with style enhancements"
            )
            refined_negative = dspy.OutputField(
                desc="Refined negative prompt with style constraints"
            )
            improvement_explanation = dspy.OutputField(
                desc="Explanation of style improvements made"
            )
            style_score = dspy.OutputField(
                desc="Float 0-1 indicating artistic style enhancement"
            )
        
        return StyleRefinementSignature
    
    def forward(self, 
                intent: Dict[str, Any], 
                scene: Dict[str, Any],
                strategies: List[AdvancedPromptStrategy] = None,
                refinement_iterations: int = 3) -> List[AdvancedGeneratedPrompts]:
        """
        Generates initial prompts based on intent and scene analysis using multiple strategies.
        
        Args:
            intent: Parsed user intent with target entities and edit type
            scene: Scene analysis data with detected entities and layout
            strategies: List of strategies to use (defaults to all)
            refinement_iterations: Number of refinement iterations to perform
            
        Returns:
            List of AdvancedGeneratedPrompts objects with refinement iterations
        """
        # Validate inputs
        if not intent:
            raise AdvancedPromptGenerationError("Intent cannot be empty")
        
        if not scene:
            raise AdvancedPromptGenerationError("Scene analysis cannot be empty")
        
        self.logger.info("Starting advanced prompt generation process")
        
        # Use all strategies if none specified
        if strategies is None:
            strategies = list(AdvancedPromptStrategy)
        
        # Generate prompts using multiple strategies
        all_prompts = []
        
        for strategy in strategies:
            try:
                strategy_name = strategy.value
                if strategy_name in self.strategies:
                    # Generate base prompts using strategy
                    base_prompts = self._generate_base_prompts_with_strategy(
                        intent, scene, strategy_name
                    )
                    
                    # Perform refinement iterations
                    refined_prompts = self._refine_prompts_with_strategies(
                        base_prompts, intent.get("naive_prompt", ""), refinement_iterations
                    )
                    
                    # Add to collection
                    all_prompts.extend(refined_prompts)
                    
                    self.logger.info(f"Prompt generation completed for strategy: {strategy_name}")
                else:
                    self.logger.warning(f"Unknown strategy: {strategy_name}")
                    
            except Exception as e:
                self.logger.error(f"Failed to generate prompts for strategy {strategy.value}: {str(e)}")
                # Continue with other strategies
        
        self.logger.info(f"Advanced prompt generation completed with {len(all_prompts)} prompt variations")
        return all_prompts
    
    def _generate_base_prompts_with_strategy(self, 
                                          intent: Dict[str, Any], 
                                          scene: Dict[str, Any],
                                          strategy: str) -> AdvancedGeneratedPrompts:
        """
        Generate base prompts using a specific strategy.
        
        Args:
            intent: Parsed user intent with target entities and edit type
            scene: Scene analysis data with detected entities and layout
            strategy: Strategy to use for generation
            
        Returns:
            AdvancedGeneratedPrompts object with base prompts
        """
        # Get the appropriate strategy generator
        if strategy not in self.strategies:
            raise AdvancedPromptGenerationError(f"Unknown strategy: {strategy}")
        
        generator = self.strategies[strategy]
        
        # Measure generation time
        start_time = time.time()
        
        try:
            # Process intent and scene analysis to create base prompts
            if strategy == "baseline":
                result = generator(
                    intent=json.dumps(intent, default=str),
                    scene_analysis=json.dumps(scene, default=str)
                )
            elif strategy == "technical":
                result = generator(
                    intent=json.dumps(intent, default=str),
                    scene_analysis=json.dumps(scene, default=str),
                    technical_requirements=json.dumps({
                        "max_resolution": "4096x4096",
                        "color_depth": "24-bit",
                        "compression": "lossless"
                    }, default=str)
                )
            elif strategy == "creative":
                result = generator(
                    intent=json.dumps(intent, default=str),
                    scene_analysis=json.dumps(scene, default=str),
                    creative_direction=json.dumps({
                        "artistic_style": "photorealistic",
                        "mood": "dramatic",
                        "lighting": "natural"
                    }, default=str)
                )
            elif strategy == "balanced":
                result = generator(
                    intent=json.dumps(intent, default=str),
                    scene_analysis=json.dumps(scene, default=str),
                    user_preferences=json.dumps({
                        "balance_preference": "technical_focus",
                        "creativity_level": "moderate",
                        "preservation_priority": "high"
                    }, default=str)
                )
            elif strategy == "contextual":
                result = generator(
                    intent=json.dumps(intent, default=str),
                    scene_analysis=json.dumps(scene, default=str),
                    context_information=json.dumps({
                        "image_type": "photograph",
                        "subject_matter": "landscape",
                        "intended_use": "personal"
                    }, default=str)
                )
            else:
                raise AdvancedPromptGenerationError(f"Unsupported strategy: {strategy}")
            
            # Calculate generation time
            generation_time = time.time() - start_time
            
            # Parse results
            positive_prompt = result.positive_prompt
            negative_prompt = result.negative_prompt
            neutral_prompt = getattr(result, 'neutral_prompt', None)
            
            # Parse quality score
            try:
                quality_score = float(result.quality_score) if result.quality_score else None
            except (ValueError, TypeError):
                quality_score = None
            
            # Parse additional outputs
            technical_specifications = getattr(result, 'technical_specifications', None)
            artistic_elements = getattr(result, 'artistic_elements', None)
            balance_factors = getattr(result, 'balance_factors', None)
            context_factors = getattr(result, 'context_factors', None)
            
            # Create quality metrics
            quality_metrics = {
                "overall_score": quality_score,
                "technical_specifications": technical_specifications,
                "artistic_elements": artistic_elements,
                "balance_factors": balance_factors,
                "context_factors": context_factors,
                "generation_time": generation_time
            }
            
            # Create base prompts object
            base_prompts = AdvancedGeneratedPrompts(
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                neutral_prompt=neutral_prompt,
                iteration=0,
                strategy=AdvancedPromptStrategy(strategy),
                quality_score=quality_score,
                generation_time=generation_time,
                metadata={
                    "generation_method": "base",
                    "strategy_used": strategy,
                    "intent_used": intent,
                    "scene_used": scene
                },
                quality_metrics=quality_metrics
            )
            
            self.logger.info(f"Base prompts generated successfully using strategy: {strategy}")
            return base_prompts
            
        except Exception as e:
            self.logger.error(f"Failed to generate base prompts using strategy {strategy}: {str(e)}")
            raise AdvancedPromptGenerationError(f"Base prompt generation failed for strategy {strategy}: {str(e)}")
    
    def _refine_prompts_with_strategies(self, 
                                      base_prompts: AdvancedGeneratedPrompts, 
                                      naive_prompt: str,
                                      iterations: int = 3) -> List[AdvancedGeneratedPrompts]:
        """
        Perform refinement iterations on base prompts using multiple strategies.
        
        Args:
            base_prompts: Base prompts to refine
            naive_prompt: Original user prompt
            iterations: Number of refinement iterations to perform
            
        Returns:
            List of AdvancedGeneratedPrompts objects with refinement iterations
        """
        # Base generation followed by refinement iterations
        all_prompts = [base_prompts]
        
        # Create both positive and negative prompts
        current_positive = base_prompts.positive_prompt
        current_negative = base_prompts.negative_prompt
        current_neutral = base_prompts.neutral_prompt
        current_quality_score = base_prompts.quality_score
        current_strategy = base_prompts.strategy
        
        # Define refinement strategies to use
        refinement_strategy_names = list(self.refinement_strategies.keys())
        
        # Iterate through refinement steps
        refinement_history = []
        for i in range(iterations):
            try:
                # Select refinement strategy (cycle through available strategies)
                strategy_name = refinement_strategy_names[i % len(refinement_strategy_names)]
                refiner = self.refinement_strategies[strategy_name]
                
                # Perform refinement
                if strategy_name == "preservation":
                    result = refiner(
                        naive_prompt=naive_prompt,
                        previous_positive=current_positive,
                        previous_negative=current_negative,
                        scene_analysis=json.dumps({"entities": [], "layout": ""}, default=str)  # Would be actual scene data
                    )
                elif strategy_name == "specificity":
                    result = refiner(
                        naive_prompt=naive_prompt,
                        previous_positive=current_positive,
                        previous_negative=current_negative,
                        technical_requirements=json.dumps({
                            "precision": "high",
                            "parameters": ["color", "contrast", "sharpness"]
                        }, default=str)
                    )
                elif strategy_name == "quality":
                    result = refiner(
                        naive_prompt=naive_prompt,
                        previous_positive=current_positive,
                        previous_negative=current_negative,
                        quality_guidelines=json.dumps({
                            "focus": "overall_image_quality",
                            "emphasis": "natural_appearance"
                        }, default=str)
                    )
                elif strategy_name == "style":
                    result = refiner(
                        naive_prompt=naive_prompt,
                        previous_positive=current_positive,
                        previous_negative=current_negative,
                        style_preferences=json.dumps({
                            "preferred_styles": ["photorealistic", "dramatic"],
                            "avoided_styles": ["cartoon", "abstract"]
                        }, default=str)
                    )
                else:
                    raise AdvancedPromptGenerationError(f"Unsupported refinement strategy: {strategy_name}")
                
                # Parse refined results
                refined_positive = result.refined_positive
                refined_negative = result.refined_negative
                improvement_explanation = result.improvement_explanation
                
                # Parse quality scores
                try:
                    preservation_score = float(getattr(result, 'preservation_score', 0.0))
                    specificity_score = float(getattr(result, 'specificity_score', 0.0))
                    quality_score = float(getattr(result, 'quality_score', 0.0))
                    style_score = float(getattr(result, 'style_score', 0.0))
                except (ValueError, TypeError):
                    preservation_score = specificity_score = quality_score = style_score = 0.0
                
                # Create refinement record
                refinement_record = {
                    "iteration": i + 1,
                    "strategy": strategy_name,
                    "improvement_explanation": improvement_explanation,
                    "quality_scores": {
                        "preservation": preservation_score,
                        "specificity": specificity_score,
                        "quality": quality_score,
                        "style": style_score
                    }
                }
                
                refinement_history.append(refinement_record)
                
                # Create refined prompts object
                refined_prompts = AdvancedGeneratedPrompts(
                    positive_prompt=refined_positive,
                    negative_prompt=refined_negative,
                    neutral_prompt=current_neutral,  # Maintain neutral prompt
                    iteration=i + 1,
                    strategy=current_strategy,
                    quality_score=quality_score,
                    metadata={
                        "generation_method": "refined",
                        "refinement_iteration": i + 1,
                        "refinement_strategy": strategy_name,
                        "improvement_explanation": improvement_explanation
                    },
                    quality_metrics={
                        "overall_score": quality_score,
                        "preservation_score": preservation_score,
                        "specificity_score": specificity_score,
                        "quality_score": quality_score,
                        "style_score": style_score
                    },
                    refinement_history=refinement_history
                )
                
                # Update current prompts for next iteration
                current_positive = refined_positive
                current_negative = refined_negative
                
                # Add to collection
                all_prompts.append(refined_prompts)
                
                self.logger.info(f"Refinement iteration {i + 1} completed using strategy: {strategy_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to refine prompts in iteration {i + 1}: {str(e)}")
                # Continue with next iteration if possible
                continue
        
        return all_prompts
    
    def validate_prompts(self, prompts: AdvancedGeneratedPrompts) -> bool:
        """
        Validate generated prompts with advanced checks.
        
        Args:
            prompts: Generated prompts to validate
            
        Returns:
            Boolean indicating if prompts are valid
        """
        # Validate prompt structure
        if not prompts.positive_prompt or not prompts.positive_prompt.strip():
            self.logger.error("Prompt validation failed: Empty positive prompt")
            return False
        
        if not prompts.negative_prompt or not prompts.negative_prompt.strip():
            self.logger.error("Prompt validation failed: Empty negative prompt")
            return False
        
        # Validate neutral prompt if provided
        if prompts.neutral_prompt is not None and not prompts.neutral_prompt.strip():
            self.logger.error("Prompt validation failed: Empty neutral prompt")
            return False
        
        # Validate iteration number
        if not isinstance(prompts.iteration, int) or prompts.iteration < 0:
            self.logger.error("Prompt validation failed: Invalid iteration number")
            return False
        
        # Validate strategy
        if not isinstance(prompts.strategy, AdvancedPromptStrategy):
            self.logger.error("Prompt validation failed: Invalid strategy")
            return False
        
        # Validate quality score if present
        if prompts.quality_score is not None:
            if not (0.0 <= prompts.quality_score <= 1.0):
                self.logger.error("Prompt validation failed: Invalid quality score")
                return False
        
        # Validate generation time if present
        if prompts.generation_time is not None:
            if prompts.generation_time < 0:
                self.logger.error("Prompt validation failed: Invalid generation time")
                return False
        
        # Validate metadata if present
        if prompts.metadata and not isinstance(prompts.metadata, dict):
            self.logger.error("Prompt validation failed: Invalid metadata format")
            return False
        
        # Validate quality metrics if present
        if prompts.quality_metrics and not isinstance(prompts.quality_metrics, dict):
            self.logger.error("Prompt validation failed: Invalid quality metrics format")
            return False
        
        # Validate refinement history if present
        if prompts.refinement_history and not isinstance(prompts.refinement_history, list):
            self.logger.error("Prompt validation failed: Invalid refinement history format")
            return False
        
        # Validate alternative prompts if present
        if prompts.alternative_prompts and not isinstance(prompts.alternative_prompts, list):
            self.logger.error("Prompt validation failed: Invalid alternative prompts format")
            return False
        
        self.logger.info("Advanced prompt validation passed")
        return True
    
    def compare_prompt_strategies(self, 
                               prompt_variations: List[AdvancedGeneratedPrompts]) -> Dict[str, Any]:
        """
        Compare different prompt strategies and provide analysis.
        
        Args:
            prompt_variations: List of prompt variations to compare
            
        Returns:
            Dictionary with comparison analysis
        """
        if not prompt_variations:
            return {"error": "No prompt variations to compare"}
        
        analysis = {
            "total_variations": len(prompt_variations),
            "strategies_used": {},
            "iterations_per_strategy": {},
            "quality_scores": [],
            "average_quality_score": 0.0,
            "best_quality_prompt": None,
            "best_strategy": None,
            "performance_metrics": {},
            "comparison_details": []
        }
        
        # Analyze each variation
        best_quality_score = -1
        best_prompt = None
        best_strategy = None
        
        for i, prompts in enumerate(prompt_variations):
            # Track strategies used
            strategy_name = prompts.strategy.value
            if strategy_name not in analysis["strategies_used"]:
                analysis["strategies_used"][strategy_name] = 0
            analysis["strategies_used"][strategy_name] += 1
            
            # Track iterations per strategy
            if strategy_name not in analysis["iterations_per_strategy"]:
                analysis["iterations_per_strategy"][strategy_name] = []
            analysis["iterations_per_strategy"][strategy_name].append(prompts.iteration)
            
            # Track quality scores
            if prompts.quality_score is not None:
                analysis["quality_scores"].append(prompts.quality_score)
                if prompts.quality_score > best_quality_score:
                    best_quality_score = prompts.quality_score
                    best_prompt = prompts
                    best_strategy = strategy_name
            
            # Track performance metrics
            if prompts.generation_time is not None:
                if "generation_times" not in analysis["performance_metrics"]:
                    analysis["performance_metrics"]["generation_times"] = []
                analysis["performance_metrics"]["generation_times"].append(prompts.generation_time)
            
            # Add variation details
            variation_detail = {
                "index": i,
                "strategy": strategy_name,
                "iteration": prompts.iteration,
                "has_quality_score": prompts.quality_score is not None,
                "quality_score": prompts.quality_score,
                "positive_prompt_length": len(prompts.positive_prompt),
                "negative_prompt_length": len(prompts.negative_prompt),
                "generation_time": prompts.generation_time,
                "has_neutral_prompt": prompts.neutral_prompt is not None
            }
            
            analysis["comparison_details"].append(variation_detail)
        
        # Calculate averages
        if analysis["quality_scores"]:
            analysis["average_quality_score"] = sum(analysis["quality_scores"]) / len(analysis["quality_scores"])
        
        # Calculate performance metrics
        if "generation_times" in analysis["performance_metrics"]:
            gen_times = analysis["performance_metrics"]["generation_times"]
            if gen_times:
                analysis["performance_metrics"]["average_generation_time"] = sum(gen_times) / len(gen_times)
                analysis["performance_metrics"]["min_generation_time"] = min(gen_times)
                analysis["performance_metrics"]["max_generation_time"] = max(gen_times)
        
        # Set best results
        analysis["best_quality_prompt"] = best_prompt
        analysis["best_strategy"] = best_strategy
        analysis["best_quality_score"] = best_quality_score
        
        return analysis
    
    def generate_alternative_prompts(self, 
                                   base_prompts: AdvancedGeneratedPrompts,
                                   variations: int = 3) -> List[AdvancedGeneratedPrompts]:
        """
        Generate alternative variations of prompts.
        
        Args:
            base_prompts: Base prompts to generate variations from
            variations: Number of variations to generate
            
        Returns:
            List of alternative prompt variations
        """
        alternative_prompts = []
        
        # Generate variations
        for i in range(variations):
            try:
                # Create slight variations of the prompts
                # This would use DSpy to generate variations in a real implementation
                
                # For this example, we'll create simple variations
                variation_positive = f"{base_prompts.positive_prompt} (variation {i+1})"
                variation_negative = f"{base_prompts.negative_prompt} (variation {i+1})"
                
                # Create alternative prompt
                alt_prompt = AdvancedGeneratedPrompts(
                    positive_prompt=variation_positive,
                    negative_prompt=variation_negative,
                    neutral_prompt=base_prompts.neutral_prompt,
                    iteration=base_prompts.iteration,
                    strategy=base_prompts.strategy,
                    quality_score=base_prompts.quality_score,
                    generation_time=base_prompts.generation_time,
                    metadata={
                        **base_prompts.metadata,
                        "variation_number": i + 1,
                        "variation_type": "alternative"
                    },
                    quality_metrics=base_prompts.quality_metrics,
                    refinement_history=base_prompts.refinement_history,
                    alternative_prompts=[]  # Empty for this variation
                )
                
                alternative_prompts.append(alt_prompt)
                
            except Exception as e:
                self.logger.error(f"Failed to generate alternative prompt {i+1}: {str(e)}")
                continue
        
        # Add alternative prompts to base prompts
        base_prompts.alternative_prompts = [
            {
                "positive": alt.positive_prompt,
                "negative": alt.negative_prompt,
                "strategy": alt.strategy.value,
                "quality_score": alt.quality_score
            }
            for alt in alternative_prompts
        ]
        
        return alternative_prompts

# Example usage
if __name__ == "__main__":
    # Initialize advanced prompt generator
    generator = MultiStrategyPromptGenerator()
    
    print("Advanced Prompt Generator initialized")
    
    # Example intent and scene analysis
    intent = {
        "naive_prompt": "make the sky more dramatic",
        "target_entities": ["sky_0"],
        "edit_type": "transform",
        "confidence": 0.85,
        "clarifying_questions": [],
        "scene_analysis": {
            "entities": [
                {"id": "sky_0", "label": "sky", "confidence": 0.95},
                {"id": "mountain_1", "label": "mountain", "confidence": 0.87}
            ]
        }
    }
    
    scene_analysis = {
        "entities": [
            {"id": "sky_0", "label": "sky", "confidence": 0.95, "bbox": [0, 0, 1920, 768]},
            {"id": "mountain_1", "label": "mountain", "confidence": 0.87, "bbox": [20, 768, 1900, 1080]},
            {"id": "tree_2", "label": "tree", "confidence": 0.92, "bbox": [100, 800, 200, 1000]}
        ],
        "spatial_layout": "sky occupies top portion, mountains at bottom, trees in foreground",
        "image_description": "landscape with sky, mountains, and trees"
    }
    
    # Generate prompts using multiple strategies
    try:
        prompt_variations = generator.forward(
            intent, 
            scene_analysis,
            strategies=[AdvancedPromptStrategy.BASELINE, AdvancedPromptStrategy.TECHNICAL],
            refinement_iterations=2
        )
        print(f"Generated {len(prompt_variations)} prompt variations")
        
        # Display each variation
        for i, prompts in enumerate(prompt_variations):
            print(f"\nVariation {i+1}:")
            print(f"  Strategy: {prompts.strategy.value}")
            print(f"  Iteration: {prompts.iteration}")
            print(f"  Positive: {prompts.positive_prompt}")
            print(f"  Negative: {prompts.negative_prompt}")
            print(f"  Neutral: {prompts.neutral_prompt}")
            print(f"  Quality Score: {prompts.quality_score}")
            print(f"  Generation Time: {prompts.generation_time:.2f}s")
            print(f"  Metadata: {prompts.metadata}")
            print(f"  Quality Metrics: {prompts.quality_metrics}")
        
        # Validate prompts
        for i, prompts in enumerate(prompt_variations):
            is_valid = generator.validate_prompts(prompts)
            print(f"Variation {i+1} validation: {'Passed' if is_valid else 'Failed'}")
        
        # Compare strategies
        comparison = generator.compare_prompt_strategies(prompt_variations)
        print(f"\nStrategy comparison:")
        print(f"  Total variations: {comparison['total_variations']}")
        print(f"  Strategies used: {comparison['strategies_used']}")
        print(f"  Average quality score: {comparison['average_quality_score']:.2f}")
        print(f"  Best strategy: {comparison['best_strategy']}")
        print(f"  Best quality score: {comparison['best_quality_score']:.2f}")
        
        # Display best prompt
        if comparison['best_quality_prompt']:
            best = comparison['best_quality_prompt']
            print(f"\nBest prompt variation:")
            print(f"  Strategy: {best.strategy.value}")
            print(f"  Positive: {best.positive_prompt}")
            print(f"  Negative: {best.negative_prompt}")
            print(f"  Quality Score: {best.quality_score}")
        
        # Generate alternative prompts
        if prompt_variations:
            base_prompts = prompt_variations[0]
            alternatives = generator.generate_alternative_prompts(base_prompts, variations=2)
            print(f"\nGenerated {len(alternatives)} alternative prompts:")
            for i, alt in enumerate(alternatives):
                print(f"  Alternative {i+1}:")
                print(f"    Positive: {alt.positive_prompt}")
                print(f"    Negative: {alt.negative_prompt}")
                print(f"    Quality Score: {alt.quality_score}")
        
    except AdvancedPromptGenerationError as e:
        print(f"Advanced prompt generation failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    print("Advanced prompt generator example completed")
```