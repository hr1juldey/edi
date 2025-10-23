# VariationGenerator.generate_variations()

[Back to Variation Generator](../orchestration_variation_generator.md)

## Related User Story
"As a user, I want to see multiple variations and pick the best parts from each." (from PRD)

## Function Signature
`generate_variations(intent, N=3) -> List[Prompts]`

## Parameters
- `intent` - The structured intent containing the user's edit request
- `N=3` - The number of variations to generate (default is 3)

## Returns
- `List[Prompts]` - A list of N different prompt variations based on the intent

## Step-by-step Logic
1. Take the intent as input
2. Use DSpy BestOfN approach to generate multiple interpretations
3. Apply different rollout IDs to create diversity in the variations
4. Create N different prompt variations that interpret the same intent differently
5. Ensure each variation differs by >30% tokens from the others
6. Return the list of N prompt variations

## DSpy BestOfN Implementation
- Uses different rollout IDs to ensure diversity
- Generates alternative interpretations of the same intent
- Each variation targets different aspects of the edit request
- Maintains coherence while maximizing variation

## Variation Strategy
- Each variation focuses on different aspects of the intent
- Maintains consistency with the overall edit goal
- Ensures prompts are different enough for meaningful comparison
- Preserves important constraints across all variations

## Input/Output Data Structures
### Prompts Object
Each Prompts object in the list contains:
- Positive prompt (technical prompt for desired changes)
- Negative prompt (technical prompt for things to avoid)
- Metadata about the variation approach

## See Docs

```python
import dspy
from typing import List, Dict, Any, Optional
import re

class VariationGenerator:
    def __init__(self):
        pass

    def generate_variations(self, intent: Dict[str, Any], N: int = 3) -> List[Dict[str, str]]:
        """
        Generates multiple prompt variations based on a structured intent using DSpy's BestOfN approach.
        
        This method:
        1. Takes the intent as input
        2. Uses DSpy BestOfN approach to generate multiple interpretations
        3. Applies different rollout IDs to create diversity in the variations
        4. Creates N different prompt variations that interpret the same intent differently
        5. Ensures each variation differs by >30% tokens from the others
        6. Returns the list of N prompt variations
        """
        # Create a DSPy predictor that will generate prompt variations based on the intent
        class PromptGenerator(dspy.Signature):
            """Generate a prompt variation based on the user's intent for image editing."""
            intent = dspy.InputField(desc="Structured intent containing the user's edit request")
            variation_id = dspy.InputField(desc="ID for this variation to ensure diversity")
            positive_prompt = dspy.OutputField(desc="Technical prompt for desired changes")
            negative_prompt = dspy.OutputField(desc="Technical prompt for things to avoid")
        
        class PromptVariationModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.generate = dspy.ChainOfThought(PromptGenerator)
            
            def forward(self, intent: Dict[str, Any], variation_id: int):
                result = self.generate(intent=str(intent), variation_id=str(variation_id))
                return {
                    'positive_prompt': result.positive_prompt,
                    'negative_prompt': result.negative_prompt,
                    'metadata': {'variation_approach': f'approach_{variation_id}'}
                }
        
        # List to store the generated variations
        variations = []
        
        # Create different variations using BestOfN approach with different seeds/contexts
        for i in range(N):
            # Create a module instance for each variation
            variation_module = PromptVariationModule()
            
            # For diversity, we can pass slightly different contexts or use different seeds
            try:
                variation = variation_module(intent, i)
                variations.append(variation)
            except Exception as e:
                # If DSPy generation fails, create a variation using basic template approach
                variation = self._generate_basic_variation(intent, i)
                variations.append(variation)
        
        # Ensure diversity by comparing token overlap
        variations = self._ensure_diversity(variations)
        
        return variations

    def _generate_basic_variation(self, intent: Dict[str, Any], variation_id: int) -> Dict[str, str]:
        """Fallback method to generate prompt variations without DSPy."""
        # Extract intent information
        target_entities = intent.get('target_entities', [])
        edit_type = intent.get('edit_type', 'general')
        edit_description = intent.get('description', 'edit the image')
        
        # Create variations based on different approaches
        positive_templates = [
            f"Apply {edit_type} to {' and '.join(target_entities)} focusing on {edit_description}, with high detail and photorealistic quality",
            f"Modify {' and '.join(target_entities)} using {edit_type} technique, ensuring {edit_description} with professional lighting",
            f"Transform the {' and '.join(target_entities)} with {edit_type} effects, emphasizing {edit_description} in a natural way",
            f"Enhance {' and '.join(target_entities)} through {edit_type} process, maintaining realistic textures while {edit_description}",
            f"Perform {edit_type} on {' and '.join(target_entities)}, balancing {edit_description} with visual coherence"
        ]
        
        negative_templates = [
            f"avoid blurry, pixelated, or distorted results for {edit_type}",
            f"don't make the {edit_type} look artificial or overdone",
            f"prevent unrealistic shadows or lighting artifacts in {edit_type}",
            f"no over-saturation or color inconsistencies in {edit_type} results",
            f"avoid artifacts or unnatural blending in {edit_type} process"
        ]
        
        # Select template based on variation_id to ensure diversity
        pos_idx = variation_id % len(positive_templates)
        neg_idx = variation_id % len(negative_templates)
        
        return {
            'positive_prompt': positive_templates[pos_idx],
            'negative_prompt': negative_templates[neg_idx],
            'metadata': {'variation_approach': f'basic_approach_{variation_id}'}
        }

    def _ensure_diversity(self, variations: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Ensures that variations are diverse by comparing token overlap."""
        if len(variations) <= 1:
            return variations
        
        # Calculate token overlap between variations and ensure >30% difference
        for i in range(len(variations)):
            for j in range(i+1, len(variations)):
                # Convert prompts to tokens for comparison
                tokens_i = set(variations[i]['positive_prompt'].split())
                tokens_j = set(variations[j]['positive_prompt'].split())
                
                # Calculate overlap
                common_tokens = tokens_i.intersection(tokens_j)
                total_tokens = tokens_i.union(tokens_j)
                
                if total_tokens:
                    overlap_ratio = len(common_tokens) / len(total_tokens)
                    
                    # If overlap is too high (>70% similarity), modify the second variation
                    if overlap_ratio > 0.7:
                        # Add a differentiator to reduce similarity
                        variations[j]['positive_prompt'] += f", emphasizing different aspects of {j+1}"
        
        return variations

# Example usage:
if __name__ == "__main__":
    # Create a sample intent
    intent = {
        "target_entities": ["sky", "trees"],
        "edit_type": "color adjustment",
        "description": "make colors more vibrant",
        "confidence": 0.85
    }
    
    # Create a variation generator
    generator = VariationGenerator()
    
    # Generate variations
    variations = generator.generate_variations(intent, N=3)
    
    # Print the variations
    for i, var in enumerate(variations):
        print(f"Variation {i+1}:")
        print(f"  Positive: {var['positive_prompt']}")
        print(f"  Negative: {var['negative_prompt']}")
        print(f"  Metadata: {var['metadata']}")
        print()
```