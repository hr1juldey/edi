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