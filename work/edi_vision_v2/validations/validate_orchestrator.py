#!/usr/bin/env python3
"""Validation script for Stage 7: Orchestrator"""

import logging
import sys
import os
import json

# Add the parent directory to the path so we can import from pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.orchestrator import VisionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    print("="*60)
    print("STAGE 7: ORCHESTRATOR VALIDATION")
    print("="*60)

    # Create pipeline
    pipeline = VisionPipeline(
        enable_validation=True,
        save_intermediate=True,
        output_dir="logs/orchestrator"
    )

    # Test cases
    test_cases = [
        {
            'image': 'test_image.jpeg',
            'prompt': 'change blue tin roofs to green'
        },
        {
            'image': 'test_image.jpeg',
            'prompt': 'turn the blue roofs of buildings to green'
        }
    ]

    for idx, test in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"TEST CASE {idx+1}: {test['prompt']}")
        print(f"{'='*60}")

        result = pipeline.process(
            image_path=test['image'],
            user_prompt=test['prompt']
        )

        if result['success']:
            print("✅ SUCCESS")
            print("\nResults:")
            print(f"  - Entity masks: {len(result['entity_masks'])}")
            print(f"  - Total time: {result['metadata']['total_time']:.2f}s")

            print("\nStage Timings:")
            for stage, timing in result['stage_timings'].items():
                print(f"  - {stage}: {timing:.3f}s")

            if result['validation']:
                print("\nVLM Validation:")
                print(f"  - Confidence: {result['validation'].confidence:.2f}")
                print(f"  - Covers all targets: {result['validation'].covers_all_targets}")
                print(f"  - Feedback: {result['validation'].feedback}")

            # Save result as JSON
            output_file = f"logs/orchestrator/test_case_{idx+1}_result.json"
            with open(output_file, 'w') as f:
                # Convert EntityMask objects to dicts for JSON
                result_serializable = {
                    'success': result['success'],
                    'entity_count': len(result['entity_masks']),
                    'stage_timings': result['stage_timings'],
                    'metadata': result['metadata'],
                    'validation': {
                        'confidence': result['validation'].confidence if result['validation'] else 0.0
                    } if result['validation'] else None
                }
                json.dump(result_serializable, f, indent=2)

            print(f"\nSaved result: {output_file}")
        else:
            print(f"❌ FAILED: {result['error']}")

if __name__ == "__main__":
    main()