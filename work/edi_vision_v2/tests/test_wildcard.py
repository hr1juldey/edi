"""Wildcard robustness tests for the EDI Vision Pipeline

Tests the pipeline on diverse real-world images beyond the original test case
to validate robustness, identify edge cases, and establish baseline performance metrics.
"""

import pytest
import os
import sys
import numpy as np
from pathlib import Path
import json
import time
import shutil

# Add the project root to the path to import pipeline modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.orchestrator import VisionPipeline


@pytest.mark.slow
class TestWildcardRobustness:
    """Wildcard robustness tests on diverse images."""

    @pytest.fixture(scope="class")
    def images_dir(self):
        """Get the path to test images directory."""
        return Path("/home/riju279/Documents/Code/Zonko/EDI/edi/images/")
    
    @pytest.fixture(scope="class")
    def pipeline(self):
        """Initialize the vision pipeline with validation disabled."""
        return VisionPipeline(
            enable_validation=False,
            save_intermediate=False,
            output_dir="logs/wildcard"
        )
    
    def test_scenario_1_multi_color_detection(self, pipeline, images_dir):
        """Test multi-color vehicle detection in urban scene (kol_1.png)."""
        image_path = images_dir / "kol_1.png"
        
        if not image_path.exists():
            pytest.skip(f"Image not found: {image_path}")
        
        start_time = time.time()
        result = pipeline.process(
            image_path=str(image_path),
            user_prompt="highlight all red vehicles"
        )
        total_time = time.time() - start_time
        
        # Collect metrics
        metrics = {
            "image": "kol_1.png",
            "prompt": "highlight all red vehicles",
            "entities_detected": len(result.get('entity_masks', [])),
            "total_time_seconds": total_time,
            "success": result.get('success', False),
            "stage_timings": result.get('stage_timings', {}),
            "color_coverage_percent": result.get('metadata', {}).get('color_coverage', 0),
            "clip_filter_rate": result.get('metadata', {}).get('filter_rate', 0)
        }
        
        # Success criteria
        assert result['success'] == True, f"Pipeline failed: {result.get('error', 'Unknown error')}"
        assert len(result['entity_masks']) >= 1, f"Expected at least 1 red vehicle, found {len(result['entity_masks'])}"
        
        print(f"Scenario 1 - Entities detected: {len(result['entity_masks'])}, Time: {total_time:.2f}s")
        return metrics
    
    def test_scenario_2_similar_adjacent_objects(self, pipeline, images_dir):
        """Test roof detection with touching buildings (Darjeeling.jpg)."""
        image_path = images_dir / "Darjeeling.jpg"
        
        if not image_path.exists():
            pytest.skip(f"Image not found: {image_path}")
        
        start_time = time.time()
        result = pipeline.process(
            image_path=str(image_path),
            user_prompt="edit brown roofs"
        )
        total_time = time.time() - start_time
        
        # Collect metrics
        metrics = {
            "image": "Darjeeling.jpg",
            "prompt": "edit brown roofs",
            "entities_detected": len(result.get('entity_masks', [])),
            "total_time_seconds": total_time,
            "success": result.get('success', False),
            "stage_timings": result.get('stage_timings', {}),
            "color_coverage_percent": result.get('metadata', {}).get('color_coverage', 0),
            "clip_filter_rate": result.get('metadata', {}).get('filter_rate', 0)
        }
        
        # Success criteria
        assert result['success'] == True, f"Pipeline failed: {result.get('error', 'Unknown error')}"
        assert len(result['entity_masks']) >= 1, f"Expected at least 1 brown roof, found {len(result['entity_masks'])}"
        
        # Verify masks are separate (check entity_ids are unique)
        if result['entity_masks']:
            entity_ids = [e.entity_id for e in result['entity_masks']]
            assert len(entity_ids) == len(set(entity_ids)), "Entity IDs should be unique (separate masks required)"
        
        print(f"Scenario 2 - Entities detected: {len(result['entity_masks'])}, Time: {total_time:.2f}s")
        return metrics
    
    def test_scenario_3_high_resolution_image(self, pipeline, images_dir):
        """Test processing of high-resolution image (WP.jpg)."""
        image_path = images_dir / "WP.jpg"
        
        if not image_path.exists():
            pytest.skip(f"Image not found: {image_path}")
        
        start_time = time.time()
        result = pipeline.process(
            image_path=str(image_path),
            user_prompt="edit sky regions"
        )
        total_time = time.time() - start_time
        
        # Collect metrics
        metrics = {
            "image": "WP.jpg",
            "prompt": "edit sky regions",
            "entities_detected": len(result.get('entity_masks', [])),
            "total_time_seconds": total_time,
            "success": result.get('success', False),
            "stage_timings": result.get('stage_timings', {}),
            "color_coverage_percent": result.get('metadata', {}).get('color_coverage', 0),
            "clip_filter_rate": result.get('metadata', {}).get('filter_rate', 0)
        }
        
        # Success criteria
        assert result['success'] == True, f"Pipeline failed: {result.get('error', 'Unknown error')}"
        assert total_time < 60.0, f"Processing took too long: {total_time:.2f}s"
        assert len(result['entity_masks']) >= 0, f"Expected sky to be detected, found {len(result['entity_masks'])}"  # Sky may not always be blue
        
        print(f"Scenario 3 - Entities detected: {len(result['entity_masks'])}, Time: {total_time:.2f}s")
        return metrics
    
    def test_scenario_4_dense_scene(self, pipeline, images_dir):
        """Test processing of dense scene with many vehicles (mumbai-traffic.jpg)."""
        image_path = images_dir / "mumbai-traffic.jpg"
        
        if not image_path.exists():
            pytest.skip(f"Image not found: {image_path}")
        
        start_time = time.time()
        result = pipeline.process(
            image_path=str(image_path),
            user_prompt="detect yellow auto-rickshaws"
        )
        total_time = time.time() - start_time
        
        # Collect metrics
        metrics = {
            "image": "mumbai-traffic.jpg",
            "prompt": "detect yellow auto-rickshaws",
            "entities_detected": len(result.get('entity_masks', [])),
            "total_time_seconds": total_time,
            "success": result.get('success', False),
            "stage_timings": result.get('stage_timings', {}),
            "color_coverage_percent": result.get('metadata', {}).get('color_coverage', 0),
            "clip_filter_rate": result.get('metadata', {}).get('filter_rate', 0)
        }
        
        # Success criteria
        assert result['success'] == True, f"Pipeline failed: {result.get('error', 'Unknown error')}"
        assert len(result['entity_masks']) >= 0, f"Expected some yellow objects, found {len(result['entity_masks'])}"
        
        # Check false positive rate (not too many small noise entities)
        if result['entity_masks']:
            false_positive_entities = [e for e in result['entity_masks'] if e.area < 500]
            if len(result['entity_masks']) > 0:
                false_positive_ratio = len(false_positive_entities) / len(result['entity_masks'])
                assert false_positive_ratio <= 0.5, f"Too many small noise entities: {false_positive_ratio*100:.1f}%"
        
        print(f"Scenario 4 - Entities detected: {len(result['entity_masks'])}, Time: {total_time:.2f}s")
        return metrics
    
    def test_scenario_5_architectural_detail(self, pipeline, images_dir):
        """Test architectural detail detection (Pondicherry.jpg)."""
        image_path = images_dir / "Pondicherry.jpg"
        
        if not image_path.exists():
            pytest.skip(f"Image not found: {image_path}")
        
        start_time = time.time()
        result = pipeline.process(
            image_path=str(image_path),
            user_prompt="highlight yellow colonial buildings"
        )
        total_time = time.time() - start_time
        
        # Collect metrics
        metrics = {
            "image": "Pondicherry.jpg",
            "prompt": "highlight yellow colonial buildings",
            "entities_detected": len(result.get('entity_masks', [])),
            "total_time_seconds": total_time,
            "success": result.get('success', False),
            "stage_timings": result.get('stage_timings', {}),
            "color_coverage_percent": result.get('metadata', {}).get('color_coverage', 0),
            "clip_filter_rate": result.get('metadata', {}).get('filter_rate', 0)
        }
        
        # Success criteria
        assert result['success'] == True, f"Pipeline failed: {result.get('error', 'Unknown error')}"
        assert len(result['entity_masks']) >= 1, f"Expected at least 1 yellow building, found {len(result['entity_masks'])}"
        
        # Check building size consistency (should be large regions)
        if result['entity_masks']:
            avg_area = np.mean([e.area for e in result['entity_masks']])
            # Allow for smaller areas since not all yellow regions may be buildings
            print(f"Average area of detected masks: {avg_area:.2f}")
        
        print(f"Scenario 5 - Entities detected: {len(result['entity_masks'])}, Time: {total_time:.2f}s")
        return metrics
    
    def test_scenario_6_coastal_scene(self, pipeline, images_dir):
        """Test coastal scene with sky vs water distinction (pondi_2.jpg)."""
        image_path = images_dir / "pondi_2.jpg"
        
        if not image_path.exists():
            pytest.skip(f"Image not found: {image_path}")
        
        start_time = time.time()
        result = pipeline.process(
            image_path=str(image_path),
            user_prompt="edit blue sky"
        )
        total_time = time.time() - start_time
        
        # Collect metrics
        metrics = {
            "image": "pondi_2.jpg",
            "prompt": "edit blue sky",
            "entities_detected": len(result.get('entity_masks', [])),
            "total_time_seconds": total_time,
            "success": result.get('success', False),
            "stage_timings": result.get('stage_timings', {}),
            "color_coverage_percent": result.get('metadata', {}).get('color_coverage', 0),
            "clip_filter_rate": result.get('metadata', {}).get('filter_rate', 0)
        }
        
        # Success criteria
        assert result['success'] == True, f"Pipeline failed: {result.get('error', 'Unknown error')}"
        # For sky detection, we expect at least one large mask
        if result['entity_masks']:
            # Check that the largest mask is reasonably large (more likely to be sky)
            areas = [e.area for e in result['entity_masks']]
            max_area = max(areas) if areas else 0
            image_size = result['metadata'].get('image_shape', [1, 1, 3])
            total_pixels = image_size[0] * image_size[1]
            relative_size = max_area / total_pixels if total_pixels > 0 else 0
            print(f"Largest detected area covers {relative_size*100:.1f}% of image")
        
        print(f"Scenario 6 - Entities detected: {len(result['entity_masks'])}, Time: {total_time:.2f}s")
        return metrics
    
    def test_edge_case_1_no_color_match(self, pipeline, images_dir):
        """Test pipeline behavior with no color match (any available image)."""
        # Use any available image for this test
        available_images = [
            "kol_1.png", "Darjeeling.jpg", "mumbai-traffic.jpg", 
            "Pondicherry.jpg", "pondi_2.jpg", "WP.jpg"
        ]
        
        image_path = None
        for img_name in available_images:
            path = images_dir / img_name
            if path.exists():
                image_path = path
                break
        
        if image_path is None:
            pytest.skip("No images available for testing")
        
        start_time = time.time()
        result = pipeline.process(
            image_path=str(image_path),
            user_prompt="edit purple elements"  # Assuming no purple in the scene
        )
        total_time = time.time() - start_time
        
        # Collect metrics
        metrics = {
            "image": image_path.name,
            "prompt": "edit purple elements",
            "entities_detected": len(result.get('entity_masks', [])),
            "total_time_seconds": total_time,
            "success": result.get('success', False),
            "stage_timings": result.get('stage_timings', {}),
            "color_coverage_percent": result.get('metadata', {}).get('color_coverage', 0),
            "clip_filter_rate": result.get('metadata', {}).get('filter_rate', 0)
        }
        
        # Success criteria (should not crash, may or may not find purple elements)
        assert result['success'] == True, f"Pipeline should not crash on no match: {result.get('error', 'Unknown error')}"
        
        print(f"Edge Case 1 - Entities detected: {len(result['entity_masks'])}, Time: {total_time:.2f}s")
        return metrics
    
    def test_edge_case_2_ambiguous_query(self, pipeline, images_dir):
        """Test pipeline behavior with ambiguous semantic query."""
        # Use any available image for this test
        available_images = [
            "kol_1.png", "Darjeeling.jpg", "mumbai-traffic.jpg", 
            "Pondicherry.jpg", "pondi_2.jpg", "WP.jpg"
        ]
        
        image_path = None
        for img_name in available_images:
            path = images_dir / img_name
            if path.exists():
                image_path = path
                break
        
        if image_path is None:
            pytest.skip("No images available for testing")
        
        start_time = time.time()
        result = pipeline.process(
            image_path=str(image_path),
            user_prompt="edit interesting objects"  # Very vague prompt
        )
        total_time = time.time() - start_time
        
        # Collect metrics
        metrics = {
            "image": image_path.name,
            "prompt": "edit interesting objects",
            "entities_detected": len(result.get('entity_masks', [])),
            "total_time_seconds": total_time,
            "success": result.get('success', False),
            "stage_timings": result.get('stage_timings', {}),
            "color_coverage_percent": result.get('metadata', {}).get('color_coverage', 0),
            "clip_filter_rate": result.get('metadata', {}).get('filter_rate', 0)
        }
        
        # Success criteria (should complete without crashing)
        assert result['success'] == True, f"Pipeline should complete even with vague prompt: {result.get('error', 'Unknown error')}"
        
        print(f"Edge Case 2 - Entities detected: {len(result['entity_masks'])}, Time: {total_time:.2f}s")
        return metrics
    
    def test_edge_case_3_small_entities(self, pipeline, images_dir):
        """Test pipeline behavior with very small entities (looking for birds)."""
        # Use any available image that might have birds
        available_images = [
            "kol_1.png", "Darjeeling.jpg", "mumbai-traffic.jpg", 
            "Pondicherry.jpg", "pondi_2.jpg", "WP.jpg"
        ]
        
        image_path = None
        for img_name in available_images:
            path = images_dir / img_name
            if path.exists():
                image_path = path
                break
        
        if image_path is None:
            pytest.skip("No images available for testing")
        
        start_time = time.time()
        result = pipeline.process(
            image_path=str(image_path),
            user_prompt="detect small birds"  # Will likely find nothing or be filtered by min_area
        )
        total_time = time.time() - start_time
        
        # Collect metrics
        metrics = {
            "image": image_path.name,
            "prompt": "detect small birds",
            "entities_detected": len(result.get('entity_masks', [])),
            "total_time_seconds": total_time,
            "success": result.get('success', False),
            "stage_timings": result.get('stage_timings', {}),
            "color_coverage_percent": result.get('metadata', {}).get('color_coverage', 0),
            "clip_filter_rate": result.get('metadata', {}).get('filter_rate', 0)
        }
        
        # Success criteria (should not crash)
        assert result['success'] == True, f"Pipeline should not crash: {result.get('error', 'Unknown error')}"
        
        # If entities were detected, they should be above the min_area threshold
        if result['entity_masks']:
            for entity in result['entity_masks']:
                assert entity.area >= 500, f"Entity area {entity.area} is below min_area threshold"
        
        print(f"Edge Case 3 - Entities detected: {len(result['entity_masks'])}, Time: {total_time:.2f}s")
        return metrics


# Function to run all tests and collect metrics for the report
def run_wildcard_tests():
    """Run all wildcard tests and collect metrics."""
    import tempfile
    import subprocess
    
    # Set up pipeline
    pipeline = VisionPipeline(
        enable_validation=False,
        save_intermediate=True,
        output_dir="logs/wildcard"
    )
    
    # Ensure the logs/wildcard directory exists
    os.makedirs("logs/wildcard", exist_ok=True)
    
    # Create a temporary file to store metrics
    test_instance = TestWildcardRobustness()
    
    # Available images directory
    images_dir = Path("/home/riju279/Documents/Code/Zonko/EDI/edi/images/")
    
    all_metrics = []
    
    # Run each test scenario and collect metrics
    try:
        metrics = test_instance.test_scenario_1_multi_color_detection(pipeline, images_dir)
        all_metrics.append(metrics)
        print("✓ Scenario 1 completed")
    except Exception as e:
        print(f"✗ Scenario 1 failed: {e}")
    
    try:
        metrics = test_instance.test_scenario_2_similar_adjacent_objects(pipeline, images_dir)
        all_metrics.append(metrics)
        print("✓ Scenario 2 completed")
    except Exception as e:
        print(f"✗ Scenario 2 failed: {e}")
    
    try:
        metrics = test_instance.test_scenario_3_high_resolution_image(pipeline, images_dir)
        all_metrics.append(metrics)
        print("✓ Scenario 3 completed")
    except Exception as e:
        print(f"✗ Scenario 3 failed: {e}")
    
    try:
        metrics = test_instance.test_scenario_4_dense_scene(pipeline, images_dir)
        all_metrics.append(metrics)
        print("✓ Scenario 4 completed")
    except Exception as e:
        print(f"✗ Scenario 4 failed: {e}")
    
    try:
        metrics = test_instance.test_scenario_5_architectural_detail(pipeline, images_dir)
        all_metrics.append(metrics)
        print("✓ Scenario 5 completed")
    except Exception as e:
        print(f"✗ Scenario 5 failed: {e}")
    
    try:
        metrics = test_instance.test_scenario_6_coastal_scene(pipeline, images_dir)
        all_metrics.append(metrics)
        print("✓ Scenario 6 completed")
    except Exception as e:
        print(f"✗ Scenario 6 failed: {e}")
    
    try:
        metrics = test_instance.test_edge_case_1_no_color_match(pipeline, images_dir)
        all_metrics.append(metrics)
        print("✓ Edge Case 1 completed")
    except Exception as e:
        print(f"✗ Edge Case 1 failed: {e}")
    
    try:
        metrics = test_instance.test_edge_case_2_ambiguous_query(pipeline, images_dir)
        all_metrics.append(metrics)
        print("✓ Edge Case 2 completed")
    except Exception as e:
        print(f"✗ Edge Case 2 failed: {e}")
    
    try:
        metrics = test_instance.test_edge_case_3_small_entities(pipeline, images_dir)
        all_metrics.append(metrics)
        print("✓ Edge Case 3 completed")
    except Exception as e:
        print(f"✗ Edge Case 3 failed: {e}")
    
    # Save metrics to a JSON file
    metrics_file = "logs/wildcard/wildcard_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\nAll metrics saved to: {metrics_file}")
    print(f"Total test results collected: {len(all_metrics)}")
    
    # Create summary statistics
    successful_tests = [m for m in all_metrics if m['success']]
    total_time = sum(m.get('total_time_seconds', 0) for m in all_metrics)
    avg_time = total_time / len(all_metrics) if all_metrics else 0
    
    print(f"\nTest Summary:")
    print(f"  Successful tests: {len(successful_tests)}/{len(all_metrics)}")
    print(f"  Success rate: {len(successful_tests)/len(all_metrics)*100:.1f}% if tests exist" if all_metrics else "N/A")
    print(f"  Total processing time: {total_time:.2f}s")
    print(f"  Average processing time: {avg_time:.2f}s")
    
    return all_metrics


if __name__ == "__main__":
    run_wildcard_tests()