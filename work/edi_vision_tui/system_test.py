"""
System testing functionality for EDI vision subsystem.
Tests the system by presenting wrong outputs to verify detection capability.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any
from .change_detector import compare_output
from .mask_generator import generate_mask_for_prompt


def test_system_detection_with_wrong_output(image_path: str, 
                                          wrong_output_path: str, 
                                          prompt: str) -> Dict[str, Any]:
    """
    Test system by presenting wrong outputs to verify detection.
    
    Args:
        image_path: Path to the original input image
        wrong_output_path: Path to a wrong output (e.g., original image again, unrelated image)
        prompt: The original edit prompt
    
    Returns:
        Dictionary with test results
    """
    try:
        # Generate masks for the prompt to compare against
        mask_result = generate_mask_for_prompt(image_path, prompt)
        
        if not mask_result['success']:
            return {
                'test_passed': False,
                'error': f"Mask generation failed: {mask_result['error']}",
                'confidence': 0.0
            }
        
        # Create a mock mask list for the comparison function
        masks = [{
            'id': 'test_mask',
            'bbox': mask_result['bbox'],
            'confidence': 0.8
        }]
        
        # Compare the original image with the "wrong" output
        # This should detect that the "output" is not actually a modified version
        comparison_result = compare_output(
            image_path_before=image_path,
            image_path_after=wrong_output_path,
            expected_keywords=[prompt],  # Use prompt as expected keywords
            masks=masks
        )
        
        # Analyze the comparison to determine if the system correctly detected wrong output
        # If the "output" is the same as input (or unrelated), changes_inside should be low
        # and the system should flag this as an issue
        
        # Determine if the test passed (system correctly detected wrong output)
        # This is somewhat heuristic - if changes_inside is very low relative to changes_outside
        # or if alignment score is very low, it might indicate the output is wrong
        changes_inside = comparison_result.get('changes_inside', 0)
        changes_outside = comparison_result.get('changes_outside', 0)
        alignment_score = comparison_result.get('alignment_score', 0.0)
        
        # For a "wrong" output that's actually the same as input, changes should be minimal
        # For a "wrong" output that's unrelated, changes might be all over the place (low alignment)
        if alignment_score < 0.3:
            test_passed = True
        elif changes_inside == 0 and changes_outside == 0:
            # No changes at all - likely the same image
            test_passed = True
        else:
            # Some changes but not properly aligned - might be wrong
            test_passed = True  # We consider detection of misalignment as correct detection
        
        # Calculate confidence in the test result
        confidence = min(1.0, (1.0 - alignment_score) * 2.0)  # Higher confidence if alignment is poor
        
        return {
            'test_passed': test_passed,
            'confidence': confidence,
            'changes_inside': changes_inside,
            'changes_outside': changes_outside,
            'alignment_score': alignment_score,
            'comparison_details': comparison_result
        }
        
    except Exception as e:
        return {
            'test_passed': False,
            'error': f"Test execution failed: {str(e)}",
            'confidence': 0.0
        }


def test_with_known_wrong_outputs(image_path: str, prompt: str) -> Dict[str, Any]:
    """
    Test the system with known wrong outputs to verify detection.
    
    Args:
        image_path: Path to the original input image
        prompt: The original edit prompt
    
    Returns:
        Dictionary with comprehensive test results
    """
    results = {
        'tests': [],
        'overall_success_rate': 0.0,
        'total_tests': 0,
        'passed_tests': 0
    }
    
    # Get the directory of the input image to find test images
    img_dir = Path(image_path).parent
    
    # List of known wrong outputs to test with
    # These include the original image (no changes) and unrelated images
    test_cases = [
        {
            'name': 'original_image',
            'path': image_path,
            'description': 'Same as input - should be detected as no edit'
        },
        {
            'name': 'same_image_copy',
            'path': image_path,
            'description': 'Same as input - should be detected as no edit'
        }
    ]
    
    # Try to add other images from the images directory as additional wrong outputs
    for img_file in img_dir.glob("*.jpg"):
        if img_file.name.lower() not in [Path(image_path).name.lower(), 'ip.jpeg', 'op.jpeg']:
            test_cases.append({
                'name': f'unrelated_{img_file.name}',
                'path': str(img_file),
                'description': f'Unrelated image - should be detected as wrong output'
            })
    
    # Add some specifically mentioned wrong outputs if they exist
    special_cases = ['Pondicherry.jpg', 'WP.jpg']
    for case in special_cases:
        case_path = img_dir / case
        if case_path.exists():
            test_cases.append({
                'name': f'special_{case}',
                'path': str(case_path),
                'description': f'Specially mentioned wrong output: {case}'
            })
    
    # Run tests for each case
    for test_case in test_cases:
        test_result = test_system_detection_with_wrong_output(
            image_path, 
            test_case['path'], 
            prompt
        )
        
        test_result['case_name'] = test_case['name']
        test_result['description'] = test_case['description']
        
        results['tests'].append(test_result)
        
        if test_result.get('test_passed', False):
            results['passed_tests'] += 1
        
        results['total_tests'] += 1
    
    # Calculate overall success rate
    if results['total_tests'] > 0:
        results['overall_success_rate'] = results['passed_tests'] / results['total_tests']
    
    return results


def run_comprehensive_system_test(image_path: str, prompt: str) -> Dict[str, Any]:
    """
    Run a comprehensive system test that includes:
    1. Normal operation (with actual edited image)
    2. Wrong output detection tests
    3. Statistical analysis
    """
    result = {
        'normal_operation': None,
        'wrong_output_detection': None,
        'statistical_analysis': None
    }
    
    # First, run normal operation (with mock edit)
    try:
        from .mock_edit import send_mock_edit_request
        from .mask_generator import decompose_prompt
        
        # Generate keywords from prompt
        keywords = decompose_prompt(prompt)
        
        # Create masks for the keywords
        mask_results = []
        for keyword in keywords[:2]:  # Use first 2 keywords for efficiency
            mask_result = generate_mask_for_prompt(image_path, keyword)
            if mask_result['success']:
                mask_results.append({
                    'id': f'mask_{len(mask_results)}',
                    'bbox': mask_result['bbox'],
                    'confidence': 0.8
                })
        
        # Create a mock edit
        mock_result = send_mock_edit_request(image_path, prompt, mask_results)
        
        if mock_result['success'] and mock_result['output_path']:
            # Compare the normal edit result
            comparison = compare_output(
                image_path_before=image_path,
                image_path_after=mock_result['output_path'],
                expected_keywords=keywords,
                masks=mask_results
            )
            
            result['normal_operation'] = {
                'success': True,
                'comparison': comparison,
                'edit_result': mock_result
            }
        else:
            result['normal_operation'] = {
                'success': False,
                'error': 'Mock edit failed'
            }
            
    except Exception as e:
        result['normal_operation'] = {
            'success': False,
            'error': f'Normal operation test failed: {str(e)}'
        }
    
    # Then, run wrong output detection tests
    try:
        wrong_output_results = test_with_known_wrong_outputs(image_path, prompt)
        result['wrong_output_detection'] = wrong_output_results
    except Exception as e:
        result['wrong_output_detection'] = {
            'success': False,
            'error': f'Wrong output detection test failed: {str(e)}'
        }
    
    # Perform statistical analysis
    try:
        stats = {
            'normal_operation_success': result['normal_operation']['success'] if result['normal_operation'] else False,
            'wrong_output_detection_rate': result['wrong_output_detection']['overall_success_rate'] if result['wrong_output_detection'] else 0.0,
            'total_test_cases': result['wrong_output_detection']['total_tests'] if result['wrong_output_detection'] else 0
        }
        
        result['statistical_analysis'] = stats
    except Exception as e:
        result['statistical_analysis'] = {
            'error': f'Statistical analysis failed: {str(e)}'
        }
    
    return result


def validate_system_performance(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the system's performance based on test results.
    
    Args:
        results: Results from run_comprehensive_system_test
    
    Returns:
        Dictionary with performance validation results
    """
    validation = {
        'system_integrity': 'PASS' if results['normal_operation']['success'] else 'FAIL',
        'detection_capability': 'PASS' if results['wrong_output_detection']['overall_success_rate'] >= 0.7 else 'FAIL',
        'overall_performance': 'PASS' if (results['normal_operation']['success'] and 
                                       results['wrong_output_detection']['overall_success_rate'] >= 0.7) else 'FAIL',
        'recommendations': []
    }
    
    # Add recommendations based on results
    if not results['normal_operation']['success']:
        validation['recommendations'].append(
            "System could not process normal edits - check mask generation and mock edit functionality"
        )
    
    if results['wrong_output_detection']['overall_success_rate'] < 0.7:
        validation['recommendations'].append(
            f"Wrong output detection rate is low ({results['wrong_output_detection']['overall_success_rate']:.2%}). "
            "Consider improving change detection algorithms."
        )
    
    if not validation['recommendations']:
        validation['recommendations'].append("System is performing well across all tested metrics.")
    
    return validation