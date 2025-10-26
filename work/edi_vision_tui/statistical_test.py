"""
Statistical testing module for EDI Vision Subsystem.
Tests the app for performance, consistency, stability, and utility.
"""

import time
import statistics
import random
from pathlib import Path
from typing import Dict, List, Any
from runnable_app import EDIVisionSystem


class StatisticalTester:
    """Class to perform statistical tests on the EDI Vision System"""
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.system = EDIVisionSystem()
        
        # Sample prompts for testing
        self.sample_prompts = [
            "edit the blue tin sheds to green",
            "change red roof to brown",
            "make the sky more dramatic",
            "enhance the green trees",
            "modify the yellow car to red",
            "adjust the building color",
            "change the water color",
            "edit grass to more green",
            "modify the person's shirt",
            "change tree color to autumn"
        ]
    
    def test_performance(self, num_runs: int = 10) -> Dict[str, Any]:
        """Test performance metrics across multiple runs"""
        times = []
        results = []
        
        print(f"Testing performance over {num_runs} runs...")
        
        for i in range(num_runs):
            prompt = random.choice(self.sample_prompts)
            
            start_time = time.time()
            result = self.system.process_vision_task(self.image_path, prompt)
            end_time = time.time()
            
            processing_time = end_time - start_time
            times.append(processing_time)
            
            if result.get('success'):
                results.append({
                    'run': i + 1,
                    'time': processing_time,
                    'alignment_score': result['summary'].get('alignment_score', 0),
                    'success': True
                })
            else:
                results.append({
                    'run': i + 1,
                    'time': processing_time,
                    'error': result.get('error', 'Unknown error'),
                    'success': False
                })
        
        # Calculate performance metrics
        successful_runs = [r for r in results if r['success']]
        successful_times = [r['time'] for r in successful_runs]
        
        performance_metrics = {
            'total_runs': num_runs,
            'successful_runs': len(successful_runs),
            'success_rate': len(successful_runs) / num_runs if num_runs > 0 else 0,
            'avg_processing_time': statistics.mean(successful_times) if successful_times else 0,
            'median_processing_time': statistics.median(successful_times) if successful_times else 0,
            'std_processing_time': statistics.stdev(successful_times) if len(successful_times) > 1 else 0,
            'min_processing_time': min(successful_times) if successful_times else 0,
            'max_processing_time': max(successful_times) if successful_times else 0,
            'all_times': successful_times
        }
        
        return {
            'results': results,
            'metrics': performance_metrics
        }
    
    def test_consistency(self, num_runs: int = 5, fixed_prompt: str = None) -> Dict[str, Any]:
        """Test consistency by running the same prompt multiple times"""
        if fixed_prompt is None:
            fixed_prompt = random.choice(self.sample_prompts)
        
        print(f"Testing consistency over {num_runs} runs with prompt: '{fixed_prompt}'")
        
        results = []
        alignment_scores = []
        
        for i in range(num_runs):
            result = self.system.process_vision_task(self.image_path, fixed_prompt)
            
            if result.get('success'):
                alignment_score = result['summary'].get('alignment_score', 0)
                alignment_scores.append(alignment_score)
                
                results.append({
                    'run': i + 1,
                    'alignment_score': alignment_score,
                    'success': True
                })
            else:
                results.append({
                    'run': i + 1,
                    'error': result.get('error', 'Unknown error'),
                    'success': False
                })
        
        # Calculate consistency metrics
        successful_runs = [r for r in results if r['success']]
        successful_scores = [r['alignment_score'] for r in successful_runs]
        
        if len(successful_scores) > 1:
            consistency_metrics = {
                'std_alignment_score': statistics.stdev(successful_scores),
                'variance_alignment_score': statistics.variance(successful_scores),
                'range_alignment_score': max(successful_scores) - min(successful_scores),
                'coefficient_of_variation': (statistics.stdev(successful_scores) / statistics.mean(successful_scores)) if statistics.mean(successful_scores) != 0 else 0
            }
        else:
            consistency_metrics = {
                'std_alignment_score': 0,
                'variance_alignment_score': 0,
                'range_alignment_score': 0,
                'coefficient_of_variation': 0
            }
        
        consistency_metrics.update({
            'total_runs': num_runs,
            'successful_runs': len(successful_runs),
            'success_rate': len(successful_runs) / num_runs if num_runs > 0 else 0,
            'avg_alignment_score': statistics.mean(successful_scores) if successful_scores else 0,
            'min_alignment_score': min(successful_scores) if successful_scores else 0,
            'max_alignment_score': max(successful_scores) if successful_scores else 0,
            'all_alignment_scores': successful_scores
        })
        
        return {
            'results': results,
            'metrics': consistency_metrics,
            'prompt_used': fixed_prompt
        }
    
    def test_stability(self, num_runs: int = 10) -> Dict[str, Any]:
        """Test stability over extended runs"""
        print(f"Testing stability over {num_runs} runs...")
        
        results = []
        system_errors = []
        processing_times = []
        
        for i in range(num_runs):
            prompt = random.choice(self.sample_prompts)
            
            try:
                start_time = time.time()
                result = self.system.process_vision_task(self.image_path, prompt)
                end_time = time.time()
                
                processing_time = end_time - start_time
                processing_times.append(processing_time)
                
                if result.get('success'):
                    results.append({
                        'run': i + 1,
                        'time': processing_time,
                        'alignment_score': result['summary'].get('alignment_score', 0),
                        'success': True
                    })
                else:
                    results.append({
                        'run': i + 1,
                        'time': processing_time,
                        'error': result.get('error', 'Unknown error'),
                        'success': False
                    })
                    system_errors.append(result.get('error', 'Unknown error'))
                    
            except Exception as e:
                results.append({
                    'run': i + 1,
                    'time': 0,
                    'error': str(e),
                    'success': False
                })
                system_errors.append(str(e))
        
        # Calculate stability metrics
        successful_runs = [r for r in results if r['success']]
        successful_times = [r['time'] for r in successful_runs]
        
        stability_metrics = {
            'total_runs': num_runs,
            'successful_runs': len(successful_runs),
            'success_rate': len(successful_runs) / num_runs if num_runs > 0 else 0,
            'error_rate': len(system_errors) / num_runs if num_runs > 0 else 0,
            'errors': system_errors,
            'avg_processing_time': statistics.mean(successful_times) if successful_times else 0,
            'max_processing_time': max(successful_times) if successful_times else 0,
            'crash_free_runs': num_runs - len([e for e in system_errors if 'crash' in e.lower() or 'segmentation' in e.lower()])
        }
        
        # Check for memory leaks by looking for increasing processing times
        stability_metrics['potential_memory_leak'] = self._check_increasing_trend(processing_times)
        
        return {
            'results': results,
            'metrics': stability_metrics
        }
    
    def test_utility(self, num_runs: int = 20) -> Dict[str, Any]:
        """Test utility by running various prompts and measuring effectiveness"""
        print(f"Testing utility over {num_runs} runs with varied prompts...")
        
        results = []
        high_quality_results = []  # Results with alignment score > 0.6
        medium_quality_results = []  # Results with alignment score 0.3-0.6
        low_quality_results = []  # Results with alignment score < 0.3
        
        for i in range(num_runs):
            # Use a random prompt
            prompt = random.choice(self.sample_prompts)
            
            result = self.system.process_vision_task(self.image_path, prompt)
            
            if result.get('success'):
                alignment_score = result['summary'].get('alignment_score', 0)
                
                # Classify based on alignment score
                if alignment_score > 0.6:
                    high_quality_results.append(result)
                elif alignment_score > 0.3:
                    medium_quality_results.append(result)
                else:
                    low_quality_results.append(result)
                
                results.append({
                    'run': i + 1,
                    'prompt': prompt,
                    'alignment_score': alignment_score,
                    'success': True
                })
            else:
                results.append({
                    'run': i + 1,
                    'prompt': prompt,
                    'error': result.get('error', 'Unknown error'),
                    'success': False
                })
        
        # Calculate utility metrics
        successful_runs = [r for r in results if r['success']]
        successful_scores = [r['alignment_score'] for r in successful_runs if 'alignment_score' in r]
        
        utility_metrics = {
            'total_runs': num_runs,
            'successful_runs': len(successful_runs),
            'success_rate': len(successful_runs) / num_runs if num_runs > 0 else 0,
            'high_quality_results': len(high_quality_results),
            'medium_quality_results': len(medium_quality_results),
            'low_quality_results': len(low_quality_results),
            'high_quality_rate': len(high_quality_results) / max(len(successful_runs), 1),
            'avg_alignment_score': statistics.mean(successful_scores) if successful_scores else 0,
            'median_alignment_score': statistics.median(successful_scores) if successful_scores else 0,
            'useful_results_rate': (len(high_quality_results) + len(medium_quality_results)) / max(len(successful_runs), 1)
        }
        
        return {
            'results': results,
            'metrics': utility_metrics
        }
    
    def _check_increasing_trend(self, times: List[float]) -> bool:
        """Check if processing times are increasing (potential memory leak)"""
        if len(times) < 5:
            return False
        
        # Compare first and last 25% of runs
        split_point = len(times) // 4
        first_quarter = times[:split_point]
        last_quarter = times[-split_point:]
        
        if not first_quarter or not last_quarter:
            return False
        
        avg_first = sum(first_quarter) / len(first_quarter)
        avg_last = sum(last_quarter) / len(last_quarter)
        
        # If last quarter is significantly slower than first quarter, potential memory leak
        return avg_last > (avg_first * 1.5)  # 50% slower
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all statistical tests and return comprehensive results"""
        print("="*60)
        print("COMPREHENSIVE STATISTICAL TESTING")
        print("="*60)
        
        results = {}
        
        # Performance test
        print("\n1. Performance Test:")
        results['performance'] = self.test_performance(num_runs=10)
        
        # Consistency test
        print("\n2. Consistency Test:")
        results['consistency'] = self.test_consistency(num_runs=5)
        
        # Stability test
        print("\n3. Stability Test:")
        results['stability'] = self.test_stability(num_runs=15)
        
        # Utility test
        print("\n4. Utility Test:")
        results['utility'] = self.test_utility(num_runs=20)
        
        # Generate summary
        summary = self._generate_summary(results)
        results['summary'] = summary
        
        return results
    
    def _generate_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of all test results"""
        perf_metrics = test_results['performance']['metrics']
        cons_metrics = test_results['consistency']['metrics']
        stab_metrics = test_results['stability']['metrics']
        util_metrics = test_results['utility']['metrics']
        
        # Determine overall ratings
        performance_rating = self._rate_performance(perf_metrics)
        consistency_rating = self._rate_consistency(cons_metrics)
        stability_rating = self._rate_stability(stab_metrics)
        utility_rating = self._rate_utility(util_metrics)
        
        overall_score = (
            performance_rating['score'] * 0.25 +
            consistency_rating['score'] * 0.25 +
            stability_rating['score'] * 0.25 +
            utility_rating['score'] * 0.25
        )
        
        return {
            'overall_score': overall_score,
            'overall_rating': self._score_to_rating(overall_score),
            'performance': performance_rating,
            'consistency': consistency_rating,
            'stability': stability_rating,
            'utility': utility_rating,
            'recommendations': self._generate_recommendations(test_results)
        }
    
    def _rate_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Rate performance based on metrics"""
        avg_time = metrics.get('avg_processing_time', float('inf'))
        
        if avg_time < 2.0:  # < 2 seconds is excellent
            score = 10.0
            rating = "EXCELLENT"
        elif avg_time < 5.0:  # < 5 seconds is good
            score = 8.0
            rating = "GOOD"
        elif avg_time < 10.0:  # < 10 seconds is fair
            score = 6.0
            rating = "FAIR"
        else:  # > 10 seconds is poor
            score = 4.0
            rating = "POOR"
        
        # Adjust for success rate
        success_rate = metrics.get('success_rate', 0)
        score *= success_rate
        
        return {
            'score': score,
            'rating': rating,
            'avg_processing_time': metrics.get('avg_processing_time', 0),
            'success_rate': success_rate
        }
    
    def _rate_consistency(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Rate consistency based on metrics"""
        std_score = metrics.get('std_alignment_score', 1.0)
        
        if std_score < 0.1:  # Very consistent
            score = 10.0
            rating = "EXCELLENT"
        elif std_score < 0.2:  # Fairly consistent
            score = 8.0
            rating = "GOOD"
        elif std_score < 0.3:  # Somewhat consistent
            score = 6.0
            rating = "FAIR"
        else:  # Inconsistent
            score = 4.0
            rating = "POOR"
        
        # Adjust for success rate
        success_rate = metrics.get('success_rate', 0)
        score *= success_rate
        
        return {
            'score': score,
            'rating': rating,
            'std_alignment_score': std_score,
            'avg_alignment_score': metrics.get('avg_alignment_score', 0),
            'success_rate': success_rate
        }
    
    def _rate_stability(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Rate stability based on metrics"""
        success_rate = metrics.get('success_rate', 0)
        error_rate = metrics.get('error_rate', 1.0)
        potential_leak = metrics.get('potential_memory_leak', False)
        
        score = 10.0
        
        # Lower score for high error rate
        score -= error_rate * 10.0
        
        # Lower score for potential memory leak
        if potential_leak:
            score -= 2.0
        
        # Ensure score doesn't go below 0
        score = max(0.0, score)
        
        if score >= 9.0:
            rating = "EXCELLENT"
        elif score >= 7.0:
            rating = "GOOD"
        elif score >= 5.0:
            rating = "FAIR"
        else:
            rating = "POOR"
        
        return {
            'score': score,
            'rating': rating,
            'success_rate': success_rate,
            'error_rate': error_rate,
            'potential_memory_leak': potential_leak
        }
    
    def _rate_utility(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Rate utility based on metrics"""
        success_rate = metrics.get('success_rate', 0)
        high_quality_rate = metrics.get('high_quality_rate', 0)
        useful_results_rate = metrics.get('useful_results_rate', 0)
        avg_score = metrics.get('avg_alignment_score', 0)
        
        # Base score on average alignment and proportion of useful results
        score = (avg_score * 5.0) + (useful_results_rate * 5.0)
        
        # Cap at 10
        score = min(10.0, score)
        
        if score >= 8.0:
            rating = "EXCELLENT"
        elif score >= 6.0:
            rating = "GOOD"
        elif score >= 4.0:
            rating = "FAIR"
        else:
            rating = "POOR"
        
        return {
            'score': score,
            'rating': rating,
            'success_rate': success_rate,
            'high_quality_rate': high_quality_rate,
            'useful_results_rate': useful_results_rate,
            'avg_alignment_score': avg_score
        }
    
    def _score_to_rating(self, score: float) -> str:
        """Convert a score to a rating"""
        if score >= 9.0:
            return "EXCELLENT"
        elif score >= 7.0:
            return "GOOD"
        elif score >= 5.0:
            return "FAIR"
        else:
            return "POOR"
    
    def _generate_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        perf_metrics = test_results['performance']['metrics']
        cons_metrics = test_results['consistency']['metrics']
        stab_metrics = test_results['stability']['metrics']
        util_metrics = test_results['utility']['metrics']
        
        # Performance recommendations
        if perf_metrics.get('avg_processing_time', 0) > 5.0:
            recommendations.append("Performance could be improved - average processing time is high")
        
        # Consistency recommendations
        if cons_metrics.get('std_alignment_score', 1.0) > 0.25:
            recommendations.append("Results are inconsistent - alignment scores vary significantly")
        
        # Stability recommendations
        if stab_metrics.get('error_rate', 0) > 0.1:  # >10% error rate
            recommendations.append("System stability needs improvement - error rate is high")
        
        if stab_metrics.get('potential_memory_leak', False):
            recommendations.append("Potential memory leak detected - processing times are increasing")
        
        # Utility recommendations
        if util_metrics.get('high_quality_rate', 0) < 0.5:  # <50% high quality
            recommendations.append("Quality of results needs improvement - too few high-quality outputs")
        
        if not recommendations:
            recommendations.append("System is performing well across all metrics!")
        
        return recommendations


def run_statistical_tests(image_path: str) -> Dict[str, Any]:
    """Run statistical tests on the EDI Vision System"""
    tester = StatisticalTester(image_path)
    return tester.run_comprehensive_test()


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Statistical Testing for EDI Vision System")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--output", help="Path to save test results")
    
    args = parser.parse_args()
    
    print("Running statistical tests on EDI Vision System...")
    print(f"Using image: {args.image}")
    print()
    
    results = run_statistical_tests(args.image)
    
    # Print summary
    summary = results['summary']
    
    print("="*60)
    print("STATISTICAL TEST SUMMARY")
    print("="*60)
    print(f"Overall Score: {summary['overall_score']:.2f}/10.0 ({summary['overall_rating']})")
    print()
    
    for category, metrics in summary.items():
        if category == 'overall_score' or category == 'overall_rating' or category == 'recommendations':
            continue
        print(f"{category.title()} Score: {metrics['score']:.2f}/10.0 ({metrics['rating']})")
    
    print()
    print("Recommendations:")
    for rec in summary['recommendations']:
        print(f"  - {rec}")
    
    # Save results if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {args.output}")