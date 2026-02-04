"""
FactoryGuard AI - Latency & Performance Testing
Tests API endpoint performance and measures latency metrics
"""

import requests
import time
import numpy as np
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import sys


class LatencyTester:
    """
    Performance testing for prediction API
    """
    
    def __init__(self, base_url='http://localhost:5000'):
        self.base_url = base_url
        self.predict_url = f"{base_url}/predict"
        self.health_url = f"{base_url}/health"
        
    def generate_sample_request(self, machine_id='M001', scenario='normal'):
        """
        Generate sample prediction request
        
        Args:
            machine_id: Machine identifier
            scenario: 'normal', 'high_risk', or 'low_risk'
            
        Returns:
            dict: Sample request data
        """
        scenarios = {
            'normal': {
                'temperature': 70.0 + np.random.randn() * 5,
                'vibration': 0.4 + np.random.randn() * 0.1,
                'pressure': 100.0 + np.random.randn() * 3
            },
            'high_risk': {
                'temperature': 90.0 + np.random.randn() * 3,
                'vibration': 0.8 + np.random.randn() * 0.1,
                'pressure': 115.0 + np.random.randn() * 5
            },
            'low_risk': {
                'temperature': 60.0 + np.random.randn() * 2,
                'vibration': 0.25 + np.random.randn() * 0.05,
                'pressure': 95.0 + np.random.randn() * 2
            }
        }
        
        data = scenarios.get(scenario, scenarios['normal'])
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'machine_id': machine_id,
            'temperature': float(data['temperature']),
            'vibration': float(data['vibration']),
            'pressure': float(data['pressure'])
        }
    
    def check_health(self):
        """
        Check if API is healthy
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            response = requests.get(self.health_url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {str(e)}")
            return False
    
    def single_request(self, request_data):
        """
        Make a single prediction request and measure latency
        
        Args:
            request_data: Request payload
            
        Returns:
            dict: Result with latency and response
        """
        start_time = time.time()
        
        try:
            response = requests.post(
                self.predict_url,
                json=request_data,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                'success': response.status_code == 200,
                'latency_ms': latency_ms,
                'status_code': response.status_code,
                'response': response.json() if response.status_code == 200 else None,
                'error': None if response.status_code == 200 else response.text
            }
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return {
                'success': False,
                'latency_ms': latency_ms,
                'status_code': None,
                'response': None,
                'error': str(e)
            }
    
    def sequential_test(self, num_requests=100, scenario='normal'):
        """
        Run sequential latency test
        
        Args:
            num_requests: Number of requests to send
            scenario: Request scenario
            
        Returns:
            list: Results for each request
        """
        print(f"\n{'='*70}")
        print(f"SEQUENTIAL LATENCY TEST - {num_requests} requests")
        print(f"{'='*70}")
        
        results = []
        
        for i in range(num_requests):
            request_data = self.generate_sample_request(
                machine_id=f'M{i%10:03d}',
                scenario=scenario
            )
            
            result = self.single_request(request_data)
            results.append(result)
            
            if (i + 1) % 20 == 0:
                print(f"Progress: {i+1}/{num_requests} requests completed")
        
        return results
    
    def concurrent_test(self, num_requests=100, num_workers=10, scenario='normal'):
        """
        Run concurrent latency test
        
        Args:
            num_requests: Total number of requests
            num_workers: Number of concurrent workers
            scenario: Request scenario
            
        Returns:
            list: Results for each request
        """
        print(f"\n{'='*70}")
        print(f"CONCURRENT LATENCY TEST - {num_requests} requests, {num_workers} workers")
        print(f"{'='*70}")
        
        results = []
        
        # Generate all requests upfront
        requests_data = [
            self.generate_sample_request(
                machine_id=f'M{i%10:03d}',
                scenario=scenario
            )
            for i in range(num_requests)
        ]
        
        # Execute concurrently
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(self.single_request, req_data)
                for req_data in requests_data
            ]
            
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                
                if completed % 20 == 0:
                    print(f"Progress: {completed}/{num_requests} requests completed")
        
        return results
    
    def analyze_results(self, results):
        """
        Analyze test results and compute metrics
        
        Args:
            results: List of result dictionaries
            
        Returns:
            dict: Performance metrics
        """
        latencies = [r['latency_ms'] for r in results if r['success']]
        successes = [r for r in results if r['success']]
        failures = [r for r in results if not r['success']]
        
        if not latencies:
            return {
                'error': 'No successful requests',
                'total_requests': len(results),
                'success_count': 0,
                'failure_count': len(failures)
            }
        
        metrics = {
            'total_requests': len(results),
            'success_count': len(successes),
            'failure_count': len(failures),
            'success_rate': (len(successes) / len(results)) * 100,
            'avg_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'std_latency_ms': np.std(latencies)
        }
        
        # Calculate throughput
        if latencies:
            # Throughput = 1000ms / avg_latency_ms
            metrics['throughput_req_per_sec'] = 1000 / metrics['avg_latency_ms']
        
        return metrics
    
    def print_report(self, metrics, test_type=''):
        """
        Print formatted performance report
        
        Args:
            metrics: Performance metrics dictionary
            test_type: Type of test (for title)
        """
        print(f"\n{'='*70}")
        print(f"PERFORMANCE REPORT - {test_type}")
        print(f"{'='*70}")
        
        if 'error' in metrics:
            print(f"\n❌ ERROR: {metrics['error']}")
            print(f"Total Requests: {metrics['total_requests']}")
            print(f"Failures: {metrics['failure_count']}")
            return
        
        print(f"\n📊 Request Statistics:")
        print(f"  Total Requests:    {metrics['total_requests']}")
        print(f"  Successful:        {metrics['success_count']}")
        print(f"  Failed:            {metrics['failure_count']}")
        print(f"  Success Rate:      {metrics['success_rate']:.2f}%")
        
        print(f"\n⏱️  Latency Metrics:")
        print(f"  Average:           {metrics['avg_latency_ms']:.2f} ms")
        print(f"  Median (P50):      {metrics['p50_latency_ms']:.2f} ms")
        print(f"  P95:               {metrics['p95_latency_ms']:.2f} ms")
        print(f"  P99:               {metrics['p99_latency_ms']:.2f} ms")
        print(f"  Min:               {metrics['min_latency_ms']:.2f} ms")
        print(f"  Max:               {metrics['max_latency_ms']:.2f} ms")
        print(f"  Std Dev:           {metrics['std_latency_ms']:.2f} ms")
        
        print(f"\n🚀 Throughput:")
        print(f"  Requests/sec:      {metrics['throughput_req_per_sec']:.2f}")
        
        # Performance assessment
        print(f"\n✅ Performance Assessment:")
        avg_latency = metrics['avg_latency_ms']
        p95_latency = metrics['p95_latency_ms']
        
        if avg_latency < 50:
            print(f"  ✓ Average latency: EXCELLENT ({avg_latency:.2f}ms < 50ms target)")
        elif avg_latency < 100:
            print(f"  ⚠ Average latency: GOOD ({avg_latency:.2f}ms, target: <50ms)")
        else:
            print(f"  ✗ Average latency: NEEDS OPTIMIZATION ({avg_latency:.2f}ms > 100ms)")
        
        if p95_latency < 100:
            print(f"  ✓ P95 latency: EXCELLENT ({p95_latency:.2f}ms < 100ms)")
        elif p95_latency < 200:
            print(f"  ⚠ P95 latency: ACCEPTABLE ({p95_latency:.2f}ms < 200ms)")
        else:
            print(f"  ✗ P95 latency: NEEDS OPTIMIZATION ({p95_latency:.2f}ms > 200ms)")
        
        print(f"\n{'='*70}")
    
    def save_results(self, metrics, filename='latency_test_results.json'):
        """
        Save test results to JSON file
        
        Args:
            metrics: Performance metrics
            filename: Output filename
        """
        output = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")


def main():
    """
    Main execution function
    """
    parser = argparse.ArgumentParser(description='FactoryGuard AI - Latency Testing')
    parser.add_argument('--url', default='http://localhost:5000', help='API base URL')
    parser.add_argument('--requests', type=int, default=100, help='Number of requests')
    parser.add_argument('--concurrent', type=int, default=0, help='Number of concurrent workers (0 for sequential)')
    parser.add_argument('--scenario', default='normal', choices=['normal', 'high_risk', 'low_risk'], help='Test scenario')
    parser.add_argument('--output', default='latency_test_results.json', help='Output file')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = LatencyTester(base_url=args.url)
    
    # Check health
    print(f"\n{'='*70}")
    print("FACTORYGUARD AI - LATENCY & PERFORMANCE TESTING")
    print(f"{'='*70}")
    print(f"\nAPI URL: {args.url}")
    print(f"Checking API health...")
    
    if not tester.check_health():
        print("\n❌ ERROR: API is not healthy or not running")
        print("Please start the API server with: python app.py")
        sys.exit(1)
    
    print("✓ API is healthy")
    
    # Run tests
    if args.concurrent > 0:
        results = tester.concurrent_test(
            num_requests=args.requests,
            num_workers=args.concurrent,
            scenario=args.scenario
        )
        test_type = f"Concurrent ({args.concurrent} workers)"
    else:
        results = tester.sequential_test(
            num_requests=args.requests,
            scenario=args.scenario
        )
        test_type = "Sequential"
    
    # Analyze and report
    metrics = tester.analyze_results(results)
    tester.print_report(metrics, test_type=test_type)
    
    # Save results
    tester.save_results(metrics, filename=args.output)
    
    # Exit with appropriate code
    if metrics.get('success_rate', 0) == 100 and metrics.get('avg_latency_ms', float('inf')) < 50:
        print("\n✅ All tests passed! API meets performance targets.")
        sys.exit(0)
    elif metrics.get('success_rate', 0) == 100:
        print("\n⚠️  Tests passed but latency exceeds target. Consider optimization.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please review errors.")
        sys.exit(1)


if __name__ == '__main__':
    main()
