#!/usr/bin/env python3
"""Direct comparison of Rust vs Python ALMA performance."""
import numpy as np
import time
import subprocess
import json
import os
import my_project

# Ensure optimizations are enabled
os.environ['PYTHONOPTIMIZE'] = '2'
os.environ['NPY_RELAXED_STRIDES_CHECKING'] = '1'

def benchmark_python_alma(data, iterations=100):
    """Benchmark Python ALMA binding."""
    # Warmup
    for _ in range(10):
        _ = my_project.alma(data, 9, 0.85, 6.0)
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = my_project.alma(data, 9, 0.85, 6.0)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    return np.median(times)

def get_rust_benchmark():
    """Run Rust benchmark and extract the time."""
    # Run the Rust benchmark
    cmd = [
        "cargo", "bench", 
        "--features", "nightly-avx",
        "--bench", "indicator_benchmark",
        "--", "alma/1M/AVX-512", "--exact",
        "--output-format", "bencher"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        
        # Parse the benchmark output
        for line in output.split('\n'):
            if 'alma/1M/AVX-512' in line and 'ns/iter' in line:
                # Extract the time value
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'ns/iter' in part:
                        # The time is the previous part
                        time_str = parts[i-1].replace(',', '')
                        return float(time_str) / 1_000_000  # Convert ns to ms
        
        # Fallback: try scalar version
        cmd[-2] = "alma/1M/scalar"
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        
        for line in output.split('\n'):
            if 'alma/1M/scalar' in line and 'ns/iter' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'ns/iter' in part:
                        time_str = parts[i-1].replace(',', '')
                        return float(time_str) / 1_000_000
                        
    except subprocess.CalledProcessError as e:
        print(f"Error running Rust benchmark: {e}")
        return None
    
    return None

def main():
    print("Direct Rust vs Python ALMA Performance Comparison")
    print("=" * 60)
    
    # Test with different sizes
    sizes = [10_000, 100_000, 1_000_000]
    
    for size in sizes:
        print(f"\nTesting with {size:,} elements:")
        print("-" * 40)
        
        # Generate test data
        data = np.random.randn(size).astype(np.float64)
        
        # Ensure C-contiguous
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        
        # Benchmark Python
        python_time = benchmark_python_alma(data, iterations=100)
        print(f"Python ALMA: {python_time:.3f} ms")
        
        # For 1M elements, try to get Rust benchmark
        if size == 1_000_000:
            print("\nAttempting to run Rust benchmark...")
            rust_time = get_rust_benchmark()
            if rust_time:
                print(f"Rust ALMA: {rust_time:.3f} ms")
                overhead = ((python_time - rust_time) / rust_time) * 100
                print(f"Python overhead: {overhead:.1f}%")
            else:
                print("Could not get Rust benchmark time")
    
    # Detailed analysis for 1M elements
    print("\n" + "=" * 60)
    print("Detailed 1M Element Analysis")
    print("=" * 60)
    
    data_1m = np.random.randn(1_000_000).astype(np.float64)
    
    # Time 1000 iterations for more accuracy
    print("\nRunning 1000 iterations for accurate measurement...")
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        _ = my_project.alma(data_1m, 9, 0.85, 6.0)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    times = np.array(times)
    print(f"Mean: {np.mean(times):.3f} ms")
    print(f"Median: {np.median(times):.3f} ms")
    print(f"Std Dev: {np.std(times):.3f} ms")
    print(f"Min: {np.min(times):.3f} ms")
    print(f"Max: {np.max(times):.3f} ms")
    print(f"P5: {np.percentile(times, 5):.3f} ms")
    print(f"P95: {np.percentile(times, 95):.3f} ms")

if __name__ == "__main__":
    main()