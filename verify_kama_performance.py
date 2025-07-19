#!/usr/bin/env python3
"""
Accurate performance verification for KAMA Python bindings.
Ensures we're comparing apples to apples with the Rust benchmark.
"""

import time
import numpy as np
import gc
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parent))

try:
    import my_project
except ImportError:
    print("Error: my_project module not found")
    sys.exit(1)

def parse_rust_benchmark_time():
    """Parse the Rust benchmark results from Criterion JSON."""
    criterion_dir = Path(__file__).parent / 'target/criterion'
    
    # Look for KAMA benchmark results
    kama_paths = {
        'scalar': criterion_dir / 'kama/kama_scalar/1M/base/estimates.json',
        'avx2': criterion_dir / 'kama/kama_avx2/1M/base/estimates.json',
        'avx512': criterion_dir / 'kama/kama_avx512/1M/base/estimates.json',
    }
    
    results = {}
    for kernel, path in kama_paths.items():
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                # Convert nanoseconds to milliseconds
                median_ns = data['median']['point_estimate']
                median_ms = median_ns / 1_000_000
                results[kernel] = median_ms
        else:
            print(f"Warning: {path} not found. Run 'cargo bench --features nightly-avx --bench indicator_benchmark -- kama' first")
    
    return results

def benchmark_python_kama(data, period, kernel=None, warmup=10, iterations=50):
    """Benchmark Python KAMA with proper warmup and timing."""
    # Warmup
    for _ in range(warmup):
        if kernel:
            my_project.kama(data, period, kernel=kernel)
        else:
            my_project.kama(data, period)
    
    # Force GC
    gc.collect()
    gc.disable()
    
    # Time iterations
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        if kernel:
            my_project.kama(data, period, kernel=kernel)
        else:
            my_project.kama(data, period)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    gc.enable()
    
    # Return median time (matching Criterion)
    return np.median(times)

def main():
    print("KAMA Python Binding Performance Verification")
    print("=" * 60)
    
    # Load the same data size as Rust benchmark (1M points)
    print("\nGenerating test data (1M points)...")
    data = np.random.randn(1_000_000).astype(np.float64)
    
    # Use default KAMA period (30) to match Rust benchmark
    period = 30
    print(f"Using period: {period} (KAMA default)")
    
    # Parse Rust benchmark results
    print("\nParsing Rust benchmark results...")
    rust_times = parse_rust_benchmark_time()
    
    if not rust_times:
        print("ERROR: No Rust benchmark results found!")
        print("Please run: cargo bench --features nightly-avx --bench indicator_benchmark -- kama")
        return
    
    # Benchmark Python bindings
    print("\nBenchmarking Python bindings...")
    python_results = {}
    
    # Test with auto kernel (should select best available)
    python_results['auto'] = benchmark_python_kama(data, period)
    print(f"Python (auto kernel): {python_results['auto']:.3f} ms")
    
    # Test with specific kernels
    for kernel in ['scalar']:  # Can add 'avx2', 'avx512' if supported
        try:
            python_results[kernel] = benchmark_python_kama(data, period, kernel=kernel)
            print(f"Python ({kernel}): {python_results[kernel]:.3f} ms")
        except Exception as e:
            print(f"Python ({kernel}): Not available - {e}")
    
    # Calculate overhead
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"{'Implementation':<20} {'Time (ms)':<12} {'Overhead':<12}")
    print("-" * 44)
    
    # Compare with best Rust time
    best_rust_kernel = min(rust_times.keys(), key=lambda k: rust_times[k])
    best_rust_time = rust_times[best_rust_kernel]
    
    for kernel, rust_time in sorted(rust_times.items(), key=lambda x: x[1]):
        print(f"Rust {kernel:<15} {rust_time:>8.3f} ms")
    
    print("-" * 44)
    
    # Show Python results and calculate overhead
    python_auto_time = python_results['auto']
    overhead_percent = ((python_auto_time - best_rust_time) / best_rust_time) * 100
    
    print(f"Python (auto)        {python_auto_time:>8.3f} ms    {overhead_percent:>6.1f}%")
    
    if 'scalar' in python_results and 'scalar' in rust_times:
        scalar_overhead = ((python_results['scalar'] - rust_times['scalar']) / rust_times['scalar']) * 100
        print(f"Python (scalar)      {python_results['scalar']:>8.3f} ms    {scalar_overhead:>6.1f}%")
    
    print("\n" + "=" * 60)
    
    # Success criteria
    if overhead_percent < 10:
        print(f"[OK] SUCCESS: Python binding overhead is {overhead_percent:.1f}% (< 10% target)")
    else:
        print(f"[FAIL] FAILURE: Python binding overhead is {overhead_percent:.1f}% (exceeds 10% target)")
    
    # Additional analysis
    print("\nDetailed Analysis:")
    print(f"- Best Rust time: {best_rust_time:.3f} ms ({best_rust_kernel})")
    print(f"- Python auto time: {python_auto_time:.3f} ms")
    print(f"- Absolute difference: {python_auto_time - best_rust_time:.3f} ms")
    
    # Memory efficiency check
    print("\nMemory Efficiency Check:")
    import tracemalloc
    tracemalloc.start()
    
    start_mem = tracemalloc.get_traced_memory()[0]
    result = my_project.kama(data, period)
    end_mem = tracemalloc.get_traced_memory()[0]
    
    additional_mem = (end_mem - start_mem) / 1024 / 1024
    expected_mem = (len(data) * 8) / 1024 / 1024
    
    print(f"- Additional memory used: {additional_mem:.2f} MB")
    print(f"- Expected output size: {expected_mem:.2f} MB")
    print(f"- Memory efficiency: {(additional_mem / expected_mem * 100):.1f}%")
    
    tracemalloc.stop()

if __name__ == "__main__":
    main()