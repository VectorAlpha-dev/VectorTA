#!/usr/bin/env python3
"""
Simple, focused benchmark for ALMA Python bindings performance.
"""

import numpy as np
import time
from my_project import alma, alma_batch, AlmaStream

def benchmark_with_timing(func, *args, warmup=5, runs=20, **kwargs):
    """Run a function multiple times and return timing statistics."""
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Actual timing
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
    }

def format_time(seconds):
    """Format time in appropriate units."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    elif seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    else:
        return f"{seconds:.3f} s"

def main():
    print("ALMA Python Bindings Performance Benchmark")
    print("=" * 50)
    
    # Test parameters
    period = 9
    offset = 0.85
    sigma = 6.0
    
    # Test different data sizes
    sizes = [100, 1_000, 10_000, 100_000, 1_000_000]
    
    print("\n1. Single Calculation Performance")
    print("-" * 50)
    print(f"{'Size':>10} {'Total Time':>15} {'Per Element':>15} {'Throughput':>15}")
    print("-" * 50)
    
    for size in sizes:
        # Generate test data
        data = np.random.randn(size).astype(np.float64) * 10 + 100
        
        # Benchmark
        result = benchmark_with_timing(alma, data, period, offset, sigma)
        
        per_element = result['mean'] / size
        throughput = size / result['mean'] / 1e6  # Million elements/second
        
        print(f"{size:>10,} {format_time(result['mean']):>15} "
              f"{format_time(per_element):>15} {throughput:>12.1f} M/s")
    
    print("\n2. Batch Processing Performance")
    print("-" * 50)
    
    # Test batch with multiple period values
    data = np.random.randn(10_000).astype(np.float64) * 10 + 100
    period_ranges = [
        ((5, 15, 1), "11 periods"),
        ((5, 25, 2), "11 periods"),
        ((10, 50, 5), "9 periods"),
    ]
    
    print(f"{'Configuration':>20} {'Total Time':>15} {'Time/Combo':>15}")
    print("-" * 50)
    
    for (start, end, step), desc in period_ranges:
        result = benchmark_with_timing(
            alma_batch, 
            data, 
            (start, end, step),
            (offset, offset, 0.0),
            (sigma, sigma, 0.0)
        )
        
        num_combos = len(range(start, end + 1, step))
        per_combo = result['mean'] / num_combos
        
        print(f"{desc:>20} {format_time(result['mean']):>15} "
              f"{format_time(per_combo):>15}")
    
    print("\n3. Streaming Performance")
    print("-" * 50)
    
    # Test streaming with different update counts
    update_counts = [1_000, 10_000, 100_000]
    
    print(f"{'Updates':>10} {'Total Time':>15} {'Per Update':>15} {'Updates/sec':>15}")
    print("-" * 50)
    
    for count in update_counts:
        data = np.random.randn(count).astype(np.float64) * 10 + 100
        
        # Create stream for each benchmark run
        def stream_benchmark():
            stream = AlmaStream(period, offset, sigma)
            for value in data:
                stream.update(value)
        
        result = benchmark_with_timing(stream_benchmark, warmup=2, runs=10)
        
        per_update = result['mean'] / count
        updates_per_sec = count / result['mean']
        
        print(f"{count:>10,} {format_time(result['mean']):>15} "
              f"{format_time(per_update):>15} {updates_per_sec:>13,.0f}")
    
    print("\n4. Zero-Copy Verification")
    print("-" * 50)
    
    # Create a large array and measure memory overhead
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # Measure baseline memory
    baseline_mem = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create large array
    large_data = np.random.randn(10_000_000).astype(np.float64)
    data_size_mb = large_data.nbytes / 1024 / 1024
    
    after_alloc = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run ALMA
    result = alma(large_data, period, offset, sigma)
    
    after_alma = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Data size: {data_size_mb:.1f} MB")
    print(f"Memory after allocation: +{after_alloc - baseline_mem:.1f} MB")
    print(f"Memory after ALMA: +{after_alma - after_alloc:.1f} MB")
    print(f"Expected overhead: ~{data_size_mb:.1f} MB (for output array)")
    
    if (after_alma - after_alloc) < data_size_mb * 1.5:
        print("✓ Zero-copy confirmed - minimal memory overhead")
    else:
        print("✗ Excessive memory usage detected")
    
    print("\n5. Kernel Comparison")
    print("-" * 50)
    
    data = np.random.randn(1_000_000).astype(np.float64) * 10 + 100
    kernels = ['auto', 'scalar']
    
    print(f"{'Kernel':>10} {'Time':>15} {'Throughput':>15}")
    print("-" * 50)
    
    results = {}
    for kernel in kernels:
        result = benchmark_with_timing(alma, data, period, offset, sigma, kernel=kernel)
        throughput = len(data) / result['mean'] / 1e6
        results[kernel] = result['mean']
        
        print(f"{kernel:>10} {format_time(result['mean']):>15} {throughput:>12.1f} M/s")
    
    if 'scalar' in results and 'auto' in results:
        speedup = results['scalar'] / results['auto']
        print(f"\nSpeedup (auto vs scalar): {speedup:.2f}x")

if __name__ == "__main__":
    main()