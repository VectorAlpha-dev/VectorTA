#!/usr/bin/env python3
"""
Direct performance comparison between Rust and Python KAMA implementations.
"""

import time
import numpy as np
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    import my_project
except ImportError:
    print("Error: my_project module not found. Run: python -m maturin develop --features python,nightly-avx --release")
    sys.exit(1)

def benchmark_function(func, warmup_runs=10, test_runs=50):
    """Benchmark a function with warmup and proper timing."""
    # Warmup
    for _ in range(warmup_runs):
        func()
    
    # Force garbage collection
    gc.collect()
    gc.disable()
    
    # Actual timing
    times = []
    for _ in range(test_runs):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds
    
    gc.enable()
    
    # Return median time
    return np.median(times)

def main():
    print("KAMA Performance Comparison")
    print("=" * 60)
    
    # Test different data sizes
    sizes = [10_000, 100_000, 1_000_000]
    
    for size in sizes:
        print(f"\nData size: {size:,}")
        print("-" * 40)
        
        # Generate test data
        data = np.random.randn(size).astype(np.float64)
        period = 14  # Match the benchmark parameter
        
        # Test Python binding
        def python_kama():
            return my_project.kama(data, period)
        
        python_time = benchmark_function(python_kama)
        print(f"Python binding: {python_time:.3f} ms")
        
        # Test with different kernels
        kernels = ["scalar", "auto"]
        for kernel in kernels:
            def python_kama_kernel():
                return my_project.kama(data, period, kernel=kernel)
            
            kernel_time = benchmark_function(python_kama_kernel)
            print(f"Python binding ({kernel}): {kernel_time:.3f} ms")
    
    # Test batch operation
    print("\n\nBatch Operation Test (100k data, 5 periods)")
    print("-" * 40)
    data = np.random.randn(100_000).astype(np.float64)
    
    def batch_test():
        return my_project.kama_batch(data, (10, 50, 10))
    
    batch_time = benchmark_function(batch_test, warmup_runs=5, test_runs=20)
    print(f"Batch operation: {batch_time:.3f} ms")
    
    # Memory efficiency test
    print("\n\nMemory Efficiency Test")
    print("-" * 40)
    import tracemalloc
    
    tracemalloc.start()
    data = np.random.randn(1_000_000).astype(np.float64)
    
    # Measure memory usage
    start_memory = tracemalloc.get_traced_memory()[0]
    result = my_project.kama(data, period)
    end_memory = tracemalloc.get_traced_memory()[0]
    
    memory_used = (end_memory - start_memory) / 1024 / 1024  # Convert to MB
    print(f"Additional memory used: {memory_used:.2f} MB")
    print(f"Expected output size: {(len(data) * 8) / 1024 / 1024:.2f} MB")
    
    tracemalloc.stop()
    
    print("\n" + "=" * 60)
    print("Note: Compare these times with Rust benchmark results:")
    print("cargo bench --features nightly-avx --bench indicator_benchmark -- kama")

if __name__ == "__main__":
    main()