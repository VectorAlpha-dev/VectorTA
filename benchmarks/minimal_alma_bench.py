#!/usr/bin/env python3
"""Minimal benchmark to test ALMA optimization impact."""
import numpy as np
import time
import my_project

# Test parameters
SIZES = [1000, 10_000, 100_000, 1_000_000]
ITERATIONS = 100
PERIOD = 9
OFFSET = 0.85
SIGMA = 6.0

def benchmark_alma(size: int, iterations: int) -> tuple[float, float]:
    """Benchmark ALMA and return mean and std time in microseconds."""
    # Generate random data once
    data = np.random.randn(size).astype(np.float64)
    
    # Ensure data is C-contiguous
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    
    # Warmup
    for _ in range(10):
        _ = my_project.alma(data, PERIOD, OFFSET, SIGMA)
    
    # Actual benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = my_project.alma(data, PERIOD, OFFSET, SIGMA)
        end = time.perf_counter()
        times.append((end - start) * 1_000_000)  # Convert to microseconds
    
    times_arr = np.array(times)
    return np.mean(times_arr), np.std(times_arr)

def main():
    print("ALMA Python Binding Performance Test")
    print("=" * 50)
    print(f"Period: {PERIOD}, Offset: {OFFSET}, Sigma: {SIGMA}")
    print(f"Iterations per size: {ITERATIONS}")
    print()
    
    for size in SIZES:
        mean_time, std_time = benchmark_alma(size, ITERATIONS)
        throughput = size / mean_time  # Elements per microsecond = Million elements per second
        
        print(f"Size: {size:>10,} | "
              f"Time: {mean_time:>8.2f} +/- {std_time:>6.2f} us | "
              f"Throughput: {throughput:>6.2f} M elem/s")
    
    # Test specifically 1M elements with more detail
    print("\nDetailed 1M element test:")
    data_1m = np.random.randn(1_000_000).astype(np.float64)
    
    # Time individual components
    start = time.perf_counter()
    result = my_project.alma(data_1m, PERIOD, OFFSET, SIGMA)
    total_time = (time.perf_counter() - start) * 1_000_000
    
    print(f"Total time for 1M elements: {total_time:.2f} us")
    print(f"Output type: {type(result)}")
    print(f"Output shape: {result.shape}")
    print(f"Output is C-contiguous: {result.flags['C_CONTIGUOUS']}")

if __name__ == "__main__":
    main()