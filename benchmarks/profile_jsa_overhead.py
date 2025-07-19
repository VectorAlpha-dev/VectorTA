#!/usr/bin/env python3
"""Profile JSA to understand Python binding overhead."""

import time
import numpy as np
import my_project

# Test with different data sizes
sizes = [1000, 10_000, 100_000, 1_000_000]

print("JSA Python binding overhead analysis")
print("=" * 60)

for size in sizes:
    data = np.random.randn(size).astype(np.float64)
    
    # Measure Python binding time
    times_py = []
    for _ in range(50):
        start = time.perf_counter()
        result = my_project.jsa(data, 30)
        times_py.append(time.perf_counter() - start)
    
    median_py = sorted(times_py)[len(times_py) // 2] * 1000
    
    # Estimate pure computation time (using scalar kernel explicitly)
    times_scalar = []
    for _ in range(50):
        start = time.perf_counter()
        result = my_project.jsa(data, 30, kernel="scalar")
        times_scalar.append(time.perf_counter() - start)
    
    median_scalar = sorted(times_scalar)[len(times_scalar) // 2] * 1000
    
    print(f"\nData size: {size:,} points")
    print(f"  Python (auto kernel): {median_py:.3f} ms")
    print(f"  Python (scalar kernel): {median_scalar:.3f} ms")
    print(f"  Difference: {abs(median_py - median_scalar):.3f} ms")
    
    # Test if result is being cached (it shouldn't be)
    data_modified = data.copy()
    data_modified[0] += 1.0
    result1 = my_project.jsa(data, 30)
    result2 = my_project.jsa(data_modified, 30)
    print(f"  Results are different (no caching): {not np.array_equal(result1, result2)}")

# Test batch operation overhead
print("\n" + "=" * 60)
print("Batch operation analysis:")
data = np.random.randn(100_000).astype(np.float64)

start = time.perf_counter()
batch_result = my_project.jsa_batch(data, (10, 50, 5))
batch_time = (time.perf_counter() - start) * 1000

print(f"  Batch time for 9 periods: {batch_time:.3f} ms")
print(f"  Time per period: {batch_time / 9:.3f} ms")

# Compare with individual calls
individual_times = []
for period in range(10, 51, 5):
    start = time.perf_counter()
    _ = my_project.jsa(data, period)
    individual_times.append(time.perf_counter() - start)

total_individual = sum(individual_times) * 1000
print(f"  Total individual calls: {total_individual:.3f} ms")
print(f"  Batch speedup: {total_individual / batch_time:.2f}x")