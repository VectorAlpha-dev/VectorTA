#!/usr/bin/env python3
"""Analyze PyO3 binding overhead in detail."""

import time
import numpy as np
import my_project

# Test with different data sizes to understand scaling
sizes = [100, 1000, 10_000, 100_000, 1_000_000]

print("PyO3 Binding Overhead Analysis")
print("=" * 60)
print("\nTesting JSA with different data sizes:")
print(f"{'Size':>10} | {'Time (ms)':>10} | {'ns/element':>12} | {'Overhead':>10}")
print("-" * 60)

baseline_overhead_ns = None

for size in sizes:
    data = np.random.randn(size).astype(np.float64)
    
    # Warmup
    for _ in range(5):
        _ = my_project.jsa(data, 30)
    
    # Measure
    times = []
    for _ in range(50):
        start = time.perf_counter()
        _ = my_project.jsa(data, 30)
        times.append(time.perf_counter() - start)
    
    median_s = sorted(times)[len(times) // 2]
    median_ms = median_s * 1000
    ns_per_element = (median_s * 1e9) / size
    
    # Estimate fixed overhead from small size
    if size == 100:
        baseline_overhead_ns = median_s * 1e9
    
    # Estimate variable overhead
    if baseline_overhead_ns:
        overhead_ms = (baseline_overhead_ns / 1e6)
        overhead_pct = (overhead_ms / median_ms) * 100
    else:
        overhead_pct = 0
    
    print(f"{size:>10,} | {median_ms:>10.3f} | {ns_per_element:>12.2f} | {overhead_pct:>9.1f}%")

# Now test the overhead of just creating arrays
print("\n" + "=" * 60)
print("NumPy array creation overhead:")
print(f"{'Size':>10} | {'Create (µs)':>12} | {'Slice (µs)':>12}")
print("-" * 60)

for size in sizes:
    # Test array creation time
    create_times = []
    for _ in range(100):
        start = time.perf_counter()
        arr = np.empty(size, dtype=np.float64)
        create_times.append(time.perf_counter() - start)
    
    # Test slice access time
    data = np.random.randn(size).astype(np.float64)
    slice_times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = data.__array_interface__['data'][0]  # Get pointer
        slice_times.append(time.perf_counter() - start)
    
    create_median = sorted(create_times)[50] * 1e6
    slice_median = sorted(slice_times)[50] * 1e6
    
    print(f"{size:>10,} | {create_median:>12.2f} | {slice_median:>12.2f}")

# Theoretical minimum overhead calculation
print("\n" + "=" * 60)
print("Theoretical minimum Python binding overhead:")
print("  - PyO3 function call: ~50-100 ns")
print("  - Input validation: ~20-50 ns")
print("  - Array slice access: ~100-200 ns")
print("  - Output array creation: ~1-10 µs (size dependent)")
print("  - GIL release/acquire: ~100-200 ns")
print("  - Error handling: ~50-100 ns")
print("  - Total fixed overhead: ~500-1000 ns minimum")
print("\nFor 1M element array:")
print("  - JSA computation: ~92 µs (0.092 ns/element)")
print("  - Minimum overhead: ~10 µs (array creation)")
print("  - Expected total: ~102 µs (0.102 ms)")
print("  - Actual: ~590 µs (0.590 ms)")
print("  - Unexplained overhead: ~488 µs")