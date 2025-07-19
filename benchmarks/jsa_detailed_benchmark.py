#!/usr/bin/env python3
"""Detailed JSA benchmark to verify optimization."""

import time
import numpy as np
import my_project

# Create test data matching Rust benchmark
data = np.array([50000.0 + np.sin(i) * 1000.0 for i in range(1_002_240)], dtype=np.float64)

print(f"JSA Python detailed benchmark:")
print(f"  Data size: {len(data)} points")
print(f"  Data dtype: {data.dtype}")
print(f"  Data is C-contiguous: {data.flags['C_CONTIGUOUS']}")
print()

# Test with different kernels
kernels = [None, "scalar", "avx2", "avx512"]

for kernel in kernels:
    # Warmup
    for _ in range(10):
        if kernel:
            _ = my_project.jsa(data, 30, kernel=kernel)
        else:
            _ = my_project.jsa(data, 30)
    
    # Benchmark
    times = []
    for _ in range(100):
        start = time.perf_counter()
        if kernel:
            _ = my_project.jsa(data, 30, kernel=kernel)
        else:
            _ = my_project.jsa(data, 30)
        times.append((time.perf_counter() - start) * 1000)
    
    times.sort()
    median = times[len(times) // 2]
    
    kernel_name = kernel or "auto"
    print(f"  Kernel '{kernel_name}':")
    print(f"    Median time: {median:.3f} ms")
    print(f"    Min time: {min(times):.3f} ms")
    print(f"    Max time: {max(times):.3f} ms")
    print()

# Test batch operation
print("Batch operation test:")
start = time.perf_counter()
result = my_project.jsa_batch(data, (10, 100, 10))
elapsed = (time.perf_counter() - start) * 1000
print(f"  Period range: (10, 100, 10)")
print(f"  Result shape: {result['values'].shape}")
print(f"  Time: {elapsed:.3f} ms")
print(f"  Time per combo: {elapsed / result['values'].shape[0]:.3f} ms")