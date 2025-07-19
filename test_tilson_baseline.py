#!/usr/bin/env python3
"""Quick baseline test for tilson performance."""

import time
import numpy as np
import my_project

# Generate test data
np.random.seed(42)
data = np.random.randn(1_000_000).astype(np.float64)

# Warmup
for _ in range(10):
    _ = my_project.tilson(data, 14, 0.7)

# Benchmark
times = []
for _ in range(100):
    start = time.perf_counter()
    _ = my_project.tilson(data, 14, 0.7)
    times.append((time.perf_counter() - start) * 1000)  # Convert to ms

print(f"Tilson baseline performance (current implementation):")
print(f"  Median: {np.median(times):.2f} ms")
print(f"  Mean: {np.mean(times):.2f} ms")
print(f"  Std: {np.std(times):.2f} ms")
print(f"  Min: {np.min(times):.2f} ms")
print(f"  Max: {np.max(times):.2f} ms")