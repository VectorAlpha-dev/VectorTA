#!/usr/bin/env python3
"""Test if import overhead is affecting benchmarks."""
import time
import numpy as np

# Time the import
start = time.perf_counter()
import my_project
import_time = (time.perf_counter() - start) * 1000

print(f"Import time: {import_time:.2f} ms")

# Test function call overhead
data = np.random.randn(1_000_000).astype(np.float64)

# First call (may have initialization overhead)
start = time.perf_counter()
result1 = my_project.alma(data, 9, 0.85, 6.0)
first_call_time = (time.perf_counter() - start) * 1000

# Second call (warmed up)
start = time.perf_counter()
result2 = my_project.alma(data, 9, 0.85, 6.0)
second_call_time = (time.perf_counter() - start) * 1000

# Many calls to get average
times = []
for _ in range(100):
    start = time.perf_counter()
    _ = my_project.alma(data, 9, 0.85, 6.0)
    times.append((time.perf_counter() - start) * 1000)

avg_time = np.mean(times)

print(f"First call time: {first_call_time:.2f} ms")
print(f"Second call time: {second_call_time:.2f} ms")
print(f"Average time (100 calls): {avg_time:.2f} ms")
print(f"Initialization overhead: {first_call_time - avg_time:.2f} ms")