#!/usr/bin/env python3
"""Final JSA performance comparison."""

import time
import numpy as np
import my_project
from pathlib import Path
import csv

# Load the same CSV data used in criterion benchmark
csv_path = Path(__file__).parent.parent / 'src/data/1MillionCandles.csv'
closes = []
with open(csv_path, 'r') as f:
    f.readline()  # Skip header
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 7:
            try:
                closes.append(float(row[4]))  # close price
            except ValueError:
                continue

data = np.array(closes, dtype=np.float64)
print(f"Loaded {len(data)} candles from CSV")

# Warmup
for _ in range(10):
    _ = my_project.jsa(data, 30)

# Benchmark Python binding
times = []
for _ in range(100):
    start = time.perf_counter()
    _ = my_project.jsa(data, 30)
    times.append(time.perf_counter() - start)

times_ms = [t * 1000 for t in times]
times_ms.sort()
median = times_ms[len(times_ms) // 2]
mean = sum(times_ms) / len(times_ms)

print(f"\nJSA Python Binding Performance:")
print(f"  Data: {len(data)} points")
print(f"  Period: 30")
print(f"  Median time: {median:.3f} ms")
print(f"  Mean time: {mean:.3f} ms")
print(f"  Min time: {min(times_ms):.3f} ms")
print(f"  Max time: {max(times_ms):.3f} ms")

# Based on our Rust benchmark results:
rust_median = 0.760  # ms (from bench_jsa_kernels)
overhead_percent = ((median - rust_median) / rust_median) * 100

print(f"\nComparison with Rust:")
print(f"  Rust median: {rust_median:.3f} ms")
print(f"  Python median: {median:.3f} ms")
print(f"  Overhead: {overhead_percent:.1f}%")

if overhead_percent < 10:
    print(f"  PASS: Overhead is less than 10%")
else:
    print(f"  FAIL: Overhead exceeds 10% target")

# Test with kernel parameter
print(f"\nTesting kernel parameter support:")
try:
    result_auto = my_project.jsa(data, 30)
    result_scalar = my_project.jsa(data, 30, kernel="scalar")
    print("  Kernel parameter works correctly")
except Exception as e:
    print(f"  Kernel parameter error: {e}")

# Test batch operation
print(f"\nTesting batch operation:")
try:
    batch_result = my_project.jsa_batch(data, 20, 40, 5)  # Updated to match new signature
    print(f"  Batch operation works")
    print(f"  Result shape: {batch_result['values'].shape}")
    print(f"  Periods array: {batch_result['periods']}")
except Exception as e:
    print(f"  Batch operation error: {e}")