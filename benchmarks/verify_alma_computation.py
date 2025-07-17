#!/usr/bin/env python3
"""Verify ALMA computation and timing."""
import numpy as np
import time
import my_project
import csv

# Load the same data used in benchmarks
csv_path = 'src/data/1MillionCandles.csv'
closes = []
with open(csv_path, 'r') as f:
    f.readline()  # Skip header
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 5:
            closes.append(float(row[4]))  # Close is column 4

close_data = np.array(closes, dtype=np.float64)
print(f"Loaded {len(close_data)} close prices")

# Verify data is C-contiguous
print(f"Data is C-contiguous: {close_data.flags['C_CONTIGUOUS']}")

# Run ALMA and verify output
print("\nRunning ALMA computation...")
start = time.perf_counter()
result = my_project.alma(close_data, 9, 0.85, 6.0)
end = time.perf_counter()
first_time = (end - start) * 1000

print(f"First run time: {first_time:.3f} ms")
print(f"Result shape: {result.shape}")
print(f"First 10 results: {result[:10]}")
print(f"Last 10 results: {result[-10:]}")
print(f"Number of NaN values: {np.sum(np.isnan(result))}")

# Benchmark with proper warmup
print("\nBenchmarking with warmup...")
# Warmup
for _ in range(10):
    _ = my_project.alma(close_data, 9, 0.85, 6.0)

# Time 100 iterations
times = []
for _ in range(100):
    start = time.perf_counter()
    _ = my_project.alma(close_data, 9, 0.85, 6.0)
    end = time.perf_counter()
    times.append((end - start) * 1000)

times = np.array(times)
print(f"\nTiming statistics (100 iterations):")
print(f"Mean: {np.mean(times):.3f} ms")
print(f"Median: {np.median(times):.3f} ms")
print(f"Std Dev: {np.std(times):.3f} ms")
print(f"Min: {np.min(times):.3f} ms")
print(f"Max: {np.max(times):.3f} ms")

# Test with different sizes to see scaling
print("\nTesting scaling behavior:")
for size in [10_000, 100_000, 500_000, 1_000_000]:
    subset = close_data[:size]
    
    # Warmup
    for _ in range(5):
        _ = my_project.alma(subset, 9, 0.85, 6.0)
    
    # Time
    times = []
    for _ in range(20):
        start = time.perf_counter()
        _ = my_project.alma(subset, 9, 0.85, 6.0)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    median_time = np.median(times)
    throughput = size / (median_time * 1000)  # Million elements per second
    print(f"Size {size:>10,}: {median_time:>6.3f} ms ({throughput:>6.1f} M elem/s)")