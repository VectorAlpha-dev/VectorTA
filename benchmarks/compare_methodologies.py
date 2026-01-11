#!/usr/bin/env python3
"""Compare different benchmark methodologies."""
import numpy as np
import time
import my_project
import gc


np.random.seed(42)
data_1m = np.random.randn(1_000_000).astype(np.float64)

print("Comparing Different Benchmark Methodologies")
print("=" * 60)


print("\nMethod 1: Simple timing with warmup")
for _ in range(10):
    _ = my_project.alma(data_1m, 9, 0.85, 6.0)

times = []
for _ in range(100):
    start = time.perf_counter()
    _ = my_project.alma(data_1m, 9, 0.85, 6.0)
    end = time.perf_counter()
    times.append((end - start) * 1000)

print(f"Mean: {np.mean(times):.3f} ms")
print(f"Median: {np.median(times):.3f} ms")


print("\nMethod 2: Criterion-style (GC disabled)")
gc_enabled = gc.isenabled()
gc.disable()


warmup_start = time.perf_counter()
warmup_iters = 0
while (time.perf_counter() - warmup_start) < 0.15:  
    _ = my_project.alma(data_1m, 9, 0.85, 6.0)
    warmup_iters += 1
print(f"Warmup iterations: {warmup_iters}")


samples = []
for _ in range(10):  
    
    batch_size = 10
    start = time.perf_counter()
    for _ in range(batch_size):
        _ = my_project.alma(data_1m, 9, 0.85, 6.0)
    end = time.perf_counter()
    sample_time = ((end - start) / batch_size) * 1000
    samples.append(sample_time)

if gc_enabled:
    gc.enable()

print(f"Mean: {np.mean(samples):.3f} ms")
print(f"Median: {np.median(samples):.3f} ms")


print("\nMethod 3: Including allocation overhead")
times = []
for _ in range(100):
    
    start = time.perf_counter()
    result = my_project.alma(data_1m, 9, 0.85, 6.0)
    _ = np.ascontiguousarray(result)  
    end = time.perf_counter()
    times.append((end - start) * 1000)

print(f"Mean: {np.mean(times):.3f} ms")
print(f"Median: {np.median(times):.3f} ms")


print("\nMethod 4: Real-world pattern (fresh data each time)")
times = []
for _ in range(50):
    
    test_data = np.random.randn(1_000_000).astype(np.float64)
    
    start = time.perf_counter()
    _ = my_project.alma(test_data, 9, 0.85, 6.0)
    end = time.perf_counter()
    times.append((end - start) * 1000)

print(f"Mean: {np.mean(times):.3f} ms")
print(f"Median: {np.median(times):.3f} ms")


print("\nMethod 5: With actual CSV data")
import csv
csv_path = 'src/data/1MillionCandles.csv'
closes = []
with open(csv_path, 'r') as f:
    f.readline()
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 5:
            closes.append(float(row[4]))

close_data = np.array(closes, dtype=np.float64)


for _ in range(10):
    _ = my_project.alma(close_data, 9, 0.85, 6.0)

times = []
for _ in range(100):
    start = time.perf_counter()
    _ = my_project.alma(close_data, 9, 0.85, 6.0)
    end = time.perf_counter()
    times.append((end - start) * 1000)

print(f"Mean: {np.mean(times):.3f} ms")
print(f"Median: {np.median(times):.3f} ms")
print(f"Data size: {len(close_data):,} elements")