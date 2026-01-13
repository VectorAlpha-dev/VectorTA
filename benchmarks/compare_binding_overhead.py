#!/usr/bin/env python3
"""Compare binding overhead for JSA vs ALMA to understand the issue."""

import time
import numpy as np
import my_project
from pathlib import Path
import csv


csv_path = Path(__file__).parent.parent / 'src/data/1MillionCandles.csv'
closes = []
with open(csv_path, 'r') as f:
    f.readline()
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 7:
            try:
                closes.append(float(row[4]))
            except ValueError:
                continue

data = np.array(closes, dtype=np.float64)
print(f"Loaded {len(data)} candles from CSV\n")

def benchmark_indicator(name, func, *args, **kwargs):
    """Benchmark a single indicator function."""

    for _ in range(10):
        _ = func(data, *args, **kwargs)


    times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = func(data, *args, **kwargs)
        times.append(time.perf_counter() - start)

    times_ms = [t * 1000 for t in times]
    times_ms.sort()
    median = times_ms[len(times_ms) // 2]

    print(f"{name}:")
    print(f"  Median: {median:.3f} ms")
    print(f"  Min: {min(times_ms):.3f} ms")
    print(f"  Max: {max(times_ms):.3f} ms")
    return median


print("JSA Indicator:")
jsa_time = benchmark_indicator("JSA (period=30)", my_project.jsa, 30)


print("\nALMA Indicator:")
alma_time = benchmark_indicator("ALMA (period=9)", my_project.alma, 9, 0.85, 6.0)


print("\nSMA Indicator:")
sma_time = benchmark_indicator("SMA (period=30)", my_project.sma, 30)


print("\nEMA Indicator:")
ema_time = benchmark_indicator("EMA (period=30)", my_project.ema, 30)

print("\nSummary:")
print(f"  JSA:  {jsa_time:.3f} ms")
print(f"  ALMA: {alma_time:.3f} ms")
print(f"  SMA:  {sma_time:.3f} ms")
print(f"  EMA:  {ema_time:.3f} ms")





print("\nOverhead Analysis (based on Rust direct write baseline):")
jsa_rust_baseline = 0.092
print(f"  JSA Rust baseline: {jsa_rust_baseline:.3f} ms")
print(f"  JSA Python: {jsa_time:.3f} ms")
print(f"  JSA Overhead: {((jsa_time - jsa_rust_baseline) / jsa_rust_baseline * 100):.1f}%")

if jsa_time < jsa_rust_baseline * 1.1:
    print("  STATUS: PASS - Overhead is less than 10%")
else:
    print("  STATUS: FAIL - Overhead exceeds 10% target")
    print(f"  Target was: {jsa_rust_baseline * 1.1:.3f} ms")