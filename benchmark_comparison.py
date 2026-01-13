import time
import numpy as np
import my_project as ta
import subprocess
import json


data = np.array([(np.sin(i * 0.123)) for i in range(1_000_000)], dtype=np.float64)

print("Running comprehensive benchmark comparison...")
print("=" * 60)


print("\n1. Running Rust benchmark...")
rust_result = subprocess.run(
    ["cargo", "run", "--release", "--example", "bench_ma"],
    capture_output=True,
    text=True
)


rust_times = {}
for line in rust_result.stdout.split('\n'):
    for ma_type in ["sma", "ema", "wma", "alma", "hma"]:
        if line.startswith(f"{ma_type}:"):
            rust_times[ma_type] = float(line.split(":")[1].strip().split()[0])

print("\nRust benchmark times:")
for ma_type, time_ms in rust_times.items():
    print(f"  {ma_type}: {time_ms:.2f} ms")


print("\n2. Running Python benchmark...")
print("  Warming up...")
for _ in range(10):
    _ = ta.ma(data, "sma", 14)

python_times = {}
for ma_type in ["sma", "ema", "wma", "alma", "hma"]:
    times = []
    for _ in range(10):
        start = time.perf_counter()
        _ = ta.ma(data, ma_type, 14)
        times.append((time.perf_counter() - start) * 1000)

    times.sort()
    python_times[ma_type] = times[len(times) // 2]

print("\nPython benchmark times:")
for ma_type, time_ms in python_times.items():
    print(f"  {ma_type}: {time_ms:.2f} ms")


print("\n3. Performance Comparison")
print("=" * 60)
print("MA Type | Rust (ms) | Python (ms) | Overhead | Status")
print("--------|-----------|-------------|----------|--------")

all_pass = True
for ma_type in ["sma", "ema", "wma", "alma", "hma"]:
    rust_time = rust_times.get(ma_type, 0)
    python_time = python_times.get(ma_type, 0)

    if rust_time > 0:
        overhead = ((python_time - rust_time) / rust_time) * 100
        status = "PASS" if overhead < 10 else "FAIL"
        if overhead >= 10:
            all_pass = False
        print(f"{ma_type:7} | {rust_time:9.2f} | {python_time:11.2f} | {overhead:8.1f}% | {status}")
    else:
        print(f"{ma_type:7} | N/A       | {python_time:11.2f} | N/A      | N/A")

print("\n" + "=" * 60)
if all_pass:
    print("✓ ALL TESTS PASS: Python binding overhead is <10% for all MA types")
else:
    print("✗ SOME TESTS FAIL: Python binding overhead exceeds 10% for some MA types")


print("\n4. Kernel Selection Performance")
print("=" * 60)
kernels = [None, "scalar", "avx2", "avx512"]
for kernel in kernels:
    try:
        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = ta.ma(data, "sma", 14, kernel)
            times.append((time.perf_counter() - start) * 1000)
        median_time = sorted(times)[len(times) // 2]
        print(f"Kernel {str(kernel):7}: {median_time:.2f} ms")
    except Exception as e:
        print(f"Kernel {str(kernel):7}: Error - {e}")