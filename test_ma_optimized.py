import time
import numpy as np
import my_project as ta

# Generate test data
data = np.random.randn(1_000_000).astype(np.float64)

# Test with kernel parameter
print("Testing ma function with kernel parameter...")
print("===========================================")

# Test default (Auto kernel)
start = time.perf_counter()
result_auto = ta.ma(data, "sma", 14)
time_auto = (time.perf_counter() - start) * 1000
print(f"Auto kernel: {time_auto:.2f} ms")

# Test with specific kernels
kernels = [None, "scalar", "avx2", "avx512"]
for kernel in kernels:
    try:
        start = time.perf_counter()
        result = ta.ma(data, "sma", 14, kernel)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"Kernel {kernel}: {elapsed:.2f} ms")
    except Exception as e:
        print(f"Kernel {kernel}: Error - {e}")

# Benchmark different MA types with kernel support
print("\n\nBenchmarking MA types with kernel support...")
print("===========================================")

ma_types = ["sma", "ema", "wma", "alma", "hma"]

# Warmup
for _ in range(10):
    _ = ta.ma(data, "sma", 14)

for ma_type in ma_types:
    times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = ta.ma(data, ma_type, 14)
        times.append((time.perf_counter() - start) * 1000)
    
    median_time = np.median(times)
    print(f"{ma_type}: {median_time:.2f} ms")

# Compare with Rust benchmark results
print("\n\nPython vs Rust Performance Comparison")
print("=====================================")
print("MA Type | Python (ms) | Rust (ms) | Overhead")
print("--------|-------------|-----------|----------")

rust_times = {
    "sma": 0.84,
    "ema": 1.07, 
    "wma": 1.20,
    "alma": 1.33,
    "hma": 1.94
}

for ma_type in ma_types:
    times = []
    for _ in range(10):
        start = time.perf_counter()
        _ = ta.ma(data, ma_type, 14)
        times.append((time.perf_counter() - start) * 1000)
    
    python_time = np.median(times)
    rust_time = rust_times.get(ma_type, 0)
    if rust_time > 0:
        overhead = ((python_time - rust_time) / rust_time) * 100
        print(f"{ma_type:7} | {python_time:11.2f} | {rust_time:11.2f} | {overhead:8.1f}%")
    else:
        print(f"{ma_type:7} | {python_time:11.2f} | N/A         | N/A")