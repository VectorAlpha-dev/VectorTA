import time
import numpy as np
import my_project

# Create test data
np.random.seed(42)
data = np.random.randn(1_000_000).astype(np.float64)

# Warmup
for _ in range(10):
    _ = my_project.srwma(data, 14)

# Benchmark single run
start = time.perf_counter()
result = my_project.srwma(data, 14)
end = time.perf_counter()
single_time = (end - start) * 1000

print(f"Single run time: {single_time:.3f} ms")

# Check result
print(f"Result shape: {result.shape}")
print(f"First 10 non-NaN values: {result[~np.isnan(result)][:10]}")
print(f"Number of NaN values: {np.sum(np.isnan(result))}")

# Run multiple iterations for better measurement
times = []
for _ in range(100):
    start = time.perf_counter()
    _ = my_project.srwma(data, 14)
    end = time.perf_counter()
    times.append((end - start) * 1000)

print(f"\nMedian time over 100 runs: {np.median(times):.3f} ms")
print(f"Mean time: {np.mean(times):.3f} ms")
print(f"Min time: {np.min(times):.3f} ms")
print(f"Max time: {np.max(times):.3f} ms")

# Test with different kernels
for kernel in ['auto', 'scalar', 'avx2', 'avx512']:
    try:
        times = []
        for _ in range(50):
            start = time.perf_counter()
            _ = my_project.srwma(data, 14, kernel)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        print(f"\nKernel '{kernel}' median time: {np.median(times):.3f} ms")
    except Exception as e:
        print(f"\nKernel '{kernel}' failed: {e}")