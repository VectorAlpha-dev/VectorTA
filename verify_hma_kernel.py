import numpy as np
import my_project
import time

# Create test data
data = np.random.randn(1_000_000).astype(np.float64)

# Test different kernels
kernels = [None, "auto", "scalar", "avx2", "avx512"]

print("Testing HMA with different kernels (1M points):")
print("-" * 50)

for kernel in kernels:
    times = []
    
    # Warmup
    for _ in range(5):
        if kernel is None:
            _ = my_project.hma(data, 5)
        else:
            _ = my_project.hma(data, 5, kernel=kernel)
    
    # Benchmark
    for _ in range(20):
        start = time.perf_counter()
        if kernel is None:
            result = my_project.hma(data, 5)
        else:
            result = my_project.hma(data, 5, kernel=kernel)
        times.append((time.perf_counter() - start) * 1000)
    
    kernel_name = "default" if kernel is None else kernel
    print(f"Kernel {kernel_name:8s}: {np.median(times):6.3f} ms (min: {np.min(times):6.3f}, max: {np.max(times):6.3f})")

# Also test batch with kernels
print("\nTesting HMA batch with different kernels:")
print("-" * 50)

for kernel in [None, "auto", "scalarbatch", "avx2batch", "avx512batch"]:
    times = []
    
    # Only 5 iterations for batch since it's slower
    for _ in range(5):
        start = time.perf_counter()
        if kernel is None:
            result = my_project.hma_batch(data, (5, 25, 5))
        else:
            result = my_project.hma_batch(data, (5, 25, 5), kernel=kernel)
        times.append((time.perf_counter() - start) * 1000)
    
    kernel_name = "default" if kernel is None else kernel
    periods_per_batch = len(range(5, 26, 5))
    print(f"Kernel {kernel_name:12s}: {np.median(times):6.3f} ms for {periods_per_batch} periods ({np.median(times)/periods_per_batch:6.3f} ms/period)")