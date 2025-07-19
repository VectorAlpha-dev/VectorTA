import time
import numpy as np
import my_project

# Test data
data = np.random.randn(1_000_000).astype(np.float64)

# Warmup
for _ in range(10):
    _ = my_project.hma(data, 5)

# Benchmark single HMA
times = []
for _ in range(50):
    start = time.perf_counter()
    result = my_project.hma(data, 5)
    times.append((time.perf_counter() - start) * 1000)

print(f"HMA single median time: {np.median(times):.3f} ms")

# Test with kernel parameter
times_kernel = []
for _ in range(50):
    start = time.perf_counter()
    result = my_project.hma(data, 5, kernel="avx2")
    times_kernel.append((time.perf_counter() - start) * 1000)

print(f"HMA with AVX2 kernel: {np.median(times_kernel):.3f} ms")

# Test batch operation
times_batch = []
for _ in range(10):
    start = time.perf_counter()
    result = my_project.hma_batch(data, (5, 25, 1))
    times_batch.append((time.perf_counter() - start) * 1000)

print(f"\nHMA batch (21 periods) median time: {np.median(times_batch):.3f} ms")
print(f"Per-period time: {np.median(times_batch) / 21:.3f} ms")

# Test streaming
stream = my_project.HmaStream(5)
start = time.perf_counter()
for val in data[:1000]:
    stream.update(val)
stream_time = (time.perf_counter() - start) * 1000
print(f"\nHMA streaming (1000 points): {stream_time:.3f} ms")