import time
import numpy as np
import gc

# Import the module from venv if possible
try:
    import sys
    sys.path.insert(0, '.venv/Lib/site-packages')
    import my_project as ta
except ImportError:
    import my_project as ta

def benchmark_gaussian(data, iterations=100):
    """Benchmark gaussian indicator"""
    # Warmup
    for _ in range(10):
        _ = ta.gaussian(data, 14, 4)
    
    # Disable GC during measurement
    gc.disable()
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = ta.gaussian(data, 14, 4)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    gc.enable()
    
    return np.median(times), np.mean(times), np.std(times)

def benchmark_gaussian_kernel(data, kernel, iterations=100):
    """Benchmark gaussian with specific kernel"""
    # Warmup
    for _ in range(10):
        _ = ta.gaussian(data, 14, 4, kernel=kernel)
    
    gc.disable()
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = ta.gaussian(data, 14, 4, kernel=kernel)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    gc.enable()
    
    return np.median(times), np.mean(times), np.std(times)

def main():
    # Test different data sizes
    sizes = [10_000, 100_000, 1_000_000]
    
    print("Gaussian Python Binding Performance Validation")
    print("=" * 60)
    
    for size in sizes:
        data = np.random.randn(size).astype(np.float64)
        print(f"\nData size: {size:,}")
        print("-" * 40)
        
        # Test auto kernel (default)
        median, mean, std = benchmark_gaussian(data)
        print(f"Auto kernel:   {median:.3f} ms (±{std:.3f} ms)")
        
        # Test specific kernels
        for kernel in ['scalar', 'avx2', 'avx512']:
            try:
                median, mean, std = benchmark_gaussian_kernel(data, kernel)
                print(f"{kernel:6} kernel: {median:.3f} ms (±{std:.3f} ms)")
            except ValueError as e:
                print(f"{kernel:6} kernel: Not available on this CPU")
    
    # Test batch operation
    print("\n\nBatch Operation Test (1M data)")
    print("-" * 40)
    data = np.random.randn(1_000_000).astype(np.float64)
    
    # Warmup
    _ = ta.gaussian_batch(data, (10, 30, 5), (2, 4, 1))
    
    gc.disable()
    start = time.perf_counter()
    result = ta.gaussian_batch(data, (10, 30, 5), (2, 4, 1))
    end = time.perf_counter()
    gc.enable()
    
    batch_time = (end - start) * 1000
    num_combos = len(result['periods'])
    
    print(f"Batch time: {batch_time:.3f} ms for {num_combos} combinations")
    print(f"Average per combo: {batch_time / num_combos:.3f} ms")
    
    # Verify correctness
    print("\nVerifying batch correctness...")
    values = result['values']
    periods = result['periods']
    poles = result['poles']
    
    # Check a few random combinations
    for i in [0, num_combos//2, num_combos-1]:
        period = int(periods[i])
        pole = int(poles[i])
        individual = ta.gaussian(data, period, pole)
        batch_row = values[i]
        if not np.allclose(individual, batch_row, rtol=1e-10):
            print(f"ERROR: Batch row {i} doesn't match individual computation!")
        else:
            print(f"OK: Row {i} (period={period}, poles={pole}) matches")

if __name__ == "__main__":
    main()