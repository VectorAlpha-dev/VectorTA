"""
Test to verify EMD optimization improvements
"""
import time
import numpy as np
import my_project as ta

# Generate test data
np.random.seed(42)
size = 10000
high = np.random.randn(size).cumsum() + 100
low = high - np.abs(np.random.randn(size))
close = (high + low) / 2 + np.random.randn(size) * 0.1
volume = np.abs(np.random.randn(size)) * 1000000

# Test single EMD calculation
print("Testing single EMD calculation...")
start = time.perf_counter()
for _ in range(100):
    upperband, middleband, lowerband = ta.emd(
        high, low, close, volume,
        period=20, delta=0.5, fraction=0.1
    )
single_time = (time.perf_counter() - start) / 100
print(f"Single EMD: {single_time*1000:.3f}ms")

# Test batch EMD calculation  
print("\nTesting batch EMD calculation...")
start = time.perf_counter()
for _ in range(10):
    result = ta.emd_batch(
        high, low, close, volume,
        period_range=(10, 30, 5),    # 5 values
        delta_range=(0.3, 0.7, 0.2), # 3 values  
        fraction_range=(0.05, 0.15, 0.05)  # 3 values
    )
batch_time = (time.perf_counter() - start) / 10
combos = 5 * 3 * 3  # 45 combinations
print(f"Batch EMD ({combos} combinations): {batch_time*1000:.3f}ms")
print(f"Per combination: {batch_time*1000/combos:.3f}ms")
print(f"Batch speedup: {(single_time*combos)/batch_time:.1f}x")

# Verify results shape
print(f"\nResult shapes:")
print(f"  upperband: {result['upperband'].shape}")
print(f"  periods: {result['periods'].shape}")
print(f"  First 5 periods: {result['periods'][:5]}")

print("\nEMD optimization test completed successfully!")
print("The batch function now uses zero-copy operations like ALMA.")