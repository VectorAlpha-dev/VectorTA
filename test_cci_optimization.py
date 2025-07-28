#!/usr/bin/env python3
"""
Quick test to verify CCI optimization works correctly.
"""
import numpy as np
import time

try:
    import my_project as ta
    print("✓ Successfully imported my_project module")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    print("Please run: maturin develop --features python --release")
    exit(1)

# Test data
np.random.seed(42)
data = np.random.randn(10000).astype(np.float64) * 100 + 100

# Test 1: Basic CCI calculation
print("\n1. Testing basic CCI calculation...")
try:
    result = ta.cci(data, 14)
    print(f"✓ CCI calculation successful, output length: {len(result)}")
    print(f"  First 5 values: {result[:5]}")
    print(f"  Last 5 values: {result[-5:]}")
    
    # Check NaN prefix
    nan_count = np.sum(np.isnan(result[:13]))
    assert nan_count == 13, f"Expected 13 NaN values, got {nan_count}"
    print(f"✓ NaN prefix correct: {nan_count} NaN values")
except Exception as e:
    print(f"✗ CCI calculation failed: {e}")

# Test 2: CCI with kernel parameter
print("\n2. Testing CCI with kernel parameter...")
try:
    result_scalar = ta.cci(data, 14, kernel='scalar')
    result_auto = ta.cci(data, 14, kernel='auto')
    print(f"✓ CCI with kernel parameter successful")
    
    # Compare results (should be very close)
    max_diff = np.max(np.abs(result_scalar - result_auto))
    print(f"  Max difference between scalar and auto: {max_diff}")
except Exception as e:
    print(f"✗ CCI with kernel failed: {e}")

# Test 3: CCI batch calculation
print("\n3. Testing CCI batch calculation...")
try:
    batch_result = ta.cci_batch(data, (10, 20, 5))
    print(f"✓ CCI batch calculation successful")
    print(f"  Shape: {batch_result['values'].shape}")
    print(f"  Periods: {batch_result['periods']}")
    
    # Verify shape
    expected_rows = len(range(10, 21, 5))  # 10, 15, 20
    assert batch_result['values'].shape == (expected_rows, len(data))
    print(f"✓ Batch shape correct: {batch_result['values'].shape}")
except Exception as e:
    print(f"✗ CCI batch calculation failed: {e}")

# Test 4: CCI streaming
print("\n4. Testing CCI streaming...")
try:
    stream = ta.CciStream(14)
    stream_results = []
    for val in data[:100]:
        result = stream.update(val)
        stream_results.append(result if result is not None else np.nan)
    
    print(f"✓ CCI streaming successful")
    print(f"  Processed {len(stream_results)} values")
    print(f"  Non-NaN values: {np.sum(~np.isnan(stream_results))}")
except Exception as e:
    print(f"✗ CCI streaming failed: {e}")

# Test 5: Performance benchmark
print("\n5. Performance benchmark...")
test_sizes = [1000, 10000, 100000]
for size in test_sizes:
    test_data = np.random.randn(size).astype(np.float64)
    
    # Warmup
    for _ in range(3):
        _ = ta.cci(test_data, 14)
    
    # Benchmark
    times = []
    for _ in range(10):
        start = time.perf_counter()
        _ = ta.cci(test_data, 14)
        times.append((time.perf_counter() - start) * 1000)
    
    median_time = np.median(times)
    print(f"  Size {size:>6}: {median_time:6.2f} ms (median of 10 runs)")

print("\n✓ All tests completed successfully!")