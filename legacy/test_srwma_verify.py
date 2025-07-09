#!/usr/bin/env python3
"""Verify SRWMA bindings are actually working and not taking shortcuts."""

import numpy as np
import sys

try:
    import my_project as ta
except ImportError:
    print("Module not found")
    sys.exit(1)

print("=== SRWMA Binding Verification ===\n")

# Test 1: Basic functionality
print("1. Basic functionality test:")
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
result = ta.srwma(data, period=3)
print(f"   Input: {data}")
print(f"   Output: {result}")
print(f"   First non-NaN index: {np.where(~np.isnan(result))[0][0] if np.any(~np.isnan(result)) else 'None'}")

# Test 2: Error handling
print("\n2. Error handling test:")
try:
    ta.srwma(np.array([1.0, 2.0]), period=5)
    print("   ERROR: Should have raised ValueError for period > data length")
except ValueError as e:
    print(f"   ✓ Correctly raised: {e}")

# Test 3: Streaming functionality
print("\n3. Streaming test:")
stream = ta.SrwmaStream(period=3)
stream_results = []
for val in data[:6]:
    res = stream.update(val)
    stream_results.append(res)
    print(f"   update({val}) -> {res}")

# Test 4: Batch processing
print("\n4. Batch processing test:")
batch_result = ta.srwma_batch(data, period_range=(2, 4, 1))
print(f"   Batch shape: {batch_result['values'].shape}")
print(f"   Periods: {batch_result['periods']}")
print(f"   First row (period=2): {batch_result['values'][0]}")

# Test 5: Kernel selection
print("\n5. Kernel selection test:")
result_auto = ta.srwma(data, period=3, kernel='auto')
result_scalar = ta.srwma(data, period=3, kernel='scalar')
print(f"   Auto kernel result: {result_auto[4:6]}")
print(f"   Scalar kernel result: {result_scalar[4:6]}")
print(f"   Results match: {np.allclose(result_auto, result_scalar, rtol=1e-10)}")

# Test 6: Leading NaN handling
print("\n6. Leading NaN handling test:")
nan_data = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, 5.0])
nan_result = ta.srwma(nan_data, period=2)
print(f"   Input with NaNs: {nan_data}")
print(f"   Result: {nan_result}")
print(f"   First valid index should be 4 (2 leading NaNs + period=2 + 1): {np.where(~np.isnan(nan_result))[0][0] if np.any(~np.isnan(nan_result)) else 'None'}")

# Test 7: Zero-copy verification
print("\n7. Zero-copy test (modifying input):")
test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
original = test_data.copy()
result = ta.srwma(test_data, period=2)
test_data[0] = 999.0  # Modify input after calling
print(f"   Original: {original}")
print(f"   Modified: {test_data}")
print(f"   Result unchanged: {np.array_equal(result, ta.srwma(original, period=2))}")

# Test 8: Large data performance
print("\n8. Performance test:")
import time
large_data = np.random.randn(100000)
start = time.time()
large_result = ta.srwma(large_data, period=50)
elapsed = time.time() - start
print(f"   Processed 100k points in {elapsed*1000:.2f}ms")
print(f"   Result shape: {large_result.shape}")
print(f"   Non-NaN values: {np.sum(~np.isnan(large_result))}")

print("\n=== All tests completed ===")