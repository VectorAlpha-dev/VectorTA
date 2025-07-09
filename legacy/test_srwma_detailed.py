#!/usr/bin/env python3
"""Detailed SRWMA test to check for issues."""

import numpy as np
import my_project as ta

print("=== Detailed SRWMA Test ===\n")

# Issue 1: Leading NaN calculation seems off
print("1. Leading NaN warmup calculation:")
data_with_nans = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
result = ta.srwma(data_with_nans, period=3)
print(f"   Data: {data_with_nans}")
print(f"   Result: {result}")
print(f"   Expected first valid at index 5 (first_non_nan=2 + period=3 + 1 = 6)")
print(f"   Actual first valid at index: {np.where(~np.isnan(result))[0][0] if np.any(~np.isnan(result)) else 'None'}")

# Issue 2: Kernel matching
print("\n2. Kernel precision test:")
data = np.random.randn(100)
r1 = ta.srwma(data, period=5, kernel='auto')
r2 = ta.srwma(data, period=5, kernel='scalar')
max_diff = np.max(np.abs(r1 - r2)[~np.isnan(r1)])
print(f"   Max difference between auto and scalar: {max_diff}")
print(f"   Are they exactly equal? {np.array_equal(r1, r2, equal_nan=True)}")

# Issue 3: Warmup period verification
print("\n3. Warmup period test:")
for period in [2, 3, 5, 10]:
    data = np.ones(20)
    result = ta.srwma(data, period=period)
    first_valid = np.where(~np.isnan(result))[0][0] if np.any(~np.isnan(result)) else None
    expected = period + 1
    print(f"   Period {period}: first valid at {first_valid}, expected at {expected} - {'✓' if first_valid == expected else '✗'}")

# Issue 4: Test streaming more thoroughly
print("\n4. Streaming vs batch comparison:")
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
batch = ta.srwma(data, period=3)

stream = ta.SrwmaStream(period=3)
stream_results = []
for val in data:
    res = stream.update(val)
    stream_results.append(res if res is not None else np.nan)

stream_results = np.array(stream_results)
print(f"   Batch:  {batch}")
print(f"   Stream: {stream_results}")
print(f"   Match: {np.allclose(batch, stream_results, rtol=1e-9, equal_nan=True)}")

# Issue 5: Check actual calculation
print("\n5. Manual calculation verification:")
# For SRWMA with period 3, weights are sqrt(3), sqrt(2), sqrt(1) = 1.732, 1.414, 1.0
# Normalized: divide by sum = 4.146
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
result = ta.srwma(data, period=3)
print(f"   Data: {data}")
print(f"   Result: {result}")

# Manual calc for index 3 (value 4.0):
# Uses values [2.0, 3.0, 4.0] with weights [sqrt(3), sqrt(2), sqrt(1)]
w1, w2, w3 = np.sqrt(3), np.sqrt(2), np.sqrt(1)
norm = w1 + w2 + w3
manual = (2.0 * w1 + 3.0 * w2 + 4.0 * w3) / norm
print(f"   Manual calc for index 3: {manual}")
print(f"   Actual result[3]: {result[3]}")
print(f"   Difference: {abs(manual - result[3]) if ~np.isnan(result[3]) else 'Result is NaN'}")

# Issue 6: Test the batch range handling
print("\n6. Batch metadata test:")
metadata = ta.srwma_batch(data, period_range=(3, 5, 1))
print(f"   Requested periods (3, 5, 1): {metadata['periods']}")
print(f"   Values shape: {metadata['values'].shape}")

print("\n=== Test completed ===")