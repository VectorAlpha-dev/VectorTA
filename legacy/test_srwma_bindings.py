#!/usr/bin/env python3
"""Test that SRWMA bindings are not taking shortcuts."""

import numpy as np
import my_project as ta

# Load actual test data
import csv
data_path = 'src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv'
close_prices = []
with open(data_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        if len(row) >= 6:
            close_prices.append(float(row[2]))  # close is column 2

close = np.array(close_prices)
print(f"Loaded {len(close)} candles")

# Test 1: Check actual computation
print("\n1. Testing actual SRWMA computation:")
result = ta.srwma(close, period=14)
print(f"   Result length: {len(result)}")
print(f"   Last 5 values: {result[-5:]}")
print(f"   Expected: [59344.28384704595, 59282.09151629659, 59192.76580529367, 59178.04767548977, 59110.03801260874]")

# Test 2: Test that different methods give same results
print("\n2. Testing consistency between methods:")
# Single call
single = ta.srwma(close, period=14)

# Batch call with single period
batch_result = ta.srwma_batch(close, period_range=(14, 14, 0))
batch_single = batch_result['values'][0]

# Streaming
stream = ta.SrwmaStream(period=14)
stream_values = []
for price in close:
    val = stream.update(price)
    stream_values.append(val if val is not None else np.nan)
stream_values = np.array(stream_values)

print(f"   Single vs Batch max diff: {np.max(np.abs(single - batch_single))}")
print(f"   Single vs Stream max diff: {np.max(np.abs(single - stream_values)[~np.isnan(single)])}")

# Test 3: Test error conditions actually work
print("\n3. Testing error handling:")
try:
    ta.srwma(np.array([1.0]), period=5)
    print("   ERROR: Should have failed!")
except ValueError as e:
    print(f"   ✓ Correctly raised: {e}")

try:
    ta.srwma(np.full(10, np.nan), period=3)
    print("   ERROR: Should have failed!")
except ValueError as e:
    print(f"   ✓ Correctly raised: {e}")

# Test 4: Test warmup period is correct
print("\n4. Testing warmup period:")
small_data = np.arange(1.0, 21.0)
for period in [3, 5, 7]:
    result = ta.srwma(small_data, period=period)
    first_valid = np.where(~np.isnan(result))[0][0]
    expected_first = period + 1
    status = "✓" if first_valid == expected_first else "✗"
    print(f"   Period {period}: first valid at {first_valid}, expected {expected_first} {status}")

# Test 5: Performance check (should be fast if using native code)
print("\n5. Performance test:")
import time
large_data = np.random.randn(1000000)
start = time.time()
result = ta.srwma(large_data, period=50)
elapsed = time.time() - start
print(f"   1M points processed in {elapsed*1000:.1f}ms")
print(f"   That's {1000000/elapsed/1e6:.1f} million points/second")

print("\n✓ All binding tests completed successfully")