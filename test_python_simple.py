#!/usr/bin/env python3
"""
Simple test script to verify Python bindings work correctly
"""

import numpy as np

# This assumes you've built and installed the module with:
# maturin develop --features python

try:
    import ta_indicators
    print("✓ Successfully imported ta_indicators module")
except ImportError as e:
    print(f"✗ Failed to import ta_indicators: {e}")
    print("\nMake sure to run: maturin develop --features python")
    exit(1)

# Test data
data = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
                 110.0, 120.0, 130.0, 140.0, 150.0], dtype=np.float64)

print("\nTesting Z-Score indicator...")
try:
    # Test zscore function
    result = ta_indicators.zscore(data, period=14, ma_type="sma", nbdev=1.0, devtype=0)
    print(f"✓ zscore returned array of length {len(result)}")
    print(f"  First few values: {result[:5]}")
    print(f"  Last few values: {result[-5:]}")
except Exception as e:
    print(f"✗ zscore failed: {e}")

print("\nTesting Z-Score stream...")
try:
    # Test zscore stream
    stream = ta_indicators.ZscoreStream(period=10, ma_type="sma", nbdev=1.0, devtype=0)
    results = []
    for val in data[:12]:
        results.append(stream.update(val))
    print(f"✓ ZscoreStream processed {len(results)} values")
    print(f"  Results: {results}")
except Exception as e:
    print(f"✗ ZscoreStream failed: {e}")

print("\nTesting ALMA indicator...")
try:
    # Test alma function
    result = ta_indicators.alma(data, period=9, offset=0.85, sigma=6.0)
    print(f"✓ alma returned array of length {len(result)}")
    print(f"  First few values: {result[:5]}")
    print(f"  Last few values: {result[-5:]}")
except Exception as e:
    print(f"✗ alma failed: {e}")

print("\nTesting ALMA stream...")
try:
    # Test alma stream
    stream = ta_indicators.AlmaStream(period=9, offset=0.85, sigma=6.0)
    results = []
    for val in data[:12]:
        results.append(stream.update(val))
    print(f"✓ AlmaStream processed {len(results)} values")
    print(f"  Results: {results}")
except Exception as e:
    print(f"✗ AlmaStream failed: {e}")

print("\nAll tests completed!")