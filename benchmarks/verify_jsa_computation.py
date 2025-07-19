#!/usr/bin/env python3
"""Verify JSA computation is actually happening."""

import numpy as np
import my_project
import hashlib

# Create test data
data = np.array([50000.0 + np.sin(i) * 1000.0 for i in range(100)], dtype=np.float64)

# Run JSA
result = my_project.jsa(data, 30)

# Verify result
print("Input data shape:", data.shape)
print("Output shape:", result.shape)
print("First 10 input values:", data[:10])
print("First 10 output values:", result[:10])
print("Last 10 output values:", result[-10:])

# Verify warmup period
nan_count = np.isnan(result).sum()
print(f"\nNaN count (warmup period): {nan_count}")
print(f"Expected warmup period: 30")

# Verify computation is correct
# JSA formula: (data[i] + data[i-period]) / 2
period = 30
for i in range(period, min(period + 5, len(data))):
    expected = (data[i] + data[i - period]) / 2
    actual = result[i]
    print(f"Index {i}: expected={expected:.6f}, actual={actual:.6f}, match={np.isclose(expected, actual)}")

# Hash the result to ensure it's deterministic
result_hash = hashlib.md5(result.tobytes()).hexdigest()
print(f"\nResult hash: {result_hash}")