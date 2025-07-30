#!/usr/bin/env python3
"""Simple test to check if VAR indicator compiles and runs"""

import sys
import numpy as np

# Try to import the module
try:
    import my_project
    print("✓ Module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import module: {e}")
    print("  Run: maturin develop --features python")
    sys.exit(1)

# Test data
test_data = np.array([100.0, 101.0, 99.5, 102.0, 100.5, 101.5, 99.0, 102.5, 101.0, 100.0,
                      101.5, 99.5, 102.0, 100.5, 101.0], dtype=np.float64)

# Test 1: Basic VAR calculation
try:
    result = my_project.var(test_data, 5, 1.0)
    print(f"✓ Basic VAR calculation: {len(result)} values")
    print(f"  First non-NaN value: {result[4]:.4f}")
except Exception as e:
    print(f"✗ Basic VAR failed: {e}")

# Test 2: VAR with default parameters
try:
    result = my_project.var(test_data)
    print(f"✓ VAR with defaults: {len(result)} values")
except Exception as e:
    print(f"✗ VAR with defaults failed: {e}")

# Test 3: VarStream
try:
    stream = my_project.VarStream(5, 1.0)
    results = []
    for val in test_data[:6]:
        results.append(stream.update(val))
    print(f"✓ VarStream: {sum(1 for r in results if r is not None)} non-None values")
except Exception as e:
    print(f"✗ VarStream failed: {e}")

# Test 4: var_batch
try:
    batch_result = my_project.var_batch(
        test_data,
        period_range=(3, 5, 1),
        nbdev_range=(1.0, 1.0, 0.0)
    )
    print(f"✓ var_batch: {batch_result['values'].shape} shape")
    print(f"  Periods: {list(batch_result['periods'])}")
except Exception as e:
    print(f"✗ var_batch failed: {e}")

print("\nAll tests completed!")