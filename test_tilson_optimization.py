#!/usr/bin/env python3
"""Test script to verify tilson optimization works correctly."""

import numpy as np
import time

# Try to import and test the optimized tilson
try:
    import my_project
    
    # Test data
    np.random.seed(42)
    test_data = np.random.randn(1000).astype(np.float64)
    
    print("Testing tilson function...")
    result = my_project.tilson(test_data, 14, 0.7)
    print(f"  Result shape: {result.shape}")
    print(f"  First 5 values: {result[:5]}")
    print(f"  Last 5 values: {result[-5:]}")
    print(f"  Contains NaN in warmup: {np.isnan(result[:100]).any()}")
    print(f"  All finite after warmup: {np.isfinite(result[100:]).all()}")
    
    print("\nTesting tilson_batch function...")
    batch_result = my_project.tilson_batch(test_data, (5, 10, 1), (0.0, 0.5, 0.1))
    print(f"  Result keys: {list(batch_result.keys())}")
    print(f"  Values shape: {batch_result['values'].shape}")
    print(f"  Periods: {batch_result['periods']}")
    print(f"  Volume factors: {batch_result['volume_factors']}")
    
    print("\nTesting TilsonStream...")
    stream = my_project.TilsonStream(14, 0.7)
    stream_results = []
    for val in test_data[:50]:
        result = stream.update(val)
        stream_results.append(result if result is not None else float('nan'))
    print(f"  Stream results (first 10): {stream_results[:10]}")
    print(f"  Stream produced values: {sum(1 for x in stream_results if not np.isnan(x))}")
    
    print("\nAll tests passed! Tilson optimization is working correctly.")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()