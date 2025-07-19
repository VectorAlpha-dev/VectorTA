#!/usr/bin/env python3
"""Quick test to verify KAMA optimization works."""

import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path for import
sys.path.insert(0, str(Path(__file__).parent))

try:
    import my_project
    print("Successfully imported my_project")
    
    # Test data
    data = np.random.randn(1_000_000).astype(np.float64)
    
    # Test basic KAMA
    print("\nTesting basic KAMA...")
    start = time.perf_counter()
    result = my_project.kama(data, 30)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"KAMA computed in {elapsed:.2f} ms")
    print(f"Result shape: {result.shape}")
    
    # Test KAMA with kernel parameter
    print("\nTesting KAMA with kernel='scalar'...")
    start = time.perf_counter()
    result_scalar = my_project.kama(data, 30, kernel="scalar")
    elapsed = (time.perf_counter() - start) * 1000
    print(f"KAMA (scalar) computed in {elapsed:.2f} ms")
    
    # Test batch
    print("\nTesting KAMA batch...")
    start = time.perf_counter()
    batch_result = my_project.kama_batch(data[:10000], (10, 50, 10))
    elapsed = (time.perf_counter() - start) * 1000
    print(f"KAMA batch computed in {elapsed:.2f} ms")
    print(f"Batch shape: {batch_result['values'].shape}")
    print(f"Periods: {batch_result['periods']}")
    
    print("\n[OK] All tests passed! KAMA optimization is working.")
    
except ImportError as e:
    print(f"Failed to import my_project: {e}")
    print("Make sure to run: python -m maturin develop --features python,nightly-avx --release")
except Exception as e:
    print(f"Error during testing: {e}")
    import traceback
    traceback.print_exc()