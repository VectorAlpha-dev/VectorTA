import numpy as np
import sys
sys.path.insert(0, '.')

try:
    import my_project
    
    # Test data
    data = np.random.randn(100).astype(np.float64)
    
    # Test batch
    result = my_project.kama_batch(data, (5, 15, 5))
    
    print(f"Batch shape: {result['values'].shape}")
    print(f"Periods: {result['periods']}")
    
    # Check first row for NaN warmup
    first_row = result['values'][0]
    print(f"\nFirst row warmup check (period=5):")
    print(f"First 10 values: {first_row[:10]}")
    print(f"NaN count in first 5: {np.sum(np.isnan(first_row[:5]))}")
    
    # Compare with single calculation
    single_result = my_project.kama(data, 5)
    print(f"\nSingle calculation first 10: {single_result[:10]}")
    
    # Check if they match
    if np.allclose(first_row, single_result, equal_nan=True):
        print("\n[OK] Batch and single calculations match!")
    else:
        print("\n[FAIL] Batch and single calculations don't match!")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()