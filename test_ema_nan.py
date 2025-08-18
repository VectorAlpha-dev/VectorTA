import numpy as np
import my_project as ta

# Generate test data
np.random.seed(42)
n = 100
data = 100 + np.cumsum(np.random.randn(n))

# Insert NaN values in the middle
data[10:20] = np.nan

print("Data shape:", n)
print("NaN range: indices 10-19")

# Run EMA
result = ta.ema(data, 14)

print(f"\nResult shape: {result.shape}")
print(f"All NaN? {np.all(np.isnan(result))}")

# Check where NaN values are in the result
nan_indices = np.where(np.isnan(result))[0]
non_nan_indices = np.where(~np.isnan(result))[0]
print(f"\nNaN count: {len(nan_indices)}")
print(f"Non-NaN count: {len(non_nan_indices)}")

if len(non_nan_indices) > 0:
    print(f"First non-NaN index: {non_nan_indices[0]}")
    print(f"Last non-NaN index: {non_nan_indices[-1]}")
    
print(f"\nChecking index 0-30: {result[0:30]}")
print(f"Any valid after index 20? {np.any(~np.isnan(result[20:]))}")