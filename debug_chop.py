import numpy as np
import my_project as ta
from test_utils import load_test_data

# Load actual test data
test_data = load_test_data()
high = test_data['high'].copy()
low = test_data['low'].copy()
close = test_data['close'].copy()

n = len(close)
print(f"Loaded test data with {n} points")

# Insert NaN values like the test does
high[10:20] = np.nan
low[15:25] = np.nan
close[5:15] = np.nan

print("Data shape:", n)
print("\nNaN ranges:")
print(f"  high NaN: indices 10-19")
print(f"  low NaN: indices 15-24")
print(f"  close NaN: indices 5-14")

# Find first valid index where all three are not NaN
first_valid = None
for i in range(n):
    if not (np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i])):
        first_valid = i
        break

print(f"\nFirst valid index (all three not NaN): {first_valid}")

# Calculate expected warmup
period = 14
if first_valid is not None:
    warmup_end = first_valid + period - 1
    print(f"Expected warmup end: {warmup_end}")
    print(f"Test checks from index 30, warmup ends at {warmup_end}")

# Run CHOP
result = ta.chop(high, low, close)

print(f"\nResult shape: {result.shape}")
print(f"All NaN? {np.all(np.isnan(result))}")

# Check for first non-NaN value
first_non_nan = None
for i in range(len(result)):
    if not np.isnan(result[i]):
        first_non_nan = i
        break

if first_non_nan is not None:
    print(f"First non-NaN value at index: {first_non_nan}")
else:
    print("No non-NaN values found!")

# Check specific ranges
print(f"\nChecking index 0-20: {result[0:20]}")
print(f"\nChecking index 20-40: {result[20:40]}")
print(f"\nChecking index 30-50: {result[30:50]}")
print(f"Any valid in 30+? {np.any(~np.isnan(result[30:]))}")

# Check where NaN values are in the result
nan_indices = np.where(np.isnan(result))[0]
non_nan_indices = np.where(~np.isnan(result))[0]
print(f"\nNaN count: {len(nan_indices)}")
print(f"Non-NaN count: {len(non_nan_indices)}")
if len(non_nan_indices) > 0:
    print(f"Non-NaN indices: {non_nan_indices[:20]}...")  # First 20
    print(f"Last non-NaN index: {non_nan_indices[-1] if len(non_nan_indices) > 0 else 'None'}")