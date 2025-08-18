import numpy as np
import my_project as ta

# Simple test without NaN
np.random.seed(42)
n = 100
high = 100 + np.cumsum(np.random.randn(n))
low = high - np.abs(np.random.randn(n))
close = low + np.random.rand(n) * (high - low)

print("Test 1: No NaN values")
result = ta.chop(high, low, close)
print(f"Result shape: {result.shape}")
print(f"All NaN? {np.all(np.isnan(result))}")
non_nan = np.where(~np.isnan(result))[0]
print(f"Non-NaN count: {len(non_nan)}")
if len(non_nan) > 0:
    print(f"First non-NaN at: {non_nan[0]}, Last at: {non_nan[-1]}")
    print(f"First few values: {result[:20]}")

print("\n" + "="*50 + "\n")

# Test with NaN in middle
high2 = high.copy()
low2 = low.copy()
close2 = close.copy()

# Add small NaN gap
high2[20:25] = np.nan
low2[20:25] = np.nan
close2[20:25] = np.nan

print("Test 2: With NaN gap at indices 20-24")
result2 = ta.chop(high2, low2, close2)
print(f"Result shape: {result2.shape}")
print(f"All NaN? {np.all(np.isnan(result2))}")
non_nan2 = np.where(~np.isnan(result2))[0]
print(f"Non-NaN count: {len(non_nan2)}")
if len(non_nan2) > 0:
    print(f"First non-NaN at: {non_nan2[0]}, Last at: {non_nan2[-1]}")
    print(f"Values around gap (15-30): {result2[15:30]}")