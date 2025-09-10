import numpy as np
import sys
sys.path.insert(0, '.')
import my_project as ta_indicators

# Test with real-world price data (Bitcoin-like prices)
np.random.seed(42)
prices = 30000 + np.cumsum(np.random.randn(500) * 100)
print(f"Price range: {prices.min():.2f} to {prices.max():.2f}")
print(f"First 5 prices: {prices[:5]}")

# Run OTTO with default params
hott, lott = ta_indicators.otto(
    prices,
    ott_period=2,
    ott_percent=0.6,
    fast_vidya_length=10,
    slow_vidya_length=25,
    correcting_constant=100000,
    ma_type="VAR"
)

# Find first non-NaN values
first_valid_hott = next((i for i, x in enumerate(hott) if not np.isnan(x)), -1)
first_valid_lott = next((i for i, x in enumerate(lott) if not np.isnan(x)), -1)

print(f"\nFirst valid HOTT at index: {first_valid_hott}")
print(f"First valid LOTT at index: {first_valid_lott}")

# Print some values around index 250
if len(hott) > 250:
    print(f"\nValues around index 250:")
    for i in range(245, 255):
        print(f"  [{i}]: price={prices[i]:.2f}, HOTT={hott[i]:.6f}, LOTT={lott[i]:.6f}")

# Print last 5 values
print(f"\nLast 5 values:")
for i in range(-5, 0):
    print(f"  [{i}]: price={prices[i]:.2f}, HOTT={hott[i]:.6f}, LOTT={lott[i]:.6f}")

# Check if values are reasonable
print(f"\nMin HOTT (non-NaN): {np.nanmin(hott):.6f}")
print(f"Max HOTT (non-NaN): {np.nanmax(hott):.6f}")
print(f"Min LOTT (non-NaN): {np.nanmin(lott):.6f}")
print(f"Max LOTT (non-NaN): {np.nanmax(lott):.6f}")