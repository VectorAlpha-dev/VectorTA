import numpy as np
import sys
sys.path.insert(0, '.')
import my_project as ta_indicators

# Generate the same test data as in test_utils
data = np.arange(0.612, 0.60805 - 0.00001, -0.00001)
print(f"Data length: {len(data)}")
print(f"First 5 values: {data[:5]}")
print(f"Last 5 values: {data[-5:]}")

# Run OTTO with default params
hott, lott = ta_indicators.otto(
    data,
    ott_period=2,
    ott_percent=0.6,
    fast_vidya_length=10,
    slow_vidya_length=25,
    correcting_constant=100000,
    ma_type="VAR"
)

print(f"\nHOTT length: {len(hott)}")
print(f"LOTT length: {len(lott)}")

# Find first non-NaN values
first_valid_hott = next((i for i, x in enumerate(hott) if not np.isnan(x)), -1)
first_valid_lott = next((i for i, x in enumerate(lott) if not np.isnan(x)), -1)

print(f"\nFirst valid HOTT at index: {first_valid_hott}")
print(f"First valid LOTT at index: {first_valid_lott}")

# Print last 10 values
print(f"\nLast 10 HOTT values:")
for i in range(-10, 0):
    print(f"  [{i}]: {hott[i]}")

print(f"\nLast 10 LOTT values:")
for i in range(-10, 0):
    print(f"  [{i}]: {lott[i]}")

# Check if values are reasonable
print(f"\nMin HOTT (non-NaN): {np.nanmin(hott)}")
print(f"Max HOTT (non-NaN): {np.nanmax(hott)}")
print(f"Min LOTT (non-NaN): {np.nanmin(lott)}")
print(f"Max LOTT (non-NaN): {np.nanmax(lott)}")