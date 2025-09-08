import sys
sys.path.append('tests/python')
import numpy as np
import my_project as ta
from test_utils import load_test_data

# Load test data
data = load_test_data()
high = data['high']
low = data['low']
close = data['close']

# Parameters from test
period = 22
mult = 3.0
use_close = True

# Calculate ATR separately to check it
atr_values = ta.atr(high, low, close, period)
print(f"ATR last 5 values: {atr_values[-5:]}")

# Calculate Chandelier Exit
long_stop, short_stop = ta.chandelier_exit(high, low, close, period, mult, use_close)

# Get last 5 non-NaN short_stop values
non_nan_indices = [i for i in range(len(short_stop)) if not np.isnan(short_stop[i])]
last_5_indices = non_nan_indices[-5:] if len(non_nan_indices) >= 5 else non_nan_indices
actual_short_stops = [short_stop[i] for i in last_5_indices]

print(f"\nShort stop last 5 values:")
for i, val in enumerate(actual_short_stops):
    print(f"  [{i}]: {val:.8f}")

# Check the range of values used
if use_close:
    print(f"\nUsing close prices for high/low calculation")
    for idx in last_5_indices[-1:]:
        start = max(0, idx - period + 1)
        window_close = close[start:idx+1]
        print(f"  Index {idx}: window close range = {window_close.min():.2f} to {window_close.max():.2f}")
        print(f"  ATR[{idx}] = {atr_values[idx]:.2f}")
        print(f"  Expected short stop ~= {window_close.min() + mult * atr_values[idx]:.2f}")
else:
    print(f"\nUsing high/low prices")

# Expected values from test
expected = [68719.23648167, 68705.54391432, 68244.42828185, 67599.49972358, 66883.02246342]
print(f"\nExpected short stop values: {expected}")
print(f"Difference from expected: {[actual_short_stops[i] - expected[i] for i in range(min(len(actual_short_stops), len(expected)))]}")

# Search for indices where short_stop might match expected values
print(f"\nSearching for expected values in short_stop array...")
tolerance = 0.01  # Small tolerance for floating point comparison
expected_indices = []
for exp_val in expected:
    for i, val in enumerate(short_stop):
        if not np.isnan(val) and abs(val - exp_val) < tolerance:
            expected_indices.append(i)
            print(f"  Found {exp_val:.8f} at index {i} (actual: {val:.8f})")
            break

# Check if they're consecutive
if expected_indices:
    print(f"\nExpected values found at indices: {expected_indices}")
    if len(expected_indices) >= 2:
        gaps = [expected_indices[i+1] - expected_indices[i] for i in range(len(expected_indices)-1)]
        print(f"Gaps between indices: {gaps}")
        
    # Show what's at those indices
    print(f"\nData at index {expected_indices[0]}:")
    idx = expected_indices[0]
    print(f"  close[{idx}] = {close[idx]:.2f}")
    print(f"  short_stop[{idx}] = {short_stop[idx]:.8f}")
    
    # Check what happens after these indices
    print(f"\nChecking pattern around indices 15386-15395:")
    for i in range(15385, min(15395, len(short_stop))):
        has_short = not np.isnan(short_stop[i])
        has_long = not np.isnan(long_stop[i])
        print(f"  [{i}]: short={'Y' if has_short else 'N'}, long={'Y' if has_long else 'N'}")
    
    # Show actual values at expected indices
    print(f"\nActual short_stop values at expected indices:")
    for i in expected_indices:
        print(f"  short_stop[{i}] = {short_stop[i]:.8f}")