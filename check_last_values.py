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

print(f"Data length: {len(close)}")
print(f"Last 5 close prices: {close[-5:]}")

# Parameters from test
period = 22
mult = 3.0
use_close = True

# Calculate Chandelier Exit
long_stop, short_stop = ta.chandelier_exit(high, low, close, period, mult, use_close)

# Get last 5 non-NaN short_stop values
non_nan_indices = [i for i in range(len(short_stop)) if not np.isnan(short_stop[i])]
print(f"\nTotal non-NaN short_stop values: {len(non_nan_indices)}")

if non_nan_indices:
    last_5_indices = non_nan_indices[-5:] if len(non_nan_indices) >= 5 else non_nan_indices
    
    print(f"\nLast 5 non-NaN short_stop indices: {last_5_indices}")
    print("\nLast 5 non-NaN short_stop values:")
    for idx in last_5_indices:
        print(f"  short_stop[{idx}] = {short_stop[idx]:.8f}")
    
    # Also check long_stop at those indices
    print("\nCorresponding long_stop values at those indices:")
    for idx in last_5_indices:
        print(f"  long_stop[{idx}] = {'NaN' if np.isnan(long_stop[idx]) else f'{long_stop[idx]:.8f}'}")

# Get last 5 non-NaN long_stop values  
non_nan_long_indices = [i for i in range(len(long_stop)) if not np.isnan(long_stop[i])]
if non_nan_long_indices:
    last_5_long_indices = non_nan_long_indices[-5:] if len(non_nan_long_indices) >= 5 else non_nan_long_indices
    
    print(f"\nLast 5 non-NaN long_stop indices: {last_5_long_indices}")
    print("\nLast 5 non-NaN long_stop values:")
    for idx in last_5_long_indices:
        print(f"  long_stop[{idx}] = {long_stop[idx]:.8f}")

# Check the very last values (regardless of NaN)
print("\nLast 10 values (including NaN):")
for i in range(max(0, len(short_stop)-10), len(short_stop)):
    s_val = 'NaN' if np.isnan(short_stop[i]) else f'{short_stop[i]:.8f}'
    l_val = 'NaN' if np.isnan(long_stop[i]) else f'{long_stop[i]:.8f}'
    print(f"  [{i}]: short={s_val}, long={l_val}, close={close[i]:.2f}")