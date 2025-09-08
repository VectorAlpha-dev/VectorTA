import my_project
import numpy as np

# Load test data
data = np.loadtxt('src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv', delimiter=',', skiprows=1)
h = np.ascontiguousarray(data[:, 2])  # high
l = np.ascontiguousarray(data[:, 3])  # low
c = np.ascontiguousarray(data[:, 4])  # close

# Run indicator with default params
u, lo, ut, lt = my_project.fvg_trailing_stop(h, l, c, 5, 9, False)

n = len(lo)
print(f"Total data points: {n}")
print("\nLast 5 values:")

for i in range(5):
    idx = n - 5 + i
    upper_str = f"{u[idx]:.8f}" if not np.isnan(u[idx]) else "NaN"
    lower_str = f"{lo[idx]:.8f}" if not np.isnan(lo[idx]) else "NaN"
    upper_ts_str = f"{ut[idx]:.8f}" if not np.isnan(ut[idx]) else "NaN"
    lower_ts_str = f"{lt[idx]:.8f}" if not np.isnan(lt[idx]) else "NaN"
    
    print(f"Index {idx}: upper={upper_str}, lower={lower_str}, upper_ts={upper_ts_str}, lower_ts={lower_ts_str}")

# Check warmup period
warmup = 2 + 9 - 1  # 2 + smoothing_length - 1
print(f"\nExpected warmup period: {warmup} values")

# Find first non-NaN
first_non_nan = None
for i in range(n):
    if not np.isnan(u[i]) or not np.isnan(lo[i]) or not np.isnan(ut[i]) or not np.isnan(lt[i]):
        first_non_nan = i
        break

print(f"First non-NaN value at index: {first_non_nan}")

# Check if upper/lower are mutually exclusive
print("\nChecking mutual exclusivity (after warmup):")
for i in range(max(warmup, 20), min(n, warmup + 10)):  # Check a few values after warmup
    upper_active = not np.isnan(u[i])
    lower_active = not np.isnan(lo[i])
    if upper_active and lower_active:
        print(f"WARNING: Both upper and lower active at index {i}")