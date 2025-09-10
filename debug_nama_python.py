import my_project as ta
import numpy as np
import csv

# Load CSV data
data = []
with open('src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        if len(row) >= 6:
            data.append(float(row[2]))  # close price

close = np.array(data)
print(f"Loaded {len(close)} close prices")
print(f"Last 5 close prices: {close[-5:]}")

# Calculate NAMA with period=30
result = ta.nama(close, period=30)
print(f"\nNAMA result length: {len(result)}")
print(f"Last 5 NAMA values: {result[-5:]}")

# Expected values from Rust
expected = [
    59304.88975909,
    59283.51109653,
    59243.52850894,
    59228.86200178,
    59137.33546742
]
print(f"\nExpected last 5: {expected}")

# Check differences
print("\nDifferences:")
for i in range(5):
    diff = result[-(5-i)] - expected[i]
    print(f"  [{i}]: {result[-(5-i)]:.8f} vs {expected[i]:.8f}, diff: {diff:.8f}")