import numpy as np
import my_project

# Test data from the test
data = [
    59949.97, 59966.42, 60163.65, 60362.22, 60405.83,
    60426.37, 60331.36, 60288.69, 60125.09, 60063.62,
    59947.23, 59764.81, 59726.84, 59605.10, 59548.58,
    59532.58, 59444.31, 59401.41, 59335.09, 59219.26,
    59092.62, 59066.76, 58911.80, 58893.76, 58935.84,
    58998.39, 59145.61, 59265.70, 59379.66, 59411.90,
    59512.25, 59642.32, 59682.52, 59763.20, 59784.75,
    59744.66, 59698.14, 59661.39, 59467.62, 59376.13,
    59311.42, 59175.62, 59086.03, 59033.63, 58964.81,
    58915.34, 58940.84, 58950.47, 58991.36, 59092.21,
]

# Expected values from PineScript
expected = [59155.78388366, 59188.30482025, 59204.22918387, 59193.09295559, 59181.69474570]

# Calculate UMA
result = my_project.uma(
    data=np.array(data, dtype=np.float64),
    accelerator=1.0,
    min_length=5,
    max_length=50,
    smooth_length=4
)

# Get last 5 non-NaN values
valid_vals = [v for v in result if not np.isnan(v)]
last_5 = valid_vals[-5:] if len(valid_vals) >= 5 else valid_vals

print("Input data (last 10):", data[-10:])
print()
print("Expected last 5 values:", expected)
print("Got last 5 values:    ", last_5)
print()

# Check differences
for i, (got, exp) in enumerate(zip(last_5, expected)):
    diff = abs(got - exp)
    pct_diff = (diff / exp) * 100
    print(f"Index {i}: got={got:.8f}, expected={exp:.8f}, diff={diff:.8f} ({pct_diff:.2f}%)")

# Debug: check what momentum factor would be
print("\nDebug momentum factor calculation:")
for price in [58915.34, 58940.84, 58950.47, 58991.36, 59092.21]:
    # Pine formula: mf = 100 - 100 / (1 + src / len)
    # Using len=50 as max_length
    mf_pine = 100 - 100 / (1 + price / 50)
    print(f"Price {price}: mf_pine = {mf_pine:.2f}")