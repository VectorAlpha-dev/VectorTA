"""Quick script to get ehlers_ecema reference values"""
import numpy as np
import sys
sys.path.insert(0, '.')

try:
    import my_project as ta_indicators
except ImportError:
    print("Module not found, trying to build...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "maturin"], check=True)
    subprocess.run([sys.executable, "-m", "maturin", "develop", "--features", "python", "--release"], check=True)
    import my_project as ta_indicators

# Test data from Rust unit test
data = np.array([
    59000.0, 59100.0, 59200.0, 59300.0, 59400.0,
    59350.0, 59300.0, 59250.0, 59200.0, 59150.0,
    59100.0, 59050.0, 59000.0, 58950.0, 58900.0,
    58850.0, 58800.0, 58750.0, 58700.0, 58650.0,
    59600.0, 59550.0, 59500.0, 59450.0, 59400.0,
])

# Test regular mode (default: pine_compatible=False, confirmed_only=False)
print("Regular mode test (length=20, gain_limit=50):")
result = ta_indicators.ehlers_ecema(data, 20, 50)
print(f"Length: {len(result)}")
print(f"First 5 values: {result[:5]}")
print(f"Last 5 values: {result[-5:]}")
print(f"First valid value index: {np.where(~np.isnan(result))[0][0] if np.any(~np.isnan(result)) else 'None'}")

# Test with Pine mode if available
print("\nPine mode test (length=20, gain_limit=50, pine_compatible=True):")
try:
    result_pine = ta_indicators.ehlers_ecema(data, 20, 50, pine_compatible=True)
    print(f"First 5 values: {result_pine[:5]}")
    print(f"Last 5 values: {result_pine[-5:]}")
    print(f"First valid value index: {np.where(~np.isnan(result_pine))[0][0] if np.any(~np.isnan(result_pine)) else 'None'}")
except TypeError as e:
    print(f"Pine mode not available: {e}")

# Test with smaller period for re-input test
print("\nSmaller period test (length=10, gain_limit=30):")
result_small = ta_indicators.ehlers_ecema(data, 10, 30)
print(f"Last 5 values: {result_small[-5:]}")

# Test re-input
print("\nRe-input test (applying ECEMA twice with length=10, gain_limit=30):")
first_pass = ta_indicators.ehlers_ecema(data, 10, 30)
second_pass = ta_indicators.ehlers_ecema(first_pass, 10, 30)
print(f"Second pass last 5 values: {second_pass[-5:]}")

# Test streaming
print("\nStreaming test (length=20, gain_limit=50):")
stream = ta_indicators.EhlersEcemaStream(20, 50)
stream_results = []
for val in data:
    # Try both methods
    try:
        result = stream.update(val)  # ALMA-style
        stream_results.append(result if result is not None else np.nan)
    except AttributeError:
        result = stream.next(val)  # Current style
        stream_results.append(result)

print(f"Stream last 5 values: {stream_results[-5:]}")
print(f"Batch last 5 values: {result[-5:]}")

# Batch test
print("\nBatch test:")
batch_result = ta_indicators.ehlers_ecema_batch(
    data,
    (15, 25, 5),  # length_range: 15, 20, 25
    (40, 60, 10)  # gain_limit_range: 40, 50, 60
)
print(f"Batch dimensions: rows={batch_result['rows']}, cols={batch_result['cols']}")
print(f"Number of combinations: {batch_result['rows']}")

# Extract default params row (length=20, gain_limit=50)
# Row index should be: length_idx=1 (20 is second), gain_idx=1 (50 is second)
# So row = 1 * 3 + 1 = 4
default_row_idx = 4
default_row_start = default_row_idx * len(data)
default_row_end = default_row_start + len(data)
default_row = batch_result['values'][default_row_start:default_row_end]
print(f"Default params row last 5: {default_row[-5:]}")