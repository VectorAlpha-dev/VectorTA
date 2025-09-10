import sys
sys.path.insert(0, "target/wheels")
import my_project

# Simple test data
data = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0]
volume = [1000.0] * 15  # Constant volume for simplicity

result = my_project.volume_adjusted_ma(data, volume, length=3, vi_factor=0.67, strict=False, sample_period=0)
print("Result:", result[-5:])
