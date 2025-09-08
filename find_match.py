import my_project
import numpy as np

# Load all data
lines = open('src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv').readlines()[1:]
all_close = [float(line.split(',')[4]) for line in lines]

# Expected values
target = 59161.970663

print(f"Total data points: {len(all_close)}")
print("Searching for data length that produces expected values...")

# Try different data lengths
for length in range(15000, len(all_close), 10):
    close = np.array(all_close[:length])
    predict, trigger = my_project.ehlers_pma(close)
    
    # Check slice -6:-1
    if len(predict) >= 6:
        slice_vals = predict[-6:-1]
        if abs(slice_vals[0] - target) < 0.01:
            print(f"\nFound match at data length {length}!")
            print(f"Slice -6:-1: {slice_vals}")
            print(f"Expected: [59161.970663, 59240.517857, 59260.298469, 59225.190051, 59192.784439]")
            break
else:
    print("\nNo match found")