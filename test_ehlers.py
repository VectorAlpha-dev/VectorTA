import my_project
import numpy as np

# Read CSV manually
with open('src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv') as f:
    lines = f.readlines()[1:]
    close = [float(line.split(',')[4]) for line in lines]

# Convert to numpy array
close_arr = np.array(close)

# Calculate ehlers_pma
predict, trigger = my_project.ehlers_pma(close_arr)

print('Total length:', len(predict))
print('Last 5 predict values:', predict[-5:])
print('Last 6 values, skip last one:', predict[-6:-1])