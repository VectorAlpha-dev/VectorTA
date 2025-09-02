import pandas as pd
import numpy as np

# Load data
csv_path = 'src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv'
# Correct column order: timestamp, open, close, high, low, volume
df = pd.read_csv(csv_path, header=None, names=['timestamp', 'open', 'close', 'high', 'low', 'volume'])

# Get last 20 rows for debugging
print("Last few rows of data:")
print(df[['open', 'high', 'low', 'close']].tail(20))

# Test with our implementation
import my_project

open_data = df['open'].values
high_data = df['high'].values
low_data = df['low'].values
close_data = df['close'].values

bulls, bears = my_project.aso(
    open_data,
    high_data,
    low_data,
    close_data,
    period=10,
    mode=0
)

print("\nOur implementation - Last 5 values:")
print("Bulls:", bulls[-5:])
print("Bears:", bears[-5:])

print("\nExpected from PineScript:")
print("Bulls:", [48.48594883, 46.37206396, 47.20522805, 46.83750720, 43.28268188])
print("Bears:", [51.51405117, 53.62793604, 52.79477195, 53.16249280, 56.71731812])

# Debug: calculate manually for last point
period = 10
last_idx = len(close_data) - 1

print("\nManual calculation for last point:")
print(f"Index: {last_idx}")

# Get the data for the last period
start_idx = last_idx - period + 1
print(f"Period data from index {start_idx} to {last_idx}")

for i in range(start_idx, last_idx + 1):
    o, h, l, c = open_data[i], high_data[i], low_data[i], close_data[i]
    print(f"  [{i}] O:{o:.2f} H:{h:.2f} L:{l:.2f} C:{c:.2f}")

# Calculate for last bar
h = high_data[last_idx]
l = low_data[last_idx]
o = open_data[last_idx]
c = close_data[last_idx]

intrarange = h - l
k1 = 1.0 if intrarange == 0.0 else intrarange
intrabarbulls = (((c - l) + (h - o)) * 50.0) / k1
intrabarbears = (((h - c) + (o - l)) * 50.0) / k1

print(f"\nIntrabar calculation:")
print(f"  intrarange: {intrarange:.2f}")
print(f"  intrabarbulls: {intrabarbulls:.2f}")
print(f"  intrabarbears: {intrabarbears:.2f}")

# Group calculations
grouplow = min(low_data[start_idx:last_idx+1])
grouphigh = max(high_data[start_idx:last_idx+1])
groupopen = open_data[start_idx]

grouprange = grouphigh - grouplow
k2 = 1.0 if grouprange == 0.0 else grouprange
groupbulls = (((c - grouplow) + (grouphigh - groupopen)) * 50.0) / k2
groupbears = (((grouphigh - c) + (groupopen - grouplow)) * 50.0) / k2

print(f"\nGroup calculation:")
print(f"  grouplow: {grouplow:.2f}, grouphigh: {grouphigh:.2f}")
print(f"  groupopen: {groupopen:.2f}")
print(f"  grouprange: {grouprange:.2f}")
print(f"  groupbulls: {groupbulls:.2f}")
print(f"  groupbears: {groupbears:.2f}")

# Average mode
avg_bulls = (intrabarbulls + groupbulls) * 0.5
avg_bears = (intrabarbears + groupbears) * 0.5

print(f"\nAverage mode (before SMA):")
print(f"  bulls: {avg_bulls:.2f}")
print(f"  bears: {avg_bears:.2f}")