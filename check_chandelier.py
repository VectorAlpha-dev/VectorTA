import pandas as pd
import numpy as np

# Load the CSV data
df = pd.read_csv('src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv', 
                 names=['timestamp', 'open', 'close', 'high', 'low', 'volume'])

# Find where prices are around 68k
mask = (df['close'] > 65000) & (df['close'] < 70000)
high_price_periods = df[mask]

print(f"Total rows: {len(df)}")
print(f"Rows with close price between 65k-70k: {len(high_price_periods)}")

if not high_price_periods.empty:
    print(f"\nFirst occurrence index: {high_price_periods.index[0]}")
    print(f"Last occurrence index: {high_price_periods.index[-1]}")
    
    # Look at the last period around 68k
    last_period_start = high_price_periods.index[-100] if len(high_price_periods) > 100 else high_price_periods.index[0]
    last_period_end = high_price_periods.index[-1]
    
    print(f"\nLast period in 65-70k range: rows {last_period_start} to {last_period_end}")
    print(f"Last 5 close prices in that range:")
    last_5 = df.loc[last_period_end-4:last_period_end, ['close', 'high', 'low']]
    print(last_5)

# Also check the very end of the data
print("\nLast 5 rows of entire dataset:")
print(df.tail()[['close', 'high', 'low']])