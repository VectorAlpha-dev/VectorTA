#!/usr/bin/env python

# Test to see what index gives us the expected values

import my_project
import numpy as np

# Read CSV
with open('src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv') as f:
    lines = f.readlines()[1:]
    close_all = [float(line.split(',')[4]) for line in lines]

print(f"Total data points: {len(close_all)}")

# Expected values from the test
expected_predict = [59161.97066327, 59240.51785714, 59260.29846939, 59225.19005102, 59192.78443878]
expected_trigger = [59020.56403061, 59141.96938776, 59214.56709184, 59232.46619898, 59220.78227041]

# Try different data lengths
for end_idx in range(len(close_all)-100, len(close_all)+1):
    close = close_all[:end_idx]
    close_arr = np.array(close)
    
    try:
        predict, trigger = my_project.ehlers_pma(close_arr)
        
        # Check last 5 values
        if len(predict) >= 5:
            last5_predict = predict[-5:]
            
            # Check if first value is close to expected
            if abs(last5_predict[0] - expected_predict[0]) < 1.0:
                print(f"\nFound potential match at data length {end_idx}:")
                print(f"Last 5 predict: {last5_predict}")
                print(f"Expected:       {expected_predict}")
                print(f"Last 5 trigger: {trigger[-5:]}")
                print(f"Expected:       {expected_trigger}")
                print(f"Last 5 close prices: {close[-5:]}")
                break
    except:
        pass
else:
    print("\nNo exact match found")
    
    # Show what we get with full data
    close_arr = np.array(close_all)
    predict, trigger = my_project.ehlers_pma(close_arr)
    print(f"\nWith full data ({len(close_all)} points):")
    print(f"Last 5 predict: {predict[-5:]}")
    print(f"Expected:       {expected_predict}")
    print(f"Difference:     {[predict[-5+i] - expected_predict[i] for i in range(5)]}")