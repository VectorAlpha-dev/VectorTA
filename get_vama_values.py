import sys
import os
os.chdir("C:/Rust Projects/my_project-bindings-1")

# Parse test output to extract all values
test_output = """
got 60249.34558277224, expected 58881.58124494
got 60283.78930990677, expected 58866.67951208
got 60173.39052862816, expected 58873.34641238
got 60260.19903965848, expected 58870.41762890
got 60226.10253226444, expected 58696.37821343
"""

# Extract the "got" values
for line in test_output.strip().split('\n'):
    if 'got' in line:
        got_val = line.split('got ')[1].split(',')[0]
        print(f"            {got_val},")
