#!/usr/bin/env python3
"""
Script to identify and fix batch Python bindings that use copy_from_slice
"""

import os
import re

indicators = ['edcf', 'ehlers_itrend', 'ema', 'epma', 'frama', 'fwma']

def check_indicator(name):
    """Check if an indicator needs fixing"""
    file_path = f"src/indicators/moving_averages/{name}.rs"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check for copy_from_slice in batch_py function
    if 'slice_out.copy_from_slice(&batch_result.values)' in content:
        print(f"✗ {name}: Uses copy_from_slice - NEEDS FIX")
        return True
    elif f'{name}_batch_inner_into' in content:
        print(f"✓ {name}: Already uses _into variant")
        return False
    else:
        print(f"? {name}: Unknown status")
        return None

print("Checking indicators for zero-copy batch implementations:\n")

for ind in indicators:
    check_indicator(ind)