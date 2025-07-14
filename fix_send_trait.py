#!/usr/bin/env python3
"""Fix Box<dyn std::error::Error> to add Send bound for thread safety"""

import os
import re
from pathlib import Path

def fix_send_trait_in_file(filepath):
    """Fix Box<dyn std::error::Error> to Box<dyn std::error::Error + Send> in a file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match Box<dyn std::error::Error> but not if already has + Send
    pattern = r'Box<dyn std::error::Error>(?!\s*\+\s*Send)'
    replacement = r'Box<dyn std::error::Error + Send>'
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False

def main():
    # List of files to fix
    indicator_files = [
        'alligator.rs', 'adxr.rs', 'adosc.rs', 'ad.rs', 'vpci.rs', 
        'squeeze_momentum.rs', 'eri.rs', 'bandpass.rs', 'wavetrend.rs', 
        'vosc.rs', 'vidya.rs', 'ultosc.rs', 'ui.rs', 'tsi.rs', 
        'supertrend.rs', 'rvi.rs', 'rsx.rs', 'rocp.rs', 'pvi.rs', 
        'natr.rs', 'mom.rs', 'mfi.rs', 'medium_ad.rs', 'mass.rs', 
        'mab.rs', 'fosc.rs', 'donchian.rs', 'dm.rs', 'devstop.rs', 
        'deviation.rs', 'dec_osc.rs', 'damiani_volatmeter.rs', 'cvi.rs', 
        'correl_hl.rs', 'chande.rs', 'bollinger_bands_width.rs', 
        'bollinger_bands.rs', 'aroonosc.rs', 'aroon.rs', 'apo.rs', 
        'gatorosc.rs', 'pma.rs', 'pivot.rs', 'obv.rs', 'nvi.rs', 
        'medprice.rs', 'heikin_ashi_candles.rs', 'bop.rs', 'avgprice.rs'
    ]
    
    moving_avg_files = [
        'alma.rs', 'cwma.rs', 'epma.rs', 'vpwma.rs', 'vwap.rs', 
        'trima.rs', 'swma.rs', 'srwma.rs', 'sqwma.rs', 'gaussian.rs', 
        'smma.rs', 'sma.rs', 'mwdx.rs', 'hwma.rs', 'hma.rs', 
        'ehlers_itrend.rs', 'ema.rs', 'edcf.rs', 'dema.rs'
    ]
    
    indicators_dir = Path('src/indicators')
    moving_avg_dir = indicators_dir / 'moving_averages'
    
    fixed_count = 0
    
    # Fix regular indicators
    for filename in indicator_files:
        filepath = indicators_dir / filename
        if filepath.exists():
            if fix_send_trait_in_file(filepath):
                print(f"Fixed: {filepath}")
                fixed_count += 1
    
    # Fix moving average indicators
    for filename in moving_avg_files:
        filepath = moving_avg_dir / filename
        if filepath.exists():
            if fix_send_trait_in_file(filepath):
                print(f"Fixed: {filepath}")
                fixed_count += 1
    
    print(f"\nTotal files fixed: {fixed_count}")

if __name__ == "__main__":
    main()