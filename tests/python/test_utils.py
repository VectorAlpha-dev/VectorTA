"""Common utilities for Python binding tests"""
import numpy as np
import os
import csv
from pathlib import Path

def load_test_data():
    """Load the same CSV data used in Rust tests"""
    data_path = Path(__file__).parent.parent.parent / 'src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv'
    
    candles = {
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': []
    }
    
    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # Skip empty rows
            if len(row) < 6:
                continue
            # CSV format matches Rust: timestamp[0], open[1], close[2], high[3], low[4], volume[5]
            candles['open'].append(float(row[1]))
            candles['close'].append(float(row[2]))
            candles['high'].append(float(row[3]))
            candles['low'].append(float(row[4]))
            candles['volume'].append(float(row[5]))
    
    return {k: np.array(v) for k, v in candles.items()}

def assert_close(actual, expected, rtol=1e-8, atol=1e-10, msg=""):
    """Assert arrays are close with better error messages"""
    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as e:
        if msg:
            raise AssertionError(f"{msg}: {str(e)}")
        raise

# Expected outputs from Rust tests - these must match EXACTLY
EXPECTED_OUTPUTS = {
    'alma': {
        'default_params': {'period': 9, 'offset': 0.85, 'sigma': 6.0},
        'last_5_values': [
            59286.72216704,
            59273.53428138,
            59204.37290721,
            59155.93381742,
            59026.92526112
        ],
        # Re-input test expected values
        'reinput_last_5': [
            59140.73195170,
            59211.58090986,
            59238.16030697,
            59222.63528822,
            59165.14427332
        ]
    },
    'cwma': {
        'default_params': {'period': 14},
        'last_5_values': [
            59224.641237300435,
            59213.64831277214,
            59171.21190130624,
            59167.01279027576,
            59039.413552249636
        ]
    },
    'dema': {
        'default_params': {'period': 30},
        'last_5_values': [
            59189.73193987478,
            59129.24920772847,
            59058.80282420511,
            59011.5555611042,
            58908.370159946775
        ]
    },
    'edcf': {
        'default_params': {'period': 15},
        'last_5_values': [
            59593.332275678375,
            59731.70263288801,
            59766.41512339413,
            59655.66162110993,
            59332.492883847
        ]
    }
}