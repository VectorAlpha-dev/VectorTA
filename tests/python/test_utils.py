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
        first_row = True
        for row in reader:
            # Skip header row and empty rows
            if first_row:
                first_row = False
                continue
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
    },
    'ema': {
        'default_params': {'period': 9},
        'last_five': [
            59302.2,
            59277.9,
            59230.2,
            59215.1,
            59103.1
        ]
    },
    'sqwma': {
        'default_params': {'period': 14},
        'last_5_values': [
            59229.72287968442,
            59211.30867850099,
            59172.516765286,
            59167.73471400394,
            59067.97928994083
        ]
    },
    'srwma': {
        'default_params': {'period': 14},
        'last_5_values': [
            59344.28384704595,
            59282.09151629659,
            59192.76580529367,
            59178.04767548977,
            59110.03801260874
        ]
    },
    'supersmoother_3_pole': {
        'default_params': {'period': 14},
        'last_5_values': [
            59072.13481064446,
            59089.08032603,
            59111.35711851466,
            59133.14402399381,
            59121.91820047289
        ]
    },
    'supersmoother': {
        'default_params': {'period': 14},
        'last_5_values': [
            59140.98229179739,
            59172.03593376982,
            59179.40342783722,
            59171.22758152845,
            59127.859841077094
        ]
    },
    'wilders': {
        'default_params': {'period': 5},
        'last_5_values': [
            59302.18156619092,
            59277.94525295273,
            59230.15620236219,
            59215.12496188975,
            59103.0999695118
        ]
    },
    'ad': {
        'default_params': {},
        'last_5_values': [
            1645918.16,
            1645876.11,
            1645824.27,
            1645828.87,
            1645728.78
        ]
    },
    'vwma': {
        'default_params': {'period': 20},
        'last_5_values': [
            59201.87047121331,
            59217.157390630266,
            59195.74526905522,
            59196.261392450084,
            59151.22059588594
        ]
    },
    'acosc': {
        'default_params': {},  # ACOSC has no parameters
        'last_5_osc': [
            273.30,
            383.72,
            357.7,
            291.25,
            176.84
        ],
        'last_5_change': [
            49.6,
            110.4,
            -26.0,
            -66.5,
            -114.4
        ]
    }
}

# Convenience constants for individual indicators
EXPECTED_SUPERSMOOTHER_3_POLE = EXPECTED_OUTPUTS['supersmoother_3_pole']['last_5_values']
EXPECTED_SUPERSMOOTHER = EXPECTED_OUTPUTS['supersmoother']['last_5_values']