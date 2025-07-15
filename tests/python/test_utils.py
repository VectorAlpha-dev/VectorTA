"""Common utilities for Python binding tests"""
import numpy as np
import os
import csv
from pathlib import Path

def load_test_data():
    """Load the same CSV data used in Rust tests"""
    data_path = Path(__file__).parent.parent.parent / 'src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv'
    
    candles = {
        'timestamp': [],
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
            candles['timestamp'].append(int(row[0]))
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

# Alias for backward compatibility
assert_array_close = assert_close

def assert_all_nan(arr, msg=""):
    """Assert all values are NaN"""
    if not np.all(np.isnan(arr)):
        raise AssertionError(f"{msg}: Not all values are NaN")

def assert_no_nan(arr, msg=""):
    """Assert no NaN values in array"""
    if np.any(np.isnan(arr)):
        raise AssertionError(f"{msg}: Found NaN values in array")

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
    'adx': {
        'default_params': {'period': 14},
        'last_5_values': [
            36.14,
            36.52,
            37.01,
            37.46,
            38.47
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
    },
    'vwap': {
        'default_params': {'anchor': '1d'},
        'last_5_values': [
            59353.05963230107,
            59330.15815713043,
            59289.94649532547,
            59274.6155462414,
            58730.0
        ],
        'anchor_1D': [
            59353.05963230107,
            59330.15815713043,
            59289.94649532547,
            59274.6155462414,
            58730.0
        ]
    },
    'zlema': {
        'default_params': {'period': 14},
        'last_5_values': [
            59015.1,
            59165.2,
            59168.1,
            59147.0,
            58978.9
        ]
    },
    'vpwma': {
        'default_params': {'period': 14, 'power': 0.382},
        'last_5_values': [
            59363.927599446455,
            59296.83894519251,
            59196.82476139941,
            59180.8040249446,
            59113.84473799056
        ]
    },
    'wma': {
        'default_params': {'period': 30},
        'last_5_values': [
            59638.52903225806,
            59563.7376344086,
            59489.4064516129,
            59432.02580645162,
            59350.58279569892
        ]
    },
    'adxr': {
        'default_params': {'period': 14},
        'last_5_values': [
            37.10,
            37.3,
            37.0,
            36.2,
            36.3
        ]
    },
    'aroon': {
        'default_params': {'length': 14},
        'last_5_up': [
            21.43,
            14.29,
            7.14,
            0.0,
            0.0
        ],
        'last_5_down': [
            71.43,
            64.29,
            57.14,
            50.0,
            42.86
        ]
    },
    'aroonosc': {
        'default_params': {'length': 14},
        'last_5_values': [-50.0, -50.0, -50.0, -50.0, -42.8571]
    },
    'adosc': {
        'default_params': {'short_period': 3, 'long_period': 10},
        'last_5_values': [-166.2175, -148.9983, -144.9052, -128.5921, -142.0772]
    },
    'bollinger_bands_width': {
        'default_params': {'period': 20, 'devup': 2.0, 'devdn': 2.0, 'matype': 'sma', 'devtype': 0},
        'last_5_values': [
            0.03715911020016619,
            0.036072736452195386,
            0.034961761824566714,
            0.03493493700573171,
            0.03624293421802348
        ]
    },
    'apo': {
        'default_params': {'short_period': 10, 'long_period': 20},
        'last_5_values': [-429.8, -401.6, -386.1, -357.9, -374.1]
    },
    'bandpass': {
        'default_params': {'period': 20, 'bandwidth': 0.3},
        'last_5_values': {
            'bp': [
                -236.23678021132827,
                -247.4846395608195,
                -242.3788746078502,
                -212.89589193350128,
                -179.97293838509464
            ],
            'bp_normalized': [
                -0.4399672555578846,
                -0.4651011734720517,
                -0.4596426251402882,
                -0.40739824942488945,
                -0.3475245023284841
            ],
            'signal': [-1.0, 1.0, 1.0, 1.0, 1.0],
            'trigger': [
                -0.4746908356434579,
                -0.4353877348116954,
                -0.3727126131420441,
                -0.2746336628365846,
                -0.18240018384226137
            ]
        }
    },
    'ao': {
        'default_params': {'short_period': 5, 'long_period': 34},
        'last_5_values': [-1671.3, -1401.6706, -1262.3559, -1178.4941, -1157.4118]
    },
    'atr': {
        'default_params': {'length': 14},
        'last_5_values': [916.89, 874.33, 838.45, 801.92, 811.57]
    },
    'bop': {
        'default_params': {},  # BOP has no parameters
        'last_5_values': [
            0.045454545454545456,
            -0.32398753894080995,
            -0.3844086021505376,
            0.3547400611620795,
            -0.5336179295624333
        ]
    }
}

# Convenience constants for individual indicators
EXPECTED_SUPERSMOOTHER_3_POLE = EXPECTED_OUTPUTS['supersmoother_3_pole']['last_5_values']
EXPECTED_SUPERSMOOTHER = EXPECTED_OUTPUTS['supersmoother']['last_5_values']