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
    
    # Convert to numpy arrays
    result = {k: np.array(v) for k, v in candles.items()}
    
    # Add calculated fields
    result['hl2'] = (result['high'] + result['low']) / 2.0
    
    return result

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
    'cg': {
        'default_params': {'period': 10},
        'last_5_values': [
            -4.99905186931943,
            -4.998559827254377,
            -4.9970065675119555,
            -4.9928483984587295,
            -5.004210799262688
        ]
    },
    'trima': {
        'default_params': {'period': 30},
        'last_5_values': [
            59957.916666666664,
            59846.770833333336,
            59750.620833333334,
            59665.2125,
            59581.612499999996
        ],
        # Re-input test expected values (period=10 on first pass result)
        'reinput_last_5': [
            60750.01069444444,
            60552.44180555555,
            60372.22486111111,
            60210.39555555556,
            60066.62458333334
        ]
    },
    'chande': {
        'default_params': {'period': 22, 'mult': 3.0, 'direction': 'long'},
        'last_5_values': [
            59444.14115983658,
            58576.49837984401,
            58649.1120898511,
            58724.56154031242,
            58713.39965211639
        ],
        'warmup_period': 21  # period - 1
    },
    'mfi': {
        'default_params': {'period': 14},
        'last_5_values': [
            38.13874339324763,
            37.44139770113819,
            31.02039511395131,
            28.092605898618896,
            25.905204729397813
        ]
    },
    'jsa': {
        'default_params': {'period': 30},
        'last_5_values': [61640.0, 61418.0, 61240.0, 61060.5, 60889.5],
        'warmup_period': 30  # first_valid + period where first_valid = 0 for this data
    },
    'donchian': {
        'default_params': {'period': 20},
        'last_5_upper': [61290.0, 61290.0, 61290.0, 61290.0, 61290.0],
        'last_5_middle': [59583.0, 59583.0, 59583.0, 59583.0, 59583.0],
        'last_5_lower': [57876.0, 57876.0, 57876.0, 57876.0, 57876.0],
        # Re-input test: Apply Donchian to the middle band output
        'reinput_last_5_upper': [61700.0, 61700.0, 61700.0, 61642.5, 61642.5],
        'reinput_last_5_middle': [60641.5, 60641.5, 60641.5, 60612.75, 60612.75],
        'reinput_last_5_lower': [59583.0, 59583.0, 59583.0, 59583.0, 59583.0]
    },
    'msw': {
        'default_params': {'period': 5},
        'last_5_sine': [
            -0.49733966449848194,
            -0.8909425976991894,
            -0.709353328514554,
            -0.40483478076837887,
            -0.8817006719953886,
        ],
        'last_5_lead': [
            -0.9651269132969991,
            -0.30888310410390457,
            -0.003182174183612666,
            0.36030983330963545,
            -0.28983704937461496,
        ],
        'warmup_period': 4  # period - 1
    },
    'correl_hl': {
        'default_params': {'period': 5},
        'last_5_values': [
            0.04589155420456278,
            0.6491664099299647,
            0.9691259236943873,
            0.9915438003818791,
            0.8460608423095615
        ]
    },
    'cfo': {
        'default_params': {'period': 14, 'scalar': 100.0},
        'last_5_values': [
            0.5998626489475746,
            0.47578011282578453,
            0.20349744599816233,
            0.0919617952835795,
            -0.5676291145560617
        ]
    },
    'cmo': {
        'default_params': {'period': 14},
        'last_5_values': [
            -13.152504931406101,
            -14.649876201213106,
            -16.760170709240303,
            -14.274505732779227,
            -21.984038127126716
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
    'linearreg_intercept': {
        'default_params': {'period': 14},
        'last_5_values': [
            60000.91428571429,
            59947.142857142855,
            59754.57142857143,
            59318.4,
            59321.91428571429
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
        ],
        # Re-input test expected values (period=10 on first pass result)
        # Note: The Rust test only verifies length, not specific values
        'reinput_last_5': None,  # Not verified in Rust tests
        'warmup_period': 13  # first + period - 1 (with no leading NaNs, first=0)
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
    'natr': {
        'default_params': {'period': 14},
        'last_5_values': [
            1.5465877404905772,
            1.4773840355794576,
            1.4201627494720954,
            1.3556212509014807,
            1.3836271128536142
        ]
    },
    'cci': {
        'default_params': {'period': 14},
        'last_5_values': [
            -51.55252564125841,
            -43.50326506381541,
            -64.05117302269149,
            -39.05150631680948,
            -152.50523930896998
        ],
        # Re-input test expected values (when CCI is applied to CCI output)
        'reinput_last_5': [
            -150.0,  # Will be calculated when tests run
            -150.0,
            -150.0,
            -150.0,
            -150.0
        ]
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
    },
    'midprice': {
        'default_params': {'period': 14},
        'last_5_values': [
            59583.0,
            59583.0,
            59583.0,
            59486.0,
            58989.0
        ]
    },
    'rsx': {
        'default_params': {'period': 14},
        'last_5_values': [
            46.11486311289701,
            46.88048640321688,
            47.174443049619995,
            47.48751360654475,
            46.552886446171684
        ]
    },
    'coppock': {
        'default_params': {'short': 11, 'long': 14, 'ma': 10, 'ma_type': 'wma'},
        'last_5_values': [
            -1.4542764618985533,
            -1.3795224034983653,
            -1.614331648987457,
            -1.9179048338714915,
            -2.1096548435774625
        ]
    },
    'decycler': {
        'default_params': {'hp_period': 125, 'k': 0.707},
        'last_5_values': [
            60289.96384058519,
            60204.010366691065,
            60114.255563805666,
            60028.535266555904,
            59934.26876964316
        ]
    },
    'nvi': {
        'default_params': {},  # NVI has no parameters
        'last_5_values': [
            154243.6925373456,
            153973.11239019397,
            153973.11239019397,
            154275.63921207888,
            154275.63921207888
        ]
    },
    'ppo': {
        'default_params': {'fast_period': 12, 'slow_period': 26, 'ma_type': 'sma'},
        'last_5_values': [
            -0.8532313608928664,
            -0.8537562894550523,
            -0.6821291938174874,
            -0.5620008722078592,
            -0.4101724140910927
        ]
    },
    'vpci': {
        'default_params': {'short_range': 5, 'long_range': 25},
        'last_5_vpci': [
            -319.65148214323426,
            -133.61700649928346,
            -144.76194155503174,
            -83.55576212490328,
            -169.53504207700533
        ],
        'last_5_vpcis': [
            -1049.2826640115732,
            -694.1067814399748,
            -519.6960416662324,
            -330.9401404636258,
            -173.004986803695
        ]
    },
    'cksp': {
        'default_params': {'p': 10, 'x': 1.0, 'q': 9},
        'long_last_5_values': [
            60306.66197802568,
            60306.66197802568,
            60306.66197802568,
            60203.29578022311,
            60201.57958198072
        ],
        'short_last_5_values': [
            58757.826484736055,
            58701.74383626245,
            58656.36945263621,
            58611.03250737258,
            58611.03250737258
        ]
    },
    'damiani_volatmeter': {
        'default_params': {
            'vis_atr': 13,
            'vis_std': 20,
            'sed_atr': 40,
            'sed_std': 100,
            'threshold': 1.4
        },
        'vol_last_5_values': [
            0.9009485470514558,
            0.8333604467044887,
            0.815318380178986,
            0.8276892636184923,
            0.879447954127426
        ],
        'anti_last_5_values': [
            1.1227721577887388,
            1.1250333024152703,
            1.1325501989919875,
            1.1403866079746106,
            1.1392919184055932
        ]
    },
    'di': {
        'default_params': {'period': 14},
        'plus_last_5_values': [
            10.99067007335658,
            11.306993269828585,
            10.948661818939213,
            10.683207768215592,
            9.802180952619183
        ],
        'minus_last_5_values': [
            28.06728094177839,
            27.331240567633152,
            27.759989125359493,
            26.951434842917386,
            30.748897303623057
        ]
    },
    'efi': {
        'default_params': {'period': 13},
        'last_5_values': [
            -44604.382026531224,
            -39811.02321812391,
            -36599.9671820205,
            -29903.28014503471,
            -55406.09054645832
        ]
    },
    'fosc': {
        'default_params': {'period': 5},
        'last_5_values': [
            -0.8904444627923475,
            -0.4763353099245297,
            -0.2379782851444668,
            0.292790128025632,
            -0.6597902988090389
        ]
    },
    'kst': {
        'default_params': {'roclen1': 10, 'roclen2': 15, 'roclen3': 20, 'roclen4': 30, 
                          'smalen1': 10, 'smalen2': 10, 'smalen3': 10, 'smalen4': 15, 'siglen': 9},
        'last_5_line': [
            -47.38570195278667,
            -44.42926180347176,
            -42.185693049429034,
            -40.10697793942024,
            -40.17466795905724
        ],
        'last_5_signal': [
            -52.66743277411538,
            -51.559775662725556,
            -50.113844191238954,
            -48.58923772989874,
            -47.01112630514571
        ]
    },
    'lrsi': {
        'default_params': {'alpha': 0.2},
        'last_5_values': [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ]
    },
    'mean_ad': {
        'default_params': {'period': 5},
        'last_5_values': [
            199.71999999999971,
            104.14000000000087,
            133.4,
            100.54000000000087,
            117.98000000000029
        ]
    },
    'pivot': {
        'default_params': {'mode': 3},  # Camarilla mode
        'last_5_r4': [59466.5, 59357.55, 59243.6, 59334.85, 59170.35],
        # Other levels can be calculated but r4 is used for verification
    },
    'correlation_cycle': {
        'default_params': {'period': 20, 'threshold': 9.0},
        'last_5_values': {
            'real': [
                -0.3348928030992766,
                -0.2908979303392832,
                -0.10648582811938148,
                -0.09118320471750277,
                0.0826798259258665
            ],
            'imag': [
                0.2902308064575494,
                0.4025192756952553,
                0.4704322460080054,
                0.5404405595224989,
                0.5418162415918566
            ],
            'angle': [
                -139.0865569687123,
                -125.8553823569915,
                -102.75438860700636,
                -99.576759208278,
                -81.32373697835556
            ]
        }
    },
    'keltner': {
        'default_params': {'period': 20, 'multiplier': 2.0, 'ma_type': 'ema'},
        'last_5_upper': [
            61619.504155205745,
            61503.56119134791,
            61387.47897150178,
            61286.61078267451,
            61206.25688331261
        ],
        'last_5_middle': [
            59758.339871629956,
            59703.35512195091,
            59640.083205574636,
            59593.884805043715,
            59504.46720456336
        ],
        'last_5_lower': [
            57897.17558805417,
            57903.14905255391,
            57892.68743964749,
            57901.158827412924,
            57802.67752581411
        ]
    },
    'sma': {
        'default_params': {'period': 9},
        'last_5_values': [59180.8, 59175.0, 59129.4, 59085.4, 59133.7],
        'reinput_last_5': None  # To be calculated if needed
    },
    'mwdx': {
        'default_params': {'factor': 0.2},
        'last_5_values': [
            59302.181566190935,
            59277.94525295275,
            59230.1562023622,
            59215.124961889764,
            59103.099969511815
        ]
    }
}

# Convenience constants for individual indicators
EXPECTED_SUPERSMOOTHER_3_POLE = EXPECTED_OUTPUTS['supersmoother_3_pole']['last_5_values']
EXPECTED_SUPERSMOOTHER = EXPECTED_OUTPUTS['supersmoother']['last_5_values']