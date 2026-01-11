"""
Python binding tests for OTTO indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

try:
    import my_project as ta_indicators
except ImportError:
    
    try:
        import my_project as ta_indicators
    except ImportError:
        pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import (
    load_test_data, 
    assert_close, 
    assert_all_nan,
    assert_no_nan,
    EXPECTED_OUTPUTS
)


class TestOtto:
    @pytest.fixture(scope='class')
    def test_data(self):
        """Load test data from CSV"""
        return load_test_data()
    
    @pytest.fixture(scope='class')
    def otto_test_data(self, test_data):
        """Use close prices from CSV data for OTTO tests"""
        return test_data['close']
    
    def test_otto_partial_params(self, otto_test_data):
        """Test OTTO with partial parameters (None values) - mirrors check_otto_partial_params"""
        data = otto_test_data
        
        
        hott, lott = ta_indicators.otto(
            data,
            ott_period=2,  
            ott_percent=0.8,
            fast_vidya_length=10,  
            slow_vidya_length=20,
            correcting_constant=100000,  
            ma_type="VAR"  
        )
        
        assert len(hott) == len(data)
        assert len(lott) == len(data)
    
    def test_otto_accuracy(self, otto_test_data):
        """Test OTTO matches Rust CSV reference values (last 5)"""
        data = otto_test_data
        expected = EXPECTED_OUTPUTS['otto']

        hott, lott = ta_indicators.otto(
            data,
            ott_period=expected['default_params']['ott_period'],
            ott_percent=expected['default_params']['ott_percent'],
            fast_vidya_length=expected['default_params']['fast_vidya_length'],
            slow_vidya_length=expected['default_params']['slow_vidya_length'],
            correcting_constant=expected['default_params']['correcting_constant'],
            ma_type=expected['default_params']['ma_type']
        )

        assert len(hott) == len(data)
        assert len(lott) == len(data)

        
        hott_last5 = hott[-5:]
        lott_last5 = lott[-5:]
        
        assert_close(hott_last5, np.array(expected['last_5_hott']), rtol=0.0, atol=1e-8,
                     msg="OTTO HOTT last 5 values mismatch")
        assert_close(lott_last5, np.array(expected['last_5_lott']), rtol=0.0, atol=1e-8,
                     msg="OTTO LOTT last 5 values mismatch")
    
    def test_otto_default_candles(self, test_data):
        """Test OTTO with default parameters - mirrors check_otto_default_candles"""
        close = test_data['close']
        
        
        hott, lott = ta_indicators.otto(
            close,
            ott_period=2,
            ott_percent=0.6,
            fast_vidya_length=10,
            slow_vidya_length=25,
            correcting_constant=100000,
            ma_type="VAR"
        )
        
        assert len(hott) == len(close)
        assert len(lott) == len(close)
        
        
        non_nan_hott = sum(1 for x in hott if not np.isnan(x))
        non_nan_lott = sum(1 for x in lott if not np.isnan(x))
        assert non_nan_hott > 0
        assert non_nan_lott > 0
    
    def test_otto_zero_period(self):
        """Test OTTO fails with zero period - mirrors check_otto_zero_period"""
        input_data = [10.0, 20.0, 30.0]
        
        with pytest.raises(ValueError):
            ta_indicators.otto(
                np.array(input_data),
                ott_period=0,
                ott_percent=0.6,
                fast_vidya_length=10,
                slow_vidya_length=25,
                correcting_constant=100000,
                ma_type="VAR"
            )
    
    def test_otto_period_exceeds_length(self):
        """Test OTTO fails when period exceeds data length - mirrors check_otto_period_exceeds_length"""
        data_small = [10.0, 20.0, 30.0]
        
        with pytest.raises(ValueError):
            ta_indicators.otto(
                np.array(data_small),
                ott_period=10,
                ott_percent=0.6,
                fast_vidya_length=10,
                slow_vidya_length=25,
                correcting_constant=100000,
                ma_type="VAR"
            )
    
    def test_otto_very_small_dataset(self):
        """Test OTTO with minimal valid dataset - mirrors check_otto_very_small_dataset"""
        
        data = [1.0] * 15
        
        
        hott, lott = ta_indicators.otto(
            np.array(data),
            ott_period=1,
            ott_percent=0.5,
            fast_vidya_length=1,
            slow_vidya_length=2,
            correcting_constant=1.0,
            ma_type="SMA"
        )
        
        assert len(hott) == len(data)
        assert len(lott) == len(data)
    
    def test_otto_empty_input(self):
        """Test OTTO fails with empty input - mirrors check_otto_empty_input"""
        empty = []
        
        with pytest.raises(ValueError):
            ta_indicators.otto(
                np.array(empty),
                ott_period=2,
                ott_percent=0.6,
                fast_vidya_length=10,
                slow_vidya_length=25,
                correcting_constant=100000,
                ma_type="VAR"
            )
    
    def test_otto_invalid_ma_type(self, otto_test_data):
        """Test OTTO fails with invalid MA type - mirrors check_otto_invalid_ma_type"""
        data = otto_test_data[:300]  
        
        with pytest.raises(ValueError):
            ta_indicators.otto(
                data,
                ott_period=2,
                ott_percent=0.6,
                fast_vidya_length=10,
                slow_vidya_length=25,
                correcting_constant=100000,
                ma_type="INVALID_MA"
            )
    
    def test_otto_all_ma_types(self, otto_test_data):
        """Test OTTO with all supported MA types - mirrors check_otto_all_ma_types"""
        data = otto_test_data[:300]  
        ma_types = ["SMA", "EMA", "WMA", "DEMA", "TMA", "VAR", "ZLEMA", "TSF", "HULL"]
        
        for ma_type in ma_types:
            hott, lott = ta_indicators.otto(
                data,
                ott_period=2,
                ott_percent=0.6,
                fast_vidya_length=10,
                slow_vidya_length=25,
                correcting_constant=100000,
                ma_type=ma_type
            )
            assert len(hott) == len(data), f"Failed for MA type: {ma_type}"
            assert len(lott) == len(data), f"Failed for MA type: {ma_type}"
    
    def test_otto_reinput(self, otto_test_data):
        """Test OTTO applied twice (re-input) - mirrors check_otto_reinput"""
        data = otto_test_data
        params = EXPECTED_OUTPUTS['otto']['default_params']
        
        
        first_hott, first_lott = ta_indicators.otto(
            data,
            ott_period=params['ott_period'],
            ott_percent=params['ott_percent'],
            fast_vidya_length=params['fast_vidya_length'],
            slow_vidya_length=params['slow_vidya_length'],
            correcting_constant=params['correcting_constant'],
            ma_type=params['ma_type']
        )
        
        assert len(first_hott) == len(data)
        assert len(first_lott) == len(data)
        
        
        second_hott, second_lott = ta_indicators.otto(
            first_hott,
            ott_period=params['ott_period'],
            ott_percent=params['ott_percent'],
            fast_vidya_length=params['fast_vidya_length'],
            slow_vidya_length=params['slow_vidya_length'],
            correcting_constant=params['correcting_constant'],
            ma_type=params['ma_type']
        )
        
        assert len(second_hott) == len(first_hott)
        assert len(second_lott) == len(first_hott)
        
        
        
        third_hott, third_lott = ta_indicators.otto(
            data,
            ott_period=params['ott_period'],
            ott_percent=params['ott_percent'],
            fast_vidya_length=params['fast_vidya_length'],
            slow_vidya_length=params['slow_vidya_length'],
            correcting_constant=params['correcting_constant'],
            ma_type=params['ma_type']
        )
        
        
        for i in range(len(data)):
            if not np.isnan(first_hott[i]) and not np.isnan(third_hott[i]):
                assert_close(first_hott[i], third_hott[i], rtol=1e-10, 
                           msg=f"OTTO HOTT determinism failed at index {i}")
            if not np.isnan(first_lott[i]) and not np.isnan(third_lott[i]):
                assert_close(first_lott[i], third_lott[i], rtol=1e-10, 
                           msg=f"OTTO LOTT determinism failed at index {i}")
    
    def test_otto_nan_handling(self, otto_test_data):
        """Test OTTO handles NaN values correctly - mirrors check_otto_nan_handling"""
        data = otto_test_data.copy()
        
        
        data[100] = np.nan
        data[150] = np.nan
        data[200] = np.nan
        
        hott, lott = ta_indicators.otto(
            data,
            ott_period=2,
            ott_percent=0.6,
            fast_vidya_length=10,
            slow_vidya_length=25,
            correcting_constant=100000,
            ma_type="VAR"
        )
        
        assert len(hott) == len(data)
        assert len(lott) == len(data)
        
        
        valid_hott = sum(1 for i in range(250, len(hott)) if not np.isnan(hott[i]))
        valid_lott = sum(1 for i in range(250, len(lott)) if not np.isnan(lott[i]))
        
        assert valid_hott > 0, "Should produce some valid HOTT values despite NaNs"
        assert valid_lott > 0, "Should produce some valid LOTT values despite NaNs"
    
    def test_otto_streaming(self, otto_test_data):
        """Test OTTO streaming matches batch calculation - mirrors check_otto_streaming"""
        data = otto_test_data
        params = EXPECTED_OUTPUTS['otto']['default_params']
        
        
        batch_hott, batch_lott = ta_indicators.otto(
            data,
            ott_period=params['ott_period'],
            ott_percent=params['ott_percent'],
            fast_vidya_length=params['fast_vidya_length'],
            slow_vidya_length=params['slow_vidya_length'],
            correcting_constant=params['correcting_constant'],
            ma_type=params['ma_type']
        )
        
        
        stream = ta_indicators.OttoStreamPy(
            ott_period=params['ott_period'],
            ott_percent=params['ott_percent'],
            fast_vidya_length=params['fast_vidya_length'],
            slow_vidya_length=params['slow_vidya_length'],
            correcting_constant=params['correcting_constant'],
            ma_type=params['ma_type']
        )
        
        stream_hott = []
        stream_lott = []
        
        for value in data:
            result = stream.update(float(value))
            if result is not None:
                hott_val, lott_val = result
                stream_hott.append(hott_val)
                stream_lott.append(lott_val)
            else:
                stream_hott.append(np.nan)
                stream_lott.append(np.nan)
        
        
        stream_hott = np.array(stream_hott)
        stream_lott = np.array(stream_lott)
        
        
        
        if len(stream_hott) >= 10:
            for i in range(-10, 0):
                if not np.isnan(stream_hott[i]) and not np.isnan(batch_hott[i]):
                    assert_close(
                        stream_hott[i], 
                        batch_hott[i], 
                        rtol=1e-3, 
                        atol=1e-4,
                        msg=f"OTTO streaming HOTT mismatch at index {i}"
                    )
                if not np.isnan(stream_lott[i]) and not np.isnan(batch_lott[i]):
                    assert_close(
                        stream_lott[i], 
                        batch_lott[i], 
                        rtol=1e-3, 
                        atol=1e-4,
                        msg=f"OTTO streaming LOTT mismatch at index {i}"
                    )
    
    def test_otto_batch_single_params(self, otto_test_data):
        """Test OTTO batch processing with single parameter set - mirrors batch tests"""
        data = otto_test_data
        params = EXPECTED_OUTPUTS['otto']['default_params']
        
        result = ta_indicators.otto_batch(
            data,
            ott_period_range=(params['ott_period'], params['ott_period'], 0),
            ott_percent_range=(params['ott_percent'], params['ott_percent'], 0),
            fast_vidya_range=(params['fast_vidya_length'], params['fast_vidya_length'], 0),
            slow_vidya_range=(params['slow_vidya_length'], params['slow_vidya_length'], 0),
            correcting_constant_range=(params['correcting_constant'], params['correcting_constant'], 0),
            ma_types=[params['ma_type']]
        )
        
        assert 'hott' in result
        assert 'lott' in result
        assert 'ott_periods' in result
        assert 'ott_percents' in result
        
        
        assert result['hott'].shape[0] == 1
        assert result['lott'].shape[0] == 1
        assert result['hott'].shape[1] == len(data)
        assert result['lott'].shape[1] == len(data)
        
        
        hott_row = result['hott'][0]
        lott_row = result['lott'][0]
        
        
        
        
        
        valid_hott = hott_row[~np.isnan(hott_row)]
        valid_lott = lott_row[~np.isnan(lott_row)]
        
        assert len(valid_hott) > 0, "Batch HOTT should have valid values"
        assert len(valid_lott) > 0, "Batch LOTT should have valid values"
        
        
        
        assert np.all((valid_hott >= 0.0) & (valid_hott <= 1.0)), \
            "Batch HOTT values should be within normalized range [0, 1]"
        assert np.all((valid_lott >= 0.0) & (valid_lott <= 1.0)), \
            "Batch LOTT values should be within normalized range [0, 1]"
    
    def test_otto_batch_sweep(self, test_data):
        """Test OTTO batch with parameter sweep - mirrors check_batch_sweep"""
        close = test_data['close'][:300]  
        
        result = ta_indicators.otto_batch(
            close,
            ott_period_range=(2, 4, 1),
            ott_percent_range=(0.5, 0.7, 0.1),
            fast_vidya_range=(10, 12, 1),
            slow_vidya_range=(20, 22, 1),
            correcting_constant_range=(100000.0, 100000.0, 0.0),
            ma_types=["VAR", "EMA"]
        )
        
        
        expected_combos = 3 * 3 * 3 * 3 * 1 * 2
        
        
        assert result['hott'].shape[0] == expected_combos
        assert result['lott'].shape[0] == expected_combos
        assert result['hott'].shape[1] == len(close)
        assert result['lott'].shape[1] == len(close)
        
        
        assert 'ott_periods' in result
        assert 'ott_percents' in result
        assert 'fast_vidya' in result
        assert 'slow_vidya' in result
        assert 'ma_types' in result
        
        assert len(result['ott_periods']) == expected_combos
        assert len(result['ott_percents']) == expected_combos
        assert len(result['fast_vidya']) == expected_combos
        assert len(result['slow_vidya']) == expected_combos
        assert len(result['ma_types']) == expected_combos
        
        
        assert all(2 <= p <= 4 for p in result['ott_periods'])
        assert all(0.49 <= p <= 0.71 for p in result['ott_percents'])
        assert all(10 <= f <= 12 for f in result['fast_vidya'])
        assert all(20 <= s <= 22 for s in result['slow_vidya'])
        assert all(m in ["VAR", "EMA"] for m in result['ma_types'])
    
    def test_otto_all_nan_input(self):
        """Test OTTO with all NaN values"""
        all_nan = [float('nan')] * 100
        
        with pytest.raises(ValueError):
            ta_indicators.otto(
                np.array(all_nan),
                ott_period=2,
                ott_percent=0.6,
                fast_vidya_length=10,
                slow_vidya_length=25,
                correcting_constant=100000,
                ma_type="VAR"
            )
    
    def test_otto_warmup_period(self, otto_test_data):
        """Test OTTO warmup period behavior"""
        data = otto_test_data
        params = EXPECTED_OUTPUTS['otto']['default_params']
        
        hott, lott = ta_indicators.otto(
            data,
            ott_period=params['ott_period'],
            ott_percent=params['ott_percent'],
            fast_vidya_length=params['fast_vidya_length'],
            slow_vidya_length=params['slow_vidya_length'],
            correcting_constant=params['correcting_constant'],
            ma_type=params['ma_type']
        )
        
        
        
        warmup = EXPECTED_OUTPUTS['otto']['warmup_period']
        
        
        for i in range(warmup, len(data)):
            assert not np.isnan(hott[i]), f"Expected valid HOTT at index {i}"
            assert not np.isnan(lott[i]), f"Expected valid LOTT at index {i}"
    
    def test_otto_stream_reset(self):
        """Test OTTO streaming reset functionality"""
        params = EXPECTED_OUTPUTS['otto']['default_params']
        
        stream = ta_indicators.OttoStreamPy(
            ott_period=params['ott_period'],
            ott_percent=params['ott_percent'],
            fast_vidya_length=params['fast_vidya_length'],
            slow_vidya_length=params['slow_vidya_length'],
            correcting_constant=params['correcting_constant'],
            ma_type=params['ma_type']
        )
        
        
        for i in range(100):
            stream.update(float(i))
        
        
        stream.reset()
        
        
        result = stream.update(1.0)
        
        assert result == (None, None), "Should return (None, None) after reset"
        
        
        data = np.arange(100, 200, dtype=float)
        for i in range(50):
            result = stream.update(data[i])
            if i < 30:  
                assert result == (None, None) or np.isnan(result[0]), f"Expected None/NaN during warmup at index {i}"
    
    def test_otto_consecutive_nan_values(self, otto_test_data):
        """Test OTTO with consecutive NaN values"""
        data = otto_test_data[:300].copy()  
        
        
        data[50:60] = np.nan  
        data[100:110] = np.nan  
        
        params = EXPECTED_OUTPUTS['otto']['default_params']
        hott, lott = ta_indicators.otto(
            data,
            ott_period=params['ott_period'],
            ott_percent=params['ott_percent'],
            fast_vidya_length=params['fast_vidya_length'],
            slow_vidya_length=params['slow_vidya_length'],
            correcting_constant=params['correcting_constant'],
            ma_type=params['ma_type']
        )
        
        assert len(hott) == len(data)
        assert len(lott) == len(data)
        
        
        valid_hott = sum(1 for i in range(200, len(hott)) if not np.isnan(hott[i]))
        valid_lott = sum(1 for i in range(200, len(lott)) if not np.isnan(lott[i]))
        
        assert valid_hott > 0, "Should produce valid HOTT values despite consecutive NaNs"
        assert valid_lott > 0, "Should produce valid LOTT values despite consecutive NaNs"
    
    def test_otto_alternating_nan_pattern(self, otto_test_data):
        """Test OTTO with alternating NaN/valid pattern"""
        data = otto_test_data[:300].copy()  
        
        
        for i in range(50, 100, 2):
            data[i] = np.nan
        
        params = EXPECTED_OUTPUTS['otto']['default_params']
        hott, lott = ta_indicators.otto(
            data,
            ott_period=params['ott_period'],
            ott_percent=params['ott_percent'],
            fast_vidya_length=params['fast_vidya_length'],
            slow_vidya_length=params['slow_vidya_length'],
            correcting_constant=params['correcting_constant'],
            ma_type=params['ma_type']
        )
        
        assert len(hott) == len(data)
        assert len(lott) == len(data)
        
        
        valid_count = 0
        for i in range(150, len(data)):
            if not np.isnan(hott[i]) and not np.isnan(lott[i]):
                valid_count += 1
        
        assert valid_count > 50, "Should produce valid values despite alternating NaN pattern"
    
    def test_otto_extreme_parameter_values(self, otto_test_data):
        """Test OTTO with extreme but valid parameter values"""
        data = otto_test_data[:300]  
        
        
        hott, lott = ta_indicators.otto(
            data,
            ott_period=2,
            ott_percent=0.01,  
            fast_vidya_length=5,
            slow_vidya_length=10,
            correcting_constant=1.0,  
            ma_type="SMA"
        )
        assert len(hott) == len(data)
        assert len(lott) == len(data)
        
        
        hott, lott = ta_indicators.otto(
            data,
            ott_period=3,
            ott_percent=0.95,  
            fast_vidya_length=10,
            slow_vidya_length=25,
            correcting_constant=1000000.0,  
            ma_type="EMA"
        )
        assert len(hott) == len(data)
        assert len(lott) == len(data)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
