"""
Python binding tests for CMO indicator.
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

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS
from rust_comparison import compare_with_rust


class TestCmo:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_cmo_partial_params(self, test_data):
        """Test CMO with partial parameters (None values) - mirrors check_cmo_partial_params"""
        close = test_data['close']
        
        
        result = ta_indicators.cmo(close)  
        assert len(result) == len(close)
        
        
        result2 = ta_indicators.cmo(close, period=10)
        assert len(result2) == len(close)
    
    def test_cmo_accuracy(self, test_data):
        """Test CMO matches expected values from Rust tests - mirrors check_cmo_accuracy"""
        close = test_data['close']
        
        
        expected_last_five = [
            -13.152504931406101,
            -14.649876201213106,
            -16.760170709240303,
            -14.274505732779227,
            -21.984038127126716,
        ]
        
        result = ta_indicators.cmo(close, period=14)
        
        assert len(result) == len(close)
        
        
        assert_close(
            result[-5:], 
            expected_last_five,
            rtol=1e-6,
            msg="CMO last 5 values mismatch"
        )
        
        
        compare_with_rust('cmo', result, 'close', {'period': 14})
    
    def test_cmo_default_candles(self, test_data):
        """Test CMO with default parameters - mirrors check_cmo_default_candles"""
        close = test_data['close']
        
        
        result = ta_indicators.cmo(close)
        assert len(result) == len(close)
    
    def test_cmo_zero_period(self):
        """Test CMO fails with zero period - mirrors check_cmo_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.cmo(input_data, period=0)
    
    def test_cmo_period_exceeds_length(self):
        """Test CMO fails when period exceeds data length - mirrors check_cmo_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.cmo(data_small, period=10)
    
    def test_cmo_very_small_dataset(self):
        """Test CMO fails with insufficient data - mirrors check_cmo_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.cmo(single_point, period=14)
    
    def test_cmo_empty_input(self):
        """Test CMO fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data provided"):
            ta_indicators.cmo(empty, period=14)
    
    def test_cmo_reinput(self, test_data):
        """Test CMO applied twice (re-input) - mirrors check_cmo_reinput"""
        close = test_data['close']
        
        
        first_result = ta_indicators.cmo(close, period=14)
        assert len(first_result) == len(close)
        
        
        second_result = ta_indicators.cmo(first_result, period=14)
        assert len(second_result) == len(first_result)
        
        
        if len(second_result) > 28:
            assert not np.any(np.isnan(second_result[28:])), "Expected no NaN after index 28"
    
    def test_cmo_nan_handling(self, test_data):
        """Test CMO handles NaN values correctly - mirrors check_cmo_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.cmo(close, period=14)
        assert len(result) == len(close)
        
        
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"
        
        
        assert np.all(np.isnan(result[:14])), "Expected NaN in warmup period"
    
    def test_cmo_streaming(self, test_data):
        """Test CMO streaming matches batch calculation"""
        close = test_data['close']
        period = 14
        
        
        batch_result = ta_indicators.cmo(close, period=period)
        
        
        stream = ta_indicators.CmoStream(period=period)
        stream_values = []
        
        for price in close:
            result = stream.update(price)
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        
        assert len(batch_result) == len(stream_values)
        
        
        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, atol=1e-9, 
                        msg=f"CMO streaming mismatch at index {i}")
    
    def test_cmo_batch(self, test_data):
        """Test CMO batch processing"""
        close = test_data['close']
        
        result = ta_indicators.cmo_batch(
            close,
            period_range=(14, 14, 0),  
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        
        default_row = result['values'][0]
        
        
        expected_last_five = [
            -13.152504931406101,
            -14.649876201213106,
            -16.760170709240303,
            -14.274505732779227,
            -21.984038127126716,
        ]
        
        
        assert_close(
            default_row[-5:],
            expected_last_five,
            rtol=1e-6,
            msg="CMO batch default row mismatch"
        )
    
    def test_cmo_batch_multiple_periods(self, test_data):
        """Test CMO batch with multiple periods"""
        close = test_data['close'][:1000]  
        
        result = ta_indicators.cmo_batch(
            close,
            period_range=(10, 20, 2),  
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        
        assert result['values'].shape[0] == 6
        assert result['values'].shape[1] == len(close)
        
        
        expected_periods = [10, 12, 14, 16, 18, 20]
        assert np.array_equal(result['periods'], expected_periods)
        
        
        for i, period in enumerate(expected_periods):
            row = result['values'][i]
            
            assert np.all(np.isnan(row[:period])), f"Expected NaN warmup for period {period}"
            
            assert not np.all(np.isnan(row[period:])), f"Expected values after warmup for period {period}"
    
    def test_cmo_all_nan_input(self):
        """Test CMO with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.cmo(all_nan, period=14)
    
    def test_cmo_kernel_parameter(self, test_data):
        """Test CMO with different kernel parameters"""
        close = test_data['close'][:1000]  
        
        
        result_auto = ta_indicators.cmo(close, period=14, kernel='auto')
        
        
        result_scalar = ta_indicators.cmo(close, period=14, kernel='scalar')
        
        
        assert len(result_auto) == len(close)
        assert len(result_scalar) == len(close)
        
        
        
        assert_close(
            result_auto[14:],
            result_scalar[14:],
            rtol=1e-10,
            msg="Kernel results should match"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
