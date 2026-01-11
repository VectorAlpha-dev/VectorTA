"""
Python binding tests for RSX indicator.
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


class TestRsx:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_rsx_partial_params(self, test_data):
        """Test RSX with partial parameters (None values) - mirrors check_rsx_partial_params"""
        close = test_data['close']
        
        
        result = ta_indicators.rsx(close, 14)  
        assert len(result) == len(close)
    
    def test_rsx_accuracy(self, test_data):
        """Test RSX matches expected values from Rust tests - mirrors check_rsx_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['rsx']
        
        result = ta_indicators.rsx(
            close,
            period=expected['default_params']['period']
        )
        
        assert len(result) == len(close)
        
        
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-1,  
            msg="RSX last 5 values mismatch"
        )
        
        
        compare_with_rust('rsx', result, 'close', expected['default_params'])
    
    def test_rsx_default_candles(self, test_data):
        """Test RSX with default parameters - mirrors check_rsx_default_candles"""
        close = test_data['close']
        
        
        result = ta_indicators.rsx(close, 14)
        assert len(result) == len(close)
    
    def test_rsx_zero_period(self):
        """Test RSX fails with zero period - mirrors check_rsx_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.rsx(input_data, period=0)
    
    def test_rsx_period_exceeds_length(self):
        """Test RSX fails when period exceeds data length - mirrors check_rsx_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.rsx(data_small, period=10)
    
    def test_rsx_very_small_dataset(self):
        """Test RSX fails with insufficient data - mirrors check_rsx_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.rsx(single_point, period=14)
    
    def test_rsx_reinput(self, test_data):
        """Test RSX applied twice (re-input) - mirrors check_rsx_reinput"""
        close = test_data['close']
        
        
        first_result = ta_indicators.rsx(close, period=14)
        assert len(first_result) == len(close)
        
        
        second_result = ta_indicators.rsx(first_result, period=14)
        assert len(second_result) == len(first_result)
    
    def test_rsx_nan_handling(self, test_data):
        """Test RSX handles NaN values correctly - mirrors check_rsx_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.rsx(close, period=14)
        assert len(result) == len(close)
        
        
        if len(result) > 50:
            assert not np.any(np.isnan(result[50:])), "Found unexpected NaN after warmup period"
    
    def test_rsx_streaming(self, test_data):
        """Test RSX streaming matches batch calculation - mirrors check_rsx_streaming"""
        close = test_data['close']
        period = 14
        
        
        batch_result = ta_indicators.rsx(close, period=period)
        
        
        stream = ta_indicators.RsxStream(period=period)
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
                        msg=f"RSX streaming mismatch at index {i}")
    
    def test_rsx_batch(self, test_data):
        """Test RSX batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        result = ta_indicators.rsx_batch(
            close,
            period_range=(14, 100, 1)  
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        
        periods = result['periods']
        default_idx = None
        for i, p in enumerate(periods):
            if p == 14:
                default_idx = i
                break
        
        assert default_idx is not None, "Default period (14) not found in batch results"
        
        
        values_2d = result['values']
        default_row = values_2d[default_idx]
        expected = EXPECTED_OUTPUTS['rsx']['last_5_values']
        
        
        assert_close(
            default_row[-5:],
            expected,
            rtol=1e-1,  
            msg="RSX batch default row mismatch"
        )
    
    def test_rsx_all_nan_input(self):
        """Test RSX with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.rsx(all_nan, period=14)
    
    def test_rsx_kernel_parameter(self, test_data):
        """Test RSX with explicit kernel parameter"""
        close = test_data['close']
        
        
        result_scalar = ta_indicators.rsx(close, period=14, kernel="scalar")
        assert len(result_scalar) == len(close)
        
        
        result_auto = ta_indicators.rsx(close, period=14, kernel="auto")
        assert len(result_auto) == len(close)
        
        
        assert_close(result_scalar, result_auto, rtol=1e-9, atol=1e-9)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])