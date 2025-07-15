"""
Python binding tests for BOP indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

try:
    import my_project as ta_indicators
except ImportError:
    # If not in virtual environment, try to import from installed location
    try:
        import my_project as ta_indicators
    except ImportError:
        pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS
from rust_comparison import compare_with_rust


class TestBop:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_bop_partial_params(self, test_data):
        """Test BOP with standard parameters - mirrors check_bop_partial_params"""
        open_data = test_data['open']
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # BOP has no parameters, just OHLC inputs
        result = ta_indicators.bop(open_data, high, low, close)
        assert len(result) == len(close)
    
    def test_bop_accuracy(self, test_data):
        """Test BOP matches expected values from Rust tests - mirrors check_bop_accuracy"""
        open_data = test_data['open']
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['bop']
        
        result = ta_indicators.bop(open_data, high, low, close)
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-10,
            msg="BOP last 5 values mismatch"
        )
        
        # BOP doesn't use a single source like other indicators, it uses all OHLC data
        # so we skip the Rust comparison which expects a single source parameter
    
    def test_bop_default_candles(self, test_data):
        """Test BOP with default parameters - mirrors check_bop_default_candles"""
        open_data = test_data['open']
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        result = ta_indicators.bop(open_data, high, low, close)
        assert len(result) == len(close)
    
    def test_bop_with_empty_data(self):
        """Test BOP fails with empty data - mirrors check_bop_with_empty_data"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="bop: Data is empty"):
            ta_indicators.bop(empty, empty, empty, empty)
    
    def test_bop_with_inconsistent_lengths(self):
        """Test BOP fails with inconsistent input lengths - mirrors check_bop_with_inconsistent_lengths"""
        open_data = np.array([1.0, 2.0, 3.0])
        high = np.array([1.5, 2.5])  # Wrong length
        low = np.array([0.8, 1.8, 2.8])
        close = np.array([1.2, 2.2, 3.2])
        
        with pytest.raises(ValueError, match="bop: Input arrays have different lengths"):
            ta_indicators.bop(open_data, high, low, close)
    
    def test_bop_very_small_dataset(self):
        """Test BOP with single data point - mirrors check_bop_very_small_dataset"""
        open_data = np.array([10.0])
        high = np.array([12.0])
        low = np.array([9.5])
        close = np.array([11.0])
        
        result = ta_indicators.bop(open_data, high, low, close)
        assert len(result) == 1
        # (11.0 - 10.0) / (12.0 - 9.5) = 1.0 / 2.5 = 0.4
        assert_close(result[0], 0.4, rtol=1e-10, msg="BOP single value calculation")
    
    def test_bop_with_slice_data_reinput(self, test_data):
        """Test BOP with slice data re-input - mirrors check_bop_with_slice_data_reinput"""
        open_data = test_data['open']
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # First pass
        first_result = ta_indicators.bop(open_data, high, low, close)
        assert len(first_result) == len(close)
        
        # Second pass - use first result as close, zeros for others
        dummy = np.zeros_like(first_result)
        second_result = ta_indicators.bop(dummy, dummy, dummy, first_result)
        assert len(second_result) == len(first_result)
        
        # All values should be 0.0 since (first_result - 0) / (0 - 0) = 0.0
        for i, val in enumerate(second_result):
            assert_close(val, 0.0, atol=1e-15, 
                        msg=f"Expected BOP=0.0 for dummy data at idx {i}")
    
    def test_bop_nan_handling(self, test_data):
        """Test BOP handles values correctly without NaN - mirrors check_bop_nan_handling"""
        open_data = test_data['open']
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        result = ta_indicators.bop(open_data, high, low, close)
        assert len(result) == len(close)
        
        # BOP should not produce NaN values after any warmup period
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after index 240"
        
        # Actually, BOP has no warmup period - it calculates from the first value
        assert not np.any(np.isnan(result)), "BOP should not produce any NaN values"
    
    def test_bop_streaming(self, test_data):
        """Test BOP streaming matches batch calculation - mirrors check_bop_streaming"""
        open_data = test_data['open']
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Batch calculation
        batch_result = ta_indicators.bop(open_data, high, low, close)
        
        # Streaming calculation
        stream = ta_indicators.BopStream()
        stream_values = []
        
        for i in range(len(close)):
            result = stream.update(open_data[i], high[i], low[i], close[i])
            stream_values.append(result)
        
        stream_values = np.array(stream_values)
        
        # Compare batch vs streaming
        assert len(batch_result) == len(stream_values)
        
        # Compare all values
        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            assert_close(b, s, rtol=1e-12, atol=1e-12, 
                        msg=f"BOP streaming mismatch at index {i}")
    
    def test_bop_batch(self, test_data):
        """Test BOP batch processing - mirrors check_batch_default_row"""
        open_data = test_data['open']
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # BOP has no parameters, but we still provide batch API for consistency
        result = ta_indicators.bop_batch(open_data, high, low, close)
        
        assert 'values' in result
        assert 'rows' in result
        assert 'cols' in result
        assert 'params' in result
        
        # Should have 1 row (no parameters to sweep)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        assert result['rows'] == 1
        assert result['cols'] == len(close)
        
        # Extract the single row
        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['bop']['last_5_values']
        
        # Check last 5 values match
        assert_close(
            default_row[-5:],
            expected,
            rtol=1e-10,
            msg="BOP batch default row mismatch"
        )
        
        # Verify it matches the regular BOP calculation
        regular_result = ta_indicators.bop(open_data, high, low, close)
        assert_close(
            default_row,
            regular_result,
            rtol=1e-12,
            msg="BOP batch should match regular calculation"
        )
    
    def test_bop_zero_range_handling(self):
        """Test BOP when high equals low (zero range)"""
        # When high == low, BOP should return 0.0
        open_data = np.array([10.0, 20.0, 30.0])
        high = np.array([15.0, 25.0, 35.0])
        low = np.array([15.0, 25.0, 35.0])  # Same as high
        close = np.array([15.0, 25.0, 35.0])
        
        result = ta_indicators.bop(open_data, high, low, close)
        
        # All values should be 0.0 since denominator is 0
        for i, val in enumerate(result):
            assert_close(val, 0.0, atol=1e-15, 
                        msg=f"Expected BOP=0.0 when high=low at idx {i}")
    
    def test_bop_with_kernel_options(self, test_data):
        """Test BOP with different kernel options"""
        open_data = test_data['open']
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Test with explicit scalar kernel
        result_scalar = ta_indicators.bop(open_data, high, low, close, kernel='scalar')
        assert len(result_scalar) == len(close)
        
        # Test with auto kernel (default)
        result_auto = ta_indicators.bop(open_data, high, low, close, kernel='auto')
        assert len(result_auto) == len(close)
        
        # Results should be very close regardless of kernel
        assert_close(result_scalar, result_auto, rtol=1e-12, 
                    msg="BOP results should match across kernels")
        
        # Test batch kernels
        result_batch = ta_indicators.bop_batch(open_data, high, low, close, kernel='scalar')
        assert result_batch['values'].shape[0] == 1
        assert result_batch['values'].shape[1] == len(close)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])