"""
Python binding tests for CCI indicator.
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


class TestCci:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_cci_partial_params(self, test_data):
        """Test CCI with partial parameters (None values) - mirrors check_cci_partial_params"""
        close = test_data['close']
        
        # Test with default params
        result = ta_indicators.cci(close, 14)  # Using default period
        assert len(result) == len(close)
        
        # Test with different sources
        high = test_data['high']
        low = test_data['low']
        
        # Create hl2 source
        hl2 = (high + low) / 2
        result_hl2 = ta_indicators.cci(hl2, 20)
        assert len(result_hl2) == len(hl2)
        
        # Create hlc3 source (default in Rust)
        hlc3 = (high + low + close) / 3
        result_hlc3 = ta_indicators.cci(hlc3, 9)
        assert len(result_hlc3) == len(hlc3)
    
    def test_cci_accuracy(self, test_data):
        """Test CCI matches expected values from Rust tests - mirrors check_cci_accuracy"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['cci']
        
        # Create hlc3 source (default in Rust)
        hlc3 = (high + low + close) / 3
        
        result = ta_indicators.cci(
            hlc3,
            period=expected['default_params']['period']
        )
        
        assert len(result) == len(hlc3)
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-6,
            msg="CCI last 5 values mismatch"
        )
        
        # Verify warmup period
        period = expected['default_params']['period']
        for i in range(period - 1):
            assert np.isnan(result[i]), f"Expected NaN at index {i} for initial period warm-up"
        
        # Compare full output with Rust
        # TODO: Add CCI to generate_references binary
        # compare_with_rust('cci', result, 'hlc3', expected['default_params'])
    
    def test_cci_default_candles(self, test_data):
        """Test CCI with default parameters - mirrors check_cci_default_candles"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Create hlc3 source (default in Rust)
        hlc3 = (high + low + close) / 3
        
        # Default params: period=14
        result = ta_indicators.cci(hlc3, 14)
        assert len(result) == len(hlc3)
    
    def test_cci_zero_period(self):
        """Test CCI fails with zero period - mirrors check_cci_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.cci(input_data, period=0)
    
    def test_cci_period_exceeds_length(self):
        """Test CCI fails when period exceeds data length - mirrors check_cci_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.cci(data_small, period=10)
    
    def test_cci_very_small_dataset(self):
        """Test CCI fails with insufficient data - mirrors check_cci_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.cci(single_point, period=9)
    
    def test_cci_empty_input(self):
        """Test CCI fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.cci(empty, period=14)
    
    def test_cci_reinput(self, test_data):
        """Test CCI applied twice (re-input) - mirrors check_cci_reinput"""
        close = test_data['close']
        
        # First pass
        first_result = ta_indicators.cci(close, period=14)
        assert len(first_result) == len(close)
        
        # Second pass - apply CCI to CCI output
        second_result = ta_indicators.cci(first_result, period=14)
        assert len(second_result) == len(first_result)
        
        # After warmup period (28), no NaN values should exist
        if len(second_result) > 28:
            for i in range(28, len(second_result)):
                assert not np.isnan(second_result[i]), f"Expected no NaN after index 28, found NaN at index {i}"
    
    def test_cci_nan_handling(self, test_data):
        """Test CCI handles NaN values correctly - mirrors check_cci_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.cci(close, period=14)
        assert len(result) == len(close)
        
        # After warmup period (240), no NaN values should exist
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"
        
        # First period-1 values should be NaN
        assert np.all(np.isnan(result[:13])), "Expected NaN in warmup period"
    
    def test_cci_streaming(self, test_data):
        """Test CCI streaming matches batch calculation - mirrors check_cci_streaming"""
        close = test_data['close']
        period = 14
        
        # Batch calculation
        batch_result = ta_indicators.cci(close, period=period)
        
        # Streaming calculation
        stream = ta_indicators.CciStream(period=period)
        stream_values = []
        
        for price in close:
            result = stream.update(price)
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        # Compare batch vs streaming
        assert len(batch_result) == len(stream_values)
        
        # Compare values where both are not NaN
        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, atol=1e-9, 
                        msg=f"CCI streaming mismatch at index {i}")
    
    def test_cci_batch(self, test_data):
        """Test CCI batch processing - mirrors check_batch_default_row"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Create hlc3 source
        hlc3 = (high + low + close) / 3
        
        result = ta_indicators.cci_batch(
            hlc3,
            period_range=(14, 14, 0)  # Default period only
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        # Should have 1 combination (default params)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(hlc3)
        
        # Extract the single row
        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['cci']['last_5_values']
        
        # Check last 5 values match
        assert_close(
            default_row[-5:],
            expected,
            rtol=1e-6,
            msg="CCI batch default row mismatch"
        )
    
    def test_cci_all_nan_input(self):
        """Test CCI with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.cci(all_nan, period=14)
    
    def test_cci_batch_sweep(self, test_data):
        """Test CCI batch with multiple periods"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        result = ta_indicators.cci_batch(
            close,
            period_range=(10, 20, 2)  # 10, 12, 14, 16, 18, 20
        )
        
        # Should have 6 combinations
        assert result['values'].shape[0] == 6
        assert result['values'].shape[1] == len(close)
        assert len(result['periods']) == 6
        
        # Verify periods
        expected_periods = [10, 12, 14, 16, 18, 20]
        assert list(result['periods']) == expected_periods
        
        # Verify each row matches individual calculation
        for i, period in enumerate(expected_periods):
            row_data = result['values'][i]
            single_result = ta_indicators.cci(close, period)
            assert_close(
                row_data,
                single_result,
                rtol=1e-10,
                msg=f"Period {period} mismatch"
            )
    
    def test_cci_batch_kernel_parameter(self, test_data):
        """Test CCI batch accepts kernel parameter"""
        close = test_data['close'][:50]
        
        # Test with explicit scalar kernel
        result_scalar = ta_indicators.cci_batch(
            close,
            period_range=(14, 14, 0),
            kernel='scalar'
        )
        
        # Test with auto kernel
        result_auto = ta_indicators.cci_batch(
            close,
            period_range=(14, 14, 0),
            kernel='auto'
        )
        
        # Results should be similar (may use different kernels)
        assert_close(
            result_scalar['values'][0],
            result_auto['values'][0],
            rtol=1e-9,
            msg="Kernel results mismatch"
        )
    
    def test_cci_kernel_parameter(self, test_data):
        """Test CCI with different kernel parameters"""
        close = test_data['close'][:100]
        
        # Test with explicit scalar kernel
        result_scalar = ta_indicators.cci(close, 14, kernel='scalar')
        
        # Test with auto kernel
        result_auto = ta_indicators.cci(close, 14, kernel='auto')
        
        # Test without kernel parameter (should default to auto)
        result_default = ta_indicators.cci(close, 14)
        
        # All results should be similar
        assert_close(result_scalar, result_auto, rtol=1e-9, msg="Scalar vs auto mismatch")
        assert_close(result_auto, result_default, rtol=1e-9, msg="Auto vs default mismatch")
    
    def test_cci_invalid_kernel(self):
        """Test CCI with invalid kernel parameter"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        with pytest.raises(ValueError, match="Unknown kernel"):
            ta_indicators.cci(data, 2, kernel='invalid_kernel')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])