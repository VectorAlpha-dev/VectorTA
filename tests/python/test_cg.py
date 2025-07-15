"""
Python binding tests for CG (Center of Gravity) indicator.
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


class TestCg:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_cg_partial_params(self, test_data):
        """Test CG with partial parameters - mirrors check_cg_partial_params"""
        close = test_data['close']
        
        # Test with custom period
        result = ta_indicators.cg(close, period=12)
        assert len(result) == len(close)
    
    def test_cg_accuracy(self, test_data):
        """Test CG matches expected values from Rust tests - mirrors check_cg_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['cg']
        
        result = ta_indicators.cg(
            close,
            period=expected['default_params']['period']
        )
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-4,  # CG uses 1e-4 tolerance in Rust tests
            msg="CG last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('cg', result, 'close', expected['default_params'])
    
    def test_cg_default_candles(self, test_data):
        """Test CG with default parameters - mirrors check_cg_default_candles"""
        close = test_data['close']
        
        # Default period is 10
        result = ta_indicators.cg(close, 10)
        assert len(result) == len(close)
    
    def test_cg_zero_period(self):
        """Test CG fails with zero period - mirrors check_cg_zero_period"""
        data = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.cg(data, period=0)
    
    def test_cg_period_exceeds_length(self):
        """Test CG fails when period exceeds data length - mirrors check_cg_period_exceeds_length"""
        data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.cg(data, period=10)
    
    def test_cg_very_small_dataset(self):
        """Test CG fails with insufficient data - mirrors check_cg_very_small_dataset"""
        data = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.cg(data, period=10)
    
    def test_cg_empty_input(self):
        """Test CG fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.cg(empty, period=10)
    
    def test_cg_nan_handling(self, test_data):
        """Test CG handles NaN values correctly - mirrors check_cg_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.cg(close, period=10)
        assert len(result) == len(close)
        
        # After warmup period, check for valid values
        check_idx = 240
        if len(result) > check_idx:
            # Find first non-NaN value after check_idx
            found_valid = False
            for i in range(check_idx, len(result)):
                if not np.isnan(result[i]):
                    found_valid = True
                    break
            assert found_valid, f"All CG values from index {check_idx} onward are NaN."
        
        # First period values should be NaN (CG starts at first + period)
        # With default period=10, first 10 values should be NaN
        assert np.all(np.isnan(result[:10])), "Expected NaN in warmup period"
    
    def test_cg_streaming(self, test_data):
        """Test CG streaming matches batch calculation - mirrors check_cg_streaming"""
        close = test_data['close']
        period = 10
        
        # Batch calculation
        batch_result = ta_indicators.cg(close, period=period)
        
        # Streaming calculation
        stream = ta_indicators.CgStream(period=period)
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
                        msg=f"CG streaming mismatch at index {i}")
    
    def test_cg_batch(self, test_data):
        """Test CG batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        result = ta_indicators.cg_batch(
            close,
            period_range=(10, 10, 0),  # Default period only
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        # Should have 1 combination (default params)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        # Extract the single row
        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['cg']['last_5_values']
        
        # Check last 5 values match
        assert_close(
            default_row[-5:],
            expected,
            rtol=1e-4,  # CG uses 1e-4 tolerance
            msg="CG batch default row mismatch"
        )
    
    def test_cg_batch_multiple_periods(self, test_data):
        """Test CG batch with multiple period values"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        # Multiple periods: 10, 12, 14
        result = ta_indicators.cg_batch(
            close,
            period_range=(10, 14, 2),
        )
        
        # Should have 3 rows
        assert result['values'].shape[0] == 3
        assert result['values'].shape[1] == 100
        assert len(result['periods']) == 3
        assert list(result['periods']) == [10, 12, 14]
        
        # Verify first row matches single calculation
        single_result = ta_indicators.cg(close, period=10)
        assert_close(
            result['values'][0],
            single_result,
            rtol=1e-10,
            msg="Batch first row mismatch"
        )
    
    def test_cg_all_nan_input(self):
        """Test CG with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.cg(all_nan, period=10)
    
    def test_cg_kernel_parameter(self, test_data):
        """Test CG with different kernel parameters"""
        close = test_data['close'][:100]  # Use smaller dataset
        
        # Test with scalar kernel
        result_scalar = ta_indicators.cg(close, period=10, kernel='scalar')
        assert len(result_scalar) == len(close)
        
        # Test with auto kernel
        result_auto = ta_indicators.cg(close, period=10, kernel='auto')
        assert len(result_auto) == len(close)
        
        # Results should be very close regardless of kernel
        # (since AVX implementations currently fall back to scalar)
        assert_close(
            result_scalar,
            result_auto,
            rtol=1e-10,
            msg="Kernel results mismatch"
        )
    
    def test_cg_warmup_period(self, test_data):
        """Test CG respects warmup period requirement"""
        close = test_data['close']
        period = 10
        
        result = ta_indicators.cg(close, period=period)
        
        # CG requires period + 1 valid points, outputs start at first + period
        # So with period=10, first 10 values should be NaN
        assert np.all(np.isnan(result[:period])), f"Expected NaN in first {period} values"
        
        # Value at index period should be valid (if input has enough data)
        if len(close) > period:
            assert not np.isnan(result[period]), f"Expected valid value at index {period}"
    
    def test_cg_edge_case_small_period(self, test_data):
        """Test CG with very small period"""
        close = test_data['close'][:20]
        
        # Period of 2 (minimum sensible value)
        result = ta_indicators.cg(close, period=2)
        assert len(result) == len(close)
        
        # First 2 values should be NaN
        assert np.all(np.isnan(result[:2]))
        # Third value onwards should be valid
        assert not np.isnan(result[2])
    
    def test_cg_batch_empty_range(self, test_data):
        """Test CG batch with single value range"""
        close = test_data['close'][:50]
        
        # Step of 0 means single value
        result = ta_indicators.cg_batch(
            close,
            period_range=(12, 12, 0),
        )
        
        assert result['values'].shape[0] == 1
        assert result['periods'][0] == 12
    
    def test_cg_batch_kernel_parameter(self, test_data):
        """Test CG batch with kernel parameter"""
        close = test_data['close'][:50]
        
        # Test batch with scalar kernel
        result = ta_indicators.cg_batch(
            close,
            period_range=(10, 12, 2),
            kernel='scalar'
        )
        
        assert result['values'].shape[0] == 2
        assert list(result['periods']) == [10, 12]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])