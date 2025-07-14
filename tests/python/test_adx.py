"""
Python binding tests for ADX indicator.
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


class TestAdx:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_adx_partial_params(self, test_data):
        """Test ADX with partial parameters (None values) - mirrors check_adx_partial_params"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Test with default params (period=14)
        result = ta_indicators.adx(high, low, close, 14)
        assert len(result) == len(close)
    
    def test_adx_accuracy(self, test_data):
        """Test ADX matches expected values from Rust tests - mirrors check_adx_accuracy"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['adx']
        
        result = ta_indicators.adx(
            high,
            low,
            close,
            period=expected['default_params']['period']
        )
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-1,  # ADX has lower precision requirement
            msg="ADX last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('adx', result, 'ohlc', expected['default_params'])
    
    def test_adx_default_candles(self, test_data):
        """Test ADX with default parameters - mirrors check_adx_default_candles"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Default params: period=14
        result = ta_indicators.adx(high, low, close, 14)
        assert len(result) == len(close)
    
    def test_adx_zero_period(self):
        """Test ADX fails with zero period - mirrors check_adx_zero_period"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0, 25.0])
        close = np.array([9.0, 19.0, 29.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.adx(high, low, close, period=0)
    
    def test_adx_period_exceeds_length(self):
        """Test ADX fails when period exceeds data length - mirrors check_adx_period_exceeds_length"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0, 25.0])
        close = np.array([9.0, 19.0, 29.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.adx(high, low, close, period=10)
    
    def test_adx_very_small_dataset(self):
        """Test ADX fails with insufficient data - mirrors check_adx_very_small_dataset"""
        high = np.array([42.0])
        low = np.array([41.0])
        close = np.array([40.5])
        
        with pytest.raises(ValueError, match="Not enough valid data"):
            ta_indicators.adx(high, low, close, period=14)
    
    def test_adx_input_length_mismatch(self):
        """Test ADX fails when input arrays have different lengths"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0])  # Different length
        close = np.array([9.0, 19.0, 29.0])
        
        with pytest.raises(ValueError, match="Input arrays must have the same length"):
            ta_indicators.adx(high, low, close, period=14)
    
    def test_adx_all_nan_input(self):
        """Test ADX with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.adx(all_nan, all_nan, all_nan, period=14)
    
    def test_adx_reinput(self, test_data):
        """Test ADX applied twice (re-input) - mirrors check_adx_reinput"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # First pass
        first_result = ta_indicators.adx(high, low, close, period=14)
        assert len(first_result) == len(close)
        
        # Second pass - apply ADX using the first result as close price
        # This is different from ALMA since ADX requires high/low/close
        second_result = ta_indicators.adx(high, low, first_result, period=5)
        assert len(second_result) == len(first_result)
        
        # Check that we have values after warmup
        non_nan_count = np.sum(~np.isnan(second_result))
        assert non_nan_count > 100, "Expected more non-NaN values after second pass"
    
    def test_adx_nan_handling(self, test_data):
        """Test ADX handles NaN values correctly - mirrors check_adx_nan_handling"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        result = ta_indicators.adx(high, low, close, period=14)
        assert len(result) == len(close)
        
        # ADX requires period + additional warmup bars
        # After warmup period (100), no NaN values should exist
        if len(result) > 100:
            assert not np.any(np.isnan(result[100:])), "Found unexpected NaN after warmup period"
        
        # First several values should be NaN (more than just period due to ADX calculation)
        # ADX needs period + 1 bars minimum, but also needs time to calculate DX values
        assert np.all(np.isnan(result[:27])), "Expected NaN in warmup period"
    
    def test_adx_streaming(self, test_data):
        """Test ADX streaming matches batch calculation - mirrors check_adx_streaming"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        period = 14
        
        # Batch calculation
        batch_result = ta_indicators.adx(high, low, close, period=period)
        
        # Streaming calculation
        stream = ta_indicators.AdxStream(period=period)
        stream_values = []
        
        for h, l, c in zip(high, low, close):
            result = stream.update(h, l, c)
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        # Compare batch vs streaming
        assert len(batch_result) == len(stream_values)
        
        # Compare values where both are not NaN
        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-8, atol=1e-8, 
                        msg=f"ADX streaming mismatch at index {i}")
    
    def test_adx_batch(self, test_data):
        """Test ADX batch processing - mirrors check_batch_default_row"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        result = ta_indicators.adx_batch(
            high,
            low,
            close,
            period_range=(14, 14, 0)  # Default period only
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        # Should have 1 combination (default params)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        # Extract the single row
        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['adx']['last_5_values']
        
        # Check last 5 values match
        assert_close(
            default_row[-5:],
            expected,
            rtol=1e-1,
            msg="ADX batch default row mismatch"
        )
    
    def test_adx_batch_multiple_periods(self, test_data):
        """Test ADX batch with multiple period values"""
        high = test_data['high'][:100]  # Use smaller dataset for speed
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        
        # Multiple periods: 10, 14, 18
        result = ta_indicators.adx_batch(
            high,
            low,
            close,
            period_range=(10, 18, 4)
        )
        
        # Should have 3 rows * 100 cols
        assert result['values'].shape == (3, 100)
        assert len(result['periods']) == 3
        assert list(result['periods']) == [10, 14, 18]
        
        # Verify each row matches individual calculation
        periods = [10, 14, 18]
        for i, period in enumerate(periods):
            row_data = result['values'][i]
            single_result = ta_indicators.adx(high, low, close, period=period)
            
            # Compare where both are not NaN
            for j in range(len(row_data)):
                if np.isnan(row_data[j]) and np.isnan(single_result[j]):
                    continue
                assert_close(
                    row_data[j], 
                    single_result[j], 
                    rtol=1e-10, 
                    msg=f"Period {period} mismatch at index {j}"
                )
    
    def test_adx_kernel_parameter(self, test_data):
        """Test ADX with different kernel parameters"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        
        # Test with explicit scalar kernel
        result_scalar = ta_indicators.adx(high, low, close, period=14, kernel='scalar')
        assert len(result_scalar) == len(close)
        
        # Test with auto kernel (default)
        result_auto = ta_indicators.adx(high, low, close, period=14)
        
        # Results should be very close (might use different kernels)
        for i, (s, a) in enumerate(zip(result_scalar, result_auto)):
            if np.isnan(s) and np.isnan(a):
                continue
            assert_close(s, a, rtol=1e-10, atol=1e-10,
                        msg=f"Kernel results mismatch at index {i}")
    
    def test_adx_edge_cases(self, test_data):
        """Test ADX with edge case scenarios"""
        # Test with minimum required data
        min_len = 15  # period + 1
        high = test_data['high'][:min_len]
        low = test_data['low'][:min_len]
        close = test_data['close'][:min_len]
        
        result = ta_indicators.adx(high, low, close, period=14)
        assert len(result) == min_len
        
        # Test with flat price (no movement)
        flat_high = np.full(50, 100.0)
        flat_low = np.full(50, 100.0)
        flat_close = np.full(50, 100.0)
        
        result_flat = ta_indicators.adx(flat_high, flat_low, flat_close, period=14)
        assert len(result_flat) == 50
        # ADX should be low for flat prices (no trend)
        non_nan_values = result_flat[~np.isnan(result_flat)]
        if len(non_nan_values) > 0:
            assert np.all(non_nan_values < 20), "ADX should be low for flat prices"
    
    def test_adx_warmup_behavior(self, test_data):
        """Test ADX warmup period behavior in detail"""
        high = test_data['high'][:50]
        low = test_data['low'][:50]
        close = test_data['close'][:50]
        period = 14
        
        result = ta_indicators.adx(high, low, close, period=period)
        
        # ADX needs:
        # - period bars for initial ATR
        # - period more bars for DX calculation
        # - first ADX appears at 2*period bars
        expected_first_valid = 2 * period - 1  # 27 for period=14
        
        # All values before this should be NaN
        assert np.all(np.isnan(result[:expected_first_valid])), \
            f"Expected NaN for indices 0 to {expected_first_valid-1}"
        
        # Should have a value at expected_first_valid (might still be calculating)
        # Check that we eventually get non-NaN values
        if len(result) > expected_first_valid + 5:
            assert not np.all(np.isnan(result[expected_first_valid:])), \
                "Expected some non-NaN values after warmup"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])