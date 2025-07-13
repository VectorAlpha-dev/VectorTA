"""
Python binding tests for WILDERS indicator.
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


class TestWilders:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_wilders_partial_params(self, test_data):
        """Test Wilders with partial parameters (None values) - mirrors check_wilders_partial_params"""
        close = test_data['close']
        
        # Test with default period (5)
        result = ta_indicators.wilders(close, 5)
        assert len(result) == len(close)
    
    def test_wilders_accuracy(self, test_data):
        """Test Wilders matches expected values from Rust tests - mirrors check_wilders_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['wilders']
        
        result = ta_indicators.wilders(
            close,
            period=expected['default_params']['period']
        )
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-8,
            msg="Wilders last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('wilders', result, 'close', expected['default_params'])
    
    def test_wilders_default_candles(self, test_data):
        """Test Wilders with default parameters - mirrors check_wilders_default_candles"""
        close = test_data['close']
        
        # Default period: 5
        result = ta_indicators.wilders(close, 5)
        assert len(result) == len(close)
    
    def test_wilders_zero_period(self):
        """Test Wilders fails with zero period - mirrors check_wilders_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.wilders(input_data, period=0)
    
    def test_wilders_period_exceeds_length(self):
        """Test Wilders fails when period exceeds data length - mirrors check_wilders_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.wilders(data_small, period=10)
    
    def test_wilders_very_small_dataset(self):
        """Test Wilders with very small dataset - mirrors check_wilders_very_small_dataset"""
        single_point = np.array([42.0])
        
        # Should work with period=1
        result = ta_indicators.wilders(single_point, period=1)
        assert len(result) == 1
    
    def test_wilders_reinput(self, test_data):
        """Test Wilders applied twice (re-input) - mirrors check_wilders_reinput"""
        close = test_data['close']
        
        # First pass
        first_result = ta_indicators.wilders(close, period=5)
        assert len(first_result) == len(close)
        
        # Second pass - apply Wilders to Wilders output with different period
        second_result = ta_indicators.wilders(first_result, period=10)
        assert len(second_result) == len(first_result)
    
    def test_wilders_nan_handling(self, test_data):
        """Test Wilders handles NaN values correctly - mirrors check_wilders_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.wilders(close, period=5)
        assert len(result) == len(close)
        
        # After warmup period (240), no NaN values should exist
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"
        
        # First period-1 values should be NaN (accounting for potential leading NaNs in data)
        # Find first non-NaN in input
        first_valid = np.argmax(~np.isnan(close))
        warmup_end = first_valid + 5 - 1  # period - 1
        assert np.all(np.isnan(result[:warmup_end])), "Expected NaN in warmup period"
    
    def test_wilders_streaming(self, test_data):
        """Test Wilders streaming matches batch calculation - mirrors check_wilders_streaming"""
        close = test_data['close']
        period = 5
        
        # Batch calculation
        batch_result = ta_indicators.wilders(close, period=period)
        
        # Streaming calculation
        stream = ta_indicators.WildersStream(period=period)
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
                        msg=f"Wilders streaming mismatch at index {i}")
    
    def test_wilders_batch(self, test_data):
        """Test Wilders batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        result = ta_indicators.wilders_batch(
            close,
            period_range=(5, 5, 0),  # Default period only
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        # Should have 1 combination (default params)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        # Extract the single row
        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['wilders']['last_5_values']
        
        # Check last 5 values match
        assert_close(
            default_row[-5:],
            expected,
            rtol=1e-8,
            msg="Wilders batch default row mismatch"
        )
    
    def test_wilders_all_nan_input(self):
        """Test Wilders with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.wilders(all_nan, period=5)
    
    def test_wilders_kernel_selection(self, test_data):
        """Test Wilders with different kernel selections"""
        close = test_data['close']
        period = 5
        
        # Test different kernels
        for kernel in ['auto', 'scalar', 'avx2', 'avx512']:
            try:
                result = ta_indicators.wilders(close, period=period, kernel=kernel)
                assert len(result) == len(close)
            except ValueError as e:
                # AVX kernels might not be available on all systems
                if "Unknown kernel" not in str(e):
                    raise
    
    def test_wilders_batch_multiple_periods(self, test_data):
        """Test Wilders batch with multiple periods"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        result = ta_indicators.wilders_batch(
            close,
            period_range=(5, 10, 1),  # periods 5, 6, 7, 8, 9, 10
        )
        
        # Should have 6 rows (one for each period)
        assert result['values'].shape[0] == 6
        assert result['values'].shape[1] == len(close)
        
        # Verify periods array
        assert np.array_equal(result['periods'], [5, 6, 7, 8, 9, 10])
        
        # Verify each row has correct warmup
        for i, period in enumerate([5, 6, 7, 8, 9, 10]):
            row = result['values'][i]
            # First period-1 values should be NaN
            assert np.all(np.isnan(row[:period-1])), f"Expected NaN in warmup for period {period}"
            # After warmup should have values
            assert not np.any(np.isnan(row[period-1:])), f"Unexpected NaN after warmup for period {period}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])