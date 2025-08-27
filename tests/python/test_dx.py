"""
Python binding tests for DX indicator.
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


class TestDx:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_dx_accuracy(self, test_data):
        """Test DX matches expected values from Rust tests - mirrors check_dx_accuracy"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['dx']
        
        result = ta_indicators.dx(
            high, low, close,
            period=expected['default_params']['period']
        )
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-4,  # DX uses less precise tolerance
            msg="DX last 5 values mismatch"
        )
        
        # Verify DX values are in valid range [0, 100]
        valid_values = result[~np.isnan(result)]
        assert np.all((valid_values >= 0) & (valid_values <= 100)), \
            "DX values should be between 0 and 100"
    
    def test_dx_default_params(self, test_data):
        """Test DX with default parameters - mirrors check_dx_default_candles"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Default period is 14
        result = ta_indicators.dx(high, low, close, period=14)
        assert len(result) == len(close)
        
        # Verify warmup period (should have NaN values)
        # Warmup = first_valid_idx + period - 1 (minimum 13)
        assert np.any(np.isnan(result[:13])), "Expected NaN in warmup period"
        
        # Verify we have valid values after warmup
        if len(result) > 50:
            assert not np.all(np.isnan(result[50:])), "Expected valid values after warmup"
    
    def test_dx_zero_period(self, test_data):
        """Test DX fails with zero period - mirrors check_dx_zero_period"""
        high = np.array([2.0, 2.5, 3.0])
        low = np.array([1.0, 1.2, 2.1])
        close = np.array([1.5, 2.3, 2.2])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.dx(high, low, close, period=0)
    
    def test_dx_period_exceeds_length(self):
        """Test DX fails when period exceeds data length - mirrors check_dx_period_exceeds_length"""
        high = np.array([3.0, 4.0])
        low = np.array([2.0, 3.0])
        close = np.array([2.5, 3.5])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.dx(high, low, close, period=14)
    
    def test_dx_very_small_dataset(self):
        """Test DX fails with insufficient data - mirrors check_dx_very_small_dataset"""
        high = np.array([3.0])
        low = np.array([2.0])
        close = np.array([2.5])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.dx(high, low, close, period=14)
    
    def test_dx_empty_input(self):
        """Test DX fails with empty input - mirrors check_dx_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.dx(empty, empty, empty, period=14)
    
    def test_dx_all_nan_input(self):
        """Test DX with all NaN values - mirrors check_dx_all_nan"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All high, low, and close values are NaN"):
            ta_indicators.dx(all_nan, all_nan, all_nan, period=14)
    
    def test_dx_mismatched_lengths(self):
        """Test DX fails with mismatched input lengths"""
        high = np.array([1.0, 2.0, 3.0])
        low = np.array([1.0, 2.0])  # Different length
        close = np.array([1.0, 2.0, 3.0])
        
        # The function should handle this gracefully by using min length
        result = ta_indicators.dx(high, low, close, period=1)
        assert len(result) == 2  # min(3, 2, 3) = 2
    
    def test_dx_nan_handling(self, test_data):
        """Test DX handles NaN values correctly - mirrors check_dx_nan_handling"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        result = ta_indicators.dx(high, low, close, period=14)
        assert len(result) == len(close)
        
        # After sufficient warmup period, no NaN values should exist
        if len(result) > 50:
            assert not np.any(np.isnan(result[50:])), \
                "Found unexpected NaN after warmup period"
        
        # First period-1 values should be NaN (minimum 13 for period=14)
        assert np.any(np.isnan(result[:13])), "Expected NaN in warmup period"
    
    def test_dx_streaming(self, test_data):
        """Test DX streaming matches batch calculation - mirrors check_dx_streaming"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        period = 14
        
        # Batch calculation
        batch_result = ta_indicators.dx(high, low, close, period=period)
        
        # Streaming calculation
        stream = ta_indicators.DxStream(period=period)
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
            assert_close(b, s, rtol=1e-9, atol=1e-9, 
                        msg=f"DX streaming mismatch at index {i}")
    
    def test_dx_batch_single_params(self, test_data):
        """Test DX batch processing with single parameter - mirrors check_batch_default_row"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        result = ta_indicators.dx_batch(
            high, low, close,
            period_range=(14, 14, 0)  # Default period only
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        # Should have 1 combination
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        # Extract the single row
        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['dx']['last_5_values']
        
        # Check last 5 values match
        assert_close(
            default_row[-5:],
            expected,
            rtol=1e-4,
            msg="DX batch default row mismatch"
        )
    
    def test_dx_batch_multiple_periods(self, test_data):
        """Test DX batch with multiple periods - mirrors check_batch_sweep"""
        high = test_data['high'][:100]  # Use smaller dataset for speed
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        
        result = ta_indicators.dx_batch(
            high, low, close,
            period_range=(10, 30, 5)  # periods: 10, 15, 20, 25, 30
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        # Should have 5 combinations
        expected_combos = 5
        assert result['values'].shape[0] == expected_combos
        assert result['values'].shape[1] == len(close)
        assert len(result['periods']) == expected_combos
        
        # Verify periods array
        expected_periods = [10, 15, 20, 25, 30]
        np.testing.assert_array_equal(result['periods'], expected_periods)
        
        # Verify each batch result matches individual computation
        for i, period in enumerate(expected_periods):
            batch_row = result['values'][i]
            single_result = ta_indicators.dx(high, low, close, period=period)
            assert_close(batch_row, single_result, rtol=1e-10,
                        msg=f"Batch result for period {period} should match single computation")
    
    def test_dx_batch_with_kernel(self, test_data):
        """Test DX batch with kernel parameter"""
        high = test_data['high'][:50]
        low = test_data['low'][:50]
        close = test_data['close'][:50]
        
        # Test with different kernels
        result_auto = ta_indicators.dx_batch(
            high, low, close,
            period_range=(14, 14, 0),
            kernel=None  # Auto kernel
        )
        
        result_scalar = ta_indicators.dx_batch(
            high, low, close,
            period_range=(14, 14, 0),
            kernel='scalar'
        )
        
        # Both should produce valid results
        assert result_auto['values'].shape == result_scalar['values'].shape
        
        # Results should be very close (might differ slightly due to kernel differences)
        assert_close(
            result_auto['values'][0],
            result_scalar['values'][0],
            rtol=1e-8,
            msg="Different kernels should produce similar results"
        )
    
    def test_dx_value_bounds(self, test_data):
        """Test that DX values are always between 0 and 100"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Test with various periods
        for period in [5, 10, 14, 20, 50]:
            if period >= len(high):
                continue
                
            result = ta_indicators.dx(high, low, close, period=period)
            
            # Check all non-NaN values are in valid range
            valid_values = result[~np.isnan(result)]
            if len(valid_values) > 0:
                assert np.all(valid_values >= -1e-9), \
                    f"DX values should be >= 0 for period {period}"
                assert np.all(valid_values <= 100.0 + 1e-9), \
                    f"DX values should be <= 100 for period {period}"
    
    def test_dx_with_kernel_parameter(self, test_data):
        """Test DX with explicit kernel parameter"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        
        # Test with scalar kernel
        result_scalar = ta_indicators.dx(
            high, low, close,
            period=14,
            kernel='scalar'
        )
        
        # Test with auto kernel
        result_auto = ta_indicators.dx(
            high, low, close,
            period=14,
            kernel=None
        )
        
        # Both should have same length
        assert len(result_scalar) == len(result_auto) == len(close)
        
        # Results should be very close
        assert_close(
            result_scalar,
            result_auto,
            rtol=1e-8,
            msg="Different kernels should produce similar results"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])