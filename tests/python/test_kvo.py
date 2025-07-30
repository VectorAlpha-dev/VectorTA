"""
Python binding tests for KVO indicator.
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

from test_utils import load_test_data, assert_close


class TestKvo:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_kvo_partial_params(self, test_data):
        """Test KVO with partial parameters (None values) - mirrors check_kvo_partial_params"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        # Test with defaults (short_period=2, long_period=5)
        result = ta_indicators.kvo(high, low, close, volume)  # Using defaults
        assert len(result) == len(close)
    
    def test_kvo_accuracy(self, test_data):
        """Test KVO matches expected values from Rust tests - mirrors check_kvo_accuracy"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        # Using default parameters: short_period=2, long_period=5
        result = ta_indicators.kvo(high, low, close, volume, short_period=2, long_period=5)
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected from Rust tests
        expected_last_five = [
            -246.42698280402647,
            530.8651474164992,
            237.2148311016648,
            608.8044103976362,
            -6339.615516805162,
        ]
        
        assert_close(
            result[-5:], 
            expected_last_five,
            rtol=1e-1,  # KVO uses less precision in tests
            msg="KVO last 5 values mismatch"
        )
    
    def test_kvo_default_candles(self, test_data):
        """Test KVO with default parameters - mirrors check_kvo_default_candles"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        # Default params: short_period=2, long_period=5
        result = ta_indicators.kvo(high, low, close, volume)
        assert len(result) == len(close)
    
    def test_kvo_zero_period(self, test_data):
        """Test KVO fails with zero short period - mirrors check_kvo_zero_period"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        with pytest.raises(ValueError, match="Invalid"):
            ta_indicators.kvo(high, low, close, volume, short_period=0, long_period=5)
    
    def test_kvo_period_invalid(self, test_data):
        """Test KVO fails when long_period < short_period - mirrors check_kvo_period_invalid"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        with pytest.raises(ValueError, match="Invalid"):
            ta_indicators.kvo(high, low, close, volume, short_period=5, long_period=2)
    
    def test_kvo_very_small_dataset(self):
        """Test KVO fails with insufficient data - mirrors check_kvo_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Not enough valid data"):
            ta_indicators.kvo(single_point, single_point, single_point, single_point, 
                            short_period=2, long_period=5)
    
    def test_kvo_empty_input(self):
        """Test KVO fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty"):
            ta_indicators.kvo(empty, empty, empty, empty)
    
    def test_kvo_nan_handling(self, test_data):
        """Test KVO handles NaN values correctly - mirrors check_kvo_nan_handling"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        result = ta_indicators.kvo(high, low, close, volume)
        assert len(result) == len(close)
        
        # After warmup period (240), no NaN values should exist
        if len(result) > 240:
            non_nan_count = np.count_nonzero(~np.isnan(result[240:]))
            assert non_nan_count == len(result) - 240, "Found unexpected NaN values after warmup"
    
    def test_kvo_streaming(self, test_data):
        """Test KVO streaming functionality"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        # Create streaming instance with default params
        stream = ta_indicators.KvoStream()  # Uses defaults: short_period=2, long_period=5
        
        # Process each point
        stream_results = []
        for i in range(len(close)):
            result = stream.update(high[i], low[i], close[i], volume[i])
            stream_results.append(result if result is not None else np.nan)
        
        # Compare with batch calculation
        batch_results = ta_indicators.kvo(high, low, close, volume)
        
        # The streaming results should match batch results
        # Note: First value will be NaN for streaming
        assert_close(
            stream_results[1:],  # Skip first NaN
            batch_results[1:],
            rtol=1e-9,
            msg="Streaming vs batch results mismatch"
        )
    
    def test_kvo_batch(self, test_data):
        """Test KVO batch functionality"""
        high = test_data['high'][:100]  # Use smaller dataset for speed
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        volume = test_data['volume'][:100]
        
        # Test batch with single parameter set
        result = ta_indicators.kvo_batch(
            high, low, close, volume,
            short_period_range=(2, 2, 0),
            long_period_range=(5, 5, 0)
        )
        
        assert 'values' in result
        assert 'short_periods' in result
        assert 'long_periods' in result
        
        # Should match single calculation
        single_result = ta_indicators.kvo(high, low, close, volume, short_period=2, long_period=5)
        batch_values = result['values'].flatten()
        
        assert_close(batch_values, single_result, rtol=1e-10, 
                    msg="Batch result with single params differs from single calculation")
    
    def test_kvo_batch_multiple_params(self, test_data):
        """Test KVO batch with multiple parameter combinations"""
        high = test_data['high'][:50]  # Use smaller dataset for speed
        low = test_data['low'][:50]
        close = test_data['close'][:50]
        volume = test_data['volume'][:50]
        
        # Multiple parameter combinations
        result = ta_indicators.kvo_batch(
            high, low, close, volume,
            short_period_range=(2, 3, 1),    # 2, 3
            long_period_range=(5, 6, 1)      # 5, 6
        )
        
        # Should have 2 * 2 = 4 combinations
        assert result['values'].shape == (4, 50)
        assert len(result['short_periods']) == 4
        assert len(result['long_periods']) == 4
        
        # Check first combination matches single calculation
        first_row = result['values'][0, :]
        single_result = ta_indicators.kvo(high, low, close, volume, short_period=2, long_period=5)
        
        assert_close(first_row, single_result, rtol=1e-10,
                    msg="First batch row doesn't match single calculation")
    
    def test_kvo_all_nan_input(self):
        """Test KVO with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.kvo(all_nan, all_nan, all_nan, all_nan)
    
    def test_kvo_kernel_parameter(self, test_data):
        """Test KVO with different kernel parameters"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        # Test with scalar kernel
        result_scalar = ta_indicators.kvo(high, low, close, volume, kernel='scalar')
        assert len(result_scalar) == len(close)
        
        # Test with auto kernel (default)
        result_auto = ta_indicators.kvo(high, low, close, volume)
        assert len(result_auto) == len(close)
        
        # Results should be very close (within floating point precision)
        assert_close(result_scalar, result_auto, rtol=1e-14,
                    msg="Scalar vs auto kernel results differ")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
