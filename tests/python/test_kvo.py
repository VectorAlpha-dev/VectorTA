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
        
        
        result = ta_indicators.kvo(high, low, close, volume)  
        assert len(result) == len(close)
    
    def test_kvo_accuracy(self, test_data):
        """Test KVO matches expected values from Rust tests - mirrors check_kvo_accuracy"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        
        result = ta_indicators.kvo(high, low, close, volume, short_period=2, long_period=5)
        
        assert len(result) == len(close)
        
        
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
            rtol=1e-1,  
            msg="KVO last 5 values mismatch"
        )
    
    def test_kvo_default_candles(self, test_data):
        """Test KVO with default parameters - mirrors check_kvo_default_candles"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        
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
        
        
        if len(result) > 240:
            non_nan_count = np.count_nonzero(~np.isnan(result[240:]))
            assert non_nan_count == len(result) - 240, "Found unexpected NaN values after warmup"
    
    def test_kvo_streaming(self, test_data):
        """Test KVO streaming functionality"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        
        stream = ta_indicators.KvoStream(short_period=2, long_period=5)
        
        
        stream_results = []
        for i in range(len(close)):
            result = stream.update(high[i], low[i], close[i], volume[i])
            stream_results.append(result if result is not None else np.nan)
        
        
        batch_results = ta_indicators.kvo(high, low, close, volume, short_period=2, long_period=5)
        
        
        assert np.isnan(stream_results[0]), "First streaming value should be NaN during warmup"
        
        
        
        for i in range(len(stream_results)):
            if np.isnan(batch_results[i]):
                assert np.isnan(stream_results[i]), f"Expected NaN at index {i} in streaming"
            elif not np.isnan(stream_results[i]):
                assert_close(
                    stream_results[i], 
                    batch_results[i],
                    rtol=1e-9,
                    msg=f"Streaming vs batch mismatch at index {i}"
                )
    
    def test_kvo_batch(self, test_data):
        """Test KVO batch functionality"""
        high = test_data['high'][:100]  
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        volume = test_data['volume'][:100]
        
        
        result = ta_indicators.kvo_batch(
            high, low, close, volume,
            short_range=(2, 2, 0),
            long_range=(5, 5, 0)
        )
        
        assert 'values' in result
        assert 'shorts' in result  
        assert 'longs' in result   
        
        
        single_result = ta_indicators.kvo(high, low, close, volume, short_period=2, long_period=5)
        batch_values = result['values'].flatten()
        
        assert_close(batch_values, single_result, rtol=1e-10, 
                    msg="Batch result with single params differs from single calculation")
    
    def test_kvo_batch_multiple_params(self, test_data):
        """Test KVO batch with multiple parameter combinations"""
        high = test_data['high'][:50]  
        low = test_data['low'][:50]
        close = test_data['close'][:50]
        volume = test_data['volume'][:50]
        
        
        result = ta_indicators.kvo_batch(
            high, low, close, volume,
            short_range=(2, 3, 1),    
            long_range=(5, 6, 1)      
        )
        
        
        assert result['values'].shape == (4, 50)
        assert len(result['shorts']) == 4  
        assert len(result['longs']) == 4   
        
        
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
        
        
        result_scalar = ta_indicators.kvo(high, low, close, volume, kernel='scalar')
        assert len(result_scalar) == len(close)
        
        
        result_auto = ta_indicators.kvo(high, low, close, volume)
        assert len(result_auto) == len(close)
        
        
        assert_close(result_scalar, result_auto, rtol=1e-14,
                    msg="Scalar vs auto kernel results differ")
    
    def test_kvo_warmup_period(self, test_data):
        """Test KVO warmup period calculation - KVO warmup = first_valid_idx + 1"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        volume = test_data['volume'][:100]
        
        
        result = ta_indicators.kvo(high, low, close, volume, short_period=2, long_period=5)
        
        
        assert np.isnan(result[0]), "First value should be NaN during warmup"
        
        
        assert not np.isnan(result[1]), "Second value should be valid after warmup"
        
        
        high_nan = high.copy()
        low_nan = low.copy()
        close_nan = close.copy()
        volume_nan = volume.copy()
        
        
        for i in range(5):
            high_nan[i] = np.nan
            low_nan[i] = np.nan
            close_nan[i] = np.nan
            volume_nan[i] = np.nan
        
        result_nan = ta_indicators.kvo(high_nan, low_nan, close_nan, volume_nan, 
                                       short_period=2, long_period=5)
        
        
        for i in range(6):
            assert np.isnan(result_nan[i]), f"Expected NaN at index {i} during warmup with NaN input"
        
        
        assert not np.isnan(result_nan[6]), "Expected valid value at index 6 after warmup"
    
    def test_kvo_batch_edge_cases(self):
        """Test KVO batch with edge cases"""
        
        test_size = 20
        high = np.random.randn(test_size) + 100
        low = high - np.abs(np.random.randn(test_size))
        close = (high + low) / 2 + np.random.randn(test_size) * 0.1
        volume = np.abs(np.random.randn(test_size)) * 1000
        
        
        result = ta_indicators.kvo_batch(
            high, low, close, volume,
            short_range=(2, 2, 0),
            long_range=(5, 5, 0)
        )
        assert result['values'].shape == (1, test_size)
        assert len(result['shorts']) == 1
        assert result['shorts'][0] == 2
        assert result['longs'][0] == 5
        
        
        result = ta_indicators.kvo_batch(
            high, low, close, volume,
            short_range=(2, 3, 10),  
            long_range=(5, 6, 10)
        )
        
        assert result['values'].shape == (1, test_size)
        assert result['shorts'][0] == 2
        assert result['longs'][0] == 5
        
        
        result = ta_indicators.kvo_batch(
            high, low, close, volume,
            short_range=(2, 4, 1),  
            long_range=(5, 7, 2)     
        )
        
        assert result['values'].shape == (6, test_size)
        assert len(result['shorts']) == 6
        assert len(result['longs']) == 6
        
        
        expected_shorts = [2, 2, 3, 3, 4, 4]
        expected_longs = [5, 7, 5, 7, 5, 7]
        np.testing.assert_array_equal(result['shorts'], expected_shorts)
        np.testing.assert_array_equal(result['longs'], expected_longs)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
