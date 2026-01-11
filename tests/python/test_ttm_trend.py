"""
Python binding tests for TTM Trend indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

try:
    import my_project as ta
except ImportError:
    
    try:
        import my_project as ta
    except ImportError:
        pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close


class TestTtmTrend:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_ttm_trend_partial_params(self, test_data):
        """Test TTM Trend with partial parameters - mirrors check_ttm_partial_params"""
        
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        hl2 = (high + low) / 2.0
        
        
        result = ta.ttm_trend(hl2, close, period=5)
        assert len(result) == len(close)
        assert result.dtype == np.float64  
    
    def test_ttm_trend_accuracy(self, test_data):
        """Test TTM Trend matches expected values from Rust tests - mirrors check_ttm_accuracy"""
        
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        hl2 = (high + low) / 2.0
        
        result = ta.ttm_trend(hl2, close, period=5)
        
        assert len(result) == len(close)
        
        
        
        expected_last_five = [True, False, False, False, False]
        actual_last_five = [v == 1.0 for v in result[-5:]]
        assert actual_last_five == expected_last_five, f"Expected {expected_last_five}, got {actual_last_five}"
    
    def test_ttm_trend_zero_period(self):
        """Test TTM Trend fails with zero period - mirrors check_ttm_zero_period"""
        src = np.array([10.0, 20.0, 30.0])
        close = np.array([12.0, 22.0, 32.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta.ttm_trend(src, close, period=0)
    
    def test_ttm_trend_period_exceeds_length(self):
        """Test TTM Trend fails when period exceeds data length - mirrors check_ttm_period_exceeds_length"""
        src = np.array([1.0, 2.0, 3.0])
        close = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta.ttm_trend(src, close, period=10)
    
    def test_ttm_trend_very_small_dataset(self):
        """Test TTM Trend fails with insufficient data - mirrors check_ttm_very_small_dataset"""
        src = np.array([42.0])
        close = np.array([43.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta.ttm_trend(src, close, period=5)
    
    def test_ttm_trend_all_nan(self):
        """Test TTM Trend fails with all NaN values - mirrors check_ttm_all_nan"""
        src = np.array([np.nan, np.nan, np.nan])
        close = np.array([np.nan, np.nan, np.nan])
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta.ttm_trend(src, close, period=5)
    
    def test_ttm_trend_empty_input(self):
        """Test TTM Trend fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError):
            ta.ttm_trend(empty, empty, period=5)
    
    def test_ttm_trend_mismatched_lengths(self):
        """Test TTM Trend handles mismatched input lengths"""
        src = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        close = np.array([1.0, 2.0, 3.0])  
        
        
        result = ta.ttm_trend(src, close, period=2)
        assert len(result) == len(close)
    
    def test_ttm_trend_streaming(self, test_data):
        """Test TTM Trend streaming functionality - mirrors check_ttm_streaming"""
        
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        hl2 = (high + low) / 2.0
        
        period = 5
        
        
        batch_result = ta.ttm_trend(hl2, close, period=period)
        
        
        stream = ta.TtmTrendStream(period=period)
        stream_values = []
        
        for i in range(len(close)):
            value = stream.update(hl2[i], close[i])
            if value is None:
                stream_values.append(False)  
            else:
                stream_values.append(value)
        
        
        batch_bool = [False if np.isnan(v) else v == 1.0 for v in batch_result]
        assert stream_values == batch_bool
    
    def test_ttm_trend_batch_single_period(self, test_data):
        """Test TTM Trend batch with single period"""
        
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        hl2 = (high + low) / 2.0
        
        
        batch_result = ta.ttm_trend_batch(
            hl2,
            close,
            period_range=(5, 5, 0)  
        )
        
        
        single_result = ta.ttm_trend(hl2, close, period=5)
        
        assert batch_result['values'].shape[0] == 1  
        assert batch_result['values'].shape[1] == len(close)  
        
        assert np.allclose(batch_result['values'][0], single_result, equal_nan=True)
        assert batch_result['periods'].tolist() == [5]
    
    def test_ttm_trend_batch_multiple_periods(self, test_data):
        """Test TTM Trend batch with multiple periods"""
        
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        hl2 = (high + low) / 2.0
        
        
        batch_result = ta.ttm_trend_batch(
            hl2,
            close,
            period_range=(5, 15, 5)
        )
        
        
        assert batch_result['values'].shape[0] == 3
        assert batch_result['values'].shape[1] == 100
        assert batch_result['periods'].tolist() == [5, 10, 15]
        
        
        periods = [5, 10, 15]
        for i, period in enumerate(periods):
            single_result = ta.ttm_trend(hl2, close, period=period)
            
            assert np.allclose(batch_result['values'][i], single_result, equal_nan=True)
    
    def test_ttm_trend_batch_invalid_range(self):
        """Test TTM Trend batch with invalid range"""
        src = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        
        with pytest.raises(ValueError):
            ta.ttm_trend_batch(
                src,
                close,
                period_range=(10, 20, 5)  
            )
    
    def test_ttm_trend_with_kernel(self, test_data):
        """Test TTM Trend with different kernel options"""
        
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        hl2 = (high + low) / 2.0
        
        
        result_auto = ta.ttm_trend(hl2, close, period=5, kernel=None)
        result_scalar = ta.ttm_trend(hl2, close, period=5, kernel="scalar")
        
        
        
        assert np.allclose(result_auto, result_scalar, equal_nan=True)
    
    def test_ttm_trend_batch_with_kernel(self, test_data):
        """Test TTM Trend batch with kernel option"""
        
        high = test_data['high'][:50]
        low = test_data['low'][:50]
        close = test_data['close'][:50]
        hl2 = (high + low) / 2.0
        
        
        result_auto = ta.ttm_trend_batch(
            hl2,
            close,
            period_range=(5, 10, 5),
            kernel=None
        )
        
        result_scalar = ta.ttm_trend_batch(
            hl2,
            close,
            period_range=(5, 10, 5),
            kernel="scalar"
        )
        
        
        
        assert np.allclose(result_auto['values'], result_scalar['values'], equal_nan=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])