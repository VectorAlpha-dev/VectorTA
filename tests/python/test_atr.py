"""
Python binding tests for ATR (Average True Range) indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""

import pytest
import numpy as np
import my_project
from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS


class TestAtr:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    def test_atr_accuracy(self, test_data):
        """Test that ATR values match expected results from Rust tests."""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['atr']
        
        
        result = my_project.atr(
            high, low, close,
            expected['default_params']['length']
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == close.shape
        
        
        
        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=0.0,
            atol=1e-2,
            msg="ATR last 5 values mismatch"
        )

    def test_atr_with_different_params(self, test_data):
        """Test ATR with non-default parameters."""
        high = test_data['high']
        low = test_data['low'] 
        close = test_data['close']
        
        
        length = 20
        result = my_project.atr(high, low, close, length)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == close.shape
        
        
        for i in range(length - 1):
            assert np.isnan(result[i]), f"Expected NaN at index {i} during warmup"
        
        
        for i in range(length, len(result)):
            assert not np.isnan(result[i]), f"Unexpected NaN at index {i}"

    def test_atr_empty_input(self, test_data):
        """Test ATR with empty input arrays."""
        empty = np.array([], dtype=np.float64)
        
        with pytest.raises(ValueError, match="No candles|no data"):
            my_project.atr(empty, empty, empty, 14)

    def test_atr_mismatched_lengths(self, test_data):
        """Test ATR with mismatched array lengths."""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0])  
        close = np.array([7.0, 17.0, 27.0])  
        
        with pytest.raises(ValueError, match="differing lengths|same length|Inconsistent slice lengths"):
            my_project.atr(high, low, close, 14)

    def test_atr_invalid_length(self, test_data):
        """Test ATR with invalid length parameters."""
        high = test_data['high'][:50]
        low = test_data['low'][:50]
        close = test_data['close'][:50]
        
        
        with pytest.raises(ValueError, match="Invalid length"):
            my_project.atr(high, low, close, 0)

    def test_atr_length_exceeds_data(self, test_data):
        """Test ATR when length exceeds data length."""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0, 25.0])
        close = np.array([7.0, 17.0, 27.0])
        
        with pytest.raises(ValueError, match="Not enough data|too short"):
            my_project.atr(high, low, close, 10)  

    def test_atr_with_nan_values(self, test_data):
        """Test ATR with NaN values in input."""
        high = test_data['high'].copy()
        low = test_data['low'].copy()
        close = test_data['close'].copy()
        
        
        high[5:10] = np.nan
        low[5:10] = np.nan
        close[5:10] = np.nan
        
        result = my_project.atr(high, low, close, 14)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == close.shape
        
        

    def test_atr_kernel_selection(self, test_data):
        """Test ATR with different kernel selections."""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        
        result_scalar = my_project.atr(high, low, close, 14, kernel="scalar")
        
        
        result_auto = my_project.atr(high, low, close, 14)
        
        
        assert_close(result_scalar, result_auto, rtol=1e-10)

    def test_atr_streaming(self, test_data):
        """Test ATR streaming functionality."""
        high = test_data['high']
        low = test_data['low'] 
        close = test_data['close']
        
        
        batch_result = my_project.atr(high, low, close, 14)
        
        
        stream = my_project.AtrStream(14)
        stream_results = []
        
        for h, l, c in zip(high, low, close):
            val = stream.update(h, l, c)
            stream_results.append(val if val is not None else np.nan)
        
        stream_results = np.array(stream_results)
        
        
        assert_close(stream_results, batch_result, rtol=1e-10,
                    msg="Streaming vs batch mismatch")

    def test_atr_re_input(self, test_data):
        """Test using ATR output as input to another calculation."""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        
        first_result = my_project.atr(high, low, close, 14)
        
        
        
        second_result = my_project.atr(first_result, first_result, first_result, 5)
        
        assert isinstance(second_result, np.ndarray)
        assert second_result.shape == first_result.shape

    def test_atr_constant_range(self):
        """Test ATR with constant range (high = low)."""
        length = 50
        constant_price = 100.0
        high = np.full(length, constant_price)
        low = np.full(length, constant_price) 
        close = np.full(length, constant_price)
        
        result = my_project.atr(high, low, close, 14)
        
        
        warmup = 14 - 1
        for i in range(warmup, length):
            assert abs(result[i]) < 1e-10, f"Expected 0 at index {i}, got {result[i]}"

    def test_atr_trending_market(self):
        """Test ATR in a trending market with expanding range."""
        length = 100
        
        base_prices = np.linspace(100, 200, length)
        ranges = np.linspace(1, 10, length)  
        
        high = base_prices + ranges
        low = base_prices - ranges
        close = base_prices
        
        result = my_project.atr(high, low, close, 14)
        
        
        
        last_quarter = result[-25:]
        first_quarter_valid = result[14:39]  
        
        assert np.mean(last_quarter) > np.mean(first_quarter_valid), \
            "ATR should increase in expanding range market"

    def test_atr_batch_single_params(self, test_data):
        """Test ATR batch calculation with single parameter set."""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        
        result = my_project.atr_batch(
            high, low, close,
            (14, 14, 0)  
        )
        
        assert 'values' in result
        assert 'lengths' in result
        
        values = result['values']
        assert values.shape == (1, len(close))  
        
        
        single_result = my_project.atr(high, low, close, 14)
        assert_close(values[0], single_result, rtol=1e-10)

    def test_atr_batch_parameter_sweep(self, test_data):
        """Test ATR batch calculation with parameter ranges."""
        high = test_data['high'][:100]  
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        
        result = my_project.atr_batch(
            high, low, close,
            (10, 20, 5)  
        )
        
        values = result['values']
        lengths = result['lengths']
        
        
        assert values.shape[0] == 3
        assert len(lengths) == 3
        assert list(lengths) == [10, 15, 20]

    def test_atr_error_coverage(self, test_data):
        """Test all error enum variants are covered."""
        high = test_data['high']
        low = test_data['low'] 
        close = test_data['close']
        
        
        with pytest.raises(ValueError, match="Invalid length"):
            my_project.atr(high[:50], low[:50], close[:50], 0)
            
        
        with pytest.raises(ValueError, match="differing lengths|same length|Inconsistent slice lengths"):
            my_project.atr(high[:50], low[:49], close[:50], 14)
            
        
        with pytest.raises(ValueError, match="No candles|no data"):
            my_project.atr(np.array([]), np.array([]), np.array([]), 14)
            
        
        with pytest.raises(ValueError, match="Not enough data|too short"):
            my_project.atr(high[:10], low[:10], close[:10], 20)

    def test_atr_real_world_conditions(self, test_data):
        """Test ATR under real-world conditions."""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        result = my_project.atr(high, low, close, 14)
        
        
        warmup = 14 - 1  
        
        
        assert all(np.isnan(result[:warmup]))
        
        
        valid_start = warmup
        assert not any(np.isnan(result[valid_start:]))
        
        
        assert len(result) == len(close)
        assert result.dtype == np.float64
        
        
        valid_values = result[valid_start:]
        assert all(valid_values >= 0), "ATR should be non-negative"
        
        
        assert any(valid_values > 0), "Should have some positive ATR values"

    def test_atr_single_data_point(self):
        """Test ATR with exactly length data points."""
        length = 14
        high = np.array([110.0] * length)
        low = np.array([90.0] * length)
        close = np.array([100.0] * length)
        
        result = my_project.atr(high, low, close, length)
        
        assert len(result) == length
        
        assert all(np.isnan(result[:length-1]))
        
        assert not np.isnan(result[-1])
        assert result[-1] >= 0  
