"""
Unit tests for Williams %R (WILLR) indicator Python bindings.

These tests verify that the Python bindings work correctly and produce
the same results as the Rust implementation.
"""

import pytest
import numpy as np
from typing import Dict, List, Tuple
import my_project as ta  

from test_utils import (
    load_test_data,
    assert_close,
    assert_all_nan,
    assert_no_nan,
    EXPECTED_OUTPUTS
)


class TestWillr:
    """Test suite for Williams %R indicator."""
    
    @pytest.fixture
    def test_data(self):
        """Load test data for WILLR tests."""
        return load_test_data()

    def test_willr_partial_params(self, test_data):
        """Test WILLR with default parameters."""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        
        result = ta.willr(high, low, close, 14)
        
        assert len(result) == len(close)
        assert np.isnan(result[0])  
    
    def test_willr_accuracy(self, test_data):
        """Test WILLR calculation accuracy against expected values."""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        result = ta.willr(high, low, close, 14)
        
        
        expected_last_five = [
            -58.72876391329818,
            -61.77504393673111,
            -65.93438781487991,
            -60.27950310559006,
            -65.00449236298293,
        ]
        
        actual_last_five = result[-5:]
        assert_close(actual_last_five, expected_last_five, rtol=1e-8)
    
    def test_willr_with_slice_data(self):
        """Test WILLR with small slice of data."""
        high = np.array([1.0, 2.0, 3.0, 4.0])
        low = np.array([0.5, 1.5, 2.5, 3.5])
        close = np.array([0.75, 1.75, 2.75, 3.75])
        
        result = ta.willr(high, low, close, 2)
        assert len(result) == 4
        
        
        assert np.isnan(result[0])
        
        
        
        
        
        assert abs(result[1] - (-16.666666666666668)) < 1e-8
    
    def test_willr_zero_period(self, test_data):
        """Test WILLR fails with zero period."""
        high = test_data['high'][:10]
        low = test_data['low'][:10]
        close = test_data['close'][:10]
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta.willr(high, low, close, 0)
    
    def test_willr_period_exceeds_length(self):
        """Test WILLR fails when period exceeds data length."""
        high = np.array([1.0, 2.0, 3.0])
        low = np.array([0.5, 1.5, 2.5])
        close = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta.willr(high, low, close, 10)
    
    def test_willr_all_nan(self):
        """Test WILLR fails with all NaN values."""
        high = np.full(10, np.nan)
        low = np.full(10, np.nan)
        close = np.full(10, np.nan)
        
        with pytest.raises(ValueError, match="All input values are NaN"):
            ta.willr(high, low, close, 5)
    
    def test_willr_not_enough_valid_data(self):
        """Test WILLR fails with not enough valid data (period <= len but insufficient valid span)."""
        
        high = np.array([np.nan, np.nan, 2.0])
        low = np.array([np.nan, np.nan, 1.0])
        close = np.array([np.nan, np.nan, 1.5])

        with pytest.raises(ValueError, match="Not enough valid data"):
            ta.willr(high, low, close, 3)
    
    def test_willr_mismatched_lengths(self):
        """Test WILLR fails with mismatched input lengths."""
        high = np.array([1.0, 2.0, 3.0])
        low = np.array([0.5, 1.5])  
        close = np.array([1.0, 2.0, 3.0])
        
        
        with pytest.raises((ValueError, RuntimeError)):
            ta.willr(high, low, close, 2)
    
    def test_willr_stream(self):
        """Test WILLR streaming functionality."""
        stream = ta.WillrStream(14)
        
        
        values = [
            (10.0, 8.0, 9.0),
            (11.0, 8.5, 10.0),
            (12.0, 9.0, 11.0),
            (13.0, 9.5, 12.0),
            (14.0, 10.0, 13.0),
            (15.0, 10.5, 14.0),
            (16.0, 11.0, 15.0),
            (17.0, 11.5, 16.0),
            (18.0, 12.0, 17.0),
            (19.0, 12.5, 18.0),
            (20.0, 13.0, 19.0),
            (21.0, 13.5, 20.0),
            (22.0, 14.0, 21.0),
            (23.0, 14.5, 22.0),
            (24.0, 15.0, 23.0),
        ]
        
        results = []
        for high, low, close in values:
            result = stream.update(high, low, close)
            results.append(result)
        
        
        assert all(r is None for r in results[:13])
        
        
        assert results[13] is not None
        assert results[14] is not None
    
    def test_willr_batch(self, test_data):
        """Test WILLR batch processing."""
        high = test_data['high'][:100]  
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        
        
        result = ta.willr_batch(high, low, close, (10, 20, 2))
        
        
        assert 'values' in result
        assert 'periods' in result
        
        
        expected_periods = [10, 12, 14, 16, 18, 20]
        assert len(result['periods']) == 6
        assert list(result['periods']) == expected_periods
        
        
        assert result['values'].shape == (6, 100)
        
        
        single_result = ta.willr(high, low, close, 10)
        np.testing.assert_array_almost_equal(result['values'][0], single_result, decimal=10)
    
    def test_willr_kernel_options(self, test_data):
        """Test WILLR with different kernel options."""
        high = test_data['high'][:50]
        low = test_data['low'][:50]
        close = test_data['close'][:50]
        
        
        result_auto = ta.willr(high, low, close, 14, kernel='auto')
        result_scalar = ta.willr(high, low, close, 14, kernel='scalar')
        
        
        np.testing.assert_array_almost_equal(result_auto, result_scalar, decimal=10)
        
        
        with pytest.raises(ValueError, match="Unknown kernel"):
            ta.willr(high, low, close, 14, kernel='invalid')
    
    def test_willr_edge_cases(self):
        """Test WILLR with edge cases."""
        
        high = np.array([10.0, 10.0, 10.0, 10.0])
        low = np.array([10.0, 10.0, 10.0, 10.0])
        close = np.array([10.0, 10.0, 10.0, 10.0])
        
        result = ta.willr(high, low, close, 2)
        
        assert result[1] == 0.0
        
        
        high = np.array([10.0, 20.0, 15.0])
        low = np.array([5.0, 10.0, 12.0])
        close = np.array([8.0, 20.0, 15.0])
        
        result = ta.willr(high, low, close, 2)
        
        assert abs(result[1]) < 1e-10  
        
        
        close = np.array([8.0, 5.0, 12.0])  
        result = ta.willr(high, low, close, 2)
        
        assert abs(result[1] - (-100.0)) < 1e-10
    
    def test_willr_batch_empty_range(self, test_data):
        """Test WILLR batch with empty range (single value)."""
        high = test_data['high'][:50]
        low = test_data['low'][:50]
        close = test_data['close'][:50]
        
        
        result = ta.willr_batch(high, low, close, (14, 14, 1))
        
        assert len(result['periods']) == 1
        assert result['periods'][0] == 14
        assert result['values'].shape == (1, 50)
    
    def test_willr_with_nans_in_data(self, test_data):
        """Test WILLR handles NaN values correctly."""
        high = test_data['high'][:50].copy()
        low = test_data['low'][:50].copy()
        close = test_data['close'][:50].copy()
        
        
        high[10:15] = np.nan
        low[10:15] = np.nan
        close[10:15] = np.nan
        
        result = ta.willr(high, low, close, 14)
        
        
        assert len(result) == 50
        
        
        assert np.isnan(result[10])
        assert np.isnan(result[14])


if __name__ == "__main__":
    pytest.main([__file__])
