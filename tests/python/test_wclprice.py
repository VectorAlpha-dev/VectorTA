"""
Python binding tests for WCLPRICE indicator.
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


class TestWclprice:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_wclprice_slices(self):
        """Test WCLPRICE with simple slice data - mirrors check_wclprice_slices"""
        high = np.array([59230.0, 59220.0, 59077.0, 59160.0, 58717.0])
        low = np.array([59222.0, 59211.0, 59077.0, 59143.0, 58708.0])
        close = np.array([59225.0, 59210.0, 59080.0, 59150.0, 58710.0])
        
        result = ta_indicators.wclprice(high, low, close)
        expected = np.array([59225.5, 59212.75, 59078.5, 59150.75, 58711.25])
        
        assert_close(result, expected, rtol=1e-5, msg="WCLPRICE values mismatch")
    
    def test_wclprice_candles(self, test_data):
        """Test WCLPRICE with full candle data - mirrors check_wclprice_candles"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        result = ta_indicators.wclprice(high, low, close)
        assert len(result) == len(close)
        
        # Check some values are reasonable (should be between low and high)
        for i in range(len(result)):
            if not np.isnan(result[i]):
                assert low[i] <= result[i] <= high[i], f"WCLPRICE value {result[i]} at index {i} is outside range [{low[i]}, {high[i]}]"

    def test_wclprice_reference_last_five(self, test_data):
        """Ensure last 5 values match Rust reference values exactly (same CSV)."""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']

        result = ta_indicators.wclprice(high, low, close)
        expected_last_five = np.array([59225.5, 59212.75, 59078.5, 59150.75, 58711.25])
        actual_last_five = result[-5:]
        assert_close(
            actual_last_five,
            expected_last_five,
            rtol=1e-8,
            msg="WCLPRICE last 5 values mismatch vs Rust references",
        )
    
    def test_wclprice_empty_data(self):
        """Test WCLPRICE fails with empty data - mirrors check_wclprice_empty_data"""
        high = np.array([])
        low = np.array([])
        close = np.array([])
        
        with pytest.raises(ValueError, match="empty input"):
            ta_indicators.wclprice(high, low, close)
    
    def test_wclprice_all_nan(self):
        """Test WCLPRICE fails with all NaN values - mirrors check_wclprice_all_nan"""
        high = np.array([np.nan, np.nan])
        low = np.array([np.nan, np.nan])
        close = np.array([np.nan, np.nan])
        
        with pytest.raises(ValueError, match="all values are NaN"):
            ta_indicators.wclprice(high, low, close)
    
    def test_wclprice_partial_nan(self):
        """Test WCLPRICE handles partial NaN values - mirrors check_wclprice_partial_nan"""
        high = np.array([np.nan, 59000.0])
        low = np.array([np.nan, 58950.0])
        close = np.array([np.nan, 58975.0])
        
        result = ta_indicators.wclprice(high, low, close)
        
        # First value should be NaN
        assert np.isnan(result[0])
        # Second value should be calculated
        expected = (59000.0 + 58950.0 + 2.0 * 58975.0) / 4.0
        assert_close(result[1], expected, rtol=1e-8, msg="WCLPRICE calculation incorrect")
    
    def test_wclprice_formula(self):
        """Test WCLPRICE formula (high + low + 2*close) / 4"""
        # Test with simple values
        high = np.array([100.0])
        low = np.array([90.0])
        close = np.array([95.0])
        
        result = ta_indicators.wclprice(high, low, close)
        expected = (100.0 + 90.0 + 2.0 * 95.0) / 4.0  # = 95.0
        
        assert_close(result[0], expected, rtol=1e-10, msg="WCLPRICE formula incorrect")
    
    def test_wclprice_mismatched_lengths(self):
        """Test WCLPRICE handles mismatched input lengths"""
        high = np.array([100.0, 101.0, 102.0])
        low = np.array([90.0, 91.0])  # Shorter
        close = np.array([95.0, 96.0, 97.0])
        
        # Should process up to the shortest length
        result = ta_indicators.wclprice(high, low, close)
        assert len(result) == 2  # min(3, 2, 3) = 2
    
    def test_wclprice_stream(self):
        """Test WCLPRICE streaming functionality"""
        stream = ta_indicators.WclpriceStream()
        
        # Test normal update
        result = stream.update(100.0, 90.0, 95.0)
        expected = (100.0 + 90.0 + 2.0 * 95.0) / 4.0
        assert result is not None
        assert_close(result, expected, rtol=1e-10, msg="Stream update incorrect")
        
        # Test NaN handling
        result_nan = stream.update(np.nan, 90.0, 95.0)
        assert result_nan is None
        
        result_nan2 = stream.update(100.0, np.nan, 95.0)
        assert result_nan2 is None
        
        result_nan3 = stream.update(100.0, 90.0, np.nan)
        assert result_nan3 is None
    
    def test_wclprice_batch(self, test_data):
        """Test WCLPRICE batch functionality"""
        high = test_data['high'][:100]  # Use smaller dataset for batch test
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        
        batch_result = ta_indicators.wclprice_batch(high, low, close)
        
        # Check batch result structure - ALMA-compatible format
        assert 'values' in batch_result
        assert 'periods' in batch_result
        assert 'offsets' in batch_result
        assert 'sigmas' in batch_result
        
        # WCLPRICE has no parameters, so param arrays are placeholders
        # Values is a 2D array with shape (1, 100), flatten for comparison
        batch_values = batch_result['values'].flatten() if batch_result['values'].ndim > 1 else batch_result['values']
        assert len(batch_values) == 100
        
        # Compare with single calculation
        single_result = ta_indicators.wclprice(high, low, close)
        assert_close(
            batch_values,
            single_result,
            rtol=1e-10,
            msg="Batch result doesn't match single calculation"
        )
    
    def test_wclprice_with_kernel(self, test_data):
        """Test WCLPRICE with different kernel selections"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        
        # Test with default kernel (None)
        result_default = ta_indicators.wclprice(high, low, close, kernel=None)
        
        # Test with scalar kernel
        result_scalar = ta_indicators.wclprice(high, low, close, kernel="scalar")
        
        # Results should be identical
        assert_close(
            result_default,
            result_scalar,
            rtol=1e-14,
            msg="Different kernels produce different results"
        )
        
        # Test invalid kernel
        with pytest.raises(ValueError, match="Unknown kernel"):
            ta_indicators.wclprice(high, low, close, kernel="invalid_kernel")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
