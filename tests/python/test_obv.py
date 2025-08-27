"""
Python binding tests for OBV indicator.
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


class TestObv:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_obv_accuracy(self, test_data):
        """Test OBV matches expected values from Rust tests"""
        close = test_data['close']
        volume = test_data['volume']
        
        # Calculate OBV
        result = ta_indicators.obv(close, volume)
        
        assert len(result) == len(close), "Output length should match input length"
        
        # Expected values from Rust tests (last 5 values)
        expected_last_five = [
            -329661.6180239202,
            -329767.87639284023,
            -329889.94421654026,
            -329801.35075036023,
            -330218.2007503602,
        ]
        
        # Check last 5 values
        for i, expected in enumerate(expected_last_five):
            actual = result[-(5-i)]
            assert_close(actual, expected, rtol=1e-6, msg=f"OBV mismatch at tail index {i}")
    
    def test_obv_empty_data(self):
        """Test OBV with empty data"""
        close = np.array([], dtype=np.float64)
        volume = np.array([], dtype=np.float64)
        
        with pytest.raises(Exception):  # Should raise an error
            ta_indicators.obv(close, volume)
    
    def test_obv_mismatched_lengths(self):
        """Test OBV with mismatched input lengths"""
        close = np.array([1.0, 2.0, 3.0])
        volume = np.array([100.0, 200.0])
        
        with pytest.raises(Exception):  # Should raise an error
            ta_indicators.obv(close, volume)
    
    def test_obv_all_nan(self):
        """Test OBV with all NaN values"""
        close = np.array([np.nan, np.nan])
        volume = np.array([np.nan, np.nan])
        
        with pytest.raises(Exception):  # Should raise an error
            ta_indicators.obv(close, volume)
    
    def test_obv_batch(self, test_data):
        """Test OBV batch functionality"""
        close = test_data['close']
        volume = test_data['volume']
        
        # OBV has no parameters, so batch returns single result
        result = ta_indicators.obv_batch(close, volume)
        
        assert 'values' in result
        values = result['values']
        assert values.shape[0] == 1  # Single row since no parameters
        assert values.shape[1] == len(close)
        
        # Should match single calculation
        single_result = ta_indicators.obv(close, volume)
        np.testing.assert_array_almost_equal(values[0], single_result, decimal=10)
    
    def test_obv_stream(self):
        """Test OBV streaming functionality"""
        stream = ta_indicators.ObvStream()
        
        # Test data points
        test_data = [
            (100.0, 1000.0),
            (101.0, 1500.0),  # Price up, volume added
            (100.5, 2000.0),  # Price down, volume subtracted
            (100.5, 1000.0),  # Price same, no change
            (102.0, 3000.0),  # Price up, volume added
        ]
        
        results = []
        for close, volume in test_data:
            result = stream.update(close, volume)
            if result is not None:
                results.append(result)
        
        # Should have results for all inputs
        assert len(results) == len(test_data)
    
    def test_obv_kernel_selection(self, test_data):
        """Test OBV with different kernel selections"""
        close = test_data['close']
        volume = test_data['volume']
        
        # Test with different kernels
        result_auto = ta_indicators.obv(close, volume)
        result_scalar = ta_indicators.obv(close, volume, kernel='scalar')
        
        # Results should be identical regardless of kernel
        np.testing.assert_array_almost_equal(result_auto, result_scalar, decimal=10)
        
        # Test with AVX kernels if available
        try:
            result_avx2 = ta_indicators.obv(close, volume, kernel='avx2')
            np.testing.assert_array_almost_equal(result_auto, result_avx2, decimal=10)
        except ValueError:
            # AVX2 not available on this machine
            pass
        
        try:
            result_avx512 = ta_indicators.obv(close, volume, kernel='avx512')
            np.testing.assert_array_almost_equal(result_auto, result_avx512, decimal=10)
        except ValueError:
            # AVX512 not available on this machine
            pass
    
    def test_obv_nan_handling(self, test_data):
        """Test OBV handles NaN values correctly"""
        close = test_data['close']
        volume = test_data['volume']
        
        result = ta_indicators.obv(close, volume)
        assert len(result) == len(close)
        
        # Find first valid data point
        first_valid = None
        for i in range(len(close)):
            if not np.isnan(close[i]) and not np.isnan(volume[i]):
                first_valid = i
                break
        
        if first_valid is not None:
            # First valid value should be 0.0 (OBV starts at 0)
            assert result[first_valid] == 0.0, f"OBV should start at 0.0, got {result[first_valid]}"
            
            # All values before first valid should be NaN
            if first_valid > 0:
                assert np.all(np.isnan(result[:first_valid])), "Expected NaN before first valid data"
            
            # No NaN values should exist after first valid (unless input has NaN)
            for i in range(first_valid + 1, len(result)):
                if not np.isnan(close[i]) and not np.isnan(volume[i]):
                    assert not np.isnan(result[i]), f"Unexpected NaN at index {i}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
