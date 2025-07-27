"""
Python binding tests for MEDPRICE indicator.
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


class TestMedprice:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_medprice_accuracy(self, test_data):
        """Test MEDPRICE matches expected values from Rust tests - mirrors check_medprice_accuracy"""
        high = test_data['high']
        low = test_data['low']
        
        result = ta_indicators.medprice(high, low)
        
        assert len(result) == len(high)
        
        # Check last 5 values match expected - from Rust test
        expected_last_five = [59166.0, 59244.5, 59118.0, 59146.5, 58767.5]
        
        for i, expected in enumerate(expected_last_five):
            actual = result[-(5-i)]
            assert_close(actual, expected, rtol=1e-3)
    
    def test_medprice_empty_data(self):
        """Test MEDPRICE with empty data - mirrors check_medprice_empty_data"""
        high = np.array([], dtype=np.float64)
        low = np.array([], dtype=np.float64)
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.medprice(high, low)
    
    def test_medprice_different_length(self):
        """Test MEDPRICE with different length arrays - mirrors check_medprice_different_length"""
        high = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        low = np.array([5.0, 15.0], dtype=np.float64)
        
        with pytest.raises(ValueError, match="Different lengths"):
            ta_indicators.medprice(high, low)
    
    def test_medprice_all_values_nan(self):
        """Test MEDPRICE with all NaN values - mirrors check_medprice_all_values_nan"""
        high = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        low = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.medprice(high, low)
    
    def test_medprice_nan_handling(self):
        """Test MEDPRICE handles NaN correctly - mirrors check_medprice_nan_handling"""
        high = np.array([np.nan, 100.0, 110.0], dtype=np.float64)
        low = np.array([np.nan, 80.0, 90.0], dtype=np.float64)
        
        result = ta_indicators.medprice(high, low)
        
        assert len(result) == 3
        assert np.isnan(result[0])
        assert_close(result[1], 90.0)
        assert_close(result[2], 100.0)
    
    def test_medprice_late_nan_handling(self):
        """Test MEDPRICE handles late NaN correctly - mirrors check_medprice_late_nan_handling"""
        high = np.array([100.0, 110.0, np.nan], dtype=np.float64)
        low = np.array([80.0, 90.0, np.nan], dtype=np.float64)
        
        result = ta_indicators.medprice(high, low)
        
        assert len(result) == 3
        assert_close(result[0], 90.0)
        assert_close(result[1], 100.0)
        assert np.isnan(result[2])
    
    def test_medprice_streaming(self):
        """Test MEDPRICE streaming functionality - mirrors check_medprice_streaming"""
        high = [100.0, 110.0, 120.0]
        low = [80.0, 90.0, 100.0]
        
        stream = ta_indicators.MedpriceStream()
        
        values = []
        for h, l in zip(high, low):
            values.append(stream.update(h, l))
        
        assert values[0] == 90.0
        assert values[1] == 100.0
        assert values[2] == 110.0
    
    def test_medprice_batch(self, test_data):
        """Test MEDPRICE batch functionality - mirrors check_medprice_batch"""
        high = np.array([100.0, 110.0, 120.0], dtype=np.float64)
        low = np.array([80.0, 90.0, 100.0], dtype=np.float64)
        
        # Since medprice has no parameters, we use None for dummy_range
        batch = ta_indicators.medprice_batch(high, low)
        
        assert 'values' in batch
        assert 'params' in batch
        
        values = batch['values']
        assert values.shape == (1, 3)
        
        assert_close(values[0, 0], 90.0)
        assert_close(values[0, 1], 100.0)
        assert_close(values[0, 2], 110.0)
    
    def test_medprice_with_kernel(self, test_data):
        """Test MEDPRICE with different kernel specifications"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        
        # Test with scalar kernel
        result_scalar = ta_indicators.medprice(high, low, kernel="scalar")
        
        # Test with auto kernel
        result_auto = ta_indicators.medprice(high, low, kernel="auto")
        
        # Results should be the same
        np.testing.assert_array_almost_equal(result_scalar, result_auto, decimal=10)
    
    def test_medprice_formula(self):
        """Test MEDPRICE formula: (high + low) / 2"""
        high = np.array([100.0, 110.0, 120.0], dtype=np.float64)
        low = np.array([80.0, 90.0, 100.0], dtype=np.float64)
        
        result = ta_indicators.medprice(high, low)
        
        # Verify the formula
        for i in range(len(high)):
            expected = (high[i] + low[i]) / 2.0
            assert_close(result[i], expected)


if __name__ == "__main__":
    test = TestMedprice()
    data = test.test_data()
    
    # Run a simple test
    test.test_medprice_accuracy(data)
    print("Basic medprice test passed!")