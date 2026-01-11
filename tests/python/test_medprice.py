"""
Python binding tests for MEDPRICE indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
Tests follow the same quality and coverage standards as ALMA tests.
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


class TestMedprice:
    """Comprehensive test suite for MEDPRICE indicator Python bindings."""
    
    @pytest.fixture(scope='class')
    def test_data(self):
        """Load test data once for all tests in the class."""
        return load_test_data()
    
    
    
    
    
    def test_medprice_accuracy(self, test_data):
        """Test MEDPRICE matches expected values from Rust tests - mirrors check_medprice_accuracy"""
        high = test_data['high']
        low = test_data['low']
        
        result = ta_indicators.medprice(high, low)
        
        assert len(result) == len(high), "Output length should match input length"
        
        
        expected_last_five = [59166.0, 59244.5, 59118.0, 59146.5, 58767.5]
        
        for i, expected in enumerate(expected_last_five):
            actual = result[-(5-i)]
            assert_close(actual, expected, rtol=1e-10, 
                        msg=f"Mismatch at last_five[{i}]: expected {expected}, got {actual}")
    
    def test_medprice_formula_verification(self):
        """Test MEDPRICE formula: (high + low) / 2 with various inputs"""
        
        high = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64)
        low = np.array([50.0, 100.0, 150.0, 200.0, 250.0], dtype=np.float64)
        
        result = ta_indicators.medprice(high, low)
        
        
        for i in range(len(high)):
            expected = (high[i] + low[i]) / 2.0
            assert_close(result[i], expected, rtol=1e-15,
                        msg=f"Formula verification failed at index {i}")
        
        
        high_frac = np.array([10.5, 20.25, 30.75, 40.125], dtype=np.float64)
        low_frac = np.array([5.25, 10.125, 15.375, 20.0625], dtype=np.float64)
        
        result_frac = ta_indicators.medprice(high_frac, low_frac)
        
        for i in range(len(high_frac)):
            expected = (high_frac[i] + low_frac[i]) / 2.0
            assert_close(result_frac[i], expected, rtol=1e-15)
    
    def test_medprice_no_warmup_period(self):
        """Test that MEDPRICE has no warmup period - first valid value appears immediately"""
        high = np.array([100.0, 110.0, 120.0, 130.0], dtype=np.float64)
        low = np.array([80.0, 90.0, 100.0, 110.0], dtype=np.float64)
        
        result = ta_indicators.medprice(high, low)
        
        
        assert not np.isnan(result[0]), "MEDPRICE should not have warmup period"
        assert_close(result[0], 90.0)  
    
    
    
    
    
    def test_medprice_empty_data(self):
        """Test MEDPRICE with empty data - mirrors check_medprice_empty_data"""
        high = np.array([], dtype=np.float64)
        low = np.array([], dtype=np.float64)
        
        with pytest.raises(ValueError, match=r"(?i)empty data"):
            ta_indicators.medprice(high, low)
    
    def test_medprice_different_length(self):
        """Test MEDPRICE with different length arrays - mirrors check_medprice_different_length"""
        high = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        low = np.array([5.0, 15.0], dtype=np.float64)
        
        with pytest.raises(ValueError, match=r"(?i)different length"):
            ta_indicators.medprice(high, low)
    
    def test_medprice_all_values_nan(self):
        """Test MEDPRICE with all NaN values - mirrors check_medprice_all_values_nan"""
        high = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        low = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        
        with pytest.raises(ValueError, match=r"(?i)all values are nan"):
            ta_indicators.medprice(high, low)
    
    
    
    
    
    def test_medprice_nan_handling_basic(self):
        """Test MEDPRICE handles NaN correctly - mirrors check_medprice_nan_handling"""
        high = np.array([np.nan, 100.0, 110.0], dtype=np.float64)
        low = np.array([np.nan, 80.0, 90.0], dtype=np.float64)
        
        result = ta_indicators.medprice(high, low)
        
        assert len(result) == 3
        assert np.isnan(result[0]), "First value should be NaN"
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
        assert np.isnan(result[2]), "Last value should be NaN"
    
    def test_medprice_nan_patterns(self):
        """Test MEDPRICE with various NaN patterns"""
        
        high_alt = np.array([100.0, np.nan, 120.0, np.nan, 140.0], dtype=np.float64)
        low_alt = np.array([80.0, np.nan, 100.0, np.nan, 120.0], dtype=np.float64)
        
        result_alt = ta_indicators.medprice(high_alt, low_alt)
        
        assert_close(result_alt[0], 90.0)
        assert np.isnan(result_alt[1])
        assert_close(result_alt[2], 110.0)
        assert np.isnan(result_alt[3])
        assert_close(result_alt[4], 130.0)
        
        
        high_cluster = np.array([100.0, 110.0, np.nan, np.nan, np.nan, 150.0, 160.0], dtype=np.float64)
        low_cluster = np.array([80.0, 90.0, np.nan, np.nan, np.nan, 130.0, 140.0], dtype=np.float64)
        
        result_cluster = ta_indicators.medprice(high_cluster, low_cluster)
        
        assert_close(result_cluster[0], 90.0)
        assert_close(result_cluster[1], 100.0)
        assert np.isnan(result_cluster[2])
        assert np.isnan(result_cluster[3])
        assert np.isnan(result_cluster[4])
        assert_close(result_cluster[5], 140.0)
        assert_close(result_cluster[6], 150.0)
    
    def test_medprice_partial_nan(self):
        """Test MEDPRICE when only one input has NaN"""
        high = np.array([100.0, np.nan, 120.0, 130.0], dtype=np.float64)
        low = np.array([80.0, 90.0, 100.0, 110.0], dtype=np.float64)
        
        result = ta_indicators.medprice(high, low)
        
        assert_close(result[0], 90.0)
        assert np.isnan(result[1]), "Should be NaN when either input is NaN"
        assert_close(result[2], 110.0)
        assert_close(result[3], 120.0)
    
    
    
    
    
    def test_medprice_boundary_values(self):
        """Test MEDPRICE with extreme values"""
        
        high_large = np.array([1e10, 1e11, 1e12], dtype=np.float64)
        low_large = np.array([5e9, 5e10, 5e11], dtype=np.float64)
        
        result_large = ta_indicators.medprice(high_large, low_large)
        
        assert_close(result_large[0], 7.5e9, rtol=1e-10)
        assert_close(result_large[1], 7.5e10, rtol=1e-10)
        assert_close(result_large[2], 7.5e11, rtol=1e-10)
        
        
        high_small = np.array([1e-10, 1e-11, 1e-12], dtype=np.float64)
        low_small = np.array([5e-11, 5e-12, 5e-13], dtype=np.float64)
        
        result_small = ta_indicators.medprice(high_small, low_small)
        
        assert_close(result_small[0], 7.5e-11, rtol=1e-10)
        assert_close(result_small[1], 7.5e-12, rtol=1e-10)
        assert_close(result_small[2], 7.5e-13, rtol=1e-10)
        
        
        high_mixed = np.array([100.0, 50.0, -25.0], dtype=np.float64)
        low_mixed = np.array([-100.0, -50.0, -75.0], dtype=np.float64)
        
        result_mixed = ta_indicators.medprice(high_mixed, low_mixed)
        
        assert_close(result_mixed[0], 0.0)  
        assert_close(result_mixed[1], 0.0)  
        assert_close(result_mixed[2], -50.0)  
    
    def test_medprice_single_value(self):
        """Test MEDPRICE with single value input"""
        high = np.array([100.0], dtype=np.float64)
        low = np.array([80.0], dtype=np.float64)
        
        result = ta_indicators.medprice(high, low)
        
        assert len(result) == 1
        assert_close(result[0], 90.0)
    
    def test_medprice_identical_values(self):
        """Test MEDPRICE when high equals low"""
        high = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        low = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        
        result = ta_indicators.medprice(high, low)
        
        for i in range(len(result)):
            assert_close(result[i], 100.0)
    
    
    
    
    
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
    
    def test_medprice_streaming_with_nan(self):
        """Test MEDPRICE streaming with NaN values"""
        stream = ta_indicators.MedpriceStream()
        
        
        result1 = stream.update(100.0, 80.0)
        assert result1 == 90.0
        
        
        result2 = stream.update(np.nan, 90.0)
        assert result2 is None
        
        result3 = stream.update(110.0, np.nan)
        assert result3 is None
        
        
        result4 = stream.update(120.0, 100.0)
        assert result4 == 110.0
    
    
    
    
    
    def test_medprice_batch(self, test_data):
        """Test MEDPRICE batch functionality - mirrors check_medprice_batch"""
        high = np.array([100.0, 110.0, 120.0], dtype=np.float64)
        low = np.array([80.0, 90.0, 100.0], dtype=np.float64)
        
        
        batch = ta_indicators.medprice_batch(high, low)
        
        assert 'values' in batch, "Batch result should have 'values'"
        assert 'params' in batch, "Batch result should have 'params'"
        
        values = batch['values']
        assert values.shape == (1, 3), "Should have 1 row, 3 columns"
        
        assert_close(values[0, 0], 90.0)
        assert_close(values[0, 1], 100.0)
        assert_close(values[0, 2], 110.0)
        
        
        assert len(batch['params']) == 0 or batch['params'].size == 0
    
    def test_medprice_batch_with_dummy_range(self):
        """Test MEDPRICE batch with explicit dummy range parameter"""
        high = np.array([100.0, 110.0, 120.0], dtype=np.float64)
        low = np.array([80.0, 90.0, 100.0], dtype=np.float64)
        
        
        batch = ta_indicators.medprice_batch(high, low, dummy_range=(0, 0, 0))
        
        assert batch['values'].shape == (1, 3)
        
        
        regular = ta_indicators.medprice(high, low)
        np.testing.assert_array_almost_equal(batch['values'][0], regular, decimal=10)
    
    def test_medprice_batch_errors(self):
        """Test MEDPRICE batch error handling"""
        
        with pytest.raises(ValueError, match=r"(?i)empty"):
            ta_indicators.medprice_batch(np.array([]), np.array([]))
        
        
        with pytest.raises(ValueError, match=r"(?i)different"):
            ta_indicators.medprice_batch(
                np.array([1.0, 2.0]), 
                np.array([1.0])
            )
    
    
    
    
    
    def test_medprice_with_kernel(self, test_data):
        """Test MEDPRICE with different kernel specifications"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        
        
        result_scalar = ta_indicators.medprice(high, low, kernel="scalar")
        
        
        result_auto = ta_indicators.medprice(high, low, kernel="auto")
        
        
        np.testing.assert_array_almost_equal(result_scalar, result_auto, decimal=15)
        
        
        with pytest.raises(ValueError, match=r"(?i)invalid kernel|unknown kernel"):
            ta_indicators.medprice(high, low, kernel="invalid_kernel")
    
    def test_medprice_kernel_consistency(self, test_data):
        """Test that all kernels produce identical results for MEDPRICE"""
        high = test_data['high'][:50]
        low = test_data['low'][:50]
        
        kernels = ["scalar", "auto"]
        results = {}
        
        for kernel in kernels:
            results[kernel] = ta_indicators.medprice(high, low, kernel=kernel)
        
        
        for kernel in kernels[1:]:
            np.testing.assert_array_almost_equal(
                results[kernels[0]], 
                results[kernel], 
                decimal=15,
                err_msg=f"Kernel {kernel} produces different results than {kernels[0]}"
            )
    
    
    
    
    
    def test_medprice_large_dataset(self, test_data):
        """Test MEDPRICE with large dataset for performance"""
        
        high = test_data['high']
        low = test_data['low']
        
        result = ta_indicators.medprice(high, low)
        
        assert len(result) == len(high)
        assert not np.all(np.isnan(result)), "Should have some valid values"
        
        
        sample_idx = len(high) // 2
        if not np.isnan(high[sample_idx]) and not np.isnan(low[sample_idx]):
            expected = (high[sample_idx] + low[sample_idx]) / 2.0
            assert_close(result[sample_idx], expected, rtol=1e-10)
    
    def test_medprice_repeated_calls(self):
        """Test MEDPRICE with repeated calls for consistency"""
        high = np.random.rand(100) * 100
        low = high * 0.8  
        
        results = []
        for _ in range(10):
            result = ta_indicators.medprice(high, low)
            results.append(result)
        
        
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i],
                                         err_msg=f"Result {i} differs from first result")


if __name__ == "__main__":
    test = TestMedprice()
    data = test.test_data()
    
    
    test.test_medprice_accuracy(data)
    print("Basic medprice test passed!")
    
    
    pytest.main([__file__, '-v'])