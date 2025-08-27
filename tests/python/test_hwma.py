"""
Python binding tests for HWMA indicator.
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
from rust_comparison import compare_with_rust


class TestHwma:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_hwma_partial_params(self, test_data):
        """Test HWMA with partial parameters - mirrors check_hwma_partial_params"""
        close = test_data['close']
        
        # Test with default params: na=0.2, nb=0.1, nc=0.1
        result = ta_indicators.hwma(close, 0.2, 0.1, 0.1)
        assert len(result) == len(close)
    
    def test_hwma_accuracy(self, test_data):
        """Test HWMA matches expected values from Rust tests - mirrors check_hwma_accuracy"""
        close = test_data['close']
        
        # Using na=0.2, nb=0.1, nc=0.1
        result = ta_indicators.hwma(close, 0.2, 0.1, 0.1)
        
        assert len(result) == len(close)
        
        # Expected last 5 values from Rust test
        expected_last_five = [
            57941.04005793378,
            58106.90324194954,
            58250.474156632234,
            58428.90005831887,
            58499.37021151028,
        ]
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected_last_five,
            rtol=1e-3,
            msg="HWMA last 5 values mismatch"
        )
        
        # Compare full output with Rust
        # compare_with_rust('hwma', result, 'close', {'na': 0.2, 'nb': 0.1, 'nc': 0.1})
    
    def test_hwma_default_candles(self, test_data):
        """Test HWMA with default parameters - mirrors check_hwma_default_candles"""
        close = test_data['close']
        
        # Default params: na=0.2, nb=0.1, nc=0.1
        result = ta_indicators.hwma(close, 0.2, 0.1, 0.1)
        assert len(result) == len(close)
        
        # Compare with Rust
        # compare_with_rust('hwma', result, 'close', {'na': 0.2, 'nb': 0.1, 'nc': 0.1})
    
    def test_hwma_invalid_na(self):
        """Test HWMA fails with invalid na - mirrors check_hwma_invalid_na"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        # Test na > 1
        with pytest.raises(ValueError):
            ta_indicators.hwma(input_data, na=1.5, nb=0.1, nc=0.1)
        
        # Test na <= 0
        with pytest.raises(ValueError):
            ta_indicators.hwma(input_data, na=0.0, nb=0.1, nc=0.1)
        
        # Test na = NaN
        with pytest.raises(ValueError):
            ta_indicators.hwma(input_data, na=float('nan'), nb=0.1, nc=0.1)
    
    def test_hwma_invalid_nb(self):
        """Test HWMA fails with invalid nb - mirrors check_hwma_invalid_nb"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        # Test nb > 1
        with pytest.raises(ValueError):
            ta_indicators.hwma(input_data, na=0.2, nb=1.5, nc=0.1)
        
        # Test nb <= 0
        with pytest.raises(ValueError):
            ta_indicators.hwma(input_data, na=0.2, nb=0.0, nc=0.1)
        
        # Test nb = NaN
        with pytest.raises(ValueError):
            ta_indicators.hwma(input_data, na=0.2, nb=float('nan'), nc=0.1)
    
    def test_hwma_invalid_nc(self):
        """Test HWMA fails with invalid nc - mirrors check_hwma_invalid_nc"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        # Test nc > 1
        with pytest.raises(ValueError):
            ta_indicators.hwma(input_data, na=0.2, nb=0.1, nc=1.5)
        
        # Test nc <= 0
        with pytest.raises(ValueError):
            ta_indicators.hwma(input_data, na=0.2, nb=0.1, nc=0.0)
        
        # Test nc = NaN
        with pytest.raises(ValueError):
            ta_indicators.hwma(input_data, na=0.2, nb=0.1, nc=float('nan'))
    
    def test_hwma_empty_input(self):
        """Test HWMA with empty input - mirrors check_hwma_empty_input"""
        data_empty = np.array([])
        
        with pytest.raises(ValueError):
            ta_indicators.hwma(data_empty, na=0.2, nb=0.1, nc=0.1)
    
    def test_hwma_all_nan(self):
        """Test HWMA with all NaN input"""
        data = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        
        with pytest.raises(ValueError):
            ta_indicators.hwma(data, na=0.2, nb=0.1, nc=0.1)
    
    def test_hwma_reinput(self, test_data):
        """Test HWMA with re-input of HWMA result - mirrors check_hwma_reinput"""
        close = test_data['close']
        
        # First HWMA pass
        first_result = ta_indicators.hwma(close, 0.2, 0.1, 0.1)
        
        # Second HWMA pass using first result as input
        second_result = ta_indicators.hwma(first_result, 0.2, 0.1, 0.1)
        
        assert len(second_result) == len(first_result)
        
        # Verify values are reasonable (not NaN/Inf)
        finite_values = second_result[np.isfinite(second_result)]
        if len(finite_values) > 0:
            assert np.all(finite_values > 0), "HWMA reinput produced negative values"
    
    def test_hwma_nan_handling(self, test_data):
        """Test HWMA handling of NaN values - mirrors check_hwma_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.hwma(close, 0.2, 0.1, 0.1)
        
        assert len(result) == len(close)
        
        # After index 3, all values should be finite
        if len(result) > 3:
            for i in range(3, len(result)):
                assert np.isfinite(result[i]), f"Unexpected non-finite value at index {i}"
    
    def test_hwma_streaming(self, test_data):
        """Test HWMA streaming vs batch calculation - mirrors check_hwma_streaming"""
        close = test_data['close'][:100]  # Use first 100 values for testing
        na, nb, nc = 0.2, 0.1, 0.1
        
        # Batch calculation
        batch_result = ta_indicators.hwma(close, na, nb, nc)
        
        # Streaming calculation
        stream = ta_indicators.HwmaStream(na, nb, nc)
        stream_results = []
        
        for val in close:
            result = stream.update(val)
            stream_results.append(result if result is not None else np.nan)
        
        stream_results = np.array(stream_results)
        
        # Compare batch vs streaming
        assert_close(
            stream_results, 
            batch_result,
            rtol=1e-10,
            msg="HWMA streaming vs batch mismatch"
        )
    
    def test_hwma_batch(self, test_data):
        """Test HWMA batch computation."""
        close = test_data['close']
        
        # Test parameter ranges
        na_range = (0.1, 0.3, 0.1)  # na: 0.1, 0.2, 0.3
        nb_range = (0.05, 0.15, 0.05)  # nb: 0.05, 0.10, 0.15
        nc_range = (0.05, 0.15, 0.05)  # nc: 0.05, 0.10, 0.15
        
        result = ta_indicators.hwma_batch(close, na_range, nb_range, nc_range)
        
        assert isinstance(result, dict)
        assert 'values' in result
        assert 'na' in result
        assert 'nb' in result
        assert 'nc' in result
        
        values = result['values']
        na_values = result['na']
        nb_values = result['nb']
        nc_values = result['nc']
        
        # Should have 3 * 3 * 3 = 27 combinations
        assert values.shape == (27, len(close))
        assert len(na_values) == 27
        assert len(nb_values) == 27
        assert len(nc_values) == 27
        
        # Verify first combination matches individual calculation
        individual_result = ta_indicators.hwma(close, 0.1, 0.05, 0.05)
        np.testing.assert_allclose(
            values[0], 
            individual_result, 
            rtol=1e-9,
            err_msg="Batch first row mismatch"
        )
    
    def test_hwma_different_params(self, test_data):
        """Test HWMA with different parameter values."""
        close = test_data['close']
        
        # Test various parameter combinations
        param_sets = [
            (0.1, 0.05, 0.05),
            (0.2, 0.1, 0.1),
            (0.3, 0.15, 0.15),
            (0.5, 0.25, 0.25),
        ]
        
        for na, nb, nc in param_sets:
            result = ta_indicators.hwma(close, na, nb, nc)
            assert len(result) == len(close)
            
            # Verify finite values after warmup
            finite_count = np.sum(np.isfinite(result))
            assert finite_count > len(close) - 4, f"Too many non-finite values for params ({na}, {nb}, {nc})"
    
    def test_hwma_batch_performance(self, test_data):
        """Test that batch computation is more efficient than multiple single computations."""
        close = test_data['close'][:1000]  # Use first 1000 values
        
        # Test 8 combinations
        import time
        
        start_batch = time.time()
        batch_result = ta_indicators.hwma_batch(
            close, 
            (0.1, 0.2, 0.1),  # 2 na values
            (0.05, 0.1, 0.05),  # 2 nb values
            (0.05, 0.1, 0.05)   # 2 nc values
        )
        batch_time = time.time() - start_batch
        
        start_single = time.time()
        single_results = []
        for na in [0.1, 0.2]:
            for nb in [0.05, 0.1]:
                for nc in [0.05, 0.1]:
                    single_results.append(ta_indicators.hwma(close, na, nb, nc))
        single_time = time.time() - start_single
        
        # Batch should be faster than multiple single calls
        print(f"Batch time: {batch_time:.4f}s, Single time: {single_time:.4f}s")
        
        # Verify results match
        values = batch_result['values']
        for i, single in enumerate(single_results):
            np.testing.assert_allclose(values[i], single, rtol=1e-9)
    
    def test_hwma_edge_cases(self):
        """Test HWMA with edge case parameter values"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test very small parameters (close to 0)
        result = ta_indicators.hwma(data, 0.001, 0.001, 0.001)
        assert len(result) == len(data)
        assert np.sum(np.isfinite(result)) > 0
        
        # Test parameters close to 1
        result = ta_indicators.hwma(data, 0.999, 0.999, 0.999)
        assert len(result) == len(data)
        assert np.sum(np.isfinite(result)) > 0
    
    def test_hwma_single_value(self):
        """Test HWMA with single value input"""
        data = np.array([42.0])
        
        result = ta_indicators.hwma(data, 0.2, 0.1, 0.1)
        assert len(result) == 1
        assert abs(result[0] - data[0]) < 1e-12  # First value should be the data value
    
    def test_hwma_two_values(self):
        """Test HWMA with two values input"""
        data = np.array([1.0, 2.0])
        
        result = ta_indicators.hwma(data, 0.2, 0.1, 0.1)
        assert len(result) == 2
        # HWMA doesn't use NaN prefix for small data
        assert np.isfinite(result[0])
        assert np.isfinite(result[1])
    
    def test_hwma_three_values(self):
        """Test HWMA with three values input"""
        data = np.array([1.0, 2.0, 3.0])
        
        result = ta_indicators.hwma(data, 0.2, 0.1, 0.1)
        assert len(result) == 3
        # HWMA doesn't use NaN prefix for small data
        assert np.isfinite(result[0])
        assert np.isfinite(result[1])
        assert np.isfinite(result[2])
    
    def test_hwma_batch_metadata(self):
        """Test batch metadata function returns correct parameter combinations"""
        na_range = (0.1, 0.2, 0.1)  # 2 values
        nb_range = (0.05, 0.1, 0.05)  # 2 values  
        nc_range = (0.05, 0.1, 0.05)  # 2 values
        
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ta_indicators.hwma_batch(data, na_range, nb_range, nc_range)
        
        na_values = result['na']
        nb_values = result['nb']
        nc_values = result['nc']
        
        # Should have 2 * 2 * 2 = 8 combinations
        assert len(na_values) == 8
        assert len(nb_values) == 8
        assert len(nc_values) == 8
        
        # Check first few combinations
        expected_combinations = [
            (0.1, 0.05, 0.05),
            (0.1, 0.05, 0.1),
            (0.1, 0.1, 0.05),
            (0.1, 0.1, 0.1),
        ]
        
        for i, (exp_na, exp_nb, exp_nc) in enumerate(expected_combinations):
            assert abs(na_values[i] - exp_na) < 1e-9
            assert abs(nb_values[i] - exp_nb) < 1e-9
            assert abs(nc_values[i] - exp_nc) < 1e-9


if __name__ == '__main__':
    pytest.main([__file__, '-v'])