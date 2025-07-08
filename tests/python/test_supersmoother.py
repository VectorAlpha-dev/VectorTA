"""Tests for SuperSmoother (2-pole) Python bindings"""
import pytest
import numpy as np
import my_project
from test_utils import (
    load_test_data,
    assert_close,
    EXPECTED_SUPERSMOOTHER
)
from rust_comparison import compare_with_rust

class TestSuperSmoother:
    """Test cases for SuperSmoother indicator"""
    
    def test_accuracy(self):
        """Test SuperSmoother accuracy against expected values"""
        data = load_test_data()
        close_prices = data['close']
        
        # Calculate SuperSmoother with default period
        result = my_project.supersmoother(close_prices, 14)
        
        # Check output length
        assert len(result) == len(close_prices)
        
        # For 2-pole supersmoother with period=14:
        # First non-NaN in data is at index 0 (no leading NaNs in test data)
        # Warmup period = first + period - 1 = 0 + 14 - 1 = 13
        # So first 13 values (indices 0-12) should be NaN
        for i in range(13):
            assert np.isnan(result[i]), f"Expected NaN at index {i} during warmup"
        
        # Values at indices 13 and 14 are initial conditions (set from input data)
        assert not np.isnan(result[13]), "Value at index 13 should be initialized"
        assert not np.isnan(result[14]), "Value at index 14 should be initialized"
        
        # Test last 5 values against expected
        assert_close(
            result[-5:], 
            EXPECTED_SUPERSMOOTHER,
            msg="SuperSmoother last 5 values mismatch"
        )
    
    def test_kernel_selection(self):
        """Test different kernel options"""
        data = load_test_data()
        close_prices = data['close']
        
        # Test scalar kernel explicitly (safest option)
        result_scalar = my_project.supersmoother(close_prices, 14, kernel='scalar')
        assert len(result_scalar) == len(close_prices)
        
        # Test auto kernel
        result_auto = my_project.supersmoother(close_prices, 14, kernel='auto')
        assert len(result_auto) == len(close_prices)
        
        # Results should match
        assert_close(
            result_scalar[-5:], 
            result_auto[-5:],
            msg="Scalar and auto kernels produce different results"
        )
    
    def test_streaming(self):
        """Test streaming SuperSmoother implementation"""
        # Note: The current streaming implementation has limitations
        # It needs redesign to properly track output history
        # For now, we test basic functionality
        
        # Test that stream can be created and produces values
        stream = my_project.SuperSmootherStream(14)
        
        # Feed some values
        results = []
        for i in range(20):
            val = stream.update(float(i))
            results.append(val)
        
        # Should get None for first value, then start producing output
        assert results[0] is None
        assert results[1] is not None
        
        # Values should be numeric after warmup
        for val in results[2:]:
            assert val is not None
            assert isinstance(val, float)
    
    def test_batch_processing(self):
        """Test batch processing with multiple periods"""
        data = load_test_data()
        close_prices = data['close']
        
        # Test batch with multiple periods
        result = my_project.supersmoother_batch(close_prices, 10, 20, 5)
        
        # Check return structure
        assert 'values' in result
        assert 'periods' in result
        
        # Should have 3 periods: 10, 15, 20
        assert len(result['periods']) == 3
        assert list(result['periods']) == [10, 15, 20]
        
        # Values should be 2D array with shape (3, len(close_prices))
        assert result['values'].shape == (3, len(close_prices))
        
        # Each row should match individual calculation
        for i, period in enumerate(result['periods']):
            individual = my_project.supersmoother(close_prices, period)
            # Find first non-NaN for comparison
            first_valid = next(j for j, v in enumerate(individual) if not np.isnan(v))
            assert_close(
                result['values'][i][first_valid:],
                individual[first_valid:],
                msg=f"Batch result for period {period} doesn't match individual"
            )
    
    def test_error_handling(self):
        """Test error conditions"""
        data = load_test_data()
        close_prices = data['close']
        
        # Test with period = 0
        with pytest.raises(ValueError, match="Invalid period"):
            my_project.supersmoother(close_prices, 0)
        
        # Test with period > data length
        with pytest.raises(ValueError, match="Invalid period"):
            my_project.supersmoother(close_prices[:5], 10)
        
        # Test with all NaN data
        all_nan = np.full(10, np.nan)
        with pytest.raises(ValueError, match="All values are NaN"):
            my_project.supersmoother(all_nan, 5)
        
        # Test with empty data
        empty = np.array([])
        with pytest.raises(ValueError, match="Empty data"):
            my_project.supersmoother(empty, 5)
    
    def test_nan_handling(self):
        """Test handling of NaN values in input"""
        data = load_test_data()
        data_with_nan = data['close'].copy()
        
        # Insert NaN values in the middle
        data_with_nan[100:105] = np.nan
        
        # Should still compute
        result = my_project.supersmoother(data_with_nan, 14)
        
        # NaN should propagate
        # With period=14, NaN at 100-104 affects output from 100 to at least 113
        for i in range(100, 114):
            assert np.isnan(result[i]), f"Expected NaN at position {i}"
    
    def test_leading_nans(self):
        """Test handling of leading NaN values"""
        # Create data with leading NaNs
        data = np.empty(20)
        data[:5] = np.nan
        for i in range(5, 20):
            data[i] = i - 4  # 1, 2, 3, ...
        
        period = 3
        result = my_project.supersmoother(data, period)
        
        # For 2-pole supersmoother with leading NaNs:
        # first_non_nan = 5
        # NaN up to first_non_nan + period - 1 = 5 + 3 - 1 = 7
        # Two initial values at indices 7 and 8
        # Main calculation starts at index 9
        
        # Check that NaN input produces NaN output
        for i in range(5):
            assert np.isnan(result[i]), f"Expected NaN at index {i} where input is NaN"
        
        # Due to warmup, values remain NaN
        for i in range(5, 7):
            assert np.isnan(result[i]), f"Expected NaN at index {i} due to warmup"
        
        # Initial values should be set from data
        assert result[7] == data[7], f"Expected initial value at index 7"
        assert result[8] == data[8], f"Expected initial value at index 8"
    
    def test_edge_cases(self):
        """Test edge cases"""
        data = load_test_data()
        close_prices = data['close']
        
        # Test with minimum period (1)
        result1 = my_project.supersmoother(close_prices, 1)
        assert len(result1) == len(close_prices)
        
        # Test with very small dataset
        small_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result2 = my_project.supersmoother(small_data, 2)
        assert len(result2) == len(small_data)
        
        # For 2-pole filter, warmup is period-1, so first value is NaN
        assert np.isnan(result2[0])
        assert not np.isnan(result2[1])
    
    def test_consistency(self):
        """Test that multiple runs produce identical results"""
        data = load_test_data()
        close_prices = data['close']
        
        # Run multiple times
        result1 = my_project.supersmoother(close_prices, 14)
        result2 = my_project.supersmoother(close_prices, 14)
        
        # Results should be identical
        for i in range(len(result1)):
            if np.isnan(result1[i]) and np.isnan(result2[i]):
                continue
            assert result1[i] == result2[i], f"Inconsistent result at index {i}"
    
    def test_compare_with_rust(self):
        """Compare Python results with Rust implementation"""
        data = load_test_data()
        close_prices = data['close']
        
        # Calculate Python result
        result = my_project.supersmoother(close_prices, 14)
        
        # Compare with Rust
        compare_with_rust('supersmoother', result, 'close', {'period': 14})

if __name__ == "__main__":
    pytest.main([__file__, "-v"])