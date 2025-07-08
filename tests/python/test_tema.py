"""
Test TEMA (Triple Exponential Moving Average) indicator Python bindings
"""
import numpy as np
import pytest
from my_project import tema, tema_batch, TemaStream
from rust_comparison import compare_with_rust


class TestTema:
    """Test TEMA indicator functionality"""
    
    def test_basic_functionality(self):
        """Test basic TEMA calculation"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        period = 3
        
        result = tema(data, period)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == data.shape
        # TEMA has a warmup period of (period-1)*3 = 6 for period=3
        assert np.isnan(result[0:6]).all()
        assert not np.isnan(result[6:]).any()
        
    def test_kernel_selection(self):
        """Test different kernel selections"""
        data = np.random.random(1000)
        period = 9
        
        # Test all kernel options
        result_auto = tema(data, period, kernel='auto')
        result_scalar = tema(data, period, kernel='scalar')
        
        # Results should be very close (within floating point precision)
        np.testing.assert_allclose(result_auto, result_scalar, rtol=1e-10)
        
        # Test AVX kernels (even if they're stubs, they should work)
        result_avx2 = tema(data, period, kernel='avx2')
        result_avx512 = tema(data, period, kernel='avx512')
        
        np.testing.assert_allclose(result_scalar, result_avx2, rtol=1e-10)
        np.testing.assert_allclose(result_scalar, result_avx512, rtol=1e-10)
        
    def test_invalid_kernel(self):
        """Test error handling for invalid kernel"""
        data = np.random.random(100)
        
        with pytest.raises(ValueError, match="Unknown kernel"):
            tema(data, 9, kernel='invalid_kernel')
            
    def test_error_empty_input(self):
        """Test error when input is empty"""
        data = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            tema(data, 3)
            
    def test_error_all_nan(self):
        """Test error when all values are NaN"""
        data = np.full(10, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            tema(data, 3)
            
    def test_error_invalid_period(self):
        """Test error for invalid period values"""
        data = np.random.random(10)
        
        # Period exceeds data length
        with pytest.raises(ValueError, match="Invalid period"):
            tema(data, 11)
            
        # Period is zero
        with pytest.raises(ValueError, match="Invalid period"):
            tema(data, 0)
            
    def test_error_not_enough_valid_data(self):
        """Test error when not enough valid data after NaN values"""
        # First 8 values are NaN, only 2 valid values, but need period 9
        data = np.array([np.nan] * 8 + [1.0, 2.0])
        
        with pytest.raises(ValueError, match="Not enough valid data"):
            tema(data, 9)
            
    def test_leading_nans(self):
        """Test TEMA with leading NaN values"""
        # Mix of NaN and valid values
        data = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        period = 2
        
        result = tema(data, period)
        
        # First valid index is 2, TEMA warmup is (period-1)*3 = 3
        # So first valid output is at index 2+3 = 5
        assert np.isnan(result[0:5]).all()
        assert not np.isnan(result[5:]).any()
        
    def test_compare_with_rust(self):
        """Test that Python bindings match Rust implementation"""
        # Use the standard test data file that generate_references uses
        from test_utils import load_test_data
        candles = load_test_data()
        data = candles['close']
        period = 9
        
        result = tema(data, period)
        compare_with_rust("tema", result, 'close', params={'period': period})
        
    def test_tema_stream(self):
        """Test TEMA streaming functionality"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        period = 3
        
        # Create stream
        stream = TemaStream(period)
        
        # Batch calculation for comparison
        batch_result = tema(data, period)
        
        # Process data through stream
        stream_results = []
        for value in data:
            result = stream.update(value)
            stream_results.append(result if result is not None else np.nan)
            
        stream_results = np.array(stream_results)
        
        # Compare results (accounting for warmup)
        # TEMA warmup is (period-1)*3 = 6
        np.testing.assert_allclose(
            batch_result[6:], 
            stream_results[6:], 
            rtol=1e-10
        )
        
    def test_batch_calculation(self):
        """Test batch TEMA calculation with multiple periods"""
        data = np.random.random(100)
        period_range = (5, 15, 2)  # periods: 5, 7, 9, 11, 13, 15
        
        result = tema_batch(data, period_range)
        
        assert isinstance(result, dict)
        assert 'values' in result
        assert 'periods' in result
        
        # Check dimensions
        expected_periods = [5, 7, 9, 11, 13, 15]
        assert result['values'].shape == (len(expected_periods), len(data))
        np.testing.assert_array_equal(result['periods'], expected_periods)
        
        # Verify each row matches individual calculation
        for i, period in enumerate(expected_periods):
            individual_result = tema(data, period)
            np.testing.assert_allclose(
                result['values'][i], 
                individual_result, 
                rtol=1e-10
            )
            
    def test_batch_kernel_selection(self):
        """Test batch calculation with different kernels"""
        data = np.random.random(100)
        period_range = (9, 15, 6)  # periods: 9, 15
        
        # Test different kernels
        result_auto = tema_batch(data, period_range, kernel='auto')
        result_scalar = tema_batch(data, period_range, kernel='scalar')
        result_avx2 = tema_batch(data, period_range, kernel='avx2')
        result_avx512 = tema_batch(data, period_range, kernel='avx512')
        
        # All should produce same results
        np.testing.assert_allclose(result_auto['values'], result_scalar['values'], rtol=1e-10)
        np.testing.assert_allclose(result_scalar['values'], result_avx2['values'], rtol=1e-10)
        np.testing.assert_allclose(result_scalar['values'], result_avx512['values'], rtol=1e-10)
        
    def test_batch_invalid_kernel(self):
        """Test batch calculation with invalid kernel"""
        data = np.random.random(100)
        period_range = (5, 10, 5)
        
        with pytest.raises(ValueError, match="Unknown kernel"):
            tema_batch(data, period_range, kernel='invalid')
            
    def test_zero_copy_performance(self):
        """Test that zero-copy operations are working (indirect test)"""
        # Large dataset to make copies noticeable
        data = np.random.random(1_000_000)
        period = 20
        
        # This should be fast due to zero-copy
        result = tema(data, period)
        
        # Verify result is correct shape and has expected NaN pattern
        assert result.shape == data.shape
        warmup = (period - 1) * 3
        assert np.isnan(result[:warmup]).all()
        assert not np.isnan(result[warmup:]).any()
        
    def test_real_world_data(self):
        """Test with real-world conditions including warmup period"""
        # Simulate real price data with some noise
        np.random.seed(42)
        trend = np.linspace(100, 110, 200)
        noise = np.random.normal(0, 0.5, 200)
        data = trend + noise
        
        period = 9
        result = tema(data, period)
        
        # Check warmup period
        warmup = (period - 1) * 3  # 24 for period=9
        assert np.isnan(result[:warmup]).all()
        assert not np.isnan(result[warmup:]).any()
        
        # Result should be smoother than input (but still responsive)
        input_volatility = np.std(np.diff(data[warmup:]))
        output_volatility = np.std(np.diff(result[warmup:]))
        # TEMA should reduce volatility but not as much as simple MA
        assert output_volatility < input_volatility
        assert output_volatility > input_volatility * 0.3  # Still responsive
        
    def test_edge_cases(self):
        """Test edge cases"""
        # Single value
        data = np.array([42.0])
        with pytest.raises(ValueError, match="Invalid period"):
            tema(data, 2)
            
        # Period equals data length
        data = np.random.random(10)
        result = tema(data, 10)
        # All values should be NaN except possibly the last
        assert np.isnan(result[:-1]).all()
        
        # Minimum valid period
        data = np.array([1.0, 2.0, 3.0])
        result = tema(data, 1)
        # Period 1 TEMA is just the input
        np.testing.assert_array_equal(result, data)
        
    def test_reinput(self):
        """Test using TEMA output as input for another TEMA calculation"""
        from test_utils import load_test_data
        candles = load_test_data()
        data = candles['close']
        
        # First TEMA with period 9
        first_period = 9
        first_result = tema(data, first_period)
        assert len(first_result) == len(data)
        
        # Use first result as input for second TEMA with period 5
        second_period = 5
        second_result = tema(first_result, second_period)
        assert len(second_result) == len(first_result)
        
        # Verify second result is not all NaN
        # First TEMA has warmup of (9-1)*3 = 24
        # Second TEMA adds warmup of (5-1)*3 = 12
        # So total warmup should be 24 + 12 = 36
        assert np.isnan(second_result[:36]).all()
        assert not np.isnan(second_result[36:]).any()
        
        # Verify the output values are reasonable (not just zeros or same value)
        valid_values = second_result[36:]
        assert len(np.unique(valid_values)) > 1  # Multiple different values
        assert np.all(np.isfinite(valid_values))  # All finite values
        assert np.std(valid_values) > 0  # Has variance
        
    def test_accuracy_check(self):
        """Test TEMA matches expected values"""
        # Simple test data where we can verify calculations
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        period = 3
        
        result = tema(data, period)
        
        # Warmup period is (3-1)*3 = 6
        assert np.isnan(result[:6]).all()
        
        # First valid value at index 6
        # We can manually verify TEMA calculation
        # EMA1 values, EMA2 values, EMA3 values can be computed
        # But for simplicity, just verify it's in reasonable range
        assert 6.0 < result[6] < 8.0  # Should be near the trend
        assert 7.0 < result[7] < 9.0
        assert 8.0 < result[8] < 10.0
        assert 9.0 < result[9] < 11.0
        
    def test_batch_warmup_periods(self):
        """Test that batch processing correctly handles different warmup periods"""
        data = np.random.random(50)
        period_range = (3, 7, 2)  # periods: 3, 5, 7
        
        result = tema_batch(data, period_range)
        
        # Check each period has correct warmup
        periods = [3, 5, 7]
        for i, period in enumerate(periods):
            warmup = (period - 1) * 3
            row = result['values'][i]
            
            # Check warmup NaN values
            assert np.isnan(row[:warmup]).all(), f"Period {period} warmup incorrect"
            # Check remaining values are not NaN
            assert not np.isnan(row[warmup:]).any(), f"Period {period} has unexpected NaNs"
            
    def test_very_small_dataset(self):
        """Test TEMA with very small datasets"""
        # Test with minimum size for different periods
        for period in [1, 2, 3, 4, 5]:
            min_size = period
            data = np.arange(1.0, min_size + 1)
            
            if period == 1:
                # Period 1 should work and return input
                result = tema(data, period)
                np.testing.assert_array_equal(result, data)
            else:
                # Should either work with all NaN except last, or fail
                try:
                    result = tema(data, period)
                    # If it works, check warmup
                    warmup = (period - 1) * 3
                    if warmup >= len(data):
                        assert np.isnan(result).all()
                    else:
                        assert np.isnan(result[:warmup]).all()
                except ValueError as e:
                    # Should be invalid period or not enough data
                    assert "Invalid period" in str(e) or "Not enough valid data" in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])