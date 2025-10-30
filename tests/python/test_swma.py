"""
Test SWMA (Symmetric Weighted Moving Average) indicator Python bindings
"""
import numpy as np
import pytest
from my_project import swma, swma_batch, SwmaStream
from rust_comparison import compare_with_rust
from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS


class TestSwma:
    """Test SWMA indicator functionality"""
    
    def test_basic_functionality(self):
        """Test basic SWMA calculation"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        period = 5
        
        result = swma(data, period)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == data.shape
        # SWMA has a warmup period of period-1
        assert np.isnan(result[0:period-1]).all()
        assert not np.isnan(result[period-1:]).any()
        
    def test_swma_empty_input(self):
        """Test SWMA fails with empty input - mirrors check_swma_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            swma(empty, 5)
    
    def test_swma_accuracy(self):
        """Test SWMA matches expected values from Rust tests - mirrors check_swma_accuracy"""
        test_data = load_test_data()
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['swma']
        
        result = swma(
            close,
            period=expected['default_params']['period']
        )
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        # Match Rust test tolerance: absolute diff <= 1e-8 (no looser)
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=0.0,
            atol=1e-8,
            msg="SWMA last 5 values mismatch"
        )
    
    def test_default_period(self):
        """Test SWMA with consistent period usage"""
        data = np.random.random(100)
        
        # Period is now required, just like ALMA
        result = swma(data, 5)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == data.shape
        
    def test_kernel_selection(self):
        """Test different kernel selections"""
        data = np.random.random(1000)
        period = 5
        
        # Test all kernel options
        result_auto = swma(data, period, kernel='auto')
        result_scalar = swma(data, period, kernel='scalar')
        
        # Results should be very close (within floating point precision)
        np.testing.assert_allclose(result_auto, result_scalar, rtol=1e-10)
        
        # Test AVX kernels when compiled and available; otherwise skip gracefully
        try:
            result_avx2 = swma(data, period, kernel='avx2')
            np.testing.assert_allclose(result_scalar, result_avx2, rtol=1e-10)
        except ValueError as e:
            msg = str(e)
            if 'not compiled' in msg or 'not available' in msg:
                pytest.skip('AVX2 kernel not available in this build/CPU')
            else:
                raise
        
        try:
            result_avx512 = swma(data, period, kernel='avx512')
            np.testing.assert_allclose(result_scalar, result_avx512, rtol=1e-10)
        except ValueError as e:
            msg = str(e)
            if 'not compiled' in msg or 'not available' in msg:
                pytest.skip('AVX512 kernel not available in this build/CPU')
            else:
                raise
        
    def test_invalid_kernel(self):
        """Test error handling for invalid kernel"""
        data = np.random.random(100)
        
        with pytest.raises(ValueError, match="Unknown kernel"):
            swma(data, 5, kernel='invalid_kernel')
            
    def test_error_all_nan(self):
        """Test error when all values are NaN"""
        data = np.full(10, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            swma(data, 5)
            
    def test_error_invalid_period(self):
        """Test error for invalid period values"""
        data = np.random.random(10)
        
        # Period exceeds data length
        with pytest.raises(ValueError, match="Invalid period"):
            swma(data, 11)
            
        # Period is zero
        with pytest.raises(ValueError, match="Invalid period"):
            swma(data, 0)
            
    def test_error_not_enough_valid_data(self):
        """Test error when not enough valid data after NaN values"""
        # First 8 values are NaN, only 2 valid values, but need period 5
        data = np.array([np.nan] * 8 + [1.0, 2.0])
        
        with pytest.raises(ValueError, match="Not enough valid data"):
            swma(data, 5)
            
    def test_leading_nans(self):
        """Test SWMA with leading NaN values"""
        # Mix of NaN and valid values
        data = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        period = 3
        
        result = swma(data, period)
        
        # First valid index is 2, so warmup completes at index 2+period-1 = 4
        assert np.isnan(result[0:4]).all()
        assert not np.isnan(result[4:]).any()
        
    def test_compare_with_rust(self):
        """Test that Python bindings match Rust implementation"""
        candles = load_test_data()
        data = candles['close']
        period = 5
        
        result = swma(data, period)
        compare_with_rust("swma", result, 'close', params={'period': period})
        
    def test_swma_stream(self):
        """Test SWMA streaming functionality"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        period = 5
        
        # Create stream
        stream = SwmaStream(period)
        
        # Batch calculation for comparison
        batch_result = swma(data, period)
        
        # Process data through stream
        stream_results = []
        for value in data:
            result = stream.update(value)
            stream_results.append(result if result is not None else np.nan)
            
        stream_results = np.array(stream_results)
        
        # Compare results (accounting for warmup)
        np.testing.assert_allclose(
            batch_result[period-1:], 
            stream_results[period-1:], 
            rtol=1e-10
        )
        
    def test_swma_stream_explicit_period(self):
        """Test SWMA stream with explicit period"""
        # Create stream with period 5
        stream = SwmaStream(5)
        
        # Feed some data
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        results = []
        for value in data:
            result = stream.update(value)
            results.append(result)
            
        # First 4 results should be None (period is 5)
        assert results[:4] == [None, None, None, None]
        assert results[4] is not None
        assert results[5] is not None
        
    def test_batch_calculation(self):
        """Test batch SWMA calculation with multiple periods"""
        data = np.random.random(100)
        period_range = (3, 10, 2)  # periods: 3, 5, 7, 9
        
        result = swma_batch(data, period_range)
        
        assert isinstance(result, dict)
        assert 'values' in result
        assert 'periods' in result
        
        # Check dimensions
        expected_periods = [3, 5, 7, 9]
        assert result['values'].shape == (len(expected_periods), len(data))
        np.testing.assert_array_equal(result['periods'], expected_periods)
        
        # Verify each row matches individual calculation
        for i, period in enumerate(expected_periods):
            individual_result = swma(data, period)
            np.testing.assert_allclose(
                result['values'][i], 
                individual_result, 
                rtol=1e-10
            )
            
    def test_batch_kernel_selection(self):
        """Test batch calculation with different kernels"""
        data = np.random.random(100)
        period_range = (5, 10, 5)
        
        # Test different kernels
        result_auto = swma_batch(data, period_range, kernel='auto')
        result_scalar = swma_batch(data, period_range, kernel='scalar')
        # All should produce same results for compiled kernels
        np.testing.assert_allclose(result_auto['values'], result_scalar['values'], rtol=1e-10)
        
        try:
            result_avx2 = swma_batch(data, period_range, kernel='avx2')
            np.testing.assert_allclose(result_scalar['values'], result_avx2['values'], rtol=1e-10)
        except ValueError as e:
            msg = str(e)
            if 'not compiled' in msg or 'not available' in msg:
                pytest.skip('AVX2 batch kernel not available in this build/CPU')
            else:
                raise
        
        try:
            result_avx512 = swma_batch(data, period_range, kernel='avx512')
            np.testing.assert_allclose(result_scalar['values'], result_avx512['values'], rtol=1e-10)
        except ValueError as e:
            msg = str(e)
            if 'not compiled' in msg or 'not available' in msg:
                pytest.skip('AVX512 batch kernel not available in this build/CPU')
            else:
                raise
        
    def test_batch_invalid_kernel(self):
        """Test batch calculation with invalid kernel"""
        data = np.random.random(100)
        period_range = (5, 10, 5)
        
        with pytest.raises(ValueError, match="Unknown kernel"):
            swma_batch(data, period_range, kernel='invalid')
            
    def test_zero_copy_performance(self):
        """Test that zero-copy operations are working (indirect test)"""
        # Large dataset to make copies noticeable
        data = np.random.random(1_000_000)
        period = 20
        
        # This should be fast due to zero-copy
        result = swma(data, period)
        
        # Verify result is correct shape and has expected NaN pattern
        assert result.shape == data.shape
        assert np.isnan(result[:period-1]).all()
        assert not np.isnan(result[period-1:]).any()
        
    def test_symmetric_weights(self):
        """Test that SWMA uses symmetric triangular weights"""
        # Create a simple pattern where we can verify the weights
        data = np.zeros(30)
        data[15] = 1.0  # Single spike in the middle
        period = 5
        
        result = swma(data, period)
        
        # With period 5, triangular weights are [1, 2, 3, 2, 1] / 9
        # When the spike is at position i in the window:
        # Position 0: weight = 1/9 ≈ 0.1111
        # Position 1: weight = 2/9 ≈ 0.2222
        # Position 2: weight = 3/9 ≈ 0.3333 (center)
        # Position 3: weight = 2/9 ≈ 0.2222
        # Position 4: weight = 1/9 ≈ 0.1111
        
        # result[i] uses data[i-period+1:i+1], so:
        # result[15] uses data[11:16] - spike at 15 is at position 4 (last)
        # result[16] uses data[12:17] - spike at 15 is at position 3
        # result[17] uses data[13:18] - spike at 15 is at position 2 (center)
        # result[18] uses data[14:19] - spike at 15 is at position 1
        # result[19] uses data[15:20] - spike at 15 is at position 0 (first)
        
        # Check the weights are correct
        assert abs(result[15] - 1/9) < 1e-10    # Position 4 in window
        assert abs(result[16] - 2/9) < 1e-10    # Position 3 in window  
        assert abs(result[17] - 3/9) < 1e-10    # Position 2 in window (center)
        assert abs(result[18] - 2/9) < 1e-10    # Position 1 in window
        assert abs(result[19] - 1/9) < 1e-10    # Position 0 in window
        
        # Verify zero outside the affected range
        assert result[10] == 0.0   # Too early
        assert result[20] == 0.0   # Too late
        
    def test_real_world_data(self):
        """Test with real-world conditions including warmup period"""
        # Simulate real price data with some noise
        np.random.seed(42)
        trend = np.linspace(100, 110, 200)
        noise = np.random.normal(0, 0.5, 200)
        data = trend + noise
        
        period = 10
        result = swma(data, period)
        
        # Check warmup period
        assert np.isnan(result[:period-1]).all()
        assert not np.isnan(result[period-1:]).any()
        
        # Result should be smoother than input
        input_volatility = np.std(np.diff(data[period:]))
        output_volatility = np.std(np.diff(result[period:]))
        assert output_volatility < input_volatility
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Single value
        data = np.array([42.0])
        with pytest.raises(ValueError, match="Invalid period"):
            swma(data, 2)
            
        # Period equals data length
        data = np.random.random(10)
        result = swma(data, 10)
        assert np.isnan(result[:-1]).all()
        assert not np.isnan(result[-1])
        
        # Minimum valid period
        data = np.array([1.0, 2.0, 3.0])
        result = swma(data, 1)
        np.testing.assert_array_equal(result, data)  # Period 1 returns input as-is


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
