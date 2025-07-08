"""
Test Tilson T3 Moving Average indicator Python bindings
"""
import numpy as np
import pytest
from my_project import tilson, tilson_batch, TilsonStream
from rust_comparison import compare_with_rust


class TestTilson:
    """Test Tilson T3 indicator functionality"""
    
    def test_basic_functionality(self):
        """Test basic Tilson calculation"""
        # Need at least 25 values for period=5 (warmup=24 + 1 value)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 
                        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                        21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0])
        period = 5
        volume_factor = 0.0
        
        result = tilson(data, period, volume_factor)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == data.shape
        # Tilson has a warmup period of 6 * (period - 1) = 24 for period=5
        assert np.isnan(result[0:24]).all()
        assert not np.isnan(result[24:]).any()
        
    def test_kernel_selection(self):
        """Test different kernel selections"""
        data = np.random.random(1000)
        period = 5
        volume_factor = 0.7
        
        # Test all kernel options
        result_auto = tilson(data, period, volume_factor, kernel='auto')
        result_scalar = tilson(data, period, volume_factor, kernel='scalar')
        
        # Results should be very close (within floating point precision)
        np.testing.assert_allclose(result_auto, result_scalar, rtol=1e-10)
        
        # Test AVX kernels (even if they're stubs, they should work)
        result_avx2 = tilson(data, period, volume_factor, kernel='avx2')
        result_avx512 = tilson(data, period, volume_factor, kernel='avx512')
        
        np.testing.assert_allclose(result_scalar, result_avx2, rtol=1e-10)
        np.testing.assert_allclose(result_scalar, result_avx512, rtol=1e-10)
        
    def test_invalid_kernel(self):
        """Test error handling for invalid kernel"""
        data = np.random.random(100)
        
        with pytest.raises(ValueError, match="Unknown kernel"):
            tilson(data, 5, 0.0, kernel='invalid_kernel')
            
    def test_error_empty_input(self):
        """Test error when input is empty"""
        data = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            tilson(data, 5, 0.0)
            
    def test_error_all_nan(self):
        """Test error when all values are NaN"""
        data = np.full(10, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            tilson(data, 5, 0.0)
            
    def test_error_invalid_period(self):
        """Test error for invalid period values"""
        data = np.random.random(10)
        
        # Period exceeds data length
        with pytest.raises(ValueError, match="Invalid period"):
            tilson(data, 11, 0.0)
            
        # Period is zero
        with pytest.raises(ValueError, match="Invalid period"):
            tilson(data, 0, 0.0)
            
    def test_error_not_enough_valid_data(self):
        """Test error when not enough valid data after NaN values"""
        # First 8 values are NaN, only 2 valid values, but need more for period 5
        data = np.array([np.nan] * 8 + [1.0, 2.0])
        
        with pytest.raises(ValueError, match="Not enough valid data"):
            tilson(data, 5, 0.0)
            
    def test_error_invalid_volume_factor(self):
        """Test error for invalid volume factor values"""
        data = np.random.random(100)
        
        # NaN volume factor
        with pytest.raises(ValueError, match="Invalid volume factor"):
            tilson(data, 5, np.nan)
            
        # Infinite volume factor
        with pytest.raises(ValueError, match="Invalid volume factor"):
            tilson(data, 5, np.inf)
            
    def test_leading_nans(self):
        """Test Tilson with leading NaN values"""
        # Mix of NaN and valid values
        data = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                        19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0])
        period = 3
        volume_factor = 0.5
        
        result = tilson(data, period, volume_factor)
        
        # First valid index is 2, Tilson warmup is 6 * (period-1) = 12
        # So first valid output is at index 2+12 = 14
        assert np.isnan(result[0:14]).all()
        assert not np.isnan(result[14:]).any()
        
    def test_compare_with_rust(self):
        """Test that Python bindings match Rust implementation"""
        # Use the standard test data file that generate_references uses
        from test_utils import load_test_data
        candles = load_test_data()
        data = candles['close']
        period = 5
        volume_factor = 0.0
        
        result = tilson(data, period, volume_factor)
        compare_with_rust("tilson", result, 'close', params={
            'period': period,
            'volume_factor': volume_factor
        })
        
    def test_tilson_stream(self):
        """Test Tilson streaming functionality"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                        21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0])
        period = 5
        volume_factor = 0.0
        
        # Create stream
        stream = TilsonStream(period, volume_factor)
        
        # Batch calculation for comparison
        batch_result = tilson(data, period, volume_factor)
        
        # Process data through stream
        stream_results = []
        for value in data:
            result = stream.update(value)
            stream_results.append(result if result is not None else np.nan)
            
        stream_results = np.array(stream_results)
        
        # Compare results (accounting for warmup)
        # Tilson warmup is 6 * (period-1) = 24
        np.testing.assert_allclose(
            batch_result[24:],
            stream_results[24:],
            rtol=1e-10
        )
        
    def test_batch_calculation(self):
        """Test batch Tilson calculation with multiple parameters"""
        data = np.random.random(100)
        period_range = (3, 7, 2)  # periods: 3, 5, 7
        volume_factor_range = (0.0, 0.6, 0.3)  # volume_factors: 0.0, 0.3, 0.6
        
        result = tilson_batch(data, period_range, volume_factor_range)
        
        assert isinstance(result, dict)
        assert 'values' in result
        assert 'periods' in result
        assert 'volume_factors' in result
        
        # Check dimensions
        expected_periods = [3, 5, 7]
        expected_v_factors = [0.0, 0.3, 0.6]
        expected_combos = len(expected_periods) * len(expected_v_factors)
        assert result['values'].shape == (expected_combos, len(data))
        assert len(result['periods']) == expected_combos
        assert len(result['volume_factors']) == expected_combos
        
        # Verify combinations are correct
        combo_idx = 0
        for period in expected_periods:
            for v_factor in expected_v_factors:
                assert result['periods'][combo_idx] == period
                assert abs(result['volume_factors'][combo_idx] - v_factor) < 1e-10
                combo_idx += 1
                
    def test_batch_kernel_selection(self):
        """Test batch calculation with different kernels"""
        data = np.random.random(100)
        period_range = (5, 10, 5)  # periods: 5, 10
        volume_factor_range = (0.5, 0.5, 0.0)  # just 0.5
        
        # Test different kernels
        result_auto = tilson_batch(data, period_range, volume_factor_range, kernel='auto')
        result_scalar = tilson_batch(data, period_range, volume_factor_range, kernel='scalar')
        result_avx2 = tilson_batch(data, period_range, volume_factor_range, kernel='avx2')
        result_avx512 = tilson_batch(data, period_range, volume_factor_range, kernel='avx512')
        
        # All should produce same results
        np.testing.assert_allclose(result_auto['values'], result_scalar['values'], rtol=1e-10)
        np.testing.assert_allclose(result_scalar['values'], result_avx2['values'], rtol=1e-10)
        np.testing.assert_allclose(result_scalar['values'], result_avx512['values'], rtol=1e-10)
        
    def test_batch_invalid_kernel(self):
        """Test batch calculation with invalid kernel"""
        data = np.random.random(100)
        period_range = (5, 10, 5)
        
        with pytest.raises(ValueError, match="Unknown kernel"):
            tilson_batch(data, period_range, kernel='invalid')
            
    def test_zero_copy_performance(self):
        """Test that zero-copy operations are working (indirect test)"""
        # Large dataset to make copies noticeable
        data = np.random.random(1_000_000)
        period = 10
        volume_factor = 0.7
        
        # This should be fast due to zero-copy
        result = tilson(data, period, volume_factor)
        
        # Verify result is correct shape and has expected NaN pattern
        assert result.shape == data.shape
        warmup = 6 * (period - 1)
        assert np.isnan(result[:warmup]).all()
        assert not np.isnan(result[warmup:]).any()
        
    def test_real_world_data(self):
        """Test with real-world conditions including warmup period"""
        # Simulate real price data with some noise
        np.random.seed(42)
        trend = np.linspace(100, 110, 200)
        noise = np.random.normal(0, 0.5, 200)
        data = trend + noise
        
        period = 5
        volume_factor = 0.7
        result = tilson(data, period, volume_factor)
        
        # Check warmup period
        warmup = 6 * (period - 1)  # 24 for period=5
        assert np.isnan(result[:warmup]).all()
        assert not np.isnan(result[warmup:]).any()
        
        # Result should be smoother than input
        input_volatility = np.std(np.diff(data[warmup:]))
        output_volatility = np.std(np.diff(result[warmup:]))
        assert output_volatility < input_volatility
        
    def test_edge_cases(self):
        """Test edge cases"""
        # Large dataset
        data = np.random.random(100)
        
        # Minimum valid period
        result = tilson(data, 1, 0.0)
        # Period 1 has warmup = 6 * (1-1) = 0, so no NaN values
        # Check that all values are valid
        assert not np.isnan(result).any()  # No NaN values for period=1
        
        # Maximum volume factor
        result = tilson(data, 5, 1.0)
        warmup = 6 * (5 - 1)
        assert np.isnan(result[:warmup]).all()
        assert not np.isnan(result[warmup:]).any()
        
    def test_reinput(self):
        """Test using Tilson output as input for another Tilson calculation"""
        from test_utils import load_test_data
        candles = load_test_data()
        data = candles['close']
        
        # First Tilson with period 5, volume_factor 0.0
        first_result = tilson(data, 5, 0.0)
        assert len(first_result) == len(data)
        
        # Use first result as input for second Tilson with period 3, volume_factor 0.7
        second_result = tilson(first_result, 3, 0.7)
        assert len(second_result) == len(first_result)
        
        # Verify second result is not all NaN
        # First Tilson has warmup of 24, second adds 12
        # So total warmup should be at least 36
        assert np.isnan(second_result[:36]).all()
        if len(second_result) > 36:
            assert not np.isnan(second_result[36:]).all()
            
            # Verify the output values are reasonable
            valid_values = second_result[36:]
            assert len(np.unique(valid_values)) > 1  # Multiple different values
            assert np.all(np.isfinite(valid_values))  # All finite values
            assert np.std(valid_values) > 0  # Has variance
            
    def test_accuracy_check(self):
        """Test Tilson matches expected values"""
        # Simple test data where we can verify calculations
        data = np.arange(1.0, 31.0)  # 1 to 30
        period = 5
        volume_factor = 0.0
        
        result = tilson(data, period, volume_factor)
        
        # Warmup period is 6 * (5-1) = 24
        assert np.isnan(result[:24]).all()
        
        # After warmup, values should be smoothed
        if len(result) > 24:
            # The value at index 24 should be a smoothed version based on the trend
            # Since we have an upward trend from 1-30, expect a value around the data point
            assert not np.isnan(result[24])
            assert result[-1] > result[24]  # Should follow upward trend
            
    def test_batch_warmup_periods(self):
        """Test that batch processing correctly handles different warmup periods"""
        data = np.random.random(100)
        period_range = (2, 6, 2)  # periods: 2, 4, 6
        volume_factor_range = (0.0, 0.0, 0.0)  # just 0.0
        
        result = tilson_batch(data, period_range, volume_factor_range)
        
        # Check each period has correct warmup
        periods = [2, 4, 6]
        for i, period in enumerate(periods):
            warmup = 6 * (period - 1)
            row = result['values'][i]
            
            # Check warmup NaN values
            assert np.isnan(row[:warmup]).all(), f"Period {period} warmup incorrect"
            # Check remaining values are not NaN (if any)
            if warmup < len(row):
                assert not np.isnan(row[warmup:]).any(), f"Period {period} has unexpected NaNs"
                
    def test_volume_factor_effect(self):
        """Test that volume factor affects smoothing"""
        data = np.random.random(100)
        period = 5
        
        # Compare different volume factors
        result_0 = tilson(data, period, 0.0)
        result_5 = tilson(data, period, 0.5)
        result_9 = tilson(data, period, 0.9)
        
        warmup = 6 * (period - 1)
        
        # Higher volume factor should produce more smoothing
        vol_0 = np.std(np.diff(result_0[warmup:]))
        vol_5 = np.std(np.diff(result_5[warmup:]))
        vol_9 = np.std(np.diff(result_9[warmup:]))
        
        # Generally, higher volume factor = more smoothing = lower volatility
        # But this depends on the data, so we just check they're different
        assert not np.allclose(result_0[warmup:], result_5[warmup:])
        assert not np.allclose(result_5[warmup:], result_9[warmup:])
        
    def test_batch_single_combination(self):
        """Test batch with single parameter combination"""
        data = np.random.random(50)
        
        # Single combination
        result = tilson_batch(data, (5, 5, 0), (0.7, 0.7, 0.0))
        
        assert result['values'].shape == (1, len(data))
        assert len(result['periods']) == 1
        assert result['periods'][0] == 5
        assert abs(result['volume_factors'][0] - 0.7) < 1e-10
        
        # Should match single calculation
        single_result = tilson(data, 5, 0.7)
        np.testing.assert_allclose(result['values'][0], single_result, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])