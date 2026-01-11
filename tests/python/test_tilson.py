"""
Test Tilson T3 Moving Average indicator Python bindings
"""
import numpy as np
import pytest
from my_project import tilson, tilson_batch, TilsonStream
from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS
from rust_comparison import compare_with_rust


class TestTilson:
    """Test Tilson T3 indicator functionality"""
    
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_tilson_accuracy(self, test_data):
        """Test Tilson matches expected values from Rust tests - mirrors check_tilson_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['tilson']
        
        result = tilson(
            close,
            period=expected['default_params']['period'],
            volume_factor=expected['default_params']['volume_factor']
        )
        
        assert len(result) == len(close)
        
        
        
        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=0.0,
            atol=1e-8,
            msg="Tilson last 5 values mismatch"
        )
        
        
        
        compare_with_rust('tilson', result, 'close', expected['default_params'], rtol=0.0, atol=1e-8)
    
    def test_tilson_default_params(self, test_data):
        """Test Tilson with default parameters - mirrors check_tilson_default_candles"""
        close = test_data['close']
        
        
        result = tilson(close, 5, 0.0)
        assert len(result) == len(close)
        
        
        warmup = 6 * (5 - 1)  
        assert np.all(np.isnan(result[:warmup]))
        assert not np.any(np.isnan(result[warmup:]))
    
    def test_basic_functionality(self):
        """Test basic Tilson calculation"""
        
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 
                        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                        21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0])
        period = 5
        volume_factor = 0.0
        
        result = tilson(data, period, volume_factor)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == data.shape
        
        assert np.isnan(result[0:24]).all()
        assert not np.isnan(result[24:]).any()
        
    def test_kernel_selection(self):
        """Test different kernel selections"""
        data = np.random.random(1000)
        period = 5
        volume_factor = 0.7
        
        
        result_auto = tilson(data, period, volume_factor, kernel='auto')
        result_scalar = tilson(data, period, volume_factor, kernel='scalar')
        
        
        np.testing.assert_allclose(result_auto, result_scalar, rtol=1e-10)
        
        
        try:
            result_avx2 = tilson(data, period, volume_factor, kernel='avx2')
            np.testing.assert_allclose(result_scalar, result_avx2, rtol=1e-10)
        except ValueError as e:
            if "AVX2 kernel not compiled" not in str(e):
                raise
                
        try:
            result_avx512 = tilson(data, period, volume_factor, kernel='avx512')
            np.testing.assert_allclose(result_scalar, result_avx512, rtol=1e-10)
        except ValueError as e:
            if "AVX512 kernel not compiled" not in str(e):
                raise
        
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
        
        
        with pytest.raises(ValueError, match="Invalid period"):
            tilson(data, 11, 0.0)
            
        
        with pytest.raises(ValueError, match="Invalid period"):
            tilson(data, 0, 0.0)
            
    def test_error_not_enough_valid_data(self):
        """Test error when not enough valid data after NaN values"""
        
        data = np.array([np.nan] * 8 + [1.0, 2.0])
        
        with pytest.raises(ValueError, match="Not enough valid data"):
            tilson(data, 5, 0.0)
            
    def test_error_invalid_volume_factor(self):
        """Test error for invalid volume factor values"""
        data = np.random.random(100)
        
        
        with pytest.raises(ValueError, match="Invalid volume factor"):
            tilson(data, 5, np.nan)
            
        
        with pytest.raises(ValueError, match="Invalid volume factor"):
            tilson(data, 5, np.inf)
            
    def test_leading_nans(self):
        """Test Tilson with leading NaN values"""
        
        data = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                        19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0])
        period = 3
        volume_factor = 0.5
        
        result = tilson(data, period, volume_factor)
        
        
        
        assert np.isnan(result[0:14]).all()
        assert not np.isnan(result[14:]).any()
        
    def test_tilson_zero_period(self):
        """Test Tilson fails with zero period - mirrors check_tilson_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            tilson(input_data, period=0, volume_factor=0.0)
    
    def test_tilson_period_exceeds_length(self):
        """Test Tilson fails when period exceeds data length - mirrors check_tilson_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            tilson(data_small, period=10, volume_factor=0.0)
    
    def test_tilson_very_small_dataset(self):
        """Test Tilson fails with insufficient data - mirrors check_tilson_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            tilson(single_point, period=5, volume_factor=0.0)
        
    def test_tilson_streaming(self, test_data):
        """Test Tilson streaming matches batch calculation - mirrors check_tilson_streaming"""
        close = test_data['close']
        period = 5
        volume_factor = 0.0
        
        
        batch_result = tilson(close, period=period, volume_factor=volume_factor)
        
        
        stream = TilsonStream(period=period, volume_factor=volume_factor)
        stream_values = []
        
        for price in close:
            result = stream.update(price)
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        
        assert len(batch_result) == len(stream_values)
        
        
        
        
        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            
            if i < 50:
                assert_close(b, s, rtol=1e-4, atol=1e-4,
                            msg=f"Tilson streaming mismatch at index {i}")
            else:
                
                assert_close(b, s, rtol=1e-7, atol=1e-7,
                            msg=f"Tilson streaming mismatch at index {i}")
        
    def test_tilson_batch(self, test_data):
        """Test Tilson batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        
        result = tilson_batch(
            close,
            period_range=(5, 5, 0),  
            volume_factor_range=(0.0, 0.0, 0.0)  
        )
        
        assert 'values' in result
        assert 'periods' in result
        assert 'volume_factors' in result
        
        
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        
        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['tilson']['last_5_values']
        
        
        assert_close(
            default_row[-5:],
            expected,
            rtol=0.0,
            atol=1e-8,
            msg="Tilson batch default row mismatch"
        )
    
    def test_tilson_batch_multiple_params(self, test_data):
        """Test batch Tilson with multiple parameter combinations"""
        close = test_data['close'][:100]  
        
        
        result = tilson_batch(
            close,
            period_range=(3, 7, 2),  
            volume_factor_range=(0.0, 0.6, 0.3)  
        )
        
        
        expected_periods = [3, 5, 7]
        expected_v_factors = [0.0, 0.3, 0.6]
        expected_combos = len(expected_periods) * len(expected_v_factors)
        assert result['values'].shape == (expected_combos, len(close))
        
        
        combo_idx = 0
        for period in expected_periods:
            for v_factor in expected_v_factors:
                single_result = tilson(close, period, v_factor)
                batch_row = result['values'][combo_idx]
                
                
                valid_mask = ~np.isnan(single_result) & ~np.isnan(batch_row)
                if np.any(valid_mask):
                    assert_close(
                        batch_row[valid_mask],
                        single_result[valid_mask],
                        rtol=1e-10,
                        msg=f"Batch mismatch for period={period}, v_factor={v_factor}"
                    )
                combo_idx += 1
                
    def test_batch_kernel_selection(self):
        """Test batch calculation with different kernels"""
        data = np.random.random(100)
        period_range = (5, 10, 5)  
        volume_factor_range = (0.5, 0.5, 0.0)  
        
        
        result_auto = tilson_batch(data, period_range, volume_factor_range, kernel='auto')
        result_scalar = tilson_batch(data, period_range, volume_factor_range, kernel='scalar')
        
        
        np.testing.assert_allclose(result_auto['values'], result_scalar['values'], rtol=1e-10)
        
        
        try:
            result_avx2 = tilson_batch(data, period_range, volume_factor_range, kernel='avx2')
            np.testing.assert_allclose(result_scalar['values'], result_avx2['values'], rtol=1e-10)
        except ValueError as e:
            if "AVX2 kernel not compiled" not in str(e):
                raise
                
        try:
            result_avx512 = tilson_batch(data, period_range, volume_factor_range, kernel='avx512')
            np.testing.assert_allclose(result_scalar['values'], result_avx512['values'], rtol=1e-10)
        except ValueError as e:
            if "AVX512 kernel not compiled" not in str(e):
                raise
        
    def test_batch_invalid_kernel(self):
        """Test batch calculation with invalid kernel"""
        data = np.random.random(100)
        period_range = (5, 10, 5)
        
        with pytest.raises(ValueError, match="Unknown kernel"):
            tilson_batch(data, period_range, kernel='invalid')
            
    def test_zero_copy_performance(self):
        """Test that zero-copy operations are working (indirect test)"""
        
        data = np.random.random(1_000_000)
        period = 10
        volume_factor = 0.7
        
        
        result = tilson(data, period, volume_factor)
        
        
        assert result.shape == data.shape
        warmup = 6 * (period - 1)
        assert np.isnan(result[:warmup]).all()
        assert not np.isnan(result[warmup:]).any()
        
    def test_real_world_data(self):
        """Test with real-world conditions including warmup period"""
        
        np.random.seed(42)
        trend = np.linspace(100, 110, 200)
        noise = np.random.normal(0, 0.5, 200)
        data = trend + noise
        
        period = 5
        volume_factor = 0.7
        result = tilson(data, period, volume_factor)
        
        
        warmup = 6 * (period - 1)  
        assert np.isnan(result[:warmup]).all()
        assert not np.isnan(result[warmup:]).any()
        
        
        input_volatility = np.std(np.diff(data[warmup:]))
        output_volatility = np.std(np.diff(result[warmup:]))
        assert output_volatility < input_volatility
        
    def test_edge_cases(self):
        """Test edge cases"""
        
        data = np.random.random(100)
        
        
        result = tilson(data, 1, 0.0)
        
        
        assert not np.isnan(result).any()  
        
        
        result = tilson(data, 5, 1.0)
        warmup = 6 * (5 - 1)
        assert np.isnan(result[:warmup]).all()
        assert not np.isnan(result[warmup:]).any()
        
    def test_tilson_nan_handling(self, test_data):
        """Test Tilson handles NaN values correctly - mirrors check_tilson_nan_handling"""
        close = test_data['close']
        
        result = tilson(close, period=5, volume_factor=0.0)
        assert len(result) == len(close)
        
        
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"
        
        
        assert np.all(np.isnan(result[:24])), "Expected NaN in warmup period"
            
    def test_batch_warmup_periods(self):
        """Test that batch processing correctly handles different warmup periods"""
        data = np.random.random(100)
        period_range = (2, 6, 2)  
        volume_factor_range = (0.0, 0.0, 0.0)  
        
        result = tilson_batch(data, period_range, volume_factor_range)
        
        
        periods = [2, 4, 6]
        for i, period in enumerate(periods):
            warmup = 6 * (period - 1)
            row = result['values'][i]
            
            
            assert np.isnan(row[:warmup]).all(), f"Period {period} warmup incorrect"
            
            if warmup < len(row):
                assert not np.isnan(row[warmup:]).any(), f"Period {period} has unexpected NaNs"
                
    def test_volume_factor_effect(self):
        """Test that volume factor affects smoothing"""
        data = np.random.random(100)
        period = 5
        
        
        result_0 = tilson(data, period, 0.0)
        result_5 = tilson(data, period, 0.5)
        result_9 = tilson(data, period, 0.9)
        
        warmup = 6 * (period - 1)
        
        
        vol_0 = np.std(np.diff(result_0[warmup:]))
        vol_5 = np.std(np.diff(result_5[warmup:]))
        vol_9 = np.std(np.diff(result_9[warmup:]))
        
        
        
        assert not np.allclose(result_0[warmup:], result_5[warmup:])
        assert not np.allclose(result_5[warmup:], result_9[warmup:])
        
    def test_tilson_all_nan_input(self):
        """Test Tilson with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            tilson(all_nan, period=5, volume_factor=0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
