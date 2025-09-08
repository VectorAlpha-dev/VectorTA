"""
Python binding tests for CoRa Wave indicator.
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
from rust_comparison import compare_with_rust

class TestCoraWave:
    @pytest.fixture(scope='class')
    def test_data(self):
        """Load test data from CSV file"""
        return load_test_data()
    
    def test_cora_wave_partial_params(self, test_data):
        """Test CoRa Wave with partial parameters - mirrors check_cora_wave_partial_params"""
        close = test_data['close']
        
        # Test with defaults
        result = ta_indicators.cora_wave(close, 20, 2.0, True)
        assert len(result) == len(close)
    
    def test_cora_wave_accuracy(self, test_data):
        """Test CoRa Wave matches expected values from Rust tests - mirrors check_cora_wave_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['cora_wave']
        
        result = ta_indicators.cora_wave(
            close,
            period=expected['default_params']['period'],
            r_multi=expected['default_params']['r_multi'],
            smooth=expected['default_params']['smooth']
        )
        
        assert len(result) == len(close), "Output length should match input length"
        
        # Check last 5 values match expected
        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=1e-8,  # Use same tight tolerance as ALMA
            msg="CoRa Wave last 5 values mismatch"
        )
        
        # Compare full output with Rust
        # Note: cora_wave not yet registered in generate_references binary
        # compare_with_rust('cora_wave', result, 'close', expected['default_params'])
    
    def test_cora_wave_default_candles(self, test_data):
        """Test CoRa Wave with default parameters - mirrors check_cora_wave_default_candles"""
        close = test_data['close']
        
        # Default params: period=20, r_multi=2.0, smooth=True
        result = ta_indicators.cora_wave(close, 20, 2.0, True)
        assert len(result) == len(close)
    
    def test_cora_wave_zero_period(self):
        """Test CoRa Wave fails with zero period - mirrors check_cora_wave_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.cora_wave(input_data, period=0, r_multi=2.0, smooth=True)
    
    def test_cora_wave_empty_input(self):
        """Test CoRa Wave fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.cora_wave(empty, 20, 2.0, True)
    
    def test_cora_wave_all_nan(self):
        """Test CoRa Wave fails with all NaN input - mirrors check_cora_wave_all_nan"""
        nan_data = np.array([np.nan, np.nan, np.nan])
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.cora_wave(nan_data, period=20, r_multi=2.0, smooth=True)
    
    def test_cora_wave_invalid_r_multi(self):
        """Test CoRa Wave with edge case r_multi values"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test with NaN r_multi - should raise error
        with pytest.raises(ValueError, match="Invalid r_multi"):
            ta_indicators.cora_wave(data, period=2, r_multi=float('nan'), smooth=False)
        
        # Test with negative r_multi - should raise error
        with pytest.raises(ValueError, match="Invalid r_multi"):
            ta_indicators.cora_wave(data, period=2, r_multi=-1.0, smooth=False)
        
        # Test with zero r_multi - currently allowed, produces valid output
        result_zero = ta_indicators.cora_wave(data, period=2, r_multi=0.0, smooth=False)
        assert len(result_zero) == len(data)
        
        # Test with infinity r_multi - should raise error
        with pytest.raises(ValueError, match="Invalid r_multi"):
            ta_indicators.cora_wave(data, period=2, r_multi=float('inf'), smooth=False)
    
    def test_cora_wave_period_exceeds_length(self):
        """Test CoRa Wave fails when period exceeds data length - mirrors check_cora_wave_period_exceeds_length"""
        small_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.cora_wave(small_data, period=10, r_multi=2.0, smooth=True)
    
    def test_cora_wave_very_small_dataset(self):
        """Test CoRa Wave fails with insufficient data - mirrors check_cora_wave_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.cora_wave(single_point, period=20, r_multi=2.0, smooth=True)
    
    def test_cora_wave_nan_handling(self, test_data):
        """Test CoRa Wave handles NaN values correctly"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['cora_wave']
        
        result = ta_indicators.cora_wave(
            close,
            period=expected['default_params']['period'],
            r_multi=expected['default_params']['r_multi'],
            smooth=expected['default_params']['smooth']
        )
        
        assert len(result) == len(close)
        
        # After warmup period (22), no NaN values should exist
        warmup = expected['warmup_period']  # Should be 22 for default params with smoothing
        if len(result) > warmup + 100:
            assert not np.any(np.isnan(result[warmup + 100:])), "Found unexpected NaN after warmup period"
        
        # First warmup values should be NaN
        assert np.all(np.isnan(result[:warmup])), "Expected NaN in warmup period"
    
    def test_cora_wave_no_smoothing(self, test_data):
        """Test CoRa Wave without smoothing"""
        close = test_data['close'][:100]  # Use subset for faster test
        
        result_smooth = ta_indicators.cora_wave(close, period=20, r_multi=2.0, smooth=True)
        result_raw = ta_indicators.cora_wave(close, period=20, r_multi=2.0, smooth=False)
        
        assert len(result_smooth) == len(close)
        assert len(result_raw) == len(close)
        
        # Results should be different when smoothing is on vs off
        # (except for NaN values at the beginning)
        valid_idx = ~np.isnan(result_smooth) & ~np.isnan(result_raw)
        if np.any(valid_idx):
            assert not np.allclose(result_smooth[valid_idx], result_raw[valid_idx], rtol=1e-10), \
                "Smoothed and raw values should be different"
    
    def test_cora_wave_different_r_multi(self, test_data):
        """Test CoRa Wave with different r_multi values"""
        close = test_data['close'][:100]  # Use subset for faster test
        
        result1 = ta_indicators.cora_wave(close, period=20, r_multi=1.0, smooth=False)
        result2 = ta_indicators.cora_wave(close, period=20, r_multi=2.0, smooth=False)
        result3 = ta_indicators.cora_wave(close, period=20, r_multi=3.0, smooth=False)
        
        # Results should be different for different r_multi values
        valid_idx = ~np.isnan(result1) & ~np.isnan(result2) & ~np.isnan(result3)
        if np.any(valid_idx):
            assert not np.allclose(result1[valid_idx], result2[valid_idx], rtol=1e-10), \
                "Different r_multi values should produce different results"
            assert not np.allclose(result2[valid_idx], result3[valid_idx], rtol=1e-10), \
                "Different r_multi values should produce different results"
    
    def test_cora_wave_streaming(self, test_data):
        """Test CoRa Wave streaming matches batch calculation"""
        close = test_data['close'][:100]  # Use subset for speed
        period = 20
        r_multi = 2.0
        smooth = True
        
        # Batch calculation
        batch_result = ta_indicators.cora_wave(close, period=period, r_multi=r_multi, smooth=smooth)
        
        # Streaming calculation
        stream = ta_indicators.CoraWaveStream(period=period, r_multi=r_multi, smooth=smooth)
        stream_values = []
        
        for price in close:
            result = stream.update(price)
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        # Compare batch vs streaming
        assert len(batch_result) == len(stream_values)
        
        # Compare values where both are not NaN
        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, atol=1e-9, 
                        msg=f"CoRa Wave streaming mismatch at index {i}")
    
    def test_cora_wave_batch(self, test_data):
        """Test CoRa Wave batch processing"""
        close = test_data['close'][:100]  # Use subset for speed
        expected = EXPECTED_OUTPUTS['cora_wave']
        
        # Single parameter combination test
        result = ta_indicators.cora_wave_batch(
            close,
            period_range=(20, 20, 0),  # Default period only
            r_multi_range=(2.0, 2.0, 0.0),  # Default r_multi only
            smooth=False  # Without smoothing for simplicity
        )
        
        assert 'values' in result
        assert 'periods' in result
        assert 'r_multis' in result
        assert 'smooth' in result
        
        # Should have 1 combination
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        # Extract the single row and verify it matches regular calculation
        default_row = result['values'][0]
        regular_result = ta_indicators.cora_wave(close, 20, 2.0, False)
        
        # Compare where both are not NaN
        valid_idx = ~np.isnan(default_row) & ~np.isnan(regular_result)
        assert_close(
            default_row[valid_idx],
            regular_result[valid_idx],
            rtol=1e-8,
            msg="CoRa Wave batch default row mismatch"
        )
    
    def test_cora_wave_batch_multiple_params(self, test_data):
        """Test CoRa Wave batch with multiple parameter combinations"""
        close = test_data['close'][:50]  # Smaller dataset for speed
        
        # Multiple parameter combinations
        result = ta_indicators.cora_wave_batch(
            close,
            period_range=(15, 20, 5),  # 15, 20
            r_multi_range=(1.5, 2.0, 0.5),  # 1.5, 2.0
            smooth=False  # Single smooth option for testing
        )
        
        # Should have 2 * 2 = 4 combinations
        assert result['values'].shape[0] == 4
        assert result['values'].shape[1] == len(close)
        assert len(result['periods']) == 4
        assert len(result['r_multis']) == 4
        
        # Verify first combination
        assert result['periods'][0] == 15
        assert result['r_multis'][0] == 1.5
        
        # Verify last combination
        assert result['periods'][3] == 20
        assert result['r_multis'][3] == 2.0
    
    def test_cora_wave_batch_sweep(self, test_data):
        """Test CoRa Wave batch with larger parameter sweep"""
        close = test_data['close'][:100]  # Use subset for speed
        expected = EXPECTED_OUTPUTS['cora_wave']
        
        # Use parameters from expected outputs
        batch_range = expected['batch_range']
        result = ta_indicators.cora_wave_batch(
            close,
            period_range=batch_range['period_range'],
            r_multi_range=batch_range['r_multi_range'],
            smooth=batch_range['smooth']
        )
        
        # Should have 3 periods * 3 r_multis = 9 combinations
        # Periods: 15, 20, 25 (range 15-25 step 5)
        # R_multis: 1.5, 2.0, 2.5 (range 1.5-2.5 step 0.5)
        assert result['values'].shape[0] == 9
        assert result['values'].shape[1] == len(close)
        
        # Find the default params row (period=20, r_multi=2.0)
        default_idx = None
        for i in range(len(result['periods'])):
            if result['periods'][i] == 20 and abs(result['r_multis'][i] - 2.0) < 1e-10:
                default_idx = i
                break
        
        assert default_idx is not None, "Default parameters not found in batch result"
        
        # Compare with single calculation (without smoothing)
        single_result = ta_indicators.cora_wave(close, 20, 2.0, False)
        batch_row = result['values'][default_idx]
        
        # Values should match where both are not NaN
        valid_idx = ~np.isnan(batch_row) & ~np.isnan(single_result)
        assert_close(
            batch_row[valid_idx],
            single_result[valid_idx],
            rtol=1e-8,
            msg="Batch row doesn't match single calculation"
        )
    
    def test_cora_wave_edge_cases(self, test_data):
        """Test CoRa Wave with various edge cases"""
        # Test with very small period
        close = test_data['close'][:20]
        result = ta_indicators.cora_wave(close, period=2, r_multi=2.0, smooth=False)
        assert len(result) == len(close)
        
        # Test with period equal to data length
        result = ta_indicators.cora_wave(close, period=len(close), r_multi=2.0, smooth=False)
        assert len(result) == len(close)
        # Should have NaN for all but the last value
        assert np.all(np.isnan(result[:-1]))
        assert not np.isnan(result[-1])
        
        # Test with large r_multi
        result = ta_indicators.cora_wave(close, period=5, r_multi=10.0, smooth=False)
        assert len(result) == len(close)
        
        # Test with very small r_multi (but positive)
        result = ta_indicators.cora_wave(close, period=5, r_multi=0.001, smooth=False)
        assert len(result) == len(close)

    def test_cora_wave_with_leading_nans(self, test_data):
        """Test CoRa Wave handles leading NaN values correctly"""
        close = test_data['close'].copy()
        # Add some NaN values at the beginning
        close[:5] = np.nan
        
        result = ta_indicators.cora_wave(close, period=20, r_multi=2.0, smooth=True)
        assert len(result) == len(close)
        
        # First 5 + warmup period should be NaN
        expected_nan_count = 5 + 22  # 5 leading NaNs + 22 warmup
        assert np.all(np.isnan(result[:expected_nan_count]))
        
        # After that, should have valid values
        if len(result) > expected_nan_count + 10:
            assert not np.any(np.isnan(result[expected_nan_count + 10:]))
    
    def test_cora_wave_performance_characteristics(self, test_data):
        """Test CoRa Wave maintains expected performance characteristics"""
        # Generate synthetic trending data
        trend_up = np.linspace(100, 200, 100)
        trend_down = np.linspace(200, 100, 100)
        choppy = np.array([100 + 10 * (i % 2) for i in range(100)], dtype=np.float64)
        
        # CoRa Wave should smooth the choppy data
        result_choppy = ta_indicators.cora_wave(choppy, period=10, r_multi=2.0, smooth=True)
        
        # Calculate the variance of the smoothed output (excluding NaN)
        valid_result = result_choppy[~np.isnan(result_choppy)]
        if len(valid_result) > 20:
            # Smoothed output should have lower variance than input
            input_var = np.var(choppy[-len(valid_result):])
            output_var = np.var(valid_result)
            assert output_var < input_var, "CoRa Wave should smooth choppy data"
        
        # For trending data, CoRa Wave should follow the trend
        result_up = ta_indicators.cora_wave(trend_up, period=10, r_multi=2.0, smooth=True)
        valid_up = result_up[~np.isnan(result_up)]
        if len(valid_up) > 2:
            # Check that the trend is preserved (mostly increasing)
            diffs = np.diff(valid_up)
            assert np.sum(diffs > 0) > len(diffs) * 0.7, "CoRa Wave should follow upward trend"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])