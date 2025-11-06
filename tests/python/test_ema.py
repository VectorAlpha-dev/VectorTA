"""
Python binding tests for EMA indicator.
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


class TestEma:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_ema_partial_params(self, test_data):
        """Test EMA with partial parameters (None values) - mirrors check_ema_partial_params"""
        close = test_data['close']
        
        # Test with default period (9)
        result = ta_indicators.ema(close, 9)
        assert len(result) == len(close)
    
    def test_ema_accuracy(self, test_data):
        """Test EMA matches expected values from Rust tests - mirrors check_ema_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['ema']
        
        result = ta_indicators.ema(
            close,
            period=expected['default_params']['period']
        )
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        # Match Rust test tolerance (abs diff < 1e-1); do not allow larger rtol
        assert_close(
            result[-5:],
            expected['last_five'],
            rtol=0.0,
            atol=1e-1,
            msg="EMA last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('ema', result, 'close', expected['default_params'])
    
    def test_ema_default_candles(self, test_data):
        """Test EMA with default candle data - mirrors check_ema_default_candles"""
        close = test_data['close']
        
        # Using default params
        result = ta_indicators.ema(close, 9)
        assert len(result) == len(close)
    
    def test_ema_zero_period(self, test_data):
        """Test EMA fails with zero period - mirrors check_ema_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.ema(input_data, 0)
    
    def test_ema_period_exceeds_length(self, test_data):
        """Test EMA fails when period exceeds data length - mirrors check_ema_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.ema(data_small, 10)
    
    def test_ema_very_small_dataset(self, test_data):
        """Test EMA with single data point - mirrors check_ema_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.ema(single_point, 9)
    
    def test_ema_empty_input(self, test_data):
        """Test EMA fails with empty input - mirrors check_ema_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty|Empty input"):
            ta_indicators.ema(empty, 9)
    
    def test_ema_all_nan_input(self, test_data):
        """Test EMA fails with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.ema(all_nan, 9)
    
    def test_ema_nan_handling(self, test_data):
        """Test EMA handles NaN values correctly - mirrors check_ema_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.ema(close, 9)
        assert len(result) == len(close)
        
        # Check that values after warm-up period are not NaN
        if len(result) > 30:
            for i, val in enumerate(result[30:]):
                assert not np.isnan(val), f"Found unexpected NaN at index {30 + i}"
    
    def test_ema_streaming(self, test_data):
        """Test EMA streaming interface - mirrors check_ema_streaming"""
        close = test_data['close']
        period = 9
        
        # Batch computation
        batch_result = ta_indicators.ema(close, period)
        
        # Streaming computation
        stream = ta_indicators.EmaStream(period)
        stream_values = []
        
        for price in close:
            stream_values.append(stream.update(price))
        
        # Compare results (skip warm-up period for comparison)
        warm_up = 240
        for i in range(warm_up, len(batch_result)):
            if not (np.isnan(batch_result[i]) and np.isnan(stream_values[i])):
                assert abs(batch_result[i] - stream_values[i]) < 1e-9, \
                    f"EMA streaming mismatch at idx {i}: batch={batch_result[i]}, stream={stream_values[i]}"
    
    def test_ema_warmup_period(self, test_data):
        """Test EMA warmup period produces NaN values correctly"""
        # Use simple data to verify warmup behavior
        simple_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        period = 5
        
        result = ta_indicators.ema(simple_data, period)
        
        # First value should NOT be NaN (EMA starts from first value)
        assert not np.isnan(result[0]), "EMA should start from first value"
        
        # After period, all values should be valid
        assert np.all(np.isfinite(result)), "All EMA values should be finite"
    
    def test_ema_batch(self, test_data):
        """Test EMA batch computation"""
        close = test_data['close']
        
        # Test batch with period range
        batch_result = ta_indicators.ema_batch(
            close,
            (5, 20, 5)  # period_range: start=5, end=20, step=5
        )
        
        # Should return a dict with 'values' and 'periods'
        assert 'values' in batch_result
        assert 'periods' in batch_result
        
        # Check dimensions
        periods = batch_result['periods']
        values = batch_result['values']
        
        expected_periods = [5, 10, 15, 20]
        assert len(periods) == len(expected_periods)
        assert all(p in periods for p in expected_periods)
        
        # Values should be a 2D array with shape (num_periods, data_length)
        assert values.shape == (len(expected_periods), len(close))
        
        # Verify each batch row matches individual calculation
        for i, period in enumerate(expected_periods):
            single_result = ta_indicators.ema(close, period)
            batch_row = values[i]
            
            # Compare values (allowing for small numerical differences)
            assert_close(batch_row, single_result, rtol=1e-10, atol=1e-10,
                        msg=f"Batch row for period {period} doesn't match single calculation")
    
    def test_ema_batch_edge_cases(self, test_data):
        """Test EMA batch with edge cases"""
        close = test_data['close'][:50]  # Use smaller dataset for speed
        
        # Test 1: Single period (no sweep)
        single_batch = ta_indicators.ema_batch(close, (9, 9, 0))
        assert single_batch['values'].shape == (1, len(close))
        assert single_batch['periods'][0] == 9
        
        # Test 2: Step larger than range (should only get start value)
        large_step_batch = ta_indicators.ema_batch(close, (5, 10, 20))
        assert large_step_batch['values'].shape == (1, len(close))
        assert large_step_batch['periods'][0] == 5
        
        # Test 3: Multiple periods with small step
        multi_batch = ta_indicators.ema_batch(close, (5, 15, 2))
        expected_periods = [5, 7, 9, 11, 13, 15]
        assert len(multi_batch['periods']) == len(expected_periods)
        assert all(p in multi_batch['periods'] for p in expected_periods)
    
    def test_ema_batch_consistency(self, test_data):
        """Test that batch results are consistent with individual calculations"""
        # Use smaller dataset for performance
        close = test_data['close'][:100]
        
        # Test various period combinations
        batch_result = ta_indicators.ema_batch(close, (10, 30, 10))
        
        for i, period in enumerate([10, 20, 30]):
            # Calculate single EMA for comparison
            single_result = ta_indicators.ema(close, period)
            batch_row = batch_result['values'][i]
            
            # They should be exactly the same
            np.testing.assert_array_almost_equal(
                batch_row, single_result, decimal=10,
                err_msg=f"Batch and single calculations differ for period {period}"
            )
