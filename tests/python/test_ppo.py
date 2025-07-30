"""
Python binding tests for PPO indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
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


class TestPpo:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_ppo_partial_params(self, test_data):
        """Test PPO with partial parameters (None values) - mirrors check_ppo_partial_params"""
        close = test_data['close']
        
        # Test with all default params (None)
        result = ta_indicators.ppo(close)  # Using defaults
        assert len(result) == len(close)
    
    def test_ppo_accuracy(self, test_data):
        """Test PPO matches expected values from Rust tests - mirrors check_ppo_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['ppo']
        
        result = ta_indicators.ppo(
            close,
            fast_period=expected['default_params']['fast_period'],
            slow_period=expected['default_params']['slow_period'],
            ma_type=expected['default_params']['ma_type']
        )
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-8,
            msg="PPO last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('ppo', result, 'close', expected['default_params'])
    
    def test_ppo_default_candles(self, test_data):
        """Test PPO with default parameters - mirrors check_ppo_default_candles"""
        close = test_data['close']
        
        # Default params: fast_period=12, slow_period=26, ma_type='sma'
        result = ta_indicators.ppo(close, fast_period=12, slow_period=26, ma_type='sma')
        assert len(result) == len(close)
    
    def test_ppo_zero_period(self):
        """Test PPO fails with zero period - mirrors check_ppo_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.ppo(input_data, fast_period=0, slow_period=26, ma_type='sma')
    
    def test_ppo_period_exceeds_length(self):
        """Test PPO fails when period exceeds data length - mirrors check_ppo_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.ppo(data_small, fast_period=12, slow_period=26, ma_type='sma')
    
    def test_ppo_very_small_dataset(self):
        """Test PPO fails with insufficient data - mirrors check_ppo_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.ppo(single_point, fast_period=12, slow_period=26, ma_type='sma')
    
    def test_ppo_empty_input(self):
        """Test PPO fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.ppo(empty, fast_period=12, slow_period=26, ma_type='sma')
    
    def test_ppo_nan_handling(self, test_data):
        """Test PPO handles NaN values correctly - mirrors check_ppo_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.ppo(close, fast_period=12, slow_period=26, ma_type='sma')
        assert len(result) == len(close)
        
        # After warmup period (30), no NaN values should exist
        if len(result) > 30:
            assert not np.any(np.isnan(result[30:])), "Found unexpected NaN after warmup period"
    
    def test_ppo_streaming(self, test_data):
        """Test PPO streaming matches batch calculation - mirrors check_ppo_streaming"""
        close = test_data['close']
        fast_period = 12
        slow_period = 26
        ma_type = 'sma'
        
        # Batch calculation
        batch_result = ta_indicators.ppo(close, fast_period=fast_period, slow_period=slow_period, ma_type=ma_type)
        
        # Streaming calculation
        stream = ta_indicators.PpoStream(fast_period=fast_period, slow_period=slow_period, ma_type=ma_type)
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
                        msg=f"PPO streaming mismatch at index {i}")
    
    def test_ppo_batch(self, test_data):
        """Test PPO batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        result = ta_indicators.ppo_batch(
            close,
            fast_period_range=(12, 12, 0),  # Default fast period only
            slow_period_range=(26, 26, 0),  # Default slow period only
            ma_type='sma'  # Default ma_type
        )
        
        assert 'values' in result
        assert 'fast_periods' in result
        assert 'slow_periods' in result
        assert 'ma_types' in result
        
        # Check dimensions
        assert result['values'].shape[0] == 1  # Single parameter combination
        assert result['values'].shape[1] == len(close)
        
        # Check parameters
        assert result['fast_periods'][0] == 12
        assert result['slow_periods'][0] == 26
        assert result['ma_types'][0] == 'sma'
        
        # Compare with single calculation
        single_result = ta_indicators.ppo(close, fast_period=12, slow_period=26, ma_type='sma')
        assert_close(
            result['values'][0], 
            single_result, 
            rtol=1e-9,
            msg="PPO batch vs single calculation mismatch"
        )
    
    def test_ppo_batch_multiple_params(self, test_data):
        """Test PPO batch processing with multiple parameter combinations"""
        close = test_data['close']
        
        result = ta_indicators.ppo_batch(
            close,
            fast_period_range=(10, 14, 2),  # 10, 12, 14
            slow_period_range=(24, 28, 2),  # 24, 26, 28
            ma_type='ema'
        )
        
        # Should have 3 * 3 = 9 combinations
        assert result['values'].shape[0] == 9
        assert result['values'].shape[1] == len(close)
        assert len(result['fast_periods']) == 9
        assert len(result['slow_periods']) == 9
        assert len(result['ma_types']) == 9
        
        # Verify parameter combinations
        expected_combinations = [
            (10, 24), (10, 26), (10, 28),
            (12, 24), (12, 26), (12, 28),
            (14, 24), (14, 26), (14, 28)
        ]
        
        for i, (fast, slow) in enumerate(expected_combinations):
            assert result['fast_periods'][i] == fast
            assert result['slow_periods'][i] == slow
            assert result['ma_types'][i] == 'ema'
    
    def test_ppo_kernel_parameter(self, test_data):
        """Test PPO with different kernel parameters"""
        close = test_data['close']
        
        # Test with auto kernel (default)
        result_auto = ta_indicators.ppo(close, kernel=None)
        assert len(result_auto) == len(close)
        
        # Test with scalar kernel
        result_scalar = ta_indicators.ppo(close, kernel='scalar')
        assert len(result_scalar) == len(close)
        
        # Results should be very close regardless of kernel
        assert_close(
            result_auto, 
            result_scalar, 
            rtol=1e-9,
            msg="PPO kernel results mismatch"
        )
    
    def test_ppo_different_ma_types(self, test_data):
        """Test PPO with different moving average types"""
        close = test_data['close']
        
        ma_types = ['sma', 'ema', 'wma']
        results = {}
        
        for ma_type in ma_types:
            results[ma_type] = ta_indicators.ppo(
                close, 
                fast_period=12, 
                slow_period=26, 
                ma_type=ma_type
            )
            assert len(results[ma_type]) == len(close)
        
        # Results should be different for different MA types
        assert not np.array_equal(results['sma'], results['ema'])
        assert not np.array_equal(results['sma'], results['wma'])
        assert not np.array_equal(results['ema'], results['wma'])


if __name__ == "__main__":
    # Run tests in this file
    pytest.main([__file__, "-v"])