"""
Python binding tests for MAC-Z VWAP indicator.
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

class TestMacz:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_macz_partial_params(self, test_data):
        """Test MAC-Z with partial parameters (default values) - mirrors check_macz_partial_params"""
        close = test_data['close']
        volume = test_data.get('volume')  # May be None
        
        # Test with minimum required params
        result = ta_indicators.macz(close, volume)
        assert len(result) == len(close)
        
        # Test with some explicit params
        result = ta_indicators.macz(
            close, 
            volume,
            fast_length=12,
            slow_length=26
        )
        assert len(result) == len(close)

    def test_macz_accuracy(self, test_data):
        """Test MAC-Z matches expected values from Rust tests - mirrors check_macz_accuracy"""
        close = test_data['close']
        volume = test_data.get('volume')
        expected = EXPECTED_OUTPUTS['macz']
        
        result = ta_indicators.macz(
            close,
            volume,
            fast_length=expected['default_params']['fast_length'],
            slow_length=expected['default_params']['slow_length'],
            signal_length=expected['default_params']['signal_length'],
            lengthz=expected['default_params']['lengthz'],
            length_stdev=expected['default_params']['length_stdev'],
            a=expected['default_params']['a'],
            b=expected['default_params']['b'],
            use_lag=expected['default_params']['use_lag'],
            gamma=expected['default_params']['gamma']
        )
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected with tight tolerance
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-9,
            atol=1e-10,
            msg="MAC-Z last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('macz', result, 'close', expected['default_params'])

    def test_macz_default_candles(self, test_data):
        """Test MAC-Z with default parameters - mirrors check_macz_default_candles"""
        close = test_data['close']
        volume = test_data.get('volume')
        
        # Default params from Rust
        result = ta_indicators.macz(
            close,
            volume,
            fast_length=20,
            slow_length=30,
            signal_length=10,
            lengthz=20,
            length_stdev=20,
            a=2.0,
            b=-1.0,
            use_lag=False,
            gamma=0.02
        )
        assert len(result) == len(close)

    def test_macz_zero_fast_length(self):
        """Test MAC-Z fails with zero fast_length - mirrors check_macz_zero_fast_length"""
        input_data = np.array([10.0, 20.0, 30.0])
        volume = np.array([100.0, 200.0, 300.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.macz(input_data, volume, fast_length=0)

    def test_macz_period_exceeds_length(self):
        """Test MAC-Z fails when period exceeds data length - mirrors check_macz_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        volume_small = np.array([100.0, 200.0, 300.0])
        
        with pytest.raises(ValueError, match="Not enough valid data"):
            ta_indicators.macz(data_small, volume_small, slow_length=10)

    def test_macz_very_small_dataset(self):
        """Test MAC-Z fails with insufficient data - mirrors check_macz_very_small_dataset"""
        single_point = np.array([42.0])
        single_volume = np.array([100.0])
        
        with pytest.raises(ValueError, match="Not enough valid data"):
            ta_indicators.macz(single_point, single_volume)

    def test_macz_empty_input(self):
        """Test MAC-Z fails with empty input - mirrors check_macz_empty_input"""
        empty = np.array([])
        empty_volume = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.macz(empty, empty_volume)

    def test_macz_invalid_a(self):
        """Test MAC-Z fails with invalid A constant - mirrors check_macz_invalid_a"""
        data = np.array([1.0, 2.0, 3.0] * 20)
        volume = np.array([100.0, 200.0, 300.0] * 20)
        
        # A out of range (> 2.0)
        with pytest.raises(ValueError, match="A out of range"):
            ta_indicators.macz(data, volume, a=3.0)
        
        # A out of range (< -2.0)
        with pytest.raises(ValueError, match="A out of range"):
            ta_indicators.macz(data, volume, a=-3.0)
    
    def test_macz_invalid_b(self):
        """Test MAC-Z fails with invalid B constant - mirrors check_macz_invalid_b"""
        data = np.array([1.0, 2.0, 3.0] * 20)
        volume = np.array([100.0, 200.0, 300.0] * 20)
        
        # B out of range (> 2.0)
        with pytest.raises(ValueError, match="B out of range"):
            ta_indicators.macz(data, volume, b=3.0)
        
        # B out of range (< -2.0)
        with pytest.raises(ValueError, match="B out of range"):
            ta_indicators.macz(data, volume, b=-3.0)
    
    def test_macz_invalid_gamma(self):
        """Test MAC-Z fails with invalid gamma - mirrors check_macz_invalid_gamma"""
        data = np.array([1.0, 2.0, 3.0] * 20)
        volume = np.array([100.0, 200.0, 300.0] * 20)
        
        # Gamma out of range (>= 1.0)
        with pytest.raises(ValueError, match="Invalid gamma"):
            ta_indicators.macz(data, volume, gamma=1.5)
        
        # Gamma out of range (< 0.0)
        with pytest.raises(ValueError, match="Invalid gamma"):
            ta_indicators.macz(data, volume, gamma=-0.1)

    def test_macz_nan_handling(self, test_data):
        """Test MAC-Z handles NaN values correctly - mirrors check_macz_nan_handling"""
        close = test_data['close']
        volume = test_data.get('volume')
        expected = EXPECTED_OUTPUTS['macz']
        
        result = ta_indicators.macz(
            close,
            volume,
            fast_length=expected['default_params']['fast_length'],
            slow_length=expected['default_params']['slow_length'],
            signal_length=expected['default_params']['signal_length'],
            lengthz=expected['default_params']['lengthz'],
            length_stdev=expected['default_params']['length_stdev'],
            a=expected['default_params']['a'],
            b=expected['default_params']['b'],
            use_lag=expected['default_params']['use_lag'],
            gamma=expected['default_params']['gamma']
        )
        
        assert len(result) == len(close)
        
        # After warmup period, no NaN values should exist
        warmup = expected['warmup_period']
        if len(result) > warmup:
            assert not np.any(np.isnan(result[warmup:])), "Found unexpected NaN after warmup period"
        
        # First warmup values should be NaN
        assert np.all(np.isnan(result[:warmup])), "Expected NaN in warmup period"

    def test_macz_streaming(self, test_data):
        """Test MAC-Z streaming matches batch calculation - mirrors check_macz_streaming"""
        close = test_data['close']
        volume = test_data.get('volume')
        expected = EXPECTED_OUTPUTS['macz']['default_params']
        
        # Batch calculation
        batch_result = ta_indicators.macz(
            close,
            volume,
            fast_length=expected['fast_length'],
            slow_length=expected['slow_length'],
            signal_length=expected['signal_length'],
            lengthz=expected['lengthz'],
            length_stdev=expected['length_stdev'],
            a=expected['a'],
            b=expected['b'],
            use_lag=expected['use_lag'],
            gamma=expected['gamma']
        )
        
        # Streaming calculation
        stream = ta_indicators.MaczStream(
            fast_length=expected['fast_length'],
            slow_length=expected['slow_length'],
            signal_length=expected['signal_length'],
            lengthz=expected['lengthz'],
            length_stdev=expected['length_stdev'],
            a=expected['a'],
            b=expected['b'],
            use_lag=expected['use_lag'],
            gamma=expected['gamma']
        )
        stream_values = []
        
        for i, price in enumerate(close):
            vol = volume[i] if volume is not None else None
            result = stream.update(price, vol)
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        # Compare batch vs streaming
        assert len(batch_result) == len(stream_values)
        
        # Compare values where both are not NaN
        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            # Use relaxed tolerance for streaming due to accumulated rounding over thousands of iterations
            # The relative tolerance of 1e-5 is reasonable for financial calculations
            assert_close(b, s, rtol=1e-5, atol=1e-8, 
                        msg=f"MAC-Z streaming mismatch at index {i}")
    
    def test_macz_batch(self, test_data):
        """Test MAC-Z batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        volume = test_data.get('volume')
        
        result = ta_indicators.macz_batch(
            close,
            volume,
            fast_length_range=(12, 12, 0),  # Default fast only
            slow_length_range=(25, 25, 0),  # Default slow only
            signal_length_range=(9, 9, 0),  # Default signal only
            lengthz_range=(20, 20, 0),  # Default lengthz only
            length_stdev_range=(25, 25, 0),  # Default stdev only
            a_range=(1.0, 1.0, 0.0),  # Default A only
            b_range=(1.0, 1.0, 0.0),  # Default B only
            use_lag_range=(False, False, False),  # Default use_lag only
            gamma_range=(0.02, 0.02, 0.0)  # Default gamma only
        )
        
        assert 'values' in result
        assert 'fast_lengths' in result
        assert 'slow_lengths' in result
        assert 'signal_lengths' in result
        
        # Should have 1 combination (default params)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        # Extract the single row
        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['macz']['last_5_values']
        
        # Check last 5 values match with tight tolerance
        assert_close(
            default_row[-5:],
            expected,
            rtol=1e-9,
            atol=1e-10,
            msg="MAC-Z batch default row mismatch"
        )
    
    def test_macz_all_nan_input(self):
        """Test MAC-Z with all NaN values"""
        all_nan = np.full(100, np.nan)
        all_nan_volume = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.macz(all_nan, all_nan_volume)
    
    def test_macz_with_volume(self, test_data):
        """Test MAC-Z with actual volume data"""
        close = test_data['close']
        volume = test_data.get('volume', np.ones_like(close) * 1000.0)
        expected = EXPECTED_OUTPUTS['macz']
        
        result = ta_indicators.macz(
            close,
            volume,
            fast_length=expected['default_params']['fast_length'],
            slow_length=expected['default_params']['slow_length'],
            signal_length=expected['default_params']['signal_length'],
            lengthz=expected['default_params']['lengthz'],
            length_stdev=expected['default_params']['length_stdev'],
            a=expected['default_params']['a'],
            b=expected['default_params']['b'],
            use_lag=expected['default_params']['use_lag'],
            gamma=expected['default_params']['gamma']
        )
        
        assert len(result) == len(close)
        assert not np.all(np.isnan(result)), "Result should not be all NaN"



if __name__ == '__main__':
    pytest.main([__file__, '-v'])