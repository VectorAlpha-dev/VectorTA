"""
Python binding tests for DEMA indicator.
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


class TestDema:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_dema_partial_params(self, test_data):
        """Test DEMA with partial parameters - mirrors check_dema_partial_params"""
        close = test_data['close']
        
        # Test with default period (30)
        result = ta_indicators.dema(close, 30)
        assert len(result) == len(close)
        
        # Test with custom period
        result_custom = ta_indicators.dema(close, 14)
        assert len(result_custom) == len(close)
    
    def test_dema_accuracy(self, test_data):
        """Test DEMA matches expected values from Rust tests - mirrors check_dema_accuracy"""
        close = test_data['close']
        
        # Use default period of 30
        result = ta_indicators.dema(close, period=30)
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        expected_last_5 = [
            59189.73193987478,
            59129.24920772847,
            59058.80282420511,
            59011.5555611042,
            58908.370159946775,
        ]
        
        assert_close(
            result[-5:], 
            expected_last_5,
            rtol=1e-6,
            msg="DEMA last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('dema', result, 'close', {'period': 30})
    
    def test_dema_default_candles(self, test_data):
        """Test DEMA with default parameters - mirrors check_dema_default_candles"""
        close = test_data['close']
        
        # Default period is 30
        result = ta_indicators.dema(close, 30)
        assert len(result) == len(close)
    
    def test_dema_zero_period(self):
        """Test DEMA fails with zero period - mirrors check_dema_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.dema(input_data, period=0)
    
    def test_dema_period_exceeds_length(self):
        """Test DEMA fails when period exceeds data length - mirrors check_dema_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough data"):
            ta_indicators.dema(data_small, period=10)
    
    def test_dema_very_small_dataset(self):
        """Test DEMA fails with insufficient data - mirrors check_dema_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough data"):
            ta_indicators.dema(single_point, period=9)
    
    def test_dema_empty_input(self):
        """Test DEMA fails with empty input - mirrors check_dema_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.dema(empty, period=30)
    
    def test_dema_reinput(self, test_data):
        """Test DEMA applied twice (re-input) - mirrors check_dema_reinput"""
        close = test_data['close']
        
        # First pass with period 80
        first_result = ta_indicators.dema(close, period=80)
        assert len(first_result) == len(close)
        
        # Second pass with period 60 - apply DEMA to DEMA output
        second_result = ta_indicators.dema(first_result, period=60)
        assert len(second_result) == len(first_result)
        
        # After warmup period (240), no NaN values should exist
        if len(second_result) > 240:
            assert not np.any(np.isnan(second_result[240:])), "Found unexpected NaN after warmup period"
    
    def test_dema_nan_handling(self, test_data):
        """Test DEMA handles NaN values correctly - mirrors check_dema_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.dema(close, period=30)
        assert len(result) == len(close)
        
        # After warmup period (240), no NaN values should exist
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"
    
    def test_dema_streaming(self, test_data):
        """Test DEMA streaming matches batch calculation - mirrors check_dema_streaming"""
        close = test_data['close']
        period = 30
        
        # Batch calculation
        batch_result = ta_indicators.dema(close, period=period)
        
        # Streaming calculation
        stream = ta_indicators.DemaStream(period=period)
        stream_values = []
        
        for price in close:
            result = stream.update(price)
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        # Compare batch vs streaming
        assert len(batch_result) == len(stream_values)
        
        # Compare values after the warmup period
        # Streaming has NaN for first period-1 values, batch may have values
        warmup = period - 1
        for i in range(warmup, len(batch_result)):
            b = batch_result[i]
            s = stream_values[i]
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, atol=1e-9, 
                        msg=f"DEMA streaming mismatch at index {i}")
    
    def test_dema_batch(self, test_data):
        """Test DEMA batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        result = ta_indicators.dema_batch(
            close,
            period_range=(30, 30, 0),  # Default period only
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        # Should have 1 combination (default params)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        # Extract the single row
        default_row = result['values'][0]
        expected_last_5 = [
            59189.73193987478,
            59129.24920772847,
            59058.80282420511,
            59011.5555611042,
            58908.370159946775,
        ]
        
        # Check last 5 values match
        assert_close(
            default_row[-5:],
            expected_last_5,
            rtol=1e-6,
            msg="DEMA batch default row mismatch"
        )
    
    def test_dema_all_nan_input(self):
        """Test DEMA with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.dema(all_nan, period=30)
    
    def test_dema_not_enough_valid_data(self):
        """Test DEMA with not enough valid data after NaN values"""
        # First two values are NaN, not enough valid data for period 3
        data = np.array([np.nan, np.nan, 1.0, 2.0])
        
        with pytest.raises(ValueError, match="Not enough valid data"):
            ta_indicators.dema(data, period=3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])