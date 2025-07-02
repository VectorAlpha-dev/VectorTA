"""
Python binding tests for CWMA indicator.
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


class TestCwma:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_cwma_partial_params(self, test_data):
        """Test CWMA with partial parameters - mirrors check_cwma_partial_params"""
        close = test_data['close']
        
        # Test with default period (14)
        result = ta_indicators.cwma(close, 14)
        assert len(result) == len(close)
    
    def test_cwma_accuracy(self, test_data):
        """Test CWMA matches expected values from Rust tests - mirrors check_cwma_accuracy"""
        close = test_data['close']
        
        # Use default period of 14
        result = ta_indicators.cwma(close, period=14)
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        expected_last_5 = [
            59224.641237300435,
            59213.64831277214,
            59171.21190130624,
            59167.01279027576,
            59039.413552249636,
        ]
        
        assert_close(
            result[-5:], 
            expected_last_5,
            rtol=1e-9,
            msg="CWMA last 5 values mismatch"
        )
    
    def test_cwma_default_candles(self, test_data):
        """Test CWMA with default parameters - mirrors check_cwma_default_candles"""
        close = test_data['close']
        
        # Default period is 14
        result = ta_indicators.cwma(close, 14)
        assert len(result) == len(close)
    
    def test_cwma_zero_period(self):
        """Test CWMA fails with zero period - mirrors check_cwma_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.cwma(input_data, period=0)
    
    def test_cwma_period_exceeds_length(self):
        """Test CWMA fails when period exceeds data length - mirrors check_cwma_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.cwma(data_small, period=10)
    
    def test_cwma_very_small_dataset(self):
        """Test CWMA fails with insufficient data - mirrors check_cwma_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.cwma(single_point, period=9)
    
    def test_cwma_empty_input(self):
        """Test CWMA fails with empty input - mirrors check_cwma_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.cwma(empty, period=9)
    
    def test_cwma_reinput(self, test_data):
        """Test CWMA applied twice (re-input) - mirrors check_cwma_reinput"""
        close = test_data['close']
        
        # First pass with period 80
        first_result = ta_indicators.cwma(close, period=80)
        assert len(first_result) == len(close)
        
        # Second pass with period 60 - apply CWMA to CWMA output
        second_result = ta_indicators.cwma(first_result, period=60)
        assert len(second_result) == len(first_result)
        
        # After warmup period (240), no NaN values should exist
        if len(second_result) > 240:
            assert not np.any(np.isnan(second_result[240:])), "Found unexpected NaN after warmup period"
    
    def test_cwma_nan_handling(self, test_data):
        """Test CWMA handles NaN values correctly - mirrors check_cwma_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.cwma(close, period=9)
        assert len(result) == len(close)
        
        # After warmup period (240), no NaN values should exist
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"
        
        # First period-1 values should be NaN
        assert np.all(np.isnan(result[:8])), "Expected NaN in warmup period"
    
    def test_cwma_streaming(self, test_data):
        """Test CWMA streaming matches batch calculation - mirrors check_cwma_streaming"""
        close = test_data['close']
        period = 9
        
        # Batch calculation
        batch_result = ta_indicators.cwma(close, period=period)
        
        # Streaming calculation
        stream = ta_indicators.CwmaStream(period=period)
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
                        msg=f"CWMA streaming mismatch at index {i}")
    
    def test_cwma_batch(self, test_data):
        """Test CWMA batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        result = ta_indicators.cwma_batch(
            close,
            period_range=(14, 14, 0),  # Default period only
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        # Should have 1 combination (default params)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        # Extract the single row
        default_row = result['values'][0]
        expected_last_5 = [
            59224.641237300435,
            59213.64831277214,
            59171.21190130624,
            59167.01279027576,
            59039.413552249636,
        ]
        
        # Check last 5 values match
        assert_close(
            default_row[-5:],
            expected_last_5,
            rtol=1e-9,
            msg="CWMA batch default row mismatch"
        )
    
    def test_cwma_all_nan_input(self):
        """Test CWMA with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.cwma(all_nan, period=9)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])