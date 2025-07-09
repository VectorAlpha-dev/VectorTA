"""
Python binding tests for TrendFlex indicator.
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


class TestTrendFlex:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_trendflex_partial_params(self, test_data):
        """Test TrendFlex with partial parameters (None values) - mirrors check_trendflex_partial_params"""
        close = test_data['close']
        
        # Test with all default params (None)
        result = ta_indicators.trendflex(close)  # Using defaults
        assert len(result) == len(close)
    
    def test_trendflex_accuracy(self, test_data):
        """Test TrendFlex matches expected values from Rust tests - mirrors check_trendflex_accuracy"""
        close = test_data['close']
        
        # First, ensure we have expected outputs for trendflex
        if 'trendflex' not in EXPECTED_OUTPUTS:
            # Add expected outputs for trendflex
            EXPECTED_OUTPUTS['trendflex'] = {
                'default_params': {'period': 20},
                'last_5_values': [
                    -0.19724678008015128,
                    -0.1238001236481444,
                    -0.10515389737087717,
                    -0.1149541079904878,
                    -0.16006869484450567,
                ],
                'reinput_last_5': []  # Will be populated during reinput test
            }
        
        expected = EXPECTED_OUTPUTS['trendflex']
        
        result = ta_indicators.trendflex(
            close,
            period=expected['default_params']['period']
        )
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-8,
            msg="TrendFlex last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('trendflex', result, 'close', expected['default_params'])
    
    def test_trendflex_default_candles(self, test_data):
        """Test TrendFlex with default parameters - mirrors check_trendflex_default_candles"""
        close = test_data['close']
        
        # Default params: period=20
        result = ta_indicators.trendflex(close)
        assert len(result) == len(close)
    
    def test_trendflex_zero_period(self):
        """Test TrendFlex fails with zero period - mirrors check_trendflex_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="period = 0"):
            ta_indicators.trendflex(input_data, period=0)
    
    def test_trendflex_period_exceeds_length(self):
        """Test TrendFlex fails when period exceeds data length - mirrors check_trendflex_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="period > data len"):
            ta_indicators.trendflex(data_small, period=10)
    
    def test_trendflex_very_small_dataset(self):
        """Test TrendFlex fails with insufficient data - mirrors check_trendflex_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="period > data len"):
            ta_indicators.trendflex(single_point, period=9)
    
    def test_trendflex_empty_input(self):
        """Test TrendFlex fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="No data provided"):
            ta_indicators.trendflex(empty)
    
    def test_trendflex_reinput(self, test_data):
        """Test TrendFlex applied twice (re-input) - mirrors check_trendflex_reinput"""
        close = test_data['close']
        
        # First pass
        first_result = ta_indicators.trendflex(close, period=20)
        assert len(first_result) == len(close)
        
        # Second pass - apply TrendFlex to TrendFlex output
        second_result = ta_indicators.trendflex(first_result, period=10)
        assert len(second_result) == len(first_result)
        
        # After warmup period, no NaN values should exist
        if len(second_result) > 240:
            assert not np.any(np.isnan(second_result[240:])), "Found unexpected NaN after warmup period"
    
    def test_trendflex_nan_handling(self, test_data):
        """Test TrendFlex handles NaN values correctly - mirrors check_trendflex_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.trendflex(close, period=20)
        assert len(result) == len(close)
        
        # After warmup period (240), no NaN values should exist
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"
        
        # First 19 values should be NaN (period-1)
        assert np.all(np.isnan(result[:19])), "Expected NaN in warmup period"
    
    def test_trendflex_streaming(self, test_data):
        """Test TrendFlex streaming matches batch calculation - mirrors check_trendflex_streaming"""
        close = test_data['close']
        period = 20
        
        # Batch calculation
        batch_result = ta_indicators.trendflex(close, period=period)
        
        # Streaming calculation
        stream = ta_indicators.TrendFlexStream(period=period)
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
                        msg=f"TrendFlex streaming mismatch at index {i}")
    
    def test_trendflex_batch(self, test_data):
        """Test TrendFlex batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        result = ta_indicators.trendflex_batch(
            close,
            period_range=(20, 20, 0),  # Default period only
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        # Should have 1 combination (default params)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        # Extract the single row
        default_row = result['values'][0]
        expected = [
            -0.19724678008015128,
            -0.1238001236481444,
            -0.10515389737087717,
            -0.1149541079904878,
            -0.16006869484450567,
        ]
        
        # Check last 5 values match
        assert_close(
            default_row[-5:],
            expected,
            rtol=1e-8,
            msg="TrendFlex batch default row mismatch"
        )
    
    def test_trendflex_all_nan_input(self):
        """Test TrendFlex with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.trendflex(all_nan, period=20)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])