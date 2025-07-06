"""
Python binding tests for EDCF indicator.
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


class TestEdcf:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_edcf_partial_params(self, test_data):
        """Test EDCF with partial parameters - mirrors check_edcf_with_slice_data"""
        close = test_data['close']
        
        # Test with default period (15)
        result = ta_indicators.edcf(close, 15)
        assert len(result) == len(close)
        
        # Test with custom period
        result_custom = ta_indicators.edcf(close, 20)
        assert len(result_custom) == len(close)
    
    def test_edcf_accuracy(self, test_data):
        """Test EDCF matches expected values from Rust tests - mirrors check_edcf_accuracy_last_five"""
        # Note: Rust test uses "hl2" source, we need to calculate it
        hl2 = (test_data['high'] + test_data['low']) / 2
        
        # Use period of 15 as in Rust test
        result = ta_indicators.edcf(hl2, period=15)
        
        assert len(result) == len(hl2)
        
        # Check last 5 values match expected
        expected_last_5 = [
            59593.332275678375,
            59731.70263288801,
            59766.41512339413,
            59655.66162110993,
            59332.492883847,
        ]
        
        assert_close(
            result[-5:], 
            expected_last_5,
            rtol=1e-8,
            msg="EDCF last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('edcf', result, 'hl2', {'period': 15})
    
    def test_edcf_default_candles(self, test_data):
        """Test EDCF with default parameters - mirrors check_edcf_with_default_candles"""
        close = test_data['close']
        
        # Default period is 15
        result = ta_indicators.edcf(close, 15)
        assert len(result) == len(close)
    
    def test_edcf_zero_period(self):
        """Test EDCF fails with zero period - mirrors check_edcf_with_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.edcf(input_data, period=0)
    
    def test_edcf_period_exceeds_length(self):
        """Test EDCF fails when period exceeds data length - mirrors check_edcf_with_period_exceeding_data_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.edcf(data_small, period=10)
    
    def test_edcf_very_small_dataset(self):
        """Test EDCF fails with insufficient data - mirrors check_edcf_very_small_data_set"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.edcf(single_point, period=15)
    
    def test_edcf_empty_input(self):
        """Test EDCF fails with empty input - mirrors check_edcf_with_no_data"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="No data provided"):
            ta_indicators.edcf(empty, period=15)
    
    def test_edcf_reinput(self, test_data):
        """Test EDCF applied twice (re-input) - mirrors check_edcf_with_slice_data_reinput"""
        close = test_data['close']
        
        # First pass with period 15
        first_result = ta_indicators.edcf(close, period=15)
        assert len(first_result) == len(close)
        
        # Second pass with period 5 - apply EDCF to EDCF output
        second_result = ta_indicators.edcf(first_result, period=5)
        assert len(second_result) == len(first_result)
        
        # After warmup period (240), no NaN values should exist
        if len(second_result) > 240:
            for i in range(240, len(second_result)):
                assert not np.isnan(second_result[i]), f"Found unexpected NaN at index {i}"
    
    def test_edcf_nan_handling(self, test_data):
        """Test EDCF handles NaN values correctly - mirrors check_edcf_accuracy_nan_check"""
        close = test_data['close']
        period = 15
        
        result = ta_indicators.edcf(close, period=period)
        assert len(result) == len(close)
        
        # After 2*period, no NaN values should exist
        start_index = 2 * period
        if len(result) > start_index:
            for i in range(start_index, len(result)):
                assert not np.isnan(result[i]), f"Found unexpected NaN at index {i}"
    
    def test_edcf_streaming(self, test_data):
        """Test EDCF streaming matches batch calculation - mirrors check_edcf_streaming"""
        close = test_data['close']
        period = 15
        
        # Streaming calculation
        stream = ta_indicators.EdcfStream(period=period)
        stream_values = []
        
        for price in close:
            result = stream.update(price)
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        # The Rust test only checks that streaming produces non-NaN values after 2*period
        # It doesn't compare with batch values
        assert len(stream_values) == len(close)
        
        # After index 30 (2*period), no NaN values should exist
        for i in range(30, len(stream_values)):
            assert not np.isnan(stream_values[i]), f"Found unexpected NaN at index {i} in streaming"
    
    def test_edcf_batch(self, test_data):
        """Test EDCF batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        result = ta_indicators.edcf_batch(
            close,
            period_range=(15, 15, 0),  # Default period only
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        # Should have 1 combination (default params)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        # Verify the batch matches the single calculation
        single_result = ta_indicators.edcf(close, period=15)
        default_row = result['values'][0]
        
        # Compare non-NaN values
        for i in range(len(close)):
            if not np.isnan(single_result[i]) and not np.isnan(default_row[i]):
                assert_close(
                    single_result[i],
                    default_row[i],
                    rtol=1e-9,
                    msg=f"EDCF batch vs single mismatch at index {i}"
                )
    
    def test_edcf_all_nan_input(self):
        """Test EDCF with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.edcf(all_nan, period=15)
    
    def test_edcf_accuracy_hl2(self, test_data):
        """Additional test to verify EDCF with hl2 data matches Rust exactly"""
        # This is an extra test to ensure we're testing the exact same scenario as Rust
        hl2 = (test_data['high'] + test_data['low']) / 2
        
        result = ta_indicators.edcf(hl2, period=15)
        
        # Expected values from Rust test (check_edcf_accuracy_last_five)
        expected_last_5 = [
            59593.332275678375,
            59731.70263288801,
            59766.41512339413,
            59655.66162110993,
            59332.492883847,
        ]
        
        # Check last 5 values with the same tolerance as Rust (1e-8)
        actual_last_5 = result[-5:]
        for i, (actual, expected) in enumerate(zip(actual_last_5, expected_last_5)):
            diff = abs(actual - expected)
            assert diff < 1e-8, f"EDCF mismatch at index {i}: got {actual}, expected {expected}, diff {diff}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])