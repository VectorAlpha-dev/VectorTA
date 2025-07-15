"""
Python binding tests for EPMA indicator.
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


class TestEpma:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_epma_partial_params(self, test_data):
        """Test EPMA with partial parameters (None values) - mirrors check_epma_partial_params"""
        close = test_data['close']
        
        # Test with all default params (None)
        result = ta_indicators.epma(close, None, None)
        assert len(result) == len(close)
        
        # Test with partial custom parameters
        result = ta_indicators.epma(close, 15, None)  # Custom period, default offset
        assert len(result) == len(close)
        
        result = ta_indicators.epma(close, None, 3)  # Default period, custom offset
        assert len(result) == len(close)
    
    def test_epma_accuracy(self, test_data):
        """Test EPMA matches expected values from Rust tests - mirrors check_epma_accuracy"""
        close = test_data['close']
        
        # Using default parameters (period=11, offset=4)
        result = ta_indicators.epma(close, 11, 4)
        
        assert len(result) == len(close)
        
        # Expected values from Rust test with period=11, offset=4
        expected_last_5 = [59174.48, 59201.04, 59167.60, 59200.32, 59117.04]
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected_last_5,
            rtol=0,
            atol=0.1,  # Using 1e-1 tolerance as in Rust test
            msg="EPMA last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('epma', result, 'close', {'period': 11, 'offset': 4})
    
    def test_epma_default_candles(self, test_data):
        """Test EPMA with default parameters - mirrors check_epma_default_candles"""
        close = test_data['close']
        
        # Default params: period=11, offset=4
        result = ta_indicators.epma(close, 11, 4)
        assert len(result) == len(close)
    
    def test_epma_zero_period(self):
        """Test EPMA fails with zero period - mirrors check_epma_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        # With period=0 and default offset=4, it will fail with "Invalid offset"
        with pytest.raises(ValueError, match="Invalid offset"):
            ta_indicators.epma(input_data, period=0, offset=None)
        
        # With period=0 and offset=0, it will fail with "Invalid offset" because 0 >= 0
        with pytest.raises(ValueError, match="Invalid offset"):
            ta_indicators.epma(input_data, period=0, offset=0)
    
    def test_epma_period_exceeds_length(self):
        """Test EPMA fails when period exceeds data length - mirrors check_epma_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.epma(data_small, period=10, offset=None)
    
    def test_epma_very_small_dataset(self):
        """Test EPMA fails with insufficient data - mirrors check_epma_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.epma(single_point, period=9, offset=None)
    
    def test_epma_empty_input(self):
        """Test EPMA fails with empty input - mirrors check_epma_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.epma(empty, period=None, offset=None)
    
    def test_epma_invalid_offset(self):
        """Test EPMA fails with invalid offset - mirrors check_epma_invalid_offset"""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Offset >= period
        with pytest.raises(ValueError, match="Invalid offset"):
            ta_indicators.epma(data, period=3, offset=3)
        
        with pytest.raises(ValueError, match="Invalid offset"):
            ta_indicators.epma(data, period=3, offset=4)
    
    def test_epma_reinput(self, test_data):
        """Test EPMA applied twice (re-input) - mirrors check_epma_reinput"""
        close = test_data['close']
        
        # First pass
        first_result = ta_indicators.epma(close, period=9, offset=None)
        assert len(first_result) == len(close)
        
        # Second pass - apply EPMA to EPMA output with period=3, offset=0
        second_result = ta_indicators.epma(first_result, period=3, offset=0)
        assert len(second_result) == len(first_result)
    
    def test_epma_nan_handling(self, test_data):
        """Test EPMA handles NaN values correctly - mirrors check_epma_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.epma(close, period=11, offset=4)
        assert len(result) == len(close)
        
        # After warmup period (240), no NaN values should exist
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"
        
        # First period+offset+1 values should be NaN (11+4+1=16)
        assert np.all(np.isnan(result[:16])), "Expected NaN in warmup period"
    
    def test_epma_streaming(self, test_data):
        """Test EPMA streaming matches batch calculation - mirrors check_epma_streaming"""
        close = test_data['close']
        period = 11
        offset = 4
        
        # Batch calculation
        batch_result = ta_indicators.epma(close, period=period, offset=offset)
        
        # Streaming calculation
        stream = ta_indicators.EpmaStream(period=period, offset=offset)
        stream_values = []
        
        for price in close:
            result = stream.update(price)
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        # Compare batch vs streaming
        assert len(batch_result) == len(stream_values)
        
        # Compare values where both are not NaN
        # Note: EPMA streaming returns the input value during warmup, not NaN
        warmup_period = period + offset + 1
        for i in range(warmup_period, len(batch_result)):
            if np.isnan(batch_result[i]) and np.isnan(stream_values[i]):
                continue
            assert_close(batch_result[i], stream_values[i], rtol=1e-9, atol=1e-9, 
                        msg=f"EPMA streaming mismatch at index {i}")
    
    def test_epma_batch_single_params(self, test_data):
        """Test EPMA batch processing with single parameter set - mirrors check_batch_default_row"""
        close = test_data['close']
        
        result = ta_indicators.epma_batch(
            close,
            period_range=(11, 11, 0),  # Default period only
            offset_range=(4, 4, 0)      # Default offset only
        )
        
        assert 'values' in result
        assert 'periods' in result
        assert 'offsets' in result
        
        # Should have 1 combination
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        # Extract the single row
        default_row = result['values'][0]
        expected_last_5 = [59174.48, 59201.04, 59167.60, 59200.32, 59117.04]
        
        # Check last 5 values match
        assert_close(
            default_row[-5:],
            expected_last_5,
            rtol=0,
            atol=0.1,
            msg="EPMA batch default row mismatch"
        )
    
    def test_epma_batch_multiple_params(self, test_data):
        """Test EPMA batch processing with multiple parameter combinations"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        result = ta_indicators.epma_batch(
            close,
            period_range=(5, 9, 2),    # periods: 5, 7, 9
            offset_range=(1, 3, 1)     # offsets: 1, 2, 3
        )
        
        # Should have 3 periods x 3 offsets = 9 combinations
        assert result['values'].shape[0] == 9
        assert result['values'].shape[1] == 100
        
        # Verify metadata
        assert len(result['periods']) == 9
        assert len(result['offsets']) == 9
        
        # Check first combination (period=5, offset=1)
        single_result = ta_indicators.epma(close, period=5, offset=1)
        # The batch values should match the single calculation
        # Use a more relaxed tolerance due to potential floating point differences
        assert_close(
            result['values'][0],
            single_result,
            rtol=1e-6,
            msg="First batch row mismatch"
        )
    
    def test_epma_batch_warmup_validation(self, test_data):
        """Test batch warmup period handling"""
        close = test_data['close'][:50]
        
        result = ta_indicators.epma_batch(
            close,
            period_range=(5, 10, 5),   # periods: 5, 10
            offset_range=(2, 4, 2)     # offsets: 2, 4
        )
        
        # Should have 2 periods x 2 offsets = 4 rows
        assert result['values'].shape == (4, 50)
        
        # Check warmup periods for each combination
        # Row 0: period=5, offset=2, warmup=5+2+1=8
        assert np.all(np.isnan(result['values'][0][:8]))
        assert not np.any(np.isnan(result['values'][0][8:]))
        
        # Row 3: period=10, offset=4, warmup=10+4+1=15
        assert np.all(np.isnan(result['values'][3][:15]))
        assert not np.any(np.isnan(result['values'][3][15:]))
    
    def test_epma_all_nan_input(self):
        """Test EPMA with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.epma(all_nan, period=11, offset=4)
    
    def test_epma_period_too_small(self):
        """Test EPMA fails with period < 2"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # With period=1 and default offset=4, it will fail with "Invalid offset"
        with pytest.raises(ValueError, match="Invalid offset"):
            ta_indicators.epma(data, period=1, offset=None)
        
        # With period=1 and offset=0, it will fail with "Invalid period"
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.epma(data, period=1, offset=0)
    
    def test_epma_not_enough_valid_data(self):
        """Test EPMA fails when there's not enough valid data after NaN prefix"""
        data = np.array([np.nan, np.nan, 1.0, 2.0, 3.0])
        
        # With period=3, offset=2, needs period+offset+1=6 valid values
        with pytest.raises(ValueError, match="Not enough valid data"):
            ta_indicators.epma(data, period=3, offset=2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])