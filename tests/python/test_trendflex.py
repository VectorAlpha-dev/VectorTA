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
        
        # Use exact reference values from Rust tests
        expected_last_five = [
            -0.19724678008015128,
            -0.1238001236481444,
            -0.10515389737087717,
            -0.1149541079904878,
            -0.16006869484450567,
        ]
        
        result = ta_indicators.trendflex(close, period=20)
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected_last_five,
            rtol=1e-8,
            msg="TrendFlex last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('trendflex', result, 'close', {'period': 20})
    
    def test_trendflex_default_candles(self, test_data):
        """Test TrendFlex with default parameters - mirrors check_trendflex_default_candles"""
        close = test_data['close']
        
        # Default params: period=20
        result = ta_indicators.trendflex(close)
        assert len(result) == len(close)
    
    def test_trendflex_zero_period(self):
        """Test TrendFlex fails with zero period - mirrors check_trendflex_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="period = 0|ZeroTrendFlexPeriod"):
            ta_indicators.trendflex(input_data, period=0)
    
    def test_trendflex_period_exceeds_length(self):
        """Test TrendFlex fails when period exceeds data length - mirrors check_trendflex_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="period > data len|TrendFlexPeriodExceedsData"):
            ta_indicators.trendflex(data_small, period=10)
    
    def test_trendflex_very_small_dataset(self):
        """Test TrendFlex fails with insufficient data - mirrors check_trendflex_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="period > data len|TrendFlexPeriodExceedsData"):
            ta_indicators.trendflex(single_point, period=9)
    
    def test_trendflex_empty_input(self):
        """Test TrendFlex fails with empty input - mirrors check_trendflex_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="No data provided|NoDataProvided"):
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
        
        # Calculate warmup period: first_valid + period
        first_valid = 0  # Since close data starts valid at index 0
        warmup = first_valid + 20
        
        # First warmup values should be NaN
        assert np.all(np.isnan(result[:warmup])), f"Expected NaN in warmup period [0:{warmup})"
        # Value at warmup index should NOT be NaN
        assert not np.isnan(result[warmup]), f"Expected valid value at index {warmup}"
    
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
    
    def test_trendflex_batch_single_param(self, test_data):
        """Test TrendFlex batch processing with single parameter - mirrors check_batch_default_row"""
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
        assert len(result['periods']) == 1
        assert result['periods'][0] == 20
        
        # Extract the single row
        default_row = result['values'][0]
        expected_last_five = [
            -0.19724678008015128,
            -0.1238001236481444,
            -0.10515389737087717,
            -0.1149541079904878,
            -0.16006869484450567,
        ]
        
        # Check last 5 values match
        assert_close(
            default_row[-5:],
            expected_last_five,
            rtol=1e-8,
            msg="TrendFlex batch default row mismatch"
        )
        
        # Verify matches single calculation
        single_result = ta_indicators.trendflex(close, period=20)
        assert_close(
            default_row,
            single_result,
            rtol=1e-10,
            msg="Batch vs single calculation mismatch"
        )
    
    def test_trendflex_batch_multiple_periods(self, test_data):
        """Test TrendFlex batch processing with multiple periods"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        result = ta_indicators.trendflex_batch(
            close,
            period_range=(10, 30, 10),  # periods: 10, 20, 30
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        # Should have 3 combinations
        assert result['values'].shape[0] == 3
        assert result['values'].shape[1] == len(close)
        assert len(result['periods']) == 3
        assert list(result['periods']) == [10, 20, 30]
        
        # Verify each row matches individual calculation
        for i, period in enumerate([10, 20, 30]):
            row_data = result['values'][i]
            single_result = ta_indicators.trendflex(close, period=period)
            assert_close(
                row_data,
                single_result,
                rtol=1e-10,
                msg=f"Batch row {i} (period={period}) mismatch"
            )
            
            # Check warmup period for each row
            warmup = period  # first_valid=0 + period
            assert np.all(np.isnan(row_data[:warmup])), f"Expected NaN in warmup [0:{warmup}) for period={period}"
            assert not np.isnan(row_data[warmup]), f"Expected valid value at index {warmup} for period={period}"
    
    def test_trendflex_batch_edge_cases(self, test_data):
        """Test TrendFlex batch processing edge cases"""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        # Single value sweep (step=0)
        single_batch = ta_indicators.trendflex_batch(
            close,
            period_range=(5, 5, 0),
        )
        assert single_batch['values'].shape[0] == 1
        assert single_batch['values'].shape[1] == len(close)
        assert len(single_batch['periods']) == 1
        
        # Step larger than range
        large_step_batch = ta_indicators.trendflex_batch(
            close,
            period_range=(5, 7, 10),  # Step larger than range
        )
        # Should only have period=5
        assert large_step_batch['values'].shape[0] == 1
        assert large_step_batch['periods'][0] == 5
        
        # Empty data should throw
        with pytest.raises(ValueError, match="No data provided|All values are NaN"):
            ta_indicators.trendflex_batch(
                np.array([]),
                period_range=(20, 20, 0),
            )
    
    def test_trendflex_all_nan_input(self):
        """Test TrendFlex with all NaN values - mirrors check_trendflex_all_nan"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN|AllValuesNaN"):
            ta_indicators.trendflex(all_nan, period=20)
    
    def test_trendflex_invalid_period(self):
        """Test TrendFlex with various invalid period values"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Period = 0
        with pytest.raises(ValueError, match="period = 0|ZeroTrendFlexPeriod"):
            ta_indicators.trendflex(data, period=0)
        
        # Period > data length
        with pytest.raises(ValueError, match="period > data len|TrendFlexPeriodExceedsData"):
            ta_indicators.trendflex(data, period=10)
        
        # Period = data length should still fail (need at least period + 1 data points)
        with pytest.raises(ValueError, match="period > data len|TrendFlexPeriodExceedsData"):
            ta_indicators.trendflex(data, period=len(data))
    
    def test_trendflex_warmup_calculation(self, test_data):
        """Test TrendFlex warmup period calculation"""
        # Test with various period values to verify warmup = first_valid + period
        test_periods = [5, 10, 20, 30, 50]
        close = test_data['close'][:200]  # Use subset for speed
        
        for period in test_periods:
            if period >= len(close):
                continue
                
            result = ta_indicators.trendflex(close, period=period)
            
            # First valid index is 0 for clean data
            first_valid = 0
            warmup = first_valid + period
            
            # Check NaN pattern
            for i in range(warmup):
                assert np.isnan(result[i]), f"Expected NaN at index {i} for period={period}"
            
            # Check first non-NaN value
            if warmup < len(result):
                assert not np.isnan(result[warmup]), f"Expected valid value at index {warmup} for period={period}"
    
    def test_trendflex_batch_metadata(self, test_data):
        """Test TrendFlex batch metadata is correctly populated"""
        close = test_data['close'][:50]  # Small dataset
        
        result = ta_indicators.trendflex_batch(
            close,
            period_range=(10, 20, 5),  # periods: 10, 15, 20
        )
        
        # Check metadata
        assert 'periods' in result
        assert len(result['periods']) == 3
        assert list(result['periods']) == [10, 15, 20]
        
        # Check values shape matches metadata
        assert result['values'].shape[0] == len(result['periods'])
        assert result['values'].shape[1] == len(close)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])