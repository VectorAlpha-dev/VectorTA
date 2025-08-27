"""
Python binding tests for SMMA indicator.
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


class TestSmma:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_smma_partial_params(self, test_data):
        """Test SMMA with partial parameters (None values) - mirrors check_smma_partial_params"""
        close = test_data['close']
        
        # Test with default params
        result = ta_indicators.smma(close, 7)  # Using default period
        assert len(result) == len(close)
    
    def test_smma_accuracy(self, test_data):
        """Test SMMA matches expected values from Rust tests - mirrors check_smma_accuracy"""
        close = test_data['close']
        
        result = ta_indicators.smma(close, period=7)
        
        assert len(result) == len(close)
        
        # Expected values from Rust test
        expected_last_five = [59434.4, 59398.2, 59346.9, 59319.4, 59224.5]
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected_last_five,
            rtol=1e-1,  # Using same tolerance as Rust test
            msg="SMMA last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('smma', result, 'close', {'period': 7})
    
    def test_smma_default_candles(self, test_data):
        """Test SMMA with default parameters - mirrors check_smma_default_candles"""
        close = test_data['close']
        
        # Default params: period=7
        result = ta_indicators.smma(close, 7)
        assert len(result) == len(close)
    
    def test_smma_zero_period(self):
        """Test SMMA fails with zero period - mirrors check_smma_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.smma(input_data, period=0)
    
    def test_smma_empty_input(self):
        """Test SMMA fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.smma(empty, period=7)
    
    def test_smma_period_exceeds_length(self):
        """Test SMMA fails when period exceeds data length - mirrors check_smma_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.smma(data_small, period=10)
    
    def test_smma_very_small_dataset(self):
        """Test SMMA fails with insufficient data - mirrors check_smma_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.smma(single_point, period=9)
    
    def test_smma_reinput(self, test_data):
        """Test SMMA applied twice (re-input) - mirrors check_smma_reinput"""
        close = test_data['close']
        
        # First pass with period 7
        first_result = ta_indicators.smma(close, period=7)
        assert len(first_result) == len(close)
        
        # Verify first pass matches expected
        expected_first_pass = [59434.4, 59398.2, 59346.9, 59319.4, 59224.5]
        assert_close(
            first_result[-5:],
            expected_first_pass,
            rtol=1e-1,
            msg="First pass SMMA values mismatch"
        )
        
        # Second pass with period 5 - apply SMMA to SMMA output
        second_result = ta_indicators.smma(first_result, period=5)
        assert len(second_result) == len(first_result)
        
        # Verify the re-input produces expected smoothing
        # The second pass should be smoother than the first
        assert np.std(second_result[~np.isnan(second_result)]) < np.std(first_result[~np.isnan(first_result)]), \
            "Second pass should produce smoother results"
    
    def test_smma_nan_handling(self, test_data):
        """Test SMMA handles NaN values correctly - mirrors check_smma_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.smma(close, period=7)
        assert len(result) == len(close)
        
        # After warmup period (240), no NaN values should exist
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"
        
        # First period-1 values should be NaN (warmup = first + period - 1)
        # Since first valid index is 0, warmup = 0 + 7 - 1 = 6
        assert np.all(np.isnan(result[:6])), "Expected NaN in warmup period (indices 0-5)"
        # First valid value should be at index 6 (period-1)
        assert not np.isnan(result[6]), "Expected valid value at index 6 (period-1)"
    
    def test_smma_streaming(self, test_data):
        """Test SMMA streaming matches batch calculation - mirrors check_smma_streaming"""
        close = test_data['close']
        period = 7
        
        # Batch calculation
        batch_result = ta_indicators.smma(close, period=period)
        
        # Streaming calculation
        stream = ta_indicators.SmmaStream(period=period)
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
            assert_close(b, s, rtol=1e-10, atol=1e-10, 
                        msg=f"SMMA streaming mismatch at index {i}")
    
    def test_smma_batch(self, test_data):
        """Test SMMA batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        result = ta_indicators.smma_batch(
            close,
            period_range=(7, 7, 0)  # Default period only
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        # Should have 1 combination (default params)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        # Extract the single row
        default_row = result['values'][0]
        expected = [59434.4, 59398.2, 59346.9, 59319.4, 59224.5]
        
        # Check last 5 values match
        assert_close(
            default_row[-5:],
            expected,
            rtol=1e-1,  # Using same tolerance as Rust test
            msg="SMMA batch default row mismatch"
        )
    
    def test_smma_all_nan_input(self):
        """Test SMMA with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.smma(all_nan, period=7)
    
    def test_smma_warmup_period(self, test_data):
        """Test SMMA warmup period is correct"""
        close = test_data['close']
        period = 10
        
        result = ta_indicators.smma(close, period=period)
        
        # First period-1 values should be NaN (warmup = first + period - 1)
        # Since first valid index is 0, warmup = 0 + period - 1 = period - 1
        assert np.all(np.isnan(result[:period-1])), f"Expected NaN in first {period-1} values"
        
        # Value at period-1 index should not be NaN
        assert not np.isnan(result[period-1]), f"Expected valid value at index {period-1}"
    
    def test_smma_kernel_options(self, test_data):
        """Test SMMA with different kernel options"""
        close = test_data['close']
        period = 7
        
        # Test auto kernel (default)
        result_auto = ta_indicators.smma(close, period=period)
        
        # Test scalar kernel
        result_scalar = ta_indicators.smma(close, period=period, kernel='scalar')
        
        # Results should be very close (scalar is always supported)
        assert_close(result_auto, result_scalar, rtol=1e-12, 
                    msg="Auto vs scalar kernel results differ")
        
        # Test invalid kernel
        with pytest.raises(ValueError, match="Unknown kernel"):
            ta_indicators.smma(close, period=period, kernel='invalid')
    
    def test_smma_batch_multiple_periods(self, test_data):
        """Test SMMA batch with multiple periods"""
        close = test_data['close']
        
        result = ta_indicators.smma_batch(
            close,
            period_range=(5, 10, 1)  # Periods 5, 6, 7, 8, 9, 10
        )
        
        assert result['values'].shape[0] == 6  # 6 periods
        assert result['values'].shape[1] == len(close)
        assert len(result['periods']) == 6
        assert list(result['periods']) == [5, 6, 7, 8, 9, 10]
    
    def test_smma_batch_large_range(self, test_data):
        """Test SMMA batch with large period range"""
        close = test_data['close']
        
        result = ta_indicators.smma_batch(
            close,
            period_range=(7, 100, 1)  # Default range from Rust
        )
        
        expected_periods = list(range(7, 101))
        assert result['values'].shape[0] == len(expected_periods)
        assert result['values'].shape[1] == len(close)
        assert list(result['periods']) == expected_periods
    
    def test_smma_batch_validation(self, test_data):
        """Test SMMA batch parameter validation"""
        close = test_data['close']
        
        # Test period exceeding data length
        small_data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.smma_batch(small_data, period_range=(5, 5, 0))
        
        # Test all NaN input
        all_nan = np.full(100, np.nan)
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.smma_batch(all_nan, period_range=(7, 7, 0))
    
    def test_smma_batch_matches_individual(self, test_data):
        """Test that batch results match individual calculations"""
        close = test_data['close'][:100]  # Use subset for speed
        
        # Test multiple periods
        periods = [5, 7, 10, 14]
        batch_result = ta_indicators.smma_batch(
            close,
            period_range=(5, 14, 1)
        )
        
        # Extract rows for our test periods
        for period in periods:
            # Find the row index for this period
            row_idx = period - 5  # Since we start at 5
            batch_row = batch_result['values'][row_idx]
            
            # Calculate individual result
            individual_result = ta_indicators.smma(close, period=period)
            
            # They should match exactly
            assert_close(
                batch_row,
                individual_result,
                rtol=1e-10,
                msg=f"Batch row for period {period} doesn't match individual calculation"
            )
    
    def test_smma_leading_nans(self):
        """Test SMMA with leading NaN values in data"""
        # Data with 3 leading NaNs
        data = np.array([np.nan, np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        period = 3
        
        result = ta_indicators.smma(data, period=period)
        
        # First valid data point is at index 3
        # Warmup = first + period - 1 = 3 + 3 - 1 = 5
        # So indices 0-4 should be NaN, index 5 should be first valid
        assert np.all(np.isnan(result[:5])), "Expected NaN through index 4"
        assert not np.isnan(result[5]), "Expected valid value at index 5"
        
        # The first valid SMMA value should be mean of [1.0, 2.0, 3.0]
        expected_first = (1.0 + 2.0 + 3.0) / 3
        assert_close(result[5], expected_first, rtol=1e-10)
    
    def test_smma_kernel_batch(self, test_data):
        """Test SMMA batch with different kernel options"""
        close = test_data['close'][:50]  # Small dataset for speed
        
        # Test with auto kernel (default)
        result_auto = ta_indicators.smma_batch(
            close,
            period_range=(7, 7, 0)
        )
        
        # Test with scalar kernel
        result_scalar = ta_indicators.smma_batch(
            close,
            period_range=(7, 7, 0),
            kernel='scalar'
        )
        
        # Results should be very close
        assert_close(
            result_auto['values'][0],
            result_scalar['values'][0],
            rtol=1e-10,
            msg="Auto vs scalar kernel batch results differ"
        )
    
    def test_smma_edge_case_period_one(self):
        """Test SMMA with period=1 (edge case)"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ta_indicators.smma(data, period=1)
        
        # With period=1, SMMA should equal the input data
        assert_close(result, data, rtol=1e-10, 
                    msg="SMMA with period=1 should equal input")
    
    def test_smma_constant_values(self):
        """Test SMMA with constant values"""
        constant_value = 42.0
        data = np.full(50, constant_value)
        
        result = ta_indicators.smma(data, period=10)
        
        # After warmup, all values should equal the constant
        non_nan_values = result[~np.isnan(result)]
        assert np.all(np.abs(non_nan_values - constant_value) < 1e-10), \
            "SMMA of constant values should converge to the constant"
    
    def test_smma_formula_verification(self):
        """Verify SMMA formula implementation"""
        # Simple test data where we can manually calculate SMMA
        data = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0])
        period = 3
        
        result = ta_indicators.smma(data, period=period)
        
        # First period-1 values should be NaN (warmup = first + period - 1)
        assert np.all(np.isnan(result[:period-1]))
        
        # First SMMA value (at index period-1) should be mean of first period values
        expected_first = np.mean(data[:period])  # (10 + 12 + 14) / 3 = 12.0
        assert_close(result[period-1], expected_first, rtol=1e-10)
        
        # Second SMMA value follows the formula: (prev * (period - 1) + new_value) / period
        expected_second = (expected_first * (period - 1) + data[period]) / period
        # (12.0 * 2 + 16) / 3 = 40 / 3 = 13.333...
        assert_close(result[period], expected_second, rtol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])