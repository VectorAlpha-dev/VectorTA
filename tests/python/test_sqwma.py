"""
Python binding tests for SQWMA indicator.
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


class TestSqwma:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_sqwma_partial_params(self, test_data):
        """Test SQWMA with partial parameters (None values) - mirrors check_sqwma_partial_params"""
        close = test_data['close']
        
        # Test with default period (14)
        result = ta_indicators.sqwma(close, 14)
        assert len(result) == len(close)
    
    def test_sqwma_accuracy(self, test_data):
        """Test SQWMA matches expected values from Rust tests - mirrors check_sqwma_accuracy"""
        close = test_data['close']
        
        # Expected values from Rust test
        expected_last_five = [
            59229.72287968442,
            59211.30867850099,
            59172.516765286,
            59167.73471400394,
            59067.97928994083,
        ]
        
        result = ta_indicators.sqwma(close, period=14)
        
        assert len(result) == len(close)
        
        # Add debug output
        print(f"\nSQWMA test_accuracy debug:")
        print(f"Input length: {len(close)}")
        print(f"Output length: {len(result)}")
        print(f"Actual last 5: {result[-5:]}")
        print(f"Expected last 5: {expected_last_five}")
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected_last_five,
            rtol=1e-8,
            msg="SQWMA last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('sqwma', result, 'close', {'period': 14})
    
    def test_sqwma_zero_period(self):
        """Test SQWMA fails with zero period - mirrors check_sqwma_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.sqwma(input_data, period=0)
    
    def test_sqwma_period_exceeds_length(self):
        """Test SQWMA fails when period exceeds data length - mirrors check_sqwma_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.sqwma(data_small, period=10)
    
    def test_sqwma_very_small_dataset(self):
        """Test SQWMA fails with insufficient data - mirrors check_sqwma_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.sqwma(single_point, period=9)
    
    def test_sqwma_nan_handling(self, test_data):
        """Test SQWMA handles NaN values correctly - mirrors check_sqwma_nan_handling"""
        close = test_data['close']
        period = 14
        
        result = ta_indicators.sqwma(close, period=period)
        assert len(result) == len(close)
        
        # Find first non-NaN in input data
        first_valid = next((i for i, v in enumerate(close) if not np.isnan(v)), 0)
        
        # SQWMA warmup period: first_non_nan + period + 1
        warmup_end = first_valid + period + 1
        
        # Check warmup period has NaN values
        assert np.all(np.isnan(result[:warmup_end])), f"Expected NaN in warmup period (indices 0-{warmup_end-1})"
        
        # After warmup period, no NaN values should exist (if we have enough data)
        if len(result) > warmup_end:
            # Check a reasonable range after warmup
            check_start = min(warmup_end, 240)
            if len(result) > check_start:
                assert not np.any(np.isnan(result[check_start:])), f"Found unexpected NaN after index {check_start}"
    
    def test_sqwma_streaming(self, test_data):
        """Test SQWMA streaming matches batch calculation - mirrors check_sqwma_streaming"""
        close = test_data['close']
        period = 14
        
        # Batch calculation
        batch_result = ta_indicators.sqwma(close, period=period)
        
        # Streaming calculation
        stream = ta_indicators.SqwmaStream(period=period)
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
                        msg=f"SQWMA streaming mismatch at index {i}")
    
    def test_sqwma_batch(self, test_data):
        """Test SQWMA batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        result = ta_indicators.sqwma_batch(
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
        expected_last_five = [
            59229.72287968442,
            59211.30867850099,
            59172.516765286,
            59167.73471400394,
            59067.97928994083,
        ]
        
        # Check last 5 values match
        assert_close(
            default_row[-5:],
            expected_last_five,
            rtol=1e-8,
            msg="SQWMA batch default row mismatch"
        )
    
    def test_sqwma_empty_input(self):
        """Test SQWMA fails with empty input - mirrors check_sqwma_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.sqwma(empty, period=14)
    
    def test_sqwma_all_nan_input(self):
        """Test SQWMA with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.sqwma(all_nan, period=14)
    
    def test_sqwma_period_less_than_2(self):
        """Test SQWMA fails with period < 2"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.sqwma(data, period=1)
    
    def test_sqwma_kernel_selection(self, test_data):
        """Test SQWMA with different kernel selections"""
        close = test_data['close']
        
        # Test scalar kernel
        result_scalar = ta_indicators.sqwma(close, period=14, kernel='scalar')
        assert len(result_scalar) == len(close)
        
        # Test auto kernel
        result_auto = ta_indicators.sqwma(close, period=14, kernel='auto')
        assert len(result_auto) == len(close)
        
        # Test invalid kernel
        with pytest.raises(ValueError, match="Unknown kernel"):
            ta_indicators.sqwma(close, period=14, kernel='invalid_kernel')
    
    def test_sqwma_batch_multiple_periods(self, test_data):
        """Test SQWMA batch with multiple periods"""
        close = test_data['close']
        
        result = ta_indicators.sqwma_batch(
            close,
            period_range=(10, 20, 2),  # periods: 10, 12, 14, 16, 18, 20
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        # Should have 6 combinations
        assert result['values'].shape[0] == 6
        assert result['values'].shape[1] == len(close)
        
        # Check periods array
        expected_periods = [10, 12, 14, 16, 18, 20]
        assert np.array_equal(result['periods'], expected_periods)
        
        # Verify each row has appropriate warmup period
        for i, period in enumerate(expected_periods):
            row = result['values'][i]
            # Check that we have NaN values in the warmup period (period + 1)
            warmup_end = period + 1
            assert np.all(np.isnan(row[:warmup_end])), f"Expected NaN in warmup for period {period}"
            # Check that we have valid values after warmup (if data is long enough)
            if len(row) > warmup_end + 10:
                assert not np.all(np.isnan(row[warmup_end:warmup_end+10])), f"Expected valid values after warmup for period {period}"
    
    def test_sqwma_edge_case_period_2(self):
        """Test SQWMA with minimum valid period (2)"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = ta_indicators.sqwma(data, period=2)
        assert len(result) == len(data)
        
        # First 3 values (period + 1) should be NaN
        assert np.all(np.isnan(result[:3]))
        # Remaining values should be valid
        assert not np.any(np.isnan(result[3:]))
    
    def test_sqwma_with_leading_nans(self):
        """Test SQWMA correctly handles data that starts with NaN values"""
        # Create data with 5 leading NaNs
        data = np.array([np.nan] * 5 + [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        result = ta_indicators.sqwma(data, period=3)
        assert len(result) == len(data)
        
        # First non-NaN is at index 5, so warmup ends at 5 + 3 + 1 = 9
        assert np.all(np.isnan(result[:9])), "Expected NaN in warmup period including leading NaNs"
        # Should have valid values starting from index 9
        assert not np.any(np.isnan(result[9:])), "Expected valid values after warmup"
        
        # Test with different kernels to ensure they all handle it correctly
        for kernel in ['scalar', 'auto']:
            result_k = ta_indicators.sqwma(data, period=3, kernel=kernel)
            assert np.all(np.isnan(result_k[:9])), f"Kernel {kernel}: Expected NaN in warmup"
            assert not np.any(np.isnan(result_k[9:])), f"Kernel {kernel}: Expected valid values after warmup"
    
    def test_sqwma_boundary_conditions(self):
        """Test SQWMA with boundary conditions"""
        # Test with exact minimum data (period = data length)
        period = 5
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = ta_indicators.sqwma(data, period=period)
        assert len(result) == len(data)
        # All should be NaN since warmup = 0 + period + 1 = 6, which exceeds data length
        assert np.all(np.isnan(result)), "Expected all NaN when data length equals period"
        
        # Test with just enough data for one output
        data_extended = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])  # length = 7
        period = 3  # warmup = 0 + 3 + 1 = 4, so we get output at indices 4,5,6
        
        result = ta_indicators.sqwma(data_extended, period=period)
        assert len(result) == len(data_extended)
        assert np.all(np.isnan(result[:4])), "Expected NaN in warmup period"
        assert not np.any(np.isnan(result[4:])), "Expected valid values after warmup"
        
        # Test with period = 2 (minimum valid period)
        data_min = np.array([1.0, 2.0, 3.0, 4.0])
        result = ta_indicators.sqwma(data_min, period=2)
        assert len(result) == len(data_min)
        # Warmup = 0 + 2 + 1 = 3
        assert np.all(np.isnan(result[:3])), "Expected NaN in first 3 values for period=2"
        assert not np.isnan(result[3]), "Expected valid value at index 3 for period=2"
    
    def test_sqwma_batch_parameter_sweep(self, test_data):
        """Test SQWMA batch with comprehensive parameter sweep"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        # Test with multiple period combinations
        result = ta_indicators.sqwma_batch(
            close,
            period_range=(5, 25, 5),  # periods: 5, 10, 15, 20, 25
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        # Should have 5 combinations
        assert result['values'].shape[0] == 5
        assert result['values'].shape[1] == len(close)
        
        # Check periods array
        expected_periods = [5, 10, 15, 20, 25]
        assert np.array_equal(result['periods'], expected_periods)
        
        # Verify each row against single calculation
        for i, period in enumerate(expected_periods):
            row = result['values'][i]
            single_result = ta_indicators.sqwma(close, period=period)
            
            # Compare where both are valid
            for j in range(len(row)):
                if np.isnan(row[j]) and np.isnan(single_result[j]):
                    continue
                assert_close(row[j], single_result[j], rtol=1e-10, atol=1e-10,
                           msg=f"Batch vs single mismatch for period {period} at index {j}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])