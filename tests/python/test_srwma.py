"""
Python binding tests for SRWMA indicator.
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


class TestSrwma:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_srwma_partial_params(self, test_data):
        """Test SRWMA with partial parameters (None values) - mirrors check_srwma_partial_params"""
        close = test_data['close']
        
        # Test with default period (14)
        result = ta_indicators.srwma(close, 14)
        assert len(result) == len(close)
    
    def test_srwma_accuracy(self, test_data):
        """Test SRWMA matches expected values from Rust tests - mirrors check_srwma_accuracy"""
        close = test_data['close']
        
        # Expected values from Rust test
        expected_last_five = [
            59344.28384704595,
            59282.09151629659,
            59192.76580529367,
            59178.04767548977,
            59110.03801260874,
        ]
        
        result = ta_indicators.srwma(close, period=14)
        
        assert len(result) == len(close)
        
        # Add debug output
        print(f"\nSRWMA test_accuracy debug:")
        print(f"Input length: {len(close)}")
        print(f"Output length: {len(result)}")
        print(f"Actual last 5: {result[-5:]}")
        print(f"Expected last 5: {expected_last_five}")
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected_last_five,
            rtol=1e-8,
            msg="SRWMA last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('srwma', result, 'close', {'period': 14})
    
    def test_srwma_zero_period(self):
        """Test SRWMA fails with zero period - mirrors check_srwma_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.srwma(input_data, period=0)
    
    def test_srwma_period_exceeds_length(self):
        """Test SRWMA fails when period exceeds data length - mirrors check_srwma_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.srwma(data_small, period=10)
    
    def test_srwma_very_small_dataset(self):
        """Test SRWMA fails with insufficient data - mirrors check_srwma_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.srwma(single_point, period=3)
    
    def test_srwma_nan_handling(self, test_data):
        """Test SRWMA handles NaN values correctly - mirrors check_srwma_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.srwma(close, period=14)
        assert len(result) == len(close)
        
        # After warmup period (50), no NaN values should exist
        if len(result) > 50:
            assert not np.any(np.isnan(result[50:])), "Found unexpected NaN after warmup period"
        
        # Check warmup period - SRWMA needs period + 1 values before producing output
        # First period+1 values should be NaN
        assert np.all(np.isnan(result[:15])), "Expected NaN in warmup period (first period+1 values)"
    
    def test_srwma_streaming(self, test_data):
        """Test SRWMA streaming matches batch calculation - mirrors check_srwma_streaming"""
        close = test_data['close']
        period = 14
        
        # Batch calculation
        batch_result = ta_indicators.srwma(close, period=period)
        
        # Streaming calculation
        stream = ta_indicators.SrwmaStream(period=period)
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
                        msg=f"SRWMA streaming mismatch at index {i}")
    
    def test_srwma_batch(self, test_data):
        """Test SRWMA batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        result = ta_indicators.srwma_batch(
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
            59344.28384704595,
            59282.09151629659,
            59192.76580529367,
            59178.04767548977,
            59110.03801260874,
        ]
        
        # Check last 5 values match
        assert_close(
            default_row[-5:],
            expected_last_five,
            rtol=1e-8,
            msg="SRWMA batch default row mismatch"
        )
    
    def test_srwma_all_nan_input(self):
        """Test SRWMA with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.srwma(all_nan, period=14)
    
    def test_srwma_period_less_than_2(self):
        """Test SRWMA with period < 2"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # SRWMA might allow period=1, so check for specific behavior
        # Based on the code, period=0 is invalid, but period=1 might be valid
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.srwma(data, period=0)
    
    def test_srwma_kernel_selection(self, test_data):
        """Test SRWMA with different kernel selections"""
        close = test_data['close']
        
        # Test scalar kernel
        result_scalar = ta_indicators.srwma(close, period=14, kernel='scalar')
        assert len(result_scalar) == len(close)
        
        # Test auto kernel
        result_auto = ta_indicators.srwma(close, period=14, kernel='auto')
        assert len(result_auto) == len(close)
        
        # Test invalid kernel
        with pytest.raises(ValueError, match="Unknown kernel"):
            ta_indicators.srwma(close, period=14, kernel='invalid_kernel')
    
    def test_srwma_batch_multiple_periods(self, test_data):
        """Test SRWMA batch with multiple periods"""
        close = test_data['close']
        
        result = ta_indicators.srwma_batch(
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
    
    def test_srwma_edge_case_period_2(self):
        """Test SRWMA with minimum valid period (2)"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = ta_indicators.srwma(data, period=2)
        assert len(result) == len(data)
        
        # First 3 values (period + 1) should be NaN
        assert np.all(np.isnan(result[:3]))
        # Remaining values should be valid
        assert not np.any(np.isnan(result[3:]))
    
    def test_srwma_with_leading_nans(self):
        """Test SRWMA correctly handles data that starts with NaN values"""
        # Create data with 5 leading NaNs
        data = np.array([np.nan] * 5 + [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        result = ta_indicators.srwma(data, period=3)
        assert len(result) == len(data)
        
        # First non-NaN is at index 5, so warmup ends at 5 + 3 + 1 = 9
        assert np.all(np.isnan(result[:9])), "Expected NaN in warmup period including leading NaNs"
        # Should have valid values starting from index 9
        assert not np.any(np.isnan(result[9:])), "Expected valid values after warmup"
        
        # Test with different kernels to ensure they all handle it correctly
        for kernel in ['scalar', 'auto']:
            result_k = ta_indicators.srwma(data, period=3, kernel=kernel)
            assert np.all(np.isnan(result_k[:9])), f"Kernel {kernel}: Expected NaN in warmup"
            assert not np.any(np.isnan(result_k[9:])), f"Kernel {kernel}: Expected valid values after warmup"
    
    def test_srwma_reinput(self, test_data):
        """Test SRWMA with reinput - mirrors check_srwma_reinput"""
        close = test_data['close']
        
        # First pass with period 14
        first_result = ta_indicators.srwma(close, period=14)
        
        # Second pass with the output of the first, using period 5
        second_result = ta_indicators.srwma(first_result, period=5)
        
        assert len(second_result) == len(first_result)
        
        # After sufficient warmup, all values should be finite
        for i in range(50, len(second_result)):
            assert np.isfinite(second_result[i])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])