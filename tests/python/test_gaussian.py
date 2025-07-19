"""
Python binding tests for Gaussian indicator.
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


class TestGaussian:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_gaussian_partial_params(self, test_data):
        """Test Gaussian with partial parameters - mirrors check_gaussian_partial_params"""
        close = test_data['close']
        
        # Test with default params: period=14, poles=4
        result = ta_indicators.gaussian(close, 14, 4)
        assert len(result) == len(close)
    
    def test_gaussian_accuracy(self, test_data):
        """Test Gaussian matches expected values from Rust tests - mirrors check_gaussian_accuracy"""
        close = test_data['close']
        
        # Using period=14, poles=4
        result = ta_indicators.gaussian(close, 14, 4)
        
        assert len(result) == len(close)
        
        # Expected last 5 values from Rust test
        expected_last_five = [
            59221.90637814869,
            59236.15215167245,
            59207.10087088464,
            59178.48276885589,
            59085.36983209433
        ]
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected_last_five,
            rtol=1e-4,  # Using 1e-4 as in Rust test
            msg="Gaussian last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('gaussian', result, 'close', {'period': 14, 'poles': 4})
    
    def test_gaussian_default_candles(self, test_data):
        """Test Gaussian with default parameters - mirrors check_gaussian_default_candles"""
        close = test_data['close']
        
        # Default params: period=14, poles=4
        result = ta_indicators.gaussian(close, 14, 4)
        assert len(result) == len(close)
        
        # Compare with Rust
        compare_with_rust('gaussian', result, 'close', {'period': 14, 'poles': 4})
    
    def test_gaussian_zero_period(self):
        """Test Gaussian fails with zero period - mirrors check_gaussian_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError):
            ta_indicators.gaussian(input_data, period=0, poles=4)
    
    def test_gaussian_period_exceeds_length(self):
        """Test Gaussian fails when period exceeds data length - mirrors check_gaussian_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError):
            ta_indicators.gaussian(data_small, period=10, poles=4)
    
    def test_gaussian_very_small_dataset(self):
        """Test Gaussian with very small dataset - mirrors check_gaussian_very_small_dataset"""
        data_single = np.array([42.0])
        
        with pytest.raises(ValueError):
            ta_indicators.gaussian(data_single, period=3, poles=4)
    
    def test_gaussian_empty_input(self):
        """Test Gaussian with empty input - mirrors check_gaussian_empty_input"""
        data_empty = np.array([])
        
        with pytest.raises(ValueError):
            ta_indicators.gaussian(data_empty, period=14, poles=4)
    
    def test_gaussian_invalid_poles(self):
        """Test Gaussian with invalid poles - mirrors check_gaussian_invalid_poles"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test poles = 0
        with pytest.raises(ValueError):
            ta_indicators.gaussian(data, period=3, poles=0)
        
        # Test poles = 5 (> 4)
        with pytest.raises(ValueError):
            ta_indicators.gaussian(data, period=3, poles=5)
    
    def test_gaussian_all_nan(self):
        """Test Gaussian with all NaN input - mirrors check_gaussian_all_nan"""
        data = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        
        with pytest.raises(ValueError):
            ta_indicators.gaussian(data, period=3, poles=4)
    
    def test_gaussian_reinput(self, test_data):
        """Test Gaussian with re-input of Gaussian result - mirrors check_gaussian_reinput"""
        close = test_data['close']
        
        # First Gaussian pass with period=14, poles=4
        first_result = ta_indicators.gaussian(close, 14, 4)
        
        # Second Gaussian pass with period=10, poles=2 using first result as input
        second_result = ta_indicators.gaussian(first_result, 10, 2)
        
        assert len(second_result) == len(first_result)
        
        # Verify no NaN values after warmup period in second result
        for i in range(240, len(second_result)):
            assert not np.isnan(second_result[i]), f"NaN found at index {i}"
    
    def test_gaussian_nan_handling(self, test_data):
        """Test Gaussian handling of NaN values - mirrors check_gaussian_nan_handling"""
        # The Rust test doesn't actually test NaN inputs, it just verifies that
        # the outputs are finite for regular data. Let's test with regular data
        close = test_data['close']
        
        result = ta_indicators.gaussian(close, period=14, poles=4)
        
        assert len(result) == len(close)
        # Skip the first few values (poles) and check that remaining are finite
        skip = 4  # poles
        for i in range(skip, len(result)):
            assert np.isfinite(result[i]), f"Non-finite value found at index {i}"
    
    def test_gaussian_streaming(self, test_data):
        """Test Gaussian streaming vs batch calculation - mirrors check_gaussian_streaming"""
        close = test_data['close'][:100]  # Use first 100 values for testing
        period = 14
        poles = 4
        
        # Batch calculation
        batch_result = ta_indicators.gaussian(close, period, poles)
        
        # Streaming calculation
        stream = ta_indicators.GaussianStream(period, poles)
        stream_results = []
        
        for val in close:
            result = stream.update(val)
            stream_results.append(result if result is not None else np.nan)
        
        stream_results = np.array(stream_results)
        
        # Compare batch vs streaming
        assert_close(
            stream_results[period:], 
            batch_result[period:],
            rtol=1e-9,
            msg="Gaussian streaming vs batch mismatch"
        )
    
    def test_gaussian_batch(self, test_data):
        """Test Gaussian batch computation."""
        close = test_data['close']
        
        # Test period range 10-20 step 5, poles range 2-4 step 1
        period_range = (10, 20, 5)  # periods: 10, 15, 20
        poles_range = (2, 4, 1)     # poles: 2, 3, 4
        
        result = ta_indicators.gaussian_batch(close, period_range, poles_range)
        
        assert isinstance(result, dict)
        assert 'values' in result
        assert 'periods' in result
        assert 'poles' in result
        
        expected_periods = [10, 10, 10, 15, 15, 15, 20, 20, 20]  # 3 periods x 3 poles
        expected_poles = [2, 3, 4, 2, 3, 4, 2, 3, 4]
        
        assert list(result['periods']) == expected_periods
        assert list(result['poles']) == expected_poles
        assert result['values'].shape == (9, len(close))  # 3 periods x 3 poles = 9 rows
        
        # Check each row corresponds to individual Gaussian calculation
        row_idx = 0
        for period in [10, 15, 20]:
            for poles in [2, 3, 4]:
                individual_result = ta_indicators.gaussian(close, period, poles)
                np.testing.assert_allclose(
                    result['values'][row_idx], 
                    individual_result, 
                    rtol=1e-9,
                    err_msg=f"Batch row {row_idx} (period={period}, poles={poles}) mismatch"
                )
                row_idx += 1
    
    def test_gaussian_different_poles(self, test_data):
        """Test Gaussian with different poles values."""
        close = test_data['close']
        period = 14
        
        # Test all valid poles values (1-4)
        for poles in [1, 2, 3, 4]:
            result = ta_indicators.gaussian(close, period, poles)
            assert len(result) == len(close)
            
            # Verify warmup period - Rust implementation returns actual values during warmup
            # not NaN values, so we just check that we get valid results
            assert len(result) == len(close)
            # Check that after warmup we have valid non-NaN values
            assert not np.isnan(result[period:]).any()
    
    def test_gaussian_kernel_parameter(self, test_data):
        """Test that kernel parameter works correctly"""
        close = test_data['close']
        period = 14
        poles = 4
        
        # Test with different kernel types
        result_auto = ta_indicators.gaussian(close, period, poles)  # Default (auto)
        result_scalar = ta_indicators.gaussian(close, period, poles, kernel='scalar')
        
        # Results should be very close (within floating point tolerance)
        valid_idx = ~np.isnan(result_auto)
        np.testing.assert_allclose(result_auto[valid_idx], result_scalar[valid_idx], rtol=1e-10)
        
        # Test invalid kernel
        with pytest.raises(ValueError):
            ta_indicators.gaussian(close, period, poles, kernel='invalid')
        
        # Test batch with kernel
        batch_result = ta_indicators.gaussian_batch(
            close, 
            period_range=(10, 20, 5),
            poles_range=(2, 4, 1),
            kernel='scalar'
        )
        assert 'values' in batch_result
        assert 'periods' in batch_result
        assert 'poles' in batch_result