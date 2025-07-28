"""
Python binding tests for Linear Regression Angle indicator.
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


class TestLinearregAngle:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_linearreg_angle_partial_params(self, test_data):
        """Test Linear Regression Angle with partial parameters - mirrors check_lra_partial_params"""
        close = test_data['close']
        
        # Test with default period (14)
        result = ta_indicators.linearreg_angle(close, 14)
        assert len(result) == len(close)
    
    def test_linearreg_angle_accuracy(self, test_data):
        """Test Linear Regression Angle matches expected values from Rust tests - mirrors check_lra_accuracy"""
        close = test_data['close']
        
        result = ta_indicators.linearreg_angle(close, period=14)
        
        assert len(result) == len(close)
        
        # Expected last 5 values from Rust test
        expected_last_5 = [
            -89.30491945492733,
            -89.28911257342405,
            -89.1088041965075,
            -86.58419429159467,
            -87.77085937059316,
        ]
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected_last_5,
            rtol=1e-5,
            msg="Linear Regression Angle last 5 values mismatch"
        )
    
    def test_linearreg_angle_zero_period(self):
        """Test Linear Regression Angle fails with zero period - mirrors check_lra_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.linearreg_angle(input_data, period=0)
    
    def test_linearreg_angle_period_exceeds_length(self):
        """Test Linear Regression Angle fails when period exceeds data length - mirrors check_lra_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.linearreg_angle(data_small, period=10)
    
    def test_linearreg_angle_very_small_dataset(self):
        """Test Linear Regression Angle fails with insufficient data - mirrors check_lra_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.linearreg_angle(single_point, period=14)
    
    def test_linearreg_angle_empty_input(self):
        """Test Linear Regression Angle fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.linearreg_angle(empty, period=14)
    
    def test_linearreg_angle_reinput(self, test_data):
        """Test Linear Regression Angle applied twice (re-input) - mirrors check_lra_reinput"""
        close = test_data['close']
        
        # First pass
        first_result = ta_indicators.linearreg_angle(close, period=14)
        assert len(first_result) == len(close)
        
        # Second pass - apply to output of first pass
        second_result = ta_indicators.linearreg_angle(first_result, period=14)
        assert len(second_result) == len(first_result)
    
    def test_linearreg_angle_nan_handling(self, test_data):
        """Test Linear Regression Angle handles NaN values correctly"""
        close = test_data['close']
        
        result = ta_indicators.linearreg_angle(close, period=14)
        assert len(result) == len(close)
        
        # First period-1 values should be NaN (warmup period)
        warmup = 14 - 1  # period - 1
        for i in range(warmup):
            assert np.isnan(result[i]), f"Expected NaN at index {i} during warmup"
        
        # After warmup, values should not be NaN (unless input was NaN)
        for i in range(warmup, len(result)):
            if not np.isnan(close[i]):
                assert not np.isnan(result[i]), f"Unexpected NaN at index {i} after warmup"
    
    def test_linearreg_angle_all_nan_input(self):
        """Test Linear Regression Angle with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.linearreg_angle(all_nan, period=14)
    
    def test_linearreg_angle_batch_single_period(self, test_data):
        """Test batch processing with single period"""
        close = test_data['close']
        
        # Batch with single period
        batch_result = ta_indicators.linearreg_angle_batch(
            close,
            period_range=(14, 14, 0)  # Single period
        )
        
        # Should have one row
        assert batch_result['values'].shape[0] == 1
        assert batch_result['values'].shape[1] == len(close)
        assert len(batch_result['periods']) == 1
        assert batch_result['periods'][0] == 14
        
        # Single batch result should match regular call
        single_result = ta_indicators.linearreg_angle(close, period=14)
        assert_close(
            batch_result['values'][0], 
            single_result, 
            rtol=1e-10,
            msg="Batch vs single calculation mismatch"
        )
    
    def test_linearreg_angle_batch_multiple_periods(self, test_data):
        """Test batch processing with multiple periods"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        # Multiple periods: 10, 12, 14, 16
        batch_result = ta_indicators.linearreg_angle_batch(
            close,
            period_range=(10, 16, 2)  # period range
        )
        
        # Should have 4 rows (10, 12, 14, 16)
        assert batch_result['values'].shape[0] == 4
        assert batch_result['values'].shape[1] == 100
        assert len(batch_result['periods']) == 4
        
        # Verify each row matches individual calculation
        periods = [10, 12, 14, 16]
        for i, period in enumerate(periods):
            single_result = ta_indicators.linearreg_angle(close, period=period)
            assert_close(
                batch_result['values'][i], 
                single_result, 
                rtol=1e-10, 
                msg=f"Period {period} mismatch"
            )
    
    def test_linearreg_angle_stream_basic(self):
        """Test streaming functionality"""
        # Create a stream with period 14
        stream = ta_indicators.Linearreg_angleStream(14)
        
        # Generate test data
        test_values = [100 + i + 5 * np.sin(i * 0.1) for i in range(50)]
        
        results = []
        for value in test_values:
            result = stream.update(value)
            results.append(result)
        
        # First period-1 results should be None
        for i in range(13):  # period - 1
            assert results[i] is None, f"Expected None at index {i}"
        
        # After that, should have values
        for i in range(13, 50):
            assert results[i] is not None, f"Expected value at index {i}"
            assert isinstance(results[i], (int, float)), f"Expected number at index {i}"
    
    def test_linearreg_angle_stream_matches_batch(self):
        """Test that streaming produces same results as batch calculation"""
        period = 14
        stream = ta_indicators.Linearreg_angleStream(period)
        
        # Generate test data
        test_data = np.array([100 + i + 5 * np.sin(i * 0.1) for i in range(100)])
        
        # Stream processing
        stream_results = []
        for value in test_data:
            result = stream.update(value)
            if result is not None:
                stream_results.append(result)
            else:
                stream_results.append(np.nan)
        
        # Batch processing
        batch_result = ta_indicators.linearreg_angle(test_data, period=period)
        
        # Compare results (streaming should match batch)
        assert_close(
            stream_results, 
            batch_result, 
            rtol=1e-10,
            msg="Stream vs batch mismatch"
        )
    
    def test_linearreg_angle_kernel_consistency(self, test_data):
        """Test that different kernels produce consistent results"""
        close = test_data['close']
        
        # Test with different kernels
        result_auto = ta_indicators.linearreg_angle(close, period=14)
        result_scalar = ta_indicators.linearreg_angle(close, period=14, kernel='scalar')
        
        # Results should be very close (within floating point tolerance)
        assert_close(
            result_auto, 
            result_scalar, 
            rtol=1e-10,
            msg="Kernel consistency check failed"
        )
        
        # Try with invalid kernel
        with pytest.raises(ValueError, match="kernel"):
            ta_indicators.linearreg_angle(close, period=14, kernel='invalid_kernel')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])