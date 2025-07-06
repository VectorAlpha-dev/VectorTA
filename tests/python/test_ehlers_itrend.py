"""
Python binding tests for Ehlers ITrend indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import numpy as np
import pytest
from test_utils import (
    load_test_data,
    assert_close,
    EXPECTED_OUTPUTS
)
from rust_comparison import compare_with_rust

# Import the indicator from the Rust module
import my_project

# Helper functions for NaN handling
def assert_no_nan(arr, msg=""):
    """Assert no NaN values in array"""
    if np.any(np.isnan(arr)):
        raise AssertionError(f"{msg}: Found NaN values in array")

def assert_all_nan(arr, msg=""):
    """Assert all values are NaN"""
    if not np.all(np.isnan(arr)):
        raise AssertionError(f"{msg}: Not all values are NaN")


class TestEhlersITrend:
    def setup_method(self):
        """Load test data before each test."""
        self.data = load_test_data()
        self.close = np.array(self.data['close'], dtype=np.float64)
    
    def test_ehlers_itrend_partial_params(self):
        """Test with default parameters - mirrors check_itrend_partial_params"""
        # Test with default parameters (warmup_bars=12, max_dc_period=50)
        result = my_project.ehlers_itrend(self.close, None, None)
        assert len(result) == len(self.close)
        
        # Test with partial custom parameters
        result = my_project.ehlers_itrend(self.close, 15, None)
        assert len(result) == len(self.close)
        
        result = my_project.ehlers_itrend(self.close, None, 40)
        assert len(result) == len(self.close)
    
    def test_ehlers_itrend_accuracy(self):
        """Test accuracy matches expected values from Rust tests - mirrors check_itrend_accuracy"""
        result = my_project.ehlers_itrend(self.close, 12, 50)
        
        # Expected values from Rust test
        expected_last_5 = [59097.88, 59145.9, 59191.96, 59217.26, 59179.68]
        
        assert_close(
            result[-5:],
            expected_last_5,
            rtol=0,
            atol=0.1,  # Using 1e-1 tolerance as in Rust test
            msg="Ehlers ITrend last 5 values mismatch"
        )
    
    def test_ehlers_itrend_error_empty_input(self):
        """Test error with empty input - mirrors check_itrend_no_data"""
        with pytest.raises(ValueError, match="Input data is empty"):
            my_project.ehlers_itrend(np.array([]), None, None)
    
    def test_ehlers_itrend_error_all_nan(self):
        """Test error with all NaN values - mirrors check_itrend_all_nan_data"""
        all_nan = np.full(100, np.nan)
        with pytest.raises(ValueError, match="All values are NaN"):
            my_project.ehlers_itrend(all_nan, None, None)
    
    def test_ehlers_itrend_error_insufficient_data(self):
        """Test error with insufficient data for warmup - mirrors check_itrend_small_data_for_warmup"""
        small_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="Not enough data for warmup"):
            my_project.ehlers_itrend(small_data, 10, None)
    
    def test_ehlers_itrend_error_zero_warmup(self):
        """Test error with zero warmup bars - mirrors check_itrend_zero_warmup"""
        with pytest.raises(ValueError, match="Invalid warmup_bars"):
            my_project.ehlers_itrend(self.close, 0, None)
    
    def test_ehlers_itrend_error_invalid_max_dc(self):
        """Test error with invalid max_dc_period - mirrors check_itrend_invalid_max_dc"""
        with pytest.raises(ValueError, match="Invalid max_dc_period"):
            my_project.ehlers_itrend(self.close, None, 0)
    
    def test_ehlers_itrend_reinput(self):
        """Test applying indicator twice - mirrors check_itrend_reinput"""
        # First pass
        first_result = my_project.ehlers_itrend(self.close, 12, 50)
        assert len(first_result) == len(self.close)
        
        # Second pass - apply to output
        second_result = my_project.ehlers_itrend(first_result, 10, 40)
        assert len(second_result) == len(first_result)
        
        # Verify no NaN values after warmup
        if len(second_result) > 20:
            assert_no_nan(second_result[20:], "Found unexpected NaN after warmup")
    
    def test_ehlers_itrend_nan_handling(self):
        """Test NaN handling - mirrors check_itrend_nan_handling"""
        result = my_project.ehlers_itrend(self.close, 12, 50)
        
        # First 12 values should echo input (warmup period)
        assert_close(
            result[:12],
            self.close[:12],
            rtol=0,
            atol=1e-10,
            msg="Warmup echo failed"
        )
        
        # After warmup, should have filtered values
        if len(result) > 12:
            assert_no_nan(result[12:], "Found unexpected NaN after warmup")
    
    def test_ehlers_itrend_streaming(self):
        """Test streaming interface - mirrors check_itrend_streaming"""
        # Batch calculation
        batch_result = my_project.ehlers_itrend(self.close, 12, 50)
        
        # Streaming calculation
        stream = my_project.EhlersITrendStream(12, 50)
        stream_result = []
        
        for price in self.close:
            value = stream.update(price)
            stream_result.append(value if value is not None else np.nan)
        
        stream_result = np.array(stream_result)
        
        # Compare batch vs streaming
        assert len(batch_result) == len(stream_result)
        
        # They should match within tolerance after warmup
        for i in range(12, len(batch_result)):
            if not (np.isnan(batch_result[i]) and np.isnan(stream_result[i])):
                assert abs(batch_result[i] - stream_result[i]) < 1e-9, \
                    f"Batch/stream mismatch at index {i}: {batch_result[i]} vs {stream_result[i]}"
    
    def test_ehlers_itrend_batch_single_params(self):
        """Test batch calculation with single parameter set"""
        # Single parameter combination
        batch_result = my_project.ehlers_itrend_batch(
            self.close[:100],  # Use smaller dataset
            (12, 12, 0),      # warmup_bars
            (50, 50, 0)       # max_dc_period
        )
        
        # Should return a dict with values and metadata
        assert isinstance(batch_result, dict)
        assert 'values' in batch_result
        assert 'warmup_bars' in batch_result
        assert 'max_dc_periods' in batch_result
        
        # Should have 1 row x 100 cols
        assert batch_result['values'].shape == (1, 100)
        
        # Compare with single calculation
        single_result = my_project.ehlers_itrend(self.close[:100], 12, 50)
        assert_close(
            batch_result['values'][0],
            single_result,
            rtol=0,
            atol=1e-10,
            msg="Batch vs single calculation mismatch"
        )
    
    def test_ehlers_itrend_batch_multiple_params(self):
        """Test batch calculation with multiple parameter combinations"""
        batch_result = my_project.ehlers_itrend_batch(
            self.close[:100],
            (10, 14, 2),      # warmup_bars: 10, 12, 14
            (40, 50, 10)      # max_dc_period: 40, 50
        )
        
        # Should have 3 warmup x 2 max_dc = 6 combinations
        assert batch_result['values'].shape[0] == 6
        assert batch_result['values'].shape[1] == 100
        
        # Verify metadata
        assert len(batch_result['warmup_bars']) == 6
        assert len(batch_result['max_dc_periods']) == 6
        
        # Check first combination (warmup=10, max_dc=40)
        single_result = my_project.ehlers_itrend(self.close[:100], 10, 40)
        assert_close(
            batch_result['values'][0],
            single_result,
            rtol=0,
            atol=1e-10,
            msg="First batch row mismatch"
        )
    
    def test_ehlers_itrend_batch_warmup_validation(self):
        """Test batch warmup period handling"""
        data = self.close[:30]
        
        batch_result = my_project.ehlers_itrend_batch(
            data,
            (10, 15, 5),      # warmup_bars: 10, 15
            (50, 50, 0)       # max_dc_period: 50
        )
        
        # Should have 2 rows
        assert batch_result['values'].shape == (2, 30)
        
        # First row (warmup=10): first 10 values should echo input
        assert_close(
            batch_result['values'][0][:10],
            data[:10],
            rtol=0,
            atol=1e-10,
            msg="First row warmup echo failed"
        )
        
        # Second row (warmup=15): first 15 values should echo input
        assert_close(
            batch_result['values'][1][:15],
            data[:15],
            rtol=0,
            atol=1e-10,
            msg="Second row warmup echo failed"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])