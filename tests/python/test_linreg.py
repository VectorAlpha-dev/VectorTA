"""
Python binding tests for LinReg indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
from test_utils import load_test_data, assert_close

# Import LinReg functions - they'll be available after building with maturin
try:
    from my_project import (
        linreg, 
        linreg_batch,
        linreg_batch_with_metadata,
        linreg_batch_2d,
        LinRegStream
    )
except ImportError:
    pytest.skip("LinReg module not available - run 'maturin develop' first", allow_module_level=True)


def test_linreg_partial_params():
    """Test LinReg with default parameters - mirrors check_linreg_partial_params"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test with default period (14)
    result = linreg(close, 14)
    assert len(result) == len(close)


def test_linreg_accuracy():
    """Test LinReg matches expected values from Rust tests - mirrors check_linreg_accuracy"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Period 14 (default)
    result = linreg(close, 14)
    
    # Expected last 5 values from Rust test
    expected_last_five = [
        58929.37142857143,
        58899.42857142857,
        58918.857142857145,
        59100.6,
        58987.94285714286,
    ]
    
    # Check last 5 values match expected
    assert_close(
        result[-5:], 
        expected_last_five, 
        atol=1e-1,
        msg="LinReg last 5 values mismatch"
    )


def test_linreg_zero_period():
    """Test LinReg fails with zero period - mirrors check_linreg_zero_period"""
    input_data = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="linreg:"):
        linreg(input_data, 0)


def test_linreg_period_exceeds_length():
    """Test LinReg fails with period exceeding data length - mirrors check_linreg_period_exceeds_length"""
    data_small = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="linreg:"):
        linreg(data_small, 10)


def test_linreg_very_small_dataset():
    """Test LinReg fails with insufficient data - mirrors check_linreg_very_small_dataset"""
    single_point = np.array([42.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="linreg:"):
        linreg(single_point, 14)


def test_linreg_empty_input():
    """Test LinReg with empty input"""
    data_empty = np.array([], dtype=np.float64)
    
    with pytest.raises(ValueError, match="linreg:"):
        linreg(data_empty, 14)


def test_linreg_all_nan():
    """Test LinReg with all NaN input"""
    data = np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
    
    with pytest.raises(ValueError, match="linreg:"):
        linreg(data, 3)


def test_linreg_reinput():
    """Test LinReg with re-input of LinReg result - mirrors check_linreg_reinput"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # First LinReg pass with period=14
    first_result = linreg(close, 14)
    
    # Second LinReg pass with period=10 using first result as input
    second_result = linreg(first_result, 10)
    
    assert len(second_result) == len(first_result)
    
    # The second pass will have its own warmup period
    # First pass warmup: 14 values
    # Second pass warmup: 10 values  
    # So we expect NaN values up to index 14+10=24
    for i in range(24, len(second_result)):
        assert np.isfinite(second_result[i]), f"Unexpected NaN at index {i}"


def test_linreg_nan_handling():
    """Test LinReg handling of NaN values - mirrors check_linreg_nan_handling"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Use clean data like the Rust test does
    result = linreg(close, 14)
    
    assert len(result) == len(close)
    
    # The Rust test checks that after index 240, there are no NaN values
    # This implies the warmup period creates NaN values at the beginning
    if len(result) > 240:
        for i in range(240, len(result)):
            assert not np.isnan(result[i]), f"Found unexpected NaN at index {i}"


def test_linreg_batch():
    """Test LinReg batch computation"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test period range 10-40 step 10
    period_start = 10
    period_end = 40
    period_step = 10  # periods: 10, 20, 30, 40
    
    batch_result, periods = linreg_batch(close, period_start, period_end, period_step)
    
    # Should have 4 periods
    assert len(periods) == 4
    assert periods == [10, 20, 30, 40]
    
    # Batch result should contain all individual results flattened
    assert len(batch_result) == 4 * len(close)
    
    # Verify first combination matches individual calculation
    individual_result = linreg(close, 10)
    batch_first_row = batch_result[:len(close)]
    assert_close(batch_first_row, individual_result, atol=1e-9, msg="First combination mismatch")


def test_linreg_batch_with_metadata():
    """Test LinReg batch computation with metadata"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test period range
    period_start = 20
    period_end = 60
    period_step = 20  # periods: 20, 40, 60
    
    batch_result, periods = linreg_batch_with_metadata(close, period_start, period_end, period_step)
    
    # Check metadata
    assert periods == [20, 40, 60]
    
    # Check result shape
    assert len(batch_result) == 3 * len(close)
    
    # Verify second combination (period=40)
    individual_result = linreg(close, 40)
    batch_second_row = batch_result[len(close):2*len(close)]
    assert_close(batch_second_row, individual_result, atol=1e-9, msg="Second combination mismatch")


def test_linreg_batch_2d():
    """Test LinReg batch computation with 2D output"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test period range
    period_start = 15
    period_end = 45
    period_step = 15  # periods: 15, 30, 45
    
    batch_result_2d, periods = linreg_batch_2d(close, period_start, period_end, period_step)
    
    # Check metadata
    assert periods == [15, 30, 45]
    
    # Check shape
    assert batch_result_2d.shape == (3, len(close))
    
    # Verify middle row (period=30)
    individual_result = linreg(close, 30)
    assert_close(batch_result_2d[1], individual_result, atol=1e-9, msg="Middle row mismatch")


def test_linreg_stream():
    """Test LinReg streaming interface - mirrors check_linreg_streaming"""
    data = load_test_data()
    close = data['close']
    period = 14
    
    # Calculate batch result for comparison
    close_array = np.array(close, dtype=np.float64)
    batch_result = linreg(close_array, period)
    
    # Test streaming
    stream = LinRegStream(period)
    stream_results = []
    
    for price in close:
        result = stream.update(price)
        # The Rust test converts None to NaN
        stream_results.append(result if result is not None else np.nan)
    
    # Compare streaming vs batch
    assert len(batch_result) == len(stream_results)
    
    for i in range(len(batch_result)):
        # Both NaN is okay
        if np.isnan(batch_result[i]) and np.isnan(stream_results[i]):
            continue
        # Otherwise, they should match closely
        assert_close(stream_results[i], batch_result[i], atol=1e-6, 
                    msg=f"Streaming mismatch at index {i}")


def test_linreg_different_periods():
    """Test LinReg with various period values"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test various period values
    for period in [5, 10, 20, 50]:
        result = linreg(close, period)
        assert len(result) == len(close)
        
        # Count valid values after warmup
        valid_count = np.sum(np.isfinite(result[period:]))
        assert valid_count > len(close) - period - 5, f"Too many NaN values for period={period}"


def test_linreg_batch_performance():
    """Test that batch computation works correctly (performance is secondary)"""
    data = load_test_data()
    close = np.array(data['close'][:1000], dtype=np.float64)  # Use first 1000 values
    
    # Test 5 periods
    batch_result, periods = linreg_batch(close, 10, 50, 10)
    assert len(periods) == 5  # periods: 10, 20, 30, 40, 50
    
    # Verify each combination
    for i, period in enumerate(periods):
        individual_result = linreg(close, period)
        start_idx = i * len(close)
        end_idx = start_idx + len(close)
        batch_row = batch_result[start_idx:end_idx]
        assert_close(batch_row, individual_result, atol=1e-9, 
                          msg=f"Batch mismatch for period={period}")


def test_linreg_edge_cases():
    """Test LinReg with edge case inputs"""
    # Test with monotonically increasing data (perfect linear regression)
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)
    result = linreg(data, 3)
    assert len(result) == len(data)
    
    # LinReg outputs the fitted value at the end of the window
    # With data [8, 9, 10] and perfect linear fit, the value at x=3 is 10.0
    assert_close(result[-1], 10.0, atol=1e-9, msg="Perfect linear regression failed")
    
    # Test with alternating valid/NaN values
    data = np.array([1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan], dtype=np.float64)
    result = linreg(data, 2)
    assert len(result) == len(data)
    
    # Test with constant values
    data = np.array([5.0] * 20, dtype=np.float64)
    result = linreg(data, 5)
    assert len(result) == len(data)
    # With constant data, LinReg should predict the same constant value
    for i in range(5, len(result)):
        assert_close(result[i], 5.0, atol=1e-9, msg=f"Constant prediction failed at index {i}")


def test_linreg_single_value():
    """Test LinReg with single value input"""
    data = np.array([42.0], dtype=np.float64)
    
    # Period=1 with single value should actually work, returning [NaN]
    result = linreg(data, 1)
    assert len(result) == 1
    assert np.isnan(result[0])


def test_linreg_two_values():
    """Test LinReg with two values input"""
    data = np.array([1.0, 2.0], dtype=np.float64)
    
    # Period=1 is mathematically undefined (can't fit a line with one point)
    # So it returns all NaN values
    result = linreg(data, 1)
    assert len(result) == 2
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    
    # Test with period=2 which should work
    result = linreg(data, 2)
    assert len(result) == 2
    assert np.isnan(result[0])  # First value is NaN (warmup)
    assert_close(result[1], 2.0, atol=1e-9, msg="Two-value prediction failed")


def test_linreg_warmup_period():
    """Test that warmup period is correctly calculated"""
    data = load_test_data()
    close = np.array(data['close'][:50], dtype=np.float64)
    
    test_cases = [
        (5, 4),    # period=5, warmup=4 (first output at index 4)
        (10, 9),   # period=10, warmup=9 (first output at index 9)
        (20, 19),  # period=20, warmup=19 (first output at index 19)
        (30, 29),  # period=30, warmup=29 (first output at index 29)
    ]
    
    for period, expected_warmup in test_cases:
        result = linreg(close, period)
        
        # LinReg has NaN values for the first 'period-1' indices
        for i in range(period - 1):
            assert np.isnan(result[i]), f"Expected NaN at index {i} for period={period}"
        
        # First valid value should be at index 'period-1'
        if period - 1 < len(result):
            assert not np.isnan(result[period - 1]), \
                f"Expected valid value at index {period - 1} for period={period}"


def test_linreg_consistency():
    """Test that LinReg produces consistent results across multiple calls"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    result1 = linreg(close, 14)
    result2 = linreg(close, 14)
    
    assert_close(result1, result2, atol=1e-15, msg="LinReg results not consistent")


def test_linreg_step_precision():
    """Test batch with very small step sizes"""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
    
    batch_result, periods = linreg_batch(data, 2, 4, 1)  # periods: 2, 3, 4
    
    assert periods == [2, 3, 4]
    assert len(batch_result) == 3 * len(data)


def test_linreg_slope_calculation():
    """Test LinReg slope calculation with known data"""
    # Create data with known slope = 2
    data = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0], dtype=np.float64)
    result = linreg(data, 4)
    
    # LinReg outputs the fitted value at the end of the window
    # With the last window [10, 12, 14, 16] and slope=2, the fitted value at x=4 is 16
    assert_close(result[-1], 16.0, atol=1e-9, msg="Slope calculation failed")


if __name__ == "__main__":
    # Run a simple test to verify the module loads correctly
    print("Testing LinReg module...")
    test_linreg_accuracy()
    print("LinReg tests passed!")