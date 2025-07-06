"""
Python binding tests for JSA indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
from test_utils import load_test_data, assert_close

# Import JSA functions - they'll be available after building with maturin
try:
    from my_project import (
        jsa, 
        jsa_batch,
        jsa_batch_with_metadata,
        jsa_batch_2d,
        JsaStream
    )
except ImportError:
    pytest.skip("JSA module not available - run 'maturin develop' first", allow_module_level=True)


def test_jsa_partial_params():
    """Test JSA with default parameters - mirrors check_jsa_partial_params"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test with default period (30)
    result = jsa(close, 30)
    assert len(result) == len(close)


def test_jsa_accuracy():
    """Test JSA matches expected values from Rust tests - mirrors check_jsa_accuracy"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Period 30 (default)
    result = jsa(close, 30)
    
    # Expected last 5 values from Rust test
    expected_last_five = [61640.0, 61418.0, 61240.0, 61060.5, 60889.5]
    
    # Check last 5 values match expected
    assert_close(
        result[-5:], 
        expected_last_five, 
        atol=1e-5,
        msg="JSA last 5 values mismatch"
    )


def test_jsa_zero_period():
    """Test JSA fails with zero period - mirrors check_jsa_zero_period"""
    input_data = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="jsa:"):
        jsa(input_data, 0)


def test_jsa_period_exceeds_length():
    """Test JSA fails with period exceeding data length - mirrors check_jsa_period_exceeds_length"""
    data_small = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="jsa:"):
        jsa(data_small, 10)


def test_jsa_very_small_dataset():
    """Test JSA fails with insufficient data - mirrors check_jsa_very_small_dataset"""
    single_point = np.array([42.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="jsa:"):
        jsa(single_point, 5)


def test_jsa_empty_input():
    """Test JSA with empty input"""
    data_empty = np.array([], dtype=np.float64)
    
    with pytest.raises(ValueError, match="jsa:"):
        jsa(data_empty, 30)


def test_jsa_all_nan():
    """Test JSA with all NaN input"""
    data = np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
    
    with pytest.raises(ValueError, match="jsa:"):
        jsa(data, 3)


def test_jsa_reinput():
    """Test JSA with re-input of JSA result - mirrors check_jsa_reinput"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # First JSA pass with period=10
    first_result = jsa(close, 10)
    
    # Second JSA pass with period=5 using first result as input
    second_result = jsa(first_result, 5)
    
    assert len(second_result) == len(first_result)
    
    # After index 30, values should be finite (not NaN)
    for i in range(30, len(second_result)):
        assert np.isfinite(second_result[i]), f"Unexpected NaN at index {i}"


def test_jsa_nan_handling():
    """Test JSA handling of NaN values"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Inject some NaN values
    close_with_nans = close.copy()
    close_with_nans[10:15] = np.nan
    
    result = jsa(close_with_nans, 30)
    
    assert len(result) == len(close_with_nans)
    
    # Check that we have NaN in the beginning (warmup period)
    assert np.isnan(result[0])
    
    # After warmup, we should have non-NaN values where input is valid
    # Note: The exact behavior depends on how JSA handles NaN in the window


def test_jsa_batch():
    """Test JSA batch computation"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test period range 10-40 step 10
    period_start = 10
    period_end = 40
    period_step = 10  # periods: 10, 20, 30, 40
    
    batch_result, periods = jsa_batch(close, period_start, period_end, period_step)
    
    # Should have 4 periods
    assert len(periods) == 4
    assert periods == [10, 20, 30, 40]
    
    # Batch result should contain all individual results flattened
    assert len(batch_result) == 4 * len(close)
    
    # Verify first combination matches individual calculation
    individual_result = jsa(close, 10)
    batch_first_row = batch_result[:len(close)]
    assert_close(batch_first_row, individual_result, atol=1e-9, msg="First combination mismatch")


def test_jsa_batch_with_metadata():
    """Test JSA batch computation with metadata"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test period range
    period_start = 20
    period_end = 60
    period_step = 20  # periods: 20, 40, 60
    
    batch_result, periods = jsa_batch_with_metadata(close, period_start, period_end, period_step)
    
    # Check metadata
    assert periods == [20, 40, 60]
    
    # Check result shape
    assert len(batch_result) == 3 * len(close)
    
    # Verify second combination (period=40)
    individual_result = jsa(close, 40)
    batch_second_row = batch_result[len(close):2*len(close)]
    assert_close(batch_second_row, individual_result, atol=1e-9, msg="Second combination mismatch")


def test_jsa_batch_2d():
    """Test JSA batch computation with 2D output"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test period range
    period_start = 15
    period_end = 45
    period_step = 15  # periods: 15, 30, 45
    
    batch_result_2d, periods = jsa_batch_2d(close, period_start, period_end, period_step)
    
    # Check metadata
    assert periods == [15, 30, 45]
    
    # Check shape
    assert batch_result_2d.shape == (3, len(close))
    
    # Verify middle row (period=30)
    individual_result = jsa(close, 30)
    assert_close(batch_result_2d[1], individual_result, atol=1e-9, msg="Middle row mismatch")


def test_jsa_stream():
    """Test JSA streaming interface"""
    data = load_test_data()
    close = data['close']
    period = 30
    
    # Calculate batch result for comparison
    close_array = np.array(close, dtype=np.float64)
    batch_result = jsa(close_array, period)
    
    # Test streaming
    stream = JsaStream(period)
    stream_results = []
    
    for price in close:
        result = stream.update(price)
        stream_results.append(result if result is not None else np.nan)
    
    # Compare streaming vs batch
    # The warmup period should have NaN values
    for i in range(period):
        assert stream_results[i] is None or np.isnan(stream_results[i])
    
    # After warmup, values should match
    for i in range(period, len(close)):
        if not np.isnan(batch_result[i]):
            assert_close(stream_results[i], batch_result[i], atol=1e-9, 
                        msg=f"Streaming mismatch at index {i}")


def test_jsa_different_periods():
    """Test JSA with various period values"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test various period values
    for period in [5, 10, 20, 50]:
        result = jsa(close, period)
        assert len(result) == len(close)
        
        # Count valid values after warmup
        valid_count = np.sum(np.isfinite(result[period:]))
        assert valid_count > len(close) - period - 5, f"Too many NaN values for period={period}"


def test_jsa_batch_performance():
    """Test that batch computation works correctly (performance is secondary)"""
    data = load_test_data()
    close = np.array(data['close'][:1000], dtype=np.float64)  # Use first 1000 values
    
    # Test 5 periods
    batch_result, periods = jsa_batch(close, 10, 50, 10)
    assert len(periods) == 5  # periods: 10, 20, 30, 40, 50
    
    # Verify each combination
    for i, period in enumerate(periods):
        individual_result = jsa(close, period)
        start_idx = i * len(close)
        end_idx = start_idx + len(close)
        batch_row = batch_result[start_idx:end_idx]
        assert_close(batch_row, individual_result, atol=1e-9, 
                          msg=f"Batch mismatch for period={period}")


def test_jsa_edge_cases():
    """Test JSA with edge case inputs"""
    # Test with single valid value and rest NaN
    data = np.array([42.0, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
    # This should work - only one valid value but period=3
    result = jsa(data, 3)
    assert len(result) == len(data)
    # All should be NaN since we need at least 3 valid values
    
    # Test with alternating valid/NaN values
    data = np.array([1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan], dtype=np.float64)
    result = jsa(data, 2)
    assert len(result) == len(data)
    
    # Test with very large period
    data = np.random.randn(100).astype(np.float64)
    result = jsa(data, 99)
    assert len(result) == len(data)
    # Last value should be average of first and last
    expected_last = (data[-1] + data[-100]) * 0.5
    assert_close(result[-1], expected_last, atol=1e-9)


def test_jsa_single_value():
    """Test JSA with single value input"""
    data = np.array([42.0], dtype=np.float64)
    
    # Period=1 with single value should work
    result = jsa(data, 1)
    assert len(result) == 1
    assert np.isnan(result[0])  # Should be NaN as it's in warmup


def test_jsa_two_values():
    """Test JSA with two values input"""
    data = np.array([1.0, 2.0], dtype=np.float64)
    
    # Should work with period=1
    result = jsa(data, 1)
    assert len(result) == 2
    # First value should be NaN (warmup)
    assert np.isnan(result[0])
    # Second value should be average of data[1] and data[0]
    assert_close(result[1], (data[1] + data[0]) * 0.5, atol=1e-9)
    
    # Should also work with period=2 (period == data.length is allowed)
    result2 = jsa(data, 2)
    assert len(result2) == 2
    # Both should be NaN as warmup = first + period = 0 + 2 = 2
    assert np.isnan(result2[0])
    assert np.isnan(result2[1])


def test_jsa_warmup_period():
    """Test that warmup period is correctly calculated"""
    data = load_test_data()
    close = np.array(data['close'][:50], dtype=np.float64)
    
    test_cases = [
        (5, 5),    # period=5, warmup=5
        (10, 10),  # period=10, warmup=10
        (20, 20),  # period=20, warmup=20
        (30, 30),  # period=30, warmup=30
    ]
    
    for period, expected_warmup in test_cases:
        result = jsa(close, period)
        
        # Check NaN values up to warmup period
        for i in range(expected_warmup):
            assert np.isnan(result[i]), f"Expected NaN at index {i} for period={period}"
        
        # Check valid values after warmup
        if expected_warmup < len(result):
            assert not np.isnan(result[expected_warmup]), \
                f"Expected valid value at index {expected_warmup} for period={period}"


def test_jsa_consistency():
    """Test that JSA produces consistent results across multiple calls"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    result1 = jsa(close, 30)
    result2 = jsa(close, 30)
    
    assert_close(result1, result2, atol=1e-15, msg="JSA results not consistent")


def test_jsa_step_precision():
    """Test batch with very small step sizes"""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
    
    batch_result, periods = jsa_batch(data, 2, 4, 1)  # periods: 2, 3, 4
    
    assert periods == [2, 3, 4]
    assert len(batch_result) == 3 * len(data)


if __name__ == "__main__":
    # Run a simple test to verify the module loads correctly
    print("Testing JSA module...")
    test_jsa_accuracy()
    print("JSA tests passed!")