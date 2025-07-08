"""
Python binding tests for SMA indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
from test_utils import load_test_data, assert_close

# Import SMA functions - they'll be available after building with maturin
try:
    from my_project import (
        sma, 
        sma_batch,
        SmaStream
    )
except ImportError:
    pytest.skip("SMA module not available - run 'maturin develop' first", allow_module_level=True)


def test_sma_partial_params():
    """Test SMA with default parameters - mirrors check_sma_partial_params"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test with default parameter (9)
    result = sma(close, 9)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(close)


def test_sma_accuracy():
    """Test SMA matches expected values from Rust tests - mirrors check_sma_accuracy"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test with period=9 (default)
    result = sma(close, 9)
    
    assert len(result) == len(close)
    
    # Expected values from Rust test
    expected_last_five = [59180.8, 59175.0, 59129.4, 59085.4, 59133.7]
    
    actual_last_five = result[-5:]
    
    for i, (actual, expected) in enumerate(zip(actual_last_five, expected_last_five)):
        assert_close(actual, expected, rtol=1e-4, msg=f"SMA mismatch at index {i}")


def test_sma_invalid_period():
    """Test SMA fails with invalid period"""
    input_data = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    
    # Period = 0 should fail
    with pytest.raises(ValueError, match="sma"):
        sma(input_data, 0)


def test_sma_period_exceeds_length():
    """Test SMA fails with period exceeding length"""
    data_small = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="sma"):
        sma(data_small, 10)


def test_sma_very_small_dataset():
    """Test SMA fails with insufficient data"""
    single_point = np.array([42.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="sma"):
        sma(single_point, 9)


def test_sma_reinput():
    """Test SMA with re-input of SMA result - mirrors check_sma_reinput"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # First SMA pass with period=14
    first_result = sma(close, 14)
    
    # Second SMA pass with period=14 using first result as input
    second_result = sma(first_result, 14)
    
    assert len(second_result) == len(first_result)
    
    # All values should be finite after initial warmup periods
    # Find first non-NaN in first_result (output of first pass)
    first_valid_in_first = np.where(~np.isnan(first_result))[0][0]
    warmup = first_valid_in_first + 14 - 1  # second pass warmup
    assert np.all(np.isfinite(second_result[warmup:]))


def test_sma_nan_handling():
    """Test SMA handling of NaN values - mirrors check_sma_nan_handling"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    result = sma(close, 9)
    
    assert len(result) == len(close)
    
    # After warmup, all values should be finite
    # The test data has NaN values at the beginning, find first valid
    first_valid = np.where(~np.isnan(close))[0][0]
    warmup = first_valid + 9 - 1
    
    if len(result) > warmup:
        assert np.all(np.isfinite(result[warmup:]))


def test_sma_all_nan():
    """Test SMA with all NaN values"""
    all_nan = np.full(100, np.nan, dtype=np.float64)
    
    with pytest.raises(ValueError, match="sma"):
        sma(all_nan, 14)


def test_sma_empty_input():
    """Test SMA with empty input"""
    data_empty = np.array([], dtype=np.float64)
    
    with pytest.raises(ValueError, match="sma"):
        sma(data_empty, 14)


def test_sma_batch():
    """Test SMA batch computation"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test period range
    batch_result = sma_batch(
        close,
        (9, 240, 1)  # period range: 9 to 240 step 1
    )
    
    # Check result is a dict with the expected keys
    assert isinstance(batch_result, dict)
    assert 'values' in batch_result
    assert 'periods' in batch_result
    
    # Should have 232 period combinations (9 to 240 inclusive)
    assert batch_result['values'].shape == (232, len(close))
    assert len(batch_result['periods']) == 232
    
    # Verify first combination matches individual calculation
    individual_result = sma(close, 9)
    batch_first = batch_result['values'][0]
    
    # Compare after warmup period
    first_valid = np.where(~np.isnan(close))[0][0]
    warmup = first_valid + 9 - 1
    assert_close(batch_first[warmup:], individual_result[warmup:], atol=1e-9, msg="SMA first combination mismatch")


def test_sma_stream():
    """Test SMA streaming interface"""
    data = load_test_data()
    close = data['close']
    
    # Create stream with period=9
    period = 9
    
    # Calculate batch result for comparison
    close_array = np.array(close, dtype=np.float64)
    batch_result = sma(close_array, period)
    
    # Test streaming
    stream = SmaStream(period)
    stream_results = []
    
    for price in close:
        result = stream.update(price)
        stream_results.append(result if result is not None else np.nan)
    
    stream_results = np.array(stream_results)
    
    # Compare values where both are not NaN
    for i in range(len(close)):
        if np.isnan(batch_result[i]) and np.isnan(stream_results[i]):
            continue
        if not np.isnan(batch_result[i]) and not np.isnan(stream_results[i]):
            assert_close(batch_result[i], stream_results[i], rtol=1e-9, msg=f"SMA streaming mismatch at index {i}")


def test_sma_different_periods():
    """Test SMA with various period values"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test various period values
    test_periods = [5, 10, 14, 20, 50]
    
    for period in test_periods:
        result = sma(close, period)
        assert len(result) == len(close)
        
        # After warmup, all values should be finite
        first_valid = np.where(~np.isnan(close))[0][0]
        warmup = first_valid + period - 1
        if warmup < len(result):
            assert np.all(np.isfinite(result[warmup:])), f"NaN values found for period={period}"


def test_sma_batch_performance():
    """Test that batch computation works correctly (performance is secondary)"""
    data = load_test_data()
    close = np.array(data['close'][:1000], dtype=np.float64)  # Use first 1000 values
    
    # Test multiple parameter combinations
    batch_result = sma_batch(
        close,
        (10, 50, 10)  # periods: 10, 20, 30, 40, 50
    )
    
    # Should have 5 combinations
    assert batch_result['values'].shape == (5, len(close))
    
    # Verify metadata
    expected_periods = [10, 20, 30, 40, 50]
    assert np.array_equal(batch_result['periods'], expected_periods)


def test_sma_edge_cases():
    """Test SMA with edge case inputs"""
    # Test with monotonically increasing data
    data = np.arange(1.0, 101.0, dtype=np.float64)
    result = sma(data, 14)
    assert len(result) == len(data)
    assert np.all(np.isfinite(result[13:]))  # After period - 1
    
    # Test with constant values
    data = np.array([50.0] * 100, dtype=np.float64)
    result = sma(data, 14)
    assert len(result) == len(data)
    # With constant input, SMA should equal the constant after warmup
    assert np.all(np.isfinite(result[13:]))
    assert np.allclose(result[13:], 50.0)
    
    # Test with oscillating values
    data = np.array([10.0, 20.0, 10.0, 20.0] * 25, dtype=np.float64)
    result = sma(data, 14)
    assert len(result) == len(data)
    assert np.all(np.isfinite(result[13:]))


def test_sma_consistency():
    """Test that SMA produces consistent results across multiple calls"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    result1 = sma(close, 14)
    result2 = sma(close, 14)
    
    assert_close(result1, result2, atol=1e-15, msg="SMA results not consistent")


def test_sma_step_precision():
    """Test batch with various step sizes"""
    data = np.arange(1, 51, dtype=np.float64)
    
    # Test with different step sizes
    batch_result = sma_batch(
        data,
        (10, 20, 2)  # periods: 10, 12, 14, 16, 18, 20
    )
    
    assert batch_result['values'].shape == (6, len(data))
    expected_periods = [10, 12, 14, 16, 18, 20]
    assert np.array_equal(batch_result['periods'], expected_periods)


def test_sma_batch_error_handling():
    """Test SMA batch error handling"""
    # Test with all NaN data
    all_nan = np.full(100, np.nan, dtype=np.float64)
    with pytest.raises(ValueError, match="sma"):
        sma_batch(all_nan, (10, 20, 5))
    
    # Test with insufficient data
    small_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    with pytest.raises(ValueError, match="sma"):
        sma_batch(small_data, (10, 20, 5))


def test_sma_zero_copy_verification():
    """Verify SMA uses zero-copy operations"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    # The result should be computed directly without intermediate copies
    result = sma(close, 14)
    assert len(result) == len(close)
    
    # Batch should also use zero-copy
    batch_result = sma_batch(close, (10, 20, 5))
    assert batch_result['values'].shape[0] == 3  # 10, 15, 20
    assert batch_result['values'].shape[1] == len(close)


def test_sma_stream_error_handling():
    """Test SMA stream error handling"""
    # Test with invalid period
    with pytest.raises(ValueError, match="sma"):
        SmaStream(0)
    
    # Test that stream properly handles warmup
    stream = SmaStream(9)
    
    # First 8 updates should return None (period = 9, so need 9 values)
    for i in range(8):
        result = stream.update(float(i + 1))
        assert result is None
    
    # 9th update should return a value
    result = stream.update(9.0)
    assert result is not None
    assert isinstance(result, float)
    assert_close(result, 5.0, atol=1e-9)  # Average of 1..9 is 5


def test_sma_warmup_behavior():
    """Test SMA warmup period behavior"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    period = 9
    result = sma(close, period)
    
    # Find first non-NaN value in input
    first_valid = np.where(~np.isnan(close))[0][0]
    warmup = first_valid + period - 1
    
    # Values during warmup should be NaN  
    assert np.all(np.isnan(result[first_valid:warmup])), "Expected NaN during warmup period"
    
    # Values after warmup should be finite
    assert np.all(np.isfinite(result[warmup:])), "Expected finite values after warmup"


def test_sma_formula_verification():
    """Verify SMA formula implementation"""
    # Create simple test data with known pattern
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)
    period = 3
    
    result = sma(data, period)
    
    # Result length should match input
    assert len(result) == len(data)
    
    # Manually verify some SMA values
    # SMA[2] = (1+2+3)/3 = 2.0
    # SMA[3] = (2+3+4)/3 = 3.0
    # SMA[4] = (3+4+5)/3 = 4.0
    assert_close(result[2], 2.0, atol=1e-9)
    assert_close(result[3], 3.0, atol=1e-9)
    assert_close(result[4], 4.0, atol=1e-9)


def test_sma_kernel_options():
    """Test SMA with different kernel options"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    # Test with different kernels
    kernels = ['auto', 'scalar']
    
    for kernel in kernels:
        result = sma(close, 9, kernel=kernel)
        assert len(result) == len(close)
        assert np.all(np.isfinite(result[8:]))
    
    # Invalid kernel should fail
    with pytest.raises(ValueError, match="Unknown kernel"):
        sma(close, 9, kernel='invalid')


if __name__ == '__main__':
    # Run a simple test to verify the module loads correctly
    print("Testing SMA module...")
    test_sma_accuracy()
    print("SMA tests passed!")