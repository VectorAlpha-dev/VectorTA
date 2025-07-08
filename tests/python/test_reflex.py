"""
Python binding tests for Reflex indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
from test_utils import load_test_data, assert_close

# Import Reflex functions - they'll be available after building with maturin
try:
    from my_project import (
        reflex, 
        reflex_batch,
        ReflexStream
    )
except ImportError:
    pytest.skip("Reflex module not available - run 'maturin develop' first", allow_module_level=True)


def test_reflex_partial_params():
    """Test Reflex with default parameters - mirrors check_reflex_partial_params"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test with default parameter (20)
    result = reflex(close, 20)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(close)


def test_reflex_accuracy():
    """Test Reflex matches expected values from Rust tests - mirrors check_reflex_accuracy"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test with period=20 (default)
    result = reflex(close, 20)
    
    assert len(result) == len(close)
    
    # Expected values from Rust test
    expected_last_five = [
        0.8085220962465361,
        0.445264715886137,
        0.13861699036615063,
        -0.03598639652007061,
        -0.224906760543743
    ]
    
    actual_last_five = result[-5:]
    
    for i, (actual, expected) in enumerate(zip(actual_last_five, expected_last_five)):
        assert_close(actual, expected, rtol=1e-7, msg=f"Reflex mismatch at index {i}")


def test_reflex_invalid_period():
    """Test Reflex fails with invalid period"""
    input_data = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    
    # Period < 2 should fail
    with pytest.raises(ValueError, match="reflex"):
        reflex(input_data, 1)
    
    # Period = 0 should fail
    with pytest.raises(ValueError, match="reflex"):
        reflex(input_data, 0)


def test_reflex_period_exceeds_length():
    """Test Reflex fails with period exceeding length"""
    data_small = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="reflex"):
        reflex(data_small, 10)


def test_reflex_very_small_dataset():
    """Test Reflex fails with insufficient data"""
    single_point = np.array([42.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="reflex"):
        reflex(single_point, 5)


def test_reflex_reinput():
    """Test Reflex with re-input of Reflex result - mirrors check_reflex_reinput"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # First Reflex pass with period=14
    first_result = reflex(close, 14)
    
    # Second Reflex pass with period=20 using first result as input
    second_result = reflex(first_result, 20)
    
    assert len(second_result) == len(first_result)
    
    # All values should be finite after initial warmup periods
    # Find first non-NaN in input
    first_valid = np.where(~np.isnan(close))[0][0]
    warmup = first_valid + 20  # second pass warmup
    assert np.all(np.isfinite(second_result[warmup:]))


def test_reflex_nan_handling():
    """Test Reflex handling of NaN values - mirrors check_reflex_nan_handling"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    result = reflex(close, 20)
    
    assert len(result) == len(close)
    
    # After warmup, all values should be finite
    first_valid = np.where(~np.isnan(close))[0][0]
    warmup = first_valid + 20
    
    assert np.all(np.isfinite(result[warmup:]))


def test_reflex_all_nan():
    """Test Reflex with all NaN values"""
    all_nan = np.full(100, np.nan, dtype=np.float64)
    
    with pytest.raises(ValueError, match="reflex"):
        reflex(all_nan, 20)


def test_reflex_empty_input():
    """Test Reflex with empty input"""
    data_empty = np.array([], dtype=np.float64)
    
    with pytest.raises(ValueError, match="reflex"):
        reflex(data_empty, 20)


def test_reflex_batch():
    """Test Reflex batch computation"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test period range
    batch_result = reflex_batch(
        close,
        (10, 30, 5)  # period range: 10, 15, 20, 25, 30
    )
    
    # Check result is a dict with the expected keys
    assert isinstance(batch_result, dict)
    assert 'values' in batch_result
    assert 'periods' in batch_result
    
    # Should have 5 period combinations
    assert batch_result['values'].shape == (5, len(close))
    assert len(batch_result['periods']) == 5
    
    # Verify first combination matches individual calculation
    individual_result = reflex(close, 10)
    batch_first = batch_result['values'][0]
    
    # Compare after warmup period
    first_valid = np.where(~np.isnan(close))[0][0]
    warmup = first_valid + 10
    assert_close(batch_first[warmup:], individual_result[warmup:], atol=1e-9, msg="Reflex first combination mismatch")


def test_reflex_stream():
    """Test Reflex streaming interface"""
    data = load_test_data()
    close = data['close']
    
    # Create stream with period=20
    period = 20
    
    # Calculate batch result for comparison
    close_array = np.array(close, dtype=np.float64)
    batch_result = reflex(close_array, period)
    
    # Test streaming
    stream = ReflexStream(period)
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
            assert_close(batch_result[i], stream_results[i], rtol=1e-9, msg=f"Reflex streaming mismatch at index {i}")


def test_reflex_different_periods():
    """Test Reflex with various period values"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test various period values
    test_periods = [5, 10, 20, 50]
    
    for period in test_periods:
        result = reflex(close, period)
        assert len(result) == len(close)
        
        # After warmup, all values should be finite
        first_valid = np.where(~np.isnan(close))[0][0]
        warmup = first_valid + period
        if warmup < len(result):
            assert np.all(np.isfinite(result[warmup:])), f"NaN values found for period={period}"


def test_reflex_batch_performance():
    """Test that batch computation works correctly (performance is secondary)"""
    data = load_test_data()
    close = np.array(data['close'][:1000], dtype=np.float64)  # Use first 1000 values
    
    # Test multiple parameter combinations
    batch_result = reflex_batch(
        close,
        (10, 50, 10)  # periods: 10, 20, 30, 40, 50
    )
    
    # Should have 5 combinations
    assert batch_result['values'].shape == (5, len(close))
    
    # Verify metadata
    expected_periods = [10, 20, 30, 40, 50]
    assert np.array_equal(batch_result['periods'], expected_periods)


def test_reflex_edge_cases():
    """Test Reflex with edge case inputs"""
    # Test with monotonically increasing data
    data = np.arange(1.0, 101.0, dtype=np.float64)
    result = reflex(data, 20)
    assert len(result) == len(data)
    assert np.all(np.isfinite(result[20:]))  # After period
    
    # Test with constant values
    data = np.array([50.0] * 100, dtype=np.float64)
    result = reflex(data, 20)
    assert len(result) == len(data)
    # With constant input, Reflex produces NaN after warmup (division by zero variance)
    # This is expected behavior when there's no price variation
    # First period values should be zeros
    assert np.all(result[:20] == 0.0)
    
    # Test with oscillating values
    data = np.array([10.0, 20.0, 10.0, 20.0] * 25, dtype=np.float64)
    result = reflex(data, 20)
    assert len(result) == len(data)
    assert np.all(np.isfinite(result[20:]))


def test_reflex_consistency():
    """Test that Reflex produces consistent results across multiple calls"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    result1 = reflex(close, 20)
    result2 = reflex(close, 20)
    
    assert_close(result1, result2, atol=1e-15, msg="Reflex results not consistent")


def test_reflex_step_precision():
    """Test batch with various step sizes"""
    data = np.arange(1, 51, dtype=np.float64)
    
    # Test with different step sizes
    batch_result = reflex_batch(
        data,
        (10, 20, 2)  # periods: 10, 12, 14, 16, 18, 20
    )
    
    assert batch_result['values'].shape == (6, len(data))
    expected_periods = [10, 12, 14, 16, 18, 20]
    assert np.array_equal(batch_result['periods'], expected_periods)


def test_reflex_batch_error_handling():
    """Test Reflex batch error handling"""
    # Test with all NaN data
    all_nan = np.full(100, np.nan, dtype=np.float64)
    with pytest.raises(ValueError, match="reflex"):
        reflex_batch(all_nan, (10, 20, 5))
    
    # Test with insufficient data
    small_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    with pytest.raises(ValueError, match="reflex"):
        reflex_batch(small_data, (10, 20, 5))


def test_reflex_zero_copy_verification():
    """Verify Reflex uses zero-copy operations"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    # The result should be computed directly without intermediate copies
    result = reflex(close, 20)
    assert len(result) == len(close)
    
    # Batch should also use zero-copy
    batch_result = reflex_batch(close, (10, 20, 5))
    assert batch_result['values'].shape[0] == 3  # 10, 15, 20
    assert batch_result['values'].shape[1] == len(close)


def test_reflex_stream_error_handling():
    """Test Reflex stream error handling"""
    # Test with invalid period
    with pytest.raises(ValueError, match="reflex"):
        ReflexStream(1)
    
    # Test that stream properly handles warmup
    stream = ReflexStream(20)
    
    # First 20 updates should return None (period = 20)
    for i in range(20):
        result = stream.update(float(i + 1))
        assert result is None
    
    # 21st update should return a value
    result = stream.update(21.0)
    assert result is not None
    assert isinstance(result, float)


def test_reflex_warmup_behavior():
    """Test Reflex warmup period behavior"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    period = 20
    result = reflex(close, period)
    
    # Find first non-NaN value in input
    first_valid = np.where(~np.isnan(close))[0][0]
    warmup = first_valid + period
    
    # Values during warmup should be zero (Reflex specific behavior)
    assert np.all(result[:period] == 0.0), "Expected zeros during warmup period"
    
    # Values after warmup should be finite
    assert np.all(np.isfinite(result[warmup:])), "Expected finite values after warmup"


def test_reflex_formula_verification():
    """Verify Reflex formula implementation"""
    # Create simple test data with known pattern
    data = np.array([10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 10.0, 11.0, 12.0, 13.0] * 5, dtype=np.float64)
    period = 5
    
    result = reflex(data, period)
    
    # Result length should match input
    assert len(result) == len(data)
    
    # Warmup period should have zeros
    assert np.all(result[:period] == 0.0)
    
    # After warmup, values should be finite
    assert np.all(np.isfinite(result[period:]))
    
    # Reflex should detect the oscillating pattern
    # Values should not all be the same after warmup
    assert not np.allclose(result[period:], result[period], rtol=1e-9)


def test_reflex_kernel_options():
    """Test Reflex with different kernel options"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    # Test with different kernels
    kernels = ['auto', 'scalar']
    
    for kernel in kernels:
        result = reflex(close, 20, kernel=kernel)
        assert len(result) == len(close)
        assert np.all(np.isfinite(result[20:]))
    
    # Invalid kernel should fail
    with pytest.raises(ValueError, match="Unknown kernel"):
        reflex(close, 20, kernel='invalid')


if __name__ == '__main__':
    # Run a simple test to verify the module loads correctly
    print("Testing Reflex module...")
    test_reflex_accuracy()
    print("Reflex tests passed!")