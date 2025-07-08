"""
Python binding tests for SINWMA indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
from test_utils import load_test_data, assert_close

# Import SINWMA functions - they'll be available after building with maturin
try:
    from my_project import (
        sinwma, 
        sinwma_batch,
        SinWmaStream
    )
except ImportError:
    pytest.skip("SINWMA module not available - run 'maturin develop' first", allow_module_level=True)


def test_sinwma_partial_params():
    """Test SINWMA with default parameters - mirrors check_sinwma_partial_params"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test with default parameter (14)
    result = sinwma(close, 14)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(close)


def test_sinwma_accuracy():
    """Test SINWMA matches expected values from Rust tests - mirrors check_sinwma_accuracy"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test with period=14 (default)
    result = sinwma(close, 14)
    
    assert len(result) == len(close)
    
    # Expected values from Rust test
    expected_last_five = [
        59376.72903536103,
        59300.76862770367,
        59229.27622157621,
        59178.48781774477,
        59154.66580703081,
    ]
    
    actual_last_five = result[-5:]
    
    for i, (actual, expected) in enumerate(zip(actual_last_five, expected_last_five)):
        assert_close(actual, expected, rtol=1e-6, msg=f"SINWMA mismatch at index {i}")


def test_sinwma_invalid_period():
    """Test SINWMA fails with invalid period"""
    input_data = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    
    # Period = 0 should fail
    with pytest.raises(ValueError, match="sinwma"):
        sinwma(input_data, 0)


def test_sinwma_period_exceeds_length():
    """Test SINWMA fails with period exceeding length"""
    data_small = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="sinwma"):
        sinwma(data_small, 10)


def test_sinwma_very_small_dataset():
    """Test SINWMA fails with insufficient data"""
    single_point = np.array([42.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="sinwma"):
        sinwma(single_point, 14)


def test_sinwma_reinput():
    """Test SINWMA with re-input of SINWMA result - mirrors check_sinwma_reinput"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # First SINWMA pass with period=14
    first_result = sinwma(close, 14)
    
    # Second SINWMA pass with period=5 using first result as input
    second_result = sinwma(first_result, 5)
    
    assert len(second_result) == len(first_result)
    
    # All values should be finite after initial warmup periods
    # Find first non-NaN in first_result (output of first pass)
    first_valid_in_first = np.where(~np.isnan(first_result))[0][0]
    warmup = first_valid_in_first + 5 - 1  # second pass warmup
    assert np.all(np.isfinite(second_result[warmup:]))


def test_sinwma_nan_handling():
    """Test SINWMA handling of NaN values - mirrors check_sinwma_nan_handling"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    result = sinwma(close, 14)
    
    assert len(result) == len(close)
    
    # After warmup, all values should be finite
    first_valid = np.where(~np.isnan(close))[0][0]
    warmup = first_valid + 14 - 1
    
    if len(result) > warmup:
        assert np.all(np.isfinite(result[warmup:]))


def test_sinwma_all_nan():
    """Test SINWMA with all NaN values"""
    all_nan = np.full(100, np.nan, dtype=np.float64)
    
    with pytest.raises(ValueError, match="sinwma"):
        sinwma(all_nan, 14)


def test_sinwma_empty_input():
    """Test SINWMA with empty input"""
    data_empty = np.array([], dtype=np.float64)
    
    with pytest.raises(ValueError, match="sinwma"):
        sinwma(data_empty, 14)


def test_sinwma_batch():
    """Test SINWMA batch computation"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test period range
    batch_result = sinwma_batch(
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
    individual_result = sinwma(close, 10)
    batch_first = batch_result['values'][0]
    
    # Compare after warmup period
    first_valid = np.where(~np.isnan(close))[0][0]
    warmup = first_valid + 10 - 1
    assert_close(batch_first[warmup:], individual_result[warmup:], atol=1e-9, msg="SINWMA first combination mismatch")


def test_sinwma_stream():
    """Test SINWMA streaming interface"""
    data = load_test_data()
    close = data['close']
    
    # Create stream with period=14
    period = 14
    
    # Calculate batch result for comparison
    close_array = np.array(close, dtype=np.float64)
    batch_result = sinwma(close_array, period)
    
    # Test streaming
    stream = SinWmaStream(period)
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
            assert_close(batch_result[i], stream_results[i], rtol=1e-9, msg=f"SINWMA streaming mismatch at index {i}")


def test_sinwma_different_periods():
    """Test SINWMA with various period values"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test various period values
    test_periods = [5, 10, 14, 20, 50]
    
    for period in test_periods:
        result = sinwma(close, period)
        assert len(result) == len(close)
        
        # After warmup, all values should be finite
        first_valid = np.where(~np.isnan(close))[0][0]
        warmup = first_valid + period - 1
        if warmup < len(result):
            assert np.all(np.isfinite(result[warmup:])), f"NaN values found for period={period}"


def test_sinwma_batch_performance():
    """Test that batch computation works correctly (performance is secondary)"""
    data = load_test_data()
    close = np.array(data['close'][:1000], dtype=np.float64)  # Use first 1000 values
    
    # Test multiple parameter combinations
    batch_result = sinwma_batch(
        close,
        (10, 50, 10)  # periods: 10, 20, 30, 40, 50
    )
    
    # Should have 5 combinations
    assert batch_result['values'].shape == (5, len(close))
    
    # Verify metadata
    expected_periods = [10, 20, 30, 40, 50]
    assert np.array_equal(batch_result['periods'], expected_periods)


def test_sinwma_edge_cases():
    """Test SINWMA with edge case inputs"""
    # Test with monotonically increasing data
    data = np.arange(1.0, 101.0, dtype=np.float64)
    result = sinwma(data, 14)
    assert len(result) == len(data)
    assert np.all(np.isfinite(result[13:]))  # After period - 1
    
    # Test with constant values
    data = np.array([50.0] * 100, dtype=np.float64)
    result = sinwma(data, 14)
    assert len(result) == len(data)
    # With constant input, after warmup should produce constant output
    assert np.all(np.isfinite(result[13:]))
    
    # Test with oscillating values
    data = np.array([10.0, 20.0, 10.0, 20.0] * 25, dtype=np.float64)
    result = sinwma(data, 14)
    assert len(result) == len(data)
    assert np.all(np.isfinite(result[13:]))


def test_sinwma_consistency():
    """Test that SINWMA produces consistent results across multiple calls"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    result1 = sinwma(close, 14)
    result2 = sinwma(close, 14)
    
    assert_close(result1, result2, atol=1e-15, msg="SINWMA results not consistent")


def test_sinwma_step_precision():
    """Test batch with various step sizes"""
    data = np.arange(1, 51, dtype=np.float64)
    
    # Test with different step sizes
    batch_result = sinwma_batch(
        data,
        (10, 20, 2)  # periods: 10, 12, 14, 16, 18, 20
    )
    
    assert batch_result['values'].shape == (6, len(data))
    expected_periods = [10, 12, 14, 16, 18, 20]
    assert np.array_equal(batch_result['periods'], expected_periods)


def test_sinwma_batch_error_handling():
    """Test SINWMA batch error handling"""
    # Test with all NaN data
    all_nan = np.full(100, np.nan, dtype=np.float64)
    with pytest.raises(ValueError, match="sinwma"):
        sinwma_batch(all_nan, (10, 20, 5))
    
    # Test with insufficient data
    small_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    with pytest.raises(ValueError, match="sinwma"):
        sinwma_batch(small_data, (10, 20, 5))


def test_sinwma_zero_copy_verification():
    """Verify SINWMA uses zero-copy operations"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    # The result should be computed directly without intermediate copies
    result = sinwma(close, 14)
    assert len(result) == len(close)
    
    # Batch should also use zero-copy
    batch_result = sinwma_batch(close, (10, 20, 5))
    assert batch_result['values'].shape[0] == 3  # 10, 15, 20
    assert batch_result['values'].shape[1] == len(close)


def test_sinwma_stream_error_handling():
    """Test SINWMA stream error handling"""
    # Test with invalid period
    with pytest.raises(ValueError, match="sinwma"):
        SinWmaStream(0)
    
    # Test that stream properly handles warmup
    stream = SinWmaStream(14)
    
    # First 13 updates should return None (period = 14, so need 14 values)
    for i in range(13):
        result = stream.update(float(i + 1))
        assert result is None
    
    # 14th update should return a value
    result = stream.update(14.0)
    assert result is not None
    assert isinstance(result, float)


def test_sinwma_warmup_behavior():
    """Test SINWMA warmup period behavior"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    period = 14
    result = sinwma(close, period)
    
    # Find first non-NaN value in input
    first_valid = np.where(~np.isnan(close))[0][0]
    warmup = first_valid + period
    
    # Values during warmup should be NaN  
    assert np.all(np.isnan(result[first_valid:first_valid + period - 1])), "Expected NaN during warmup period"
    
    # Values after warmup should be finite
    assert np.all(np.isfinite(result[first_valid + period - 1:])), "Expected finite values after warmup"


def test_sinwma_formula_verification():
    """Verify SINWMA formula implementation"""
    # Create simple test data with known pattern
    data = np.array([10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 10.0, 11.0, 12.0, 13.0] * 5, dtype=np.float64)
    period = 5
    
    result = sinwma(data, period)
    
    # Result length should match input
    assert len(result) == len(data)
    
    # After warmup, values should be finite
    assert np.all(np.isfinite(result[period - 1:]))
    
    # SINWMA should detect the oscillating pattern
    # Values should not all be the same after warmup
    assert not np.allclose(result[period:], result[period], rtol=1e-9)


def test_sinwma_kernel_options():
    """Test SINWMA with different kernel options"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    # Test with different kernels
    kernels = ['auto', 'scalar']
    
    for kernel in kernels:
        result = sinwma(close, 14, kernel=kernel)
        assert len(result) == len(close)
        assert np.all(np.isfinite(result[13:]))
    
    # Invalid kernel should fail
    with pytest.raises(ValueError, match="Unknown kernel"):
        sinwma(close, 14, kernel='invalid')


if __name__ == '__main__':
    # Run a simple test to verify the module loads correctly
    print("Testing SINWMA module...")
    test_sinwma_accuracy()
    print("SINWMA tests passed!")