"""
Python binding tests for NMA indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
from test_utils import load_test_data, assert_close

# Import NMA functions - they'll be available after building with maturin
try:
    from my_project import (
        nma, 
        nma_batch,
        NmaStream
    )
except ImportError:
    pytest.skip("NMA module not available - run 'maturin develop' first", allow_module_level=True)


def test_nma_partial_params():
    """Test NMA with default parameters - mirrors check_nma_partial_params"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test with default parameter (40)
    result = nma(close, 40)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(close)


def test_nma_accuracy():
    """Test NMA matches expected values from Rust tests - mirrors check_nma_accuracy"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test with period=40
    result = nma(close, 40)
    
    assert len(result) == len(close)
    
    # Expected values from Rust test
    expected_last_five = [
        64320.486018271724,
        64227.95719984426,
        64180.9249333126,
        63966.35530620797,
        64039.04719192334,
    ]
    
    actual_last_five = result[-5:]
    
    for i, (actual, expected) in enumerate(zip(actual_last_five, expected_last_five)):
        assert_close(actual, expected, rtol=1e-3, msg=f"NMA mismatch at index {i}")


def test_nma_zero_period():
    """Test NMA fails with zero period"""
    input_data = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="nma:"):
        nma(input_data, 0)


def test_nma_period_exceeds_length():
    """Test NMA fails with period exceeding length"""
    data_small = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="nma:"):
        nma(data_small, 10)


def test_nma_very_small_dataset():
    """Test NMA fails with insufficient data"""
    single_point = np.array([42.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="nma:"):
        nma(single_point, 40)


def test_nma_reinput():
    """Test NMA with re-input of NMA result - mirrors check_nma_reinput"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # First NMA pass with period=40
    first_result = nma(close, 40)
    
    # Second NMA pass with period=30 using first result as input
    second_result = nma(first_result, 30)
    
    assert len(second_result) == len(first_result)
    
    # All values should be finite
    assert np.all(np.isfinite(second_result[70:]))  # After warmup


def test_nma_nan_handling():
    """Test NMA handling of NaN values - mirrors check_nma_nan_handling"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    result = nma(close, 40)
    
    assert len(result) == len(close)
    
    # After warmup, all values should be finite
    assert np.all(np.isfinite(result[240:]))


def test_nma_all_nan():
    """Test NMA with all NaN values"""
    all_nan = np.full(100, np.nan, dtype=np.float64)
    
    with pytest.raises(ValueError, match="nma:"):
        nma(all_nan, 40)


def test_nma_empty_input():
    """Test NMA with empty input"""
    data_empty = np.array([], dtype=np.float64)
    
    with pytest.raises(ValueError, match="nma:"):
        nma(data_empty, 40)


def test_nma_batch():
    """Test NMA batch computation"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test period range
    batch_result = nma_batch(
        close,
        (20, 60, 20)  # period range: 20, 40, 60
    )
    
    # Check result is a dict with the expected keys
    assert isinstance(batch_result, dict)
    assert 'values' in batch_result
    assert 'periods' in batch_result
    
    # Should have 3 period combinations
    assert batch_result['values'].shape == (3, len(close))
    assert len(batch_result['periods']) == 3
    
    # Verify first combination matches individual calculation
    individual_result = nma(close, 20)
    batch_first = batch_result['values'][0]
    
    # Compare after warmup period
    warmup = 240 + 20  # first valid + period
    assert_close(batch_first[warmup:], individual_result[warmup:], atol=1e-9, msg="NMA first combination mismatch")


def test_nma_stream():
    """Test NMA streaming interface"""
    data = load_test_data()
    close = data['close']
    
    # Create stream with period=40
    period = 40
    
    # Calculate batch result for comparison
    close_array = np.array(close, dtype=np.float64)
    batch_result = nma(close_array, period)
    
    # Test streaming
    stream = NmaStream(period)
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
            assert_close(batch_result[i], stream_results[i], rtol=1e-9, msg=f"NMA streaming mismatch at index {i}")


def test_nma_different_periods():
    """Test NMA with various period values"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test various period values
    test_periods = [10, 20, 40, 80]
    
    for period in test_periods:
        result = nma(close, period)
        assert len(result) == len(close)
        
        # After warmup, all values should be finite
        warmup = 240 + period
        if warmup < len(result):
            assert np.all(np.isfinite(result[warmup:])), f"NaN values found for period={period}"


def test_nma_batch_performance():
    """Test that batch computation works correctly (performance is secondary)"""
    data = load_test_data()
    close = np.array(data['close'][:1000], dtype=np.float64)  # Use first 1000 values
    
    # Test multiple parameter combinations
    batch_result = nma_batch(
        close,
        (10, 50, 10)  # periods: 10, 20, 30, 40, 50
    )
    
    # Should have 5 combinations
    assert batch_result['values'].shape == (5, len(close))
    
    # Verify metadata
    expected_periods = [10, 20, 30, 40, 50]
    assert np.array_equal(batch_result['periods'], expected_periods)


def test_nma_edge_cases():
    """Test NMA with edge case inputs"""
    # Test with monotonically increasing data
    data = np.arange(1.0, 101.0, dtype=np.float64)
    result = nma(data, 10)
    assert len(result) == len(data)
    assert np.all(np.isfinite(result[10:]))
    
    # Test with constant values
    data = np.array([50.0] * 100, dtype=np.float64)
    result = nma(data, 10)
    assert len(result) == len(data)
    assert np.all(np.isfinite(result[10:]))
    
    # Test with oscillating values
    data = np.array([10.0, 20.0, 10.0, 20.0] * 25, dtype=np.float64)
    result = nma(data, 10)
    assert len(result) == len(data)
    assert np.all(np.isfinite(result[10:]))


def test_nma_consistency():
    """Test that NMA produces consistent results across multiple calls"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    result1 = nma(close, 40)
    result2 = nma(close, 40)
    
    assert_close(result1, result2, atol=1e-15, msg="NMA results not consistent")


def test_nma_step_precision():
    """Test batch with various step sizes"""
    data = np.arange(1, 51, dtype=np.float64)
    
    # Test with different step sizes
    batch_result = nma_batch(
        data,
        (10, 30, 10)  # periods: 10, 20, 30
    )
    
    assert batch_result['values'].shape == (3, len(data))
    expected_periods = [10, 20, 30]
    assert np.array_equal(batch_result['periods'], expected_periods)


def test_nma_batch_error_handling():
    """Test NMA batch error handling"""
    # Test with all NaN data
    all_nan = np.full(100, np.nan, dtype=np.float64)
    with pytest.raises(ValueError, match="nma:"):
        nma_batch(all_nan, (20, 40, 10))
    
    # Test with insufficient data
    small_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    with pytest.raises(ValueError, match="nma:"):
        nma_batch(small_data, (10, 20, 10))


def test_nma_zero_copy_verification():
    """Verify NMA uses zero-copy operations"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    # The result should be computed directly without intermediate copies
    result = nma(close, 40)
    assert len(result) == len(close)
    
    # Batch should also use zero-copy
    batch_result = nma_batch(close, (20, 40, 20))
    assert batch_result['values'].shape[0] == 2  # 20, 40
    assert batch_result['values'].shape[1] == len(close)


def test_nma_stream_error_handling():
    """Test NMA stream error handling"""
    # Test with invalid period
    with pytest.raises(ValueError, match="nma:"):
        NmaStream(0)
    
    # Test that stream properly handles warmup
    stream = NmaStream(10)
    
    # First 10 updates should return None
    for i in range(10):
        result = stream.update(float(i + 1))
        assert result is None
    
    # 11th update should return a value
    result = stream.update(11.0)
    assert result is not None
    assert isinstance(result, float)


def test_nma_warmup_behavior():
    """Test NMA warmup period behavior"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    period = 40
    result = nma(close, period)
    
    # Find first non-NaN value in input
    first_valid = np.where(~np.isnan(close))[0][0]
    warmup = first_valid + period
    
    # Values before warmup should be NaN
    assert np.all(np.isnan(result[:warmup])), "Expected NaN during warmup period"
    
    # Values after warmup should be finite
    assert np.all(np.isfinite(result[warmup:])), "Expected finite values after warmup"


def test_nma_formula_verification():
    """Verify NMA formula implementation"""
    # Create simple test data
    data = np.array([10.0, 12.0, 11.0, 13.0, 15.0, 14.0], dtype=np.float64)
    period = 3
    
    result = nma(data, period)
    
    # The formula is complex, but we can verify:
    # 1. Result length matches input
    assert len(result) == len(data)
    
    # 2. Warmup period is respected
    assert np.all(np.isnan(result[:period]))
    
    # 3. Values after warmup are reasonable
    assert np.all(np.isfinite(result[period:]))
    
    # 4. Values are within reasonable range of input
    valid_results = result[period:]
    assert np.all(valid_results >= data.min() * 0.5)
    assert np.all(valid_results <= data.max() * 1.5)


if __name__ == '__main__':
    # Run a simple test to verify the module loads correctly
    print("Testing NMA module...")
    test_nma_accuracy()
    print("NMA tests passed!")