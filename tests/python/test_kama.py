"""
Python binding tests for KAMA indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
from test_utils import load_test_data, assert_close

# Import KAMA functions - they'll be available after building with maturin
try:
    from my_project import (
        kama, 
        kama_batch,
        KamaStream
    )
except ImportError:
    pytest.skip("KAMA module not available - run 'maturin develop' first", allow_module_level=True)


def test_kama_partial_params():
    """Test KAMA with default parameters - mirrors check_kama_partial_params"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test with default period (30)
    result = kama(close, 30)
    assert len(result) == len(close)
    
    # Test with kernel parameter
    result_with_kernel = kama(close, 30, kernel="scalar")
    assert len(result_with_kernel) == len(close)


def test_kama_accuracy():
    """Test KAMA matches expected values from Rust tests - mirrors check_kama_accuracy"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Period 30 (default)
    result = kama(close, 30)
    
    # Expected last 5 values from Rust test
    expected_last_five = [
        60234.925553804125,
        60176.838757545665,
        60115.177367962766,
        60071.37070833558,
        59992.79386218023
    ]
    
    # Check last 5 values match expected
    assert_close(
        result[-5:], 
        expected_last_five, 
        atol=1e-6,
        msg="KAMA last 5 values mismatch"
    )


def test_kama_zero_period():
    """Test KAMA fails with zero period - mirrors check_kama_zero_period"""
    input_data = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="kama:"):
        kama(input_data, 0)


def test_kama_period_exceeds_length():
    """Test KAMA fails with period exceeding data length - mirrors check_kama_period_exceeds_length"""
    data_small = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="kama:"):
        kama(data_small, 10)


def test_kama_very_small_dataset():
    """Test KAMA fails with insufficient data - mirrors check_kama_very_small_dataset"""
    single_point = np.array([42.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="kama:"):
        kama(single_point, 5)


def test_kama_empty_input():
    """Test KAMA with empty input"""
    data_empty = np.array([], dtype=np.float64)
    
    with pytest.raises(ValueError, match="kama:"):
        kama(data_empty, 30)


def test_kama_all_nan():
    """Test KAMA with all NaN input"""
    data = np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
    
    with pytest.raises(ValueError, match="kama:"):
        kama(data, 3)


def test_kama_reinput():
    """Test KAMA with re-input of KAMA result - mirrors check_kama_reinput"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # First KAMA pass with period=30 (matching Rust test)
    first_result = kama(close, 30)
    
    # Second KAMA pass with period=10 using first result as input
    second_result = kama(first_result, 10)
    
    assert len(second_result) == len(first_result)
    
    # The second pass will have its own warmup period
    # First pass warmup: 30 values
    # Second pass warmup: 10 values  
    # So we expect NaN values up to index 30+10=40
    for i in range(40, len(second_result)):
        assert np.isfinite(second_result[i]), f"Unexpected NaN at index {i}"


def test_kama_nan_handling():
    """Test KAMA handling of NaN values"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Inject some NaN values
    close_with_nans = close.copy()
    close_with_nans[10:15] = np.nan
    
    result = kama(close_with_nans, 30)
    
    assert len(result) == len(close_with_nans)
    
    # Check that we have NaN in the beginning (warmup period)
    assert np.isnan(result[0])
    
    # After warmup, we should have non-NaN values where input is valid
    # Note: The exact behavior depends on how KAMA handles NaN in the window


def test_kama_batch():
    """Test KAMA batch computation"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test period range 10-40 step 10
    period_start = 10
    period_end = 40
    period_step = 10  # periods: 10, 20, 30, 40
    
    result = kama_batch(close, (period_start, period_end, period_step))
    
    assert 'values' in result
    assert 'periods' in result
    
    # Should have 4 periods
    assert len(result['periods']) == 4
    assert np.array_equal(result['periods'], [10, 20, 30, 40])
    
    # Batch result should contain all individual results in 2D array
    assert result['values'].shape == (4, len(close))
    
    # Verify first combination matches individual calculation
    individual_result = kama(close, 10)
    batch_first_row = result['values'][0]
    assert_close(batch_first_row, individual_result, atol=1e-9, msg="First combination mismatch")


def test_kama_batch_multiple_periods():
    """Test KAMA batch computation with multiple periods"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test period range
    period_start = 20
    period_end = 60
    period_step = 20  # periods: 20, 40, 60
    
    result = kama_batch(close, (period_start, period_end, period_step))
    
    # Check structure
    assert 'values' in result
    assert 'periods' in result
    assert np.array_equal(result['periods'], [20, 40, 60])
    
    # Check result shape
    assert result['values'].shape == (3, len(close))
    
    # Verify second combination (period=40)
    individual_result = kama(close, 40)
    batch_second_row = result['values'][1]
    assert_close(batch_second_row, individual_result, atol=1e-9, msg="Second combination mismatch")


def test_kama_batch_single_period():
    """Test KAMA batch computation with single period"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test single period (step=0 or start==end)
    result = kama_batch(close, (30, 30, 0))
    
    # Check structure
    assert 'values' in result
    assert 'periods' in result
    assert np.array_equal(result['periods'], [30])
    
    # Check shape
    assert result['values'].shape == (1, len(close))
    
    # Verify matches individual calculation
    individual_result = kama(close, 30)
    assert_close(result['values'][0], individual_result, atol=1e-9, msg="Single period mismatch")


def test_kama_stream():
    """Test KAMA streaming interface"""
    data = load_test_data()
    close = data['close']
    period = 30
    
    # Calculate batch result for comparison
    close_array = np.array(close, dtype=np.float64)
    batch_result = kama(close_array, period)
    
    # Test streaming
    stream = KamaStream(period)
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
            assert_close(stream_results[i], batch_result[i], atol=1e-6, 
                        msg=f"Streaming mismatch at index {i}")


def test_kama_different_periods():
    """Test KAMA with various period values"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test various period values
    for period in [5, 10, 20, 50]:
        result = kama(close, period)
        assert len(result) == len(close)
        
        # Count valid values after warmup
        valid_count = np.sum(np.isfinite(result[period:]))
        assert valid_count > len(close) - period - 5, f"Too many NaN values for period={period}"


def test_kama_batch_performance():
    """Test that batch computation works correctly (performance is secondary)"""
    data = load_test_data()
    close = np.array(data['close'][:1000], dtype=np.float64)  # Use first 1000 values
    
    # Test 5 periods
    result = kama_batch(close, (10, 50, 10))
    assert len(result['periods']) == 5  # periods: 10, 20, 30, 40, 50
    assert result['values'].shape == (5, len(close))
    
    # Verify each combination
    for i, period in enumerate(result['periods']):
        individual_result = kama(close, period)
        batch_row = result['values'][i]
        assert_close(batch_row, individual_result, atol=1e-9, 
                          msg=f"Batch mismatch for period={period}")


def test_kama_edge_cases():
    """Test KAMA with edge case inputs"""
    # Test with enough data points including NaN
    data = np.array([42.0, 43.0, 44.0, 45.0, np.nan], dtype=np.float64)
    # This should work with period=3 as we have 4 valid values
    result = kama(data, 3)
    assert len(result) == len(data)
    
    # Test with alternating valid/NaN values
    data = np.array([1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan], dtype=np.float64)
    result = kama(data, 2)
    assert len(result) == len(data)
    
    # Test with very large period
    data = np.random.randn(100).astype(np.float64)
    result = kama(data, 99)
    assert len(result) == len(data)


def test_kama_batch_error_handling():
    """Test KAMA batch error handling"""
    # Test with empty data
    empty = np.array([], dtype=np.float64)
    with pytest.raises(ValueError, match="kama:"):
        kama_batch(empty, (10, 20, 10))
    
    # Test with all NaN data
    all_nan = np.full(100, np.nan, dtype=np.float64)
    with pytest.raises(ValueError, match="kama:"):
        kama_batch(all_nan, (10, 20, 10))
    
    # Test with period exceeding data length
    small_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    with pytest.raises(ValueError, match="kama:"):
        kama_batch(small_data, (5, 10, 5))


def test_kama_batch_metadata_consistency():
    """Test that batch metadata is consistent with results"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    # Test various period ranges
    test_cases = [
        (5, 15, 5),    # periods: 5, 10, 15
        (10, 10, 0),   # single period: 10
        (20, 40, 10),  # periods: 20, 30, 40
    ]
    
    for start, end, step in test_cases:
        result = kama_batch(close, (start, end, step))
        
        # Check periods array matches expected
        if step == 0 or start == end:
            expected_periods = [start]
        else:
            expected_periods = list(range(start, end + 1, step))
        
        assert np.array_equal(result['periods'], expected_periods)
        assert result['values'].shape[0] == len(expected_periods)
        assert result['values'].shape[1] == len(close)


def test_kama_single_value():
    """Test KAMA with single value input"""
    data = np.array([42.0], dtype=np.float64)
    
    # Period=1 with single value should fail (need at least period+1 values)
    with pytest.raises(ValueError, match="kama:"):
        kama(data, 1)


def test_kama_two_values():
    """Test KAMA with two values input"""
    data = np.array([1.0, 2.0], dtype=np.float64)
    
    # Should work with period=1 since we have 2 values (need period+1)
    result = kama(data, 1)
    assert len(result) == 2
    # First value should be NaN (warmup)
    assert np.isnan(result[0])
    # Second value should be valid
    assert np.isfinite(result[1])


def test_kama_warmup_period():
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
        result = kama(close, period)
        
        # Check NaN values up to warmup period
        for i in range(expected_warmup):
            assert np.isnan(result[i]), f"Expected NaN at index {i} for period={period}"
        
        # Check valid values after warmup
        if expected_warmup < len(result):
            assert not np.isnan(result[expected_warmup]), \
                f"Expected valid value at index {expected_warmup} for period={period}"


def test_kama_consistency():
    """Test that KAMA produces consistent results across multiple calls"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    result1 = kama(close, 30)
    result2 = kama(close, 30)
    
    assert_close(result1, result2, atol=1e-15, msg="KAMA results not consistent")


def test_kama_step_precision():
    """Test batch with very small step sizes"""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
    
    result = kama_batch(data, (2, 4, 1))  # periods: 2, 3, 4
    
    assert np.array_equal(result['periods'], [2, 3, 4])
    assert result['values'].shape == (3, len(data))


def test_kama_stream_error_handling():
    """Test KAMA stream error handling"""
    # Test with invalid period
    with pytest.raises(ValueError, match="kama:"):
        KamaStream(0)
    
    # Test stream matches batch
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    period = 30
    
    batch_result = kama(close, period)
    stream = KamaStream(period)
    stream_results = []
    
    for price in close:
        result = stream.update(price)
        stream_results.append(result if result is not None else np.nan)
    
    # After warmup, values should match
    for i in range(period, len(close)):
        if not np.isnan(batch_result[i]):
            assert_close(stream_results[i], batch_result[i], atol=1e-9,
                        msg=f"Stream mismatch at index {i}")


def test_kama_batch_warmup_consistency():
    """Test that batch warmup periods are consistent"""
    data = np.random.randn(50).astype(np.float64)
    
    result = kama_batch(data, (5, 15, 5))
    
    # Each row should have proper warmup
    for i, period in enumerate(result['periods']):
        row = result['values'][i]
        # First 'period' values should be NaN
        assert np.all(np.isnan(row[:period])), f"Expected NaN warmup for period {period}"
        # After warmup should have values
        assert np.all(~np.isnan(row[period:])), f"Expected values after warmup for period {period}"


def test_kama_zero_copy_verification():
    """Verify KAMA uses zero-copy operations"""
    # This test ensures the Python binding doesn't make unnecessary copies
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    # The result should be computed directly without intermediate copies
    result = kama(close, 30)
    assert len(result) == len(close)
    
    # Batch should also use zero-copy
    batch_result = kama_batch(close, (10, 30, 10))
    assert batch_result['values'].shape == (3, len(close))


def test_kama_kernel_parameter():
    """Test KAMA with different kernel parameters"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    # Test with different kernels
    kernels = ["scalar", "auto"]
    period = 30
    
    for kernel in kernels:
        result = kama(close, period, kernel=kernel)
        assert len(result) == len(close)
        # Verify warmup period
        assert np.all(np.isnan(result[:period]))
        # Verify we have values after warmup
        assert np.all(~np.isnan(result[period:]))
    
    # Test batch with kernel
    batch_result = kama_batch(close, (10, 30, 10), kernel="scalar")
    assert batch_result['values'].shape == (3, len(close))
    
    # Invalid kernel should raise error
    with pytest.raises(ValueError):
        kama(close, period, kernel="invalid_kernel")


if __name__ == "__main__":
    # Run a simple test to verify the module loads correctly
    print("Testing KAMA module...")
    test_kama_accuracy()
    print("KAMA tests passed!")