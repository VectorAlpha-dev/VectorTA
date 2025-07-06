"""
Python binding tests for MAAQ indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
from test_utils import load_test_data, assert_close

# Import MAAQ functions - they'll be available after building with maturin
try:
    from my_project import (
        maaq, 
        maaq_batch,
        maaq_batch_with_metadata,
        maaq_batch_2d,
        MaaqStream
    )
except ImportError:
    pytest.skip("MAAQ module not available - run 'maturin develop' first", allow_module_level=True)


def test_maaq_partial_params():
    """Test MAAQ with default parameters - mirrors check_maaq_partial_params"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test with default periods (11, 2, 30)
    result = maaq(close, 11, 2, 30)
    assert len(result) == len(close)


def test_maaq_accuracy():
    """Test MAAQ matches expected values from Rust tests - mirrors check_maaq_accuracy"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Default parameters
    result = maaq(close, 11, 2, 30)
    
    # Expected last 5 values from Rust test
    expected_last_five = [
        59747.657115949725,
        59740.803138018055,
        59724.24153333905,
        59720.60576365108,
        59673.9954445178,
    ]
    
    # Check last 5 values match expected
    assert_close(
        result[-5:], 
        expected_last_five, 
        atol=1e-2,
        msg="MAAQ last 5 values mismatch"
    )


def test_maaq_zero_period():
    """Test MAAQ fails with zero period - mirrors check_maaq_zero_period"""
    input_data = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    
    # Test with zero period
    with pytest.raises(ValueError, match="maaq:"):
        maaq(input_data, 0, 2, 30)
    
    # Test with zero fast_period
    with pytest.raises(ValueError, match="maaq:"):
        maaq(input_data, 11, 0, 30)
    
    # Test with zero slow_period
    with pytest.raises(ValueError, match="maaq:"):
        maaq(input_data, 11, 2, 0)


def test_maaq_period_exceeds_length():
    """Test MAAQ fails with period exceeding data length - mirrors check_maaq_period_exceeds_length"""
    data_small = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="maaq:"):
        maaq(data_small, 10, 2, 30)


def test_maaq_very_small_dataset():
    """Test MAAQ fails with insufficient data - mirrors check_maaq_very_small_dataset"""
    single_point = np.array([42.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="maaq:"):
        maaq(single_point, 11, 2, 30)


def test_maaq_empty_input():
    """Test MAAQ with empty input"""
    data_empty = np.array([], dtype=np.float64)
    
    with pytest.raises(ValueError, match="maaq:"):
        maaq(data_empty, 11, 2, 30)


def test_maaq_all_nan():
    """Test MAAQ with all NaN input"""
    data = np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
    
    with pytest.raises(ValueError, match="maaq:"):
        maaq(data, 3, 2, 5)


def test_maaq_reinput():
    """Test MAAQ with re-input of MAAQ result - mirrors check_maaq_reinput"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # First MAAQ pass with default params
    first_result = maaq(close, 11, 2, 30)
    
    # Second MAAQ pass with different params using first result as input
    second_result = maaq(first_result, 20, 3, 25)
    
    assert len(second_result) == len(first_result)
    
    # The second pass will have its own warmup period
    # Check that we have some valid values
    for i in range(40, len(second_result)):
        assert np.isfinite(second_result[i]), f"Unexpected NaN at index {i}"


def test_maaq_nan_handling():
    """Test MAAQ handling of NaN values - mirrors check_maaq_nan_handling"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Use clean data like the Rust test does
    result = maaq(close, 11, 2, 30)
    
    assert len(result) == len(close)
    
    # The Rust test checks that after index 240, there are no NaN values
    # This implies the warmup period creates NaN values at the beginning
    if len(result) > 240:
        for i in range(240, len(result)):
            assert not np.isnan(result[i]), f"Found unexpected NaN at index {i}"


def test_maaq_batch():
    """Test MAAQ batch computation"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test period range 11-50 step 10, static fast/slow
    batch_result = maaq_batch(
        close, 
        11, 41, 10,      # period range (end is inclusive, so 11, 21, 31, 41)
        2, 2, 0,         # fast_period static
        30, 30, 0        # slow_period static
    )
    
    # Should have 4 periods: 11, 21, 31, 41
    # Batch result should contain all individual results flattened
    assert len(batch_result) == 4 * len(close)
    
    # Verify first combination matches individual calculation
    individual_result = maaq(close, 11, 2, 30)
    batch_first_row = batch_result[:len(close)]
    assert_close(batch_first_row, individual_result, atol=1e-9, msg="First combination mismatch")


def test_maaq_batch_with_metadata():
    """Test MAAQ batch computation with metadata"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test with varying all parameters
    batch_result, metadata = maaq_batch_with_metadata(
        close, 
        11, 31, 10,      # period range: 11, 21, 31
        2, 4, 2,         # fast_period range: 2, 4
        25, 35, 10       # slow_period range: 25, 35
    )
    
    # Check metadata contains all combinations (3 * 2 * 2 = 12)
    assert len(metadata) == 12
    
    # Check result shape
    assert len(batch_result) == 12 * len(close)
    
    # Verify a specific combination
    # Find the index for (21, 2, 25)
    idx = metadata.index((21, 2, 25))
    individual_result = maaq(close, 21, 2, 25)
    batch_row = batch_result[idx * len(close):(idx + 1) * len(close)]
    assert_close(batch_row, individual_result, atol=1e-9, msg="Specific combination mismatch")


def test_maaq_batch_2d():
    """Test MAAQ batch computation with 2D output"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test with simple parameter ranges
    batch_result_2d, metadata = maaq_batch_2d(
        close, 
        11, 31, 20,      # period range: 11, 31
        2, 3, 1,         # fast_period range: 2, 3
        30, 30, 0        # slow_period static: 30
    )
    
    # Check metadata (2 * 2 * 1 = 4 combinations)
    assert metadata == [(11, 2, 30), (11, 3, 30), (31, 2, 30), (31, 3, 30)]
    
    # Check shape
    assert batch_result_2d.shape == (4, len(close))
    
    # Verify a middle row
    individual_result = maaq(close, 11, 3, 30)
    assert_close(batch_result_2d[1], individual_result, atol=1e-9, msg="2D row mismatch")


def test_maaq_stream():
    """Test MAAQ streaming interface - mirrors check_maaq_streaming"""
    data = load_test_data()
    close = data['close']
    
    # Default parameters
    period = 11
    fast_period = 2
    slow_period = 30
    
    # Calculate batch result for comparison
    close_array = np.array(close, dtype=np.float64)
    batch_result = maaq(close_array, period, fast_period, slow_period)
    
    # Test streaming
    stream = MaaqStream(period, fast_period, slow_period)
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
        assert_close(stream_results[i], batch_result[i], atol=1e-9, 
                    msg=f"Streaming mismatch at index {i}")


def test_maaq_different_periods():
    """Test MAAQ with various period values"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test various period combinations
    test_cases = [
        (5, 2, 10),
        (10, 3, 20),
        (20, 5, 40),
        (50, 10, 100),
    ]
    
    for period, fast_p, slow_p in test_cases:
        result = maaq(close, period, fast_p, slow_p)
        assert len(result) == len(close)
        
        # Count valid values after warmup
        valid_count = np.sum(np.isfinite(result[period:]))
        assert valid_count > len(close) - period - 5, \
            f"Too many NaN values for params=({period}, {fast_p}, {slow_p})"


def test_maaq_batch_performance():
    """Test that batch computation works correctly (performance is secondary)"""
    data = load_test_data()
    close = np.array(data['close'][:1000], dtype=np.float64)  # Use first 1000 values
    
    # Test multiple period combinations
    batch_result = maaq_batch(
        close, 
        10, 30, 10,      # periods: 10, 20, 30
        2, 2, 0,         # fast_period fixed at 2
        25, 35, 5        # slow_periods: 25, 30, 35
    )
    
    # Should have 9 combinations: 3 periods * 1 fast * 3 slow = 9
    assert len(batch_result) == 9 * len(close)
    
    # Verify first combination (10, 2, 25)
    individual_result = maaq(close, 10, 2, 25)
    batch_first_row = batch_result[:len(close)]
    assert_close(batch_first_row, individual_result, atol=1e-9, 
                msg=f"Batch mismatch for params=(10, 2, 25)")


def test_maaq_edge_cases():
    """Test MAAQ with edge case inputs"""
    # Test with monotonically increasing data
    data = np.arange(1.0, 101.0, dtype=np.float64)
    result = maaq(data, 10, 2, 20)
    assert len(result) == len(data)
    
    # After warmup, values should be smoothed
    assert np.all(np.isfinite(result[10:]))
    
    # Test with alternating values
    data = np.array([1.0, 10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 10.0] * 20, dtype=np.float64)
    result = maaq(data, 5, 2, 10)
    assert len(result) == len(data)
    
    # Test with constant values
    data = np.array([5.0] * 100, dtype=np.float64)
    result = maaq(data, 10, 2, 20)
    assert len(result) == len(data)
    # With constant data, MAAQ should converge to the constant value
    for i in range(20, len(result)):
        assert_close(result[i], 5.0, atol=1e-9, msg=f"Constant value failed at index {i}")


def test_maaq_warmup_period():
    """Test that warmup period is correctly calculated"""
    data = load_test_data()
    close = np.array(data['close'][:50], dtype=np.float64)
    
    test_cases = [
        (5, 2, 10),    # period=5
        (10, 3, 20),   # period=10
        (20, 5, 30),   # period=20
        (30, 10, 40),  # period=30
    ]
    
    for period, fast_p, slow_p in test_cases:
        result = maaq(close, period, fast_p, slow_p)
        
        # MAAQ outputs equal to input during warmup period
        for i in range(period):
            assert_close(result[i], close[i], atol=1e-9,
                        msg=f"Warmup value mismatch at index {i} for period={period}")


def test_maaq_consistency():
    """Test that MAAQ produces consistent results across multiple calls"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    result1 = maaq(close, 11, 2, 30)
    result2 = maaq(close, 11, 2, 30)
    
    assert_close(result1, result2, atol=1e-15, msg="MAAQ results not consistent")


def test_maaq_step_precision():
    """Test batch with very small step sizes"""
    data = np.arange(1, 51, dtype=np.float64)
    
    # Use batch_with_metadata to get the metadata
    batch_result, metadata = maaq_batch_with_metadata(
        data, 
        5, 7, 1,         # periods: 5, 6, 7
        2, 3, 1,         # fast_periods: 2, 3
        10, 10, 0        # slow_period: 10
    )
    
    assert metadata == [
        (5, 2, 10), (5, 3, 10),
        (6, 2, 10), (6, 3, 10),
        (7, 2, 10), (7, 3, 10)
    ]
    assert len(batch_result) == 6 * len(data)


if __name__ == "__main__":
    # Run a simple test to verify the module loads correctly
    print("Testing MAAQ module...")
    test_maaq_accuracy()
    print("MAAQ tests passed!")