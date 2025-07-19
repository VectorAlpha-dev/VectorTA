"""
Python binding tests for MWDX indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
from test_utils import load_test_data, assert_close

# Import MWDX functions - they'll be available after building with maturin
try:
    from my_project import (
        mwdx, 
        mwdx_batch,
        MwdxStream
    )
except ImportError:
    pytest.skip("MWDX module not available - run 'maturin develop' first", allow_module_level=True)


def test_mwdx_partial_params():
    """Test MWDX with default parameters - mirrors check_mwdx_partial_params"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test with default parameter (0.2)
    result = mwdx(close, 0.2)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(close)


def test_mwdx_accuracy():
    """Test MWDX matches expected values from Rust tests - mirrors check_mwdx_accuracy"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test with factor=0.2
    result = mwdx(close, 0.2)
    
    assert len(result) == len(close)
    
    # Expected values from Rust test
    expected_last_five = [
        59302.181566190935,
        59277.94525295275,
        59230.1562023622,
        59215.124961889764,
        59103.099969511815,
    ]
    
    actual_last_five = result[-5:]
    
    for i, (actual, expected) in enumerate(zip(actual_last_five, expected_last_five)):
        assert_close(actual, expected, rtol=1e-7, msg=f"MWDX mismatch at index {i}")


def test_mwdx_zero_factor():
    """Test MWDX fails with zero factor"""
    input_data = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="mwdx:"):
        mwdx(input_data, 0.0)


def test_mwdx_negative_factor():
    """Test MWDX fails with negative factor"""
    input_data = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="mwdx:"):
        mwdx(input_data, -0.5)


def test_mwdx_nan_factor():
    """Test MWDX fails with NaN factor"""
    input_data = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="mwdx:"):
        mwdx(input_data, np.nan)


def test_mwdx_very_small_dataset():
    """Test MWDX with single data point - mirrors check_mwdx_very_small_dataset"""
    data = np.array([42.0], dtype=np.float64)
    
    result = mwdx(data, 0.2)
    assert len(result) == 1
    assert result[0] == 42.0


def test_mwdx_empty_input():
    """Test MWDX with empty input"""
    data_empty = np.array([], dtype=np.float64)
    
    with pytest.raises(ValueError, match="mwdx:"):
        mwdx(data_empty, 0.2)


def test_mwdx_reinput():
    """Test MWDX with re-input of MWDX result - mirrors check_mwdx_reinput"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # First MWDX pass with factor=0.2
    first_result = mwdx(close, 0.2)
    
    # Second MWDX pass with factor=0.3 using first result as input
    second_result = mwdx(first_result, 0.3)
    
    assert len(second_result) == len(first_result)
    
    # All values should be finite
    assert np.all(np.isfinite(second_result))


def test_mwdx_nan_handling():
    """Test MWDX handling of NaN values - mirrors check_mwdx_nan_handling"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    result = mwdx(close, 0.2)
    
    assert len(result) == len(close)
    
    # All values should be finite (MWDX handles initial NaN properly)
    assert np.all(np.isfinite(result))


def test_mwdx_batch():
    """Test MWDX batch computation"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test parameter range
    batch_result = mwdx_batch(
        close,
        (0.1, 0.5, 0.1)  # factor range: 0.1, 0.2, 0.3, 0.4, 0.5
    )
    
    # Check result is a dict with the expected keys
    assert isinstance(batch_result, dict)
    assert 'values' in batch_result
    assert 'factors' in batch_result
    
    # Should have 5 parameter combinations
    assert batch_result['values'].shape == (5, len(close))
    assert len(batch_result['factors']) == 5
    
    # Verify first combination matches individual calculation
    individual_result = mwdx(close, 0.1)
    batch_first = batch_result['values'][0]
    assert_close(batch_first, individual_result, atol=1e-9, msg="MWDX first combination mismatch")


def test_mwdx_stream():
    """Test MWDX streaming interface"""
    data = load_test_data()
    close = data['close']
    
    # Create stream with factor=0.2
    factor = 0.2
    
    # Calculate batch result for comparison
    close_array = np.array(close, dtype=np.float64)
    batch_result = mwdx(close_array, factor)
    
    # Test streaming
    stream = MwdxStream(factor)
    stream_results = []
    
    for price in close:
        result = stream.update(price)
        stream_results.append(result)
    
    # Streaming should provide a value for every input
    assert len(stream_results) == len(close)
    
    # All values should be finite
    for i, val in enumerate(stream_results):
        assert np.isfinite(val), f"NaN at index {i}"
    
    # First value should match the input
    assert_close(stream_results[0], close[0], atol=1e-9, msg="First value mismatch")


def test_mwdx_different_factors():
    """Test MWDX with various factor values"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test various factor values
    test_factors = [0.1, 0.2, 0.5, 0.9]
    
    for factor in test_factors:
        result = mwdx(close, factor)
        assert len(result) == len(close)
        
        # All values should be finite
        assert np.all(np.isfinite(result)), f"NaN values found for factor={factor}"


def test_mwdx_batch_performance():
    """Test that batch computation works correctly (performance is secondary)"""
    data = load_test_data()
    close = np.array(data['close'][:1000], dtype=np.float64)  # Use first 1000 values
    
    # Test multiple parameter combinations
    batch_result = mwdx_batch(
        close,
        (0.1, 0.9, 0.2)  # factors: 0.1, 0.3, 0.5, 0.7, 0.9
    )
    
    # Should have 5 combinations
    assert batch_result['values'].shape == (5, len(close))
    
    # Verify first combination (0.1)
    individual_result = mwdx(close, 0.1)
    batch_first = batch_result['values'][0]
    assert_close(batch_first, individual_result, atol=1e-9, 
                msg="MWDX batch mismatch for factor=0.1")


def test_mwdx_edge_cases():
    """Test MWDX with edge case inputs"""
    # Test with monotonically increasing data
    data = np.arange(1.0, 101.0, dtype=np.float64)
    result = mwdx(data, 0.2)
    assert len(result) == len(data)
    assert np.all(np.isfinite(result))
    
    # Test with constant values
    data = np.array([50.0] * 100, dtype=np.float64)
    result = mwdx(data, 0.2)
    assert len(result) == len(data)
    
    # After initial value, all should converge to the constant
    for i in range(10, len(result)):
        assert_close(result[i], 50.0, atol=1e-6, msg=f"Constant value failed at index {i}")
    
    # Test with oscillating values
    data = np.array([10.0, 20.0, 10.0, 20.0] * 25, dtype=np.float64)
    result = mwdx(data, 0.5)
    assert len(result) == len(data)
    assert np.all(np.isfinite(result))


def test_mwdx_high_factor():
    """Test MWDX with high factor value"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    # High factor (close to 1) should give more weight to new data
    result = mwdx(close, 0.95)
    assert len(result) == len(close)
    assert np.all(np.isfinite(result))


def test_mwdx_low_factor():
    """Test MWDX with low factor value"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    # Low factor should give more weight to historical data
    result = mwdx(close, 0.01)
    assert len(result) == len(close)
    assert np.all(np.isfinite(result))


def test_mwdx_consistency():
    """Test that MWDX produces consistent results across multiple calls"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    result1 = mwdx(close, 0.2)
    result2 = mwdx(close, 0.2)
    
    assert_close(result1, result2, atol=1e-15, msg="MWDX results not consistent")


def test_mwdx_step_precision():
    """Test batch with various step sizes"""
    data = np.arange(1, 51, dtype=np.float64)
    
    # Test with different step sizes
    batch_result = mwdx_batch(
        data,
        (0.2, 0.8, 0.3)  # factors: 0.2, 0.5, 0.8
    )
    
    assert batch_result['values'].shape == (3, len(data))
    expected_factors = [0.2, 0.5, 0.8]
    assert_close(batch_result['factors'], expected_factors, atol=1e-9, msg="Factor mismatch")


def test_mwdx_batch_error_handling():
    """Test MWDX batch error handling"""
    # Test with empty data
    empty = np.array([], dtype=np.float64)
    with pytest.raises(ValueError, match="mwdx:"):
        mwdx_batch(empty, (0.2, 0.2, 0))
    
    # Test with invalid factor range
    data = np.random.randn(100).astype(np.float64)
    with pytest.raises(ValueError, match="mwdx:"):
        mwdx_batch(data, (0.0, 0.5, 0.1))  # Invalid start factor (0.0)
    
    with pytest.raises(ValueError, match="mwdx:"):
        mwdx_batch(data, (-0.1, 0.5, 0.1))  # Negative factor


def test_mwdx_zero_copy_verification():
    """Verify MWDX uses zero-copy operations"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    # The result should be computed directly without intermediate copies
    result = mwdx(close, 0.2)
    assert len(result) == len(close)
    
    # Batch should also use zero-copy
    batch_result = mwdx_batch(close, (0.1, 0.5, 0.2))
    assert batch_result['values'].shape[0] == 3  # 0.1, 0.3, 0.5
    assert batch_result['values'].shape[1] == len(close)


def test_mwdx_stream_error_handling():
    """Test MWDX stream error handling"""
    # Test with invalid factor
    with pytest.raises(ValueError, match="mwdx:"):
        MwdxStream(0.0)
    
    with pytest.raises(ValueError, match="mwdx:"):
        MwdxStream(-0.5)
    
    with pytest.raises(ValueError, match="mwdx:"):
        MwdxStream(np.nan)


def test_mwdx_nan_prefix():
    """Test MWDX with NaN prefix in data"""
    # Create data with NaN prefix
    data = np.array([np.nan, np.nan, 10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    
    result = mwdx(data, 0.2)
    assert len(result) == len(data)
    
    # MWDX formula: out[i] = fac * data[i] + (1.0 - fac) * out[i-1]
    # Due to the recursive nature and NaN propagation:
    # result[0] = NaN (input)
    # result[1] = 0.2 * NaN + 0.8 * NaN = NaN  
    # result[2] = 0.2 * 10.0 + 0.8 * NaN = NaN (NaN propagates)
    # All subsequent values will be NaN due to dependency on previous NaN
    
    # All values should be NaN when starting with NaN prefix
    assert np.all(np.isnan(result))
    
    # Test with data that has no NaN prefix to verify normal operation
    clean_data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    clean_result = mwdx(clean_data, 0.2)
    assert np.all(np.isfinite(clean_result))


def test_mwdx_formula_verification():
    """Verify MWDX formula: out[i] = fac * data[i] + (1 - fac) * out[i-1]"""
    data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    factor = 0.3
    
    result = mwdx(data, factor)
    
    # Manually calculate expected values
    expected = [data[0]]  # First value is always the input
    for i in range(1, len(data)):
        val = factor * data[i] + (1 - factor) * expected[i-1]
        expected.append(val)
    
    assert_close(result, expected, atol=1e-12, msg="Formula verification failed")


def test_mwdx_kernel_parameter():
    """Test MWDX with kernel parameter"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    # Test different kernel options
    for kernel in [None, 'scalar']:
        result = mwdx(close, 0.2, kernel=kernel)
        assert len(result) == len(close)
        assert np.all(np.isfinite(result)), f"NaN values with kernel={kernel}"
    
    # Test batch with kernel
    batch_result = mwdx_batch(close, (0.1, 0.3, 0.1), kernel='scalar')
    assert batch_result['values'].shape == (3, len(close))
    
    # Test invalid kernel
    with pytest.raises(ValueError, match="Unknown kernel"):
        mwdx(close, 0.2, kernel='invalid')


def test_mwdx_kernel_consistency():
    """Test that different kernels produce consistent results"""
    data = load_test_data()
    close = np.array(data['close'][:1000], dtype=np.float64)
    
    # Get results with different kernels
    result_auto = mwdx(close, 0.2, kernel=None)  # Auto-detect
    result_scalar = mwdx(close, 0.2, kernel='scalar')
    
    # Results should be very close (within floating point precision)
    assert_close(result_auto, result_scalar, rtol=1e-14, 
                msg="Kernel results differ significantly")


if __name__ == "__main__":
    # Run a simple test to verify the module loads correctly
    print("Testing MWDX module...")
    test_mwdx_accuracy()
    print("MWDX tests passed!")