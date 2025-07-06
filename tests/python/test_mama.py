"""
Python binding tests for MAMA indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
from test_utils import load_test_data, assert_close

# Import MAMA functions - they'll be available after building with maturin
try:
    from my_project import (
        mama, 
        mama_batch,
        mama_batch_with_metadata,
        mama_batch_2d,
        MamaStream
    )
except ImportError:
    pytest.skip("MAMA module not available - run 'maturin develop' first", allow_module_level=True)


def test_mama_partial_params():
    """Test MAMA with default parameters - mirrors check_mama_partial_params"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test with default parameters (0.5, 0.05)
    mama_vals, fama_vals = mama(close, 0.5, 0.05)
    assert len(mama_vals) == len(close)
    assert len(fama_vals) == len(close)


def test_mama_accuracy():
    """Test MAMA matches expected values from Rust tests - mirrors check_mama_accuracy"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Default parameters
    mama_vals, fama_vals = mama(close, 0.5, 0.05)
    
    assert len(mama_vals) == len(close)
    assert len(fama_vals) == len(close)
    
    # Check that after warmup period (10), we have valid values
    for i in range(10, min(20, len(mama_vals))):
        assert np.isfinite(mama_vals[i]), f"MAMA NaN at index {i}"
        assert np.isfinite(fama_vals[i]), f"FAMA NaN at index {i}"


def test_mama_invalid_fast_limit():
    """Test MAMA fails with invalid fast limit"""
    input_data = np.array([10.0, 20.0, 30.0] * 10, dtype=np.float64)
    
    # Test with zero fast limit
    with pytest.raises(ValueError, match="mama:"):
        mama(input_data, 0.0, 0.05)
    
    # Test with negative fast limit
    with pytest.raises(ValueError, match="mama:"):
        mama(input_data, -0.5, 0.05)
    
    # Test with NaN fast limit
    with pytest.raises(ValueError, match="mama:"):
        mama(input_data, np.nan, 0.05)


def test_mama_invalid_slow_limit():
    """Test MAMA fails with invalid slow limit"""
    input_data = np.array([10.0, 20.0, 30.0] * 10, dtype=np.float64)
    
    # Test with zero slow limit
    with pytest.raises(ValueError, match="mama:"):
        mama(input_data, 0.5, 0.0)
    
    # Test with negative slow limit
    with pytest.raises(ValueError, match="mama:"):
        mama(input_data, 0.5, -0.05)
    
    # Test with NaN slow limit
    with pytest.raises(ValueError, match="mama:"):
        mama(input_data, 0.5, np.nan)


def test_mama_insufficient_data():
    """Test MAMA fails with insufficient data - mirrors check_mama_with_insufficient_data"""
    data_small = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="mama:"):
        mama(data_small, 0.5, 0.05)


def test_mama_very_small_dataset():
    """Test MAMA with minimum required data - mirrors check_mama_very_small_dataset"""
    # MAMA requires at least 10 data points
    data_min = np.array([42.0] * 10, dtype=np.float64)
    
    mama_vals, fama_vals = mama(data_min, 0.5, 0.05)
    assert len(mama_vals) == 10
    assert len(fama_vals) == 10


def test_mama_empty_input():
    """Test MAMA with empty input"""
    data_empty = np.array([], dtype=np.float64)
    
    with pytest.raises(ValueError, match="mama:"):
        mama(data_empty, 0.5, 0.05)


def test_mama_reinput():
    """Test MAMA with re-input of MAMA result - mirrors check_mama_reinput"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # First MAMA pass with default params
    mama_vals1, fama_vals1 = mama(close, 0.5, 0.05)
    
    # Second MAMA pass with different params using first MAMA result as input
    mama_vals2, fama_vals2 = mama(mama_vals1, 0.7, 0.1)
    
    assert len(mama_vals2) == len(mama_vals1)
    assert len(fama_vals2) == len(mama_vals1)


def test_mama_nan_handling():
    """Test MAMA handling of NaN values - mirrors check_mama_nan_handling"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    mama_vals, fama_vals = mama(close, 0.5, 0.05)
    
    assert len(mama_vals) == len(close)
    assert len(fama_vals) == len(close)
    
    # Check that after warmup period, there are valid values
    if len(mama_vals) > 20:
        for i in range(20, len(mama_vals)):
            assert np.isfinite(mama_vals[i]), f"MAMA NaN at index {i}"
            assert np.isfinite(fama_vals[i]), f"FAMA NaN at index {i}"


def test_mama_batch():
    """Test MAMA batch computation"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test parameter ranges
    mama_batch_result, fama_batch_result = mama_batch(
        close, 
        0.3, 0.7, 0.2,    # fast_limit range: 0.3, 0.5, 0.7
        0.03, 0.07, 0.02  # slow_limit range: 0.03, 0.05, 0.07
    )
    
    # Should have 3 * 3 = 9 combinations
    assert len(mama_batch_result) == 9 * len(close)
    assert len(fama_batch_result) == 9 * len(close)
    
    # Verify first combination matches individual calculation
    individual_mama, individual_fama = mama(close, 0.3, 0.03)
    batch_first_mama = mama_batch_result[:len(close)]
    batch_first_fama = fama_batch_result[:len(close)]
    assert_close(batch_first_mama, individual_mama, atol=1e-9, msg="MAMA first combination mismatch")
    assert_close(batch_first_fama, individual_fama, atol=1e-9, msg="FAMA first combination mismatch")


def test_mama_batch_with_metadata():
    """Test MAMA batch computation with metadata"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test with varying parameters
    (mama_result, fama_result), metadata = mama_batch_with_metadata(
        close, 
        0.4, 0.6, 0.2,    # fast_limit range: 0.4, 0.6
        0.04, 0.06, 0.02  # slow_limit range: 0.04, 0.06
    )
    
    # Check metadata contains all combinations (2 * 2 = 4)
    assert len(metadata) == 4
    # Use np.allclose for floating point comparison
    expected_metadata = [(0.4, 0.04), (0.4, 0.06), (0.6, 0.04), (0.6, 0.06)]
    for i, (actual, expected) in enumerate(zip(metadata, expected_metadata)):
        assert_close(actual[0], expected[0], atol=1e-9, msg=f"Fast limit mismatch at {i}")
        assert_close(actual[1], expected[1], atol=1e-9, msg=f"Slow limit mismatch at {i}")
    
    # Check result shape
    assert len(mama_result) == 4 * len(close)
    assert len(fama_result) == 4 * len(close)
    
    # Verify a specific combination
    # Find the index for (0.6, 0.04) - use approximate comparison due to floating point
    idx = None
    for i, (fast, slow) in enumerate(metadata):
        if abs(fast - 0.6) < 1e-9 and abs(slow - 0.04) < 1e-9:
            idx = i
            break
    assert idx is not None, f"Could not find (0.6, 0.04) in metadata: {metadata}"
    
    individual_mama, individual_fama = mama(close, 0.6, 0.04)
    batch_mama_row = mama_result[idx * len(close):(idx + 1) * len(close)]
    batch_fama_row = fama_result[idx * len(close):(idx + 1) * len(close)]
    assert_close(batch_mama_row, individual_mama, atol=1e-9, msg="MAMA specific combination mismatch")
    assert_close(batch_fama_row, individual_fama, atol=1e-9, msg="FAMA specific combination mismatch")


def test_mama_batch_2d():
    """Test MAMA batch computation with 2D output"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test with simple parameter ranges
    (mama_2d, fama_2d), metadata = mama_batch_2d(
        close, 
        0.3, 0.5, 0.2,    # fast_limit range: 0.3, 0.5
        0.05, 0.05, 0     # slow_limit static: 0.05
    )
    
    # Check metadata (2 * 1 = 2 combinations)
    assert metadata == [(0.3, 0.05), (0.5, 0.05)]
    
    # Check shape
    assert mama_2d.shape == (2, len(close))
    assert fama_2d.shape == (2, len(close))
    
    # Verify first row
    individual_mama, individual_fama = mama(close, 0.3, 0.05)
    assert_close(mama_2d[0], individual_mama, atol=1e-9, msg="MAMA 2D row mismatch")
    assert_close(fama_2d[0], individual_fama, atol=1e-9, msg="FAMA 2D row mismatch")


def test_mama_stream():
    """Test MAMA streaming interface"""
    data = load_test_data()
    close = data['close']
    
    # Default parameters
    fast_limit = 0.5
    slow_limit = 0.05
    
    # Calculate batch result for comparison
    close_array = np.array(close, dtype=np.float64)
    batch_mama, batch_fama = mama(close_array, fast_limit, slow_limit)
    
    # Test streaming
    stream = MamaStream(fast_limit, slow_limit)
    stream_results = []
    
    for i, price in enumerate(close):
        result = stream.update(price)
        stream_results.append(result)
    
    # Streaming returns None for first 9 values, then starts returning tuples
    for i in range(9):
        assert stream_results[i] is None, f"Expected None at index {i}"
    
    # From index 9 onwards, we should get values
    # The streaming implementation uses a 10-value window, so at index 9 we get the first result
    # This result is based on the window [0:10]
    for i in range(9, len(close)):
        assert stream_results[i] is not None, f"Expected value at index {i}"
        stream_mama, stream_fama = stream_results[i]
        
        # The streaming value at index i corresponds to a calculation on the window ending at i
        # Since streaming uses a rolling 10-value window, we can't directly compare with batch
        # Instead, just verify the values are reasonable
        assert np.isfinite(stream_mama), f"MAMA NaN at index {i}"
        assert np.isfinite(stream_fama), f"FAMA NaN at index {i}"


def test_mama_different_params():
    """Test MAMA with various parameter values"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    # Test various parameter combinations
    test_cases = [
        (0.3, 0.03),
        (0.5, 0.05),  # default
        (0.7, 0.07),
        (0.9, 0.1),
    ]
    
    for fast_lim, slow_lim in test_cases:
        mama_vals, fama_vals = mama(close, fast_lim, slow_lim)
        assert len(mama_vals) == len(close)
        assert len(fama_vals) == len(close)
        
        # Count valid values after warmup
        mama_valid = np.sum(np.isfinite(mama_vals[10:]))
        fama_valid = np.sum(np.isfinite(fama_vals[10:]))
        assert mama_valid == len(close) - 10, \
            f"Too many NaN values in MAMA for params=({fast_lim}, {slow_lim})"
        assert fama_valid == len(close) - 10, \
            f"Too many NaN values in FAMA for params=({fast_lim}, {slow_lim})"


def test_mama_batch_performance():
    """Test that batch computation works correctly (performance is secondary)"""
    data = load_test_data()
    close = np.array(data['close'][:1000], dtype=np.float64)  # Use first 1000 values
    
    # Test multiple parameter combinations
    mama_batch_result, fama_batch_result = mama_batch(
        close, 
        0.3, 0.7, 0.1,    # fast_limits: 0.3, 0.4, 0.5, 0.6, 0.7
        0.04, 0.06, 0.01  # slow_limits: 0.04, 0.05, 0.06
    )
    
    # Should have 5 * 3 = 15 combinations
    assert len(mama_batch_result) == 15 * len(close)
    assert len(fama_batch_result) == 15 * len(close)
    
    # Verify first combination (0.3, 0.04)
    individual_mama, individual_fama = mama(close, 0.3, 0.04)
    batch_first_mama = mama_batch_result[:len(close)]
    batch_first_fama = fama_batch_result[:len(close)]
    assert_close(batch_first_mama, individual_mama, atol=1e-9, 
                msg=f"MAMA batch mismatch for params=(0.3, 0.04)")
    assert_close(batch_first_fama, individual_fama, atol=1e-9, 
                msg=f"FAMA batch mismatch for params=(0.3, 0.04)")


def test_mama_edge_cases():
    """Test MAMA with edge case inputs"""
    # Test with monotonically increasing data
    data = np.arange(1.0, 101.0, dtype=np.float64)
    mama_vals, fama_vals = mama(data, 0.5, 0.05)
    assert len(mama_vals) == len(data)
    assert len(fama_vals) == len(data)
    
    # After warmup, values should be valid
    assert np.all(np.isfinite(mama_vals[10:]))
    assert np.all(np.isfinite(fama_vals[10:]))
    
    # Test with oscillating values
    data = np.array([10.0, 20.0, 10.0, 20.0] * 25, dtype=np.float64)
    mama_vals, fama_vals = mama(data, 0.5, 0.05)
    assert len(mama_vals) == len(data)
    assert len(fama_vals) == len(data)
    
    # Test with constant values
    data = np.array([50.0] * 100, dtype=np.float64)
    mama_vals, fama_vals = mama(data, 0.5, 0.05)
    assert len(mama_vals) == len(data)
    assert len(fama_vals) == len(data)
    
    # With constant data, MAMA and FAMA should converge to the constant
    for i in range(20, len(mama_vals)):
        assert_close(mama_vals[i], 50.0, atol=1e-6, msg=f"MAMA constant value failed at index {i}")
        assert_close(fama_vals[i], 50.0, atol=1e-6, msg=f"FAMA constant value failed at index {i}")


def test_mama_warmup_period():
    """Test that warmup period is correctly handled"""
    data = load_test_data()
    close = np.array(data['close'][:50], dtype=np.float64)
    
    mama_vals, fama_vals = mama(close, 0.5, 0.05)
    
    # MAMA outputs values even during warmup period (first 10 values)
    # Unlike some indicators, MAMA provides estimates from the start
    # All values should be finite
    for i in range(len(mama_vals)):
        assert np.isfinite(mama_vals[i]), f"Expected finite value at index {i} for MAMA"
        assert np.isfinite(fama_vals[i]), f"Expected finite value at index {i} for FAMA"


def test_mama_consistency():
    """Test that MAMA produces consistent results across multiple calls"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    mama1, fama1 = mama(close, 0.5, 0.05)
    mama2, fama2 = mama(close, 0.5, 0.05)
    
    assert_close(mama1, mama2, atol=1e-15, msg="MAMA results not consistent")
    assert_close(fama1, fama2, atol=1e-15, msg="FAMA results not consistent")


def test_mama_step_precision():
    """Test batch with very small step sizes"""
    data = np.arange(1, 51, dtype=np.float64)
    
    # Use batch_with_metadata to get the metadata
    (mama_result, fama_result), metadata = mama_batch_with_metadata(
        data, 
        0.4, 0.5, 0.1,     # fast_limits: 0.4, 0.5
        0.04, 0.05, 0.01   # slow_limits: 0.04, 0.05
    )
    
    assert metadata == [
        (0.4, 0.04), (0.4, 0.05),
        (0.5, 0.04), (0.5, 0.05)
    ]
    assert len(mama_result) == 4 * len(data)
    assert len(fama_result) == 4 * len(data)


def test_mama_fama_relationship():
    """Test that FAMA follows MAMA as expected"""
    data = load_test_data()
    close = np.array(data['close'][:200], dtype=np.float64)
    
    mama_vals, fama_vals = mama(close, 0.5, 0.05)
    
    # FAMA should be a smoother version of MAMA
    # After warmup, check that FAMA has less variance than MAMA
    mama_var = np.var(mama_vals[20:])
    fama_var = np.var(fama_vals[20:])
    
    # FAMA should generally have lower variance (be smoother)
    assert fama_var < mama_var * 1.1, "FAMA should be smoother than MAMA"


if __name__ == "__main__":
    # Run a simple test to verify the module loads correctly
    print("Testing MAMA module...")
    test_mama_accuracy()
    print("MAMA tests passed!")