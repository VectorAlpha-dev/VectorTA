"""
Python unit tests for the Donchian Channel indicator.
These tests verify that Python bindings correctly wrap the Rust implementation.
"""

import pytest
import numpy as np
from test_utils import load_test_data, assert_close, get_expected_output

# Import from the Rust module
try:
    import my_project as ta
except ImportError:
    ta = None
    pytest.skip("Rust module not available", allow_module_level=True)


def test_donchian_basic():
    """Test basic Donchian Channel calculation."""
    # Generate simple test data
    high = np.array([10.0, 12.0, 15.0, 11.0, 13.0, 16.0, 14.0, 12.0, 18.0, 17.0])
    low = np.array([8.0, 9.0, 11.0, 9.0, 10.0, 12.0, 11.0, 10.0, 14.0, 15.0])
    period = 3
    
    upper, middle, lower = ta.donchian(high, low, period)
    
    # Verify output shape
    assert len(upper) == len(high)
    assert len(middle) == len(high)
    assert len(lower) == len(high)
    
    # First period-1 values should be NaN
    assert np.isnan(upper[0])
    assert np.isnan(upper[1])
    assert np.isnan(middle[0])
    assert np.isnan(middle[1])
    assert np.isnan(lower[0])
    assert np.isnan(lower[1])
    
    # Check some calculated values
    # At index 2: high[0:3] = [10, 12, 15], low[0:3] = [8, 9, 11]
    assert upper[2] == 15.0  # max of high
    assert lower[2] == 8.0   # min of low
    assert middle[2] == (15.0 + 8.0) / 2  # average


def test_donchian_with_real_data():
    """Test Donchian Channel with real market data."""
    data = load_test_data()
    period = 20
    
    upper, middle, lower = ta.donchian(data['high'], data['low'], period)
    
    # Check dimensions
    assert len(upper) == len(data['high'])
    assert len(middle) == len(data['high'])
    assert len(lower) == len(data['high'])
    
    # Check NaN handling
    assert np.isnan(upper[period-2])
    assert not np.isnan(upper[period-1])
    
    # Verify last values match expected
    expected = get_expected_output('donchian')
    if expected:
        expected_upper = expected.get('upper_last5', [61290.0, 61290.0, 61290.0, 61290.0, 61290.0])
        expected_middle = expected.get('middle_last5', [59583.0, 59583.0, 59583.0, 59583.0, 59583.0])
        expected_lower = expected.get('lower_last5', [57876.0, 57876.0, 57876.0, 57876.0, 57876.0])
        
        for i in range(5):
            assert_close(upper[-5+i], expected_upper[i], tol=0.1)
            assert_close(middle[-5+i], expected_middle[i], tol=0.1)
            assert_close(lower[-5+i], expected_lower[i], tol=0.1)


def test_donchian_with_kernels():
    """Test Donchian Channel with different kernel options."""
    data = load_test_data()
    period = 20
    
    # Test with different kernels
    kernels = [None, 'scalar', 'avx2', 'avx512', 'auto']
    results = []
    
    for kernel in kernels:
        try:
            if kernel:
                upper, middle, lower = ta.donchian(data['high'], data['low'], period, kernel=kernel)
            else:
                upper, middle, lower = ta.donchian(data['high'], data['low'], period)
            results.append((upper, middle, lower))
        except ValueError as e:
            if 'not supported' in str(e):
                continue
            raise
    
    # All results should be identical
    for i in range(1, len(results)):
        np.testing.assert_array_almost_equal(results[0][0], results[i][0])
        np.testing.assert_array_almost_equal(results[0][1], results[i][1])
        np.testing.assert_array_almost_equal(results[0][2], results[i][2])


def test_donchian_edge_cases():
    """Test edge cases and error handling."""
    # Empty arrays
    with pytest.raises(ValueError):
        ta.donchian(np.array([]), np.array([]), 20)
    
    # Mismatched lengths
    with pytest.raises(ValueError):
        ta.donchian(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), 2)
    
    # Zero period
    with pytest.raises(ValueError):
        ta.donchian(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 0)
    
    # Period exceeds data length
    with pytest.raises(ValueError):
        ta.donchian(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 5)
    
    # All NaN data
    with pytest.raises(ValueError):
        ta.donchian(np.full(10, np.nan), np.full(10, np.nan), 5)


def test_donchian_streaming():
    """Test Donchian streaming functionality."""
    stream = ta.DonchianStream(3)
    
    # Test data
    test_data = [
        (10.0, 8.0),
        (12.0, 9.0),
        (15.0, 11.0),
        (11.0, 9.0),
        (13.0, 10.0),
    ]
    
    results = []
    for high, low in test_data:
        result = stream.update(high, low)
        results.append(result)
    
    # First period-1 updates should return None
    assert results[0] is None
    assert results[1] is None
    
    # After period, should get (upper, middle, lower) tuples
    assert results[2] is not None
    upper, middle, lower = results[2]
    assert upper == 15.0  # max of [10, 12, 15]
    assert lower == 8.0   # min of [8, 9, 11]
    assert middle == (15.0 + 8.0) / 2


def test_donchian_batch():
    """Test batch Donchian calculation with parameter sweeps."""
    data = load_test_data()
    
    # Test period range
    period_range = (10, 30, 10)  # 10, 20, 30
    
    result = ta.donchian_batch(data['high'], data['low'], period_range)
    
    # Check result structure
    assert 'upper' in result
    assert 'middle' in result
    assert 'lower' in result
    assert 'periods' in result
    
    # Check dimensions
    expected_combinations = 3  # [10, 20, 30]
    assert result['upper'].shape == (expected_combinations, len(data['high']))
    assert result['middle'].shape == (expected_combinations, len(data['high']))
    assert result['lower'].shape == (expected_combinations, len(data['high']))
    assert len(result['periods']) == expected_combinations
    
    # Verify periods
    np.testing.assert_array_equal(result['periods'], [10, 20, 30])
    
    # Verify each row matches individual calculation
    for i, period in enumerate([10, 20, 30]):
        upper_single, middle_single, lower_single = ta.donchian(data['high'], data['low'], period)
        np.testing.assert_array_almost_equal(result['upper'][i], upper_single)
        np.testing.assert_array_almost_equal(result['middle'][i], middle_single)
        np.testing.assert_array_almost_equal(result['lower'][i], lower_single)


def test_donchian_batch_single_param():
    """Test batch with single parameter (no sweep)."""
    data = load_test_data()
    
    # Single period
    period_range = (20, 20, 0)
    
    result = ta.donchian_batch(data['high'], data['low'], period_range)
    
    # Should have 1 combination
    assert result['upper'].shape == (1, len(data['high']))
    assert len(result['periods']) == 1
    assert result['periods'][0] == 20
    
    # Should match single calculation
    upper_single, middle_single, lower_single = ta.donchian(data['high'], data['low'], 20)
    np.testing.assert_array_almost_equal(result['upper'][0], upper_single)
    np.testing.assert_array_almost_equal(result['middle'][0], middle_single)
    np.testing.assert_array_almost_equal(result['lower'][0], lower_single)


def test_donchian_with_nan_values():
    """Test handling of NaN values in input."""
    # Create data with some NaN values
    high = np.array([np.nan, 12.0, 15.0, 11.0, 13.0, 16.0, 14.0, 12.0, 18.0, 17.0])
    low = np.array([np.nan, 9.0, 11.0, 9.0, 10.0, 12.0, 11.0, 10.0, 14.0, 15.0])
    period = 3
    
    upper, middle, lower = ta.donchian(high, low, period)
    
    # Output shape should match input
    assert len(upper) == len(high)
    
    # Check that calculation starts after NaN values
    assert np.isnan(upper[0])
    assert np.isnan(upper[1])
    assert np.isnan(upper[2])
    assert not np.isnan(upper[3])  # First valid calculation


def test_donchian_performance():
    """Basic performance test for Donchian Channel."""
    # Generate large dataset
    size = 100_000
    high = np.random.randn(size).cumsum() + 100
    low = high - np.abs(np.random.randn(size))
    
    import time
    
    # Time the calculation
    start = time.perf_counter()
    upper, middle, lower = ta.donchian(high, low, 20)
    elapsed = time.perf_counter() - start
    
    print(f"Donchian calculation for {size:,} points took {elapsed*1000:.2f} ms")
    
    # Basic sanity checks
    assert len(upper) == size
    assert np.all(upper >= middle)
    assert np.all(middle >= lower)


if __name__ == "__main__":
    # Run basic tests
    test_donchian_basic()
    test_donchian_with_real_data()
    test_donchian_streaming()
    test_donchian_batch()
    print("All Donchian tests passed!")