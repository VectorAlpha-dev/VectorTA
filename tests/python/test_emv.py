"""
Python unit tests for the Ease of Movement (EMV) indicator.
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


def test_emv_basic():
    """Test basic EMV calculation."""
    # Generate simple test data
    high = np.array([10.0, 12.0, 13.0, 15.0, 14.0, 16.0])
    low = np.array([5.0, 7.0, 8.0, 10.0, 11.0, 12.0])
    close = np.array([7.5, 9.0, 10.5, 12.5, 12.5, 14.0])
    volume = np.array([10000.0, 20000.0, 25000.0, 30000.0, 15000.0, 35000.0])
    
    result = ta.emv(high, low, close, volume)
    
    # Verify output shape
    assert len(result) == len(high)
    
    # First value should be NaN (need previous midpoint)
    assert np.isnan(result[0])
    
    # After first value, should have calculated values
    assert not np.isnan(result[1])
    
    # Test specific calculation for index 1
    # mid[0] = (10 + 5) / 2 = 7.5
    # mid[1] = (12 + 7) / 2 = 9.5
    # range[1] = 12 - 7 = 5
    # br[1] = 20000 / 10000 / 5 = 0.4
    # emv[1] = (9.5 - 7.5) / 0.4 = 5.0
    assert_close(result[1], 5.0, tol=0.01)


def test_emv_with_real_data():
    """Test EMV with real market data."""
    data = load_test_data()
    
    result = ta.emv(data['high'], data['low'], data['close'], data['volume'])
    
    # Check dimensions
    assert len(result) == len(data['high'])
    
    # First value should be NaN
    assert np.isnan(result[0])
    
    # Check some values are not NaN after warmup
    assert not np.isnan(result[10])
    
    # Verify last 5 values match expected from Rust tests
    expected_last_five = [
        -6488905.579799851,
        2371436.7401001123,
        -3855069.958128531,
        1051939.877943717,
        -8519287.22257077,
    ]
    
    for i in range(5):
        assert_close(result[-5+i], expected_last_five[i], tol=0.0001)


def test_emv_with_kernels():
    """Test EMV with different kernel options."""
    data = load_test_data()
    
    # Test with different kernels
    kernels = [None, "scalar", "avx2", "avx512"]
    results = []
    
    for kernel in kernels:
        try:
            if kernel:
                result = ta.emv(data['high'], data['low'], data['close'], data['volume'], kernel=kernel)
            else:
                result = ta.emv(data['high'], data['low'], data['close'], data['volume'])
            results.append(result)
        except Exception as e:
            if "not supported" in str(e) or "not available" in str(e):
                continue
            raise
    
    # All results should be close
    for i in range(1, len(results)):
        np.testing.assert_allclose(results[0], results[i], rtol=1e-10, equal_nan=True)


def test_emv_empty_data():
    """Test EMV with empty data."""
    empty = np.array([])
    
    with pytest.raises(Exception, match="empty|EmptyData"):
        ta.emv(empty, empty, empty, empty)


def test_emv_all_nan():
    """Test EMV with all NaN values."""
    all_nan = np.full(10, np.nan)
    
    with pytest.raises(Exception, match="NaN|AllValuesNaN"):
        ta.emv(all_nan, all_nan, all_nan, all_nan)


def test_emv_insufficient_data():
    """Test EMV with insufficient data."""
    # EMV needs at least 2 valid points
    high = np.array([10.0, np.nan])
    low = np.array([9.0, np.nan])
    close = np.array([9.5, np.nan])
    volume = np.array([1000.0, np.nan])
    
    with pytest.raises(Exception, match="NotEnoughData|not enough"):
        ta.emv(high, low, close, volume)


def test_emv_partial_nan_handling():
    """Test EMV with partial NaN values."""
    high = np.array([np.nan, 12.0, 15.0, np.nan, 13.0, 16.0])
    low = np.array([np.nan, 9.0, 11.0, np.nan, 10.0, 12.0])
    close = np.array([np.nan, 10.0, 13.0, np.nan, 11.5, 14.0])
    volume = np.array([np.nan, 10000.0, 20000.0, np.nan, 15000.0, 25000.0])
    
    result = ta.emv(high, low, close, volume)
    
    # Check shape
    assert len(result) == len(high)
    
    # First few should be NaN
    assert np.isnan(result[0])
    assert np.isnan(result[1])  # Need previous value
    
    # Should have valid values after enough data
    assert not np.isnan(result[2])


def test_emv_zero_range():
    """Test EMV when high equals low (zero range)."""
    high = np.array([10.0, 10.0, 12.0, 13.0])
    low = np.array([9.0, 10.0, 11.0, 12.0])  # At index 1: high == low
    close = np.array([9.5, 10.0, 11.5, 12.5])
    volume = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    
    result = ta.emv(high, low, close, volume)
    
    # When range is zero, EMV should be NaN
    assert np.isnan(result[1])
    
    # Other values should be calculated
    assert not np.isnan(result[2])


def test_emv_streaming():
    """Test EMV streaming functionality."""
    data = load_test_data()
    
    # Calculate batch result
    batch_result = ta.emv(data['high'], data['low'], data['close'], data['volume'])
    
    # Create stream and process same data
    stream = ta.EmvStream()
    stream_result = []
    
    for i in range(len(data['high'])):
        value = stream.update(data['high'][i], data['low'][i], data['close'][i], data['volume'][i])
        stream_result.append(value if value is not None else np.nan)
    
    # Results should match
    np.testing.assert_allclose(batch_result, stream_result, rtol=1e-10, equal_nan=True)


def test_emv_batch():
    """Test EMV batch operations."""
    data = load_test_data()
    
    # EMV has no parameters, so batch just runs once
    result = ta.emv_batch(data['high'], data['low'], data['close'], data['volume'])
    
    # Check structure
    assert 'values' in result
    assert result['values'].shape == (1, len(data['high']))
    
    # Values should match single calculation
    single_result = ta.emv(data['high'], data['low'], data['close'], data['volume'])
    np.testing.assert_allclose(result['values'][0], single_result, rtol=1e-10, equal_nan=True)


def test_emv_batch_with_kernel():
    """Test EMV batch with kernel parameter."""
    data = load_test_data()
    
    # Test scalar kernel
    result = ta.emv_batch(data['high'], data['low'], data['close'], data['volume'], kernel="scalar")
    
    assert 'values' in result
    assert result['values'].shape == (1, len(data['high']))


def test_emv_mismatched_lengths():
    """Test EMV with mismatched input lengths."""
    high = np.array([10.0, 12.0, 13.0])
    low = np.array([9.0, 11.0])  # Different length
    close = np.array([9.5, 11.5, 12.0])
    volume = np.array([1000.0, 2000.0, 3000.0])
    
    # Should still work but use minimum length
    result = ta.emv(high, low, close, volume)
    assert len(result) == min(len(high), len(low), len(close), len(volume))


def test_emv_reinput():
    """Test EMV applied to its own output."""
    data = load_test_data()
    
    # First pass
    first_result = ta.emv(data['high'], data['low'], data['close'], data['volume'])
    
    # Create synthetic inputs from first result
    # Use absolute values since EMV can be negative
    abs_result = np.abs(first_result)
    synthetic_high = abs_result * 1.1
    synthetic_low = abs_result * 0.9
    synthetic_close = abs_result
    synthetic_volume = np.full_like(abs_result, 10000.0)
    
    # Replace NaN with valid values
    mask = ~np.isnan(abs_result)
    synthetic_high[~mask] = 100.0
    synthetic_low[~mask] = 90.0
    synthetic_close[~mask] = 95.0
    synthetic_volume[~mask] = 10000.0
    
    # Second pass should work
    second_result = ta.emv(synthetic_high, synthetic_low, synthetic_close, synthetic_volume)
    assert len(second_result) == len(first_result)