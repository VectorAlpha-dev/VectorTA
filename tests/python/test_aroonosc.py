"""
Python binding tests for Aroon Oscillator indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import my_project


def load_test_data():
    """Load the standard test data"""
    data = np.loadtxt(
        'src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv',
        delimiter=',',
        skiprows=1,
        usecols=(1, 3, 4, 2),  
        unpack=True
    )
    
    return tuple(np.ascontiguousarray(arr) for arr in data)


def test_aroonosc_partial_params():
    """Test Aroon Oscillator with partial parameters - mirrors check_aroonosc_partial_params"""
    _, high, low, _ = load_test_data()
    
    
    result = my_project.aroonosc(high, low, length=20)
    
    assert len(result) == len(high)
    assert isinstance(result, np.ndarray)
    
    
    for i in range(20):
        assert np.isnan(result[i])


def test_aroonosc_accuracy():
    """Test Aroon Oscillator matches expected values from Rust tests - mirrors check_aroonosc_accuracy"""
    _, high, low, _ = load_test_data()
    
    
    result = my_project.aroonosc(high, low)  
    
    
    expected_last_five = [-50.0, -50.0, -50.0, -50.0, -42.8571]
    
    assert len(result) >= 5, "Not enough Aroon Osc values"
    
    
    last_five = result[-5:]
    for i, (actual, expected) in enumerate(zip(last_five, expected_last_five)):
        assert abs(actual - expected) < 1e-2, \
            f"Aroon Osc mismatch at index {i}: expected {expected}, got {actual}"
    
    
    length = 14
    for i in range(length, len(result)):
        if not np.isnan(result[i]):
            assert np.isfinite(result[i]), f"Aroon Osc should be finite after enough data at index {i}"


def test_aroonosc_zero_length():
    """Test Aroon Oscillator fails with zero length - mirrors Rust test for zero period"""
    high = np.array([10.0, 20.0, 30.0])
    low = np.array([5.0, 15.0, 25.0])
    
    with pytest.raises(ValueError, match="Invalid length"):
        my_project.aroonosc(high, low, length=0)


def test_aroonosc_length_exceeds_data():
    """Test Aroon Oscillator fails when length exceeds data length - mirrors Rust test"""
    high = np.array([10.0, 20.0, 30.0])
    low = np.array([5.0, 15.0, 25.0])
    
    with pytest.raises(ValueError, match="Not enough data"):
        my_project.aroonosc(high, low, length=10)


def test_aroonosc_mismatched_arrays():
    """Test Aroon Oscillator fails with mismatched high/low arrays"""
    high = np.array([10.0, 20.0, 30.0])
    low = np.array([5.0, 15.0])  
    
    with pytest.raises(ValueError, match="High and low arrays must have same length"):
        my_project.aroonosc(high, low)


def test_aroonosc_nan_handling():
    """Test Aroon Oscillator handles NaN values correctly - mirrors check_aroonosc_nan_handling"""
    _, high, low, _ = load_test_data()
    
    result = my_project.aroonosc(high, low)
    
    
    if len(result) > 50:
        for i in range(50, len(result)):
            assert not np.isnan(result[i]), f"Expected no NaN after index {i}, but found NaN"


def test_aroonosc_streaming():
    """Test streaming Aroon Oscillator implementation"""
    _, high, low, _ = load_test_data()
    
    
    stream = my_project.AroonOscStream(length=14)
    
    
    stream_results = []
    for h, l in zip(high, low):
        val = stream.update(h, l)
        stream_results.append(val if val is not None else np.nan)
    
    
    batch_result = my_project.aroonosc(high, low, length=14)
    
    
    assert len(stream_results) == len(batch_result)
    
    
    
    
    
    
    warmup = 14
    assert all(np.isnan(v) for v in stream_results[:warmup]), \
        "Stream should return NaN during warmup"
    assert any(not np.isnan(v) for v in stream_results[warmup:]), \
        "Stream should produce values after warmup"
    
    
    
    
    
    
    
    


def test_aroonosc_batch_single_params():
    """Test batch with single parameter combination"""
    _, high, low, _ = load_test_data()
    
    
    batch_result = my_project.aroonosc_batch(
        high,
        low,
        length_range=(14, 14, 0)  
    )
    
    
    assert 'values' in batch_result
    assert 'lengths' in batch_result
    
    values = batch_result['values']
    lengths = batch_result['lengths']
    
    
    assert values.shape[0] == 1
    assert values.shape[1] == len(high)
    assert len(lengths) == 1
    assert lengths[0] == 14
    
    
    single_result = my_project.aroonosc(high, low, length=14)
    np.testing.assert_array_almost_equal(values[0], single_result, decimal=10)


def test_aroonosc_batch_multiple_lengths():
    """Test batch with multiple length values"""
    _, high, low, _ = load_test_data()
    
    
    batch_result = my_project.aroonosc_batch(
        high,
        low,
        length_range=(10, 18, 4)
    )
    
    values = batch_result['values']
    lengths = batch_result['lengths']
    
    
    assert values.shape[0] == 3
    assert values.shape[1] == len(high)
    assert len(lengths) == 3
    assert list(lengths) == [10, 14, 18]
    
    
    for i, length in enumerate(lengths):
        single_result = my_project.aroonosc(high, low, length=int(length))
        np.testing.assert_array_almost_equal(values[i], single_result, decimal=10,
                                           err_msg=f"Mismatch for length={length}")


def test_aroonosc_batch_kernel_options():
    """Test batch with different kernel options"""
    data = load_test_data()
    _, high, low, _ = data[0][:100], data[1][:100], data[2][:100], data[3][:100]  
    
    
    batch_scalar = my_project.aroonosc_batch(
        high,
        low,
        length_range=(10, 14, 2),
        kernel='scalar'
    )
    
    
    batch_auto = my_project.aroonosc_batch(
        high,
        low,
        length_range=(10, 14, 2),
        kernel=None  
    )
    
    
    np.testing.assert_array_almost_equal(
        batch_scalar['values'],
        batch_auto['values'],
        decimal=10
    )


def test_aroonosc_with_edge_cases():
    """Test Aroon Oscillator with edge case values"""
    
    high = np.full(50, 100.0)
    low = np.full(50, 100.0)
    
    result = my_project.aroonosc(high, low, length=14)
    
    
    for i in range(14, len(result)):
        assert abs(result[i]) < 1e-10, f"Expected 0 for constant values at index {i}, got {result[i]}"
    
    
    high = np.arange(50, dtype=float) + 100.0
    low = np.full(50, 100.0)
    
    result = my_project.aroonosc(high, low, length=14)
    
    
    for i in range(20, len(result)):
        assert result[i] > 0, f"Expected positive value for uptrend at index {i}, got {result[i]}"


def test_aroonosc_batch_empty_data():
    """Test batch fails with empty data"""
    high = np.array([])
    low = np.array([])
    
    with pytest.raises(ValueError):
        my_project.aroonosc_batch(high, low, length_range=(10, 20, 2))


def test_aroonosc_reinput():
    """Test Aroon Oscillator with reinput of its own output - mirrors Rust test"""
    _, high, low, _ = load_test_data()
    
    
    first_result = my_project.aroonosc(high, low, length=10)
    
    
    second_result = my_project.aroonosc(first_result, first_result, length=5)
    
    assert len(second_result) == len(first_result)
    
    
    for i in range(20, len(second_result)):
        assert not np.isnan(second_result[i])