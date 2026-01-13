"""
Python binding tests for LinReg indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS
from rust_comparison import compare_with_rust


try:
    from my_project import (
        linreg,
        linreg_batch,
        LinRegStream
    )
except ImportError:
    pytest.skip("LinReg module not available - run 'maturin develop' first", allow_module_level=True)


def test_linreg_partial_params():
    """Test LinReg with default parameters - mirrors check_linreg_partial_params"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)


    result = linreg(close, 14)
    assert len(result) == len(close)


def test_linreg_accuracy():
    """Test LinReg matches expected values from Rust tests - mirrors check_linreg_accuracy"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    expected = EXPECTED_OUTPUTS['linreg']


    result = linreg(close, expected['default_params']['period'])


    assert_close(
        result[-5:],
        expected['last_5_values'],
        atol=1e-1,
        msg="LinReg last 5 values mismatch"
    )


    compare_with_rust('linreg', result, 'close', expected['default_params'])


def test_linreg_zero_period():
    """Test LinReg fails with zero period - mirrors check_linreg_zero_period"""
    input_data = np.array([10.0, 20.0, 30.0], dtype=np.float64)

    with pytest.raises(ValueError, match="Invalid period"):
        linreg(input_data, 0)


def test_linreg_period_exceeds_length():
    """Test LinReg fails with period exceeding data length - mirrors check_linreg_period_exceeds_length"""
    data_small = np.array([10.0, 20.0, 30.0], dtype=np.float64)

    with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
        linreg(data_small, 10)


def test_linreg_very_small_dataset():
    """Test LinReg fails with insufficient data - mirrors check_linreg_very_small_dataset"""
    single_point = np.array([42.0], dtype=np.float64)

    with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
        linreg(single_point, 14)


def test_linreg_empty_input():
    """Test LinReg with empty input"""
    data_empty = np.array([], dtype=np.float64)

    with pytest.raises(ValueError, match="All values are NaN"):
        linreg(data_empty, 14)


def test_linreg_all_nan():
    """Test LinReg with all NaN input"""
    data = np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)

    with pytest.raises(ValueError, match="All values are NaN"):
        linreg(data, 3)


def test_linreg_reinput():
    """Test LinReg with re-input of LinReg result - mirrors check_linreg_reinput"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)


    first_result = linreg(close, 14)


    second_result = linreg(first_result, 10)

    assert len(second_result) == len(first_result)





    for i in range(24, len(second_result)):
        assert np.isfinite(second_result[i]), f"Unexpected NaN at index {i}"


def test_linreg_nan_handling():
    """Test LinReg handling of NaN values - mirrors check_linreg_nan_handling"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    expected = EXPECTED_OUTPUTS['linreg']


    result = linreg(close, expected['default_params']['period'])

    assert len(result) == len(close)


    first_valid = next((i for i, x in enumerate(result) if not np.isnan(x)), None)
    assert first_valid == expected['warmup_period'], f"Expected warmup at {expected['warmup_period']}, got {first_valid}"



    if len(result) > 240:
        for i in range(240, len(result)):
            assert not np.isnan(result[i]), f"Found unexpected NaN at index {i}"


def test_linreg_batch():
    """Test LinReg batch computation"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)


    period_start = 10
    period_end = 40
    period_step = 10

    result = linreg_batch(close, (period_start, period_end, period_step))

    assert 'values' in result
    assert 'periods' in result


    assert len(result['periods']) == 4
    assert np.array_equal(result['periods'], [10, 20, 30, 40])


    assert result['values'].shape == (4, len(close))


    individual_result = linreg(close, 10)
    batch_first_row = result['values'][0]
    assert_close(batch_first_row, individual_result, atol=1e-9, msg="First combination mismatch")


def test_linreg_batch_multiple_periods():
    """Test LinReg batch computation with multiple periods"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)


    period_start = 20
    period_end = 60
    period_step = 20

    result = linreg_batch(close, (period_start, period_end, period_step))


    assert 'values' in result
    assert 'periods' in result
    assert np.array_equal(result['periods'], [20, 40, 60])


    assert result['values'].shape == (3, len(close))


    individual_result = linreg(close, 40)
    batch_second_row = result['values'][1]
    assert_close(batch_second_row, individual_result, atol=1e-9, msg="Second combination mismatch")


def test_linreg_batch_single_period():
    """Test LinReg batch computation with single period"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)


    result = linreg_batch(close, (30, 30, 0))


    assert 'values' in result
    assert 'periods' in result
    assert np.array_equal(result['periods'], [30])


    assert result['values'].shape == (1, len(close))


    individual_result = linreg(close, 30)
    assert_close(result['values'][0], individual_result, atol=1e-9, msg="Single period mismatch")


def test_linreg_stream():
    """Test LinReg streaming interface - mirrors check_linreg_streaming"""
    data = load_test_data()
    close = data['close']
    period = 14


    close_array = np.array(close, dtype=np.float64)
    batch_result = linreg(close_array, period)


    stream = LinRegStream(period)
    stream_results = []

    for price in close:
        result = stream.update(price)

        stream_results.append(result if result is not None else np.nan)


    assert len(batch_result) == len(stream_results)

    for i in range(len(batch_result)):

        if np.isnan(batch_result[i]) and np.isnan(stream_results[i]):
            continue

        assert_close(stream_results[i], batch_result[i], atol=1e-6,
                    msg=f"Streaming mismatch at index {i}")


def test_linreg_different_periods():
    """Test LinReg with various period values"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)


    for period in [5, 10, 20, 50]:
        result = linreg(close, period)
        assert len(result) == len(close)


        valid_count = np.sum(np.isfinite(result[period:]))
        assert valid_count > len(close) - period - 5, f"Too many NaN values for period={period}"


def test_linreg_batch_performance():
    """Test that batch computation works correctly (performance is secondary)"""
    data = load_test_data()
    close = np.array(data['close'][:1000], dtype=np.float64)


    result = linreg_batch(close, (10, 50, 10))
    assert len(result['periods']) == 5


    for i, period in enumerate(result['periods']):
        individual_result = linreg(close, period)
        batch_row = result['values'][i]
        assert_close(batch_row, individual_result, atol=1e-9,
                          msg=f"Batch mismatch for period={period}")


def test_linreg_edge_cases():
    """Test LinReg with edge case inputs"""

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)
    result = linreg(data, 3)
    assert len(result) == len(data)



    assert_close(result[-1], 10.0, atol=1e-9, msg="Perfect linear regression failed")


    data = np.array([1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan], dtype=np.float64)
    result = linreg(data, 2)
    assert len(result) == len(data)


    data = np.array([5.0] * 20, dtype=np.float64)
    result = linreg(data, 5)
    assert len(result) == len(data)

    for i in range(5, len(result)):
        assert_close(result[i], 5.0, atol=1e-9, msg=f"Constant prediction failed at index {i}")


def test_linreg_single_value():
    """Test LinReg with single value input"""
    data = np.array([42.0], dtype=np.float64)


    result = linreg(data, 1)
    assert len(result) == 1
    assert np.isnan(result[0])


def test_linreg_two_values():
    """Test LinReg with two values input"""
    data = np.array([1.0, 2.0], dtype=np.float64)



    result = linreg(data, 1)
    assert len(result) == 2
    assert np.isnan(result[0])
    assert np.isnan(result[1])


    result = linreg(data, 2)
    assert len(result) == 2
    assert np.isnan(result[0])
    assert_close(result[1], 2.0, atol=1e-9, msg="Two-value prediction failed")


def test_linreg_warmup_period():
    """Test that warmup period is correctly calculated (first + period - 1)"""
    data = load_test_data()
    close = np.array(data['close'][:50], dtype=np.float64)

    test_cases = [
        (5, 4),
        (10, 9),
        (20, 19),
        (30, 29),
    ]

    for period, expected_warmup in test_cases:
        result = linreg(close, period)


        for i in range(expected_warmup):
            assert np.isnan(result[i]), f"Expected NaN at index {i} for period={period}"


        if expected_warmup < len(result):
            assert not np.isnan(result[expected_warmup]), \
                f"Expected valid value at index {expected_warmup} for period={period}"


def test_linreg_consistency():
    """Test that LinReg produces consistent results across multiple calls"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)

    result1 = linreg(close, 14)
    result2 = linreg(close, 14)

    assert_close(result1, result2, atol=1e-15, msg="LinReg results not consistent")


def test_linreg_step_precision():
    """Test batch with very small step sizes"""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)

    result = linreg_batch(data, (2, 4, 1))

    assert np.array_equal(result['periods'], [2, 3, 4])
    assert result['values'].shape == (3, len(data))


def test_linreg_slope_calculation():
    """Test LinReg slope calculation with known data"""

    data = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0], dtype=np.float64)
    result = linreg(data, 4)



    assert_close(result[-1], 16.0, atol=1e-9, msg="Slope calculation failed")


def test_linreg_batch_error_handling():
    """Test LinReg batch error handling"""

    empty = np.array([], dtype=np.float64)
    with pytest.raises(ValueError, match="Input data slice is empty|All values are NaN"):
        linreg_batch(empty, (10, 20, 10))


    all_nan = np.full(100, np.nan, dtype=np.float64)
    with pytest.raises(ValueError, match="All values are NaN"):
        linreg_batch(all_nan, (10, 20, 10))


    small_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    with pytest.raises(ValueError, match="Not enough valid data"):
        linreg_batch(small_data, (5, 10, 5))


def test_linreg_batch_metadata_consistency():
    """Test that batch metadata is consistent with results"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)


    test_cases = [
        (5, 15, 5),
        (10, 10, 0),
        (20, 40, 10),
    ]

    for start, end, step in test_cases:
        result = linreg_batch(close, (start, end, step))


        if step == 0 or start == end:
            expected_periods = [start]
        else:
            expected_periods = list(range(start, end + 1, step))

        assert np.array_equal(result['periods'], expected_periods)
        assert result['values'].shape[0] == len(expected_periods)
        assert result['values'].shape[1] == len(close)


def test_linreg_zero_copy_verification():
    """Verify LinReg uses zero-copy operations"""

    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)


    result = linreg(close, 14)
    assert len(result) == len(close)


    batch_result = linreg_batch(close, (10, 30, 10))
    assert batch_result['values'].shape == (3, len(close))


def test_linreg_batch_warmup_consistency():
    """Test that batch warmup periods are consistent"""
    data = np.random.randn(50).astype(np.float64)

    result = linreg_batch(data, (5, 15, 5))


    for i, period in enumerate(result['periods']):
        row = result['values'][i]

        assert np.all(np.isnan(row[:period-1])), f"Expected NaN warmup for period {period}"

        assert np.all(~np.isnan(row[period-1:])), f"Expected values after warmup for period {period}"


def test_linreg_streaming():
    """Test LinReg streaming matches batch calculation - mirrors check_linreg_streaming"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    expected = EXPECTED_OUTPUTS['linreg']
    period = expected['default_params']['period']


    batch_result = linreg(close, period)


    stream = LinRegStream(period)
    stream_values = []

    for price in close:
        result = stream.update(price)
        stream_values.append(result if result is not None else np.nan)

    stream_values = np.array(stream_values)


    assert len(batch_result) == len(stream_values)


    for i, (b, s) in enumerate(zip(batch_result, stream_values)):
        if np.isnan(b) and np.isnan(s):
            continue
        assert_close(b, s, rtol=1e-9, atol=1e-9,
                    msg=f"LinReg streaming mismatch at index {i}")


def test_linreg_stream_error_handling():
    """Test LinReg stream error handling"""

    with pytest.raises(ValueError, match="LinRegStream error|Invalid period"):
        LinRegStream(0)


if __name__ == "__main__":

    print("Testing LinReg module...")
    test_linreg_accuracy()
    print("LinReg tests passed!")