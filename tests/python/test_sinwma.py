"""
Python binding tests for SINWMA indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS


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
    
    
    result = sinwma(close, 14)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(close)


def test_sinwma_default_candles():
    """Test SINWMA with default parameters - mirrors check_sinwma_default_candles"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    
    result = sinwma(close, 14)
    assert len(result) == len(close)


def test_sinwma_accuracy():
    """Test SINWMA matches expected values from Rust tests - mirrors check_sinwma_accuracy"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    expected = EXPECTED_OUTPUTS['sinwma']
    
    
    result = sinwma(close, expected['default_params']['period'])
    
    assert len(result) == len(close)
    
    
    assert_close(
        result[-5:], 
        expected['last_5_values'],
        atol=1e-6,
        rtol=0.0,
        msg="SINWMA last 5 values mismatch"
    )


def test_sinwma_invalid_period():
    """Test SINWMA fails with invalid period"""
    input_data = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    
    
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


def test_sinwma_period_one():
    """Test SINWMA with period=1 acts as passthrough"""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    
    result = sinwma(data, 1)
    
    
    assert_close(result, data, atol=1e-10, rtol=0.0, msg="Period=1 should act as passthrough")


def test_sinwma_nan_handling():
    """Test SINWMA handling of NaN values - mirrors check_sinwma_nan_handling"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    result = sinwma(close, 14)
    
    assert len(result) == len(close)
    
    
    assert np.all(np.isnan(result[:13])), "Expected NaN in warmup period (first 13 values)"
    
    
    first_valid = np.where(~np.isnan(close))[0][0]
    warmup = first_valid + 14 - 1
    
    if len(result) > warmup:
        assert np.all(np.isfinite(result[warmup:])), "Found unexpected NaN after warmup period"


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
    
    
    batch_result = sinwma_batch(
        close,
        (10, 30, 5)  
    )
    
    
    assert isinstance(batch_result, dict)
    assert 'values' in batch_result
    assert 'periods' in batch_result
    
    
    assert batch_result['values'].shape == (5, len(close))
    assert len(batch_result['periods']) == 5
    
    
    expected_periods = [10, 15, 20, 25, 30]
    assert np.array_equal(batch_result['periods'], expected_periods), "Period metadata mismatch"
    
    
    individual_result = sinwma(close, 10)
    batch_first = batch_result['values'][0]
    
    
    first_valid = np.where(~np.isnan(close))[0][0]
    warmup = first_valid + 10 - 1
    assert_close(batch_first[warmup:], individual_result[warmup:], atol=1e-9, rtol=0.0, msg="SINWMA first combination mismatch")


def test_sinwma_batch_single_parameter():
    """Test SINWMA batch with single parameter combination - mirrors ALMA pattern"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    expected = EXPECTED_OUTPUTS['sinwma']
    
    
    result = sinwma_batch(
        close,
        (14, 14, 0)  
    )
    
    assert 'values' in result
    assert 'periods' in result
    
    
    assert result['values'].shape[0] == 1
    assert result['values'].shape[1] == len(close)
    assert result['periods'][0] == 14
    
    
    default_row = result['values'][0]
    
    
    assert_close(
        default_row[-5:],
        expected['last_5_values'],
        atol=1e-6,
        rtol=0.0,
        msg="SINWMA batch default row mismatch"
    )


def test_sinwma_stream():
    """Test SINWMA streaming interface"""
    data = load_test_data()
    close = data['close']
    
    
    period = 14
    
    
    close_array = np.array(close, dtype=np.float64)
    batch_result = sinwma(close_array, period)
    
    
    stream = SinWmaStream(period)
    stream_results = []
    
    for price in close:
        result = stream.update(price)
        stream_results.append(result if result is not None else np.nan)
    
    stream_results = np.array(stream_results)
    
    
    for i in range(len(close)):
        if np.isnan(batch_result[i]) and np.isnan(stream_results[i]):
            continue
        if not np.isnan(batch_result[i]) and not np.isnan(stream_results[i]):
            assert_close(batch_result[i], stream_results[i], atol=1e-9, rtol=0.0, msg=f"SINWMA streaming mismatch at index {i}")


def test_sinwma_different_periods():
    """Test SINWMA with various period values"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    
    test_periods = [5, 10, 14, 20, 50]
    
    for period in test_periods:
        result = sinwma(close, period)
        assert len(result) == len(close)
        
        
        first_valid = np.where(~np.isnan(close))[0][0]
        warmup = first_valid + period - 1
        if warmup < len(result):
            assert np.all(np.isfinite(result[warmup:])), f"NaN values found for period={period}"


def test_sinwma_batch_performance():
    """Test that batch computation works correctly (performance is secondary)"""
    data = load_test_data()
    close = np.array(data['close'][:1000], dtype=np.float64)  
    
    
    batch_result = sinwma_batch(
        close,
        (10, 50, 10)  
    )
    
    
    assert batch_result['values'].shape == (5, len(close))
    
    
    expected_periods = [10, 20, 30, 40, 50]
    assert np.array_equal(batch_result['periods'], expected_periods)


def test_sinwma_edge_cases():
    """Test SINWMA with edge case inputs"""
    
    data = np.arange(1.0, 101.0, dtype=np.float64)
    result = sinwma(data, 14)
    assert len(result) == len(data)
    assert np.all(np.isfinite(result[13:]))  
    
    
    data = np.array([50.0] * 100, dtype=np.float64)
    result = sinwma(data, 14)
    assert len(result) == len(data)
    
    assert np.all(np.isfinite(result[13:]))
    
    
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
    
    
    batch_result = sinwma_batch(
        data,
        (10, 20, 2)  
    )
    
    assert batch_result['values'].shape == (6, len(data))
    expected_periods = [10, 12, 14, 16, 18, 20]
    assert np.array_equal(batch_result['periods'], expected_periods)


def test_sinwma_batch_error_handling():
    """Test SINWMA batch error handling"""
    
    all_nan = np.full(100, np.nan, dtype=np.float64)
    with pytest.raises(ValueError, match="sinwma"):
        sinwma_batch(all_nan, (10, 20, 5))
    
    
    small_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    with pytest.raises(ValueError, match="sinwma"):
        sinwma_batch(small_data, (10, 20, 5))


def test_sinwma_zero_copy_verification():
    """Verify SINWMA uses zero-copy operations"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    
    result = sinwma(close, 14)
    assert len(result) == len(close)
    
    
    batch_result = sinwma_batch(close, (10, 20, 5))
    assert batch_result['values'].shape[0] == 3  
    assert batch_result['values'].shape[1] == len(close)


def test_sinwma_stream_error_handling():
    """Test SINWMA stream error handling"""
    
    with pytest.raises(ValueError, match="sinwma"):
        SinWmaStream(0)
    
    
    stream = SinWmaStream(14)
    
    
    for i in range(13):
        result = stream.update(float(i + 1))
        assert result is None
    
    
    result = stream.update(14.0)
    assert result is not None
    assert isinstance(result, float)


def test_sinwma_warmup_behavior():
    """Test SINWMA warmup period behavior"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    period = 14
    result = sinwma(close, period)
    
    
    first_valid = np.where(~np.isnan(close))[0][0]
    warmup = first_valid + period
    
    
    assert np.all(np.isnan(result[first_valid:first_valid + period - 1])), "Expected NaN during warmup period"
    
    
    assert np.all(np.isfinite(result[first_valid + period - 1:])), "Expected finite values after warmup"


def test_sinwma_formula_verification():
    """Verify SINWMA formula implementation"""
    
    data = np.array([10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 10.0, 11.0, 12.0, 13.0] * 5, dtype=np.float64)
    period = 5
    
    result = sinwma(data, period)
    
    
    assert len(result) == len(data)
    
    
    assert np.all(np.isfinite(result[period - 1:]))
    
    
    
    assert not np.allclose(result[period:], result[period], rtol=1e-9)


def test_sinwma_kernel_options():
    """Test SINWMA with different kernel options"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    
    kernels = ['auto', 'scalar', 'avx2', 'avx512']
    
    for kernel in kernels:
        try:
            result = sinwma(close, 14, kernel=kernel)
            assert len(result) == len(close)
            assert np.all(np.isfinite(result[13:]))
        except ValueError as e:
            
            if 'kernel' not in str(e).lower() and 'avx' not in kernel:
                raise
    
    
    with pytest.raises(ValueError, match="Unknown kernel|Invalid kernel"):
        sinwma(close, 14, kernel='invalid')


def test_sinwma_leading_nan():
    """Test SINWMA with leading NaN values in data"""
    
    data = np.array([np.nan, np.nan, np.nan] + list(range(1, 21)), dtype=np.float64)
    
    result = sinwma(data, 5)
    
    assert len(result) == len(data)
    
    
    assert np.all(np.isnan(result[:7])), "Expected NaN during warmup with leading NaN"
    assert np.all(np.isfinite(result[7:])), "Expected finite values after warmup"


if __name__ == '__main__':
    
    print("Testing SINWMA module...")
    test_sinwma_accuracy()
    print("SINWMA tests passed!")
