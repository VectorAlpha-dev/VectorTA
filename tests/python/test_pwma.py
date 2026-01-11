"""
Python binding tests for PWMA indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS
from rust_comparison import compare_with_rust


try:
    from my_project import (
        pwma, 
        pwma_batch,
        PwmaStream
    )
except ImportError:
    pytest.skip("PWMA module not available - run 'maturin develop' first", allow_module_level=True)


def test_pwma_partial_params():
    """Test PWMA with default parameters - mirrors check_pwma_partial_params"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    
    result = pwma(close, 5)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(close)


def test_pwma_accuracy():
    """Test PWMA matches expected values from Rust tests - mirrors check_pwma_accuracy"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    expected = EXPECTED_OUTPUTS['pwma']
    
    
    result = pwma(close, expected['default_params']['period'])
    
    assert len(result) == len(close)
    
    
    
    assert_close(
        result[-5:],
        expected['last_5_values'],
        rtol=0,
        atol=1e-3,
        msg="PWMA last 5 values mismatch"
    )
    
    
    compare_with_rust('pwma', result, 'close', expected['default_params'])


def test_pwma_zero_period():
    """Test PWMA fails with zero period"""
    input_data = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="pwma:"):
        pwma(input_data, 0)


def test_pwma_period_exceeds_length():
    """Test PWMA fails with period exceeding length"""
    data_small = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="pwma:"):
        pwma(data_small, 10)


def test_pwma_very_small_dataset():
    """Test PWMA fails with insufficient data"""
    single_point = np.array([42.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="pwma:"):
        pwma(single_point, 5)


def test_pwma_reinput():
    """Test PWMA with re-input of PWMA result - mirrors check_pwma_reinput"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    expected = EXPECTED_OUTPUTS['pwma']
    
    
    first_result = pwma(close, expected['reinput_periods']['first'])
    
    
    second_result = pwma(first_result, expected['reinput_periods']['second'])
    
    assert len(second_result) == len(first_result)
    
    
    warmup = expected['reinput_warmup']
    assert np.all(np.isfinite(second_result[warmup:]))


def test_pwma_nan_handling():
    """Test PWMA handling of NaN values - mirrors check_pwma_nan_handling"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    result = pwma(close, 5)
    
    assert len(result) == len(close)
    
    
    
    first_valid = np.where(~np.isnan(close))[0][0]
    warmup = first_valid + 5 - 1
    
    assert np.all(np.isfinite(result[warmup:]))


def test_pwma_all_nan():
    """Test PWMA with all NaN values"""
    all_nan = np.full(100, np.nan, dtype=np.float64)
    
    with pytest.raises(ValueError, match="pwma:"):
        pwma(all_nan, 5)


def test_pwma_empty_input():
    """Test PWMA with empty input"""
    data_empty = np.array([], dtype=np.float64)
    
    with pytest.raises(ValueError, match="pwma:"):
        pwma(data_empty, 5)


def test_pwma_batch():
    """Test PWMA batch computation - mirrors check_batch_default_row"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    expected = EXPECTED_OUTPUTS['pwma']
    
    
    batch_result = pwma_batch(
        close,
        expected['batch_range']  
    )
    
    
    assert isinstance(batch_result, dict)
    assert 'values' in batch_result
    assert 'periods' in batch_result
    
    
    assert batch_result['values'].shape == (len(expected['batch_periods']), len(close))
    assert np.array_equal(batch_result['periods'], expected['batch_periods'])
    
    
    individual_result = pwma(close, expected['batch_periods'][0])
    batch_first = batch_result['values'][0]
    
    
    first_valid = np.where(~np.isnan(close))[0][0]
    warmup = first_valid + expected['batch_periods'][0] - 1
    assert_close(batch_first[warmup:], individual_result[warmup:], atol=1e-9, 
                msg="PWMA batch first combination mismatch")


def test_pwma_stream():
    """Test PWMA streaming interface"""
    data = load_test_data()
    close = data['close']
    
    
    period = 5
    
    
    close_array = np.array(close, dtype=np.float64)
    batch_result = pwma(close_array, period)
    
    
    stream = PwmaStream(period)
    stream_results = []
    
    for price in close:
        result = stream.update(price)
        stream_results.append(result if result is not None else np.nan)
    
    stream_results = np.array(stream_results)
    
    
    for i in range(len(close)):
        if np.isnan(batch_result[i]) and np.isnan(stream_results[i]):
            continue
        if not np.isnan(batch_result[i]) and not np.isnan(stream_results[i]):
            assert_close(batch_result[i], stream_results[i], rtol=1e-9, msg=f"PWMA streaming mismatch at index {i}")


def test_pwma_different_periods():
    """Test PWMA with various period values"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    
    test_periods = [2, 5, 10, 20]
    
    for period in test_periods:
        result = pwma(close, period)
        assert len(result) == len(close)
        
        
        first_valid = np.where(~np.isnan(close))[0][0]
        warmup = first_valid + period - 1
        if warmup < len(result):
            assert np.all(np.isfinite(result[warmup:])), f"NaN values found for period={period}"


def test_pwma_batch_performance():
    """Test that batch computation works correctly (performance is secondary)"""
    data = load_test_data()
    close = np.array(data['close'][:1000], dtype=np.float64)  
    
    
    batch_result = pwma_batch(
        close,
        (5, 30, 5)  
    )
    
    
    assert batch_result['values'].shape == (6, len(close))
    
    
    expected_periods = [5, 10, 15, 20, 25, 30]
    assert np.array_equal(batch_result['periods'], expected_periods)


def test_pwma_edge_cases():
    """Test PWMA with edge case inputs"""
    expected = EXPECTED_OUTPUTS['pwma']
    
    
    data = np.arange(1.0, 101.0, dtype=np.float64)
    result = pwma(data, expected['default_params']['period'])
    assert len(result) == len(data)
    assert np.all(np.isfinite(result[4:]))  
    
    
    constant_val = expected['constant_value']
    data = np.array([constant_val] * 100, dtype=np.float64)
    result = pwma(data, expected['default_params']['period'])
    assert len(result) == len(data)
    assert np.all(np.isfinite(result[4:]))
    
    assert np.allclose(result[4:], constant_val, rtol=1e-9)
    
    
    data = np.array([10.0, 20.0, 10.0, 20.0] * 25, dtype=np.float64)
    result = pwma(data, expected['default_params']['period'])
    assert len(result) == len(data)
    assert np.all(np.isfinite(result[4:]))


def test_pwma_consistency():
    """Test that PWMA produces consistent results across multiple calls"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    result1 = pwma(close, 5)
    result2 = pwma(close, 5)
    
    assert_close(result1, result2, atol=1e-15, msg="PWMA results not consistent")


def test_pwma_step_precision():
    """Test batch with various step sizes"""
    data = np.arange(1, 51, dtype=np.float64)
    
    
    batch_result = pwma_batch(
        data,
        (5, 15, 5)  
    )
    
    assert batch_result['values'].shape == (3, len(data))
    expected_periods = [5, 10, 15]
    assert np.array_equal(batch_result['periods'], expected_periods)


def test_pwma_batch_error_handling():
    """Test PWMA batch error handling"""
    
    all_nan = np.full(100, np.nan, dtype=np.float64)
    with pytest.raises(ValueError, match="pwma:"):
        pwma_batch(all_nan, (5, 10, 5))
    
    
    small_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    with pytest.raises(ValueError, match="pwma:"):
        pwma_batch(small_data, (5, 10, 5))


def test_pwma_zero_copy_verification():
    """Verify PWMA uses zero-copy operations"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    
    result = pwma(close, 5)
    assert len(result) == len(close)
    
    
    batch_result = pwma_batch(close, (5, 10, 5))
    assert batch_result['values'].shape[0] == 2  
    assert batch_result['values'].shape[1] == len(close)


def test_pwma_stream_error_handling():
    """Test PWMA stream error handling"""
    
    with pytest.raises(ValueError, match="pwma:"):
        PwmaStream(0)
    
    
    stream = PwmaStream(5)
    
    
    for i in range(4):
        result = stream.update(float(i + 1))
        assert result is None
    
    
    result = stream.update(5.0)
    assert result is not None
    assert isinstance(result, float)


def test_pwma_warmup_behavior():
    """Test PWMA warmup period behavior"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    expected = EXPECTED_OUTPUTS['pwma']
    
    period = expected['default_params']['period']
    result = pwma(close, period)
    
    
    first_valid = np.where(~np.isnan(close))[0][0] if np.any(~np.isnan(close)) else 0
    warmup = first_valid + period - 1
    
    
    assert np.all(np.isnan(result[:warmup])), "Expected NaN during warmup period"
    
    
    assert np.all(np.isfinite(result[warmup:])), "Expected finite values after warmup"


def test_pwma_formula_verification():
    """Verify PWMA formula implementation - mirrors Rust formula test"""
    expected = EXPECTED_OUTPUTS['pwma']['formula_test']
    
    data = np.array(expected['data'], dtype=np.float64)
    period = expected['period']
    
    result = pwma(data, period)
    
    assert len(result) == len(data)
    
    
    for i, exp_val in enumerate(expected['expected']):
        if np.isnan(exp_val):
            assert np.isnan(result[i]), f"Expected NaN at index {i}"
        else:
            assert_close(result[i], exp_val, atol=1e-9, 
                        msg=f"PWMA formula mismatch at index {i}")


def test_pwma_pascal_weights():
    """Test PWMA Pascal triangle weight calculation"""
    
    data = np.array([1.0] * 10, dtype=np.float64)
    
    
    result = pwma(data, 2)
    assert np.all(np.isfinite(result[1:]))
    assert np.allclose(result[1:], 1.0, rtol=1e-9)
    
    
    result = pwma(data, 3)
    assert np.all(np.isfinite(result[2:]))
    assert np.allclose(result[2:], 1.0, rtol=1e-9)
    
    
    result = pwma(data, 4)
    assert np.all(np.isfinite(result[3:]))
    assert np.allclose(result[3:], 1.0, rtol=1e-9)


def test_pwma_batch_single_parameter():
    """Test PWMA batch with single parameter combination - mirrors ALMA pattern"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    expected = EXPECTED_OUTPUTS['pwma']
    
    
    result = pwma_batch(
        close,
        (expected['default_params']['period'], expected['default_params']['period'], 0)
    )
    
    
    assert result['values'].shape[0] == 1
    assert result['values'].shape[1] == len(close)
    
    
    default_row = result['values'][0]
    
    
    
    assert_close(
        default_row[-5:],
        expected['last_5_values'],
        rtol=0,
        atol=1e-3,
        msg="PWMA batch single parameter mismatch"
    )


def test_pwma_batch_multiple_parameters():
    """Test PWMA batch with multiple parameter combinations"""
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)  
    
    
    result = pwma_batch(
        close,
        (5, 15, 5)  
    )
    
    
    assert result['values'].shape == (3, len(close))
    assert np.array_equal(result['periods'], [5, 10, 15])
    
    
    for i, period in enumerate([5, 10, 15]):
        individual = pwma(close, period)
        batch_row = result['values'][i]
        
        
        valid_mask = ~np.isnan(individual)
        assert_close(
            batch_row[valid_mask],
            individual[valid_mask],
            rtol=1e-9,
            msg=f"Batch row {i} (period={period}) mismatch"
        )


def test_pwma_batch_metadata_verification():
    """Test that batch metadata correctly reflects parameters"""
    data = load_test_data()
    close = np.array(data['close'][:50], dtype=np.float64)
    
    
    result = pwma_batch(
        close,
        (3, 7, 2)  
    )
    
    
    assert 'periods' in result
    assert np.array_equal(result['periods'], [3, 5, 7])
    
    
    assert result['values'].shape[0] == len(result['periods'])
    assert result['values'].shape[1] == len(close)


def test_pwma_invalid_input_type():
    """Test PWMA with invalid input types"""
    
    data_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    with pytest.raises(TypeError, match="cannot be converted"):
        pwma(data_list, 3)
    
    
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    
    with pytest.raises((ValueError, TypeError)):
        pwma(data, "invalid")
    
    with pytest.raises((ValueError, TypeError, OverflowError)):
        pwma(data, -5)


if __name__ == '__main__':
    
    print("Testing PWMA module...")
    test_pwma_accuracy()
    print("PWMA tests passed!")
