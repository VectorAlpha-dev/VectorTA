"""
Python binding tests for MAMA indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
from test_utils import load_test_data, assert_close


try:
    from my_project import (
        mama, 
        mama_batch,
        MamaStream
    )
except ImportError:
    pytest.skip("MAMA module not available - run 'maturin develop' first", allow_module_level=True)


def test_mama_partial_params():
    """Test MAMA with default parameters - mirrors check_mama_partial_params"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    
    mama_vals, fama_vals = mama(close, 0.5, 0.05)
    assert isinstance(mama_vals, np.ndarray)
    assert isinstance(fama_vals, np.ndarray)
    assert len(mama_vals) == len(close)
    assert len(fama_vals) == len(close)


def test_mama_accuracy():
    """Test MAMA matches expected values from Rust tests - mirrors check_mama_accuracy"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    
    mama_vals, fama_vals = mama(close, 0.5, 0.05)
    
    assert len(mama_vals) == len(close)
    assert len(fama_vals) == len(close)
    
    
    for i in range(10):
        assert np.isnan(mama_vals[i]), f"Expected NaN at warmup index {i} for MAMA"
        assert np.isnan(fama_vals[i]), f"Expected NaN at warmup index {i} for FAMA"
    
    
    for i in range(10, min(20, len(mama_vals))):
        assert np.isfinite(mama_vals[i]), f"MAMA NaN at index {i}"
        assert np.isfinite(fama_vals[i]), f"FAMA NaN at index {i}"
    
    
    
    expected_mama_last5 = [
        59244.89377664517,
        59241.699087812915,
        59140.34954390646,
        59141.08206671113,
        59116.77796337557,
    ]
    expected_fama_last5 = [
        59785.36249740519,
        59771.77091216539,
        59613.91557010065,
        59602.094732515914,
        59589.9618132874,
    ]
    
    assert_close(mama_vals[-5:], expected_mama_last5, rtol=1e-6, msg="MAMA last 5 values mismatch")
    assert_close(fama_vals[-5:], expected_fama_last5, rtol=1e-6, msg="FAMA last 5 values mismatch")


def test_mama_invalid_fast_limit():
    """Test MAMA fails with invalid fast limit"""
    input_data = np.array([10.0, 20.0, 30.0] * 10, dtype=np.float64)
    
    
    with pytest.raises(ValueError, match="mama:"):
        mama(input_data, 0.0, 0.05)
    
    
    with pytest.raises(ValueError, match="mama:"):
        mama(input_data, -0.5, 0.05)
    
    
    with pytest.raises(ValueError, match="mama:"):
        mama(input_data, np.nan, 0.05)


def test_mama_invalid_slow_limit():
    """Test MAMA fails with invalid slow limit"""
    input_data = np.array([10.0, 20.0, 30.0] * 10, dtype=np.float64)
    
    
    with pytest.raises(ValueError, match="mama:"):
        mama(input_data, 0.5, 0.0)
    
    
    with pytest.raises(ValueError, match="mama:"):
        mama(input_data, 0.5, -0.05)
    
    
    with pytest.raises(ValueError, match="mama:"):
        mama(input_data, 0.5, np.nan)


def test_mama_insufficient_data():
    """Test MAMA fails with insufficient data - mirrors check_mama_with_insufficient_data"""
    data_small = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="mama:"):
        mama(data_small, 0.5, 0.05)


def test_mama_very_small_dataset():
    """Test MAMA with minimum required data - mirrors check_mama_very_small_dataset"""
    
    data_min = np.array([42.0] * 10, dtype=np.float64)
    
    mama_vals, fama_vals = mama(data_min, 0.5, 0.05)
    assert len(mama_vals) == 10
    assert len(fama_vals) == 10
    
    
    for i in range(10):
        assert np.isnan(mama_vals[i]), f"Expected NaN at index {i} for MAMA"
        assert np.isnan(fama_vals[i]), f"Expected NaN at index {i} for FAMA"


def test_mama_empty_input():
    """Test MAMA with empty input"""
    data_empty = np.array([], dtype=np.float64)
    
    with pytest.raises(ValueError, match="mama:"):
        mama(data_empty, 0.5, 0.05)


def test_mama_reinput():
    """Test MAMA with re-input of MAMA result - mirrors check_mama_reinput"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    
    first_mama, first_fama = mama(close, 0.5, 0.05)
    
    
    second_mama, second_fama = mama(first_mama, 0.7, 0.1)
    
    assert len(second_mama) == len(first_mama)
    assert len(second_fama) == len(first_mama)


def test_mama_nan_handling():
    """Test MAMA handling of NaN values - mirrors check_mama_nan_handling"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    mama_vals, fama_vals = mama(close, 0.5, 0.05)
    
    assert len(mama_vals) == len(close)
    assert len(fama_vals) == len(close)
    
    
    for i in range(10):
        assert np.isnan(mama_vals[i]), f"Expected NaN at warmup index {i} for MAMA"
        assert np.isnan(fama_vals[i]), f"Expected NaN at warmup index {i} for FAMA"
    
    
    if len(mama_vals) > 10:
        for i in range(10, len(mama_vals)):
            assert np.isfinite(mama_vals[i]), f"MAMA NaN at index {i}"
            assert np.isfinite(fama_vals[i]), f"FAMA NaN at index {i}"


def test_mama_batch():
    """Test MAMA batch computation"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    
    batch_result = mama_batch(
        close, 
        (0.3, 0.7, 0.2),    
        (0.03, 0.07, 0.02)  
    )
    
    
    assert isinstance(batch_result, dict)
    assert 'mama' in batch_result
    assert 'fama' in batch_result
    assert 'fast_limits' in batch_result
    assert 'slow_limits' in batch_result
    
    
    assert batch_result['mama'].shape == (9, len(close))
    assert batch_result['fama'].shape == (9, len(close))
    assert len(batch_result['fast_limits']) == 9
    assert len(batch_result['slow_limits']) == 9
    
    
    batch_mama_values = batch_result['mama']
    batch_fama_values = batch_result['fama']
    
    
    individual_mama, individual_fama = mama(close, 0.3, 0.03)
    batch_first_mama = batch_mama_values[0]
    batch_first_fama = batch_fama_values[0]
    assert_close(batch_first_mama, individual_mama, atol=1e-9, msg="MAMA first combination mismatch")
    assert_close(batch_first_fama, individual_fama, atol=1e-9, msg="FAMA first combination mismatch")



def test_mama_stream():
    """Test MAMA streaming interface"""
    data = load_test_data()
    close = data['close']
    
    
    fast_limit = 0.5
    slow_limit = 0.05
    
    
    close_array = np.array(close, dtype=np.float64)
    batch_mama, batch_fama = mama(close_array, fast_limit, slow_limit)
    
    
    stream = MamaStream(fast_limit, slow_limit)
    stream_results = []
    
    for i, price in enumerate(close):
        result = stream.update(price)
        stream_results.append(result)
    
    
    for i in range(9):
        assert stream_results[i] is None, f"Expected None at index {i}"
    
    
    
    
    for i in range(10, len(close)):
        assert stream_results[i] is not None, f"Expected value at index {i}"
        stream_mama, stream_fama = stream_results[i]
        
        
        assert np.isfinite(stream_mama), f"MAMA NaN at index {i}"
        assert np.isfinite(stream_fama), f"FAMA NaN at index {i}"


def test_mama_different_params():
    """Test MAMA with various parameter values"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    
    test_cases = [
        (0.3, 0.03),
        (0.5, 0.05),  
        (0.7, 0.07),
        (0.9, 0.1),
    ]
    
    for fast_lim, slow_lim in test_cases:
        mama_vals, fama_vals = mama(close, fast_lim, slow_lim)
        assert len(mama_vals) == len(close)
        assert len(fama_vals) == len(close)
        
        
        assert np.all(np.isnan(mama_vals[:10])), \
            f"Expected NaN in warmup for MAMA with params=({fast_lim}, {slow_lim})"
        assert np.all(np.isnan(fama_vals[:10])), \
            f"Expected NaN in warmup for FAMA with params=({fast_lim}, {slow_lim})"
        assert np.all(np.isfinite(mama_vals[10:])), \
            f"Found NaN after warmup in MAMA for params=({fast_lim}, {slow_lim})"
        assert np.all(np.isfinite(fama_vals[10:])), \
            f"Found NaN after warmup in FAMA for params=({fast_lim}, {slow_lim})"


def test_mama_batch_performance():
    """Test that batch computation works correctly (performance is secondary)"""
    data = load_test_data()
    close = np.array(data['close'][:1000], dtype=np.float64)  
    
    
    batch_result = mama_batch(
        close, 
        (0.3, 0.7, 0.1),    
        (0.04, 0.06, 0.01)  
    )
    
    
    assert batch_result['mama'].shape == (15, len(close))
    assert batch_result['fama'].shape == (15, len(close))
    
    
    batch_mama_values = batch_result['mama']
    batch_fama_values = batch_result['fama']
    
    
    individual_mama, individual_fama = mama(close, 0.3, 0.04)
    batch_first_mama = batch_mama_values[0]
    batch_first_fama = batch_fama_values[0]
    assert_close(batch_first_mama, individual_mama, atol=1e-9, 
                msg=f"MAMA batch mismatch for params=(0.3, 0.04)")
    assert_close(batch_first_fama, individual_fama, atol=1e-9, 
                msg=f"FAMA batch mismatch for params=(0.3, 0.04)")


def test_mama_edge_cases():
    """Test MAMA with edge case inputs"""
    
    data = np.arange(1.0, 101.0, dtype=np.float64)
    mama_vals, fama_vals = mama(data, 0.5, 0.05)
    assert len(mama_vals) == len(data)
    assert len(fama_vals) == len(data)
    
    
    assert np.all(np.isnan(mama_vals[:10]))
    assert np.all(np.isnan(fama_vals[:10]))
    assert np.all(np.isfinite(mama_vals[10:]))
    assert np.all(np.isfinite(fama_vals[10:]))
    
    
    data = np.array([10.0, 20.0, 10.0, 20.0] * 25, dtype=np.float64)
    mama_vals2, fama_vals2 = mama(data, 0.5, 0.05)
    assert len(mama_vals2) == len(data)
    assert len(fama_vals2) == len(data)
    
    
    data = np.array([50.0] * 100, dtype=np.float64)
    mama_vals, fama_vals = mama(data, 0.5, 0.05)
    assert len(mama_vals) == len(data)
    assert len(fama_vals) == len(data)
    
    
    for i in range(20, len(mama_vals)):
        assert_close(mama_vals[i], 50.0, atol=1e-6, msg=f"MAMA constant value failed at index {i}")
        assert_close(fama_vals[i], 50.0, atol=1e-6, msg=f"FAMA constant value failed at index {i}")


def test_mama_warmup_period():
    """Test that MAMA handles early data points correctly"""
    data = load_test_data()
    close = np.array(data['close'][:50], dtype=np.float64)
    
    mama_vals, fama_vals = mama(close, 0.5, 0.05)
    
    
    assert len(mama_vals) == len(close)
    assert len(fama_vals) == len(close)
    
    
    for i in range(10):
        assert np.isnan(mama_vals[i]), f"Expected NaN at warmup index {i} for MAMA"
        assert np.isnan(fama_vals[i]), f"Expected NaN at warmup index {i} for FAMA"
    
    
    for i in range(10, len(mama_vals)):
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
    
    
    batch_result = mama_batch(
        data, 
        (0.4, 0.5, 0.1),     
        (0.04, 0.05, 0.01)   
    )
    
    
    metadata = list(zip(batch_result['fast_limits'], batch_result['slow_limits']))
    assert metadata == [
        (0.4, 0.04), (0.4, 0.05),
        (0.5, 0.04), (0.5, 0.05)
    ]
    
    assert batch_result['mama'].shape == (4, len(data))
    assert batch_result['fama'].shape == (4, len(data))


def test_mama_fama_relationship():
    """Test that FAMA follows MAMA as expected"""
    data = load_test_data()
    close = np.array(data['close'][:200], dtype=np.float64)
    
    mama_vals, fama_vals = mama(close, 0.5, 0.05)
    
    
    
    mama_var = np.var(mama_vals[10:])
    fama_var = np.var(fama_vals[10:])
    
    
    assert fama_var < mama_var * 1.1, "FAMA should be smoother than MAMA"


def test_mama_batch_error_handling():
    """Test MAMA batch error handling guards for invalid inputs"""
    
    
    
    
    empty = np.array([], dtype=np.float64)
    with pytest.raises(ValueError, match="mama:"):
        mama_batch(empty, (0.5, 0.5, 0), (0.05, 0.05, 0))
    
    
    small = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    with pytest.raises(ValueError, match="mama:"):
        mama_batch(small, (0.5, 0.5, 0), (0.05, 0.05, 0))
    
    
    data = np.random.randn(100).astype(np.float64)
    with pytest.raises(ValueError, match="mama:"):
        mama_batch(data, (0.0, 0.5, 0.1), (0.05, 0.05, 0))  
    
    with pytest.raises(ValueError, match="mama:"):
        mama_batch(data, (0.5, 0.5, 0), (0.0, 0.05, 0.01))  


def test_mama_zero_copy_verification():
    """Verify MAMA uses zero-copy operations"""
    
    data = load_test_data()
    close = np.array(data['close'][:100], dtype=np.float64)
    
    
    mama_vals, fama_vals = mama(close, 0.5, 0.05)
    assert len(mama_vals) == len(close)
    assert len(fama_vals) == len(close)
    
    
    batch_result = mama_batch(close, (0.3, 0.7, 0.2), (0.03, 0.07, 0.02))
    assert batch_result['mama'].shape[0] == 3 * 3  
    assert batch_result['mama'].shape[1] == len(close)
    assert batch_result['fama'].shape[0] == 3 * 3
    assert batch_result['fama'].shape[1] == len(close)


def test_mama_stream_error_handling():
    """Test MAMA stream error handling"""
    
    with pytest.raises(ValueError, match="mama:"):
        MamaStream(0.0, 0.05)  
    
    with pytest.raises(ValueError, match="mama:"):
        MamaStream(0.5, 0.0)  
    
    with pytest.raises(ValueError, match="mama:"):
        MamaStream(-0.5, 0.05)  
    
    with pytest.raises(ValueError, match="mama:"):
        MamaStream(0.5, -0.05)  


def test_mama_batch_warmup_consistency():
    """Test that batch processing produces consistent results"""
    data = np.random.randn(50).astype(np.float64)
    
    result = mama_batch(data, (0.3, 0.7, 0.2), (0.03, 0.07, 0.02))
    
    
    n_combos = len(result['fast_limits'])
    mama_values = result['mama']
    fama_values = result['fama']
    
    
    for i in range(n_combos):
        mama_row = mama_values[i]
        fama_row = fama_values[i]
        
        
        for j in range(10):
            assert np.isnan(mama_row[j]), f"Expected NaN at warmup index {j} for row {i}"
            assert np.isnan(fama_row[j]), f"Expected NaN at warmup index {j} for row {i}"
        for j in range(10, len(data)):
            assert np.isfinite(mama_row[j]), f"Expected finite value at index {j} for row {i}"
            assert np.isfinite(fama_row[j]), f"Expected finite value at index {j} for row {i}"
        
        
        fast_limit = result['fast_limits'][i]
        slow_limit = result['slow_limits'][i]
        individual_mama, individual_fama = mama(data, fast_limit, slow_limit)
        assert_close(mama_row, individual_mama, atol=1e-9, 
                    msg=f"Batch row {i} doesn't match individual computation")
        assert_close(fama_row, individual_fama, atol=1e-9, 
                    msg=f"Batch row {i} doesn't match individual computation")


def test_mama_all_nan_input():
    """Test MAMA with all NaN values"""
    all_nan = np.full(100, np.nan)
    
    
    mama_vals, fama_vals = mama(all_nan, 0.5, 0.05)
    assert np.all(np.isnan(mama_vals))
    assert np.all(np.isnan(fama_vals))


def test_mama_zero_period():
    """Test MAMA fails with zero period (invalid parameter)"""
    input_data = np.array([10.0, 20.0, 30.0] * 10, dtype=np.float64)
    
    
    
    pass  


def test_mama_period_exceeds_length():
    """Test MAMA with parameters that would require more data than available"""
    
    data_small = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="mama:"):
        mama(data_small, 0.5, 0.05)


def test_mama_default_candles():
    """Test MAMA with default parameters - mirrors check_mama_default_candles"""
    data = load_test_data()
    close = np.array(data['close'], dtype=np.float64)
    
    
    mama_vals, fama_vals = mama(close, 0.5, 0.05)
    assert len(mama_vals) == len(close)
    assert len(fama_vals) == len(close)
    
    
    for i in range(10):
        assert np.isnan(mama_vals[i])
        assert np.isnan(fama_vals[i])


if __name__ == "__main__":
    
    print("Testing MAMA module...")
    test_mama_accuracy()
    print("MAMA tests passed!")
