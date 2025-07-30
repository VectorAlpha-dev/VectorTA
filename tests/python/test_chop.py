import pytest
import numpy as np
import sys
from pathlib import Path

try:
    import my_project as ta
except ImportError:
    # If not in virtual environment, try to import from installed location
    try:
        import my_project as ta
    except ImportError:
        pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS


@pytest.fixture
def test_data():
    """Load test data"""
    return load_test_data()


def test_chop_basic(test_data):
    """Test basic CHOP functionality"""
    high = test_data['high']
    low = test_data['low']
    close = test_data['close']
    
    # Test with default parameters
    result = ta.chop(high, low, close)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == close.shape
    assert result.dtype == np.float64
    
    # First period-1 values should be NaN
    assert np.all(np.isnan(result[:13]))  # Default period is 14
    
    # Values after warmup should be finite
    non_nan_mask = ~np.isnan(result[13:])
    assert np.any(non_nan_mask), "All values are NaN after warmup period"
    
    # CHOP values should be in reasonable range (0-100 typically)
    valid_values = result[~np.isnan(result)]
    assert np.all(valid_values >= 0), "CHOP values should be non-negative"
    assert np.all(valid_values <= 200), "CHOP values seem too large"


def test_chop_with_params(test_data):
    """Test CHOP with custom parameters"""
    high = test_data['high']
    low = test_data['low']
    close = test_data['close']
    
    # Test with custom parameters
    period = 20
    scalar = 50.0
    drift = 2
    
    result = ta.chop(high, low, close, period=period, scalar=scalar, drift=drift)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == close.shape
    
    # First period-1 values should be NaN
    assert np.all(np.isnan(result[:period-1]))
    
    # Check that parameters affect the result
    result_default = ta.chop(high, low, close)
    valid_idx = ~(np.isnan(result) | np.isnan(result_default))
    assert not np.allclose(result[valid_idx], result_default[valid_idx]), \
        "Custom parameters should produce different results"


def test_chop_kernel_selection(test_data):
    """Test CHOP with different kernel selections"""
    high = test_data['high']
    low = test_data['low']
    close = test_data['close']
    
    # Test with auto kernel (default)
    result_auto = ta.chop(high, low, close, kernel="auto")
    
    # Test with scalar kernel
    result_scalar = ta.chop(high, low, close, kernel="scalar")
    
    # Results should be very close regardless of kernel
    np.testing.assert_allclose(result_auto, result_scalar, rtol=1e-10, atol=1e-10)
    
    # Test invalid kernel
    with pytest.raises(ValueError, match="Unknown kernel"):
        ta.chop(high, low, close, kernel="invalid")


def test_chop_edge_cases():
    """Test CHOP with edge cases"""
    # Empty arrays
    with pytest.raises(ValueError):
        ta.chop(np.array([]), np.array([]), np.array([]))
    
    # Mismatched lengths
    with pytest.raises(ValueError):
        ta.chop(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0, 2.0]))
    
    # All NaN values
    all_nan = np.full(100, np.nan)
    with pytest.raises(ValueError, match="All.*NaN"):
        ta.chop(all_nan, all_nan, all_nan)
    
    # Period exceeds data length
    small_data = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="period"):
        ta.chop(small_data, small_data, small_data, period=10)
    
    # Zero period
    data = np.random.randn(100)
    with pytest.raises(ValueError, match="period"):
        ta.chop(data, data, data, period=0)
    
    # Zero drift
    with pytest.raises(ValueError, match="drift"):
        ta.chop(data, data, data, drift=0)


def test_chop_nan_handling(test_data):
    """Test CHOP with NaN values in input"""
    high = test_data['high'].copy()
    low = test_data['low'].copy()
    close = test_data['close'].copy()
    
    # Insert some NaN values
    high[10:20] = np.nan
    low[15:25] = np.nan
    close[5:15] = np.nan
    
    result = ta.chop(high, low, close)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == close.shape
    
    # Should handle NaN values gracefully
    # Values after sufficient non-NaN data should be valid
    assert np.any(~np.isnan(result[30:]))


def test_chop_streaming(test_data):
    """Test CHOP streaming functionality"""
    high = test_data['high'][:100]
    low = test_data['low'][:100]
    close = test_data['close'][:100]
    
    # Batch calculation
    batch_result = ta.chop(high, low, close)
    
    # Streaming calculation
    stream = ta.ChopStream()
    stream_results = []
    
    for i in range(len(close)):
        value = stream.update(high[i], low[i], close[i])
        if value is None:
            stream_results.append(np.nan)
        else:
            stream_results.append(value)
    
    stream_result = np.array(stream_results)
    
    # Results should match
    np.testing.assert_allclose(batch_result, stream_result, rtol=1e-10, atol=1e-10)


def test_chop_streaming_with_params():
    """Test CHOP streaming with custom parameters"""
    period = 10
    scalar = 75.0
    drift = 3
    
    # Create stream with custom parameters
    stream = ta.ChopStream(period=period, scalar=scalar, drift=drift)
    
    # Generate some test data
    np.random.seed(42)
    n = 50
    high = 100 + np.cumsum(np.random.randn(n))
    low = high - np.abs(np.random.randn(n))
    close = low + np.random.rand(n) * (high - low)
    
    # Stream calculation
    stream_results = []
    for i in range(n):
        value = stream.update(high[i], low[i], close[i])
        if value is None:
            stream_results.append(np.nan)
        else:
            stream_results.append(value)
    
    stream_result = np.array(stream_results)
    
    # Batch calculation with same parameters
    batch_result = ta.chop(high, low, close, period=period, scalar=scalar, drift=drift)
    
    # Results should match
    np.testing.assert_allclose(batch_result, stream_result, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(False, reason="Long running test")
def test_chop_batch_simple():
    """Test CHOP batch processing with simple parameter sweep"""
    # Generate test data
    np.random.seed(42)
    n = 100
    high = 100 + np.cumsum(np.random.randn(n))
    low = high - np.abs(np.random.randn(n))
    close = low + np.random.rand(n) * (high - low)
    
    # Simple batch test with single parameter set
    result = ta.chop_batch(
        high, low, close,
        period_range=(14, 14, 0),
        scalar_range=(100.0, 100.0, 0.0),
        drift_range=(1, 1, 0)
    )
    
    assert 'values' in result
    assert 'periods' in result
    assert 'scalars' in result
    assert 'drifts' in result
    
    # Should have one combination
    assert result['values'].shape == (1, n)
    assert len(result['periods']) == 1
    assert result['periods'][0] == 14
    assert result['scalars'][0] == 100.0
    assert result['drifts'][0] == 1
    
    # Compare with single calculation
    single_result = ta.chop(high, low, close)
    np.testing.assert_allclose(result['values'][0], single_result, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(False, reason="Long running test") 
def test_chop_batch_parameter_sweep():
    """Test CHOP batch processing with parameter sweep"""
    # Generate test data
    np.random.seed(42)
    n = 100
    high = 100 + np.cumsum(np.random.randn(n))
    low = high - np.abs(np.random.randn(n))
    close = low + np.random.rand(n) * (high - low)
    
    # Parameter sweep
    result = ta.chop_batch(
        high, low, close,
        period_range=(10, 20, 5),      # 10, 15, 20
        scalar_range=(50.0, 100.0, 50.0),  # 50, 100
        drift_range=(1, 2, 1)           # 1, 2
    )
    
    # Should have 3 * 2 * 2 = 12 combinations
    expected_combos = 12
    assert result['values'].shape == (expected_combos, n)
    assert len(result['periods']) == expected_combos
    assert len(result['scalars']) == expected_combos
    assert len(result['drifts']) == expected_combos
    
    # Verify all parameter combinations exist
    periods_set = set(result['periods'])
    scalars_set = set(result['scalars'])
    drifts_set = set(result['drifts'])
    
    assert periods_set == {10, 15, 20}
    assert scalars_set == {50.0, 100.0}
    assert drifts_set == {1, 2}
    
    # Verify one specific combination matches single calculation
    single_result = ta.chop(high, low, close, period=10, scalar=50.0, drift=1)
    
    # Find the row with these parameters
    for i in range(expected_combos):
        if (result['periods'][i] == 10 and 
            result['scalars'][i] == 50.0 and 
            result['drifts'][i] == 1):
            np.testing.assert_allclose(result['values'][i], single_result, rtol=1e-10, atol=1e-10)
            break
    else:
        pytest.fail("Could not find expected parameter combination")


def test_chop_accuracy(test_data):
    """Test CHOP calculation accuracy against expected values"""
    high = test_data['high']
    low = test_data['low'] 
    close = test_data['close']
    
    result = ta.chop(high, low, close)
    
    # Test last 5 values against expected
    expected_last_5 = [
        49.98214330294626,
        48.90450693742312,
        46.63648608318844,
        46.19823574588033,
        56.22876423352909,
    ]
    
    np.testing.assert_allclose(result[-5:], expected_last_5, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(False, reason="Long running test")
def test_chop_performance(test_data):
    """Test CHOP performance with full dataset"""
    high = test_data['high']
    low = test_data['low']
    close = test_data['close']
    
    import time
    
    # Warmup
    for _ in range(10):
        _ = ta.chop(high, low, close)
    
    # Time the calculation
    times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = ta.chop(high, low, close)
        times.append(time.perf_counter() - start)
    
    median_time = np.median(times) * 1000  # Convert to ms
    print(f"\nCHOP median time: {median_time:.2f} ms for {len(close)} data points")
    
    # Should be reasonably fast (adjust threshold as needed)
    assert median_time < 50, f"CHOP calculation too slow: {median_time:.2f} ms"