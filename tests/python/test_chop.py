"""
Python binding tests for CHOP indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

try:
    import my_project as ta_indicators
except ImportError:
    # If not in virtual environment, try to import from installed location
    try:
        import my_project as ta_indicators
    except ImportError:
        pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS
from rust_comparison import compare_with_rust


class TestChop:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()


    def test_chop_partial_params(self, test_data):
        """Test CHOP with partial parameters - mirrors check_chop_partial_params"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Test with period specified, others default
        result = ta_indicators.chop(high, low, close, period=30, scalar=100.0, drift=1)
        assert len(result) == len(close)
        assert isinstance(result, np.ndarray)
        
        # First period-1 values should be NaN
        assert np.all(np.isnan(result[:29]))
        
        # Should have valid values after warmup
        assert np.any(~np.isnan(result[29:]))
    
    def test_chop_accuracy(self, test_data):
        """Test CHOP matches expected values from Rust tests - mirrors check_chop_accuracy"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Expected values from Rust tests
        expected_last_5 = [
            49.98214330294626,
            48.90450693742312,
            46.63648608318844,
            46.19823574588033,
            56.22876423352909,
        ]
        
        result = ta_indicators.chop(high, low, close, period=14, scalar=100.0, drift=1)
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        assert_close(
            result[-5:],
            expected_last_5,
            rtol=1e-8,
            msg="CHOP last 5 values mismatch"
        )
        
        # Compare full output with Rust
        params = {'period': 14, 'scalar': 100.0, 'drift': 1}
        compare_with_rust('chop', result, 'hlc', params)
    
    def test_chop_default_candles(self, test_data):
        """Test CHOP with default parameters - mirrors check_chop_default_candles"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Default params: period=14, scalar=100.0, drift=1
        result = ta_indicators.chop(high, low, close, period=14, scalar=100.0, drift=1)
        assert len(result) == len(close)
        
        # First period-1 values should be NaN
        assert np.all(np.isnan(result[:13]))
        
        # Values after warmup should be finite
        assert np.any(~np.isnan(result[13:]))


    def test_chop_zero_period(self, test_data):
        """Test CHOP fails with zero period - mirrors check_chop_zero_period"""
        high = test_data['high'][:10]
        low = test_data['low'][:10]
        close = test_data['close'][:10]
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.chop(high, low, close, period=0, scalar=100.0, drift=1)
    
    def test_chop_period_exceeds_length(self, test_data):
        """Test CHOP fails when period exceeds data length - mirrors check_chop_period_exceeds_length"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0, 25.0])
        close = np.array([7.0, 17.0, 27.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.chop(high, low, close, period=10, scalar=100.0, drift=1)
    
    def test_chop_very_small_dataset(self):
        """Test CHOP fails with insufficient data"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.chop(single_point, single_point, single_point, period=14, scalar=100.0, drift=1)
    
    def test_chop_empty_input(self):
        """Test CHOP fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.chop(empty, empty, empty, period=14, scalar=100.0, drift=1)
    
    def test_chop_invalid_drift(self):
        """Test CHOP fails with invalid drift"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        with pytest.raises(ValueError, match="Invalid drift|drift"):
            ta_indicators.chop(data, data, data, period=3, scalar=100.0, drift=0)
    
    def test_chop_all_nan_input(self):
        """Test CHOP with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All.*NaN"):
            ta_indicators.chop(all_nan, all_nan, all_nan, period=14, scalar=100.0, drift=1)
    
    def test_chop_with_params(self, test_data):
        """Test CHOP with custom parameters"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Test with custom parameters
        period = 20
        scalar = 50.0
        drift = 2
        
        result = ta_indicators.chop(high, low, close, period=period, scalar=scalar, drift=drift)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == close.shape
        
        # First period-1 values should be NaN
        assert np.all(np.isnan(result[:period-1]))
        
        # Check that parameters affect the result
        result_default = ta_indicators.chop(high, low, close, period=14, scalar=100.0, drift=1)
        valid_idx = ~(np.isnan(result) | np.isnan(result_default))
        assert not np.allclose(result[valid_idx], result_default[valid_idx]), \
            "Custom parameters should produce different results"


    def test_chop_nan_handling(self, test_data):
        """Test CHOP handles NaN values correctly - mirrors check_chop_nan_handling"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        result = ta_indicators.chop(high, low, close, period=14, scalar=100.0, drift=1)
        assert len(result) == len(close)
        
        # After warmup period (240), no NaN values should exist if data is clean
        if len(result) > 240:
            # Check that not all values are NaN after index 240
            assert not np.all(np.isnan(result[240:])), "All values are NaN after index 240"
        
        # First period-1 values should be NaN
        assert np.all(np.isnan(result[:13])), "Expected NaN in warmup period"
    
    def test_chop_streaming(self, test_data):
        """Test CHOP streaming matches batch calculation - mirrors check_chop_streaming"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        period = 14
        scalar = 100.0
        drift = 1
        
        # Batch calculation
        batch_result = ta_indicators.chop(high, low, close, period=period, scalar=scalar, drift=drift)
        
        # Streaming calculation
        stream = ta_indicators.ChopStream(period=period, scalar=scalar, drift=drift)
        stream_values = []
        
        for i in range(len(close)):
            result = stream.update(high[i], low[i], close[i])
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        # Compare batch vs streaming
        assert len(batch_result) == len(stream_values)
        
        # Compare values where both are not NaN
        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, atol=1e-9,
                        msg=f"CHOP streaming mismatch at index {i}")
    
    def test_chop_batch(self, test_data):
        """Test CHOP batch processing - mirrors batch tests"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        
        result = ta_indicators.chop_batch(
            high, low, close,
            period_range=(14, 14, 0),  # Default period only
            scalar_range=(100.0, 100.0, 0.0),  # Default scalar only
            drift_range=(1, 1, 0)  # Default drift only
        )
        
        assert 'values' in result
        assert 'periods' in result
        assert 'scalars' in result
        assert 'drifts' in result
        
        # Should have 1 combination (default params)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        # Extract the single row
        default_row = result['values'][0]
        
        # Compare with single calculation
        single_result = ta_indicators.chop(high, low, close, period=14, scalar=100.0, drift=1)
        assert_close(
            default_row,
            single_result,
            rtol=1e-8,
            msg="CHOP batch default row mismatch"
        )
    
    def test_chop_batch_parameter_sweep(self, test_data):
        """Test CHOP batch processing with parameter sweep"""
        high = test_data['high'][:50]
        low = test_data['low'][:50]
        close = test_data['close'][:50]
        
        # Parameter sweep
        result = ta_indicators.chop_batch(
            high, low, close,
            period_range=(10, 20, 5),      # 10, 15, 20
            scalar_range=(50.0, 100.0, 50.0),  # 50, 100
            drift_range=(1, 2, 1)           # 1, 2
        )
        
        # Should have 3 * 2 * 2 = 12 combinations
        expected_combos = 12
        assert result['values'].shape == (expected_combos, len(close))
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
        single_result = ta_indicators.chop(high, low, close, period=10, scalar=50.0, drift=1)
        
        # Find the row with these parameters
        for i in range(expected_combos):
            if (result['periods'][i] == 10 and
                result['scalars'][i] == 50.0 and
                result['drifts'][i] == 1):
                assert_close(
                    result['values'][i],
                    single_result,
                    rtol=1e-8,
                    msg="Batch row doesn't match single calculation"
                )
                break
        else:
            pytest.fail("Could not find expected parameter combination")
    
    def test_chop_kernel_selection(self, test_data):
        """Test CHOP with different kernel selections"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        
        # Test with auto kernel (default)
        result_auto = ta_indicators.chop(high, low, close, period=14, scalar=100.0, drift=1, kernel="auto")
        
        # Test with scalar kernel
        result_scalar = ta_indicators.chop(high, low, close, period=14, scalar=100.0, drift=1, kernel="scalar")
        
        # Results should be very close regardless of kernel
        np.testing.assert_allclose(result_auto, result_scalar, rtol=1e-10, atol=1e-10)
        
        # Test invalid kernel
        with pytest.raises(ValueError, match="Unknown kernel"):
            ta_indicators.chop(high, low, close, period=14, scalar=100.0, drift=1, kernel="invalid")




if __name__ == '__main__':
    pytest.main([__file__, '-v'])