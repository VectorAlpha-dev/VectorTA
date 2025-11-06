"""
Python binding tests for SuperTrend indicator.
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

from test_utils import load_test_data, assert_close


class TestSuperTrend:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_supertrend_partial_params(self, test_data):
        """Test SuperTrend with partial parameters - mirrors check_supertrend_partial_params"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Test with default params (period=10, factor=3.0)
        trend, changed = ta_indicators.supertrend(high, low, close, 10, 3.0)
        assert len(trend) == len(high)
        assert len(changed) == len(high)
    
    def test_supertrend_accuracy(self, test_data):
        """Test SuperTrend matches expected values from Rust tests - mirrors check_supertrend_accuracy"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Using standard parameters: period=10, factor=3.0
        trend, changed = ta_indicators.supertrend(high, low, close, 10, 3.0)
        
        assert len(trend) == len(high)
        assert len(changed) == len(high)
        
        # Check last 5 values match expected
        expected_last_five_trend = [
            61811.479454208165,
            61721.73150878735,
            61459.10835790861,
            61351.59752211775,
            61033.18776990598,
        ]
        expected_last_five_changed = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Match Rust test tolerance exactly: absolute 1e-4 (no relative slack)
        assert_close(
            trend[-5:], 
            expected_last_five_trend,
            rtol=0.0,
            atol=1e-4,
            msg="SuperTrend trend last 5 values mismatch"
        )
        
        # Changed values match at absolute 1e-9 like Rust
        assert_close(
            changed[-5:], 
            expected_last_five_changed,
            rtol=0.0,
            atol=1e-9,
            msg="SuperTrend changed last 5 values mismatch"
        )
    
    def test_supertrend_zero_period(self):
        """Test SuperTrend fails with zero period - mirrors check_supertrend_zero_period"""
        high = np.array([10.0, 12.0, 13.0])
        low = np.array([9.0, 11.0, 12.5])
        close = np.array([9.5, 11.5, 13.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.supertrend(high, low, close, 0, 3.0)
    
    def test_supertrend_period_exceeds_length(self):
        """Test SuperTrend fails when period exceeds data length - mirrors check_supertrend_period_exceeds_length"""
        high = np.array([10.0, 12.0, 13.0])
        low = np.array([9.0, 11.0, 12.5])
        close = np.array([9.5, 11.5, 13.0])
        
        # Classic fast path may return Not enough valid data instead of Invalid period
        with pytest.raises(ValueError, match=r"Invalid period|Not enough valid data"):
            ta_indicators.supertrend(high, low, close, 10, 3.0)
    
    def test_supertrend_very_small_dataset(self):
        """Test SuperTrend fails with dataset smaller than period - mirrors check_supertrend_very_small_dataset"""
        high = np.array([42.0])
        low = np.array([40.0])
        close = np.array([41.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.supertrend(high, low, close, 10, 3.0)
    
    def test_supertrend_empty_input(self):
        """Test SuperTrend fails with empty input"""
        empty = np.array([])
        
        # Classic fast path may classify empty input as All values are NaN
        with pytest.raises(ValueError, match=r"Empty data|All values are NaN"):
            ta_indicators.supertrend(empty, empty, empty, 10, 3.0)
    
    def test_supertrend_all_nan(self):
        """Test SuperTrend with all NaN values"""
        high = np.full(100, np.nan)
        low = np.full(100, np.nan)
        close = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.supertrend(high, low, close, 10, 3.0)
    
    def test_supertrend_nan_handling(self, test_data):
        """Test SuperTrend handles NaN values correctly - mirrors check_supertrend_nan_handling"""
        high = test_data['high'].copy()
        low = test_data['low'].copy()
        close = test_data['close'].copy()
        
        # Insert some NaN values
        high[:5] = np.nan
        low[:5] = np.nan
        close[:5] = np.nan
        
        trend, changed = ta_indicators.supertrend(high, low, close, 10, 3.0)
        assert len(trend) == len(high)
        assert len(changed) == len(high)
        
        # First few values should be NaN
        assert np.all(np.isnan(trend[:5]))
    
    def test_supertrend_reinput(self, test_data):
        """Test SuperTrend applied to its own output - mirrors check_supertrend_reinput"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # First pass
        trend1, changed1 = ta_indicators.supertrend(high, low, close, 10, 3.0)
        
        # Second pass - apply SuperTrend to trend output
        trend2, changed2 = ta_indicators.supertrend(trend1, trend1, trend1, 5, 2.0)
        
        assert len(trend2) == len(trend1)
        assert len(changed2) == len(changed1)
    
    def test_supertrend_batch_single_set(self, test_data):
        """Test batch with single parameter combination"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Single parameter set
        result = ta_indicators.supertrend_batch(
            high, low, close,
            period_range=(10, 10, 0),
            factor_range=(3.0, 3.0, 0.0)
        )
        
        assert 'trend' in result
        assert 'changed' in result
        assert 'periods' in result
        assert 'factors' in result
        
        # Should have 1 row
        assert result['trend'].shape == (1, len(high))
        assert result['changed'].shape == (1, len(high))
        
        # Should match single calculation
        single_trend, single_changed = ta_indicators.supertrend(high, low, close, 10, 3.0)
        assert_close(result['trend'][0], single_trend, rtol=1e-10)
        assert_close(result['changed'][0], single_changed, rtol=1e-10)
    
    def test_supertrend_batch_multiple_params(self, test_data):
        """Test batch with multiple parameter values"""
        high = test_data['high'][:100]  # Use smaller dataset for speed
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        
        # Multiple periods and factors
        result = ta_indicators.supertrend_batch(
            high, low, close,
            period_range=(10, 14, 2),     # 10, 12, 14
            factor_range=(2.0, 3.0, 0.5)   # 2.0, 2.5, 3.0
        )
        
        # Should have 3 * 3 = 9 combinations
        assert result['trend'].shape == (9, 100)
        assert result['changed'].shape == (9, 100)
        assert len(result['periods']) == 9
        assert len(result['factors']) == 9
        
        # Verify parameter combinations
        expected_periods = [10, 10, 10, 12, 12, 12, 14, 14, 14]
        expected_factors = [2.0, 2.5, 3.0, 2.0, 2.5, 3.0, 2.0, 2.5, 3.0]
        
        assert_close(result['periods'], expected_periods, rtol=1e-10)
        assert_close(result['factors'], expected_factors, rtol=1e-10)
    
    def test_supertrend_streaming(self, test_data):
        """Test SuperTrend streaming functionality"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Create stream
        stream = ta_indicators.SuperTrendStream(period=10, factor=3.0)
        
        # Process data points
        stream_results = []
        for i in range(min(50, len(high))):
            result = stream.update(high[i], low[i], close[i])
            stream_results.append(result)
        
        # Early results should be None (warmup period)
        assert stream_results[0] is None
        
        # Later results should be tuples (trend, changed)
        valid_results = [r for r in stream_results if r is not None]
        if valid_results:
            assert isinstance(valid_results[0], tuple)
            assert len(valid_results[0]) == 2
    
    def test_supertrend_kernel_parameter(self, test_data):
        """Test SuperTrend with different kernel parameters"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        
        # Test with different kernels (if supported)
        kernels = [None, 'scalar', 'avx2', 'avx512']
        
        for kernel in kernels:
            try:
                if kernel:
                    trend, changed = ta_indicators.supertrend(high, low, close, 10, 3.0, kernel=kernel)
                else:
                    trend, changed = ta_indicators.supertrend(high, low, close, 10, 3.0)
                
                assert len(trend) == len(high)
                assert len(changed) == len(high)
            except ValueError as e:
                # Kernel might not be supported on this system
                if "kernel" not in str(e).lower():
                    raise
    
    def test_supertrend_edge_cases(self):
        """Test SuperTrend with edge cases"""
        # Test with very small factor
        high = np.array([100.0, 101.0, 102.0, 101.5, 100.5])
        low = np.array([99.0, 100.0, 101.0, 100.5, 99.5])
        close = np.array([99.5, 100.5, 101.5, 101.0, 100.0])
        
        trend, changed = ta_indicators.supertrend(high, low, close, 2, 0.1)
        assert len(trend) == len(high)
        assert len(changed) == len(high)
        
        # Test with large factor
        trend, changed = ta_indicators.supertrend(high, low, close, 2, 10.0)
        assert len(trend) == len(high)
        assert len(changed) == len(high)
