"""
Python binding tests for KDJ indicator.
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


class TestKdj:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_kdj_partial_params(self, test_data):
        """Test KDJ with partial parameters - mirrors check_kdj_partial_params"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Test with partial params (using defaults for some)
        k, d, j = ta_indicators.kdj(
            high, low, close,
            fast_k_period=9,  # default
            slow_k_period=4,  # custom
            slow_k_ma_type="sma",  # default
            slow_d_period=3,  # default
            slow_d_ma_type="sma"  # default
        )
        
        assert len(k) == len(close)
        assert len(d) == len(close)
        assert len(j) == len(close)
    
    def test_kdj_accuracy(self, test_data):
        """Test KDJ matches expected values from Rust tests - mirrors check_kdj_accuracy"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Test with default parameters
        k, d, j = ta_indicators.kdj(
            high, low, close,
            fast_k_period=9,
            slow_k_period=3,
            slow_k_ma_type="sma",
            slow_d_period=3,
            slow_d_ma_type="sma"
        )
        
        # Expected values from Rust tests
        expected_k = [
            58.04341315415984,
            61.56034740940419,
            58.056304282719545,
            56.10961365678364,
            51.43992326447119,
        ]
        expected_d = [
            49.57659409278555,
            56.81719223571944,
            59.22002161542779,
            58.57542178296905,
            55.20194706799139,
        ]
        expected_j = [
            74.97705127690843,
            71.04665775677368,
            55.72886961730306,
            51.17799740441281,
            43.91587565743079,
        ]
        
        # Check last 5 values match expected
        assert_close(k[-5:], expected_k, rtol=1e-4, msg="KDJ K last 5 values mismatch")
        assert_close(d[-5:], expected_d, rtol=1e-4, msg="KDJ D last 5 values mismatch")
        assert_close(j[-5:], expected_j, rtol=1e-4, msg="KDJ J last 5 values mismatch")
    
    def test_kdj_default_params(self, test_data):
        """Test KDJ with default parameters - mirrors check_kdj_default_candles"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # All default params
        k, d, j = ta_indicators.kdj(high, low, close)
        
        assert len(k) == len(close)
        assert len(d) == len(close)
        assert len(j) == len(close)
    
    def test_kdj_zero_period(self):
        """Test KDJ fails with zero period - mirrors check_kdj_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.kdj(
                input_data, input_data, input_data,
                fast_k_period=0
            )
    
    def test_kdj_period_exceeds_length(self):
        """Test KDJ fails when period exceeds data length - mirrors check_kdj_period_exceeds_length"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.kdj(
                input_data, input_data, input_data,
                fast_k_period=10
            )
    
    def test_kdj_very_small_dataset(self):
        """Test KDJ fails with insufficient data - mirrors check_kdj_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.kdj(
                single_point, single_point, single_point,
                fast_k_period=9
            )
    
    def test_kdj_all_nan(self):
        """Test KDJ fails with all-NaN data - mirrors check_kdj_all_nan"""
        input_data = np.array([np.nan, np.nan, np.nan])
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.kdj(input_data, input_data, input_data)
    
    def test_kdj_empty_input(self):
        """Test KDJ fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.kdj(empty, empty, empty)
    
    def test_kdj_mismatched_lengths(self):
        """Test KDJ fails with mismatched input lengths"""
        high = np.array([1.0, 2.0, 3.0])
        low = np.array([1.0, 2.0])
        close = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="must have the same length"):
            ta_indicators.kdj(high, low, close)
    
    def test_kdj_nan_handling(self, test_data):
        """Test KDJ handles NaN values correctly - mirrors check_kdj_nan_handling"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        k, d, j = ta_indicators.kdj(high, low, close)
        
        # Check that after warmup period, no NaN values exist
        if len(k) > 50:
            for i in range(50, len(k)):
                assert not np.isnan(k[i]), f"Expected no NaN in K after index 50 at {i}"
                assert not np.isnan(d[i]), f"Expected no NaN in D after index 50 at {i}"
                assert not np.isnan(j[i]), f"Expected no NaN in J after index 50 at {i}"
    
    def test_kdj_stream(self):
        """Test KDJ streaming functionality"""
        stream = ta_indicators.KdjStream(
            fast_k_period=9,
            slow_k_period=3,
            slow_k_ma_type="sma",
            slow_d_period=3,
            slow_d_ma_type="sma"
        )
        
        # Feed some data
        values = [
            (10.0, 5.0, 7.0),
            (12.0, 6.0, 8.0),
            (15.0, 8.0, 10.0),
            (14.0, 9.0, 11.0),
            (16.0, 10.0, 13.0),
            (18.0, 11.0, 14.0),
            (17.0, 12.0, 15.0),
            (20.0, 13.0, 16.0),
            (22.0, 14.0, 18.0),
            (21.0, 15.0, 19.0),
        ]
        
        results = []
        for high, low, close in values:
            result = stream.update(high, low, close)
            if result is not None:
                results.append(result)
        
        # After feeding 10 values with period 9, we should have at least 1 result
        assert len(results) >= 1
        
        # Each result should be a tuple of (k, d, j)
        for result in results:
            assert len(result) == 3
            k, d, j = result
            assert isinstance(k, float)
            assert isinstance(d, float)
            assert isinstance(j, float)
    
    def test_kdj_batch(self, test_data):
        """Test KDJ batch processing"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Test batch with parameter ranges
        result = ta_indicators.kdj_batch(
            high, low, close,
            fast_k_period_range=(5, 15, 5),  # 5, 10, 15
            slow_k_period_range=(3, 3, 0),   # just 3
            slow_k_ma_type="sma",
            slow_d_period_range=(3, 3, 0),   # just 3
            slow_d_ma_type="sma"
        )
        
        # Check result structure
        assert 'k_values' in result
        assert 'd_values' in result
        assert 'j_values' in result
        assert 'fast_k_periods' in result
        assert 'slow_k_periods' in result
        assert 'slow_d_periods' in result
        
        # Should have 3 parameter combinations (fast_k_period = 5, 10, 15)
        assert result['k_values'].shape[0] == 3
        assert result['d_values'].shape[0] == 3
        assert result['j_values'].shape[0] == 3
        
        # Each row should have same length as input
        assert result['k_values'].shape[1] == len(close)
        assert result['d_values'].shape[1] == len(close)
        assert result['j_values'].shape[1] == len(close)
        
        # Check parameter arrays
        assert np.array_equal(result['fast_k_periods'], [5, 10, 15])
        assert np.array_equal(result['slow_k_periods'], [3, 3, 3])
        assert np.array_equal(result['slow_d_periods'], [3, 3, 3])
    
    def test_kdj_kernel_parameter(self, test_data):
        """Test KDJ with kernel parameter"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Test with different kernels
        for kernel in [None, "scalar"]:
            k, d, j = ta_indicators.kdj(
                high, low, close,
                kernel=kernel
            )
            assert len(k) == len(close)
            assert len(d) == len(close)
            assert len(j) == len(close)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])