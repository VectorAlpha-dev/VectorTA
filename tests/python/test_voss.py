"""
Python binding tests for VOSS indicator.
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
from rust_comparison import compare_with_rust


class TestVoss:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_voss_partial_params(self, test_data):
        """Test VOSS with partial parameters (None values) - mirrors check_voss_partial_params"""
        close = test_data['close']
        
        # Test with some default params (None)
        voss_result, filt_result = ta_indicators.voss(close, period=None, predict=2, bandwidth=None)
        assert len(voss_result) == len(close)
        assert len(filt_result) == len(close)
    
    def test_voss_accuracy(self, test_data):
        """Test VOSS matches expected values from Rust tests - mirrors check_voss_accuracy"""
        close = test_data['close']
        
        voss_result, filt_result = ta_indicators.voss(
            close,
            period=20,
            predict=3,
            bandwidth=0.25
        )
        
        assert len(voss_result) == len(close)
        assert len(filt_result) == len(close)
        
        # Expected values from Rust tests
        expected_voss_last_five = [
            -290.430249544605,
            -269.74949153549596,
            -241.08179139844515,
            -149.2113276943419,
            -138.60361772412466,
        ]
        expected_filt_last_five = [
            -228.0283989610523,
            -257.79056527053103,
            -270.3220395771822,
            -257.4282859799144,
            -235.78021136041997,
        ]
        
        # Check last 5 values match expected
        assert_close(
            voss_result[-5:], 
            expected_voss_last_five,
            rtol=1e-1,  # Using 1e-1 like in Rust tests
            msg="VOSS last 5 values mismatch"
        )
        
        assert_close(
            filt_result[-5:], 
            expected_filt_last_five,
            rtol=1e-1,  # Using 1e-1 like in Rust tests
            msg="Filt last 5 values mismatch"
        )
    
    def test_voss_default_candles(self, test_data):
        """Test VOSS with default parameters - mirrors check_voss_default_candles"""
        close = test_data['close']
        
        # Default params: period=20, predict=3, bandwidth=0.25
        voss_result, filt_result = ta_indicators.voss(close)
        assert len(voss_result) == len(close)
        assert len(filt_result) == len(close)
    
    def test_voss_zero_period(self):
        """Test VOSS fails with zero period - mirrors check_voss_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.voss(input_data, period=0, predict=3, bandwidth=0.25)
    
    def test_voss_period_exceeds_length(self):
        """Test VOSS fails when period exceeds data length - mirrors check_voss_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.voss(data_small, period=10, predict=3, bandwidth=0.25)
    
    def test_voss_very_small_dataset(self):
        """Test VOSS fails with insufficient data - mirrors check_voss_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.voss(single_point, period=20, predict=3, bandwidth=0.25)
    
    def test_voss_empty_input(self):
        """Test VOSS fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.voss(empty, period=20, predict=3, bandwidth=0.25)
    
    def test_voss_reinput(self, test_data):
        """Test VOSS applied twice (re-input) - mirrors check_voss_reinput"""
        close = test_data['close']
        
        # First pass
        first_voss, first_filt = ta_indicators.voss(
            close, period=10, predict=2, bandwidth=0.2
        )
        assert len(first_voss) == len(close)
        assert len(first_filt) == len(close)
        
        # Second pass - apply VOSS to VOSS output
        second_voss, second_filt = ta_indicators.voss(
            first_voss, period=10, predict=2, bandwidth=0.2
        )
        assert len(second_voss) == len(first_voss)
        assert len(second_filt) == len(first_voss)
    
    def test_voss_all_nan_input(self):
        """Test VOSS with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.voss(all_nan, period=20, predict=3, bandwidth=0.25)
    
    def test_voss_kernel_parameter(self, test_data):
        """Test VOSS with kernel parameter"""
        close = test_data['close']
        
        # Test with scalar kernel
        voss_scalar, filt_scalar = ta_indicators.voss(
            close, period=20, predict=3, bandwidth=0.25, kernel="scalar"
        )
        
        # Test with auto kernel (default)
        voss_auto, filt_auto = ta_indicators.voss(
            close, period=20, predict=3, bandwidth=0.25
        )
        
        # Results should be very close regardless of kernel
        assert_close(voss_scalar, voss_auto, rtol=1e-10)
        assert_close(filt_scalar, filt_auto, rtol=1e-10)
    
    def test_voss_stream(self):
        """Test VOSS streaming functionality"""
        # Create stream with default parameters
        stream = ta_indicators.VossStream()
        
        # Test with custom parameters
        stream_custom = ta_indicators.VossStream(period=10, predict=2, bandwidth=0.2)
        
        # Feed some values
        test_values = [50.0, 51.0, 52.0, 51.5, 50.5, 49.5, 50.0, 51.0, 52.5, 53.0]
        
        results = []
        for val in test_values * 5:  # Repeat to ensure we get past warmup
            result = stream_custom.update(val)
            if result is not None:
                results.append(result)
        
        # Should eventually start producing results
        assert len(results) > 0
        # Each result should be a tuple of (voss, filt)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    
    def test_voss_batch_single_parameter_set(self, test_data):
        """Test batch with single parameter combination"""
        close = test_data['close']
        
        # Single parameter set
        result = ta_indicators.voss_batch(
            close,
            period_range=(20, 20, 0),
            predict_range=(3, 3, 0),
            bandwidth_range=(0.25, 0.25, 0.0)
        )
        
        # Should match single calculation
        single_voss, single_filt = ta_indicators.voss(close, period=20, predict=3, bandwidth=0.25)
        
        assert 'voss' in result
        assert 'filt' in result
        assert 'periods' in result
        assert 'predicts' in result
        assert 'bandwidths' in result
        
        # Check shapes
        assert result['voss'].shape == (1, len(close))
        assert result['filt'].shape == (1, len(close))
        
        # Check values match single calculation
        assert_close(result['voss'][0], single_voss, rtol=1e-10)
        assert_close(result['filt'][0], single_filt, rtol=1e-10)
    
    def test_voss_batch_multiple_periods(self, test_data):
        """Test batch with multiple period values"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        # Multiple periods: 10, 12, 14
        result = ta_indicators.voss_batch(
            close,
            period_range=(10, 14, 2),
            predict_range=(3, 3, 0),
            bandwidth_range=(0.25, 0.25, 0.0)
        )
        
        # Should have 3 rows
        assert result['voss'].shape == (3, 100)
        assert result['filt'].shape == (3, 100)
        assert len(result['periods']) == 3
        assert len(result['predicts']) == 3
        assert len(result['bandwidths']) == 3
        
        # Verify each row matches individual calculation
        periods = [10, 12, 14]
        for i, period in enumerate(periods):
            single_voss, single_filt = ta_indicators.voss(close, period=period, predict=3, bandwidth=0.25)
            assert_close(result['voss'][i], single_voss, rtol=1e-10)
            assert_close(result['filt'][i], single_filt, rtol=1e-10)
    
    def test_voss_batch_full_parameter_sweep(self, test_data):
        """Test full parameter sweep"""
        close = test_data['close'][:50]  # Small dataset for speed
        
        result = ta_indicators.voss_batch(
            close,
            period_range=(10, 12, 2),      # 2 periods
            predict_range=(2, 3, 1),        # 2 predicts
            bandwidth_range=(0.2, 0.3, 0.1) # 2 bandwidths
        )
        
        # Should have 2 * 2 * 2 = 8 combinations
        assert result['voss'].shape == (8, 50)
        assert result['filt'].shape == (8, 50)
        assert len(result['periods']) == 8
        assert len(result['predicts']) == 8
        assert len(result['bandwidths']) == 8
        
        # Verify parameter combinations
        expected_combos = [
            (10, 2, 0.2), (10, 2, 0.3),
            (10, 3, 0.2), (10, 3, 0.3),
            (12, 2, 0.2), (12, 2, 0.3),
            (12, 3, 0.2), (12, 3, 0.3),
        ]
        
        for i, (period, predict, bandwidth) in enumerate(expected_combos):
            assert result['periods'][i] == period
            assert result['predicts'][i] == predict
            assert abs(result['bandwidths'][i] - bandwidth) < 1e-10
    
    def test_voss_batch_edge_cases(self, test_data):
        """Test edge cases for batch processing"""
        close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
        
        # Step larger than range
        result = ta_indicators.voss_batch(
            close,
            period_range=(5, 7, 10),  # Step larger than range
            predict_range=(2, 2, 0),
            bandwidth_range=(0.25, 0.25, 0.0)
        )
        
        # Should only have period=5
        assert result['voss'].shape == (1, 10)
        assert result['periods'][0] == 5
        
        # Empty data should throw
        with pytest.raises(ValueError):
            ta_indicators.voss_batch(
                np.array([]),
                period_range=(20, 20, 0),
                predict_range=(3, 3, 0),
                bandwidth_range=(0.25, 0.25, 0.0)
            )