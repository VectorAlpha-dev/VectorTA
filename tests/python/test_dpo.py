"""
Python binding tests for DPO (Detrended Price Oscillator) indicator.
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


class TestDpo:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_dpo_partial_params(self, test_data):
        """Test DPO with partial parameters - mirrors check_dpo_partial_params"""
        close = test_data['close']
        
        # Test with default period (5)
        result = ta_indicators.dpo(close, 5)
        assert len(result) == len(close)
    
    def test_dpo_accuracy(self, test_data):
        """Test DPO matches expected values from Rust tests - mirrors check_dpo_accuracy"""
        close = test_data['close']
        
        result = ta_indicators.dpo(close, period=5)
        assert len(result) == len(close)
        
        # Check last 5 values match expected from Rust tests
        expected_last_five = [
            65.3999999999287,
            131.3999999999287,
            32.599999999925785,
            98.3999999999287,
            117.99999999992724,
        ]
        
        # Match Rust test tolerance: absolute <= 1e-1 (no relaxed relative error)
        assert_close(
            result[-5:],
            expected_last_five,
            rtol=0.0,
            atol=1e-1,
            msg="DPO last 5 values mismatch"
        )
        
        # Compare full output with Rust.
        # DPO can produce values extremely close to 0 where tiny FMA/rounding
        # differences between build artifacts show up at ~1e-11. Keep this
        # strict but avoid flaking on sub-ulp noise.
        compare_with_rust('dpo', result, 'close', {'period': 5}, rtol=1e-10, atol=2e-11)
    
    def test_dpo_default_candles(self, test_data):
        """Test DPO with default parameters - mirrors check_dpo_default_candles"""
        close = test_data['close']
        
        # Default period is 5
        result = ta_indicators.dpo(close, 5)
        assert len(result) == len(close)
    
    def test_dpo_zero_period(self):
        """Test DPO fails with zero period - mirrors check_dpo_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.dpo(input_data, period=0)
    
    def test_dpo_period_exceeds_length(self):
        """Test DPO fails when period exceeds data length - mirrors check_dpo_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.dpo(data_small, period=10)
    
    def test_dpo_very_small_dataset(self):
        """Test DPO fails with insufficient data - mirrors check_dpo_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.dpo(single_point, period=5)
    
    def test_dpo_reinput(self, test_data):
        """Test DPO applied twice (re-input) - mirrors check_dpo_reinput"""
        close = test_data['close']
        
        # First pass
        first_result = ta_indicators.dpo(close, period=5)
        assert len(first_result) == len(close)
        
        # Second pass - apply DPO to DPO output
        second_result = ta_indicators.dpo(first_result, period=5)
        assert len(second_result) == len(first_result)
    
    def test_dpo_nan_handling(self, test_data):
        """Test DPO handles NaN values correctly - mirrors check_dpo_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.dpo(close, period=5)
        assert len(result) == len(close)
        
        # After warmup period, no NaN values should exist
        # Based on Rust test which checks after index 20
        if len(result) > 20:
            for i in range(20, len(result)):
                assert not np.isnan(result[i]), f"Found unexpected NaN at index {i}"
    
    def test_dpo_batch_single_parameter(self, test_data):
        """Test batch with single parameter combination"""
        close = test_data['close']
        
        batch_result = ta_indicators.dpo_batch(
            close,
            period_range=(5, 5, 0)  # Single period
        )
        
        # Should match single calculation
        single_result = ta_indicators.dpo(close, 5)
        
        assert batch_result['values'].shape == (1, len(close))
        # Allow tiny FP noise between batch and single paths
        assert_close(
            batch_result['values'][0],
            single_result,
            rtol=5e-8,
            atol=1e-10,
            msg="Batch vs single mismatch"
        )
    
    def test_dpo_batch_multiple_periods(self, test_data):
        """Test batch with multiple period values"""
        close = test_data['close'][:200]  # Use smaller dataset for speed
        
        # Multiple periods: 5, 10, 15
        batch_result = ta_indicators.dpo_batch(
            close,
            period_range=(5, 15, 5)
        )
        
        # Should have 3 rows * 200 cols
        assert batch_result['values'].shape == (3, 200)
        assert len(batch_result['periods']) == 3
        
        # Verify each row matches individual calculation
        periods = [5, 10, 15]
        for i, period in enumerate(periods):
            single_result = ta_indicators.dpo(close, period)
            assert_close(
                batch_result['values'][i],
                single_result,
                rtol=5e-8,
                atol=1e-10,
                msg=f"Batch row {i} (period={period}) mismatch"
            )
    
    def test_dpo_batch_parameter_sweep(self, test_data):
        """Test batch with parameter sweep"""
        close = test_data['close'][:50]
        
        batch_result = ta_indicators.dpo_batch(
            close,
            period_range=(5, 20, 5)  # 5, 10, 15, 20
        )
        
        # Should have 4 rows * 50 cols
        assert batch_result['values'].shape == (4, 50)
        assert len(batch_result['periods']) == 4
        assert batch_result['periods'].tolist() == [5, 10, 15, 20]
    
    def test_dpo_stream(self):
        """Test DPO streaming functionality"""
        stream = ta_indicators.DpoStream(period=5)
        
        # Feed data points one by one
        test_data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        results = []
        
        for value in test_data:
            result = stream.update(value)
            results.append(result)
        
        # First period-1 values should be None (warmup)
        for i in range(4):
            assert results[i] is None, f"Expected None at index {i}, got {results[i]}"
        
        # After warmup, should have values
        for i in range(4, len(results)):
            assert results[i] is not None, f"Expected value at index {i}, got None"
    
    def test_dpo_kernel_parameter(self, test_data):
        """Test DPO with optional kernel parameter"""
        close = test_data['close']
        
        # Test with auto kernel (default)
        result_auto = ta_indicators.dpo(close, 5)
        
        # Test with scalar kernel
        result_scalar = ta_indicators.dpo(close, 5, kernel="scalar")
        
        # Results should be very close
        assert_close(
            result_auto,
            result_scalar,
            rtol=1e-10,
            msg="Auto vs scalar kernel results differ"
        )
        
        # Test invalid kernel
        with pytest.raises(ValueError):
            ta_indicators.dpo(close, 5, kernel="invalid_kernel")
    
    def test_dpo_all_nan_input(self):
        """Test DPO with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.dpo(all_nan, 5)
    
    def test_dpo_empty_input(self):
        """Test DPO with empty input"""
        empty = np.array([])
        
        # Should fail with empty input data error
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.dpo(empty, 5)


if __name__ == "__main__":
    pytest.main([__file__])
