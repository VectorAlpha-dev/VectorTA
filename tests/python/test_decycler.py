"""
Python binding tests for Decycler indicator.
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


class TestDecycler:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_decycler_partial_params(self, test_data):
        """Test Decycler with partial parameters (None values) - mirrors check_decycler_partial_params"""
        close = test_data['close']
        
        # Test with all default params (None)
        result = ta_indicators.decycler(close)  # Using defaults
        assert len(result) == len(close)
        
        # Test with partial params
        result = ta_indicators.decycler(close, hp_period=50)
        assert len(result) == len(close)
        
        result = ta_indicators.decycler(close, hp_period=30, k=None)
        assert len(result) == len(close)
    
    def test_decycler_accuracy(self, test_data):
        """Test Decycler matches expected values from Rust tests - mirrors check_decycler_accuracy"""
        close = test_data['close']
        
        result = ta_indicators.decycler(
            close,
            hp_period=125,
            k=None  # Default k=0.707
        )
        
        assert len(result) == len(close)
        
        # Expected values from Rust tests
        expected_last_5 = [
            60289.96384058519,
            60204.010366691065,
            60114.255563805666,
            60028.535266555904,
            59934.26876964316
        ]
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected_last_5,
            rtol=1e-6,
            msg="Decycler last 5 values mismatch"
        )
    
    def test_decycler_default_params(self, test_data):
        """Test Decycler with default parameters"""
        close = test_data['close']
        
        # Default params: hp_period=125, k=0.707
        result = ta_indicators.decycler(close)
        assert len(result) == len(close)
    
    def test_decycler_zero_period(self):
        """Test Decycler fails with zero period - mirrors check_decycler_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.decycler(input_data, hp_period=0)
    
    def test_decycler_period_exceeds_length(self):
        """Test Decycler fails when period exceeds data length - mirrors check_decycler_period_exceed_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.decycler(data_small, hp_period=10)
    
    def test_decycler_very_small_dataset(self):
        """Test Decycler fails with insufficient data - mirrors check_decycler_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.decycler(single_point, hp_period=2)
    
    def test_decycler_empty_input(self):
        """Test Decycler fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.decycler(empty)
    
    def test_decycler_invalid_k(self):
        """Test Decycler fails with invalid k"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        with pytest.raises(ValueError, match="Invalid k"):
            ta_indicators.decycler(data, hp_period=2, k=0.0)
        
        with pytest.raises(ValueError, match="Invalid k"):
            ta_indicators.decycler(data, hp_period=2, k=-1.0)
        
        with pytest.raises(ValueError, match="Invalid k"):
            ta_indicators.decycler(data, hp_period=2, k=float('nan'))
    
    def test_decycler_reinput(self, test_data):
        """Test Decycler applied twice (re-input) - mirrors check_decycler_reinput"""
        close = test_data['close']
        
        # First pass
        first_result = ta_indicators.decycler(close, hp_period=30)
        assert len(first_result) == len(close)
        
        # Second pass - apply Decycler to Decycler output
        second_result = ta_indicators.decycler(first_result, hp_period=30)
        assert len(second_result) == len(first_result)
    
    def test_decycler_nan_handling(self, test_data):
        """Test Decycler handles NaN values correctly - mirrors check_decycler_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.decycler(close, hp_period=125)
        assert len(result) == len(close)
        
        # After warmup period (240), no NaN values should exist
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"
    
    def test_decycler_streaming(self, test_data):
        """Test Decycler streaming matches batch calculation"""
        close = test_data['close']
        hp_period = 125
        k = 0.707
        
        # Batch calculation
        batch_result = ta_indicators.decycler(close, hp_period=hp_period, k=k)
        
        # Streaming calculation
        stream = ta_indicators.DecyclerStream(hp_period=hp_period, k=k)
        stream_values = []
        
        for price in close:
            result = stream.update(price)
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        # Compare batch vs streaming
        assert len(batch_result) == len(stream_values)
        
        # Compare values where both are not NaN
        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, atol=1e-9, 
                        msg=f"Decycler streaming mismatch at index {i}")
    
    def test_decycler_batch(self, test_data):
        """Test Decycler batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        result = ta_indicators.decycler_batch(
            close,
            hp_period_range=(125, 125, 0),  # Default period only
            k_range=(0.707, 0.707, 0.0)  # Default k only
        )
        
        assert 'values' in result
        assert 'params' in result
        assert 'rows' in result
        assert 'cols' in result
        
        # Should have 1 combination (default params)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        # Extract the single row
        default_row = result['values'][0]
        expected_last_5 = [
            60289.96384058519,
            60204.010366691065,
            60114.255563805666,
            60028.535266555904,
            59934.26876964316
        ]
        
        # Check last 5 values match
        assert_close(
            default_row[-5:],
            expected_last_5,
            rtol=1e-6,
            msg="Decycler batch default row mismatch"
        )
    
    def test_decycler_batch_multiple_params(self, test_data):
        """Test Decycler batch with multiple parameter combinations"""
        close = test_data['close']
        
        result = ta_indicators.decycler_batch(
            close,
            hp_period_range=(100, 150, 25),  # 100, 125, 150
            k_range=(0.5, 0.7, 0.1)  # 0.5, 0.6, 0.7
        )
        
        # Should have 3 * 3 = 9 combinations
        assert result['values'].shape[0] == 9
        assert result['values'].shape[1] == len(close)
        assert len(result['params']) == 9
    
    def test_decycler_all_nan_input(self):
        """Test Decycler with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.decycler(all_nan)
    
    def test_decycler_kernel_parameter(self, test_data):
        """Test Decycler with different kernel parameters"""
        close = test_data['close']
        
        # Test with scalar kernel
        result_scalar = ta_indicators.decycler(close, kernel="scalar")
        assert len(result_scalar) == len(close)
        
        # Test with auto kernel (default)
        result_auto = ta_indicators.decycler(close, kernel=None)
        assert len(result_auto) == len(close)
        
        # Test with invalid kernel
        with pytest.raises(ValueError, match="Invalid kernel"):
            ta_indicators.decycler(close, kernel="invalid")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])