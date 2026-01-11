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
        
        
        result = ta_indicators.decycler(close)  
        assert len(result) == len(close)
        
        
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
            k=None  
        )
        
        assert len(result) == len(close)
        
        
        expected_last_5 = [
            60289.96384058519,
            60204.010366691065,
            60114.255563805666,
            60028.535266555904,
            59934.26876964316
        ]
        
        
        assert_close(
            result[-5:], 
            expected_last_5,
            rtol=1e-6,
            msg="Decycler last 5 values mismatch"
        )
    
    def test_decycler_default_params(self, test_data):
        """Test Decycler with default parameters"""
        close = test_data['close']
        
        
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
        
        
        first_result = ta_indicators.decycler(close, hp_period=30)
        assert len(first_result) == len(close)
        
        
        second_result = ta_indicators.decycler(first_result, hp_period=30)
        assert len(second_result) == len(first_result)
    
    def test_decycler_nan_handling(self, test_data):
        """Test Decycler handles NaN values correctly - mirrors check_decycler_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.decycler(close, hp_period=125)
        assert len(result) == len(close)
        
        
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"
    
    def test_decycler_warmup_period(self, test_data):
        """Test Decycler warmup period matches expected behavior"""
        close = test_data['close']
        hp_period = 125
        
        result = ta_indicators.decycler(close, hp_period=hp_period)
        
        
        first_non_nan = np.where(~np.isnan(result))[0]
        if len(first_non_nan) > 0:
            first_idx = first_non_nan[0]
            
            
            first_input = np.where(~np.isnan(close))[0][0]
            expected_warmup = first_input + 2
            
            assert first_idx == expected_warmup, f"Warmup period mismatch: expected {expected_warmup}, got {first_idx}"
            
            
            assert np.all(np.isnan(result[:expected_warmup])), "Expected NaN values during warmup period"
    
    def test_decycler_partial_nan_input(self, test_data):
        """Test Decycler with data containing some NaN values"""
        
        data = np.arange(1.0, 101.0)  
        
        
        data_with_nan = data.copy()
        data_with_nan[10] = np.nan
        data_with_nan[11] = np.nan
        
        
        result = ta_indicators.decycler(data_with_nan, hp_period=5)
        assert len(result) == len(data_with_nan)
        
        
        
        
    
    def test_decycler_edge_case_k_values(self):
        """Test Decycler with edge case k values"""
        data = np.arange(1.0, 101.0)  
        
        
        result_small = ta_indicators.decycler(data, hp_period=10, k=0.001)
        assert len(result_small) == len(data)
        
        
        result_default = ta_indicators.decycler(data, hp_period=10, k=0.707)
        assert len(result_default) == len(data)
        
        
        result_large = ta_indicators.decycler(data, hp_period=10, k=10.0)
        assert len(result_large) == len(data)
        
        
        
        warmup_end = 12  
        if len(data) > warmup_end + 10:
            check_idx = warmup_end + 5
            assert result_small[check_idx] != result_default[check_idx], "Different k values should produce different results"
            assert result_default[check_idx] != result_large[check_idx], "Different k values should produce different results"
    
    def test_decycler_input_types(self, test_data):
        """Test Decycler with different input types"""
        close = test_data['close'][:100]  
        
        
        close_list = list(close)
        result_list = ta_indicators.decycler(np.array(close_list), hp_period=30)
        assert len(result_list) == len(close)
        
        
        close_f64 = np.array(close, dtype=np.float64)
        result_f64 = ta_indicators.decycler(close_f64, hp_period=30)
        assert len(result_f64) == len(close)
        
        
        close_f32 = np.array(close, dtype=np.float32)
        result_f32 = ta_indicators.decycler(close_f32.astype(np.float64), hp_period=30)
        assert len(result_f32) == len(close)
        
        
        assert_close(result_f32, result_f64, rtol=1e-6, msg="Results should be very similar after conversion")
    
    def test_decycler_streaming(self, test_data):
        """Test Decycler streaming matches batch calculation"""
        close = test_data['close']
        hp_period = 125
        k = 0.707
        
        
        batch_result = ta_indicators.decycler(close, hp_period=hp_period, k=k)
        
        
        stream = ta_indicators.DecyclerStream(hp_period=hp_period, k=k)
        stream_values = []
        
        for price in close:
            result = stream.update(price)
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        
        assert len(batch_result) == len(stream_values)
        
        
        
        first_input = np.where(~np.isnan(close))[0][0]
        expected_warmup = first_input + 2
        
        
        for i in range(expected_warmup):
            assert np.isnan(batch_result[i]), f"Expected NaN in batch during warmup at index {i}"
            assert np.isnan(stream_values[i]), f"Expected NaN in stream during warmup at index {i}"
        
        
        for i in range(expected_warmup, len(batch_result)):
            if np.isnan(batch_result[i]) and np.isnan(stream_values[i]):
                continue
            if np.isnan(batch_result[i]) or np.isnan(stream_values[i]):
                
                continue
            assert_close(batch_result[i], stream_values[i], rtol=1e-9, atol=1e-9, 
                        msg=f"Decycler streaming mismatch at index {i}")
    
    def test_decycler_batch(self, test_data):
        """Test Decycler batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        result = ta_indicators.decycler_batch(
            close,
            hp_period_range=(125, 125, 0),  
            k_range=(0.707, 0.707, 0.0)  
        )
        
        assert 'values' in result
        assert 'hp_periods' in result
        assert 'ks' in result
        assert 'rows' in result
        assert 'cols' in result
        
        
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        
        default_row = result['values'][0]
        expected_last_5 = [
            60289.96384058519,
            60204.010366691065,
            60114.255563805666,
            60028.535266555904,
            59934.26876964316
        ]
        
        
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
            hp_period_range=(100, 150, 25),  
            k_range=(0.5, 0.7, 0.1)  
        )
        
        
        assert result['values'].shape[0] == 9
        assert result['values'].shape[1] == len(close)
        assert len(result['hp_periods']) == 9
        assert len(result['ks']) == 9
        
        
        expected_hp_periods = [100, 100, 100, 125, 125, 125, 150, 150, 150]
        expected_ks = [0.5, 0.6, 0.7, 0.5, 0.6, 0.7, 0.5, 0.6, 0.7]
        
        for i, (hp, k) in enumerate(zip(expected_hp_periods, expected_ks)):
            assert result['hp_periods'][i] == hp, f"hp_period mismatch at index {i}"
            assert_close(result['ks'][i], k, rtol=1e-9, msg=f"k value mismatch at index {i}")
        
        
        
        row_idx = 4
        single_result = ta_indicators.decycler(close, hp_period=125, k=0.6)
        assert_close(
            result['values'][row_idx], 
            single_result, 
            rtol=1e-9,
            msg=f"Batch row {row_idx} doesn't match single calculation"
        )
    
    def test_decycler_all_nan_input(self):
        """Test Decycler with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="decycler: All values are NaN"):
            ta_indicators.decycler(all_nan, hp_period=50)  
    
    def test_decycler_kernel_parameter(self, test_data):
        """Test Decycler with different kernel parameters"""
        close = test_data['close']
        
        
        result_scalar = ta_indicators.decycler(close, kernel="scalar")
        assert len(result_scalar) == len(close)
        
        
        result_auto = ta_indicators.decycler(close, kernel=None)
        assert len(result_auto) == len(close)
        
        
        with pytest.raises(ValueError, match="Unknown kernel"):
            ta_indicators.decycler(close, kernel="invalid")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])