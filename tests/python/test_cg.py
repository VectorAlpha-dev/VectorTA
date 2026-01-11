"""
Python binding tests for CG (Center of Gravity) indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.

Warmup Period: CG requires period + 1 valid data points.
Output starts at index: first_valid + period
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


class TestCg:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_cg_partial_params(self, test_data):
        """Test CG with partial parameters - mirrors check_cg_partial_params"""
        close = test_data['close']
        
        
        result = ta_indicators.cg(close, period=12)
        assert len(result) == len(close)
    
    def test_cg_accuracy(self, test_data):
        """Test CG matches expected values from Rust tests - mirrors check_cg_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['cg']
        
        result = ta_indicators.cg(
            close,
            period=expected['default_params']['period']
        )
        
        assert len(result) == len(close)
        
        
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-4,  
            msg="CG last 5 values mismatch"
        )
        
        
        compare_with_rust('cg', result, 'close', expected['default_params'])
    
    def test_cg_default_candles(self, test_data):
        """Test CG with default parameters - mirrors check_cg_default_candles"""
        close = test_data['close']
        
        
        result = ta_indicators.cg(close, 10)
        assert len(result) == len(close)
    
    def test_cg_zero_period(self):
        """Test CG fails with zero period - mirrors check_cg_zero_period"""
        data = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.cg(data, period=0)
    
    def test_cg_period_exceeds_length(self):
        """Test CG fails when period exceeds data length - mirrors check_cg_period_exceeds_length"""
        data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.cg(data, period=10)
    
    def test_cg_very_small_dataset(self):
        """Test CG fails with insufficient data - mirrors check_cg_very_small_dataset"""
        data = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.cg(data, period=10)
    
    def test_cg_empty_input(self):
        """Test CG fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.cg(empty, period=10)
    
    def test_cg_nan_handling(self, test_data):
        """Test CG handles NaN values correctly - mirrors check_cg_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.cg(close, period=10)
        assert len(result) == len(close)
        
        
        check_idx = 240
        if len(result) > check_idx:
            
            found_valid = False
            for i in range(check_idx, len(result)):
                if not np.isnan(result[i]):
                    found_valid = True
                    break
            assert found_valid, f"All CG values from index {check_idx} onward are NaN."
        
        
        
        assert np.all(np.isnan(result[:10])), "Expected NaN in warmup period"
    
    def test_cg_streaming(self, test_data):
        """Test CG streaming matches batch calculation - mirrors check_cg_streaming"""
        close = test_data['close']
        period = 10
        
        
        batch_result = ta_indicators.cg(close, period=period)
        
        
        stream = ta_indicators.CgStream(period=period)
        stream_values = []
        
        for price in close:
            result = stream.update(price)
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        
        assert len(batch_result) == len(stream_values)
        
        
        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, atol=1e-9, 
                        msg=f"CG streaming mismatch at index {i}")
    
    def test_cg_batch(self, test_data):
        """Test CG batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        result = ta_indicators.cg_batch(
            close,
            period_range=(10, 10, 0),  
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        
        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['cg']['last_5_values']
        
        
        assert_close(
            default_row[-5:],
            expected,
            rtol=1e-4,  
            msg="CG batch default row mismatch"
        )
    
    def test_cg_batch_multiple_periods(self, test_data):
        """Test CG batch with multiple period values"""
        close = test_data['close'][:100]  
        
        
        result = ta_indicators.cg_batch(
            close,
            period_range=(10, 14, 2),
        )
        
        
        assert result['values'].shape[0] == 3
        assert result['values'].shape[1] == 100
        assert len(result['periods']) == 3
        assert list(result['periods']) == [10, 12, 14]
        
        
        single_result = ta_indicators.cg(close, period=10)
        assert_close(
            result['values'][0],
            single_result,
            rtol=1e-10,
            msg="Batch first row mismatch"
        )
    
    def test_cg_all_nan_input(self):
        """Test CG with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.cg(all_nan, period=10)
    
    def test_cg_kernel_parameter(self, test_data):
        """Test CG with different kernel parameters"""
        close = test_data['close'][:100]  
        
        
        result_scalar = ta_indicators.cg(close, period=10, kernel='scalar')
        assert len(result_scalar) == len(close)
        
        
        result_auto = ta_indicators.cg(close, period=10, kernel='auto')
        assert len(result_auto) == len(close)
        
        
        
        assert_close(
            result_scalar,
            result_auto,
            rtol=1e-10,
            msg="Kernel results mismatch"
        )
    
    def test_cg_warmup_period(self, test_data):
        """Test CG respects warmup period requirement"""
        close = test_data['close']
        period = 10
        
        result = ta_indicators.cg(close, period=period)
        
        
        
        assert np.all(np.isnan(result[:period])), f"Expected NaN in first {period} values"
        
        
        if len(close) > period:
            assert not np.isnan(result[period]), f"Expected valid value at index {period}"
    
    def test_cg_edge_case_small_period(self, test_data):
        """Test CG with very small period"""
        close = test_data['close'][:20]
        
        
        result = ta_indicators.cg(close, period=2)
        assert len(result) == len(close)
        
        
        assert np.all(np.isnan(result[:2]))
        
        assert not np.isnan(result[2])
    
    def test_cg_batch_empty_range(self, test_data):
        """Test CG batch with single value range"""
        close = test_data['close'][:50]
        
        
        result = ta_indicators.cg_batch(
            close,
            period_range=(12, 12, 0),
        )
        
        assert result['values'].shape[0] == 1
        assert result['periods'][0] == 12
    
    def test_cg_batch_kernel_parameter(self, test_data):
        """Test CG batch with kernel parameter"""
        close = test_data['close'][:50]
        
        
        result = ta_indicators.cg_batch(
            close,
            period_range=(10, 12, 2),
            kernel='scalar'
        )
        
        assert result['values'].shape[0] == 2
        assert list(result['periods']) == [10, 12]
    
    def test_cg_nan_injection(self, test_data):
        """Test CG handles NaN values injected at specific positions"""
        close = test_data['close'][:100].copy()
        
        
        close[20:25] = np.nan
        
        
        result = ta_indicators.cg(close, period=10)
        assert len(result) == len(close)
        
        
        
        
        assert np.all(np.isnan(result[:10]))
        
        
        
        has_non_nan_after_injection = False
        for i in range(30, len(result)):
            if not np.isnan(result[i]) and result[i] != 0.0:
                has_non_nan_after_injection = True
                break
        assert has_non_nan_after_injection, "Expected non-zero values after NaN injection"
    
    def test_cg_batch_accuracy(self, test_data):
        """Test CG batch matches expected accuracy for default params"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['cg']
        
        
        result = ta_indicators.cg_batch(
            close,
            period_range=(10, 10, 0),
        )
        
        
        assert_close(
            result['values'][0][-5:],
            expected['last_5_values'],
            rtol=1e-4,
            msg="CG batch accuracy mismatch"
        )
    
    def test_cg_batch_parameter_sweep(self, test_data):
        """Test comprehensive batch parameter sweep like ALMA"""
        close = test_data['close']
        
        
        result = ta_indicators.cg_batch(
            close,
            period_range=(5, 20, 5),  
        )
        
        
        assert result['values'].shape[0] == 4
        assert result['values'].shape[1] == len(close)
        assert list(result['periods']) == [5, 10, 15, 20]
        
        
        for i, period in enumerate(result['periods']):
            row = result['values'][i]
            
            assert np.all(np.isnan(row[:period]))
            
            if len(close) > period:
                assert not np.isnan(row[period])
    
    def test_cg_numerical_stability(self):
        """Test CG with extreme values for numerical stability"""
        
        large_data = np.array([1e15, 1e15, 1e15, 1e15, 1e15, 1e15])
        result_large = ta_indicators.cg(large_data, period=2)
        assert len(result_large) == len(large_data)
        
        assert not np.isnan(result_large[2])  
        
        
        small_data = np.array([1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15])
        result_small = ta_indicators.cg(small_data, period=2)
        assert len(result_small) == len(small_data)
        assert not np.isnan(result_small[2])  
    
    def test_cg_full_dataset(self, test_data):
        """Test CG with full dataset instead of slices"""
        close = test_data['close']  
        
        result = ta_indicators.cg(close, period=10)
        assert len(result) == len(close)
        
        
        assert np.all(np.isnan(result[:10]))
        
        
        assert not np.any(np.isnan(result[10:]))
    
    @pytest.mark.skipif(sys.platform != "win32", reason="Poison detection only in debug builds")
    def test_cg_poison_detection(self, test_data):
        """Test for uninitialized memory patterns (debug builds only)"""
        close = test_data['close'][:100]
        
        
        single_result = ta_indicators.cg(close, period=10)
        batch_result = ta_indicators.cg_batch(
            close,
            period_range=(5, 15, 5),
        )
        
        
        for i, val in enumerate(single_result):
            if not np.isnan(val):
                
                val_bits = np.float64(val).view(np.uint64)
                assert val_bits != 0x1111111111111111, f"Found poison value at index {i}"
                assert val_bits != 0x2222222222222222, f"Found poison value at index {i}"
                assert val_bits != 0x3333333333333333, f"Found poison value at index {i}"
        
        
        for row_idx in range(batch_result['values'].shape[0]):
            for col_idx in range(batch_result['values'].shape[1]):
                val = batch_result['values'][row_idx, col_idx]
                if not np.isnan(val):
                    val_bits = np.float64(val).view(np.uint64)
                    assert val_bits != 0x1111111111111111, f"Found poison at [{row_idx},{col_idx}]"
                    assert val_bits != 0x2222222222222222, f"Found poison at [{row_idx},{col_idx}]"
                    assert val_bits != 0x3333333333333333, f"Found poison at [{row_idx},{col_idx}]"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])