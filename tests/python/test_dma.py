"""
Python binding tests for DMA (Dickson Moving Average) indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
import os
from pathlib import Path


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import my_project as ta_indicators
except ImportError:
    pytest.skip("Python module not built", allow_module_level=True)

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS


class TestDma:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_dma_partial_params(self, test_data):
        """Test DMA with default parameters - mirrors check_dma_partial_params"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['dma']
        
        
        result = ta_indicators.dma(
            close,
            hull_length=expected['default_params']['hull_length'],
            ema_length=expected['default_params']['ema_length'],
            ema_gain_limit=expected['default_params']['ema_gain_limit'],
            hull_ma_type=expected['default_params']['hull_ma_type']
        )
        assert len(result) == len(close)
        
        
        hull_len = expected['default_params']['hull_length']
        ema_len = expected['default_params']['ema_length']
        warmup = max(hull_len, ema_len) - 1
        
        assert np.all(np.isnan(result[:warmup])), f"Expected NaN during warmup period (first {warmup} values)"
        assert np.sum(~np.isnan(result)) > len(close) - warmup - 10, "Should have values after warmup"
    
    def test_dma_accuracy(self, test_data):
        """Test DMA matches expected values from Rust tests - mirrors check_dma_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['dma']
        
        result = ta_indicators.dma(
            close,
            hull_length=expected['default_params']['hull_length'],
            ema_length=expected['default_params']['ema_length'],
            ema_gain_limit=expected['default_params']['ema_gain_limit'],
            hull_ma_type=expected['default_params']['hull_ma_type']
        )
        
        assert len(result) == len(close)
        
        
        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=0.001,  
            atol=0.001,
            msg="DMA last 5 values mismatch"
        )
    
    def test_dma_default_candles(self, test_data):
        """Test DMA with default parameters - mirrors check_dma_default_candles"""
        close = test_data['close']
        
        
        result = ta_indicators.dma(close, 7, 20, 50, "WMA")
        assert len(result) == len(close)
        
        
        warmup = max(7, 20) - 1
        
        assert np.all(np.isnan(result[:warmup])), f"Expected NaN in first {warmup} values"
        assert not np.all(np.isnan(result[warmup:])), "Should have values after warmup"
    
    def test_dma_zero_hull_period(self):
        """Test DMA fails with zero hull period - mirrors check_dma_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.dma(input_data, hull_length=0, ema_length=20, ema_gain_limit=50, hull_ma_type="WMA")
    
    def test_dma_zero_ema_period(self):
        """Test DMA fails with zero EMA period - mirrors check_dma_zero_ema_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.dma(input_data, hull_length=7, ema_length=0, ema_gain_limit=50, hull_ma_type="WMA")
    
    def test_dma_period_exceeds_length(self):
        """Test DMA fails when period exceeds data length - mirrors check_dma_period_exceeds_length"""
        small_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.dma(small_data, hull_length=10, ema_length=20, ema_gain_limit=50, hull_ma_type="WMA")
    
    def test_dma_very_small_dataset(self):
        """Test DMA fails with insufficient data - mirrors check_dma_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError):
            ta_indicators.dma(single_point, hull_length=7, ema_length=20, ema_gain_limit=50, hull_ma_type="WMA")
    
    def test_dma_empty_input(self):
        """Test DMA fails with empty input - mirrors check_dma_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.dma(empty, hull_length=7, ema_length=20, ema_gain_limit=50, hull_ma_type="WMA")
    
    def test_dma_all_nan(self):
        """Test DMA fails with all NaN values - mirrors check_dma_all_nan"""
        nan_data = np.array([np.nan, np.nan, np.nan])
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.dma(nan_data, hull_length=7, ema_length=20, ema_gain_limit=50, hull_ma_type="WMA")
    
    def test_dma_invalid_hull_type(self):
        """Test DMA fails with invalid hull_ma_type - mirrors check_dma_invalid_hull_type"""
        input_data = np.array([10.0] * 50)
        
        with pytest.raises(ValueError, match="Invalid Hull MA type"):
            ta_indicators.dma(
                input_data,
                hull_length=7,
                ema_length=20,
                ema_gain_limit=50,
                hull_ma_type="INVALID"
            )
    
    def test_dma_ema_hull_type(self):
        """Test DMA works with EMA hull type - mirrors check_dma_ema_hull_type"""
        input_data = np.array([float(i) for i in range(100)])
        
        result = ta_indicators.dma(
            input_data,
            hull_length=7,
            ema_length=20,
            ema_gain_limit=50,
            hull_ma_type="EMA"
        )
        
        assert len(result) == len(input_data)
        
        assert np.sum(~np.isnan(result)) > 0, "Should produce non-NaN values after warmup"
    
    def test_dma_nan_handling(self, test_data):
        """Test DMA handles NaN values correctly - mirrors check_dma_nan_handling"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['dma']
        
        result = ta_indicators.dma(
            close,
            hull_length=expected['default_params']['hull_length'],
            ema_length=expected['default_params']['ema_length'],
            ema_gain_limit=expected['default_params']['ema_gain_limit'],
            hull_ma_type=expected['default_params']['hull_ma_type']
        )
        
        assert len(result) == len(close)
        
        
        hull_len = expected['default_params']['hull_length']
        ema_len = expected['default_params']['ema_length']
        warmup = max(hull_len, ema_len) - 1
        
        
        if len(result) > warmup + 100:
            
            assert not np.any(np.isnan(result[warmup+100:])), "Found unexpected NaN after warmup period"
        
        
        assert np.all(np.isnan(result[:warmup])), f"Expected NaN in warmup period (first {warmup} values)"
    
    
    
    
    def test_dma_batch_default_params(self, test_data):
        """Test DMA batch processing with default parameters only"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['dma']
        
        result = ta_indicators.dma_batch(
            close,
            hull_length_range=(7, 7, 0),  
            ema_length_range=expected['batch_ema_range'],      
            ema_gain_limit_range=expected['batch_gain_range'], 
            hull_ma_type="WMA"
        )
        
        assert 'values' in result
        assert 'hull_lengths' in result
        assert 'ema_lengths' in result
        assert 'ema_gain_limits' in result
        
        
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        
        default_row = result['values'][0]
        
        
        assert_close(
            default_row[-5:],
            expected['batch_default_row'],
            rtol=0.001,
            atol=0.001,
            msg="DMA batch default row mismatch"
        )
    
    def test_dma_batch_hull_sweep(self, test_data):
        """Test DMA batch processing with hull length parameter sweep"""
        close = test_data['close'][:500]  
        expected = EXPECTED_OUTPUTS['dma']
        
        result = ta_indicators.dma_batch(
            close,
            hull_length_range=expected['batch_hull_range'],  
            ema_length_range=(20, 20, 0),  
            ema_gain_limit_range=(50, 50, 0),  
            hull_ma_type="WMA"
        )
        
        
        assert result['values'].shape[0] == 4
        assert result['values'].shape[1] == len(close)
        assert len(result['hull_lengths']) == 4
        assert list(result['hull_lengths']) == expected['batch_hull_lengths']
        
        
        for i, hull_len in enumerate(result['hull_lengths']):
            row = result['values'][i]
            
            warmup = max(hull_len, 20) - 1
            
            
            assert np.all(np.isnan(row[:warmup])), f"Expected NaN in warmup for hull_length={hull_len}"
            
            
            if len(row) > warmup + 10:
                non_nan_count = np.sum(~np.isnan(row[warmup:]))
                assert non_nan_count > 0, f"Should have values after warmup for hull_length={hull_len}"
    
    def test_dma_batch_full_sweep(self, test_data):
        """Test DMA batch with multiple parameter combinations"""
        close = test_data['close'][:100]  
        
        result = ta_indicators.dma_batch(
            close,
            hull_length_range=(5, 7, 2),      
            ema_length_range=(15, 20, 5),     
            ema_gain_limit_range=(40, 50, 10), 
            hull_ma_type="WMA"
        )
        
        
        assert result['values'].shape[0] == 8
        assert result['values'].shape[1] == len(close)
        
        
        assert len(result['hull_lengths']) == 8
        assert len(result['ema_lengths']) == 8
        assert len(result['ema_gain_limits']) == 8
        
        
        assert result['hull_lengths'][0] == 5
        assert result['ema_lengths'][0] == 15
        assert result['ema_gain_limits'][0] == 40
        
        assert result['hull_lengths'][-1] == 7
        assert result['ema_lengths'][-1] == 20
        assert result['ema_gain_limits'][-1] == 50
    
    def test_dma_constant_input(self):
        """Test DMA with constant input values"""
        constant_val = EXPECTED_OUTPUTS['dma']['constant_value']
        input_data = np.full(100, constant_val)
        
        result = ta_indicators.dma(input_data, 7, 20, 50, "WMA")
        
        assert len(result) == len(input_data)
        
        
        
        last_10 = result[-10:]
        valid_values = last_10[~np.isnan(last_10)]
        
        if len(valid_values) > 0:
            
            assert_close(
                valid_values,
                np.full(len(valid_values), constant_val),
                rtol=0.01,  
                msg="DMA should converge to constant input value"
            )
    
    
    
    
    def test_dma_batch_edge_cases(self):
        """Test DMA batch processing edge cases"""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 5)  
        
        
        single_batch = ta_indicators.dma_batch(
            close,
            hull_length_range=(5, 5, 0),
            ema_length_range=(10, 10, 0),
            ema_gain_limit_range=(30, 30, 0),
            hull_ma_type="WMA"
        )
        
        assert single_batch['values'].shape[0] == 1
        assert len(single_batch['hull_lengths']) == 1
        
        
        large_step = ta_indicators.dma_batch(
            close,
            hull_length_range=(5, 7, 10),  
            ema_length_range=(10, 10, 0),
            ema_gain_limit_range=(30, 30, 0),
            hull_ma_type="WMA"
        )
        
        
        assert large_step['values'].shape[0] == 1
        assert large_step['hull_lengths'][0] == 5
    
    def test_dma_mixed_hull_types(self):
        """Test DMA with different Hull MA types"""
        input_data = np.array([float(i * 10 + np.sin(i)) for i in range(100)])
        results = {}
        
        for hull_type in EXPECTED_OUTPUTS['dma']['hull_ma_types']:
            result = ta_indicators.dma(
                input_data,
                hull_length=7,
                ema_length=20,
                ema_gain_limit=50,
                hull_ma_type=hull_type
            )
            
            assert len(result) == len(input_data), f"Result length mismatch for hull_type={hull_type}"
            
            
            non_nan_count = np.sum(~np.isnan(result))
            assert non_nan_count > 0, f"Should produce values for hull_type={hull_type}"
            
            results[hull_type] = result
        
        
        if 'EMA' in results and 'WMA' in results:
            ema_result = results['EMA']
            wma_result = results['WMA']
            valid_idx = ~(np.isnan(ema_result) | np.isnan(wma_result))
            if np.sum(valid_idx) > 0:
                assert not np.allclose(ema_result[valid_idx], wma_result[valid_idx], rtol=1e-10), \
                    "EMA and WMA hull types should produce different results"
    
    def test_dma_gain_limit_edge_cases(self):
        """Test DMA with edge case gain limit values"""
        input_data = np.array([float(i) for i in range(50)])
        
        
        result_zero = ta_indicators.dma(
            input_data,
            hull_length=7,
            ema_length=20,
            ema_gain_limit=0,
            hull_ma_type="WMA"
        )
        assert len(result_zero) == len(input_data), "Zero gain limit should produce output"
        
        
        result_large = ta_indicators.dma(
            input_data,
            hull_length=7,
            ema_length=20,
            ema_gain_limit=1000,
            hull_ma_type="WMA"
        )
        assert len(result_large) == len(input_data), "Large gain limit should produce output"
    
    def test_dma_trending_data(self):
        """Test DMA with perfectly trending data"""
        
        uptrend = np.array([float(i) for i in range(100)])
        result_up = ta_indicators.dma(uptrend, 7, 20, 50, "WMA")
        assert len(result_up) == len(uptrend)
        
        
        downtrend = np.array([100.0 - float(i) for i in range(100)])
        result_down = ta_indicators.dma(downtrend, 7, 20, 50, "WMA")
        assert len(result_down) == len(downtrend)
        
        
        warmup = 19  
        
        
        if len(result_up) > warmup + 10:
            last_10 = result_up[-10:]
            valid = last_10[~np.isnan(last_10)]
            if len(valid) > 1:
                
                diffs = np.diff(valid)
                assert np.mean(diffs) > 0, "DMA should follow uptrend"
    
    def test_dma_oscillating_data(self):
        """Test DMA with oscillating/choppy data"""
        
        x = np.linspace(0, 4 * np.pi, 100)
        oscillating = 50.0 + 10.0 * np.sin(x)
        
        result = ta_indicators.dma(
            oscillating,
            hull_length=7,
            ema_length=20,
            ema_gain_limit=50,
            hull_ma_type="WMA"
        )
        
        assert len(result) == len(oscillating)
        
        
        warmup = 19
        if len(result) > warmup:
            valid_result = result[warmup:]
            valid_input = oscillating[warmup:]
            
            
            result_var = np.nanvar(valid_result)
            input_var = np.var(valid_input)
            
            
            assert result_var < input_var, "DMA should smooth oscillating data"
    
    def test_dma_intermittent_nan(self):
        """Test DMA with data containing intermittent NaN values"""
        data = np.array([float(i) for i in range(100)])
        
        data[25] = np.nan
        data[50] = np.nan
        data[75] = np.nan
        
        
        result = ta_indicators.dma(data, 7, 20, 50, "WMA")
        assert len(result) == len(data)
        
        
        non_nan_count = np.sum(~np.isnan(result))
        assert non_nan_count > 0, "Should produce some non-NaN values"
    
    def test_dma_extreme_ratios(self):
        """Test DMA with extreme hull/ema length ratios"""
        input_data = np.array([float(i * 2 + np.random.randn()) for i in range(100)])
        
        
        result1 = ta_indicators.dma(
            input_data,
            hull_length=3,
            ema_length=50,
            ema_gain_limit=50,
            hull_ma_type="WMA"
        )
        assert len(result1) == len(input_data)
        
        
        result2 = ta_indicators.dma(
            input_data,
            hull_length=50,
            ema_length=3,
            ema_gain_limit=50,
            hull_ma_type="WMA"
        )
        assert len(result2) == len(input_data)
        
        
        valid_idx = ~(np.isnan(result1) | np.isnan(result2))
        if np.sum(valid_idx) > 0:
            assert not np.allclose(result1[valid_idx], result2[valid_idx], rtol=0.01), \
                "Different hull/ema ratios should produce different results"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])