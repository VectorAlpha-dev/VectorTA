"""
Python binding tests for TSI indicator.
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

from test_utils import load_test_data, assert_close


class TestTsi:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_tsi_partial_params(self, test_data):
        """Test TSI with partial parameters (None values) - mirrors check_tsi_partial_params"""
        close = test_data['close']
        
        
        result = ta_indicators.tsi(close, 25, 13)
        assert len(result) == len(close)
    
    def test_tsi_accuracy(self, test_data):
        """Test TSI matches expected values from Rust tests - mirrors check_tsi_accuracy"""
        close = test_data['close']
        
        result = ta_indicators.tsi(close, long_period=25, short_period=13)
        
        assert len(result) == len(close)
        
        
        expected_last_five = [
            -17.757654061849838,
            -17.367527062626184,
            -17.305577681249513,
            -16.937565646991143,
            -17.61825617316731,
        ]
        
        
        assert_close(
            result[-5:], 
            expected_last_five,
            rtol=1e-7,
            msg="TSI last 5 values mismatch"
        )
    
    def test_tsi_default_candles(self, test_data):
        """Test TSI with default parameters - mirrors check_tsi_default_candles"""
        close = test_data['close']
        
        
        result = ta_indicators.tsi(close, 25, 13)
        assert len(result) == len(close)
    
    def test_tsi_zero_period(self):
        """Test TSI fails with zero period - mirrors check_tsi_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.tsi(input_data, long_period=0, short_period=13)
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.tsi(input_data, long_period=25, short_period=0)
    
    def test_tsi_period_exceeds_length(self):
        """Test TSI fails when period exceeds data length - mirrors check_tsi_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.tsi(data_small, long_period=25, short_period=13)
    
    def test_tsi_very_small_dataset(self):
        """Test TSI fails with insufficient data - mirrors check_tsi_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.tsi(single_point, long_period=25, short_period=13)
    
    def test_tsi_reinput(self, test_data):
        """Test TSI applied twice (re-input) - mirrors check_tsi_reinput"""
        close = test_data['close']
        
        
        first_result = ta_indicators.tsi(close, long_period=25, short_period=13)
        assert len(first_result) == len(close)
        
        
        second_result = ta_indicators.tsi(first_result, long_period=25, short_period=13)
        assert len(second_result) == len(first_result)
    
    def test_tsi_nan_handling(self, test_data):
        """Test TSI handles NaN values correctly - mirrors check_tsi_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.tsi(close, long_period=25, short_period=13)
        assert len(result) == len(close)
        
        
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"
        
        
        
        warmup = 25 + 13  
        assert np.all(np.isnan(result[:warmup])), f"Expected NaN in warmup period (first {warmup} values)"
    
    def test_tsi_all_nan_input(self):
        """Test TSI with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.tsi(all_nan, long_period=25, short_period=13)
    
    def test_tsi_streaming(self, test_data):
        """Test TSI streaming matches batch calculation - mirrors check_tsi_streaming"""
        close = test_data['close']
        long_period = 25
        short_period = 13
        
        
        batch_result = ta_indicators.tsi(close, long_period=long_period, short_period=short_period)
        
        
        stream = ta_indicators.TsiStream(long_period=long_period, short_period=short_period)
        stream_values = []
        
        for price in close:
            result = stream.update(price)
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        
        assert len(batch_result) == len(stream_values)
        
        
        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            if not (np.isnan(b) or np.isnan(s)):
                assert_close(b, s, rtol=1e-9, atol=1e-9, 
                            msg=f"TSI streaming mismatch at index {i}")
    
    def test_tsi_batch(self, test_data):
        """Test TSI batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        result = ta_indicators.tsi_batch(
            close,
            long_period_range=(25, 25, 0),  
            short_period_range=(13, 13, 0)  
        )
        
        assert 'values' in result
        assert 'long_periods' in result
        assert 'short_periods' in result
        
        
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        
        default_row = result['values'][0]
        expected_last_five = [
            -17.757654061849838,
            -17.367527062626184,
            -17.305577681249513,
            -16.937565646991143,
            -17.61825617316731,
        ]
        
        
        assert_close(
            default_row[-5:],
            expected_last_five,
            rtol=1e-7,
            msg="TSI batch default row mismatch"
        )
    
    def test_tsi_batch_multiple_params(self, test_data):
        """Test TSI batch with multiple parameter combinations"""
        close = test_data['close']
        
        result = ta_indicators.tsi_batch(
            close,
            long_period_range=(20, 30, 5),  
            short_period_range=(10, 15, 5)  
        )
        
        assert 'values' in result
        assert 'long_periods' in result
        assert 'short_periods' in result
        
        
        assert result['values'].shape[0] == 6
        assert result['values'].shape[1] == len(close)
        
        
        expected_longs = [20, 20, 25, 25, 30, 30]
        expected_shorts = [10, 15, 10, 15, 10, 15]
        
        np.testing.assert_array_equal(result['long_periods'], expected_longs)
        np.testing.assert_array_equal(result['short_periods'], expected_shorts)
    
    def test_tsi_mid_series_nan(self):
        """Test TSI handles mid-series NaN values correctly"""
        
        data = np.array([
            100.0, 102.0, 101.0, 103.0, 104.0,
            np.nan, np.nan,  
            105.0, 106.0, 107.0, 108.0, 109.0, 
            110.0, 111.0, 112.0, 113.0, 114.0, 
            115.0, 116.0, 117.0
        ])
        
        
        result = ta_indicators.tsi(data, long_period=5, short_period=3)
        assert len(result) == len(data)
        
        
        assert np.isnan(result[5])
        assert np.isnan(result[6])
        
        
        
        valid_after_gap = result[10:]
        valid_count = np.sum(~np.isnan(valid_after_gap))
        assert valid_count > 0, "TSI should recover after mid-series NaN gap"
    
    def test_tsi_constant_data(self):
        """Test TSI with constant data (zero momentum)"""
        
        constant = np.full(50, 100.0)
        result = ta_indicators.tsi(constant, long_period=10, short_period=5)
        
        
        warmup = 10 + 5
        assert np.all(np.isnan(result[warmup:])), "TSI should be NaN for constant prices"
    
    def test_tsi_step_data(self):
        """Test TSI with step function data"""
        
        data = np.concatenate([
            np.full(25, 100.0),
            np.full(25, 150.0)
        ])
        
        result = ta_indicators.tsi(data, long_period=10, short_period=5)
        assert len(result) == len(data)
        
        
        
        last_values = result[-5:]
        valid_last = last_values[~np.isnan(last_values)]
        if len(valid_last) > 0:
            
            assert np.all(valid_last >= -100) and np.all(valid_last <= 100), \
                "TSI values should be in [-100, 100] range"
    
    def test_tsi_kernel_options(self, test_data):
        """Test TSI with different kernel options"""
        close = test_data['close']
        
        
        kernels = ['scalar', 'avx2', 'avx512', None]  
        
        for kernel in kernels:
            try:
                if kernel is None:
                    result = ta_indicators.tsi(close, 25, 13)
                else:
                    result = ta_indicators.tsi(close, 25, 13, kernel=kernel)
                assert len(result) == len(close)
            except (TypeError, ValueError):
                
                pass
    
    def test_tsi_edge_cases(self):
        """Test TSI edge cases"""
        
        min_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = ta_indicators.tsi(min_data, long_period=3, short_period=2)
        assert len(result) == len(min_data)
        
        
        data = np.random.randn(100) * 10 + 100
        result = ta_indicators.tsi(data, long_period=10, short_period=10)
        assert len(result) == len(data)
        
        
        result = ta_indicators.tsi(data, long_period=2, short_period=1)
        assert len(result) == len(data)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])