"""
Python binding tests for DEVIATION indicator.
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


class TestDeviation:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    
    
    def test_deviation_accuracy(self, test_data):
        """Test DEVIATION matches expected values from Rust tests - mirrors check_deviation_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['deviation']
        
        result = ta_indicators.deviation(
            close,
            period=expected['default_params']['period'],
            devtype=expected['default_params']['devtype']
        )
        
        assert len(result) == len(close)
        
        
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-8,
            msg="Deviation last 5 values mismatch"
        )
        
        
        compare_with_rust('deviation', result, 'close', expected['default_params'])
    
    def test_deviation_default_params(self, test_data):
        """Test DEVIATION with default parameters - mirrors check_deviation_default_params"""
        close = test_data['close']
        
        
        result = ta_indicators.deviation(close, 9, 0)
        assert len(result) == len(close)
    
    def test_deviation_partial_params(self, test_data):
        """Test DEVIATION with partial parameters - mirrors check_deviation_partial_params"""
        close = test_data['close']
        
        
        result = ta_indicators.deviation(close, 9, 0)  
        assert len(result) == len(close)
    
    
    
    def test_deviation_mean_absolute(self, test_data):
        """Test mean absolute deviation - mirrors check_deviation_mean_absolute"""
        close = test_data['close']
        
        
        result = ta_indicators.deviation(close, period=20, devtype=1)
        assert len(result) == len(close)
        
        
        assert np.all(np.isnan(result[:19])), "Expected NaN in warmup period"
        
        
        assert not np.any(np.isnan(result[19:])), "Found unexpected NaN after warmup"
        
        
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= 0), "Mean absolute deviation should be non-negative"
    
    def test_deviation_median_absolute(self, test_data):
        """Test median absolute deviation - mirrors check_deviation_median_absolute"""
        close = test_data['close']
        
        
        result = ta_indicators.deviation(close, period=20, devtype=2)
        assert len(result) == len(close)
        
        
        assert np.all(np.isnan(result[:19])), "Expected NaN in warmup period"
        
        
        assert not np.any(np.isnan(result[19:])), "Found unexpected NaN after warmup"
        
        
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= 0), "Median absolute deviation should be non-negative"
    
    def test_deviation_mode(self, test_data):
        """Test mode deviation (devtype=3)"""
        close = test_data['close']
        
        
        result = ta_indicators.deviation(close, period=20, devtype=3)
        assert len(result) == len(close)
        
        
        assert np.all(np.isnan(result[:19])), "Expected NaN in warmup period"
        
        
        assert not np.any(np.isnan(result[19:])), "Found unexpected NaN after warmup"
    
    
    
    def test_deviation_zero_period(self):
        """Test DEVIATION fails with zero period - mirrors check_deviation_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.deviation(input_data, period=0, devtype=0)
    
    def test_deviation_period_exceeds_length(self):
        """Test DEVIATION fails when period exceeds data length - mirrors check_deviation_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough"):
            ta_indicators.deviation(data_small, period=10, devtype=0)
    
    def test_deviation_very_small_dataset(self):
        """Test DEVIATION fails with insufficient data - mirrors check_deviation_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough"):
            ta_indicators.deviation(single_point, period=9, devtype=0)
    
    def test_deviation_empty_input(self):
        """Test DEVIATION fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="EmptyData|empty"):
            ta_indicators.deviation(empty, period=9, devtype=0)
    
    def test_deviation_invalid_devtype(self):
        """Test DEVIATION fails with invalid devtype - mirrors check_deviation_invalid_devtype"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        
        with pytest.raises(ValueError, match="Invalid devtype|calculation error"):
            ta_indicators.deviation(data, period=2, devtype=4)
        
        
        
        with pytest.raises((ValueError, OverflowError), match="Invalid devtype|calculation error|can't convert"):
            ta_indicators.deviation(data, period=2, devtype=255)
    
    def test_deviation_all_nan_input(self):
        """Test DEVIATION with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.deviation(all_nan, period=9, devtype=0)
    
    
    
    def test_deviation_nan_handling(self, test_data):
        """Test DEVIATION handles NaN values correctly - mirrors check_deviation_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.deviation(close, period=20, devtype=0)
        assert len(result) == len(close)
        
        
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"
        
        
        assert np.all(np.isnan(result[:19])), "Expected NaN in warmup period"
    
    def test_deviation_nan_window_recovery(self):
        """Test DEVIATION recovers when NaN leaves the window"""
        
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        result = ta_indicators.deviation(data, period=5, devtype=0)
        assert len(result) == len(data)
        
        
        assert np.isnan(result[2]), "NaN should propagate at index 2"
        assert np.isnan(result[3]), "NaN should affect window at index 3"
        assert np.isnan(result[4]), "NaN should affect window at index 4"
        assert np.isnan(result[5]), "NaN should affect window at index 5"
        assert np.isnan(result[6]), "NaN should affect window at index 6"
        
        
        assert not np.isnan(result[7]), "Should recover after NaN leaves window"
        assert not np.isnan(result[8]), "Should have valid value at index 8"
        assert not np.isnan(result[9]), "Should have valid value at index 9"
    
    
    
    def test_deviation_streaming(self, test_data):
        """Test DEVIATION streaming matches batch calculation - mirrors check_deviation_streaming"""
        close = test_data['close'][:100]  
        period = 20
        devtype = 0
        
        
        batch_result = ta_indicators.deviation(close, period=period, devtype=devtype)
        
        
        stream = ta_indicators.DeviationStream(period=period, devtype=devtype)
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
                        msg=f"Deviation streaming mismatch at index {i}")
    
    def test_deviation_batch_single_params(self, test_data):
        """Test DEVIATION batch processing with single parameter set"""
        close = test_data['close'][:100]  
        
        result = ta_indicators.deviation_batch(
            close,
            period_range=(20, 20, 0),  
            devtype_range=(0, 0, 0)    
        )
        
        assert 'values' in result
        assert 'periods' in result
        assert 'devtypes' in result
        
        
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        
        batch_row = result['values'][0]
        
        
        single_result = ta_indicators.deviation(close, period=20, devtype=0)
        assert_close(batch_row, single_result, rtol=1e-9, 
                    msg="Batch single params should match single calculation")
    
    def test_deviation_batch_multiple_params(self, test_data):
        """Test DEVIATION batch processing with multiple parameter combinations"""
        close = test_data['close'][:100]  
        
        result = ta_indicators.deviation_batch(
            close,
            period_range=(10, 30, 10),   
            devtype_range=(0, 2, 1)       
        )
        
        assert 'values' in result
        assert 'periods' in result
        assert 'devtypes' in result
        
        
        assert result['values'].shape[0] == 9
        assert result['values'].shape[1] == len(close)
        
        
        first_combo = result['values'][0]
        expected_first = ta_indicators.deviation(close, period=10, devtype=0)
        assert_close(first_combo, expected_first, rtol=1e-9,
                    msg="First batch combination should match single calculation")
    
    def test_deviation_reinput(self, test_data):
        """Test DEVIATION applied twice (re-input)"""
        close = test_data['close'][:100]  
        
        
        first_result = ta_indicators.deviation(close, period=20, devtype=0)
        assert len(first_result) == len(close)
        
        
        
        valid_first = first_result[~np.isnan(first_result)]
        if len(valid_first) > 20:  
            second_result = ta_indicators.deviation(valid_first, period=10, devtype=0)
            assert len(second_result) == len(valid_first)
            
            
            assert not np.all(np.isnan(second_result)), "Second pass should have some valid values"
    
    
    
    def test_deviation_constant_values(self):
        """Test DEVIATION with constant values (should return 0)"""
        
        constant_data = np.full(50, 42.0)
        
        result = ta_indicators.deviation(constant_data, period=10, devtype=0)
        assert len(result) == len(constant_data)
        
        
        valid_values = result[~np.isnan(result)]
        assert_close(valid_values, np.zeros_like(valid_values), atol=1e-10,
                    msg="Standard deviation of constant values should be 0")
    
    def test_deviation_period_one(self):
        """Test special case of period=1 (always returns 0 for standard deviation)"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = ta_indicators.deviation(data, period=1, devtype=0)
        assert len(result) == len(data)
        
        
        assert_close(result, np.zeros_like(result), atol=1e-10,
                    msg="Standard deviation with period=1 should always be 0")
    
    def test_deviation_different_kernels(self, test_data):
        """Test DEVIATION with different kernel specifications"""
        close = test_data['close'][:100]
        
        
        kernels = [None, 'scalar', 'avx2', 'avx512']
        results = []
        
        for kernel in kernels:
            try:
                result = ta_indicators.deviation(close, period=20, devtype=0, kernel=kernel)
                results.append(result)
            except (ValueError, TypeError):
                
                continue
        
        
        if len(results) > 1:
            for i in range(1, len(results)):
                
                mask = ~(np.isnan(results[0]) | np.isnan(results[i]))
                if np.any(mask):
                    assert_close(results[0][mask], results[i][mask], rtol=1e-8,
                               msg=f"Kernel results should match")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])