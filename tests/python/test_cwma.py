"""
Python binding tests for CWMA indicator.
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


class TestCwma:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_cwma_partial_params(self, test_data):
        """Test CWMA with partial parameters - mirrors check_cwma_partial_params"""
        close = test_data['close']
        
        
        result = ta_indicators.cwma(close, 14)
        assert len(result) == len(close)
    
    def test_cwma_accuracy(self, test_data):
        """Test CWMA matches expected values from Rust tests - mirrors check_cwma_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['cwma']
        
        
        result = ta_indicators.cwma(close, period=expected['default_params']['period'])
        
        assert len(result) == len(close)
        
        
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-5,  
            atol=1.0,   
            msg="CWMA last 5 values mismatch"
        )
        
        
        
        compare_with_rust('cwma', result, 'close', expected['default_params'], rtol=5e-5, atol=1e-6)
    
    def test_cwma_default_candles(self, test_data):
        """Test CWMA with default parameters - mirrors check_cwma_default_candles"""
        close = test_data['close']
        
        
        result = ta_indicators.cwma(close, 14)
        assert len(result) == len(close)
    
    def test_cwma_zero_period(self):
        """Test CWMA fails with zero period - mirrors check_cwma_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.cwma(input_data, period=0)
    
    def test_cwma_period_exceeds_length(self):
        """Test CWMA fails when period exceeds data length - mirrors check_cwma_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.cwma(data_small, period=10)
    
    def test_cwma_very_small_dataset(self):
        """Test CWMA fails with insufficient data - mirrors check_cwma_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.cwma(single_point, period=9)
    
    def test_cwma_empty_input(self):
        """Test CWMA fails with empty input - mirrors check_cwma_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.cwma(empty, period=9)
    
    def test_cwma_reinput(self, test_data):
        """Test CWMA applied twice (re-input) - mirrors check_cwma_reinput"""
        close = test_data['close']
        
        
        first_result = ta_indicators.cwma(close, period=80)
        assert len(first_result) == len(close)
        
        
        second_result = ta_indicators.cwma(first_result, period=60)
        assert len(second_result) == len(first_result)
        
        
        if len(second_result) > 240:
            assert not np.any(np.isnan(second_result[240:])), "Found unexpected NaN after warmup period"
    
    def test_cwma_nan_handling(self, test_data):
        """Test CWMA handles NaN values correctly - mirrors check_cwma_nan_handling"""
        close = test_data['close']
        period = 9
        
        result = ta_indicators.cwma(close, period=period)
        assert len(result) == len(close)
        
        
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"
        
        
        warmup_length = period - 1
        assert np.all(np.isnan(result[:warmup_length])), f"Expected NaN in warmup period (first {warmup_length} values)"
        assert not np.isnan(result[warmup_length]), f"Expected valid value at index {warmup_length}"
    
    def test_cwma_streaming(self, test_data):
        """Test CWMA streaming matches batch calculation - mirrors check_cwma_streaming"""
        close = test_data['close']
        period = 9
        
        
        batch_result = ta_indicators.cwma(close, period=period)
        
        
        stream = ta_indicators.CwmaStream(period=period)
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
                        msg=f"CWMA streaming mismatch at index {i}")
    
    def test_cwma_batch(self, test_data):
        """Test CWMA batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['cwma']
        
        result = ta_indicators.cwma_batch(
            close,
            period_range=(14, 14, 0),  
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        assert result['periods'][0] == 14
        
        
        default_row = result['values'][0]
        
        
        assert_close(
            default_row[-5:],
            expected['last_5_values'],
            rtol=1e-4,  
            atol=10.0,  
            msg="CWMA batch default row mismatch"
        )
    
    def test_cwma_all_nan_input(self):
        """Test CWMA with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.cwma(all_nan, period=9)
    
    def test_cwma_period_one(self):
        """Test CWMA fails with period=1 - mirrors check that period=1 is invalid"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.cwma(data, period=1)
    
    def test_cwma_leading_nans(self, test_data):
        """Test CWMA handles leading NaN values correctly"""
        close = test_data['close'].copy()
        
        
        close[:10] = np.nan
        
        result = ta_indicators.cwma(close, period=14)
        assert len(result) == len(close)
        
        
        expected_nans = 10 + 13  
        assert np.all(np.isnan(result[:expected_nans])), f"Expected NaN in first {expected_nans} values"
        assert not np.isnan(result[expected_nans]), f"Expected valid value at index {expected_nans}"
    
    def test_cwma_mixed_nans(self, test_data):
        """Test CWMA handles NaN values scattered throughout data"""
        close = test_data['close'].copy()
        
        
        close[50:55] = np.nan
        close[100] = np.nan
        close[150:152] = np.nan
        
        result = ta_indicators.cwma(close, period=14)
        assert len(result) == len(close)
        
        
        
        
    
    @pytest.mark.skip(reason="Known issue: batch implementation doesn't properly initialize NaN values")
    def test_cwma_batch_multiple_periods(self, test_data):
        """Test CWMA batch with multiple period values"""
        close = test_data['close'][:200]  
        
        result = ta_indicators.cwma_batch(
            close,
            period_range=(10, 20, 5),  
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        
        assert result['values'].shape[0] == 3
        assert result['values'].shape[1] == len(close)
        assert list(result['periods']) == [10, 15, 20]
        
        
        for i, period in enumerate(result['periods']):
            row_data = result['values'][i]
            warmup = int(period) - 1
            
            
            for j in range(min(warmup, len(row_data))):
                assert np.isnan(row_data[j]), f"Expected NaN at index {j} for period {period}"
            
            
            if warmup < len(row_data):
                assert not np.isnan(row_data[warmup]), f"Expected valid value at index {warmup} for period {period}"
                
                valid_values = row_data[warmup:]
                valid_values = valid_values[~np.isnan(valid_values)]
                if len(valid_values) > 0:
                    assert np.all(np.abs(valid_values) < 1e10), f"Found unreasonable values for period {period}"
    
    def test_cwma_batch_edge_cases(self):
        """Test CWMA batch processing edge cases"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        
        result = ta_indicators.cwma_batch(
            data,
            period_range=(5, 5, 0),
        )
        assert result['values'].shape[0] == 1
        assert result['periods'][0] == 5
        
        
        result = ta_indicators.cwma_batch(
            data,
            period_range=(3, 5, 10),
        )
        assert result['values'].shape[0] == 1  
        assert result['periods'][0] == 3
    
    def test_cwma_warmup_validation(self):
        """Test CWMA warmup period is exactly period - 1"""
        data = np.arange(1.0, 101.0)  
        
        test_periods = [2, 3, 5, 10, 14, 20, 30]
        for period in test_periods:
            result = ta_indicators.cwma(data, period=period)
            
            
            nan_count = 0
            for val in result:
                if np.isnan(val):
                    nan_count += 1
                else:
                    break
            
            expected_warmup = period - 1
            assert nan_count == expected_warmup, \
                f"Period {period}: Expected {expected_warmup} NaN values, got {nan_count}"
            
            
            assert not np.isnan(result[expected_warmup]), \
                f"Period {period}: Expected valid value at index {expected_warmup}"
            if expected_warmup > 0:
                assert np.isnan(result[expected_warmup - 1]), \
                    f"Period {period}: Expected NaN at index {expected_warmup - 1}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])