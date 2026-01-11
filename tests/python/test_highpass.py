"""
Python binding tests for HighPass indicator.
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


class TestHighPass:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_highpass_partial_params(self, test_data):
        """Test HighPass with partial parameters - mirrors check_highpass_partial_params"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['highpass']
        
        
        result = ta_indicators.highpass(close, expected['default_params']['period'])
        assert len(result) == len(close)
        
        
        result_default = ta_indicators.highpass(close)
        assert len(result_default) == len(close)
        
        
        np.testing.assert_array_equal(result, result_default)
    
    def test_highpass_accuracy(self, test_data):
        """Test HighPass matches expected values from Rust tests - mirrors check_highpass_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['highpass']
        
        result = ta_indicators.highpass(
            close,
            period=expected['default_params']['period']
        )
        
        assert len(result) == len(close)
        
        
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-6,  
            msg="HighPass last 5 values mismatch"
        )
        
        
        compare_with_rust('highpass', result, 'close', expected['default_params'])
    
    def test_highpass_default_candles(self, test_data):
        """Test HighPass with default parameters - mirrors check_highpass_default_candles"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['highpass']
        
        
        result = ta_indicators.highpass(close, expected['default_params']['period'])
        assert len(result) == len(close)
        
        
        compare_with_rust('highpass', result, 'close', expected['default_params'])
    
    def test_highpass_zero_period(self):
        """Test HighPass fails with zero period - mirrors check_highpass_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError):
            ta_indicators.highpass(input_data, period=0)
    
    def test_highpass_period_exceeds_length(self):
        """Test HighPass fails when period exceeds data length - mirrors check_highpass_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError):
            ta_indicators.highpass(data_small, period=48)
    
    def test_highpass_very_small_dataset(self):
        """Test HighPass with very small dataset - mirrors check_highpass_very_small_dataset"""
        data_small = np.array([42.0, 43.0])
        
        
        with pytest.raises(ValueError):
            ta_indicators.highpass(data_small, period=2)
    
    def test_highpass_empty_input(self):
        """Test HighPass with empty input - mirrors check_highpass_empty_input"""
        data_empty = np.array([])
        
        with pytest.raises(ValueError):
            ta_indicators.highpass(data_empty, period=48)
    
    def test_highpass_invalid_alpha(self):
        """Test HighPass with invalid alpha - mirrors check_highpass_invalid_alpha"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        
        with pytest.raises(ValueError):
            ta_indicators.highpass(data, period=4)
    
    def test_highpass_all_nan(self):
        """Test HighPass with all NaN input"""
        data = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        
        with pytest.raises(ValueError):
            ta_indicators.highpass(data, period=3)
    
    def test_highpass_warmup_period(self, test_data):
        """Test that HighPass has no warmup period - starts from index 0"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['highpass']
        
        result = ta_indicators.highpass(close, expected['default_params']['period'])
        
        
        assert not np.isnan(result[0]), "HighPass should produce value at index 0 (no warmup)"
        
        
        assert expected['has_warmup'] == False, "HighPass should have no warmup period"
        assert expected['warmup_length'] == 0, "HighPass warmup length should be 0"
        
        
        for i in range(len(result)):
            assert not np.isnan(result[i]), f"Unexpected NaN at index {i}"
    
    def test_highpass_nan_handling(self, test_data):
        """Test HighPass handling of NaN values - mirrors check_highpass_nan_handling"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['highpass']
        period = expected['default_params']['period']
        
        result = ta_indicators.highpass(close, period)
        
        assert len(result) == len(close)
        
        
        
        assert not np.isnan(result[0]), "HighPass should produce value at index 0"
        
        
        for i in range(len(result)):
            assert not np.isnan(result[i]), f"Unexpected NaN at index {i}"
    
    def test_highpass_streaming(self, test_data):
        """Test HighPass streaming vs batch calculation"""
        close = test_data['close'][:100]  
        period = 48
        
        
        batch_result = ta_indicators.highpass(close, period)
        
        
        stream = ta_indicators.HighPassStream(period)
        stream_results = []
        
        for val in close:
            result = stream.update(val)
            stream_results.append(result if result is not None else np.nan)
        
        stream_results = np.array(stream_results)
        
        
        assert_close(
            stream_results, 
            batch_result,
            rtol=1e-8,
            msg="HighPass streaming vs batch mismatch"
        )
    
    def test_highpass_reinput(self, test_data):
        """Test HighPass with re-input of HighPass result - kept for backward compatibility"""
        close = test_data['close']
        
        
        first_result = ta_indicators.highpass(close, 36)
        
        
        second_result = ta_indicators.highpass(first_result, 24)
        
        assert len(second_result) == len(first_result)
        
        
        for i in range(240, len(second_result)):
            assert not np.isnan(second_result[i]), f"NaN found at index {i}"
    
    def test_highpass_batch(self, test_data):
        """Test HighPass batch computation."""
        close = test_data['close']
        
        
        period_range = (30, 60, 10)  
        
        result = ta_indicators.highpass_batch(close, period_range)
        
        assert isinstance(result, dict)
        assert 'values' in result
        assert 'periods' in result
        
        values = result['values']
        periods = result['periods']
        
        expected_periods = [30, 40, 50, 60]
        
        assert list(periods) == expected_periods
        assert values.shape == (4, len(close))  
        
        
        row_idx = 0
        for period in [30, 40, 50, 60]:
            individual_result = ta_indicators.highpass(close, period)
            np.testing.assert_allclose(
                values[row_idx], 
                individual_result, 
                rtol=1e-9,
                err_msg=f"Batch row {row_idx} (period={period}) mismatch"
            )
            row_idx += 1
    
    def test_highpass_different_periods(self, test_data):
        """Test HighPass with different period values."""
        close = test_data['close']
        
        
        for period in [10, 20, 30, 48, 60]:
            result = ta_indicators.highpass(close, period)
            assert len(result) == len(close)
            
            
            first_valid = None
            for i in range(len(result)):
                if not np.isnan(result[i]):
                    first_valid = i
                    break
            
            
            assert first_valid is not None, f"No valid data found for period={period}"
            
            
            for i in range(first_valid, len(result)):
                assert not np.isnan(result[i]), f"Unexpected NaN at index {i} for period={period}"
    
    def test_highpass_leading_nan_input(self, test_data):
        """Test HighPass with leading NaN values in input"""
        close = test_data['close'][:100].copy()
        
        
        close[:5] = np.nan
        
        
        result = ta_indicators.highpass(close, 48)
        assert len(result) == len(close)
        
        
        
        assert all(np.isnan(result)), "IIR filter should propagate NaN through entire output"
    
    def test_highpass_batch_multiple_params(self, test_data):
        """Test HighPass batch with multiple parameter combinations"""
        close = test_data['close'][:500]  
        
        
        period_range = (10, 50, 10)  
        
        result = ta_indicators.highpass_batch(close, period_range)
        
        assert isinstance(result, dict)
        assert 'values' in result
        assert 'periods' in result
        
        values = result['values']
        periods = result['periods']
        
        expected_periods = [10, 20, 30, 40, 50]
        assert list(periods) == expected_periods
        assert values.shape == (5, len(close))  
        
        
        for row in range(5):
            assert not np.isnan(values[row, 0]), f"Row {row} should have value at index 0"
            
            
            individual = ta_indicators.highpass(close, expected_periods[row])
            np.testing.assert_allclose(
                values[row], 
                individual, 
                rtol=1e-9,
                err_msg=f"Batch row {row} (period={expected_periods[row]}) mismatch"
            )
    
    def test_highpass_edge_cases(self, test_data):
        """Test HighPass with edge case inputs"""
        
        period = 10
        data_exact = test_data['close'][:period]
        
        result = ta_indicators.highpass(data_exact, period)
        assert len(result) == len(data_exact)
        assert not np.isnan(result[0]), "Should have value at index 0"
        
        
        data_plus_one = test_data['close'][:period+1]
        result = ta_indicators.highpass(data_plus_one, period)
        assert len(result) == len(data_plus_one)
        assert not np.isnan(result[0]), "Should have value at index 0"
        
        
        constant_data = np.full(100, 50.0)
        result = ta_indicators.highpass(constant_data, 20)
        
        
        
        stabilized_start = 3 * 20
        for i in range(stabilized_start, len(result)):
            assert abs(result[i]) < 1e-3, f"DC component not removed at index {i}: {result[i]}"