"""
Python binding tests for FWMA indicator.
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
from rust_comparison import compare_with_rust


class TestFwma:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_fwma_partial_params(self, test_data):
        """Test FWMA with partial parameters - mirrors check_fwma_partial_params"""
        close = test_data['close']
        
        
        result = ta_indicators.fwma(close, 5)
        assert len(result) == len(close)
    
    def test_fwma_accuracy(self, test_data):
        """Test FWMA matches expected values from Rust tests - mirrors check_fwma_accuracy"""
        close = test_data['close']
        
        
        result = ta_indicators.fwma(close, 5)
        
        assert len(result) == len(close)
        
        
        period = 5
        assert np.all(np.isnan(result[:period-1])), f"Expected NaN in warmup period (first {period-1} values)"
        
        assert not np.isnan(result[period-1]), f"Expected valid value at index {period-1}"
        
        
        expected_last_five = [
            59273.583333333336,
            59252.5,
            59167.083333333336,
            59151.0,
            58940.333333333336
        ]
        
        
        assert_close(
            result[-5:], 
            expected_last_five,
            rtol=1e-8,
            msg="FWMA last 5 values mismatch"
        )
        
        
        compare_with_rust('fwma', result, 'close', {'period': 5})
    
    def test_fwma_default_candles(self, test_data):
        """Test FWMA with default parameters - mirrors check_fwma_default_candles"""
        close = test_data['close']
        
        
        result = ta_indicators.fwma(close, 5)
        assert len(result) == len(close)
        
        
        compare_with_rust('fwma', result, 'close', {'period': 5})
    
    def test_fwma_zero_period(self):
        """Test FWMA fails with zero period - mirrors check_fwma_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.fwma(input_data, period=0)
    
    def test_fwma_empty_input(self):
        """Test FWMA fails with empty input - mirrors ALMA's test_alma_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.fwma(empty, period=5)
    
    def test_fwma_all_nan_input(self):
        """Test FWMA fails with all NaN values"""
        all_nan = np.array([np.nan] * 10)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.fwma(all_nan, period=3)
    
    def test_fwma_period_exceeds_length(self):
        """Test FWMA fails when period exceeds data length - mirrors check_fwma_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.fwma(data_small, period=5)
    
    def test_fwma_very_small_dataset(self):
        """Test FWMA with very small dataset - mirrors check_fwma_very_small_dataset"""
        data_single = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.fwma(data_single, period=5)
    
    def test_fwma_reinput(self, test_data):
        """Test FWMA with re-input of FWMA result - mirrors check_fwma_reinput"""
        close = test_data['close']
        
        
        first_result = ta_indicators.fwma(close, 5)
        
        
        second_result = ta_indicators.fwma(first_result, 3)
        
        assert len(second_result) == len(first_result)
        
        
        for i in range(240, len(second_result)):
            assert not np.isnan(second_result[i]), f"NaN found at index {i}"
    
    def test_fwma_nan_handling(self, test_data):
        """Test FWMA handling of NaN values - mirrors check_fwma_nan_handling"""
        data = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        period = 3
        
        result = ta_indicators.fwma(data, period)
        
        assert len(result) == len(data)
        
        assert np.isnan(result[:2 + period - 1]).all()
        
        assert not np.isnan(result[2 + period - 1:]).any()
    
    def test_fwma_streaming(self, test_data):
        """Test FWMA streaming vs batch calculation - mirrors check_fwma_streaming"""
        close = test_data['close'][:100]  
        period = 5
        
        
        batch_result = ta_indicators.fwma(close, period)
        
        
        stream = ta_indicators.FwmaStream(period)
        stream_results = []
        
        for val in close:
            result = stream.update(val)
            stream_results.append(result if result is not None else np.nan)
        
        stream_results = np.array(stream_results)
        
        
        assert_close(
            stream_results[period-1:], 
            batch_result[period-1:],
            rtol=1e-9,
            msg="FWMA streaming vs batch mismatch"
        )
    
    def test_fwma_batch(self, test_data):
        """Test FWMA batch computation with multiple parameter sets."""
        close = test_data['close'][:100]  
        
        
        period_range = (3, 10, 2)  
        
        result = ta_indicators.fwma_batch(close, period_range)
        
        assert isinstance(result, dict)
        assert 'values' in result
        assert 'periods' in result
        
        expected_periods = [3, 5, 7, 9]
        assert result['values'].shape == (len(expected_periods), len(close))
        assert list(result['periods']) == expected_periods
        
        
        for i, period in enumerate(expected_periods):
            individual_result = ta_indicators.fwma(close, period)
            np.testing.assert_allclose(result['values'][i], individual_result, rtol=1e-9)
        
        
        single_result = ta_indicators.fwma_batch(close, (5, 5, 0))
        assert single_result['values'].shape == (1, len(close))
        assert list(single_result['periods']) == [5]
        
        
        individual_5 = ta_indicators.fwma(close, 5)
        np.testing.assert_allclose(single_result['values'][0], individual_5, rtol=1e-9)
        
        
        for i, period in enumerate(expected_periods):
            row_values = result['values'][i]
            
            assert np.all(np.isnan(row_values[:period-1])), \
                f"Row {i} (period {period}): Expected NaN in warmup period"
            
            assert not np.isnan(row_values[period-1]), \
                f"Row {i} (period {period}): Expected valid value at index {period-1}"
    
    def test_fwma_fibonacci_weights(self):
        """Test that FWMA correctly applies Fibonacci weights."""
        
        
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        period = 5
        
        result = ta_indicators.fwma(data, period)
        
        
        
        expected = (1*1 + 2*1 + 3*2 + 4*3 + 5*5) / 12
        assert np.isclose(result[-1], expected, rtol=1e-9)