"""
Python binding tests for Midpoint indicator.
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


class TestMidpoint:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_midpoint_partial_params(self, test_data):
        """Test Midpoint with partial parameters (None values) - mirrors check_midpoint_partial_params"""
        close = test_data['close']
        
        
        result = ta_indicators.midpoint(close)
        assert len(result) == len(close)
    
    def test_midpoint_accuracy(self, test_data):
        """Test Midpoint matches expected values from Rust tests - mirrors check_midpoint_accuracy"""
        close = test_data['close']
        
        
        result = ta_indicators.midpoint(close, period=14)
        
        assert len(result) == len(close)
        
        
        expected_last_five = [59578.5, 59578.5, 59578.5, 58886.0, 58886.0]
        assert_close(
            result[-5:], 
            expected_last_five,
            rtol=1e-1,
            msg="Midpoint last 5 values mismatch"
        )
        
        
        compare_with_rust('midpoint', result, 'close', {'period': 14})
    
    def test_midpoint_default_candles(self, test_data):
        """Test Midpoint with default parameters - mirrors check_midpoint_default_candles"""
        close = test_data['close']
        
        
        result = ta_indicators.midpoint(close)
        assert len(result) == len(close)
    
    def test_midpoint_zero_period(self):
        """Test Midpoint fails with zero period - mirrors check_midpoint_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.midpoint(input_data, period=0)
    
    def test_midpoint_period_exceeds_length(self):
        """Test Midpoint fails when period exceeds data length - mirrors check_midpoint_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.midpoint(data_small, period=10)
    
    def test_midpoint_very_small_dataset(self):
        """Test Midpoint fails with insufficient data - mirrors check_midpoint_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.midpoint(single_point, period=9)
    
    def test_midpoint_empty_input(self):
        """Test Midpoint fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.midpoint(empty)
    
    def test_midpoint_reinput(self, test_data):
        """Test Midpoint applied twice (re-input) - mirrors check_midpoint_reinput"""
        close = test_data['close']
        
        
        first_result = ta_indicators.midpoint(close, period=14)
        assert len(first_result) == len(close)
        
        
        second_result = ta_indicators.midpoint(first_result, period=14)
        assert len(second_result) == len(first_result)
    
    def test_midpoint_nan_handling(self, test_data):
        """Test Midpoint handles NaN values correctly - mirrors check_midpoint_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.midpoint(close, period=14)
        assert len(result) == len(close)
        
        
        if len(result) > 240:
            non_nan_after_240 = ~np.isnan(result[240:])
            assert np.all(non_nan_after_240), "Found unexpected NaN after index 240"
        
        
        assert np.all(np.isnan(result[:13])), "Expected NaN in warmup period"
    
    def test_midpoint_all_nan_input(self):
        """Test Midpoint with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.midpoint(all_nan)
    
    def test_midpoint_streaming(self, test_data):
        """Test Midpoint streaming functionality - mirrors check_midpoint_streaming"""
        close = test_data['close']
        period = 14
        
        
        batch_result = ta_indicators.midpoint(close, period=period)
        
        
        stream = ta_indicators.MidpointStream(period=period)
        stream_values = []
        
        for price in close:
            value = stream.update(price)
            stream_values.append(value if value is not None else np.nan)
        
        assert len(batch_result) == len(stream_values)
        
        
        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, msg=f"Streaming mismatch at index {i}")
    
    def test_midpoint_batch_single_parameter(self, test_data):
        """Test batch calculation with single parameter combination"""
        close = test_data['close'][:100]  
        
        
        batch_result = ta_indicators.midpoint_batch(
            close,
            period_range=(14, 14, 0)
        )
        
        
        single_result = ta_indicators.midpoint(close, period=14)
        
        assert 'values' in batch_result
        assert 'periods' in batch_result
        assert batch_result['values'].shape == (1, len(close))
        
        
        batch_values = batch_result['values'][0]
        assert_close(batch_values, single_result, rtol=1e-10, msg="Batch vs single mismatch")
    
    def test_midpoint_batch_multiple_periods(self, test_data):
        """Test batch calculation with multiple period values"""
        close = test_data['close'][:100]  
        
        
        batch_result = ta_indicators.midpoint_batch(
            close,
            period_range=(10, 18, 4)
        )
        
        assert 'values' in batch_result
        assert 'periods' in batch_result
        assert batch_result['values'].shape == (3, len(close))
        assert len(batch_result['periods']) == 3
        assert list(batch_result['periods']) == [10, 14, 18]
        
        
        periods = [10, 14, 18]
        for i, period in enumerate(periods):
            single_result = ta_indicators.midpoint(close, period=period)
            batch_row = batch_result['values'][i]
            assert_close(batch_row, single_result, rtol=1e-10, msg=f"Period {period} mismatch")
    
    def test_midpoint_kernel_option(self, test_data):
        """Test that kernel parameter is accepted (optional)"""
        close = test_data['close'][:100]
        
        
        result_auto = ta_indicators.midpoint(close, period=14)
        result_scalar = ta_indicators.midpoint(close, period=14, kernel="scalar")
        
        
        assert_close(result_auto, result_scalar, rtol=1e-10)
    
    def test_midpoint_simple_case(self):
        """Test Midpoint with simple known values"""
        
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        period = 3
        
        result = ta_indicators.midpoint(data, period=period)
        
        
        assert np.all(np.isnan(result[:2]))
        
        
        assert_close(result[2], 2.0, rtol=1e-10)
        
        
        assert_close(result[3], 3.0, rtol=1e-10)
        
        
        assert_close(result[4], 4.0, rtol=1e-10)