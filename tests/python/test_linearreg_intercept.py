"""
Python binding tests for Linear Regression Intercept indicator.
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


class TestLinearRegIntercept:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_linearreg_intercept_partial_params(self, test_data):
        """Test Linear Regression Intercept with partial parameters (None values) - mirrors check_linreg_partial_params"""
        close = test_data['close']
        
        
        result = ta_indicators.linearreg_intercept(close, 14)
        assert len(result) == len(close)
    
    def test_linearreg_intercept_accuracy(self, test_data):
        """Test Linear Regression Intercept matches expected values from Rust tests - mirrors check_linreg_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['linearreg_intercept']
        
        result = ta_indicators.linearreg_intercept(close, period=expected['default_params']['period'])
        
        assert len(result) == len(close)
        
        
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-6,
            msg="Linear Regression Intercept last 5 values mismatch"
        )
    
    def test_linearreg_intercept_default_candles(self, test_data):
        """Test Linear Regression Intercept with default parameters - mirrors check_linreg_default_candles"""
        close = test_data['close']
        
        
        result = ta_indicators.linearreg_intercept(close, 14)
        assert len(result) == len(close)
    
    def test_linearreg_intercept_zero_period(self):
        """Test Linear Regression Intercept fails with zero period - mirrors check_linreg_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.linearreg_intercept(input_data, period=0)
    
    def test_linearreg_intercept_period_exceeds_length(self):
        """Test Linear Regression Intercept fails when period exceeds data length - mirrors check_linreg_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.linearreg_intercept(data_small, period=10)
    
    def test_linearreg_intercept_very_small_dataset(self):
        """Test Linear Regression Intercept fails with insufficient data - mirrors check_linreg_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.linearreg_intercept(single_point, period=14)
    
    def test_linearreg_intercept_empty_input(self):
        """Test Linear Regression Intercept fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.linearreg_intercept(empty, period=14)
    
    def test_linearreg_intercept_all_nan_input(self):
        """Test Linear Regression Intercept fails with all NaN values - mirrors check for AllValuesNaN"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.linearreg_intercept(all_nan, period=14)
    
    def test_linearreg_intercept_period_one(self):
        """Test Linear Regression Intercept with period=1 returns input values - special case"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        result = ta_indicators.linearreg_intercept(data, period=1)
        
        
        assert_close(result, data, rtol=1e-10, msg="Period=1 should return input values")
    
    def test_linearreg_intercept_linear_trend(self):
        """Test Linear Regression Intercept with perfect linear data - property test"""
        
        data = np.array([2.0 * i + 10.0 for i in range(50)])
        period = 10
        
        result = ta_indicators.linearreg_intercept(data, period=period)
        
        
        
        warmup = period - 1
        for i in range(warmup + 5, warmup + 10):
            window_start = i - period + 1
            expected = data[window_start]  
            assert_close(result[i], expected, rtol=1e-9, 
                        msg=f"Linear trend mismatch at index {i}")
    
    def test_linearreg_intercept_nan_handling(self, test_data):
        """Test Linear Regression Intercept handles NaN values correctly - mirrors check_linreg_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.linearreg_intercept(close, period=14)
        
        assert len(result) == len(close)
        
        
        
        non_nan_start = next((i for i, val in enumerate(close) if not np.isnan(val)), 0)
        expected_valid_start = non_nan_start + 14 - 1  
        
        if expected_valid_start < len(result):
            for i in range(expected_valid_start, min(expected_valid_start + 40, len(result))):
                assert not np.isnan(result[i]), f"Found unexpected NaN at index {i}"
    
    def test_linearreg_intercept_reinput(self, test_data):
        """Test Linear Regression Intercept with reinput - mirrors check_linreg_reinput"""
        close = test_data['close']
        
        
        first_result = ta_indicators.linearreg_intercept(close, period=14)
        
        
        second_result = ta_indicators.linearreg_intercept(first_result, period=14)
        
        assert len(second_result) == len(first_result)
        
        
        start = next((i for i, v in enumerate(second_result) if not np.isnan(v)), len(second_result))
        
        
        for i in range(start, len(second_result)):
            assert not np.isnan(second_result[i]), f"Unexpected NaN at index {i} after reinput"
    
    def test_linearreg_intercept_kernel_parameter(self, test_data):
        """Test Linear Regression Intercept with different kernel parameters"""
        close = test_data['close']
        
        
        result_scalar = ta_indicators.linearreg_intercept(close, period=14, kernel="scalar")
        assert len(result_scalar) == len(close)
        
        
        result_auto = ta_indicators.linearreg_intercept(close, period=14)
        assert len(result_auto) == len(close)
        
        
        assert_close(result_scalar[-5:], result_auto[-5:], rtol=1e-10)


class TestLinearRegInterceptStream:
    def test_stream_basic(self):
        """Test Linear Regression Intercept streaming functionality"""
        stream = ta_indicators.LinearRegInterceptStream(period=5)
        
        
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        results = []
        
        for value in data:
            result = stream.update(value)
            results.append(result)
        
        
        for i in range(4):
            assert results[i] is None
        
        
        for i in range(4, len(results)):
            assert results[i] is not None
            assert not np.isnan(results[i])
    
    def test_stream_period_one(self):
        """Test Linear Regression Intercept stream with period=1"""
        stream = ta_indicators.LinearRegInterceptStream(period=1)
        
        
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        
        for val in data:
            result = stream.update(val)
            assert result is not None
            assert_close(result, val, rtol=1e-10, msg=f"Period=1 stream should return input value {val}")
    
    def test_stream_nan_handling(self):
        """Test Linear Regression Intercept stream handles NaN correctly"""
        stream = ta_indicators.LinearRegInterceptStream(period=3)
        
        
        assert stream.update(1.0) is None
        assert stream.update(2.0) is None
        result = stream.update(3.0)
        assert result is not None
        
        
        result = stream.update(4.0)
        assert result is not None
        assert not np.isnan(result)


class TestLinearRegInterceptBatch:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_batch_basic(self, test_data):
        """Test Linear Regression Intercept batch processing"""
        close = test_data['close']
        
        
        result = ta_indicators.linearreg_intercept_batch(
            close,
            period_range=(10, 20, 5)  
        )
        
        
        assert 'values' in result
        assert 'periods' in result
        
        
        assert len(result['periods']) == 3
        assert list(result['periods']) == [10, 15, 20]
        
        
        assert result['values'].shape == (3, len(close))
    
    def test_batch_single_period(self, test_data):
        """Test Linear Regression Intercept batch with single period"""
        close = test_data['close']
        
        
        result = ta_indicators.linearreg_intercept_batch(
            close,
            period_range=(14, 14, 0)
        )
        
        
        assert len(result['periods']) == 1
        assert result['periods'][0] == 14
        assert result['values'].shape == (1, len(close))
    
    def test_batch_kernel_parameter(self, test_data):
        """Test Linear Regression Intercept batch with kernel parameter"""
        close = test_data['close']
        
        
        result = ta_indicators.linearreg_intercept_batch(
            close,
            period_range=(10, 15, 5),
            kernel="scalar"
        )
        
        assert 'values' in result
        assert result['values'].shape[0] == 2  