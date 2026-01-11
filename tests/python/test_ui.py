"""
Python binding tests for UI (Ulcer Index) indicator.
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


class TestUI:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_ui_partial_params(self, test_data):
        """Test UI with partial parameters - mirrors check_ui_partial_params"""
        close = test_data['close']
        
        
        result = ta_indicators.ui(close, period=14, scalar=100.0)
        assert len(result) == len(close)
    
    def test_ui_accuracy(self, test_data):
        """Test UI matches expected values from Rust tests - mirrors check_ui_accuracy"""
        close = test_data['close']
        
        result = ta_indicators.ui(close, period=14, scalar=100.0)
        
        assert len(result) == len(close)
        
        
        expected_last_five = [
            3.514342861283708,
            3.304986039846459,
            3.2011859814326304,
            3.1308860017483373,
            2.909612553474519,
        ]
        
        
        assert_close(
            result[-5:], 
            expected_last_five,
            rtol=1e-6,
            msg="UI last 5 values mismatch"
        )
        
        
        period = 14
        for i in range(period - 1):
            assert np.isnan(result[i]), f"Expected NaN at index {i} during warmup period"
        
        
        params = {'period': 14, 'scalar': 100.0}
        compare_with_rust('ui', result, 'close', params)
    
    def test_ui_default_candles(self, test_data):
        """Test UI with default parameters - mirrors check_ui_default_candles"""
        close = test_data['close']
        
        
        result = ta_indicators.ui(close, period=14, scalar=100.0)
        assert len(result) == len(close)
    
    def test_ui_zero_period(self):
        """Test UI fails with zero period - mirrors check_ui_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.ui(input_data, period=0, scalar=100.0)
    
    def test_ui_period_exceeds_length(self):
        """Test UI fails when period exceeds data length - mirrors check_ui_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.ui(data_small, period=10, scalar=100.0)
    
    def test_ui_very_small_dataset(self):
        """Test UI fails with insufficient data - mirrors check_ui_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.ui(single_point, period=14, scalar=100.0)
    
    def test_ui_empty_input(self):
        """Test UI fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty|Empty"):
            ta_indicators.ui(empty, period=14, scalar=100.0)
    
    def test_ui_all_nan_input(self):
        """Test UI fails with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.ui(all_nan, period=14, scalar=100.0)
    
    def test_ui_different_scalars(self, test_data):
        """Test UI with different scalar values"""
        close = test_data['close']
        
        
        result1 = ta_indicators.ui(close, period=14, scalar=50.0)
        assert len(result1) == len(close)
        
        
        result2 = ta_indicators.ui(close, period=14, scalar=200.0)
        assert len(result2) == len(close)
        
        
        
        
        result_default = ta_indicators.ui(close, period=14, scalar=100.0)
        for i in range(14*2-2, len(close)):  
            if not np.isnan(result_default[i]) and not np.isnan(result2[i]):
                ratio = result2[i] / result_default[i]
                assert abs(ratio - 2.0) < 0.01, f"Scalar scaling incorrect at index {i}"
    
    def test_ui_kernel_parameter(self, test_data):
        """Test UI with different kernel parameters"""
        close = test_data['close']
        
        
        result1 = ta_indicators.ui(close, period=14, scalar=100.0, kernel=None)
        assert len(result1) == len(close)
        
        
        result2 = ta_indicators.ui(close, period=14, scalar=100.0, kernel="scalar")
        assert len(result2) == len(close)
        
        
        assert_close(result1, result2, rtol=1e-10, msg="Kernel results mismatch")
    
    def test_ui_batch_operations(self, test_data):
        """Test UI batch operations"""
        close = test_data['close']
        
        
        batch_result = ta_indicators.ui_batch(
            close,
            period_range=(14, 14, 1),
            scalar_range=(100.0, 100.0, 0.0)
        )
        
        
        single_result = ta_indicators.ui(close, period=14, scalar=100.0)
        
        assert 'values' in batch_result
        assert 'periods' in batch_result
        assert 'scalars' in batch_result
        
        batch_values = batch_result['values']
        assert batch_values.shape == (1, len(close))
        
        
        assert_close(
            batch_values[0], 
            single_result, 
            rtol=1e-10, 
            msg="Batch vs single mismatch"
        )
    
    def test_ui_batch_multiple_periods(self, test_data):
        """Test UI batch with multiple period values"""
        close = test_data['close'][:100]  
        
        
        batch_result = ta_indicators.ui_batch(
            close,
            period_range=(10, 14, 2),
            scalar_range=(100.0, 100.0, 0.0)
        )
        
        assert batch_result['values'].shape == (3, 100)
        assert len(batch_result['periods']) == 3
        assert list(batch_result['periods']) == [10, 12, 14]
        
        
        for i, period in enumerate([10, 12, 14]):
            single_result = ta_indicators.ui(close, period=period, scalar=100.0)
            assert_close(
                batch_result['values'][i], 
                single_result, 
                rtol=1e-10, 
                msg=f"Period {period} mismatch"
            )
    
    def test_ui_batch_multiple_scalars(self, test_data):
        """Test UI batch with multiple scalar values"""
        close = test_data['close'][:100]  
        
        
        batch_result = ta_indicators.ui_batch(
            close,
            period_range=(14, 14, 1),
            scalar_range=(50.0, 150.0, 50.0)
        )
        
        assert batch_result['values'].shape == (3, 100)
        assert len(batch_result['scalars']) == 3
        assert_close(
            batch_result['scalars'], 
            [50.0, 100.0, 150.0], 
            rtol=1e-10
        )
    
    def test_ui_streaming(self):
        """Test UI streaming functionality"""
        
        stream = ta_indicators.UiStream(period=5, scalar=100.0)
        
        
        
        data = [10.0, 12.0, 8.0, 15.0, 11.0, 9.0, 13.0, 10.0, 14.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0]
        
        results = []
        for value in data:
            result = stream.update(value)
            results.append(result)
        
        
        non_none_count = sum(1 for r in results if r is not None)
        assert non_none_count > 0, f"Stream should produce some non-None values. Results: {results}"
        
        
        for i, result in enumerate(results):
            if result is not None:
                assert result >= 0, f"UI at index {i} should be non-negative, got {result}"
                assert result < 1000, f"UI at index {i} should be reasonable, got {result}"
    
    def test_ui_nan_handling(self, test_data):
        """Test UI handles NaN values correctly"""
        close = test_data['close'].copy()
        
        
        close[50:55] = np.nan
        
        
        result = ta_indicators.ui(close, period=14, scalar=100.0)
        assert len(result) == len(close)
        
        
        valid_count = np.sum(~np.isnan(result[60:]))
        assert valid_count > 0, "Should have valid values after NaN region"


if __name__ == "__main__":
    pytest.main([__file__])