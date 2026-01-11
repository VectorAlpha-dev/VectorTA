import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from pathlib import Path
import sys


try:
    import my_project as ta
except ImportError:
    
    try:
        import my_project as ta
    except ImportError:
        pytest.skip("my_project module not available. Run 'maturin develop --features python' first.", allow_module_level=True)

from test_utils import load_test_data

@pytest.fixture(scope="module")
def test_data():
    data = load_test_data()
    return data['close']

class TestER:
    """Test cases for ER (Kaufman Efficiency Ratio) indicator.
    These tests mirror the Rust unit tests to ensure Python bindings work correctly.
    """
    
    def test_er_accuracy(self, test_data):
        """Test ER matches expected values from Rust tests - mirrors check_er_accuracy"""
        from test_utils import EXPECTED_OUTPUTS
        
        expected = EXPECTED_OUTPUTS['er']
        result = ta.er(test_data, period=expected['default_params']['period'])
        
        assert len(result) == len(test_data)
        
        
        assert_array_almost_equal(
            result[-5:], 
            expected['last_5_values'],
            decimal=8,
            err_msg="ER last 5 values mismatch"
        )
        
        
        assert_array_almost_equal(
            result[100:105],
            expected['values_at_100_104'],
            decimal=8,
            err_msg="ER values at indices 100-104 mismatch"
        )
    
    def test_er_partial_params(self, test_data):
        """Test ER with partial parameters - mirrors check_er_partial_params"""
        
        result = ta.er(test_data, period=5)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(test_data)
    
    def test_er_default_candles(self, test_data):
        """Test ER with default parameters - mirrors check_er_default_candles"""
        
        result = ta.er(test_data, period=5)
        assert len(result) == len(test_data)
        
        
        first_valid = np.where(~np.isnan(test_data))[0][0] if np.any(~np.isnan(test_data)) else 0
        warmup_end = first_valid + 5 - 1
        
        
        assert all(np.isnan(result[:warmup_end]))
    
    def test_er_zero_period(self, test_data):
        """Test ER fails with zero period - mirrors check_er_zero_period"""
        with pytest.raises(ValueError, match="Invalid period"):
            ta.er(test_data, period=0)
    
    def test_er_period_exceeds_length(self):
        """Test ER fails when period exceeds data length - mirrors check_er_period_exceeds_length"""
        small_data = np.array([10.0, 20.0, 30.0])
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta.er(small_data, period=10)
    
    def test_er_very_small_dataset(self):
        """Test ER fails with insufficient data - mirrors check_er_very_small_dataset"""
        single_point = np.array([42.0])
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta.er(single_point, period=5)
    
    def test_er_empty_input(self):
        """Test ER fails with empty input - mirrors check_er_empty_input"""
        empty = np.array([])
        with pytest.raises(ValueError, match="Input data slice is empty|All values are NaN"):
            ta.er(empty, period=5)
    
    def test_er_all_nan_input(self):
        """Test ER fails with all NaN values"""
        all_nan = np.full(100, np.nan)
        with pytest.raises(ValueError, match="All input data values are NaN"):
            ta.er(all_nan, period=5)
    
    def test_er_nan_handling(self, test_data):
        """Test ER handles NaN values correctly - mirrors check_er_nan_handling"""
        result = ta.er(test_data, period=5)
        assert len(result) == len(test_data)
        
        
        if len(result) > 240:
            assert not any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"
        
        
        first_valid = np.where(~np.isnan(test_data))[0][0] if np.any(~np.isnan(test_data)) else 0
        warmup_end = first_valid + 5 - 1
        assert all(np.isnan(result[:warmup_end])), "Expected NaN in warmup period"
    
    def test_er_reinput(self, test_data):
        """Test ER applied twice (re-input)"""
        
        first_result = ta.er(test_data, period=5)
        assert len(first_result) == len(test_data)
        
        
        second_result = ta.er(first_result, period=5)
        assert len(second_result) == len(first_result)
        
        
        valid_values = second_result[~np.isnan(second_result)]
        assert all(v >= 0.0 and v <= 1.0 for v in valid_values)
    
    def test_er_streaming_vs_batch(self, test_data):
        """Test that streaming ER matches batch calculation"""
        period = 5
        
        
        batch_result = ta.er(test_data, period=period)
        
        
        stream = ta.ErStream(period=period)
        stream_result = []
        
        for value in test_data:
            result = stream.update(value)
            if result is not None:
                stream_result.append(result)
            else:
                stream_result.append(np.nan)
        
        
        assert_array_almost_equal(batch_result, stream_result, decimal=10)
    
    def test_er_batch_single_period(self, test_data):
        """Test batch ER with single period - mirrors check_batch_default_row"""
        result = ta.er_batch(test_data, period_range=(5, 5, 0))
        
        assert 'values' in result
        assert 'periods' in result
        assert 'rows' in result  
        assert 'cols' in result
        
        assert result['rows'] == 1
        assert result['cols'] == len(test_data)
        assert result['values'].shape == (1, len(test_data))  
        assert list(result['periods']) == [5]
        
        
        single_result = ta.er(test_data, period=5)
        assert_array_almost_equal(result['values'][0], single_result, decimal=10)
    
    def test_er_batch_multiple_periods(self, test_data):
        """Test batch ER with multiple periods"""
        
        data_subset = test_data[:100]
        
        
        result = ta.er_batch(data_subset, period_range=(5, 15, 5))
        
        assert result['rows'] == 3
        assert result['cols'] == 100
        assert result['values'].shape == (3, 100)  
        assert list(result['periods']) == [5, 10, 15]
        
        
        for i, period in enumerate([5, 10, 15]):
            row_data = result['values'][i]
            
            single_result = ta.er(data_subset, period=period)
            assert_array_almost_equal(row_data, single_result, decimal=10)
    
    def test_er_batch_edge_cases(self):
        """Test edge cases for batch processing"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
        
        
        result = ta.er_batch(data, period_range=(5, 5, 1))
        assert result['values'].shape == (1, 10)  
        assert list(result['periods']) == [5]
        
        
        result = ta.er_batch(data, period_range=(5, 7, 10))
        
        assert result['values'].shape == (1, 10)  
        assert list(result['periods']) == [5]
    
    def test_er_with_kernel_parameter(self, test_data):
        """Test ER with different kernel parameters"""
        
        result_auto = ta.er(test_data, period=5)
        
        
        result_scalar = ta.er(test_data, period=5, kernel="scalar")
        
        
        assert_array_almost_equal(result_auto, result_scalar, decimal=8)
    
    def test_er_consistency(self):
        """Test ER produces consistent results for known data"""
        from test_utils import EXPECTED_OUTPUTS
        expected = EXPECTED_OUTPUTS['er']
        
        
        trending_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
        result = ta.er(trending_data, period=5)
        
        
        valid_values = result[4:]  
        assert_array_almost_equal(
            valid_values,
            expected['trending_data_values'],
            decimal=10,
            err_msg="ER trending data mismatch"
        )
        
        
        choppy_data = np.array([1, 5, 2, 6, 3, 7, 4, 8, 5, 9], dtype=np.float64)
        result = ta.er(choppy_data, period=5)
        
        
        valid_values = result[4:]  
        assert_array_almost_equal(
            valid_values,
            expected['choppy_data_values'],
            decimal=10,
            err_msg="ER choppy data mismatch"
        )