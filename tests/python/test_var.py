"""
Python binding tests for VAR indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'target/wheels'))

try:
    import my_project as ta_indicators
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close


class TestVar:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_var_accuracy(self, test_data):
        """Test VAR matches expected values from Rust tests"""
        close_prices = test_data['close']
        period = 14
        nbdev = 1.0
        
        
        result = ta_indicators.var(close_prices, period, nbdev)
        
        assert len(result) == len(close_prices), "VAR length mismatch"
        
        
        expected_last_five = [
            350987.4081501961,
            348493.9183540344,
            302611.06121110916,
            106092.2499871254,
            121941.35202789307,
        ]
        
        
        assert len(result) >= 5, "VAR length too short"
        result_last_five = result[-5:]
        
        for i, (actual, expected) in enumerate(zip(result_last_five, expected_last_five)):
            assert abs(actual - expected) < 1e-1, f"VAR mismatch at index {i}: expected {expected}, got {actual}"
    
    def test_var_partial_params(self, test_data):
        """Test VAR with partial parameters"""
        close_prices = test_data['close']
        
        
        result_default = ta_indicators.var(close_prices)
        assert len(result_default) == len(close_prices)
        
        
        result_period_20 = ta_indicators.var(close_prices, 20)
        assert len(result_period_20) == len(close_prices)
        
        
        result_nbdev_2 = ta_indicators.var(close_prices, 14, 2.0)
        assert len(result_nbdev_2) == len(close_prices)
    
    def test_var_errors(self):
        """Test error handling"""
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.var(np.array([10.0, 20.0, 30.0]), 0)
        
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.var(np.array([10.0, 20.0, 30.0]), 10)
        
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.var(np.array([42.0]), 14)
        
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.var(np.array([]))
        
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.var(np.array([np.nan, np.nan, np.nan]), 14)
        
        
        with pytest.raises(ValueError, match="nbdev is NaN"):
            ta_indicators.var(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 2, np.nan)
        
        
        with pytest.raises(ValueError, match="nbdev is NaN"):
            ta_indicators.var(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 2, np.inf)
    
    def test_var_stream(self):
        """Test VarStream class"""
        stream = ta_indicators.VarStream(14, 1.0)
        
        
        results = []
        test_values = [100.0, 101.0, 99.5, 102.0, 100.5, 101.5, 99.0, 102.5, 101.0, 100.0,
                      101.5, 99.5, 102.0, 100.5, 101.0]
        
        for val in test_values:
            result = stream.update(val)
            results.append(result)
        
        
        for i in range(13):
            assert results[i] is None, f"Expected None at index {i}, got {results[i]}"
        
        
        assert results[13] is not None, "Expected value at index 13"
        assert results[14] is not None, "Expected value at index 14"
    
    def test_var_batch(self, test_data):
        """Test batch processing"""
        close_prices = test_data['close']
        
        
        batch_result = ta_indicators.var_batch(
            close_prices,
            period_range=(10, 20, 5),  
            nbdev_range=(1.0, 1.0, 0.0)  
        )
        
        assert 'values' in batch_result
        assert 'periods' in batch_result
        assert 'nbdevs' in batch_result
        
        values = batch_result['values']
        periods = batch_result['periods']
        nbdevs = batch_result['nbdevs']
        
        
        assert len(periods) == 3
        assert list(periods) == [10, 15, 20]
        assert list(nbdevs) == [1.0, 1.0, 1.0]
        
        
        assert values.shape == (3, len(close_prices))
        
        
        batch_result2 = ta_indicators.var_batch(
            close_prices,
            period_range=(14, 14, 0),  
            nbdev_range=(0.5, 2.0, 0.5)  
        )
        
        periods2 = batch_result2['periods']
        nbdevs2 = batch_result2['nbdevs']
        
        
        assert len(periods2) == 4
        assert list(periods2) == [14, 14, 14, 14]
        assert len(nbdevs2) == 4
        
        expected_nbdevs = [0.5, 1.0, 1.5, 2.0]
        for actual, expected in zip(nbdevs2, expected_nbdevs):
            assert abs(actual - expected) < 1e-10
    
    def test_var_nan_handling(self, test_data):
        """Test handling of NaN values"""
        close_prices = test_data['close']
        period = 14
        
        result = ta_indicators.var(close_prices, period)
        
        
        for i in range(period - 1):
            assert np.isnan(result[i]), f"Expected NaN at index {i}"
        
        
        if len(result) > 30:
            for i in range(30, len(result)):
                assert not np.isnan(result[i]), f"Unexpected NaN at index {i}"
    
    def test_var_kernel_selection(self, test_data):
        """Test with different kernel selections"""
        close_prices = test_data['close']
        
        
        result_auto = ta_indicators.var(close_prices, kernel=None)
        assert len(result_auto) == len(close_prices)
        
        
        result_scalar = ta_indicators.var(close_prices, kernel="scalar")
        assert len(result_scalar) == len(close_prices)
        
        
        for i, (auto, scalar) in enumerate(zip(result_auto, result_scalar)):
            if not (np.isnan(auto) and np.isnan(scalar)):
                assert abs(auto - scalar) < 1e-10, f"Kernel mismatch at index {i}"