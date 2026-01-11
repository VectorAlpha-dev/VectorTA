"""
Python binding tests for HMA indicator.
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


class TestHma:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_hma_partial_params(self, test_data):
        """Test HMA with partial parameters - mirrors check_hma_partial_params"""
        close = test_data['close']
        
        
        result = ta_indicators.hma(close, 5)
        assert len(result) == len(close)
    
    def test_hma_accuracy(self, test_data):
        """Test HMA matches expected values from Rust tests - mirrors check_hma_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['hma']
        
        
        result = ta_indicators.hma(close, expected['default_params']['period'])
        
        assert len(result) == len(close)
        
        
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-12,
            atol=1e-3,  
            msg="HMA last 5 values mismatch"
        )
        
        
        
    
    def test_hma_default_candles(self, test_data):
        """Test HMA with default parameters - mirrors check_hma_default_candles"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['hma']
        
        
        result = ta_indicators.hma(close, expected['default_params']['period'])
        assert len(result) == len(close)
        
        
        
    
    def test_hma_zero_period(self):
        """Test HMA fails with zero period - mirrors check_hma_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.hma(input_data, period=0)
    
    def test_hma_period_exceeds_length(self):
        """Test HMA fails when period exceeds data length - mirrors check_hma_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.hma(data_small, period=10)
    
    def test_hma_very_small_dataset(self):
        """Test HMA with very small dataset - mirrors check_hma_very_small_dataset"""
        data_single = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.hma(data_single, period=5)
    
    def test_hma_empty_input(self):
        """Test HMA with empty input - mirrors check_hma_empty_input"""
        data_empty = np.array([])
        
        with pytest.raises(ValueError, match="No data provided|All values are NaN"):
            ta_indicators.hma(data_empty, period=5)
    
    def test_hma_all_nan(self):
        """Test HMA with all NaN input - mirrors check_hma_all_nan"""
        data = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.hma(data, period=3)
    
    def test_hma_nan_handling(self, test_data):
        """Test HMA handling of NaN values - mirrors check_hma_nan_handling"""
        close = test_data['close']
        period = 5
        
        result = ta_indicators.hma(close, period)
        
        assert len(result) == len(close)
        
        
        sqrt_period = int(np.sqrt(period))
        warmup_period = period + sqrt_period - 2
        
        
        if len(result) > warmup_period:
            for i in range(warmup_period, len(result)):
                assert not np.isnan(result[i]), f"Unexpected NaN at index {i}"
    
    def test_hma_warmup_period(self, test_data):
        """Test HMA warmup period calculation is correct"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['hma']
        
        
        test_cases = [
            {'period': 3, 'expected_warmup': 3 + 1 - 2},  
            {'period': 5, 'expected_warmup': 5 + 2 - 2},  
            {'period': 10, 'expected_warmup': 10 + 3 - 2}, 
            {'period': 16, 'expected_warmup': 16 + 4 - 2}, 
        ]
        
        for test_case in test_cases:
            period = test_case['period']
            expected_warmup = test_case['expected_warmup']
            
            result = ta_indicators.hma(close[:50], period)  
            
            
            for i in range(min(expected_warmup, len(result))):
                assert np.isnan(result[i]), f"Expected NaN at index {i} for period={period}"
            
            
            if expected_warmup < len(result):
                assert not np.isnan(result[expected_warmup]), \
                    f"Expected valid value at index {expected_warmup} for period={period}"
    
    def test_hma_batch(self, test_data):
        """Test HMA batch computation."""
        close = test_data['close']
        
        
        period_range = (3, 9, 2)  
        
        result = ta_indicators.hma_batch(close, period_range)
        
        assert isinstance(result, dict)
        assert 'values' in result
        assert 'periods' in result
        
        values = result['values']
        periods = result['periods']
        
        expected_periods = [3, 5, 7, 9]
        
        assert list(periods) == expected_periods
        assert values.shape == (4, len(close))  
        
        
        row_idx = 0
        for period in [3, 5, 7, 9]:
            individual_result = ta_indicators.hma(close, period)
            np.testing.assert_allclose(
                values[row_idx], 
                individual_result, 
                rtol=1e-9,
                err_msg=f"Batch row {row_idx} (period={period}) mismatch"
            )
            row_idx += 1
    
    def test_hma_batch_single_period(self, test_data):
        """Test HMA batch with single period matches single computation"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['hma']
        
        
        period_range = (5, 5, 0)
        
        batch_result = ta_indicators.hma_batch(close, period_range)
        single_result = ta_indicators.hma(close, 5)
        
        assert batch_result['values'].shape == (1, len(close))
        assert list(batch_result['periods']) == [5]
        
        np.testing.assert_allclose(
            batch_result['values'][0],
            single_result,
            rtol=1e-10,
            err_msg="Batch single period mismatch"
        )
        
        
        assert_close(
            batch_result['values'][0, -5:],
            expected['last_5_values'],
            rtol=1e-12,
            atol=1e-3,  
            msg="HMA batch last 5 values mismatch"
        )
    
    def test_hma_different_periods(self, test_data):
        """Test HMA with different period values."""
        close = test_data['close']
        
        
        for period in [3, 5, 10, 20]:
            result = ta_indicators.hma(close, period)
            assert len(result) == len(close)
            
            
            sqrt_period = int(np.sqrt(period))
            warmup = period + sqrt_period - 2
            
            
            for i in range(warmup, len(result)):
                assert not np.isnan(result[i]), f"Unexpected NaN at index {i} for period={period}"
    
    def test_hma_zero_half(self):
        """Test HMA fails when period/2 is zero"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        
        with pytest.raises(ValueError, match="Cannot calculate half of period|Invalid period|Zero"):
            ta_indicators.hma(data, period=1)
    
    def test_hma_zero_sqrt_period(self):
        """Test HMA with period where sqrt(period) < 1"""
        
        
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        
        
        result = ta_indicators.hma(data, period=2)
        assert len(result) == len(data)
        
        
        assert np.isnan(result[0])
        assert not np.isnan(result[1])
    
    def test_hma_not_enough_valid_data(self):
        """Test HMA with insufficient valid data after NaN prefix"""
        data = np.array([np.nan, np.nan, 1.0, 2.0, 3.0])
        
        
        
        with pytest.raises(ValueError, match="Not enough valid data"):
            ta_indicators.hma(data, period=4)
    
    def test_hma_batch_edge_cases(self, test_data):
        """Test HMA batch processing edge cases"""
        close = test_data['close'][:100]  
        
        
        result = ta_indicators.hma_batch(close, (5, 7, 10))
        assert result['values'].shape[0] == 1  
        assert list(result['periods']) == [5]
        
        
        result = ta_indicators.hma_batch(close, (5, 5, 0))
        assert result['values'].shape[0] == 1
        assert list(result['periods']) == [5]
    
    def test_hma_consistency_across_periods(self, test_data):
        """Test that HMA produces consistent results for different periods"""
        
        
        close = test_data['close'][:200]  
        
        hma5 = ta_indicators.hma(close, 5)
        hma10 = ta_indicators.hma(close, 10)
        hma20 = ta_indicators.hma(close, 20)
        
        
        def responsiveness(arr):
            valid_diffs = []
            for i in range(1, len(arr)):
                if not np.isnan(arr[i]) and not np.isnan(arr[i-1]):
                    valid_diffs.append(abs(arr[i] - arr[i-1]))
            return np.mean(valid_diffs) if valid_diffs else 0
        
        resp5 = responsiveness(hma5)
        resp10 = responsiveness(hma10)
        resp20 = responsiveness(hma20)
        
        
        assert resp5 > resp10, \
            f"HMA(5) responsiveness {resp5} should be > HMA(10) responsiveness {resp10}"
        assert resp10 > resp20, \
            f"HMA(10) responsiveness {resp10} should be > HMA(20) responsiveness {resp20}"
        
        
        assert np.sum(~np.isnan(hma5)) > 0, "HMA(5) should produce valid values"
        assert np.sum(~np.isnan(hma10)) > 0, "HMA(10) should produce valid values"
        assert np.sum(~np.isnan(hma20)) > 0, "HMA(20) should produce valid values"
    
    def test_hma_with_specific_data_patterns(self):
        """Test HMA with specific data patterns"""
        
        
        constant_data = np.full(50, 100.0)
        result = ta_indicators.hma(constant_data, 5)
        valid_result = result[~np.isnan(result)]
        np.testing.assert_allclose(valid_result, 100.0, rtol=1e-10,
                                  err_msg="HMA of constant data should equal the constant")
        
        
        linear_data = np.arange(1.0, 51.0)
        result = ta_indicators.hma(linear_data, 5)
        
        
        assert len(result) == len(linear_data)
    
    def test_hma_error_messages(self):
        """Test that HMA produces appropriate error messages"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        
        test_cases = [
            ([], 5, "No data provided"),
            ([np.nan] * 5, 3, "All values are NaN"),
            (data, 0, "Invalid period"),
            (data, 10, "Invalid period"),  
            ([42.0], 5, "Not enough valid data|Invalid period"),
        ]
        
        for input_data, period, expected_msg in test_cases:
            with pytest.raises(ValueError, match=expected_msg):
                ta_indicators.hma(np.array(input_data), period)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
