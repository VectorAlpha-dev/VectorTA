"""
Python binding tests for SuperSmoother indicator.
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


class TestSuperSmoother:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_supersmoother_partial_params(self, test_data):
        """Test SuperSmoother with partial parameters - mirrors check_supersmoother_partial_params"""
        close = test_data['close']
        
        
        result = ta_indicators.supersmoother(close, 14)  
        assert len(result) == len(close)
    
    def test_supersmoother_accuracy(self, test_data):
        """Test SuperSmoother matches expected values from Rust tests - mirrors check_supersmoother_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['supersmoother']
        
        result = ta_indicators.supersmoother(
            close,
            period=expected['default_params']['period']
        )
        
        assert len(result) == len(close)
        
        
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-8,
            msg="SuperSmoother last 5 values mismatch"
        )
        
        
        compare_with_rust('supersmoother', result, 'close', expected['default_params'])
    
    def test_supersmoother_default_candles(self, test_data):
        """Test SuperSmoother with default parameters - mirrors check_supersmoother_default_candles"""
        close = test_data['close']
        
        
        result = ta_indicators.supersmoother(close, 14)
        assert len(result) == len(close)
    
    def test_supersmoother_zero_period(self):
        """Test SuperSmoother fails with zero period - mirrors check_supersmoother_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.supersmoother(input_data, period=0)
    
    def test_supersmoother_period_exceeds_length(self):
        """Test SuperSmoother fails when period exceeds data length - mirrors check_supersmoother_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.supersmoother(data_small, period=10)
    
    def test_supersmoother_very_small_dataset(self):
        """Test SuperSmoother fails with insufficient data - mirrors check_supersmoother_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.supersmoother(single_point, period=14)
    
    def test_supersmoother_empty_input(self):
        """Test SuperSmoother fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.supersmoother(empty, period=14)
    
    def test_supersmoother_reinput(self, test_data):
        """Test SuperSmoother applied twice (re-input) - mirrors check_supersmoother_reinput"""
        close = test_data['close']
        
        
        first_result = ta_indicators.supersmoother(close, period=14)
        assert len(first_result) == len(close)
        
        
        second_result = ta_indicators.supersmoother(first_result, period=10)
        assert len(second_result) == len(first_result)
        
        
        
    
    def test_supersmoother_nan_handling(self, test_data):
        """Test SuperSmoother handles NaN values correctly - mirrors check_supersmoother_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.supersmoother(close, period=14)
        assert len(result) == len(close)
        
        
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"
        
        
        assert np.all(np.isnan(result[:13])), "Expected NaN in warmup period"
        
        
        assert not np.isnan(result[13]), "Value at index 13 should be initialized"
        if len(result) > 14:
            assert not np.isnan(result[14]), "Value at index 14 should be initialized"
    
    def test_supersmoother_streaming(self, test_data):
        """Test SuperSmoother streaming matches batch calculation - mirrors check_supersmoother_streaming"""
        close = test_data['close']
        period = 14
        
        
        batch_result = ta_indicators.supersmoother(close, period=period)
        
        
        stream = ta_indicators.SuperSmootherStream(period=period)
        stream_values = []
        
        for price in close:
            result = stream.update(price)
            
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        
        assert len(batch_result) == len(stream_values)
        
        
        
        
        
        
        
        valid_count = np.sum(~np.isnan(stream_values))
        assert valid_count > 0, "Streaming should produce some valid values"
    
    def test_supersmoother_batch(self, test_data):
        """Test SuperSmoother batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        
        result = ta_indicators.supersmoother_batch(
            close,
            period_start=14,
            period_end=14,
            period_step=0
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        assert result['periods'][0] == 14
        
        
        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['supersmoother']['last_5_values']
        
        
        assert_close(
            default_row[-5:],
            expected,
            rtol=1e-8,
            msg="SuperSmoother batch default row mismatch"
        )
    
    def test_supersmoother_batch_multiple_periods(self, test_data):
        """Test SuperSmoother batch with multiple periods"""
        close = test_data['close']
        
        
        result = ta_indicators.supersmoother_batch(
            close,
            period_start=10,
            period_end=20,
            period_step=5
        )
        
        
        assert len(result['periods']) == 3
        assert list(result['periods']) == [10, 15, 20]
        
        
        assert result['values'].shape == (3, len(close))
        
        
        for i, period in enumerate(result['periods']):
            individual = ta_indicators.supersmoother(close, period)
            
            first_valid = next((j for j, v in enumerate(individual) if not np.isnan(v)), 0)
            assert_close(
                result['values'][i][first_valid:],
                individual[first_valid:],
                msg=f"Batch result for period {period} doesn't match individual"
            )
    
    def test_supersmoother_batch_edge_cases(self, test_data):
        """Test SuperSmoother batch edge cases"""
        close = test_data['close'][:50]  
        
        
        result = ta_indicators.supersmoother_batch(close, 10, 10, 0)
        assert result['values'].shape[0] == 1
        assert result['periods'][0] == 10
        
        
        result = ta_indicators.supersmoother_batch(close, 10, 12, 5)
        assert result['values'].shape[0] == 1  
        assert result['periods'][0] == 10
    
    def test_supersmoother_all_nan_input(self):
        """Test SuperSmoother with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.supersmoother(all_nan, period=14)
    
    def test_supersmoother_leading_nans(self):
        """Test handling of leading NaN values"""
        
        data = np.empty(20)
        data[:5] = np.nan
        for i in range(5, 20):
            data[i] = i - 4  
        
        period = 3
        result = ta_indicators.supersmoother(data, period)
        
        
        
        
        
        
        
        
        for i in range(5):
            assert np.isnan(result[i]), f"Expected NaN at index {i} where input is NaN"
        
        
        for i in range(5, 7):
            assert np.isnan(result[i]), f"Expected NaN at index {i} due to warmup"
        
        
        assert result[7] == data[7], f"Expected initial value at index 7"
        assert result[8] == data[8], f"Expected initial value at index 8"
    
    def test_supersmoother_consistency(self):
        """Test that multiple runs produce identical results"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        
        result1 = ta_indicators.supersmoother(data, 3)
        result2 = ta_indicators.supersmoother(data, 3)
        
        
        for i in range(len(result1)):
            if np.isnan(result1[i]) and np.isnan(result2[i]):
                continue
            assert result1[i] == result2[i], f"Inconsistent result at index {i}"
    
    def test_supersmoother_kernel_selection(self, test_data):
        """Test different kernel options"""
        close = test_data['close']
        
        
        result_scalar = ta_indicators.supersmoother(close, 14, kernel='scalar')
        assert len(result_scalar) == len(close)
        
        
        result_auto = ta_indicators.supersmoother(close, 14, kernel='auto')
        assert len(result_auto) == len(close)
        
        
        assert_close(
            result_scalar[-5:], 
            result_auto[-5:],
            rtol=1e-9,
            msg="Scalar and auto kernels produce different results"
        )
    
    def test_supersmoother_error_conditions(self):
        """Test various error conditions"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        
        with pytest.raises(ValueError):
            ta_indicators.supersmoother(data, 3, kernel='invalid')
    
    def test_supersmoother_compare_with_rust(self, test_data):
        """Compare Python results with Rust implementation"""
        close = test_data['close']
        
        
        result = ta_indicators.supersmoother(close, 14)
        
        
        compare_with_rust('supersmoother', result, 'close', {'period': 14})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])