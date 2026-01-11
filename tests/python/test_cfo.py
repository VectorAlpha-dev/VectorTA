"""
Python binding tests for CFO indicator.
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


class TestCfo:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_cfo_partial_params(self, test_data):
        """Test CFO with partial parameters (None values) - mirrors check_cfo_partial_params"""
        close = test_data['close']
        
        
        result = ta_indicators.cfo(close, 14, 100.0)  
        assert len(result) == len(close)
    
    def test_cfo_accuracy(self, test_data):
        """Test CFO matches expected values from Rust tests - mirrors check_cfo_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['cfo']
        
        result = ta_indicators.cfo(
            close,
            period=expected['default_params']['period'],
            scalar=expected['default_params']['scalar']
        )
        
        assert len(result) == len(close)
        
        
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-6,
            msg="CFO last 5 values mismatch"
        )
        
        
        compare_with_rust('cfo', result, 'close', expected['default_params'])
    
    def test_cfo_default_candles(self, test_data):
        """Test CFO with default parameters - mirrors check_cfo_default_candles"""
        close = test_data['close']
        
        
        result = ta_indicators.cfo(close, 14, 100.0)
        assert len(result) == len(close)
    
    def test_cfo_zero_period(self):
        """Test CFO fails with zero period - mirrors check_cfo_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.cfo(input_data, period=0, scalar=100.0)
    
    def test_cfo_period_exceeds_length(self):
        """Test CFO fails when period exceeds data length - mirrors check_cfo_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.cfo(data_small, period=10, scalar=100.0)
    
    def test_cfo_very_small_dataset(self):
        """Test CFO fails with insufficient data - mirrors check_cfo_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.cfo(single_point, period=14, scalar=100.0)
    
    def test_cfo_empty_input(self):
        """Test CFO fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="No data provided"):
            ta_indicators.cfo(empty, period=14, scalar=100.0)
    
    def test_cfo_reinput(self, test_data):
        """Test CFO applied twice (re-input) - mirrors check_cfo_reinput"""
        close = test_data['close']
        
        
        first_result = ta_indicators.cfo(close, period=14, scalar=100.0)
        assert len(first_result) == len(close)
        
        
        second_result = ta_indicators.cfo(first_result, period=14, scalar=100.0)
        assert len(second_result) == len(first_result)
        
        
        if len(second_result) > 240:
            assert not np.any(np.isnan(second_result[240:])), "Found unexpected NaN after warmup period"
    
    def test_cfo_nan_handling(self, test_data):
        """Test CFO handles NaN values correctly - mirrors check_cfo_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.cfo(close, period=14, scalar=100.0)
        assert len(result) == len(close)
        
        
        first_valid = next((i for i, v in enumerate(close) if not np.isnan(v)), 0)
        warmup_period = first_valid + 14 - 1
        
        
        assert np.all(np.isnan(result[:warmup_period])), f"Expected NaN in warmup period [0:{warmup_period})"
        
        
        if len(result) > warmup_period:
            
            assert not np.all(np.isnan(result[warmup_period:])), "Expected some valid values after warmup"
    
    def test_cfo_streaming(self, test_data):
        """Test CFO streaming matches batch calculation - mirrors check_cfo_streaming"""
        close = test_data['close']
        period = 14
        scalar = 100.0
        
        
        batch_result = ta_indicators.cfo(close, period=period, scalar=scalar)
        
        
        stream = ta_indicators.CfoStream(period=period, scalar=scalar)
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
                        msg=f"CFO streaming mismatch at index {i}")
    
    def test_cfo_batch(self, test_data):
        """Test CFO batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        result = ta_indicators.cfo_batch(
            close,
            period_range=(14, 14, 0),  
            scalar_range=(100.0, 100.0, 0.0)  
        )
        
        assert isinstance(result, dict)
        assert 'values' in result
        assert 'periods' in result
        assert 'scalars' in result
        
        
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        
        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['cfo']['last_5_values']
        
        
        assert_close(
            default_row[-5:],
            expected,
            rtol=1e-6,
            msg="CFO batch default row mismatch"
        )
    
    def test_cfo_all_nan_input(self):
        """Test CFO with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.cfo(all_nan, period=14, scalar=100.0)

    def test_cfo_batch_sweep(self, test_data):
        """Test CFO batch with parameter sweep"""
        close = test_data['close'][:100]  
        
        result = ta_indicators.cfo_batch(
            close,
            period_range=(10, 20, 5),  
            scalar_range=(50.0, 150.0, 50.0)  
        )
        
        
        assert result['values'].shape[0] == 9
        assert result['values'].shape[1] == len(close)
        assert len(result['periods']) == 9
        assert len(result['scalars']) == 9
        
        
        expected_periods = [10, 10, 10, 15, 15, 15, 20, 20, 20]
        expected_scalars = [50.0, 100.0, 150.0] * 3
        
        np.testing.assert_array_equal(result['periods'], expected_periods)
        np.testing.assert_array_almost_equal(result['scalars'], expected_scalars, decimal=10)

    def test_cfo_with_kernel(self, test_data):
        """Test CFO with different kernel selections"""
        close = test_data['close'][:100]  
        
        
        result_scalar = ta_indicators.cfo(close, 14, 100.0, kernel="scalar")
        assert len(result_scalar) == len(close)
        
        
        result_auto = ta_indicators.cfo(close, 14, 100.0)
        assert len(result_auto) == len(close)
        
        
        assert_close(result_scalar, result_auto, rtol=1e-10, atol=1e-10,
                    msg="Kernel results should match")

    def test_cfo_constant_values(self, test_data):
        """Test CFO with constant input values"""
        constant = np.full(50, 42.0)
        
        result = ta_indicators.cfo(constant, period=14, scalar=100.0)
        assert len(result) == len(constant)
        
        
        
        non_nan_values = result[~np.isnan(result)]
        assert np.all(np.abs(non_nan_values) < 1e-10), "CFO should be ~0 for constant series"

    def test_cfo_linear_trend(self):
        """Test CFO with perfect linear trend"""
        
        x = np.arange(100, dtype=float)
        
        result = ta_indicators.cfo(x, period=14, scalar=100.0)
        assert len(result) == len(x)
        
        
        
        non_nan_values = result[~np.isnan(result)]
        assert np.all(np.abs(non_nan_values) < 1e-10), "CFO should be ~0 for perfect linear trend"

    def test_cfo_scalar_edge_cases(self):
        """Test CFO with edge case scalar values"""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        
        
        result = ta_indicators.cfo(data, period=2, scalar=float('nan'))
        assert np.all(np.isnan(result)), "NaN scalar should produce all NaN output"
        
        
        result = ta_indicators.cfo(data, period=2, scalar=float('inf'))
        
        non_warmup = result[1:]  
        assert np.all(np.isnan(non_warmup) | np.isinf(non_warmup)), \
            "Infinite scalar should produce inf/nan output"
        
        
        result = ta_indicators.cfo(data, period=2, scalar=0.0)
        non_warmup = result[1:]
        assert np.all(non_warmup == 0.0), "Zero scalar should produce zero output"
    
    def test_cfo_negative_scalar(self, test_data):
        """Test CFO with negative scalar values (should work)"""
        close = test_data['close'][:100]
        
        
        result_pos = ta_indicators.cfo(close, period=14, scalar=100.0)
        result_neg = ta_indicators.cfo(close, period=14, scalar=-100.0)
        
        assert len(result_pos) == len(result_neg)
        
        
        for i, (pos, neg) in enumerate(zip(result_pos, result_neg)):
            if np.isnan(pos) and np.isnan(neg):
                continue
            assert_close(-pos, neg, rtol=1e-10, atol=1e-10,
                        msg=f"Negative scalar mismatch at index {i}")
    
    def test_cfo_warmup_period(self, test_data):
        """Test CFO warmup period calculation is correct"""
        close = test_data['close']
        
        
        for period in [5, 10, 14, 20, 30]:
            result = ta_indicators.cfo(close, period=period, scalar=100.0)
            
            
            first_valid = next((i for i, v in enumerate(close) if not np.isnan(v)), 0)
            expected_warmup = first_valid + period - 1
            
            
            assert np.all(np.isnan(result[:expected_warmup])), \
                f"Period {period}: Expected NaN before index {expected_warmup}"
            
            
            if expected_warmup < len(result):
                assert not np.isnan(result[expected_warmup]), \
                    f"Period {period}: Expected valid value at index {expected_warmup}"
    
    def test_cfo_edge_values(self):
        """Test CFO with edge case values in data"""
        
        data_with_zero = np.array([10.0, 20.0, 30.0, 0.0, 40.0, 50.0, 60.0])
        result = ta_indicators.cfo(data_with_zero, period=3, scalar=100.0)
        
        
        assert np.isnan(result[3]), "Expected NaN when current value is 0"
        
        
        data_with_inf = np.array([10.0, 20.0, 30.0, np.inf, 40.0, 50.0, 60.0])
        result = ta_indicators.cfo(data_with_inf, period=3, scalar=100.0)
        
        
        assert np.isnan(result[3]), "Expected NaN when current value is inf"
        
        
        data_small = np.array([1e-10, 2e-10, 3e-10, 4e-10, 5e-10])
        result = ta_indicators.cfo(data_small, period=2, scalar=100.0)
        assert len(result) == len(data_small)
        assert not np.all(np.isnan(result)), "Should handle very small values"
    
    def test_cfo_batch_invalid_params(self, test_data):
        """Test CFO batch with invalid parameter ranges"""
        close = test_data['close'][:100]
        
        
        with pytest.raises(ValueError, match="Invalid|period"):
            ta_indicators.cfo_batch(
                close,
                period_range=(20, 10, 1),  
                scalar_range=(100.0, 100.0, 0.0)
            )
        
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.cfo_batch(
                close,
                period_range=(0, 0, 0),  
                scalar_range=(100.0, 100.0, 0.0)
            )
    
    def test_cfo_batch_edge_cases(self, test_data):
        """Test CFO batch with edge case parameter combinations"""
        close = test_data['close'][:50]
        
        
        result = ta_indicators.cfo_batch(
            close,
            period_range=(14, 14, 0),
            scalar_range=(100.0, 100.0, 0.0)
        )
        assert result['values'].shape[0] == 1, "Should have 1 combination"
        
        
        result = ta_indicators.cfo_batch(
            close,
            period_range=(14, 14, 0),
            scalar_range=(-100.0, 100.0, 100.0)  
        )
        assert result['values'].shape[0] == 3, "Should have 3 scalar values"
        
        
        row_neg = result['values'][0]  
        row_pos = result['values'][2]  
        
        for i in range(len(close)):
            if np.isnan(row_neg[i]) and np.isnan(row_pos[i]):
                continue
            if not np.isnan(row_neg[i]) and not np.isnan(row_pos[i]):
                assert_close(-row_neg[i], row_pos[i], rtol=1e-10, atol=1e-10,
                            msg=f"Batch negative scalar mismatch at index {i}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])