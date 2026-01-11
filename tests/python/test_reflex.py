"""
Python binding tests for Reflex indicator.
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


class TestReflex:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_reflex_partial_params(self, test_data):
        """Test Reflex with default parameters - mirrors check_reflex_partial_params"""
        close = test_data['close']
        
        
        result = ta_indicators.reflex(close, 20)
        assert len(result) == len(close)
    
    def test_reflex_accuracy(self, test_data):
        """Test Reflex matches expected values from Rust tests - mirrors check_reflex_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS.get('reflex', {})
        
        
        result = ta_indicators.reflex(close, period=20)
        
        assert len(result) == len(close)
        
        
        expected_last_five = [
            0.8085220962465361,
            0.445264715886137,
            0.13861699036615063,
            -0.03598639652007061,
            -0.224906760543743
        ]
        
        
        assert_close(
            result[-5:], 
            expected_last_five,
            rtol=1e-7,
            msg="Reflex last 5 values mismatch"
        )
        
        
        compare_with_rust('reflex', result, 'close', {'period': 20})
    
    def test_reflex_default_candles(self, test_data):
        """Test Reflex with default parameters - mirrors check_reflex_default_candles"""
        close = test_data['close']
        
        
        result = ta_indicators.reflex(close, 20)
        assert len(result) == len(close)
    
    def test_reflex_zero_period(self):
        """Test Reflex fails with zero period - mirrors check_reflex_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="period must be >=2"):
            ta_indicators.reflex(input_data, period=0)
    
    def test_reflex_period_less_than_two(self):
        """Test Reflex fails when period < 2 - mirrors check_reflex_period_less_than_two"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="period must be >=2"):
            ta_indicators.reflex(input_data, period=1)
    
    def test_reflex_period_exceeds_length(self):
        """Test Reflex fails when period exceeds data length - mirrors check_reflex_very_small_data_set"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Not enough data"):
            ta_indicators.reflex(data_small, period=10)
    
    def test_reflex_very_small_dataset(self):
        """Test Reflex fails with insufficient data - mirrors check_reflex_very_small_data_set"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Not enough data"):
            ta_indicators.reflex(single_point, period=5)
    
    def test_reflex_empty_input(self):
        """Test Reflex fails with empty input - mirrors empty input test"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="No data available"):
            ta_indicators.reflex(empty, period=20)
    
    def test_reflex_nan_handling(self, test_data):
        """Test Reflex handles NaN values correctly - mirrors check_reflex_nan_handling"""
        close = test_data['close']
        period = 14
        
        result = ta_indicators.reflex(close, period=period)
        assert len(result) == len(close)
        
        
        if len(result) > period:
            
            for i in range(period, len(result)):
                if not np.isnan(close[i]):
                    assert np.isfinite(result[i]), f"Found unexpected non-finite value at index {i}"
        
        
        assert np.all(result[:period] == 0.0), "Expected zeros in warmup period"
    
    def test_reflex_streaming(self, test_data):
        """Test Reflex streaming matches batch calculation - mirrors check_reflex_streaming"""
        close = test_data['close']
        period = 14
        
        
        batch_result = ta_indicators.reflex(close, period=period)
        
        
        stream = ta_indicators.ReflexStream(period=period)
        stream_values = []
        
        for price in close:
            result = stream.update(price)
            
            stream_values.append(result if result is not None else 0.0)
        
        stream_values = np.array(stream_values)
        
        
        assert len(batch_result) == len(stream_values)
        
        
        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            assert_close(b, s, rtol=1e-9, atol=1e-9, 
                        msg=f"Reflex streaming mismatch at index {i}")
    
    def test_reflex_batch(self, test_data):
        """Test Reflex batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        
        result = ta_indicators.reflex_batch(
            close,
            (20, 20, 0)  
        )
        
        assert 'values' in result
        assert 'periods' in result
        assert result['values'].shape == (1, len(close))
        
        
        single_result = ta_indicators.reflex(close, period=20)
        assert_close(
            result['values'][0], 
            single_result,
            rtol=1e-9,
            msg="Reflex batch vs single mismatch"
        )
        
        
        expected_last_five = [
            0.8085220962465361,
            0.445264715886137,
            0.13861699036615063,
            -0.03598639652007061,
            -0.224906760543743
        ]
        
        assert_close(
            result['values'][0][-5:],
            expected_last_five,
            rtol=1e-7,
            msg="Reflex batch last 5 values mismatch"
        )
    
    def test_reflex_batch_multiple_periods(self, test_data):
        """Test Reflex batch computation with multiple periods"""
        close = test_data['close']
        
        
        batch_result = ta_indicators.reflex_batch(
            close,
            (10, 30, 5)  
        )
        
        
        assert isinstance(batch_result, dict)
        assert 'values' in batch_result
        assert 'periods' in batch_result
        
        
        assert batch_result['values'].shape == (5, len(close))
        assert len(batch_result['periods']) == 5
        assert np.array_equal(batch_result['periods'], [10, 15, 20, 25, 30])
        
        
        individual_result = ta_indicators.reflex(close, 10)
        batch_first = batch_result['values'][0]
        
        
        warmup = 10
        assert_close(batch_first[warmup:], individual_result[warmup:], atol=1e-9, 
                    msg="Reflex batch first combination mismatch")
    
    def test_reflex_all_nan_input(self):
        """Test Reflex with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values.*NaN"):
            ta_indicators.reflex(all_nan, period=20)
    
    def test_reflex_batch_error_conditions(self):
        """Test Reflex batch error handling"""
        
        all_nan = np.full(100, np.nan)
        with pytest.raises(ValueError, match="All values.*NaN"):
            ta_indicators.reflex_batch(all_nan, (10, 20, 5))
        
        
        small_data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Not enough data"):
            ta_indicators.reflex_batch(small_data, (10, 20, 5))
    
    def test_reflex_edge_cases(self, test_data):
        """Test Reflex with edge case inputs"""
        
        data = np.arange(1.0, 101.0)
        result = ta_indicators.reflex(data, 20)
        assert len(result) == len(data)
        assert np.all(result[:20] == 0.0)  
        assert np.all(np.isfinite(result[20:]))  
        
        
        data = np.array([50.0] * 100)
        result = ta_indicators.reflex(data, 20)
        assert len(result) == len(data)
        
        assert np.all(result[:20] == 0.0)
        
        
        
        
        data = np.array([10.0, 20.0, 10.0, 20.0] * 25)
        result = ta_indicators.reflex(data, 20)
        assert len(result) == len(data)
        assert np.all(result[:20] == 0.0)  
        assert np.all(np.isfinite(result[20:]))
    
    def test_reflex_warmup_behavior(self, test_data):
        """Test Reflex warmup period behavior"""
        close = test_data['close']
        period = 20
        
        result = ta_indicators.reflex(close, period)
        
        
        assert np.all(result[:period] == 0.0), "Expected zeros during warmup period"
        
        
        first_valid = np.where(~np.isnan(close))[0][0] if np.any(~np.isnan(close)) else 0
        warmup = first_valid + period
        
        
        if warmup < len(result):
            assert np.all(np.isfinite(result[warmup:])), "Expected finite values after warmup"
    
    def test_reflex_consistency(self, test_data):
        """Test that Reflex produces consistent results across multiple calls"""
        close = test_data['close'][:100]
        
        result1 = ta_indicators.reflex(close, 20)
        result2 = ta_indicators.reflex(close, 20)
        
        assert_close(result1, result2, atol=1e-15, msg="Reflex results not consistent")
    
    def test_reflex_formula_verification(self):
        """Verify Reflex formula implementation with simple pattern"""
        
        data = np.array([10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 10.0, 11.0, 12.0, 13.0] * 5)
        period = 5
        
        result = ta_indicators.reflex(data, period)
        
        
        assert len(result) == len(data)
        
        
        assert np.all(result[:period] == 0.0)
        
        
        assert np.all(np.isfinite(result[period:]))
        
        
        
        unique_values = np.unique(result[period:])
        assert len(unique_values) > 1, "Expected varying values for oscillating input"


if __name__ == '__main__':
    
    print("Testing Reflex module...")
    test = TestReflex()
    test_data = load_test_data()
    test.test_reflex_accuracy(test_data)
    print("Reflex tests passed!")