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
    # If not in virtual environment, try to import from installed location
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
        
        # Test with default parameter (20)
        result = ta_indicators.reflex(close, 20)
        assert len(result) == len(close)
    
    def test_reflex_accuracy(self, test_data):
        """Test Reflex matches expected values from Rust tests - mirrors check_reflex_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS.get('reflex', {})
        
        # Test with period=20 (default)
        result = ta_indicators.reflex(close, period=20)
        
        assert len(result) == len(close)
        
        # Expected values from Rust test (check_reflex_accuracy and check_batch_default_row)
        expected_last_five = [
            0.8085220962465361,
            0.445264715886137,
            0.13861699036615063,
            -0.03598639652007061,
            -0.224906760543743
        ]
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected_last_five,
            rtol=1e-7,
            msg="Reflex last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('reflex', result, 'close', {'period': 20})
    
    def test_reflex_default_candles(self, test_data):
        """Test Reflex with default parameters - mirrors check_reflex_default_candles"""
        close = test_data['close']
        
        # Default param: period=20
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
        
        # After warmup period, no NaN values should exist (except from NaN propagation)
        if len(result) > period:
            # Check values after period are finite
            for i in range(period, len(result)):
                if not np.isnan(close[i]):
                    assert np.isfinite(result[i]), f"Found unexpected non-finite value at index {i}"
        
        # First period values should be 0.0 (Reflex specific warmup behavior)
        assert np.all(result[:period] == 0.0), "Expected zeros in warmup period"
    
    def test_reflex_streaming(self, test_data):
        """Test Reflex streaming matches batch calculation - mirrors check_reflex_streaming"""
        close = test_data['close']
        period = 14
        
        # Batch calculation
        batch_result = ta_indicators.reflex(close, period=period)
        
        # Streaming calculation
        stream = ta_indicators.ReflexStream(period=period)
        stream_values = []
        
        for price in close:
            result = stream.update(price)
            # Reflex returns 0.0 during warmup, not None
            stream_values.append(result if result is not None else 0.0)
        
        stream_values = np.array(stream_values)
        
        # Compare batch vs streaming
        assert len(batch_result) == len(stream_values)
        
        # Compare values
        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            assert_close(b, s, rtol=1e-9, atol=1e-9, 
                        msg=f"Reflex streaming mismatch at index {i}")
    
    def test_reflex_batch(self, test_data):
        """Test Reflex batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        # Test with default period only
        result = ta_indicators.reflex_batch(
            close,
            (20, 20, 0)  # Default period only
        )
        
        assert 'values' in result
        assert 'periods' in result
        assert result['values'].shape == (1, len(close))
        
        # Values should match single calculation
        single_result = ta_indicators.reflex(close, period=20)
        assert_close(
            result['values'][0], 
            single_result,
            rtol=1e-9,
            msg="Reflex batch vs single mismatch"
        )
        
        # Check last 5 values match expected from Rust
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
        
        # Test period range
        batch_result = ta_indicators.reflex_batch(
            close,
            (10, 30, 5)  # period range: 10, 15, 20, 25, 30
        )
        
        # Check result is a dict with the expected keys
        assert isinstance(batch_result, dict)
        assert 'values' in batch_result
        assert 'periods' in batch_result
        
        # Should have 5 period combinations
        assert batch_result['values'].shape == (5, len(close))
        assert len(batch_result['periods']) == 5
        assert np.array_equal(batch_result['periods'], [10, 15, 20, 25, 30])
        
        # Verify first combination matches individual calculation
        individual_result = ta_indicators.reflex(close, 10)
        batch_first = batch_result['values'][0]
        
        # Compare after warmup period
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
        # Test with all NaN data
        all_nan = np.full(100, np.nan)
        with pytest.raises(ValueError, match="All values.*NaN"):
            ta_indicators.reflex_batch(all_nan, (10, 20, 5))
        
        # Test with insufficient data
        small_data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Not enough data"):
            ta_indicators.reflex_batch(small_data, (10, 20, 5))
    
    def test_reflex_edge_cases(self, test_data):
        """Test Reflex with edge case inputs"""
        # Test with monotonically increasing data
        data = np.arange(1.0, 101.0)
        result = ta_indicators.reflex(data, 20)
        assert len(result) == len(data)
        assert np.all(result[:20] == 0.0)  # Warmup period has zeros
        assert np.all(np.isfinite(result[20:]))  # After period
        
        # Test with constant values
        data = np.array([50.0] * 100)
        result = ta_indicators.reflex(data, 20)
        assert len(result) == len(data)
        # First period values should be zeros
        assert np.all(result[:20] == 0.0)
        # With constant input, Reflex produces NaN after warmup (division by zero variance)
        # This is expected behavior
        
        # Test with oscillating values
        data = np.array([10.0, 20.0, 10.0, 20.0] * 25)
        result = ta_indicators.reflex(data, 20)
        assert len(result) == len(data)
        assert np.all(result[:20] == 0.0)  # Warmup
        assert np.all(np.isfinite(result[20:]))
    
    def test_reflex_warmup_behavior(self, test_data):
        """Test Reflex warmup period behavior"""
        close = test_data['close']
        period = 20
        
        result = ta_indicators.reflex(close, period)
        
        # Values during warmup should be zeros (Reflex specific behavior)
        assert np.all(result[:period] == 0.0), "Expected zeros during warmup period"
        
        # Find first non-NaN value in input
        first_valid = np.where(~np.isnan(close))[0][0] if np.any(~np.isnan(close)) else 0
        warmup = first_valid + period
        
        # Values after warmup should be finite
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
        # Create simple test data with known pattern
        data = np.array([10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 10.0, 11.0, 12.0, 13.0] * 5)
        period = 5
        
        result = ta_indicators.reflex(data, period)
        
        # Result length should match input
        assert len(result) == len(data)
        
        # Warmup period should have zeros
        assert np.all(result[:period] == 0.0)
        
        # After warmup, values should be finite
        assert np.all(np.isfinite(result[period:]))
        
        # Reflex should detect the oscillating pattern
        # Values should not all be the same after warmup
        unique_values = np.unique(result[period:])
        assert len(unique_values) > 1, "Expected varying values for oscillating input"


if __name__ == '__main__':
    # Run a simple test to verify the module loads correctly
    print("Testing Reflex module...")
    test = TestReflex()
    test_data = load_test_data()
    test.test_reflex_accuracy(test_data)
    print("Reflex tests passed!")