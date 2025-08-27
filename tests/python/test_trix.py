"""
Python binding tests for TRIX indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import the built module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'target/wheels'))

try:
    import my_project as ta_indicators
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close


class TestTrix:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_trix_accuracy(self, test_data):
        """Test TRIX matches expected values from Rust tests"""
        close_prices = test_data['close']
        period = 18
        
        # Test default parameters
        result = ta_indicators.trix(close_prices, period)
        
        assert len(result) == len(close_prices), "TRIX length mismatch"
        
        # Expected values from Rust tests
        expected_last_five = [-16.03736447, -15.92084231, -15.76171478, -15.53571033, -15.34967155]
        
        # Check last 5 values
        assert len(result) >= 5, "TRIX length too short"
        result_last_five = result[-5:]
        
        for i, (actual, expected) in enumerate(zip(result_last_five, expected_last_five)):
            assert abs(actual - expected) < 1e-6, f"TRIX mismatch at index {i}: expected {expected}, got {actual}"
    
    def test_trix_partial_params(self, test_data):
        """Test TRIX with partial parameters"""
        close_prices = test_data['close']
        
        # Test with default period (should be 18)
        result_default = ta_indicators.trix(close_prices, 18)
        assert len(result_default) == len(close_prices)
        
        # Test with different period
        result_period_14 = ta_indicators.trix(close_prices, 14)
        assert len(result_period_14) == len(close_prices)
        
        # Test with custom period
        result_custom = ta_indicators.trix(close_prices, 20)
        assert len(result_custom) == len(close_prices)
    
    def test_trix_errors(self):
        """Test error handling"""
        # Test zero period
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.trix(np.array([10.0, 20.0, 30.0]), 0)
        
        # Test period exceeds length
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.trix(np.array([10.0, 20.0, 30.0]), 10)
        
        # Test very small dataset
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.trix(np.array([42.0]), 18)
        
        # Test empty data
        with pytest.raises(ValueError, match="Empty"):
            ta_indicators.trix(np.array([]), 18)
        
        # Test all NaN - use enough data points so period check passes
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.trix(np.full(100, np.nan), 18)
    
    def test_trix_stream(self):
        """Test TrixStream class"""
        stream = ta_indicators.TrixStream(18)
        
        # Feed some values
        results = []
        test_values = [100.0, 101.0, 99.5, 102.0, 100.5, 101.5, 99.0, 102.5, 101.0, 100.0]
        
        for val in test_values:
            result = stream.update(val)
            results.append(result)
        
        # First several values should be None until enough data
        # TRIX needs 3*(period-1)+1 values before producing output
        assert results[0] is None
        
    def test_trix_batch(self, test_data):
        """Test batch computation"""
        close_prices = test_data['close']
        
        # Test batch with single parameter
        result = ta_indicators.trix_batch(close_prices, (18, 18, 0))
        assert 'values' in result
        assert 'periods' in result
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close_prices)
        
        # Test batch with range
        result_range = ta_indicators.trix_batch(close_prices, (10, 20, 5))
        assert result_range['values'].shape[0] == 3  # 10, 15, 20
        assert result_range['values'].shape[1] == len(close_prices)
        assert np.array_equal(result_range['periods'], [10, 15, 20])
    
    def test_trix_kernel_selection(self, test_data):
        """Test kernel parameter"""
        close_prices = test_data['close']
        
        # Test with different kernels
        result_auto = ta_indicators.trix(close_prices, 18)
        result_scalar = ta_indicators.trix(close_prices, 18, 'scalar')
        
        # Results should be very close regardless of kernel
        assert_close(result_auto, result_scalar, rtol=1e-10)
    
    def test_trix_reinput(self, test_data):
        """Test TRIX on its own output"""
        close_prices = test_data['close']
        period = 10
        
        # First TRIX calculation
        first_result = ta_indicators.trix(close_prices, period)
        
        # Apply TRIX to its own output
        second_result = ta_indicators.trix(first_result, period)
        
        assert len(first_result) == len(second_result)
        
        # The second result should have more NaN values at the beginning
        first_valid = np.where(~np.isnan(first_result))[0]
        second_valid = np.where(~np.isnan(second_result))[0]
        
        if len(first_valid) > 0 and len(second_valid) > 0:
            assert second_valid[0] > first_valid[0]
    
    def test_trix_nan_handling(self, test_data):
        """Test TRIX handles NaN values correctly"""
        close_prices = test_data['close'].copy()
        
        # Insert some NaN values
        close_prices[100:110] = np.nan
        close_prices[200] = np.nan
        close_prices[300:305] = np.nan
        
        # Should not raise error
        result = ta_indicators.trix(close_prices, 18)
        assert len(result) == len(close_prices)
        
        # TRIX uses triple EMA which propagates NaN through the calculation
        # Once a NaN is encountered, it affects all subsequent values due to the exponential smoothing
        # So we expect NaN to propagate from the first NaN position onwards
        # Check that we have valid values before the first NaN region
        valid_before_first_nan = ~np.isnan(result[90:100])
        assert np.any(valid_before_first_nan), "Should have valid values before first NaN region"
        
        # After NaN is introduced at position 100, all subsequent values should be NaN
        # due to the nature of exponential moving average propagation
        all_nan_after = np.isnan(result[110:])
        assert np.all(all_nan_after), "TRIX should propagate NaN through subsequent calculations"
    
    def test_trix_empty_input(self):
        """Test TRIX with empty input"""
        with pytest.raises(ValueError, match="Empty"):
            ta_indicators.trix(np.array([]), 18)
    
    def test_trix_all_nan_input(self):
        """Test TRIX with all NaN input"""
        all_nan = np.full(100, np.nan)
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.trix(all_nan, 18)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
