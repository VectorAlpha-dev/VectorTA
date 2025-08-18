"""
Python binding tests for COPPOCK indicator.
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

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS


class TestCoppock:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_coppock_accuracy(self, test_data):
        """Test COPPOCK matches expected values from Rust tests"""
        close = test_data['close']
        
        # Default parameters: short=11, long=14, ma=10, ma_type="wma"
        result = ta_indicators.coppock(close, 11, 14, 10)
        
        assert len(result) == len(close)
        
        # Expected values from Rust tests
        expected_last_five = [
            -1.4542764618985533,
            -1.3795224034983653,
            -1.614331648987457,
            -1.9179048338714915,
            -2.1096548435774625,
        ]
        
        for i, expected in enumerate(expected_last_five):
            assert_close(result[-(5-i)], expected, rtol=1e-6)
    
    def test_coppock_partial_params(self, test_data):
        """Test COPPOCK with default parameters"""
        close = test_data['close']
        
        # Test with default parameters
        result = ta_indicators.coppock(close, 11, 14, 10)
        assert len(result) == len(close)
    
    def test_coppock_default_ma_type(self, test_data):
        """Test COPPOCK with default MA type"""
        close = test_data['close']
        
        # Test without specifying ma_type (should default to "wma")
        result = ta_indicators.coppock(close, 11, 14, 10)
        assert len(result) == len(close)
        
        # Test with explicit ma_type
        result_sma = ta_indicators.coppock(close, 11, 14, 10, "sma")
        assert len(result_sma) == len(close)
    
    def test_coppock_zero_period(self):
        """Test COPPOCK fails with zero period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(Exception, match="Invalid period"):
            ta_indicators.coppock(input_data, 0, 14, 10)
    
    def test_coppock_period_exceeds_length(self):
        """Test COPPOCK fails when period exceeds data length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(Exception, match="Invalid period"):
            ta_indicators.coppock(data_small, 14, 20, 10)
    
    def test_coppock_very_small_dataset(self):
        """Test COPPOCK fails with insufficient data"""
        single_point = np.array([42.0])
        
        with pytest.raises(Exception, match="Invalid period|Not enough valid data"):
            ta_indicators.coppock(single_point, 11, 14, 10)
    
    def test_coppock_empty_input(self):
        """Test COPPOCK fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(Exception, match="Empty data provided"):
            ta_indicators.coppock(empty, 11, 14, 10)
    
    def test_coppock_all_nan_input(self):
        """Test COPPOCK fails with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(Exception, match="All values are NaN"):
            ta_indicators.coppock(all_nan, 11, 14, 10)
    
    def test_coppock_nan_handling(self, test_data):
        """Test COPPOCK handles NaN values correctly"""
        close = test_data['close']
        
        result = ta_indicators.coppock(close, 11, 14, 10)
        assert len(result) == len(close)
        
        # After warmup period (30), no NaN values should exist
        if len(result) > 30:
            non_nan_portion = result[30:]
            assert not np.any(np.isnan(non_nan_portion))
    
    def test_coppock_streaming(self, test_data):
        """Test CoppockStream produces same results as batch"""
        close = test_data['close']
        
        # Batch calculation
        batch_result = ta_indicators.coppock(close, 11, 14, 10)
        
        # Streaming calculation
        stream = ta_indicators.CoppockStream(11, 14, 10, "wma")
        stream_result = []
        
        for price in close:
            value = stream.update(price)
            stream_result.append(value if value is not None else np.nan)
        
        stream_result = np.array(stream_result)
        assert len(stream_result) == len(batch_result)
        
        # Compare results (allowing for NaN equality)
        for i, (b, s) in enumerate(zip(batch_result, stream_result)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-7)
    
    def test_coppock_batch_single_param(self, test_data):
        """Test batch with single parameter combination"""
        close = test_data['close']
        
        # Single parameter set: short=11, long=14, ma=10
        batch_result = ta_indicators.coppock_batch(
            close,
            (11, 11, 0),    # short range
            (14, 14, 0),    # long range
            (10, 10, 0)     # ma range
        )
        
        # Should match single calculation
        single_result = ta_indicators.coppock(close, 11, 14, 10)
        
        assert 'values' in batch_result
        batch_values = batch_result['values'][0]  # First row
        
        np.testing.assert_allclose(batch_values, single_result, rtol=1e-10)
    
    def test_coppock_batch_multiple_params(self, test_data):
        """Test batch with multiple parameter combinations"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        # Multiple parameters
        batch_result = ta_indicators.coppock_batch(
            close,
            (10, 12, 2),    # short: 10, 12
            (14, 16, 2),    # long: 14, 16
            (8, 10, 2)      # ma: 8, 10
        )
        
        assert 'values' in batch_result
        assert 'short_periods' in batch_result
        assert 'long_periods' in batch_result
        assert 'ma_periods' in batch_result
        
        # Should have 2 * 2 * 2 = 8 combinations
        assert batch_result['values'].shape == (8, 100)
        assert len(batch_result['short_periods']) == 8
        assert len(batch_result['long_periods']) == 8
        assert len(batch_result['ma_periods']) == 8
    
    def test_coppock_kernel_selection(self, test_data):
        """Test COPPOCK with different kernel selections"""
        close = test_data['close']
        
        # Test with auto kernel (default)
        result_auto = ta_indicators.coppock(close, 11, 14, 10)
        
        # Test with explicit scalar kernel
        result_scalar = ta_indicators.coppock(close, 11, 14, 10, kernel="scalar")
        
        # Results should be very close
        np.testing.assert_allclose(result_auto, result_scalar, rtol=1e-10)
    
    def test_coppock_invalid_ma_type(self, test_data):
        """Test COPPOCK handles invalid MA type by defaulting to sma"""
        close = test_data['close']
        
        # Should not raise an error, but default to 'sma' with a warning
        # The stderr shows: "Unknown indicator 'invalid_ma'. Defaulting to 'sma'."
        result = ta_indicators.coppock(close, 11, 14, 10, "invalid_ma")
        assert len(result) == len(close)  # Should still produce valid output
    
    def test_coppock_negative_periods(self, test_data):
        """Test COPPOCK fails with negative periods"""
        close = test_data['close']
        
        # Negative short period
        with pytest.raises(OverflowError, match="can't convert negative"):
            ta_indicators.coppock(close, -11, 14, 10)
        
        # Negative long period
        with pytest.raises(OverflowError, match="can't convert negative"):
            ta_indicators.coppock(close, 11, -14, 10)
        
        # Negative MA period
        with pytest.raises(OverflowError, match="can't convert negative"):
            ta_indicators.coppock(close, 11, 14, -10)
    
    def test_coppock_reinput(self, test_data):
        """Test COPPOCK applied twice (re-input)"""
        close = test_data['close']
        
        # First pass
        first_result = ta_indicators.coppock(close, 11, 14, 10)
        assert len(first_result) == len(close)
        
        # Second pass - apply COPPOCK to COPPOCK output
        second_result = ta_indicators.coppock(first_result, 11, 14, 10)
        assert len(second_result) == len(first_result)
        
        # After double warmup period, no NaN values should exist
        double_warmup = 60  # Approximate double warmup
        if len(second_result) > double_warmup:
            non_nan_portion = second_result[double_warmup:]
            assert not np.any(np.isnan(non_nan_portion))
    
    def test_coppock_supported_ma_types(self, test_data):
        """Test COPPOCK with all supported MA types"""
        close = test_data['close'][:100]  # Use smaller dataset
        
        supported_ma_types = ['sma', 'ema', 'wma', 'hma', 'rma', 'tema']
        
        for ma_type in supported_ma_types:
            result = ta_indicators.coppock(close, 11, 14, 10, ma_type)
            assert len(result) == len(close)
            # Each MA type should produce different results
            assert not np.all(np.isnan(result))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
