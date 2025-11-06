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
        expected = EXPECTED_OUTPUTS['coppock']
        
        # Use default parameters from expected outputs
        result = ta_indicators.coppock(
            close, 
            expected['default_params']['short'],
            expected['default_params']['long'],
            expected['default_params']['ma']
        )
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        for i, expected_val in enumerate(expected['last_5_values']):
            assert_close(result[-(5-i)], expected_val, rtol=1e-8)
    
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
        
        short = 11
        long = 14
        ma = 10
        
        result = ta_indicators.coppock(close, short, long, ma)
        assert len(result) == len(close)
        
        # Warmup period: max(short, long) + (ma - 1)
        warmup = max(short, long) + (ma - 1)  # 14 + 9 = 23
        
        # After warmup period, no NaN values should exist
        if len(result) > warmup:
            non_nan_portion = result[warmup:]
            assert not np.any(np.isnan(non_nan_portion)), f"Found NaN after warmup period {warmup}"
    
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
        
        # Allow tiny FP differences between batch and single.
        # Use absolute tolerance tighter than Rust's 1e-7 reference threshold.
        np.testing.assert_allclose(batch_values, single_result, rtol=0.0, atol=1e-9)
    
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
        assert 'shorts' in batch_result
        assert 'longs' in batch_result
        assert 'ma_periods' in batch_result
        
        # Should have 2 * 2 * 2 = 8 combinations
        assert batch_result['values'].shape == (8, 100)
        assert len(batch_result['shorts']) == 8
        assert len(batch_result['longs']) == 8
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
        """Test COPPOCK fails with invalid MA type"""
        close = test_data['close']
        
        # Should raise an error for unknown MA type
        with pytest.raises(Exception, match="Unknown moving average type|Underlying MA error"):
            ta_indicators.coppock(close, 11, 14, 10, "invalid_ma")
    
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
    
    def test_coppock_warmup_period(self, test_data):
        """Test COPPOCK warmup period matches expected formula"""
        close = test_data['close']
        
        short = 11
        long = 14
        ma = 10
        
        result = ta_indicators.coppock(close, short, long, ma)
        
        # Warmup period formula: first_valid + max(short, long) + (ma_period - 1)
        # For real data, first_valid is typically 0
        expected_warmup = max(short, long) + (ma - 1)
        
        # Check NaN pattern in warmup period
        for i in range(min(expected_warmup - 1, len(result))):
            assert np.isnan(result[i]), f"Expected NaN at index {i} during warmup"
        
        # Should have valid values after warmup
        if len(result) > expected_warmup:
            assert not np.isnan(result[expected_warmup]), f"Expected valid value at index {expected_warmup}"
    
    def test_coppock_supported_ma_types(self, test_data):
        """Test COPPOCK with supported MA types"""
        close = test_data['close'][:100]  # Use smaller dataset
        
        # Only test MA types that are actually supported
        supported_ma_types = ['sma', 'ema', 'wma']
        
        results = {}
        for ma_type in supported_ma_types:
            result = ta_indicators.coppock(close, 11, 14, 10, ma_type)
            assert len(result) == len(close)
            # Each MA type should produce different results
            assert not np.all(np.isnan(result))
            results[ma_type] = result
        
        # Verify different MA types produce different results
        assert not np.allclose(results['sma'], results['ema'], equal_nan=True)
        assert not np.allclose(results['wma'], results['ema'], equal_nan=True)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
