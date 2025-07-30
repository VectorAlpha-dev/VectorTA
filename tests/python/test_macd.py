"""
Python binding tests for MACD indicator.
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

from test_utils import load_test_data, assert_close
from rust_comparison import compare_with_rust


class TestMacd:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_macd_partial_params(self, test_data):
        """Test MACD with default parameters - mirrors check_macd_partial_params"""
        close = test_data['close']
        
        # Test with default params
        macd, signal, hist = ta_indicators.macd(close, 12, 26, 9, "ema")
        assert len(macd) == len(close)
        assert len(signal) == len(close)
        assert len(hist) == len(close)
    
    def test_macd_accuracy(self, test_data):
        """Test MACD matches expected values from Rust tests - mirrors check_macd_accuracy"""
        close = test_data['close']
        
        macd, signal, hist = ta_indicators.macd(close, 12, 26, 9, "ema")
        
        assert len(macd) == len(close)
        assert len(signal) == len(close)
        assert len(hist) == len(close)
        
        # Check last 5 values match expected from Rust tests
        expected_macd = [
            -629.8674025082801,
            -600.2986584356258,
            -581.6188884820076,
            -551.1020443476082,
            -560.798510688488,
        ]
        expected_signal = [
            -721.9744591891067,
            -697.6392990384105,
            -674.4352169271299,
            -649.7685824112256,
            -631.9745680666781,
        ]
        expected_hist = [
            92.10705668082664,
            97.34064060278467,
            92.81632844512228,
            98.6665380636174,
            71.17605737819008,
        ]
        
        assert_close(
            macd[-5:], 
            expected_macd,
            rtol=1e-1,
            msg="MACD last 5 values mismatch"
        )
        assert_close(
            signal[-5:], 
            expected_signal,
            rtol=1e-1,
            msg="Signal last 5 values mismatch"
        )
        assert_close(
            hist[-5:], 
            expected_hist,
            rtol=1e-1,
            msg="Histogram last 5 values mismatch"
        )
    
    def test_macd_zero_period(self):
        """Test MACD fails with zero period - mirrors check_macd_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid"):
            ta_indicators.macd(input_data, 0, 26, 9, "ema")
    
    def test_macd_period_exceeds_length(self):
        """Test MACD fails when period exceeds data length - mirrors check_macd_period_exceeds_length"""
        data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid"):
            ta_indicators.macd(data, 12, 26, 9, "ema")
    
    def test_macd_very_small_dataset(self):
        """Test MACD fails with insufficient data - mirrors check_macd_very_small_dataset"""
        data = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid|Not enough"):
            ta_indicators.macd(data, 12, 26, 9, "ema")
    
    def test_macd_nan_handling(self, test_data):
        """Test MACD NaN handling - mirrors check_macd_nan_handling"""
        close = test_data['close']
        
        macd, signal, hist = ta_indicators.macd(close, 12, 26, 9, "ema")
        n = len(macd)
        
        # After warmup period (approximately 240 values), there should be no NaNs
        if n > 240:
            for i in range(240, n):
                assert not np.isnan(macd[i])
                assert not np.isnan(signal[i])
                assert not np.isnan(hist[i])
    
    def test_macd_streaming(self):
        """Test MACD streaming functionality"""
        stream = ta_indicators.MacdStream(12, 26, 9, "ema")
        
        # Test adding values one by one
        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        result = None
        
        for i, value in enumerate(values):
            result = stream.update(value)
            # Should start returning values after we have enough data
            # Need at least slow_period + signal_period - 1 = 26 + 9 - 1 = 34 values
        
        # After 10 values, we don't have enough data yet
        assert result is None
        
        # Add more values to get results
        for i in range(100):
            result = stream.update(50.0 + i)
        
        # Should have results now
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 3  # (macd, signal, hist)
    
    def test_macd_batch(self, test_data):
        """Test MACD batch processing"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        # Test batch with single parameter set
        result = ta_indicators.macd_batch(
            close,
            (12, 12, 1),  # fast_period_range
            (26, 26, 1),  # slow_period_range
            (9, 9, 1),    # signal_period_range
            "ema"
        )
        
        assert 'macd' in result
        assert 'signal' in result
        assert 'hist' in result
        assert 'fast_periods' in result
        assert 'slow_periods' in result
        assert 'signal_periods' in result
        
        # Should have 1 row for single parameter set
        assert result['macd'].shape == (1, 100)
        assert result['signal'].shape == (1, 100)
        assert result['hist'].shape == (1, 100)
        
        # Compare with single calculation
        single_macd, single_signal, single_hist = ta_indicators.macd(close, 12, 26, 9, "ema")
        
        assert_close(
            result['macd'][0], 
            single_macd, 
            rtol=1e-10, 
            msg='Batch MACD should match single calculation'
        )
        assert_close(
            result['signal'][0], 
            single_signal, 
            rtol=1e-10, 
            msg='Batch signal should match single calculation'
        )
        assert_close(
            result['hist'][0], 
            single_hist, 
            rtol=1e-10, 
            msg='Batch hist should match single calculation'
        )
    
    def test_macd_batch_multiple_params(self, test_data):
        """Test MACD batch with multiple parameter combinations"""
        close = test_data['close'][:50]  # Use smaller dataset for speed
        
        # Multiple parameter combinations
        result = ta_indicators.macd_batch(
            close,
            (10, 14, 2),   # fast: 10, 12, 14
            (24, 28, 2),   # slow: 24, 26, 28
            (8, 10, 1),    # signal: 8, 9, 10
            "ema"
        )
        
        # Should have 3 * 3 * 3 = 27 combinations
        assert result['macd'].shape == (27, 50)
        assert result['signal'].shape == (27, 50)
        assert result['hist'].shape == (27, 50)
        assert len(result['fast_periods']) == 27
        assert len(result['slow_periods']) == 27
        assert len(result['signal_periods']) == 27
        
        # Check first combination matches single calculation
        first_macd, first_signal, first_hist = ta_indicators.macd(close, 10, 24, 8, "ema")
        
        assert_close(
            result['macd'][0], 
            first_macd, 
            rtol=1e-10, 
            msg='First batch row should match single calculation'
        )
    
    def test_macd_kernel_parameter(self, test_data):
        """Test MACD with different kernel parameters"""
        close = test_data['close'][:100]
        
        # Test with scalar kernel
        macd_scalar, signal_scalar, hist_scalar = ta_indicators.macd(close, 12, 26, 9, "ema", "scalar")
        assert len(macd_scalar) == len(close)
        
        # Test with auto kernel (default)
        macd_auto, signal_auto, hist_auto = ta_indicators.macd(close, 12, 26, 9, "ema")
        assert len(macd_auto) == len(close)
        
        # Results should be very close regardless of kernel
        assert_close(macd_scalar, macd_auto, rtol=1e-12)
        assert_close(signal_scalar, signal_auto, rtol=1e-12)
        assert_close(hist_scalar, hist_auto, rtol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])