"""
Python binding tests for VWMACD indicator.
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


class TestVwmacd:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_vwmacd_accuracy(self, test_data):
        """Test VWMACD matches expected values from Rust tests"""
        close = test_data.close
        volume = test_data.volume
        
        # Test with default parameters
        macd, signal, hist = ta_indicators.vwmacd(close, volume)
        
        assert len(macd) == len(close)
        assert len(signal) == len(close)
        assert len(hist) == len(close)
        
        # Expected values from Rust tests
        expected_macd = [
            -394.95161155,
            -508.29106210,
            -490.70190723,
            -388.94996199,
            -341.13720646,
        ]
        
        expected_signal = [
            -539.48861567,
            -533.24910496,
            -524.73966541,
            -497.58172247,
            -466.29282108,
        ]
        
        expected_histogram = [
            144.53700412,
            24.95804286,
            34.03775818,
            108.63176274,
            125.15561462,
        ]
        
        # Check last 5 values
        for i in range(5):
            assert_close(macd[-5 + i], expected_macd[i], 1e-3, f"MACD at index {i}")
            assert_close(signal[-5 + i], expected_signal[i], 1e-3, f"Signal at index {i}")
            assert_close(hist[-5 + i], expected_histogram[i], 1e-3, f"Histogram at index {i}")
    
    def test_vwmacd_errors(self):
        """Test error handling"""
        # Test with NaN data
        nan_data = np.full(10, np.nan)
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.vwmacd(nan_data, nan_data)
        
        # Test with zero period
        data = np.array([1.0, 2.0, 3.0])
        volume = np.array([100.0, 200.0, 300.0])
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.vwmacd(data, volume, fast_period=0)
        
        # Test with period exceeding data length
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.vwmacd(data, volume, slow_period=10)
    
    def test_vwmacd_custom_params(self, test_data):
        """Test VWMACD with custom parameters"""
        close = test_data.close
        volume = test_data.volume
        
        # Test with custom periods
        macd1, signal1, hist1 = ta_indicators.vwmacd(close, volume, 10, 20, 5)
        assert len(macd1) == len(close)
        
        # Test with custom MA types
        macd2, signal2, hist2 = ta_indicators.vwmacd(
            close, volume,
            fast_ma_type="ema",
            slow_ma_type="wma",
            signal_ma_type="sma"
        )
        assert len(macd2) == len(close)
        
        # Results should be different with different parameters
        assert not np.allclose(macd1, macd2, equal_nan=True)
    
    def test_vwmacd_batch(self, test_data):
        """Test batch processing"""
        close = test_data.close[:100]  # Use smaller dataset for speed
        volume = test_data.volume[:100]
        
        result = ta_indicators.vwmacd_batch(
            close, volume,
            fast_range=(10, 14, 2),
            slow_range=(20, 26, 3),
            signal_range=(5, 9, 2)
        )
        
        assert 'values' in result
        assert 'fast_periods' in result
        assert 'slow_periods' in result
        assert 'signal_periods' in result
        
        # Should have 3 * 3 * 3 = 27 combinations
        expected_combos = 3 * 3 * 3
        assert result['values'].shape[0] == expected_combos
        assert result['values'].shape[1] == len(close)
    
    def test_vwmacd_stream(self, test_data):
        """Test streaming functionality"""
        stream = ta_indicators.VwmacdStream()
        
        # Feed some data
        results = []
        for i in range(50):
            result = stream.update(test_data.close[i], test_data.volume[i])
            if result is not None:
                results.append(result)
        
        # Should eventually produce results
        assert len(results) > 0
    
    def test_vwmacd_kernel_selection(self, test_data):
        """Test different kernel selections"""
        close = test_data.close[:1000]
        volume = test_data.volume[:1000]
        
        # Test auto kernel
        macd_auto, _, _ = ta_indicators.vwmacd(close, volume, kernel='auto')
        
        # Test scalar kernel
        macd_scalar, _, _ = ta_indicators.vwmacd(close, volume, kernel='scalar')
        
        # Results should be close (within floating point tolerance)
        np.testing.assert_allclose(macd_auto, macd_scalar, rtol=1e-10, equal_nan=True)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
