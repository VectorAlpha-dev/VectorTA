"""
Python binding tests for VWMACD indicator.
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


class TestVwmacd:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_vwmacd_partial_params(self, test_data):
        """Test VWMACD with default parameters - mirrors check_vwmacd_partial_params"""
        close = test_data['close']
        volume = test_data['volume']
        
        # Test with all default params
        macd, signal, hist = ta_indicators.vwmacd(
            close, volume, 
            fast_period=12, slow_period=26, signal_period=9
        )
        assert len(macd) == len(close)
        assert len(signal) == len(close)
        assert len(hist) == len(close)
    
    def test_vwmacd_accuracy(self, test_data):
        """Test VWMACD matches expected values from Rust tests - mirrors check_vwmacd_accuracy"""
        close = test_data['close']
        volume = test_data['volume']
        
        macd, signal, hist = ta_indicators.vwmacd(
            close, volume,
            fast_period=12, slow_period=26, signal_period=9,
            fast_ma_type="sma", slow_ma_type="sma", signal_ma_type="ema"
        )
        
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
            125.15561462
        ]
        
        # Check last 5 values match expected
        assert_close(
            macd[-5:], 
            expected_macd,
            rtol=1e-3,
            msg="VWMACD MACD last 5 values mismatch"
        )
        
        assert_close(
            signal[-5:], 
            expected_signal,
            rtol=1e-3,
            msg="VWMACD Signal last 5 values mismatch"
        )
        
        assert_close(
            hist[-5:], 
            expected_histogram,
            rtol=1e-3,
            msg="VWMACD Histogram last 5 values mismatch"
        )
    
    def test_vwmacd_with_custom_ma_types(self, test_data):
        """Test VWMACD with custom MA types - mirrors check_vwmacd_with_custom_ma_types"""
        close = test_data['close']
        volume = test_data['volume']
        
        # Test with custom MA types
        macd1, signal1, hist1 = ta_indicators.vwmacd(
            close, volume,
            fast_period=12, slow_period=26, signal_period=9,
            fast_ma_type="ema", slow_ma_type="wma", signal_ma_type="sma"
        )
        
        # Test with default MA types
        macd2, signal2, hist2 = ta_indicators.vwmacd(
            close, volume,
            fast_period=12, slow_period=26, signal_period=9,
            fast_ma_type="sma", slow_ma_type="sma", signal_ma_type="ema"
        )
        
        # Results should be different
        # Skip the first 50 values to avoid warmup period
        different_count = 0
        for i in range(50, len(macd1)):
            if not np.isnan(macd1[i]) and not np.isnan(macd2[i]):
                if abs(macd1[i] - macd2[i]) > 1e-10:
                    different_count += 1
        
        assert different_count > 0, "Custom MA types should produce different results"
    
    def test_vwmacd_nan_data(self):
        """Test VWMACD with all NaN data - mirrors check_vwmacd_nan_data"""
        close = np.array([np.nan, np.nan])
        volume = np.array([np.nan, np.nan])
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.vwmacd(close, volume, 12, 26, 9)
    
    def test_vwmacd_zero_period(self):
        """Test VWMACD fails with zero period - mirrors check_vwmacd_zero_period"""
        close = np.array([10.0, 20.0, 30.0])
        volume = np.array([1.0, 1.0, 1.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.vwmacd(close, volume, fast_period=0, slow_period=26, signal_period=9)
    
    def test_vwmacd_period_exceeds(self):
        """Test VWMACD fails when period exceeds data length - mirrors check_vwmacd_period_exceeds"""
        close = np.array([10.0, 20.0, 30.0])
        volume = np.array([100.0, 200.0, 300.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.vwmacd(close, volume, fast_period=12, slow_period=26, signal_period=9)
    
    def test_vwmacd_mismatched_lengths(self):
        """Test VWMACD fails when close and volume have different lengths"""
        close = np.array([10.0, 20.0, 30.0])
        volume = np.array([100.0, 200.0])
        
        with pytest.raises(ValueError, match="Close and volume arrays must have the same length"):
            ta_indicators.vwmacd(close, volume, 2, 3, 2)
    
    def test_vwmacd_batch_single_param(self, test_data):
        """Test VWMACD batch with single parameter set"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        volume = test_data['volume'][:100]
        
        result = ta_indicators.vwmacd_batch(
            close, volume,
            fast_range=(12, 12, 0),
            slow_range=(26, 26, 0),
            signal_range=(9, 9, 0)
        )
        
        # Should match single calculation
        macd, signal, hist = ta_indicators.vwmacd(close, volume, 12, 26, 9)
        
        assert 'values' in result
        assert 'fast_periods' in result
        assert 'slow_periods' in result
        assert 'signal_periods' in result
        
        values = result['values']
        assert values.shape == (1, 100)
        assert_close(values[0], macd, rtol=1e-10, msg="Batch vs single mismatch")
    
    def test_vwmacd_batch_multiple_params(self, test_data):
        """Test VWMACD batch with multiple parameter combinations"""
        close = test_data['close'][:50]  # Use smaller dataset for speed
        volume = test_data['volume'][:50]
        
        result = ta_indicators.vwmacd_batch(
            close, volume,
            fast_range=(10, 14, 2),    # 10, 12, 14
            slow_range=(20, 26, 3),    # 20, 23, 26
            signal_range=(5, 9, 2)     # 5, 7, 9
        )
        
        # Should have 3 * 3 * 3 = 27 combinations
        values = result['values']
        assert values.shape == (27, 50)
        
        fast_periods = result['fast_periods']
        slow_periods = result['slow_periods']
        signal_periods = result['signal_periods']
        
        assert len(fast_periods) == 27
        assert len(slow_periods) == 27
        assert len(signal_periods) == 27
        
        # Verify first combination
        assert fast_periods[0] == 10
        assert slow_periods[0] == 20
        assert signal_periods[0] == 5
    
    def test_vwmacd_stream(self):
        """Test VWMACD streaming functionality"""
        # Create stream
        stream = ta_indicators.VwmacdStream(
            fast_period=12, 
            slow_period=26, 
            signal_period=9
        )
        
        # Update with values (returns None for now as not implemented)
        result = stream.update(100.0, 1000.0)
        assert result is None