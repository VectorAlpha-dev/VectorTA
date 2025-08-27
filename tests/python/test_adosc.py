"""
Python binding tests for ADOSC indicator.
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


class TestAdosc:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_adosc_partial_params(self, test_data):
        """Test ADOSC with partial parameters - mirrors check_adosc_partial_params"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        # Test with short period only (default long = 10)
        result = ta_indicators.adosc(high, low, close, volume, 2, 10)
        assert len(result) == len(close)
        
        # Test with long period only (default short = 3)
        result2 = ta_indicators.adosc(high, low, close, volume, 3, 12)
        assert len(result2) == len(close)
    
    def test_adosc_accuracy(self, test_data):
        """Test ADOSC matches expected values from Rust tests - mirrors check_adosc_accuracy"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        result = ta_indicators.adosc(
            high, low, close, volume,
            short_period=3,  # Default short period
            long_period=10   # Default long period
        )
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected from Rust tests
        expected_last_five = [-166.2175, -148.9983, -144.9052, -128.5921, -142.0772]
        assert_close(
            result[-5:], 
            expected_last_five,
            rtol=1e-1,  # Using same tolerance as Rust test
            msg="ADOSC last 5 values mismatch"
        )
        
        # All values should be finite
        assert np.all(np.isfinite(result)), "All ADOSC values should be finite"
        
        # ADOSC has no warmup period - first value should be calculated
        assert not np.isnan(result[0]), "First ADOSC value should not be NaN (no warmup period)"
    
    def test_adosc_default_candles(self, test_data):
        """Test ADOSC with default parameters - mirrors check_adosc_default_candles"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        # Default params: short_period=3, long_period=10
        result = ta_indicators.adosc(high, low, close, volume, 3, 10)
        assert len(result) == len(close)
    
    def test_adosc_zero_period(self):
        """Test ADOSC fails with zero period - mirrors check_adosc_zero_period"""
        high = np.array([10.0, 10.0, 10.0])
        low = np.array([5.0, 5.0, 5.0])
        close = np.array([7.0, 7.0, 7.0])
        volume = np.array([1000.0, 1000.0, 1000.0])
        
        # Zero short period
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.adosc(high, low, close, volume, short_period=0, long_period=10)
        
        # Zero long period
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.adosc(high, low, close, volume, short_period=3, long_period=0)
    
    def test_adosc_period_exceeds_length(self):
        """Test ADOSC fails when period exceeds data length - mirrors check_adosc_period_exceeds_length"""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([5.0, 5.5, 6.0])
        close = np.array([7.0, 8.0, 9.0])
        volume = np.array([1000.0, 1000.0, 1000.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.adosc(high, low, close, volume, short_period=3, long_period=10)
    
    def test_adosc_very_small_dataset(self):
        """Test ADOSC fails with insufficient data - mirrors check_adosc_very_small_dataset"""
        high = np.array([10.0])
        low = np.array([5.0])
        close = np.array([7.0])
        volume = np.array([1000.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.adosc(high, low, close, volume, short_period=3, long_period=10)
    
    def test_adosc_empty_input(self):
        """Test ADOSC fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="empty"):
            ta_indicators.adosc(empty, empty, empty, empty, short_period=3, long_period=10)
    
    def test_adosc_mismatched_lengths(self):
        """Test ADOSC fails when input arrays have different lengths"""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([5.0, 5.5])  # Different length
        close = np.array([7.0, 8.0, 9.0])
        volume = np.array([1000.0, 1000.0, 1000.0])
        
        with pytest.raises(ValueError, match="All input arrays must have the same length"):
            ta_indicators.adosc(high, low, close, volume, short_period=2, long_period=3)
    
    def test_adosc_short_greater_than_long(self):
        """Test ADOSC fails when short period >= long period"""
        high = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        low = np.array([5.0, 5.5, 6.0, 6.5, 7.0])
        close = np.array([7.0, 8.0, 9.0, 10.0, 11.0])
        volume = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
        
        # short = long
        with pytest.raises(ValueError, match="short_period must be less than long_period"):
            ta_indicators.adosc(high, low, close, volume, short_period=3, long_period=3)
        
        # short > long
        with pytest.raises(ValueError, match="short_period must be less than long_period"):
            ta_indicators.adosc(high, low, close, volume, short_period=5, long_period=3)
    
    def test_adosc_all_nan_input(self):
        """Test ADOSC with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.adosc(all_nan, all_nan, all_nan, all_nan, short_period=3, long_period=10)
    
    def test_adosc_nan_handling(self, test_data):
        """Test ADOSC handles data correctly - mirrors check_adosc_nan_handling"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        result = ta_indicators.adosc(high, low, close, volume, short_period=3, long_period=10)
        assert len(result) == len(close)
        
        # ADOSC starts calculating from index 0 (no warmup period)
        # After index 240, no NaN values should exist
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after index 240"
        
        # First value should not be NaN (ADOSC calculates from the start)
        assert not np.isnan(result[0]), "First ADOSC value should not be NaN"
    
    def test_adosc_streaming(self, test_data):
        """Test ADOSC streaming matches batch calculation - mirrors check_adosc_streaming"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        short_period = 3
        long_period = 10
        
        # Batch calculation
        batch_result = ta_indicators.adosc(
            high, low, close, volume,
            short_period=short_period,
            long_period=long_period
        )
        
        # Streaming calculation
        stream = ta_indicators.AdoscStream(short_period=short_period, long_period=long_period)
        stream_values = []
        
        for h, l, c, v in zip(high, low, close, volume):
            result = stream.update(h, l, c, v)
            stream_values.append(result)
        
        stream_values = np.array(stream_values)
        
        # Compare batch vs streaming
        assert len(batch_result) == len(stream_values)
        
        # Compare values - should be identical
        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            assert_close(b, s, rtol=1e-9, atol=1e-9, 
                        msg=f"ADOSC streaming mismatch at index {i}")
    
    def test_adosc_batch(self, test_data):
        """Test ADOSC batch processing - mirrors check_batch_default_row"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        result = ta_indicators.adosc_batch(
            high, low, close, volume,
            short_period_range=(3, 3, 0),  # Default short period only
            long_period_range=(10, 10, 0)  # Default long period only
        )
        
        assert 'values' in result
        assert 'shorts' in result
        assert 'longs' in result
        
        # Should have 1 combination (default params)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        # Extract the single row
        default_row = result['values'][0]
        
        # Should match single calculation
        single_result = ta_indicators.adosc(high, low, close, volume, short_period=3, long_period=10)
        assert_close(default_row, single_result, rtol=1e-9, msg="ADOSC batch default row mismatch")
    
    def test_adosc_batch_sweep(self, test_data):
        """Test ADOSC batch processing with parameter sweep"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        result = ta_indicators.adosc_batch(
            high, low, close, volume,
            short_period_range=(2, 5, 1),  # 2, 3, 4, 5
            long_period_range=(8, 12, 2)   # 8, 10, 12
        )
        
        # Check structure
        assert 'values' in result
        assert 'shorts' in result
        assert 'longs' in result
        
        # Should have valid combinations only (where short < long)
        # Valid: (2,8), (2,10), (2,12), (3,8), (3,10), (3,12), (4,8), (4,10), (4,12), (5,8), (5,10), (5,12)
        # But (5,8) is invalid since we need short < long, so 11 valid combinations
        valid_count = sum(1 for s in [2, 3, 4, 5] for l in [8, 10, 12] if s < l)
        assert result['values'].shape[0] == valid_count
        assert result['values'].shape[1] == len(close)
        
        # Verify periods match
        assert len(result['shorts']) == valid_count
        assert len(result['longs']) == valid_count
    
    def test_adosc_zero_volume(self):
        """Test ADOSC with zero volume"""
        high = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        low = np.array([5.0, 5.5, 6.0, 6.5, 7.0])
        close = np.array([7.0, 8.0, 9.0, 10.0, 11.0])
        volume = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # All zero volume
        
        result = ta_indicators.adosc(high, low, close, volume, short_period=2, long_period=3)
        assert len(result) == len(close)
        
        # With zero volume, ADOSC should be 0
        assert_close(result, np.zeros_like(result), rtol=1e-9, msg="ADOSC with zero volume should be 0")
    
    def test_adosc_constant_price(self):
        """Test ADOSC with constant price (high = low = close)"""
        price = 10.0
        high = np.full(10, price)
        low = np.full(10, price)
        close = np.full(10, price)
        volume = np.array([1000.0] * 10)
        
        result = ta_indicators.adosc(high, low, close, volume, short_period=3, long_period=5)
        assert len(result) == len(close)
        
        # With constant price (high = low), MFM is undefined (0/0), should be treated as 0
        # So ADOSC should be 0
        assert_close(result, np.zeros_like(result), rtol=1e-9, msg="ADOSC with constant price should be 0")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])