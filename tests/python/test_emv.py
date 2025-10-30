"""
Python binding tests for EMV indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
from pathlib import Path

try:
    import my_project as ta_indicators
except ImportError:
    ta_indicators = None
    pytest.skip("Rust module not available", allow_module_level=True)

from test_utils import load_test_data, assert_close
from rust_comparison import compare_with_rust


class TestEmv:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_emv_basic_calculation(self):
        """Test basic EMV calculation - mirrors check_emv_basic_calculation"""
        # Generate simple test data
        high = np.array([10.0, 12.0, 13.0, 15.0, 14.0, 16.0])
        low = np.array([5.0, 7.0, 8.0, 10.0, 11.0, 12.0])
        close = np.array([7.5, 9.0, 10.5, 12.5, 12.5, 14.0])
        volume = np.array([10000.0, 20000.0, 25000.0, 30000.0, 15000.0, 35000.0])
        
        result = ta_indicators.emv(high, low, close, volume)
        
        # Verify output shape
        assert len(result) == len(high)
        
        # First value should be NaN (need previous midpoint)
        assert np.isnan(result[0])
        
        # After first value, should have calculated values
        assert not np.isnan(result[1])
        
        # Test specific calculation for index 1
        # mid[0] = (10 + 5) / 2 = 7.5
        # mid[1] = (12 + 7) / 2 = 9.5
        # range[1] = 12 - 7 = 5
        # br[1] = 20000 / 10000 / 5 = 0.4
        # emv[1] = (9.5 - 7.5) / 0.4 = 5.0
        assert_close(result[1], 5.0, rtol=0.01)
    
    def test_emv_accuracy(self, test_data):
        """Test EMV matches expected values from Rust tests - mirrors check_emv_accuracy"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        result = ta_indicators.emv(high, low, close, volume)
        
        # Check dimensions
        assert len(result) == len(high)
        
        # Verify last 5 values match expected from Rust tests
        expected_last_five = [
            -6488905.579799851,
            2371436.7401001123,
            -3855069.958128531,
            1051939.877943717,
            -8519287.22257077,
        ]
        
        assert_close(
            result[-5:],
            expected_last_five,
            rtol=0.0001,
            msg="EMV last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('emv', result, 'ohlcv')
    
    def test_emv_warmup_period(self, test_data):
        """Test EMV warmup period behavior - mirrors check_emv_nan_handling"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        result = ta_indicators.emv(high, low, close, volume)
        
        # EMV warmup period is 1 (first value is always NaN)
        assert np.isnan(result[0]), "First EMV value should be NaN (warmup)"
        
        # Find first valid data point
        first_valid = None
        for i in range(len(high)):
            if not any(np.isnan([high[i], low[i], volume[i]])):
                first_valid = i
                break
        
        if first_valid is not None:
            # After first valid point, next should have a value
            if first_valid + 1 < len(result):
                assert not np.isnan(result[first_valid + 1]), \
                    f"Expected valid EMV at index {first_valid + 1} after first valid data"
    
    def test_emv_empty_data(self):
        """Test EMV with empty data - mirrors check_emv_empty_data"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="(?i)empty"):
            ta_indicators.emv(empty, empty, empty, empty)
    
    def test_emv_all_nan(self):
        """Test EMV with all NaN values - mirrors check_emv_all_nan"""
        all_nan = np.full(10, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN|AllValuesNaN"):
            ta_indicators.emv(all_nan, all_nan, all_nan, all_nan)
    
    def test_emv_not_enough_data(self):
        """Test EMV with insufficient data - mirrors check_emv_not_enough_data"""
        # EMV needs at least 2 valid points
        high = np.array([10.0, np.nan])
        low = np.array([9.0, np.nan])
        close = np.array([9.5, np.nan])
        volume = np.array([1000.0, np.nan])
        
        with pytest.raises(ValueError, match="(?i)not enough"):
            ta_indicators.emv(high, low, close, volume)
    
    def test_emv_with_kernels(self, test_data):
        """Test EMV with different kernel options"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        # Test with different kernels
        kernels = [None, "scalar", "avx2", "avx512"]
        results = []
        
        for kernel in kernels:
            try:
                if kernel:
                    result = ta_indicators.emv(high, low, close, volume, kernel=kernel)
                else:
                    result = ta_indicators.emv(high, low, close, volume)
                results.append(result)
            except Exception as e:
                # Allow runs without nightly AVX features by skipping when
                # specific kernels aren't compiled or supported in this build.
                emsg = str(e).lower()
                if (
                    "not supported" in emsg
                    or "not available" in emsg
                    or "not compiled" in emsg
                ):
                    continue
                raise
        
        # All results should be close
        for i in range(1, len(results)):
            np.testing.assert_allclose(results[0], results[i], rtol=1e-10, equal_nan=True)
    
    def test_emv_partial_nan_handling(self):
        """Test EMV with partial NaN values"""
        high = np.array([np.nan, 12.0, 15.0, np.nan, 13.0, 16.0])
        low = np.array([np.nan, 9.0, 11.0, np.nan, 10.0, 12.0])
        close = np.array([np.nan, 10.0, 13.0, np.nan, 11.5, 14.0])
        volume = np.array([np.nan, 10000.0, 20000.0, np.nan, 15000.0, 25000.0])
        
        result = ta_indicators.emv(high, low, close, volume)
        
        # Check shape
        assert len(result) == len(high)
        
        # First few should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])  # Need previous value
        
        # Should have valid values after enough data
        assert not np.isnan(result[2])
    
    def test_emv_zero_range(self):
        """Test EMV when high equals low (zero range)"""
        high = np.array([10.0, 10.0, 12.0, 13.0])
        low = np.array([9.0, 10.0, 11.0, 12.0])  # At index 1: high == low
        close = np.array([9.5, 10.0, 11.5, 12.5])
        volume = np.array([1000.0, 2000.0, 3000.0, 4000.0])
        
        result = ta_indicators.emv(high, low, close, volume)
        
        # When range is zero, EMV should be NaN
        assert np.isnan(result[1])
        
        # Other values should be calculated
        assert not np.isnan(result[2])
    
    def test_emv_streaming(self, test_data):
        """Test EMV streaming functionality - mirrors check_emv_streaming"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        # Calculate batch result
        batch_result = ta_indicators.emv(high, low, close, volume)
        
        # Create stream and process same data
        stream = ta_indicators.EmvStream()
        stream_result = []
        
        for i in range(len(high)):
            value = stream.update(high[i], low[i], close[i], volume[i])
            stream_result.append(value if value is not None else np.nan)
        
        stream_result = np.array(stream_result)
        
        # Results should match
        assert len(batch_result) == len(stream_result)
        
        # Compare values where both are not NaN
        for i, (b, s) in enumerate(zip(batch_result, stream_result)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, atol=1e-9,
                        msg=f"EMV streaming mismatch at index {i}")
    
    def test_emv_batch(self, test_data):
        """Test EMV batch operations - mirrors check_batch_row"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        # EMV has no parameters, so batch just runs once
        result = ta_indicators.emv_batch(high, low, close, volume)
        
        # Check structure
        assert 'values' in result
        assert result['values'].shape == (1, len(high))
        
        # Values should match single calculation
        single_result = ta_indicators.emv(high, low, close, volume)
        np.testing.assert_allclose(result['values'][0], single_result, rtol=1e-10, equal_nan=True)
        
        # Verify last 5 values match expected
        expected_last_five = [
            -6488905.579799851,
            2371436.7401001123,
            -3855069.958128531,
            1051939.877943717,
            -8519287.22257077,
        ]
        
        assert_close(
            result['values'][0, -5:],
            expected_last_five,
            rtol=0.0001,
            msg="EMV batch last 5 values mismatch"
        )
    
    def test_emv_batch_with_kernel(self, test_data):
        """Test EMV batch with kernel parameter"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        # Test scalar kernel
        result = ta_indicators.emv_batch(high, low, close, volume, kernel="scalar")
        
        assert 'values' in result
        assert result['values'].shape == (1, len(high))
    
    def test_emv_mismatched_lengths(self):
        """Test EMV with mismatched input lengths"""
        high = np.array([10.0, 12.0, 13.0])
        low = np.array([9.0, 11.0])  # Different length
        close = np.array([9.5, 11.5, 12.0])
        volume = np.array([1000.0, 2000.0, 3000.0])
        
        # Should still work but use minimum length
        result = ta_indicators.emv(high, low, close, volume)
        assert len(result) == min(len(high), len(low), len(close), len(volume))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
