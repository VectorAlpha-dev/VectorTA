"""
Python binding tests for AD indicator.
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


class TestAd:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_ad_partial_params(self, test_data):
        """Test AD with default parameters - mirrors check_ad_partial_params"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        # AD has no parameters, just test basic functionality
        result = ta_indicators.ad(high, low, close, volume)
        assert len(result) == len(close)
    
    def test_ad_accuracy(self, test_data):
        """Test AD matches expected values from Rust tests - mirrors check_ad_accuracy"""
        high = test_data['high']
        low = test_data['low'] 
        close = test_data['close']
        volume = test_data['volume']
        expected = EXPECTED_OUTPUTS['ad']
        
        result = ta_indicators.ad(high, low, close, volume)
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        # Match Rust unit test tolerance (absolute 1e-1)
        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=0.0,
            atol=1e-1,
            msg="AD last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('ad', result, 'ohlcv', expected['default_params'])
    
    def test_ad_reinput(self, test_data):
        """Test AD applied with reinput data - mirrors check_ad_with_slice_data_reinput"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        # First pass
        first_result = ta_indicators.ad(high, low, close, volume)
        assert len(first_result) == len(close)
        
        # Second pass - use AD output as all inputs (unrealistic but tests the function)
        second_result = ta_indicators.ad(first_result, first_result, first_result, first_result)
        assert len(second_result) == len(first_result)
        
        # Check no NaN after index 50
        if len(second_result) > 50:
            assert not np.any(np.isnan(second_result[50:])), "Found unexpected NaN after index 50"
    
    def test_ad_nan_check(self, test_data):
        """Test AD handles NaN values correctly - mirrors check_ad_accuracy_nan_check"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        result = ta_indicators.ad(high, low, close, volume)
        assert len(result) == len(close)
        
        # AD has no warmup period, but check after index 50 for consistency
        if len(result) > 50:
            assert not np.any(np.isnan(result[50:])), "Found unexpected NaN after index 50"
    
    def test_ad_streaming(self, test_data):
        """Test AD streaming matches batch calculation - mirrors check_ad_streaming"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        # Batch calculation
        batch_result = ta_indicators.ad(high, low, close, volume)
        
        # Streaming calculation
        stream = ta_indicators.AdStream()
        stream_values = []
        
        for i in range(len(close)):
            result = stream.update(high[i], low[i], close[i], volume[i])
            stream_values.append(result)
        
        stream_values = np.array(stream_values)
        
        # Compare batch vs streaming
        assert len(batch_result) == len(stream_values)
        assert_close(batch_result, stream_values, rtol=1e-9, atol=1e-9,
                    msg="AD streaming mismatch")
    
    def test_ad_data_length_mismatch(self, test_data):
        """Test AD fails with mismatched input lengths"""
        high = test_data['high']
        low = test_data['low'][:100]  # Shorter array
        close = test_data['close']
        volume = test_data['volume']
        
        with pytest.raises(ValueError, match="Data length mismatch"):
            ta_indicators.ad(high, low, close, volume)
    
    def test_ad_empty_input(self):
        """Test AD fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Not enough data"):
            ta_indicators.ad(empty, empty, empty, empty)
    
    def test_ad_kernel_selection(self, test_data):
        """Test AD with different kernel selections"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        # Test different kernels
        for kernel in ['auto', 'scalar', 'avx2', 'avx512']:
            try:
                result = ta_indicators.ad(high, low, close, volume, kernel=kernel)
                assert len(result) == len(close)
            except ValueError as e:
                # AVX kernels might not be available on all systems
                if "Unknown kernel" not in str(e) and "not available on this CPU" not in str(e) and "not compiled in this build" not in str(e):
                    raise
    
    def test_ad_batch(self, test_data):
        """Test AD batch processing"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        # Create multiple securities (3 copies for testing)
        highs = [high, high, high]
        lows = [low, low, low]
        closes = [close, close, close]
        volumes = [volume, volume, volume]
        
        result = ta_indicators.ad_batch(highs, lows, closes, volumes)
        
        assert 'values' in result
        assert result['values'].shape[0] == 3  # 3 securities
        assert result['values'].shape[1] == len(close)
        
        # Each row should match the single calculation
        single_result = ta_indicators.ad(high, low, close, volume)
        for i in range(3):
            assert_close(
                result['values'][i],
                single_result,
                rtol=1e-8,
                msg=f"AD batch row {i} mismatch"
            )
    
    def test_ad_batch_different_lengths(self):
        """Test AD batch fails with different length inputs"""
        highs = [np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0])]  # Different lengths
        lows = [np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0])]
        closes = [np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0])]
        volumes = [np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0])]
        
        # Should fail because rows have different lengths
        with pytest.raises(ValueError, match="Data length mismatch"):
            ta_indicators.ad_batch(highs, lows, closes, volumes)
    
    def test_ad_all_nan_input(self):
        """Test AD with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        # AD returns all NaN when input is all NaN (doesn't throw error)
        result = ta_indicators.ad(all_nan, all_nan, all_nan, all_nan)
        assert len(result) == len(all_nan)
        assert np.all(np.isnan(result)), "AD should return all NaN when input is all NaN"
    
    def test_ad_no_warmup_period(self, test_data):
        """Test AD has no warmup period - starts from 0"""
        high = test_data['high'][:50]
        low = test_data['low'][:50]
        close = test_data['close'][:50]
        volume = test_data['volume'][:50]
        
        result = ta_indicators.ad(high, low, close, volume)
        
        # AD should start at 0, not NaN
        assert not np.isnan(result[0]), "AD should not have NaN at index 0"
        # First AD value calculation: if high == low, CLV = 0, so AD[0] = 0
        # Otherwise AD[0] = CLV * volume
        if high[0] == low[0]:
            assert result[0] == 0, "First AD value should be 0 when high equals low"
    
    def test_ad_high_low_validation(self):
        """Test AD handles invalid high/low relationships"""
        high = np.array([100.0, 90.0, 95.0])  
        low = np.array([105.0, 95.0, 90.0])  # low > high (invalid)
        close = np.array([102.0, 92.0, 93.0])
        volume = np.array([1000.0, 1500.0, 1200.0])
        
        # Should handle gracefully (CLV becomes 0 when high==low)
        result = ta_indicators.ad(high, low, close, volume)
        assert len(result) == len(close)
    
    def test_ad_zero_volume(self, test_data):
        """Test AD with zero volume periods"""
        high = test_data['high'][:10]
        low = test_data['low'][:10]
        close = test_data['close'][:10]
        volume = np.array(test_data['volume'][:10])
        volume[5] = 0.0  # Set one volume to 0
        
        result = ta_indicators.ad(high, low, close, volume)
        assert len(result) == len(close)
        # When volume is 0, AD should not change from previous value
        if len(result) > 6:
            # AD[i] = AD[i-1] + CLV * volume, so if volume=0, AD[i] = AD[i-1]
            assert_close(result[5], result[4], rtol=1e-10, atol=1e-10,
                        msg="AD should not change when volume is 0")
    
    def test_ad_batch_metadata(self, test_data):
        """Test AD batch returns proper metadata"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        highs = [high, high]
        lows = [low, low]
        closes = [close, close]
        volumes = [volume, volume]
        
        result = ta_indicators.ad_batch(highs, lows, closes, volumes)
        
        assert 'values' in result
        assert 'rows' in result
        assert 'cols' in result
        assert result['rows'] == 2
        assert result['cols'] == len(close)
    
    def test_ad_streaming_reset(self):
        """Test AD streaming can be reset"""
        stream = ta_indicators.AdStream()
        
        # Add some values
        for i in range(5):
            val = float(i)
            result = stream.update(val+1, val, val+0.5, 100.0)
            assert result is not None
        
        # Reset stream (create new instance)
        stream = ta_indicators.AdStream()
        
        # First value after reset should start fresh
        result = stream.update(10.0, 9.0, 9.5, 1000.0)
        assert result is not None
        # AD is cumulative, so first value after reset should be the CLV * volume
        clv = (2 * 9.5 - 10.0 - 9.0) / (10.0 - 9.0)
        expected = clv * 1000.0
        assert_close(result, expected, rtol=1e-10, atol=1e-10,
                    msg="First AD value after reset should be CLV * volume")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
