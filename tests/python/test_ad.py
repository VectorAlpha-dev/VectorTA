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
        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=1e-4,  # AD values are large, so use relative tolerance
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
        
        # Should still work as each security is processed independently
        result = ta_indicators.ad_batch(highs, lows, closes, volumes)
        assert result['values'].shape[0] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])