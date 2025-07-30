"""
Python binding tests for MEAN_AD indicator.
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


class TestMean_Ad:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_mean_ad_accuracy(self, test_data):
        """Test MEAN_AD matches expected values from Rust tests"""
        # Test with hl2 source like in Rust
        hl2 = (test_data['high'] + test_data['low']) / 2
        result = ta_indicators.mean_ad(hl2, period=5)
        
        expected = EXPECTED_OUTPUTS['mean_ad']['last_5_values']
        assert_close(result[-5:], expected, rtol=1e-1, atol=1e-1,
                    msg="mean_ad last 5 values mismatch")
    
    def test_mean_ad_default_params(self, test_data):
        """Test with default parameters"""
        result = ta_indicators.mean_ad(test_data['close'], period=5)
        assert len(result) == len(test_data['close'])
        
        # Check warmup period (first + 2 * period - 2 values should be NaN)
        # For period=5, warmup should be 8 values (0 + 2*5 - 2 = 8)
        assert np.all(np.isnan(result[:8]))
        assert not np.any(np.isnan(result[240:]))  # No NaN after warmup
    
    def test_mean_ad_errors(self):
        """Test error handling"""
        # Test with empty data
        with pytest.raises(Exception):
            ta_indicators.mean_ad(np.array([]), period=5)
        
        # Test with zero period
        with pytest.raises(Exception):
            ta_indicators.mean_ad(np.array([1.0, 2.0, 3.0]), period=0)
        
        # Test with period exceeding data length
        with pytest.raises(Exception):
            ta_indicators.mean_ad(np.array([1.0, 2.0, 3.0]), period=10)
        
        # Test with all NaN values
        with pytest.raises(Exception):
            ta_indicators.mean_ad(np.array([np.nan, np.nan, np.nan]), period=2)
    
    def test_mean_ad_streaming(self, test_data):
        """Test streaming functionality"""
        stream = ta_indicators.MeanAdStream(period=5)
        batch_result = ta_indicators.mean_ad(test_data['close'], period=5)
        
        stream_result = []
        for price in test_data['close']:
            value = stream.update(price)
            stream_result.append(value if value is not None else np.nan)
        
        stream_result = np.array(stream_result)
        
        # Compare batch vs streaming (they should match)
        valid_idx = ~(np.isnan(batch_result) | np.isnan(stream_result))
        assert_close(batch_result[valid_idx], stream_result[valid_idx], 
                    rtol=1e-9, msg="Batch vs streaming mismatch")
    
    def test_mean_ad_batch(self, test_data):
        """Test batch computation"""
        result = ta_indicators.mean_ad_batch(test_data['close'], period_range=(5, 50, 1))
        
        assert 'values' in result
        assert 'periods' in result
        
        # Should have 46 combinations (5 to 50 inclusive, step 1)
        assert result['values'].shape[0] == 46
        assert result['values'].shape[1] == len(test_data['close'])
        assert len(result['periods']) == 46


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
