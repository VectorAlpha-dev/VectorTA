"""
Python binding tests for NVI indicator.
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


class TestNvi:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_nvi_accuracy(self, test_data):
        """Test NVI matches expected values from Rust tests"""
        close = test_data['close']
        volume = test_data['volume']
        
        # Run NVI
        result = ta_indicators.nvi(close, volume)
        
        # Check last 5 values match expected
        expected = EXPECTED_OUTPUTS['nvi']['last_5_values']
        assert_close(result[-5:], expected, rtol=1e-7, atol=1e-5,
                     msg="NVI accuracy test failed")
    
    def test_nvi_empty_data(self):
        """Test error handling with empty data"""
        empty_close = np.array([])
        empty_volume = np.array([])
        
        with pytest.raises(ValueError):
            ta_indicators.nvi(empty_close, empty_volume)
    
    def test_nvi_not_enough_data(self):
        """Test error handling with insufficient valid data"""
        # Only one valid data point
        close = np.array([np.nan, 100.0])
        volume = np.array([np.nan, 120.0])
        
        with pytest.raises(ValueError):
            ta_indicators.nvi(close, volume)
    
    def test_nvi_all_nan(self):
        """Test error handling when all values are NaN"""
        close = np.array([np.nan, np.nan, np.nan])
        volume = np.array([np.nan, np.nan, np.nan])
        
        with pytest.raises(ValueError):
            ta_indicators.nvi(close, volume)
    
    def test_nvi_streaming(self, test_data):
        """Test NVI streaming mode"""
        close = test_data['close']
        volume = test_data['volume']
        
        # Create stream
        stream = ta_indicators.NviStream()
        
        # Process data through stream
        stream_results = []
        for i in range(len(close)):
            result = stream.update(close[i], volume[i])
            stream_results.append(result if result is not None else np.nan)
        
        # Compare with batch calculation
        batch_result = ta_indicators.nvi(close, volume)
        
        # Both should have same length
        assert len(stream_results) == len(batch_result)
        
        # Values should match (allowing for NaN handling)
        for i, (stream_val, batch_val) in enumerate(zip(stream_results, batch_result)):
            if np.isnan(batch_val):
                assert np.isnan(stream_val), f"Mismatch at index {i}: batch is NaN but stream is {stream_val}"
            else:
                assert abs(stream_val - batch_val) < 1e-9, \
                    f"Mismatch at index {i}: stream={stream_val}, batch={batch_val}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
