"""
Python binding tests for WAD indicator.
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


class TestWad:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_wad_accuracy(self, test_data):
        """Test WAD matches expected values from Rust tests"""
        # Extract price data
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Run WAD calculation
        result = ta_indicators.wad(high, low, close)
        
        # Expected values from Rust test
        expected_last_five = [
            158503.46790000016,
            158279.46790000016,
            158014.46790000016,
            158186.46790000016,
            157605.46790000016,
        ]
        
        # Verify output
        assert len(result) == len(close), "WAD length mismatch"
        
        # Check last 5 values
        for i, expected in enumerate(expected_last_five):
            assert_close(result[-(5-i)], expected, 1e-4)
    
    def test_wad_errors(self):
        """Test error handling"""
        # Test empty data
        with pytest.raises(Exception):
            ta_indicators.wad([], [], [])
        
        # Test mismatched array lengths
        with pytest.raises(Exception):
            ta_indicators.wad([1.0, 2.0], [1.0], [1.0, 2.0])
        
        # Test all NaN values
        with pytest.raises(Exception):
            ta_indicators.wad([np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan])
    
    def test_wad_batch(self, test_data):
        """Test WAD batch calculation"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Run batch calculation (WAD has no parameters, so this is simplified)
        result = ta_indicators.wad_batch(high, low, close)
        
        # Check result structure
        assert 'values' in result
        
        # Verify dimensions
        assert result['values'].shape == (1, len(close))
        
        # Compare with single calculation
        single_result = ta_indicators.wad(high, low, close)
        np.testing.assert_array_almost_equal(result['values'][0], single_result, decimal=10)
    
    def test_wad_stream(self):
        """Test WAD streaming calculation"""
        # Create stream
        stream = ta_indicators.WadStream()
        
        # Test data
        test_high = [10.0, 11.0, 12.0, 11.5, 12.5]
        test_low = [9.0, 9.5, 11.0, 10.5, 11.0]
        test_close = [9.5, 10.5, 11.5, 11.0, 12.0]
        results = []
        
        # Update stream
        for h, l, c in zip(test_high, test_low, test_close):
            result = stream.update(h, l, c)
            results.append(result)
        
        # Expected values from Rust test (check_wad_small_example)
        expected = [0.0, 1.0, 2.0, 1.5, 2.5]
        
        # Verify results
        for i, (result, expected_val) in enumerate(zip(results, expected)):
            assert abs(result - expected_val) < 1e-10, f"Stream mismatch at index {i}: {result} vs {expected_val}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])