"""
Python binding tests for EMD indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import the built module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'target/wheels'))

try:
    import my_project as ta
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS


class TestEmd:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_emd_accuracy(self, test_data):
        """Test EMD matches expected values from Rust tests"""
        # EMD only uses high and low data
        upperband, middleband, lowerband = ta.emd(
            test_data['high'],
            test_data['low'],
            period=20,
            delta=0.5,
            fraction=0.1
        )
        
        # Expected values from Rust tests (last 5 values)
        expected_last_five_upper = [
            50.33760237677157,
            50.28850695686447,
            50.23941153695737,
            50.19031611705027,
            48.709744457737344,
        ]
        expected_last_five_middle = [
            -368.71064280396706,
            -399.11033986231377,
            -421.9368852621732,
            -437.879217150269,
            -447.3257167904511,
        ]
        expected_last_five_lower = [
            -60.67834136221248,
            -60.93110347122829,
            -61.68154077026321,
            -62.43197806929814,
            -63.18241536833306,
        ]
        
        # Check last 5 values
        for i in range(5):
            assert_close(upperband[-5+i], expected_last_five_upper[i], 
                        rtol=1e-6, atol=1e-6, msg=f"EMD upperband mismatch at index {-5+i}")
            assert_close(middleband[-5+i], expected_last_five_middle[i], 
                        rtol=1e-6, atol=1e-6, msg=f"EMD middleband mismatch at index {-5+i}")
            assert_close(lowerband[-5+i], expected_last_five_lower[i], 
                        rtol=1e-6, atol=1e-6, msg=f"EMD lowerband mismatch at index {-5+i}")
    
    def test_emd_errors(self):
        """Test error handling"""
        # Test with empty data
        with pytest.raises(ValueError, match="Invalid input length"):
            ta.emd(
                np.array([]),
                np.array([]),
                period=20,
                delta=0.5,
                fraction=0.1
            )
        
        # Test with all NaN
        nan_data = np.full(10, np.nan)
        with pytest.raises(ValueError, match="All values are NaN"):
            ta.emd(
                nan_data,
                nan_data,
                period=20,
                delta=0.5,
                fraction=0.1
            )
        
        # Test with invalid period (0)
        data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Invalid period"):
            ta.emd(
                data,
                data,
                period=0,
                delta=0.5,
                fraction=0.1
            )
        
        # Test with not enough data
        small_data = np.array([10.0] * 10)
        with pytest.raises(ValueError, match="Invalid period"):
            ta.emd(
                small_data,
                small_data,
                period=20,  # Needs at least 2*period = 40 or 50 data points
                delta=0.5,
                fraction=0.1
            )
    
    def test_emd_streaming(self):
        """Test EMD streaming functionality"""
        # Create stream
        stream = ta.EmdStream(period=20, delta=0.5, fraction=0.1)
        
        # Feed some data
        for i in range(100):
            value = 50.0 + i * 0.1
            upperband, middleband, lowerband = stream.update(value, value)
            
            # After warmup, all bands should have values
            if i >= 50:  # Warmup is max(50, 2*period-1) = 50
                assert upperband is not None, f"Upperband should have value at index {i}"
                assert lowerband is not None, f"Lowerband should have value at index {i}"
            if i >= 39:  # Middleband warmup is 2*period-1 = 39
                assert middleband is not None, f"Middleband should have value at index {i}"
    
    def test_emd_batch(self, test_data):
        """Test batch processing"""
        result = ta.emd_batch(
            test_data['high'],
            test_data['low'],
            period_range=(20, 22, 2),    # 2 values: 20, 22
            delta_range=(0.5, 0.6, 0.1), # 2 values: 0.5, 0.6
            fraction_range=(0.1, 0.2, 0.1)  # 2 values: 0.1, 0.2
        )
        
        # Should have 2*2*2 = 8 combinations
        assert result['upper'].shape[0] == 8, "Expected 8 rows in batch output"
        assert result['middle'].shape[0] == 8, "Expected 8 rows in batch output"
        assert result['lower'].shape[0] == 8, "Expected 8 rows in batch output"
        
        # Check parameter arrays
        assert len(result['periods']) == 8
        assert len(result['deltas']) == 8
        assert len(result['fractions']) == 8
        
        # Verify first combo matches single calculation
        upperband, middleband, lowerband = ta.emd(
            test_data['high'],
            test_data['low'],
            period=20,
            delta=0.5,
            fraction=0.1
        )
        
        # Compare with first row of batch result
        np.testing.assert_array_almost_equal(result['upper'][0], upperband, decimal=6)
        np.testing.assert_array_almost_equal(result['middle'][0], middleband, decimal=6)
        np.testing.assert_array_almost_equal(result['lower'][0], lowerband, decimal=6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
