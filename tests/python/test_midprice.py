"""
Python binding tests for MIDPRICE indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import the built module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'target/wheels'))

try:
    import rust_backtester as ta
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS


class TestMidprice:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_midprice_accuracy(self, test_data):
        """Test MIDPRICE matches expected values from Rust tests"""
        high = test_data['high']
        low = test_data['low']
        period = 14
        
        result = ta.midprice(high, low, period)
        
        # Check output length
        assert len(result) == len(high)
        
        # Check expected values
        expected = EXPECTED_OUTPUTS['midprice']['last_5_values']
        assert_close(result[-5:], expected, rtol=1e-1, msg="MIDPRICE last 5 values")
    
    def test_midprice_partial_params(self, test_data):
        """Test MIDPRICE with partial parameters - mirrors Rust tests"""
        high = test_data['high']
        low = test_data['low']
        
        # Test with default period (14)
        result = ta.midprice(high, low, 14)
        assert len(result) == len(high)
        
        # Verify warmup behavior
        assert np.all(np.isnan(result[:13]))
        assert not np.any(np.isnan(result[20:]))
    
    def test_midprice_with_default_params(self, test_data):
        """Test MIDPRICE with default parameters"""
        high = test_data['high']
        low = test_data['low']
        
        result = ta.midprice(high, low, 14)
        assert len(result) == len(high)
        
        # Check that warmup period has NaN values
        assert np.all(np.isnan(result[:13]))
        # Check that after warmup we have valid values
        assert not np.any(np.isnan(result[20:]))
    
    def test_midprice_errors(self):
        """Test error handling"""
        # Empty data
        with pytest.raises(ValueError):
            ta.midprice(np.array([]), np.array([]), 14)
        
        # Mismatched lengths
        with pytest.raises(ValueError):
            ta.midprice(np.array([1.0, 2.0]), np.array([1.0]), 14)
        
        # Invalid period
        with pytest.raises(ValueError):
            ta.midprice(np.array([1.0]), np.array([1.0]), 0)
        
        # Period exceeds data length
        with pytest.raises(ValueError):
            ta.midprice(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 10)
    
    def test_midprice_zero_period(self):
        """Test MIDPRICE fails with zero period - mirrors Rust tests"""
        high = np.array([10.0, 14.0, 12.0])
        low = np.array([5.0, 6.0, 7.0])
        
        with pytest.raises(ValueError):
            ta.midprice(high, low, 0)
    
    def test_midprice_period_exceeds_length(self):
        """Test MIDPRICE fails when period exceeds data length"""
        high = np.array([10.0, 14.0, 12.0])
        low = np.array([5.0, 6.0, 7.0])
        
        with pytest.raises(ValueError):
            ta.midprice(high, low, 10)
    
    def test_midprice_very_small_dataset(self):
        """Test MIDPRICE fails with insufficient data"""
        high = np.array([42.0])
        low = np.array([36.0])
        
        with pytest.raises(ValueError):
            ta.midprice(high, low, 14)
    
    def test_midprice_all_nan(self):
        """Test handling of all NaN values"""
        high = np.array([np.nan, np.nan, np.nan])
        low = np.array([np.nan, np.nan, np.nan])
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta.midprice(high, low, 2)
    
    def test_midprice_reinput(self, test_data):
        """Test MIDPRICE applied twice (re-input)"""
        high = test_data['high']
        low = test_data['low']
        period = 10
        
        # First calculation
        first_result = ta.midprice(high, low, period)
        
        # Use output as input for second calculation
        second_result = ta.midprice(first_result, first_result, period)
        
        assert len(second_result) == len(first_result)
        # Check that we still have valid values after double application
        assert not np.all(np.isnan(second_result[30:]))
    
    def test_midprice_nan_handling(self, test_data):
        """Test MIDPRICE handles NaN values correctly"""
        high = test_data['high']
        low = test_data['low']
        
        result = ta.midprice(high, low, 14)
        
        # Check output length matches input
        assert len(result) == len(high)
        
        # After warmup period, should have no NaN values (assuming input has no NaN)
        if len(result) > 20:
            assert not np.any(np.isnan(result[20:]))
    
    def test_midprice_streaming(self, test_data):
        """Test streaming functionality"""
        high = test_data['high']
        low = test_data['low']
        period = 14
        
        # Batch calculation
        batch_result = ta.midprice(high, low, period)
        
        # Streaming calculation
        stream = ta.MidpriceStream(period)
        stream_result = []
        
        for h, l in zip(high, low):
            val = stream.update(h, l)
            stream_result.append(val if val is not None else np.nan)
        
        # Compare results
        assert_close(batch_result, stream_result, rtol=1e-9, msg="Streaming vs batch")
    
    def test_midprice_batch(self, test_data):
        """Test batch computation with parameter sweep"""
        high = test_data['high']
        low = test_data['low']
        
        # Test period range
        result = ta.midprice_batch(high, low, (10, 20, 5))
        
        assert 'values' in result
        assert 'periods' in result
        
        # Check shape
        values = result['values']
        periods = result['periods']
        
        assert values.shape == (3, len(high))  # 3 periods: 10, 15, 20
        assert len(periods) == 3
        assert list(periods) == [10, 15, 20]
        
        # Verify one of the results matches single computation
        single_result = ta.midprice(high, low, 10)
        assert_close(values[0], single_result, rtol=1e-9, msg="Batch row 0 (period=10)")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
