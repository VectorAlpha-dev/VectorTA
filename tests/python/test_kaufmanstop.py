"""
Python binding tests for KAUFMANSTOP indicator.
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


class TestKaufmanstop:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_kaufmanstop_accuracy(self, test_data):
        """Test KAUFMANSTOP matches expected values from Rust tests"""
        # Use default parameters: period=22, mult=2.0, direction="long", ma_type="sma"
        result = ta_indicators.kaufmanstop(
            test_data['high'], 
            test_data['low'], 
            period=22, 
            mult=2.0, 
            direction="long", 
            ma_type="sma"
        )
        
        # Expected last 5 values from Rust tests
        expected_last_five = [
            56711.545454545456,
            57132.72727272727,
            57015.72727272727,
            57137.18181818182,
            56516.09090909091,
        ]
        
        # Check last 5 values
        for i, expected in enumerate(expected_last_five):
            assert_close(result[-(5-i)], expected, rtol=1e-6)
    
    def test_kaufmanstop_zero_period(self):
        """Test that kaufmanstop fails with zero period"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0, 25.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.kaufmanstop(high, low, period=0)
    
    def test_kaufmanstop_period_exceeds_length(self):
        """Test that kaufmanstop fails when period exceeds data length"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0, 25.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.kaufmanstop(high, low, period=10)
    
    def test_kaufmanstop_mismatched_lengths(self):
        """Test that kaufmanstop fails when high and low have different lengths"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0])
        
        with pytest.raises(ValueError, match="High and low arrays must have the same length"):
            ta_indicators.kaufmanstop(high, low)
    
    def test_kaufmanstop_stream(self):
        """Test KaufmanstopStream class"""
        stream = ta_indicators.KaufmanstopStream(period=22, mult=2.0, direction="long", ma_type="sma")
        
        # Test that update returns None until enough data
        for i in range(21):
            result = stream.update(100.0 + i, 95.0 + i)
            assert result is None
        
        # 22nd value should return a result
        result = stream.update(122.0, 117.0)
        assert result is not None
        assert isinstance(result, float)
    
    def test_kaufmanstop_batch(self, test_data):
        """Test batch processing"""
        result = ta_indicators.kaufmanstop_batch(
            test_data['high'],
            test_data['low'],
            period_range=(20, 24, 2),  # 20, 22, 24
            mult_range=(1.5, 2.5, 0.5),  # 1.5, 2.0, 2.5
            direction="long",
            ma_type="sma"
        )
        
        # Check structure
        assert 'values' in result
        assert 'combos' in result
        assert 'rows' in result
        assert 'cols' in result
        
        # Should have 3 periods × 3 mults = 9 rows
        assert result['rows'] == 9
        assert result['cols'] == len(test_data['high'])
        assert len(result['values']) == 9 * len(test_data['high'])
        assert len(result['combos']) == 9
        
        # Verify combos have correct structure
        for combo in result['combos']:
            assert 'period' in combo
            assert 'mult' in combo
            assert 'direction' in combo
            assert 'ma_type' in combo
    
    def test_kaufmanstop_kernel_options(self, test_data):
        """Test different kernel options"""
        kernels = [None, "scalar", "auto"]
        results = []
        
        for kernel in kernels:
            result = ta_indicators.kaufmanstop(
                test_data['high'], 
                test_data['low'],
                kernel=kernel
            )
            results.append(result)
        
        # All kernel options should produce the same results
        for i in range(1, len(results)):
            np.testing.assert_allclose(results[0], results[i], rtol=1e-10)
    
    def test_kaufmanstop_short_direction(self, test_data):
        """Test short direction"""
        result_long = ta_indicators.kaufmanstop(
            test_data['high'][:100], 
            test_data['low'][:100],
            direction="long"
        )
        
        result_short = ta_indicators.kaufmanstop(
            test_data['high'][:100], 
            test_data['low'][:100],
            direction="short"
        )
        
        # Results should be different
        assert not np.allclose(result_long, result_short)
        
        # Short stops should be above price (high), long stops below (low)
        # After warmup period, check a few values
        for i in range(30, 40):
            if not np.isnan(result_long[i]) and not np.isnan(result_short[i]):
                # Long stop should be below low price
                assert result_long[i] < test_data['low'][i] + 0.01
                # Short stop should be above high price  
                assert result_short[i] > test_data['high'][i] - 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
