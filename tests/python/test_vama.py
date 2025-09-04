"""
Python binding tests for VAMA indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    import my_project
except ImportError:
    pytest.skip("Python module not built", allow_module_level=True)

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS


class TestVama:
    @pytest.fixture(scope='class')
    def test_data(self):
        """Load test data from CSV file"""
        return load_test_data()
    
    def test_vama_fast_accuracy(self, test_data):
        """Test VAMA with fast parameters (length=13) matches expected values"""
        close = test_data['close']
        volume = test_data['volume']
        expected = EXPECTED_OUTPUTS['vama']
        
        # Test with fast parameters (length=13)
        result = my_project.vama(
            close,
            volume,
            length=expected['default_params']['length'],
            vi_factor=expected['default_params']['vi_factor'],
            strict=expected['default_params']['strict'],
            sample_period=expected['default_params']['sample_period']
        )
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        assert_close(
            result[-5:],
            expected['fast_values'],
            rtol=1e-6,
            msg="VAMA fast last 5 values mismatch"
        )
    
    def test_vama_slow_accuracy(self, test_data):
        """Test VAMA with slow parameters (length=55) matches expected values"""
        close = test_data['close']
        volume = test_data['volume']
        expected = EXPECTED_OUTPUTS['vama']
        
        # Test with slow parameters (length=55)
        result = my_project.vama(
            close,
            volume,
            length=expected['slow_params']['length'],
            vi_factor=expected['slow_params']['vi_factor'],
            strict=expected['slow_params']['strict'],
            sample_period=expected['slow_params']['sample_period']
        )
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        assert_close(
            result[-5:],
            expected['slow_values'],
            rtol=1e-6,
            msg="VAMA slow last 5 values mismatch"
        )
    
    def test_vama_default_params(self, test_data):
        """Test VAMA with default parameters"""
        close = test_data['close']
        volume = test_data['volume']
        
        # Default params: length=13, vi_factor=0.67, strict=True, sample_period=0
        result = my_project.vama(close, volume, 13, 0.67, True, 0)
        
        assert len(result) == len(close)
    
    def test_vama_empty_input(self):
        """Test VAMA fails with empty input"""
        empty_price = np.array([])
        empty_volume = np.array([])
        
        with pytest.raises(ValueError, match="[Ee]mpty"):
            my_project.vama(empty_price, empty_volume)
    
    def test_vama_all_nan(self):
        """Test VAMA fails with all NaN values"""
        all_nan = np.full(100, np.nan)
        volume = np.full(100, 100.0)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            my_project.vama(all_nan, volume)
    
    def test_vama_mismatched_lengths(self):
        """Test VAMA fails when price and volume have different lengths"""
        price = np.array([10.0, 20.0, 30.0])
        volume = np.array([100.0, 200.0])  # Different length
        
        with pytest.raises(ValueError, match="length mismatch"):
            my_project.vama(price, volume)
    
    def test_vama_invalid_period(self):
        """Test VAMA fails with zero period"""
        price = np.array([10.0, 20.0, 30.0])
        volume = np.array([100.0, 200.0, 300.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            my_project.vama(price, volume, length=0)
    
    def test_vama_invalid_vi_factor(self):
        """Test VAMA fails with invalid vi_factor"""
        price = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        volume = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        
        # Zero vi_factor
        with pytest.raises(ValueError, match="Invalid vi_factor"):
            my_project.vama(price, volume, length=2, vi_factor=0.0)
        
        # Negative vi_factor
        with pytest.raises(ValueError, match="Invalid vi_factor"):
            my_project.vama(price, volume, length=2, vi_factor=-1.0)
    
    def test_vama_period_exceeds_length(self):
        """Test VAMA fails when period exceeds data length"""
        small_price = np.array([10.0, 20.0, 30.0])
        small_volume = np.array([100.0, 200.0, 300.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough"):
            my_project.vama(small_price, small_volume, length=10)
    
    def test_vama_nan_handling(self, test_data):
        """Test VAMA handles NaN values correctly"""
        close = test_data['close']
        volume = test_data['volume']
        expected = EXPECTED_OUTPUTS['vama']
        
        result = my_project.vama(close, volume, length=13, vi_factor=0.67, strict=True, sample_period=0)
        
        assert len(result) == len(close)
        
        # After warmup period, no NaN values should exist
        warmup = expected['warmup_period']  # Should be 12 (length - 1)
        
        # Check no NaN after warmup
        if len(result) > warmup:
            assert not np.any(np.isnan(result[warmup+1:])), f"Found unexpected NaN after warmup period {warmup}"
        
        # First warmup values should be NaN
        assert np.all(np.isnan(result[:warmup])), f"Expected NaN in warmup period (first {warmup} values)"
    
    def test_vama_strict_vs_non_strict(self, test_data):
        """Test VAMA with strict=True vs strict=False"""
        close = test_data['close'][:100]  # Use smaller dataset
        volume = test_data['volume'][:100]
        
        # Test with strict=True
        result_strict = my_project.vama(close, volume, length=13, vi_factor=0.67, strict=True, sample_period=0)
        
        # Test with strict=False
        result_non_strict = my_project.vama(close, volume, length=13, vi_factor=0.67, strict=False, sample_period=0)
        
        assert len(result_strict) == len(close)
        assert len(result_non_strict) == len(close)
        
        # Results may differ but both should be valid
        assert not np.all(np.isnan(result_strict[13:]))
        assert not np.all(np.isnan(result_non_strict[13:]))
    
    def test_vama_sample_period(self, test_data):
        """Test VAMA with different sample periods"""
        close = test_data['close'][:100]  # Use smaller dataset
        volume = test_data['volume'][:100]
        
        # Test with sample_period=0 (all bars)
        result_all = my_project.vama(close, volume, length=13, vi_factor=0.67, strict=True, sample_period=0)
        
        # Test with fixed sample_period
        result_fixed = my_project.vama(close, volume, length=13, vi_factor=0.67, strict=True, sample_period=20)
        
        assert len(result_all) == len(close)
        assert len(result_fixed) == len(close)
        
        # Results may differ but both should be valid
        assert not np.all(np.isnan(result_all[13:]))
        assert not np.all(np.isnan(result_fixed[13:]))
    
    def test_vama_different_vi_factors(self, test_data):
        """Test VAMA with different vi_factor values"""
        close = test_data['close'][:100]  # Use smaller dataset
        volume = test_data['volume'][:100]
        
        # Test with different vi_factors
        result1 = my_project.vama(close, volume, length=13, vi_factor=0.5, strict=True, sample_period=0)
        result2 = my_project.vama(close, volume, length=13, vi_factor=0.67, strict=True, sample_period=0)
        result3 = my_project.vama(close, volume, length=13, vi_factor=1.0, strict=True, sample_period=0)
        
        assert len(result1) == len(close)
        assert len(result2) == len(close)
        assert len(result3) == len(close)
        
        # Results should differ with different vi_factors
        assert not np.array_equal(result1[-10:], result2[-10:])
        assert not np.array_equal(result2[-10:], result3[-10:])
    
    def test_vama_constant_volume(self):
        """Test VAMA with constant volume"""
        # Create price series with some variation
        price = np.array([50.0, 51.0, 49.0, 52.0, 48.0, 53.0, 47.0, 54.0, 46.0, 55.0] * 5)
        # Constant volume
        volume = np.array([1000.0] * 50)
        
        result = my_project.vama(price, volume, length=5, vi_factor=0.67, strict=True, sample_period=0)
        
        assert len(result) == len(price)
        # With constant volume, VAMA should still produce valid results
        assert not np.all(np.isnan(result[5:]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])