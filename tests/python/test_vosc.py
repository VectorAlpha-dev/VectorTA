"""
Python binding tests for VOSC indicator.
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


class TestVosc:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_vosc_accuracy(self, test_data):
        """Test VOSC matches expected values from Rust tests"""
        # Extract volume data
        volume = test_data.volume
        
        # Parameters from Rust test
        short_period = 2
        long_period = 5
        
        # Run VOSC calculation
        result = ta_indicators.vosc(volume, short_period, long_period)
        
        # Expected values from Rust test
        expected_last_five = [
            -39.478510754298895,
            -25.886077312645188,
            -21.155087549723756,
            -12.36093768813373,
            48.70809369473075,
        ]
        
        # Verify output
        assert len(result) == len(volume), "VOSC length mismatch"
        
        # Check last 5 values
        for i, expected in enumerate(expected_last_five):
            assert_close(result[-(5-i)], expected, 1e-1)
        
        # Check warmup period (should be NaN)
        warmup_period = long_period - 1
        for i in range(warmup_period):
            assert np.isnan(result[i]), f"Expected NaN at index {i}"
    
    def test_vosc_errors(self):
        """Test error handling"""
        # Test empty data
        with pytest.raises(Exception):
            ta_indicators.vosc([], 2, 5)
        
        # Test zero period
        with pytest.raises(Exception):
            ta_indicators.vosc([1.0, 2.0, 3.0], 0, 5)
        
        # Test short period > long period
        with pytest.raises(Exception):
            ta_indicators.vosc([1.0, 2.0, 3.0, 4.0, 5.0], 5, 2)
        
        # Test all NaN values
        with pytest.raises(Exception):
            ta_indicators.vosc([np.nan, np.nan, np.nan], 2, 3)
    
    def test_vosc_batch(self, test_data):
        """Test VOSC batch calculation"""
        volume = test_data.volume
        
        # Run batch calculation with ranges
        result = ta_indicators.vosc_batch(
            volume,
            (2, 10, 1),  # short_period_range
            (5, 20, 1)   # long_period_range
        )
        
        # Check result structure
        assert 'values' in result
        assert 'short_periods' in result
        assert 'long_periods' in result
        
        # Verify dimensions
        n_combos = len(result['short_periods'])
        assert result['values'].shape == (n_combos, len(volume))
    
    def test_vosc_stream(self):
        """Test VOSC streaming calculation"""
        # Create stream
        stream = ta_indicators.VoscStream(2, 5)
        
        # Test data
        test_values = [100.0, 120.0, 110.0, 130.0, 125.0, 140.0, 135.0]
        results = []
        
        # Update stream
        for value in test_values:
            result = stream.update(value)
            results.append(result)
        
        # First few values should be None (warmup)
        assert all(r is None for r in results[:4])
        
        # After warmup, should get values
        assert results[4] is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
