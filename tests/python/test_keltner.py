"""
Python binding tests for KELTNER indicator.
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


class TestKeltner:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_keltner_accuracy(self, test_data):
        """Test KELTNER matches expected values from Rust tests"""
        expected = EXPECTED_OUTPUTS['keltner']
        
        result = ta_indicators.keltner(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            test_data['close'],  # source
            expected['default_params']['period'],
            expected['default_params']['multiplier'],
            expected['default_params']['ma_type']
        )
        
        # Check all three bands
        assert 'upper_band' in result
        assert 'middle_band' in result
        assert 'lower_band' in result
        
        # Verify lengths
        assert len(result['upper_band']) == len(test_data['close'])
        assert len(result['middle_band']) == len(test_data['close'])
        assert len(result['lower_band']) == len(test_data['close'])
        
        # Check last 5 values for each band
        assert_close(
            result['upper_band'][-5:],
            expected['last_5_upper'],
            rtol=1e-1,  # Same tolerance as Rust tests
            msg="Upper band mismatch"
        )
        
        assert_close(
            result['middle_band'][-5:],
            expected['last_5_middle'],
            rtol=1e-1,
            msg="Middle band mismatch"
        )
        
        assert_close(
            result['lower_band'][-5:],
            expected['last_5_lower'],
            rtol=1e-1,
            msg="Lower band mismatch"
        )
    
    def test_keltner_errors(self):
        """Test error handling"""
        # Test with empty data
        with pytest.raises(ValueError, match="empty data"):
            ta_indicators.keltner(
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                20,
                2.0,
                "ema"
            )
        
        # Test with period = 0
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="invalid period"):
            ta_indicators.keltner(
                data,
                data,
                data,
                data,
                0,
                2.0,
                "ema"
            )
        
        # Test with period > data length
        with pytest.raises(ValueError, match="invalid period"):
            ta_indicators.keltner(
                data,
                data,
                data,
                data,
                10,
                2.0,
                "ema"
            )
    
    def test_keltner_streaming(self, test_data):
        """Test streaming functionality"""
        expected = EXPECTED_OUTPUTS['keltner']
        
        # Create streaming instance
        stream = ta_indicators.KeltnerStream(
            expected['default_params']['period'],
            expected['default_params']['multiplier'],
            expected['default_params']['ma_type']
        )
        
        # Process data through stream
        results = []
        for i in range(len(test_data['close'])):
            result = stream.update(
                test_data['high'][i],
                test_data['low'][i],
                test_data['close'][i],
                test_data['close'][i]  # source
            )
            results.append(result)
        
        # Extract non-None results
        non_none_results = [r for r in results if r is not None]
        
        # Should have results after warmup period
        assert len(non_none_results) > 0
        
        # Last result should match expected values
        if len(non_none_results) >= 5:
            last_5_upper = [r[0] for r in non_none_results[-5:]]
            last_5_middle = [r[1] for r in non_none_results[-5:]]
            last_5_lower = [r[2] for r in non_none_results[-5:]]
            
            assert_close(
                last_5_upper,
                expected['last_5_upper'],
                rtol=1e-1,
                msg="Streaming upper band mismatch"
            )
            assert_close(
                last_5_middle,
                expected['last_5_middle'],
                rtol=1e-1,
                msg="Streaming middle band mismatch"
            )
            assert_close(
                last_5_lower,
                expected['last_5_lower'],
                rtol=1e-1,
                msg="Streaming lower band mismatch"
            )
    
    def test_keltner_batch(self, test_data):
        """Test batch functionality"""
        result = ta_indicators.keltner_batch(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            test_data['close'],  # source
            (10, 30, 10),      # period range
            (1.0, 3.0, 1.0),   # multiplier range
            "ema"
        )
        
        assert 'upper_band' in result
        assert 'middle_band' in result
        assert 'lower_band' in result
        assert 'periods' in result
        assert 'multipliers' in result
        
        # Check shape - should have 3 periods x 3 multipliers = 9 rows
        expected_rows = 3 * 3
        assert result['upper_band'].shape == (expected_rows, len(test_data['close']))
        assert result['middle_band'].shape == (expected_rows, len(test_data['close']))
        assert result['lower_band'].shape == (expected_rows, len(test_data['close']))
        
        # Check parameter arrays
        assert len(result['periods']) == expected_rows
        assert len(result['multipliers']) == expected_rows


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
