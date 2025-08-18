"""
Python binding tests for CORRELATION_CYCLE indicator.
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


class TestCorrelationCycle:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_correlation_cycle_accuracy(self, test_data):
        """Test CORRELATION_CYCLE matches expected values from Rust tests"""
        close = test_data['close']
        
        # Test with default parameters
        result = ta_indicators.correlation_cycle(close)
        
        # Check that we get a dictionary with 4 outputs
        assert isinstance(result, dict)
        assert 'real' in result
        assert 'imag' in result
        assert 'angle' in result
        assert 'state' in result
        
        # Check output length
        assert len(result['real']) == len(close)
        assert len(result['imag']) == len(close)
        assert len(result['angle']) == len(close)
        assert len(result['state']) == len(close)
        
        # Test with specific parameters
        result = ta_indicators.correlation_cycle(close, period=20, threshold=9.0)
        
        # Expected values from Rust tests
        expected_last_five_real = [
            -0.3348928030992766,
            -0.2908979303392832,
            -0.10648582811938148,
            -0.09118320471750277,
            0.0826798259258665,
        ]
        expected_last_five_imag = [
            0.2902308064575494,
            0.4025192756952553,
            0.4704322460080054,
            0.5404405595224989,
            0.5418162415918566,
        ]
        expected_last_five_angle = [
            -139.0865569687123,
            -125.8553823569915,
            -102.75438860700636,
            -99.576759208278,
            -81.32373697835556,
        ]
        
        # Check last 5 values
        for i in range(5):
            assert_close(result['real'][-5 + i], expected_last_five_real[i], rtol=1e-8)
            assert_close(result['imag'][-5 + i], expected_last_five_imag[i], rtol=1e-8)
            assert_close(result['angle'][-5 + i], expected_last_five_angle[i], rtol=1e-8)
    
    def test_correlation_cycle_partial_params(self, test_data):
        """Test with partial parameters"""
        close = test_data['close']
        
        # Test with None parameters (should use defaults)
        result = ta_indicators.correlation_cycle(close, period=None, threshold=None)
        assert len(result['real']) == len(close)
    
    def test_correlation_cycle_kernel_parameter(self, test_data):
        """Test with different kernel parameters"""
        close = test_data['close']
        
        # Test with scalar kernel
        result_scalar = ta_indicators.correlation_cycle(close, kernel='scalar')
        assert len(result_scalar['real']) == len(close)
        
        # Test with auto kernel
        result_auto = ta_indicators.correlation_cycle(close, kernel=None)
        assert len(result_auto['real']) == len(close)
    
    def test_correlation_cycle_errors(self):
        """Test error handling"""
        # Test with empty data
        with pytest.raises(ValueError):
            ta_indicators.correlation_cycle(np.array([]))
        
        # Test with all NaN values
        with pytest.raises(ValueError):
            ta_indicators.correlation_cycle(np.full(10, np.nan))
        
        # Test with zero period
        with pytest.raises(ValueError):
            ta_indicators.correlation_cycle(np.array([1.0, 2.0, 3.0]), period=0)
        
        # Test with period exceeding data length
        with pytest.raises(ValueError):
            ta_indicators.correlation_cycle(np.array([1.0, 2.0, 3.0]), period=10)
    
    def test_correlation_cycle_batch(self, test_data):
        """Test batch operations"""
        close = test_data['close']
        
        # Test batch with default ranges
        result = ta_indicators.correlation_cycle_batch(close)
        
        # Check that we get a dictionary with expected keys
        assert isinstance(result, dict)
        assert 'real' in result
        assert 'imag' in result
        assert 'angle' in result
        assert 'state' in result
        assert 'periods' in result
        assert 'thresholds' in result
        
        # Test batch with specific ranges
        result = ta_indicators.correlation_cycle_batch(
            close,
            period_range=(10, 30, 10),
            threshold_range=(5.0, 15.0, 5.0)
        )
        
        # Should have 3 periods * 3 thresholds = 9 combinations
        assert result['real'].shape[0] == 9
        assert result['real'].shape[1] == len(close)
    
    def test_correlation_cycle_stream(self):
        """Test streaming functionality"""
        # Create a stream
        stream = ta_indicators.CorrelationCycleStream(period=20, threshold=9.0)
        
        # Test stream with values
        values = [float(i) for i in range(50)]
        results = []
        
        for val in values:
            result = stream.update(val)
            results.append(result)
        
        # First 20 values should return None (warmup period)
        for i in range(20):
            assert results[i] is None
        
        # After warmup, should get tuples with 4 values
        for i in range(20, 50):
            assert results[i] is not None
            assert isinstance(results[i], tuple)
            assert len(results[i]) == 4  # real, imag, angle, state
    
    def test_correlation_cycle_nan_handling(self, test_data):
        """Test handling of NaN values in input"""
        close = test_data['close'].copy()
        
        # Insert some NaN values
        close[10:15] = np.nan
        
        # Should still work
        result = ta_indicators.correlation_cycle(close)
        assert len(result['real']) == len(close)
        
        # Check that warmup period has NaN values
        assert np.isnan(result['real'][0])
        assert np.isnan(result['imag'][0])
        assert np.isnan(result['angle'][0])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
