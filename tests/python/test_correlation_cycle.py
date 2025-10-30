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
        """Test CORRELATION_CYCLE matches expected values from Rust tests - mirrors check_cc_accuracy"""
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
        
        # Check last 5 values for all outputs including state
        for i in range(5):
            # Match Rust test tolerance: absolute <= 1e-8 (no looser relative tolerance)
            assert_close(
                result['real'][-5 + i], expected_last_five_real[i], rtol=0.0, atol=1e-8,
                msg=f"Real value mismatch at index {i}"
            )
            assert_close(
                result['imag'][-5 + i], expected_last_five_imag[i], rtol=0.0, atol=1e-8,
                msg=f"Imag value mismatch at index {i}"
            )
            assert_close(
                result['angle'][-5 + i], expected_last_five_angle[i], rtol=0.0, atol=1e-8,
                msg=f"Angle value mismatch at index {i}"
            )
        
        # Verify state values are -1, 0, or 1 after warmup
        warmup = 20  # period
        for i in range(warmup + 1, len(result['state'])):
            assert result['state'][i] in [-1.0, 0.0, 1.0], f"State at index {i} should be -1, 0 or 1, got {result['state'][i]}"
    
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
        """Test handling of NaN values in input - mirrors check_cc_nan_handling"""
        close = test_data['close'].copy()
        
        # Insert some NaN values
        close[10:15] = np.nan
        
        # Should still work
        result = ta_indicators.correlation_cycle(close, period=20, threshold=9.0)
        assert len(result['real']) == len(close)
        
        # Check that warmup period has NaN values
        assert np.isnan(result['real'][0])
        assert np.isnan(result['imag'][0])
        assert np.isnan(result['angle'][0])
        
        # After sufficient data (beyond NaN region + warmup), should have valid values
        if len(result['real']) > 40:
            # Check no unexpected NaN values after position 40
            for i in range(40, min(50, len(result['real']))):
                assert not np.isnan(result['real'][i]), f"Unexpected NaN in real at index {i}"
                assert not np.isnan(result['imag'][i]), f"Unexpected NaN in imag at index {i}"
                assert not np.isnan(result['angle'][i]), f"Unexpected NaN in angle at index {i}"
    
    def test_correlation_cycle_default_candles(self, test_data):
        """Test CORRELATION_CYCLE with default parameters - mirrors check_cc_default_candles"""
        close = test_data['close']
        
        # Default params: period=20, threshold=9.0
        result = ta_indicators.correlation_cycle(close)
        assert len(result['real']) == len(close)
        assert len(result['imag']) == len(close)
        assert len(result['angle']) == len(close)
        assert len(result['state']) == len(close)
    
    def test_correlation_cycle_zero_period(self):
        """Test CORRELATION_CYCLE fails with zero period - mirrors check_cc_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.correlation_cycle(input_data, period=0)
    
    def test_correlation_cycle_period_exceeds_length(self):
        """Test CORRELATION_CYCLE fails when period exceeds data length - mirrors check_cc_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.correlation_cycle(data_small, period=10)
    
    def test_correlation_cycle_very_small_dataset(self):
        """Test CORRELATION_CYCLE fails with insufficient data - mirrors check_cc_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.correlation_cycle(single_point, period=20)
    
    def test_correlation_cycle_empty_input(self):
        """Test CORRELATION_CYCLE fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.correlation_cycle(empty)
    
    def test_correlation_cycle_invalid_threshold(self):
        """Test CORRELATION_CYCLE with invalid threshold values"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        # Test with NaN threshold - should use default
        result = ta_indicators.correlation_cycle(data, period=3, threshold=float('nan'))
        assert len(result['real']) == len(data)
        
        # Test with negative threshold - should work (no restriction in Rust)
        result = ta_indicators.correlation_cycle(data, period=3, threshold=-1.0)
        assert len(result['real']) == len(data)
        
        # Test with zero threshold - should work
        result = ta_indicators.correlation_cycle(data, period=3, threshold=0.0)
        assert len(result['real']) == len(data)
    
    def test_correlation_cycle_reinput(self, test_data):
        """Test CORRELATION_CYCLE applied twice (re-input) - mirrors check_cc_reinput"""
        # Use smaller dataset for reinput test
        data = np.array([10.0, 10.5, 11.0, 11.5, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
        
        # First pass
        first_result = ta_indicators.correlation_cycle(data, period=4, threshold=2.0)
        assert len(first_result['real']) == len(data)
        
        # Second pass - apply correlation_cycle to real output
        second_result = ta_indicators.correlation_cycle(first_result['real'], period=4, threshold=2.0)
        assert len(second_result['real']) == len(data)
        
        # Both should have proper structure
        assert len(first_result['real']) == len(second_result['real'])
        assert len(first_result['imag']) == len(second_result['imag'])
    
    def test_correlation_cycle_all_nan_input(self):
        """Test CORRELATION_CYCLE with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.correlation_cycle(all_nan)
    
    def test_correlation_cycle_batch_accuracy(self, test_data):
        """Test CORRELATION_CYCLE batch processing accuracy - mirrors check_batch_default_row"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['correlation_cycle']
        
        # Test batch with default parameters only
        result = ta_indicators.correlation_cycle_batch(
            close,
            period_range=(20, 20, 0),  # Default period only
            threshold_range=(9.0, 9.0, 0.0)  # Default threshold only
        )
        
        assert 'real' in result
        assert 'imag' in result
        assert 'angle' in result
        assert 'state' in result
        assert 'periods' in result
        assert 'thresholds' in result
        
        # Should have 1 combination (default params)
        assert result['real'].shape[0] == 1
        assert result['real'].shape[1] == len(close)
        
        # Extract the single row for each output
        default_real = result['real'][0]
        default_imag = result['imag'][0]
        default_angle = result['angle'][0]
        
        # Check last 5 values match expected
        # Match Rust test tolerance: absolute <= 1e-8 across arrays
        assert_close(
            default_real[-5:], expected['last_5_values']['real'], rtol=0.0, atol=1e-8,
            msg="Batch real output mismatch"
        )
        assert_close(
            default_imag[-5:], expected['last_5_values']['imag'], rtol=0.0, atol=1e-8,
            msg="Batch imag output mismatch"
        )
        assert_close(
            default_angle[-5:], expected['last_5_values']['angle'], rtol=0.0, atol=1e-8,
            msg="Batch angle output mismatch"
        )
    
    def test_correlation_cycle_batch_edge_cases(self, test_data):
        """Test batch edge cases"""
        close = test_data['close'][:50]  # Use smaller dataset
        
        # Single value sweep
        result = ta_indicators.correlation_cycle_batch(
            close,
            period_range=(10, 10, 1),
            threshold_range=(5.0, 5.0, 1.0)
        )
        
        # Should have 1 combination
        assert result['real'].shape[0] == 1
        assert result['periods'][0] == 10
        assert result['thresholds'][0] == 5.0
        
        # Step larger than range
        result = ta_indicators.correlation_cycle_batch(
            close,
            period_range=(10, 15, 10),  # Step larger than range
            threshold_range=(5.0, 5.0, 0.0)
        )
        
        # Should only have period=10
        assert result['real'].shape[0] == 1
        assert result['periods'][0] == 10
        
        # Empty data should throw
        with pytest.raises(ValueError):
            ta_indicators.correlation_cycle_batch(
                np.array([]),
                period_range=(10, 10, 0),
                threshold_range=(5.0, 5.0, 0.0)
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
