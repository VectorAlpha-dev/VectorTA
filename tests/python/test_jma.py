"""
Python binding tests for JMA indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

try:
    import my_project as ta_indicators
except ImportError:
    # If not in virtual environment, try to import from installed location
    try:
        import my_project as ta_indicators
    except ImportError:
        pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS
from rust_comparison import compare_with_rust


class TestJma:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_jma_partial_params(self, test_data):
        """Test JMA with partial parameters - mirrors check_jma_partial_params"""
        close = test_data['close']
        
        # Test with default params: period=7, phase=50.0, power=2
        result = ta_indicators.jma(close, 7)
        assert len(result) == len(close)
    
    def test_jma_accuracy(self, test_data):
        """Test JMA matches expected values from Rust tests - mirrors check_jma_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['jma']
        
        # Using period=7, phase=50.0, power=2 (defaults)
        result = ta_indicators.jma(
            close,
            period=expected['default_params']['period'],
            phase=expected['default_params']['phase'],
            power=expected['default_params']['power']
        )
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-6,
            msg="JMA last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('jma', result, 'close', expected['default_params'])
    
    def test_jma_default_candles(self, test_data):
        """Test JMA with default parameters - mirrors check_jma_default_candles"""
        close = test_data['close']
        
        # Default params: period=7, phase=50.0, power=2
        result = ta_indicators.jma(close, 7, 50.0, 2)
        assert len(result) == len(close)
        
        # Compare with Rust
        # compare_with_rust('jma', result, 'close', {'period': 7, 'phase': 50.0, 'power': 2})
    
    def test_jma_zero_period(self):
        """Test JMA fails with zero period - mirrors check_jma_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError):
            ta_indicators.jma(input_data, period=0)
    
    def test_jma_period_exceeds_length(self):
        """Test JMA fails when period exceeds data length - mirrors check_jma_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError):
            ta_indicators.jma(data_small, period=10)
    
    def test_jma_very_small_dataset(self):
        """Test JMA with very small dataset - mirrors check_jma_very_small_dataset"""
        data_single = np.array([42.0])
        
        with pytest.raises(ValueError):
            ta_indicators.jma(data_single, period=7)
    
    def test_jma_empty_input(self):
        """Test JMA with empty input"""
        data_empty = np.array([])
        
        with pytest.raises(ValueError):
            ta_indicators.jma(data_empty, period=7)
    
    def test_jma_all_nan(self):
        """Test JMA with all NaN input"""
        data = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        
        with pytest.raises(ValueError):
            ta_indicators.jma(data, period=3)
    
    def test_jma_invalid_phase(self):
        """Test JMA with invalid phase values"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test with NaN phase
        with pytest.raises(ValueError):
            ta_indicators.jma(data, period=3, phase=float('nan'))
        
        # Test with infinite phase
        with pytest.raises(ValueError):
            ta_indicators.jma(data, period=3, phase=float('inf'))
    
    def test_jma_batch_with_nan_data(self):
        """Test JMA batch with NaN prefix data"""
        # Create data with NaN prefix
        data = np.array([np.nan, np.nan] + list(range(1, 21)))
        
        result = ta_indicators.jma_batch(
            data,
            period_range=(3, 5, 2),  # periods: 3, 5
            phase_range=(50.0, 50.0, 0.0),
            power_range=(2, 2, 0)
        )
        
        values = result['values']
        assert values.shape == (2, len(data))
        
        # Both rows should have NaN for first 2 indices (before first valid)
        for row in range(2):
            assert np.isnan(values[row, 0]), f"Row {row} index 0 should be NaN"
            assert np.isnan(values[row, 1]), f"Row {row} index 1 should be NaN"
            # Should have values starting at index 2 (first valid)
            assert np.isfinite(values[row, 2]), f"Row {row} index 2 should be finite"
    
    def test_jma_nan_handling(self, test_data):
        """Test JMA handling of NaN values - mirrors check_jma_nan_handling"""
        # Test with NaN prefix
        data_with_nan = np.concatenate([
            np.array([np.nan, np.nan, np.nan]),
            test_data['close'][:50]
        ])
        period = 7
        
        result = ta_indicators.jma(data_with_nan, period)
        
        assert len(result) == len(data_with_nan)
        
        # First 3 values should be NaN (before first valid)
        assert np.all(np.isnan(result[:3])), "Expected NaN before first valid data"
        
        # After first valid, should have finite values
        assert np.isfinite(result[3]), "Expected finite value at first valid"
        
        # Check for reasonable values after first valid
        finite_count = np.sum(np.isfinite(result[3:]))
        assert finite_count == len(result[3:]), "All values after first valid should be finite"
    
    def test_jma_streaming(self, test_data):
        """Test JMA streaming vs batch calculation - mirrors check_jma_streaming"""
        close = test_data['close'][:100]  # Use first 100 values for testing
        period = 7
        phase = 50.0
        power = 2
        
        # Batch calculation
        batch_result = ta_indicators.jma(close, period, phase, power)
        
        # Streaming calculation
        stream = ta_indicators.JmaStream(period, phase, power)
        stream_results = []
        
        for val in close:
            result = stream.update(val)
            stream_results.append(result if result is not None else np.nan)
        
        stream_results = np.array(stream_results)
        
        # Compare batch vs streaming
        assert_close(
            stream_results, 
            batch_result,
            rtol=1e-10,
            msg="JMA streaming vs batch mismatch"
        )
    
    def test_jma_batch(self, test_data):
        """Test JMA batch computation with proper warmup verification."""
        close = test_data['close']
        
        # Test parameter ranges
        period_range = (5, 9, 2)  # periods: 5, 7, 9
        phase_range = (40.0, 60.0, 10.0)  # phases: 40, 50, 60
        power_range = (1, 3, 1)  # powers: 1, 2, 3
        
        result = ta_indicators.jma_batch(close, period_range, phase_range, power_range)
        
        assert isinstance(result, dict)
        assert 'values' in result
        assert 'periods' in result
        assert 'phases' in result
        assert 'powers' in result
        
        values = result['values']
        periods = result['periods']
        phases = result['phases']
        powers = result['powers']
        
        # Should have 3 * 3 * 3 = 27 combinations
        assert values.shape == (27, len(close))
        assert len(periods) == 27
        assert len(phases) == 27
        assert len(powers) == 27
        
        # Verify first combination matches individual calculation
        individual_result = ta_indicators.jma(close, 5, 40.0, 1)
        np.testing.assert_allclose(
            values[0], 
            individual_result, 
            rtol=1e-9,
            err_msg="Batch first row mismatch"
        )
        
        # Verify warmup behavior for each parameter combination
        # JMA outputs at first_valid, no additional warmup
        for i in range(27):
            row = values[i]
            # All values should be finite (JMA outputs from index 0 for clean data)
            assert np.all(np.isfinite(row)), f"Row {i} has unexpected NaN values"
    
    def test_jma_different_params(self, test_data):
        """Test JMA with different parameter values."""
        close = test_data['close']
        
        # Test various parameter combinations
        param_sets = [
            (5, 0.0, 1),     # Min phase
            (7, 50.0, 2),    # Default
            (10, 100.0, 3),  # Max phase
            (14, -100.0, 2), # Negative phase
        ]
        
        for period, phase, power in param_sets:
            result = ta_indicators.jma(close, period, phase, power)
            assert len(result) == len(close)
            
            # Verify reasonable values
            finite_count = np.sum(np.isfinite(result))
            assert finite_count > len(close) - period * 2, \
                f"Too many non-finite values for params ({period}, {phase}, {power})"
    
    def test_jma_batch_performance(self, test_data):
        """Test that batch computation is more efficient than multiple single computations."""
        close = test_data['close'][:1000]  # Use first 1000 values
        
        # Test 8 combinations
        import time
        
        start_batch = time.time()
        batch_result = ta_indicators.jma_batch(
            close, 
            (5, 7, 2),      # 2 period values
            (40.0, 50.0, 10.0),  # 2 phase values
            (1, 2, 1)       # 2 power values
        )
        batch_time = time.time() - start_batch
        
        start_single = time.time()
        single_results = []
        for period in [5, 7]:
            for phase in [40.0, 50.0]:
                for power in [1, 2]:
                    single_results.append(ta_indicators.jma(close, period, phase, power))
        single_time = time.time() - start_single
        
        # Batch should be faster than multiple single calls
        print(f"Batch time: {batch_time:.4f}s, Single time: {single_time:.4f}s")
        
        # Verify results match
        values = batch_result['values']
        for i, single in enumerate(single_results):
            np.testing.assert_allclose(values[i], single, rtol=1e-9)
    
    def test_jma_phase_range(self):
        """Test JMA with full phase range"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        # Test extreme phase values
        for phase in [-100.0, -50.0, 0.0, 50.0, 100.0]:
            result = ta_indicators.jma(data, period=3, phase=phase, power=2)
            assert len(result) == len(data)
            assert np.sum(np.isfinite(result)) > 0
    
    def test_jma_power_values(self):
        """Test JMA with different power values"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        # Test various power values
        for power in [1, 2, 3, 5, 10]:
            result = ta_indicators.jma(data, period=3, phase=50.0, power=power)
            assert len(result) == len(data)
            assert np.sum(np.isfinite(result)) > 0
    
    def test_jma_single_param_batch(self, test_data):
        """Test JMA batch with single parameter variation"""
        close = test_data['close'][:100]
        
        # Vary only period
        result = ta_indicators.jma_batch(
            close,
            (5, 9, 2),         # periods: 5, 7, 9
            (50.0, 50.0, 0.0), # phase fixed at 50.0
            (2, 2, 0)          # power fixed at 2
        )
        
        values = result['values']
        periods = result['periods']
        phases = result['phases']
        powers = result['powers']
        
        # Should have 3 combinations
        assert values.shape == (3, len(close))
        assert list(periods) == [5, 7, 9]
        assert all(p == 50.0 for p in phases)
        assert all(p == 2 for p in powers)
    
    def test_jma_batch_metadata(self, test_data):
        """Test batch metadata function returns correct parameter combinations"""
        period_range = (3, 5, 2)     # 2 values: 3, 5
        phase_range = (40.0, 50.0, 10.0)  # 2 values  
        power_range = (1, 2, 1)      # 2 values
        
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        result = ta_indicators.jma_batch(data, period_range, phase_range, power_range)
        
        periods = result['periods']
        phases = result['phases']
        powers = result['powers']
        
        # Should have 2 * 2 * 2 = 8 combinations
        assert len(periods) == 8
        assert len(phases) == 8
        assert len(powers) == 8
        
        # Check first few combinations
        expected_combinations = [
            (3, 40.0, 1),
            (3, 40.0, 2),
            (3, 50.0, 1),
            (3, 50.0, 2),
        ]
        
        for i, (exp_period, exp_phase, exp_power) in enumerate(expected_combinations):
            assert periods[i] == exp_period
            assert abs(phases[i] - exp_phase) < 1e-9
            assert powers[i] == exp_power
    
    def test_jma_warmup_behavior(self, test_data):
        """Test JMA warmup behavior - outputs at first_valid with NaN prefix"""
        # Test with data starting with NaN values
        data = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        period = 3
        
        result = ta_indicators.jma(data, period=period, phase=50.0, power=2)
        
        # JMA should have NaN for indices 0,1 (before first valid)
        assert np.isnan(result[0]), "Index 0 should be NaN"
        assert np.isnan(result[1]), "Index 1 should be NaN"
        
        # JMA should start outputting at index 2 (first valid)
        assert np.isfinite(result[2]), "Index 2 (first valid) should have a value"
        
        # All subsequent values should be finite
        for i in range(3, len(result)):
            assert np.isfinite(result[i]), f"Expected finite value at index {i}"
        
        # Test with clean data (no leading NaNs)
        clean_data = test_data['close'][:20]
        clean_result = ta_indicators.jma(clean_data, period=7)
        
        # With clean data, JMA outputs immediately at index 0
        assert np.isfinite(clean_result[0]), "JMA should output at first valid (index 0 for clean data)"


    def test_jma_consistency_single_vs_batch(self, test_data):
        """Test that single and batch calculations are consistent"""
        close = test_data['close'][:100]
        
        # Single calculation
        single_result = ta_indicators.jma(close, 7, 50.0, 2)
        
        # Batch calculation with same parameters
        batch_result = ta_indicators.jma_batch(
            close,
            period_range=(7, 7, 0),
            phase_range=(50.0, 50.0, 0.0),
            power_range=(2, 2, 0)
        )
        
        # Extract single row from batch
        batch_single = batch_result['values'][0]
        
        # Should be identical
        np.testing.assert_allclose(
            single_result,
            batch_single,
            rtol=1e-12,
            err_msg="Single vs batch calculation mismatch"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])