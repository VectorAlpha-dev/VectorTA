"""
Python binding tests for VPWMA indicator.
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


class TestVpwma:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_vpwma_partial_params(self, test_data):
        """Test VPWMA with partial parameters (None values) - mirrors check_vpwma_partial_params"""
        close = test_data['close']
        
        # Test with all default params
        result = ta_indicators.vpwma(close, 14, 0.382)  # Using defaults
        assert len(result) == len(close)
    
    def test_vpwma_accuracy(self, test_data):
        """Test VPWMA matches expected values from Rust tests - mirrors check_vpwma_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['vpwma']
        
        result = ta_indicators.vpwma(
            close,
            period=expected['default_params']['period'],
            power=expected['default_params']['power']
        )
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-4,  # Using 1e-4 as per Rust test which uses 1e-2
            msg="VPWMA last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('vpwma', result, 'close', expected['default_params'])
    
    def test_vpwma_zero_period(self):
        """Test VPWMA fails with zero period - mirrors check_vpwma_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.vpwma(input_data, period=0, power=0.382)
    
    def test_vpwma_period_exceeds_length(self):
        """Test VPWMA fails when period exceeds data length - mirrors check_vpwma_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.vpwma(data_small, period=10, power=0.382)
    
    def test_vpwma_very_small_dataset(self):
        """Test VPWMA fails with insufficient data - mirrors check_vpwma_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.vpwma(single_point, period=2, power=0.382)
    
    def test_vpwma_empty_input(self):
        """Test VPWMA fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.vpwma(empty, period=14, power=0.382)
    
    def test_vpwma_invalid_power(self):
        """Test VPWMA fails with invalid power - mirrors vpwma power validation"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test with NaN power
        with pytest.raises(ValueError, match="Invalid power"):
            ta_indicators.vpwma(data, period=2, power=float('nan'))
        
        # Test with infinite power
        with pytest.raises(ValueError, match="Invalid power"):
            ta_indicators.vpwma(data, period=2, power=float('inf'))
    
    def test_vpwma_reinput(self, test_data):
        """Test VPWMA applied twice (re-input) - mirrors check_vpwma_reinput"""
        close = test_data['close']
        
        # First pass
        first_result = ta_indicators.vpwma(close, period=14, power=0.382)
        assert len(first_result) == len(close)
        
        # Second pass - apply VPWMA to VPWMA output
        second_result = ta_indicators.vpwma(first_result, period=5, power=0.5)
        assert len(second_result) == len(first_result)
        
        # Check that values after warmup are not NaN
        if len(second_result) > 240:
            for i in range(240, len(second_result)):
                assert not np.isnan(second_result[i])
    
    def test_vpwma_nan_handling(self, test_data):
        """Test VPWMA handles NaN values correctly - mirrors check_vpwma_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.vpwma(close, period=14, power=0.382)
        assert len(result) == len(close)
        
        # After warmup period, no NaN values should exist
        if len(result) > 50:
            assert not np.any(np.isnan(result[50:])), "Found unexpected NaN after warmup period"
        
        # First period-1 values should be NaN
        assert np.all(np.isnan(result[:13])), "Expected NaN in warmup period"
    
    def test_vpwma_warmup_period(self, test_data):
        """Test VPWMA warmup period behavior - matches real-world conditions"""
        close = test_data['close']
        
        # Test with different periods
        test_cases = [
            {'period': 5, 'power': 0.5, 'expected_warmup': 4},
            {'period': 10, 'power': 0.382, 'expected_warmup': 9},
            {'period': 14, 'power': 0.382, 'expected_warmup': 13},
            {'period': 20, 'power': 0.3, 'expected_warmup': 19}
        ]
        
        for tc in test_cases:
            result = ta_indicators.vpwma(close, period=tc['period'], power=tc['power'])
            
            # Check warmup period has NaN values
            warmup_values = result[:tc['expected_warmup']]
            assert np.all(np.isnan(warmup_values)), \
                f"Expected all NaN in warmup period for period={tc['period']}, got non-NaN values"
            
            # Check first non-NaN value appears at expected index
            if len(result) > tc['expected_warmup']:
                assert not np.isnan(result[tc['expected_warmup']]), \
                    f"Expected first valid value at index {tc['expected_warmup']} for period={tc['period']}"
            
            # Verify warmup calculation: warmup = first_valid + period - 1
            # For clean data starting at index 0, warmup = 0 + period - 1 = period - 1
            assert tc['expected_warmup'] == tc['period'] - 1, \
                f"Warmup calculation mismatch for period={tc['period']}"
    
    def test_vpwma_streaming(self, test_data):
        """Test VPWMA streaming matches batch calculation - mirrors check_vpwma_streaming"""
        close = test_data['close']
        period = 14
        power = 0.382
        
        # Batch calculation
        batch_result = ta_indicators.vpwma(close, period=period, power=power)
        
        # Streaming calculation
        stream = ta_indicators.VpwmaStream(period=period, power=power)
        stream_values = []
        
        for price in close:
            result = stream.update(price)
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        # Compare batch vs streaming
        assert len(batch_result) == len(stream_values)
        
        # Compare values where both are not NaN
        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, atol=1e-9, 
                        msg=f"VPWMA streaming mismatch at index {i}")
    
    def test_vpwma_batch(self, test_data):
        """Test VPWMA batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        result = ta_indicators.vpwma_batch(
            close,
            period_range=(14, 14, 0),  # Default period only
            power_range=(0.382, 0.382, 0.0)  # Default power only
        )
        
        assert 'values' in result
        assert 'periods' in result
        assert 'powers' in result
        
        # Should have 1 combination (default params)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        # Extract the single row
        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['vpwma']['last_5_values']
        
        # Check last 5 values match
        assert_close(
            default_row[-5:],
            expected,
            rtol=1e-4,
            msg="VPWMA batch default row mismatch"
        )
    
    def test_vpwma_batch_multiple_params(self, test_data):
        """Test VPWMA batch with multiple parameter combinations"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        # Test with multiple periods and powers
        result = ta_indicators.vpwma_batch(
            close,
            period_range=(10, 14, 2),    # periods: 10, 12, 14
            power_range=(0.3, 0.4, 0.1)  # powers: 0.3, 0.4
        )
        
        assert 'values' in result
        assert 'periods' in result
        assert 'powers' in result
        
        # Should have 3 * 2 = 6 combinations
        assert result['values'].shape[0] == 6
        assert result['values'].shape[1] == 100
        assert len(result['periods']) == 6
        assert len(result['powers']) == 6
        
        # Verify parameter combinations
        expected_periods = [10, 10, 12, 12, 14, 14]
        expected_powers = [0.3, 0.4, 0.3, 0.4, 0.3, 0.4]
        
        np.testing.assert_array_equal(result['periods'], expected_periods)
        np.testing.assert_array_almost_equal(result['powers'], expected_powers, decimal=10)
        
        # Verify each row matches individual calculation
        for i in range(6):
            row_data = result['values'][i]
            period = result['periods'][i]
            power = result['powers'][i]
            
            # Calculate single result for comparison
            single_result = ta_indicators.vpwma(close, period=period, power=power)
            
            assert_close(
                row_data,
                single_result,
                rtol=1e-10,
                msg=f"Batch row {i} (period={period}, power={power}) mismatch"
            )
    
    def test_vpwma_batch_edge_cases(self, test_data):
        """Test VPWMA batch edge cases"""
        close = test_data['close'][:50]
        
        # Test 1: Single parameter sweep
        result = ta_indicators.vpwma_batch(
            close,
            period_range=(14, 14, 1),
            power_range=(0.382, 0.382, 0.1)
        )
        assert result['values'].shape[0] == 1
        
        # Test 2: Step larger than range
        result = ta_indicators.vpwma_batch(
            close,
            period_range=(10, 12, 5),  # Step > range
            power_range=(0.3, 0.3, 0)
        )
        # Should only have period=10
        assert result['values'].shape[0] == 1
        assert result['periods'][0] == 10
        
        # Test 3: Zero step (single value)
        result = ta_indicators.vpwma_batch(
            close,
            period_range=(14, 14, 0),
            power_range=(0.382, 0.382, 0)
        )
        assert result['values'].shape[0] == 1
        
        # Test 4: Multiple periods, single power
        result = ta_indicators.vpwma_batch(
            close,
            period_range=(10, 14, 2),  # 3 periods
            power_range=(0.382, 0.382, 0)  # 1 power
        )
        assert result['values'].shape[0] == 3
        assert len(np.unique(result['powers'])) == 1
    
    def test_vpwma_all_nan_input(self):
        """Test VPWMA with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.vpwma(all_nan, period=14, power=0.382)
    
    def test_vpwma_batch_invalid_params(self):
        """Test VPWMA batch with invalid parameters"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        # Test with period exceeding data length
        with pytest.raises(ValueError, match="Not enough valid data|Invalid period"):
            ta_indicators.vpwma_batch(
                data,
                period_range=(20, 25, 5),  # Periods too large
                power_range=(0.382, 0.382, 0)
            )
        
        # Test with invalid power (NaN)
        with pytest.raises(ValueError, match="Invalid power"):
            ta_indicators.vpwma_batch(
                data,
                period_range=(5, 5, 0),
                power_range=(float('nan'), float('nan'), 0)
            )
        
        # Test with empty data
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.vpwma_batch(
                np.array([]),
                period_range=(14, 14, 0),
                power_range=(0.382, 0.382, 0)
            )
    
    def test_vpwma_batch_metadata(self, test_data):
        """Test VPWMA batch metadata extraction"""
        close = test_data['close'][:100]
        
        result = ta_indicators.vpwma_batch(
            close,
            period_range=(12, 16, 2),    # 3 periods: 12, 14, 16
            power_range=(0.3, 0.5, 0.2)  # 2 powers: 0.3, 0.5
        )
        
        # Should have complete metadata
        assert 'values' in result
        assert 'periods' in result
        assert 'powers' in result
        
        # Verify dimensions match
        num_combos = len(result['periods'])
        assert num_combos == 6  # 3 periods * 2 powers
        assert result['values'].shape == (num_combos, len(close))
        
        # Verify warmup periods are correct for each combination
        for i in range(num_combos):
            period = result['periods'][i]
            row_data = result['values'][i]
            expected_warmup = period - 1
            
            # Check NaN in warmup period
            assert np.all(np.isnan(row_data[:expected_warmup])), \
                f"Expected NaN in warmup for combo {i} (period={period})"
            
            # Check values after warmup
            if len(row_data) > expected_warmup:
                assert not np.isnan(row_data[expected_warmup]), \
                    f"Expected valid value after warmup for combo {i}"
    
    def test_vpwma_kernel_selection(self, test_data):
        """Test VPWMA with different kernel selections"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        # Test different kernels
        kernels = ['auto', 'scalar', 'avx2', 'avx512']
        results = {}
        available_kernels = []
        
        for kernel in kernels:
            try:
                results[kernel] = ta_indicators.vpwma(
                    close, 
                    period=14, 
                    power=0.382,
                    kernel=kernel
                )
                available_kernels.append(kernel)
            except ValueError as e:
                # Some kernels might not be available or compiled on this system/build
                msg = str(e)
                allowed = (
                    "Unknown kernel" in msg or
                    "not available" in msg.lower() or
                    "not compiled in this build" in msg
                )
                if not allowed:
                    raise
        
        # Should have at least scalar and auto
        assert 'auto' in available_kernels, "Auto kernel should always be available"
        assert 'scalar' in available_kernels, "Scalar kernel should always be available"
        
        # All available kernels should produce similar results
        if len(results) > 1:
            scalar_result = results['scalar']
            for kernel, result in results.items():
                if kernel != 'scalar':
                    # Compare non-NaN values only
                    valid_mask = ~np.isnan(scalar_result)
                    assert_close(
                        result[valid_mask],
                        scalar_result[valid_mask],
                        rtol=1e-9,  # Slightly relaxed for different SIMD implementations
                        msg=f"Kernel {kernel} mismatch with scalar"
                    )
        
        # Verify auto kernel selects something
        if 'auto' in results:
            assert len(results['auto']) == len(close), "Auto kernel should produce full output"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
