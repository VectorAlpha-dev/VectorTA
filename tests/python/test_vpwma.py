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
    
    def test_vpwma_all_nan_input(self):
        """Test VPWMA with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.vpwma(all_nan, period=14, power=0.382)
    
    def test_vpwma_kernel_selection(self, test_data):
        """Test VPWMA with different kernel selections"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        # Test different kernels
        kernels = ['auto', 'scalar', 'avx2', 'avx512']
        results = {}
        
        for kernel in kernels:
            try:
                results[kernel] = ta_indicators.vpwma(
                    close, 
                    period=14, 
                    power=0.382,
                    kernel=kernel
                )
            except ValueError as e:
                # Some kernels might not be available on all systems
                if "Unknown kernel" not in str(e):
                    raise
        
        # All available kernels should produce similar results
        if 'scalar' in results:
            for kernel, result in results.items():
                if kernel != 'scalar':
                    assert_close(
                        result,
                        results['scalar'],
                        rtol=1e-10,
                        msg=f"Kernel {kernel} mismatch with scalar"
                    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])