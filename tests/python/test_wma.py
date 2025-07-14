"""
Python binding tests for WMA indicator.
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


class TestWma:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_wma_partial_params(self, test_data):
        """Test WMA with partial parameters (None values) - mirrors check_wma_partial_params"""
        close = test_data['close']
        
        # Test with default params (period=30)
        result = ta_indicators.wma(close, 30)  # Using default
        assert len(result) == len(close)
    
    def test_wma_accuracy(self, test_data):
        """Test WMA matches expected values from Rust tests - mirrors check_wma_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['wma']
        
        result = ta_indicators.wma(
            close,
            period=expected['default_params']['period']
        )
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-6,  # Using 1e-6 as per Rust test
            msg="WMA last 5 values mismatch"
        )
    
    def test_wma_zero_period(self):
        """Test WMA fails with zero period - mirrors check_wma_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.wma(input_data, period=0)
    
    def test_wma_period_exceeds_length(self):
        """Test WMA fails when period exceeds data length - mirrors check_wma_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.wma(data_small, period=10)
    
    def test_wma_very_small_dataset(self):
        """Test WMA fails with insufficient data - mirrors check_wma_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.wma(single_point, period=9)
    
    def test_wma_reinput(self, test_data):
        """Test WMA applied twice (re-input) - mirrors check_wma_reinput"""
        close = test_data['close']
        
        # First pass
        first_result = ta_indicators.wma(close, period=14)
        assert len(first_result) == len(close)
        
        # Second pass - apply WMA to WMA output
        second_result = ta_indicators.wma(first_result, period=5)
        assert len(second_result) == len(first_result)
        
        # Check that values after warmup are not NaN
        if len(second_result) > 50:
            for i in range(50, len(second_result)):
                assert not np.isnan(second_result[i])
    
    def test_wma_nan_handling(self, test_data):
        """Test WMA handles NaN values correctly - mirrors check_wma_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.wma(close, period=14)
        assert len(result) == len(close)
        
        # After warmup period, no NaN values should exist
        if len(result) > 50:
            assert not np.any(np.isnan(result[50:])), "Found unexpected NaN after warmup period"
        
        # First period-1 values should be NaN (warmup is period-1 for WMA)
        # For period=14, indices 0-12 should be NaN
        assert np.all(np.isnan(result[:13])), "Expected NaN in warmup period"
        # First valid value should be at index period-1 = 13
        assert not np.isnan(result[13]), "Expected valid value at index 13"
    
    def test_wma_streaming(self, test_data):
        """Test WMA streaming matches batch calculation - mirrors check_wma_streaming"""
        close = test_data['close']
        period = 30
        
        # Batch calculation
        batch_result = ta_indicators.wma(close, period=period)
        
        # Streaming calculation
        stream = ta_indicators.WmaStream(period=period)
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
            assert_close(b, s, rtol=1e-8, atol=1e-8, 
                        msg=f"WMA streaming mismatch at index {i}")
    
    def test_wma_batch(self, test_data):
        """Test WMA batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        result = ta_indicators.wma_batch(
            close,
            period_range=(30, 30, 0)  # Default period only
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        # Should have 1 combination (default params)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        # Extract the single row
        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['wma']['last_5_values']
        
        # Check last 5 values match
        assert_close(
            default_row[-5:],
            expected,
            rtol=1e-6,
            msg="WMA batch default row mismatch"
        )
    
    def test_wma_all_nan_input(self):
        """Test WMA with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.wma(all_nan, period=30)
    
    def test_wma_kernel_selection(self, test_data):
        """Test WMA with different kernel selections"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        # Test different kernels
        kernels = ['auto', 'scalar', 'avx2', 'avx512']
        results = {}
        
        for kernel in kernels:
            try:
                results[kernel] = ta_indicators.wma(
                    close, 
                    period=30,
                    kernel=kernel
                )
            except ValueError as e:
                # Some kernels might not be available on all systems
                if "Unknown kernel" not in str(e) and "not available on this CPU" not in str(e) and "not compiled in this build" not in str(e):
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
