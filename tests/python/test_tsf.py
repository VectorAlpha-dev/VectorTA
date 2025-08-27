"""
Python binding tests for TSF indicator.
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

from test_utils import load_test_data, assert_close
from rust_comparison import compare_with_rust


class TestTsf:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_tsf_partial_params(self, test_data):
        """Test TSF with partial parameters (None values) - mirrors check_tsf_partial_params"""
        close = test_data['close']
        
        # Test with default period (14)
        result = ta_indicators.tsf(close, 14)
        assert len(result) == len(close)
    
    def test_tsf_accuracy(self, test_data):
        """Test TSF matches expected values from Rust tests - mirrors check_tsf_accuracy"""
        close = test_data['close']
        
        # Expected values from Rust tests
        expected_last_five = [
            58846.945054945056,
            58818.83516483516,
            58854.57142857143,
            59083.846153846156,
            58962.25274725275,
        ]
        
        result = ta_indicators.tsf(close, 14)  # Default period
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected with small tolerance
        last_5 = result[-5:]
        for i, (actual, expected) in enumerate(zip(last_5, expected_last_five)):
            assert_close(actual, expected, atol=0.1, 
                        msg=f"TSF value mismatch at index {i}")
    
    def test_tsf_from_slice(self, test_data):
        """Test TSF from slice data - mirrors check_tsf_from_slice"""
        close = test_data['close']
        
        # Test with custom period
        result = ta_indicators.tsf(close, 20)
        assert len(result) == len(close)
    
    def test_tsf_zero_period(self, test_data):
        """Test TSF with zero period - should raise error"""
        close = test_data['close']
        
        with pytest.raises(ValueError, match="Period must be at least 2"):
            ta_indicators.tsf(close, 0)
    
    def test_tsf_period_one(self, test_data):
        """Test TSF with period=1 - should raise error"""
        close = test_data['close']
        
        with pytest.raises(ValueError, match="Period must be at least 2"):
            ta_indicators.tsf(close, 1)
    
    def test_tsf_period_exceeds_length(self):
        """Test TSF when period exceeds data length - should raise error"""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.tsf(data, 5)
    
    def test_tsf_very_small_dataset(self):
        """Test TSF with very small dataset"""
        data = np.array([100.0], dtype=np.float64)
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.tsf(data, 2)
    
    def test_tsf_nan_handling(self, test_data):
        """Test TSF handles NaN values correctly"""
        close = test_data['close'].copy()
        # Add some NaN values
        close[10:15] = np.nan
        
        result = ta_indicators.tsf(close, 14)
        assert len(result) == len(close)
        
        # Check that NaN values are handled (should not crash)
        # First few values should be NaN due to warmup period
        assert np.isnan(result[0])
    
    def test_tsf_streaming(self, test_data):
        """Test TSF streaming functionality - mirrors check_tsf_streaming"""
        close = test_data['close']
        period = 14
        
        # Batch calculation
        batch_result = ta_indicators.tsf(close, period)
        
        # Streaming calculation
        stream = ta_indicators.TsfStream(period)
        stream_result = []
        
        for value in close:
            result = stream.update(value)
            stream_result.append(result if result is not None else np.nan)
        
        # Compare batch vs streaming results
        # Allow small tolerance for floating point differences
        for i, (b, s) in enumerate(zip(batch_result, stream_result)):
            if not np.isnan(b) and not np.isnan(s):
                assert_close(b, s, atol=1e-10,
                           msg=f"Batch vs streaming mismatch at index {i}")
            else:
                # Both should be NaN
                assert np.isnan(b) and np.isnan(s), \
                    f"NaN mismatch at index {i}: batch={b}, stream={s}"
    
    def test_tsf_batch_operation(self, test_data):
        """Test TSF batch operations with parameter sweeps"""
        close = test_data['close'][:1000]  # Use smaller dataset for speed
        
        # Test batch with period range
        result = ta_indicators.tsf_batch(
            close,
            period_range=(10, 20, 2)  # periods: 10, 12, 14, 16, 18, 20
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        values = result['values']
        periods = result['periods']
        
        # Should have 6 parameter combinations
        assert len(periods) == 6
        assert values.shape == (6, len(close))
        
        # Check that periods are correct
        expected_periods = [10, 12, 14, 16, 18, 20]
        assert all(p == exp for p, exp in zip(periods, expected_periods))
    
    def test_tsf_kernel_options(self, test_data):
        """Test TSF with different kernel options"""
        close = test_data['close'][:500]  # Smaller dataset
        
        # Test with different kernels
        kernels = ["scalar", "auto"]  # AVX kernels will be stubs
        
        results = {}
        for kernel in kernels:
            try:
                results[kernel] = ta_indicators.tsf(close, 14, kernel=kernel)
            except ValueError:
                # Some kernels might not be available
                continue
        
        # All available kernels should produce same results
        if len(results) > 1:
            base_kernel = list(results.keys())[0]
            base_result = results[base_kernel]
            
            for kernel, result in results.items():
                if kernel != base_kernel:
                    for i, (a, b) in enumerate(zip(base_result, result)):
                        if not np.isnan(a) and not np.isnan(b):
                            assert_close(a, b, atol=1e-10,
                                       msg=f"Kernel {kernel} differs at index {i}")
    
    def test_tsf_rust_comparison(self, test_data):
        """Compare Python binding output with direct Rust implementation"""
        close = test_data['close']
        
        # Test with default parameters (period=14) since generate_references uses defaults
        result = ta_indicators.tsf(close, 14)
        compare_with_rust('tsf', result, 'close', {'period': 14})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])