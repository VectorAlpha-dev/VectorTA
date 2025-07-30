"""
Python binding tests for STC indicator.
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


class TestStc:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_stc_default_params(self, test_data):
        """Test STC with default parameters"""
        close = test_data['close']
        
        # Default params: fast=23, slow=50, k=10, d=3, fast_ma_type="ema", slow_ma_type="ema"
        result = ta_indicators.stc(close)
        assert len(result) == len(close)
        
        # Check that we have some valid values after warmup
        warmup = 50  # max of default periods
        assert not np.all(np.isnan(result[warmup:]))
    
    def test_stc_with_params(self, test_data):
        """Test STC with custom parameters"""
        close = test_data['close']
        
        result = ta_indicators.stc(
            close,
            fast_period=12,
            slow_period=26,
            k_period=9,
            d_period=3,
            fast_ma_type="sma",
            slow_ma_type="sma"
        )
        
        assert len(result) == len(close)
    
    def test_stc_accuracy(self, test_data):
        """Test STC accuracy with expected values if available"""
        close = test_data['close']
        
        # Use default parameters
        result = ta_indicators.stc(close, 23, 50, 10, 3, "ema", "ema")
        assert len(result) == len(close)
        
        # Check range - STC should be between 0 and 100
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= -0.1), "STC values should be >= 0"
        assert np.all(valid_values <= 100.1), "STC values should be <= 100"
        
        # Compare full output with Rust if expected values exist
        if 'stc' in EXPECTED_OUTPUTS:
            expected = EXPECTED_OUTPUTS['stc']
            assert_close(
                result[-5:], 
                expected['last_5_values'],
                rtol=1e-6,
                msg="STC last 5 values mismatch"
            )
            compare_with_rust('stc', result, 'close', expected['default_params'])
    
    def test_stc_zero_period(self):
        """Test STC fails with zero period"""
        input_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        
        with pytest.raises(ValueError, match="Not enough valid data"):
            ta_indicators.stc(input_data, fast_period=0, slow_period=50, k_period=10, d_period=3)
        
        with pytest.raises(ValueError, match="Not enough valid data"):
            ta_indicators.stc(input_data, fast_period=23, slow_period=0, k_period=10, d_period=3)
    
    def test_stc_period_exceeds_length(self):
        """Test STC fails when period exceeds data length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Not enough valid data"):
            ta_indicators.stc(data_small, fast_period=10, slow_period=50, k_period=10, d_period=3)
    
    def test_stc_empty_data(self):
        """Test STC fails with empty data"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.stc(empty)
    
    def test_stc_all_nan(self):
        """Test STC handles all NaN input"""
        all_nan = np.array([np.nan] * 100)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.stc(all_nan)
    
    def test_stc_nan_handling(self, test_data):
        """Test STC handles NaN values correctly"""
        close = test_data['close'].copy()
        
        # Insert some NaN values
        close[10:20] = np.nan
        
        result = ta_indicators.stc(close)
        assert len(result) == len(close)
        
        # Should still produce some valid values after the NaN section
        assert not np.all(np.isnan(result[100:]))
    
    def test_stc_stream(self, test_data):
        """Test STC streaming functionality"""
        close = test_data['close']
        
        # Create stream with default parameters
        stream = ta_indicators.StcStream(23, 50, 10, 3)
        
        # Process values one by one
        results = []
        for value in close[:100]:  # Test first 100 values
            result = stream.update(value)
            results.append(result if result is not None else np.nan)
        
        # Convert to numpy array
        stream_result = np.array(results)
        
        # Compare with batch calculation (allowing for warmup differences)
        batch_result = ta_indicators.stc(close[:100])
        
        # Find first non-NaN values in both
        stream_valid = np.where(~np.isnan(stream_result))[0]
        batch_valid = np.where(~np.isnan(batch_result))[0]
        
        if len(stream_valid) > 0 and len(batch_valid) > 0:
            # Compare where both have valid values
            start_idx = max(stream_valid[0], batch_valid[0])
            assert_close(
                stream_result[start_idx:],
                batch_result[start_idx:],
                rtol=1e-6,
                msg="Stream vs batch mismatch"
            )
    
    def test_stc_batch(self, test_data):
        """Test STC batch processing"""
        close = test_data['close']
        
        result = ta_indicators.stc_batch(
            close,
            fast_period_range=(20, 30, 5),    # 20, 25, 30
            slow_period_range=(45, 55, 5),    # 45, 50, 55  
            k_period_range=(10, 10, 1),       # 10 only
            d_period_range=(3, 3, 1)          # 3 only
        )
        
        assert 'values' in result
        assert 'fast_periods' in result
        assert 'slow_periods' in result
        assert 'k_periods' in result
        assert 'd_periods' in result
        
        # Should have 3 * 3 * 1 * 1 = 9 combinations
        assert result['values'].shape[0] == 9
        assert result['values'].shape[1] == len(close)
        
        # Verify parameters
        assert len(result['fast_periods']) == 9
        assert len(result['slow_periods']) == 9
        assert len(result['k_periods']) == 9
        assert len(result['d_periods']) == 9
        
        # Check first combination matches single calculation
        first_row = result['values'][0, :]
        single_result = ta_indicators.stc(close, 20, 45, 10, 3, "ema", "ema")
        
        assert_close(
            first_row,
            single_result,
            rtol=1e-10,
            msg="Batch first row vs single calculation mismatch"
        )
    
    def test_stc_batch_single_param(self, test_data):
        """Test STC batch with single parameter combination"""
        close = test_data['close']
        
        result = ta_indicators.stc_batch(
            close,
            fast_period_range=(23, 23, 0),
            slow_period_range=(50, 50, 0),
            k_period_range=(10, 10, 0),
            d_period_range=(3, 3, 0)
        )
        
        # Should have exactly 1 combination
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        # Should match single calculation
        single_result = ta_indicators.stc(close)
        assert_close(
            result['values'][0, :],
            single_result,
            rtol=1e-10,
            msg="Batch single param vs single calculation mismatch"
        )
    
    def test_stc_kernel_parameter(self, test_data):
        """Test STC with different kernel parameters"""
        close = test_data['close']
        
        # Test with explicit scalar kernel
        result_scalar = ta_indicators.stc(close, kernel="scalar")
        assert len(result_scalar) == len(close)
        
        # Test with auto kernel (default)
        result_auto = ta_indicators.stc(close, kernel=None)
        assert len(result_auto) == len(close)
        
        # Results might differ slightly due to SIMD optimizations
        # but should be very close
        assert_close(
            result_scalar[~np.isnan(result_scalar)],
            result_auto[~np.isnan(result_auto)],
            rtol=1e-10,
            msg="Scalar vs auto kernel mismatch"
        )


if __name__ == "__main__":
    pytest.main([__file__])