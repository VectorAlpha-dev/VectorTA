"""
Python binding tests for ZLEMA indicator.
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


class TestZlema:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_zlema_partial_params(self, test_data):
        """Test ZLEMA with default parameters - mirrors check_zlema_partial_params"""
        data = test_data['close']
        
        # Test with default period (None maps to 14)
        result = ta_indicators.zlema(data, period=14)
        assert len(result) == len(data)
    
    def test_zlema_accuracy(self, test_data):
        """Test ZLEMA matches expected values from Rust tests - mirrors check_zlema_accuracy"""
        data = test_data['close']
        expected = EXPECTED_OUTPUTS['zlema']
        
        result = ta_indicators.zlema(
            data,
            period=expected['default_params']['period']
        )
        
        assert len(result) == len(data)
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-1,  # Using 1e-1 as in Rust test
            atol=1e-1,
            msg="ZLEMA last 5 values mismatch"
        )
    
    def test_zlema_zero_period(self):
        """Test ZLEMA fails with zero period - mirrors check_zlema_zero_period"""
        data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.zlema(data, period=0)
    
    def test_zlema_period_exceeds_length(self):
        """Test ZLEMA fails when period exceeds data length - mirrors check_zlema_period_exceeds_length"""
        data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.zlema(data, period=10)
    
    def test_zlema_very_small_dataset(self):
        """Test ZLEMA fails with insufficient data - mirrors check_zlema_very_small_dataset"""
        data = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.zlema(data, period=14)
    
    def test_zlema_reinput(self, test_data):
        """Test ZLEMA re-input behavior - mirrors check_zlema_reinput"""
        data = test_data['close']
        
        # First pass with period 21
        first_result = ta_indicators.zlema(data, period=21)
        
        # Second pass with period 14 on first result
        second_result = ta_indicators.zlema(first_result, period=14)
        
        assert len(second_result) == len(first_result)
        
        # Check that values after warmup period are finite
        for idx, val in enumerate(second_result[14:], start=14):
            assert np.isfinite(val), f"NaN found at index {idx}"
    
    def test_zlema_nan_handling(self, test_data):
        """Test ZLEMA NaN handling - mirrors check_zlema_nan_handling"""
        data = test_data['close']
        
        result = ta_indicators.zlema(data, period=14)
        assert len(result) == len(data)
        
        # Check that values after warmup period don't have NaN
        if len(result) > 20:
            for i, val in enumerate(result[20:], start=20):
                assert not np.isnan(val), f"Found unexpected NaN at index {i}"
    
    def test_zlema_streaming(self, test_data):
        """Test ZLEMA streaming functionality"""
        data = test_data['close']
        
        # Batch calculation
        batch_result = ta_indicators.zlema(data, period=14)
        
        # Streaming calculation
        stream = ta_indicators.ZlemaStream(period=14)
        stream_values = []
        
        for value in data:
            result = stream.update(value)
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        # Compare batch vs streaming
        assert len(batch_result) == len(stream_values)
        
        # Compare values where both are not NaN
        valid_mask = ~np.isnan(batch_result) & ~np.isnan(stream_values)
        assert_close(
            batch_result[valid_mask], 
            stream_values[valid_mask], 
            rtol=1e-9, 
            atol=1e-9,
            msg="ZLEMA streaming mismatch"
        )
    
    def test_zlema_batch(self, test_data):
        """Test ZLEMA batch processing - mirrors check_batch_default_row"""
        data = test_data['close']
        
        result = ta_indicators.zlema_batch(
            data,
            period_range=(14, 40, 1)  # Default batch range from Rust
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        # Should have 27 combinations: 14, 15, ..., 40
        assert result['values'].shape[0] == 27
        assert result['values'].shape[1] == len(data)
        assert len(result['periods']) == 27
        
        # Check that period 14 row matches single ZLEMA calculation
        idx_14 = list(result['periods']).index(14)
        single_zlema = ta_indicators.zlema(data, period=14)
        assert_close(
            result['values'][idx_14],
            single_zlema,
            rtol=1e-9,
            msg="ZLEMA batch period 14 row mismatch"
        )
    
    def test_zlema_kernel_selection(self, test_data):
        """Test ZLEMA with different kernel selections"""
        data = test_data['close']
        
        # Test with auto kernel (default)
        result_auto = ta_indicators.zlema(data, period=14)
        
        # Test with scalar kernel
        result_scalar = ta_indicators.zlema(data, period=14, kernel="scalar")
        
        # Results should be identical (within tolerance) since AVX2/AVX512 are stubs
        assert_close(
            result_auto,
            result_scalar,
            rtol=1e-9,
            msg="ZLEMA kernel results mismatch"
        )
        
        # Test with invalid kernel
        with pytest.raises(ValueError, match="Unknown kernel"):
            ta_indicators.zlema(data, period=14, kernel="invalid_kernel")
    
    def test_zlema_all_nan_input(self):
        """Test ZLEMA fails with all NaN input"""
        data = np.array([np.nan, np.nan, np.nan])
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.zlema(data, period=2)
    
    def test_zlema_empty_input(self):
        """Test ZLEMA fails with empty input"""
        data = np.array([])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.zlema(data, period=14)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])