"""
Python binding tests for DTI indicator.
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


class TestDti:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_dti_partial_params(self, test_data):
        """Test DTI with partial parameters (None values) - mirrors check_dti_partial_params"""
        high = test_data['high']
        low = test_data['low']
        
        # Test with all default params
        result = ta_indicators.dti(high, low, 14, 10, 5)  # Using defaults
        assert len(result) == len(high)
    
    def test_dti_accuracy(self, test_data):
        """Test DTI matches expected values from Rust tests - mirrors check_dti_accuracy"""
        high = test_data['high']
        low = test_data['low']
        expected = EXPECTED_OUTPUTS.get('dti', {})
        
        # If expected values not yet added to test_utils.py, skip
        if not expected:
            pytest.skip("DTI expected values not yet added to EXPECTED_OUTPUTS")
        
        result = ta_indicators.dti(
            high, low,
            r=expected['default_params']['r'],
            s=expected['default_params']['s'],
            u=expected['default_params']['u']
        )
        
        assert len(result) == len(high)
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-6,
            msg="DTI last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('dti', result, 'high,low', expected['default_params'])
    
    def test_dti_default_candles(self, test_data):
        """Test DTI with default parameters - mirrors check_dti_default_candles"""
        high = test_data['high']
        low = test_data['low']
        
        # Default params: r=14, s=10, u=5
        result = ta_indicators.dti(high, low, 14, 10, 5)
        assert len(result) == len(high)
    
    def test_dti_zero_period(self):
        """Test DTI fails with zero period - mirrors check_dti_zero_period"""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0, 11.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.dti(high, low, r=0, s=10, u=5)
    
    def test_dti_period_exceeds_length(self):
        """Test DTI fails when period exceeds data length - mirrors check_dti_period_exceeds_length"""
        high = np.array([10.0, 11.0])
        low = np.array([9.0, 10.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.dti(high, low, r=14, s=10, u=5)
    
    def test_dti_all_nan(self):
        """Test DTI fails with all NaN values - mirrors check_dti_all_nan"""
        high = np.array([np.nan, np.nan, np.nan])
        low = np.array([np.nan, np.nan, np.nan])
        
        with pytest.raises(ValueError, match="All.*values are NaN"):
            ta_indicators.dti(high, low, r=14, s=10, u=5)
    
    def test_dti_empty_data(self):
        """Test DTI fails with empty data - mirrors check_dti_empty_data"""
        high = np.array([])
        low = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.dti(high, low, r=14, s=10, u=5)
    
    def test_dti_mismatched_lengths(self):
        """Test DTI fails when high/low have different lengths"""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.dti(high, low, r=14, s=10, u=5)
    
    def test_dti_streaming(self, test_data):
        """Test DTI streaming matches batch calculation - mirrors check_dti_streaming"""
        high = test_data['high']
        low = test_data['low']
        r = 14
        s = 10
        u = 5
        
        # Batch calculation
        batch_result = ta_indicators.dti(high, low, r=r, s=s, u=u)
        
        # Streaming calculation
        stream = ta_indicators.DtiStream(r=r, s=s, u=u)
        stream_values = []
        
        for h, l in zip(high, low):
            result = stream.update(h, l)
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        # Compare batch vs streaming
        assert len(batch_result) == len(stream_values)
        
        # Compare values where both are not NaN
        for i, (b, s_val) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s_val):
                continue
            assert_close(b, s_val, rtol=1e-9, atol=1e-9, 
                        msg=f"DTI streaming mismatch at index {i}")
    
    def test_dti_batch(self, test_data):
        """Test DTI batch processing - mirrors check_batch_default_row"""
        high = test_data['high']
        low = test_data['low']
        
        result = ta_indicators.dti_batch(
            high, low,
            r_range=(14, 14, 1),  # Default r only
            s_range=(10, 10, 1),  # Default s only
            u_range=(5, 5, 1)     # Default u only
        )
        
        assert 'values' in result
        assert 'r_values' in result
        assert 's_values' in result
        assert 'u_values' in result
        
        # Should have 1 combination (default params)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(high)
        
        # Extract the single row
        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS.get('dti', {}).get('last_5_values', [])
        
        if expected:
            # Check last 5 values match
            assert_close(
                default_row[-5:],
                expected,
                rtol=1e-6,
                msg="DTI batch default row mismatch"
            )
    
    def test_dti_nan_handling(self, test_data):
        """Test DTI handles NaN values correctly"""
        high = test_data['high']
        low = test_data['low']
        
        result = ta_indicators.dti(high, low, r=14, s=10, u=5)
        assert len(result) == len(high)
        
        # Check warmup period
        warmup = 14  # max(r, s, u)
        
        # After warmup period, no NaN values should exist
        if len(result) > warmup + 100:
            assert not np.any(np.isnan(result[warmup + 100:])), "Found unexpected NaN after warmup period"
    
    def test_dti_kernel_parameter(self, test_data):
        """Test DTI with different kernel parameters"""
        high = test_data['high'][:1000]  # Use smaller dataset for speed
        low = test_data['low'][:1000]
        
        # Test with default kernel (auto)
        result_auto = ta_indicators.dti(high, low, r=14, s=10, u=5)
        
        # Test with scalar kernel
        result_scalar = ta_indicators.dti(high, low, r=14, s=10, u=5, kernel='scalar')
        
        assert len(result_auto) == len(high)
        assert len(result_scalar) == len(high)
        
        # Results should be very close
        for i, (a, s) in enumerate(zip(result_auto, result_scalar)):
            if np.isnan(a) and np.isnan(s):
                continue
            assert_close(a, s, rtol=1e-9, atol=1e-9,
                        msg=f"Kernel results mismatch at index {i}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
