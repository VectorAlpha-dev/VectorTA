"""
Python binding tests for ACOSC indicator.
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


class TestAcosc:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_acosc_partial_params(self, test_data):
        """Test ACOSC with default parameters - mirrors check_acosc_partial_params"""
        high = test_data['high']
        low = test_data['low']
        
        # ACOSC returns a tuple of (osc, change)
        osc, change = ta_indicators.acosc(high, low)
        assert len(osc) == len(high)
        assert len(change) == len(high)
    
    def test_acosc_accuracy(self, test_data):
        """Test ACOSC matches expected values from Rust tests - mirrors check_acosc_accuracy"""
        high = test_data['high']
        low = test_data['low']
        expected = EXPECTED_OUTPUTS['acosc']
        
        osc, change = ta_indicators.acosc(high, low)
        
        assert len(osc) == len(high)
        assert len(change) == len(high)
        
        # Check last 5 osc values match expected
        assert_close(
            osc[-5:], 
            expected['last_5_osc'],
            rtol=1e-1,  # ACOSC uses 1e-1 tolerance in Rust tests
            msg="ACOSC osc last 5 values mismatch"
        )
        
        # Check last 5 change values match expected
        assert_close(
            change[-5:],
            expected['last_5_change'],
            rtol=1e-1,  # ACOSC uses 1e-1 tolerance in Rust tests
            msg="ACOSC change last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('acosc', {'osc': osc, 'change': change}, 'high_low', expected['default_params'])
    
    def test_acosc_too_short(self):
        """Test ACOSC fails with insufficient data - mirrors check_acosc_too_short"""
        high = np.array([100.0, 101.0])
        low = np.array([99.0, 98.0])
        
        with pytest.raises(ValueError, match="Not enough data"):
            ta_indicators.acosc(high, low)
    
    def test_acosc_length_mismatch(self):
        """Test ACOSC fails when high and low lengths don't match"""
        high = np.array([100.0, 101.0, 102.0])
        low = np.array([99.0, 98.0])  # Shorter array
        
        with pytest.raises(ValueError, match="Mismatch"):
            ta_indicators.acosc(high, low)
    
    def test_acosc_nan_handling(self, test_data):
        """Test ACOSC handles NaN values correctly - mirrors check_acosc_nan_handling"""
        high = test_data['high']
        low = test_data['low']
        
        osc, change = ta_indicators.acosc(high, low)
        
        # First 38 values should be NaN (warmup period = 34 + 5 - 1)
        assert np.all(np.isnan(osc[:38])), "Expected NaN in warmup period for osc"
        assert np.all(np.isnan(change[:38])), "Expected NaN in warmup period for change"
        
        # After warmup period, no NaN values should exist
        if len(osc) > 240:
            assert not np.any(np.isnan(osc[240:])), "Found unexpected NaN in osc after warmup period"
            assert not np.any(np.isnan(change[240:])), "Found unexpected NaN in change after warmup period"
    
    def test_acosc_streaming(self, test_data):
        """Test ACOSC streaming matches batch calculation - mirrors check_acosc_streaming"""
        high = test_data['high']
        low = test_data['low']
        
        # Batch calculation
        batch_osc, batch_change = ta_indicators.acosc(high, low)
        
        # Streaming calculation
        stream = ta_indicators.AcoscStream()
        stream_osc = []
        stream_change = []
        
        for h, l in zip(high, low):
            result = stream.update(h, l)
            if result is not None:
                stream_osc.append(result[0])
                stream_change.append(result[1])
            else:
                stream_osc.append(np.nan)
                stream_change.append(np.nan)
        
        stream_osc = np.array(stream_osc)
        stream_change = np.array(stream_change)
        
        # Compare batch vs streaming
        assert len(batch_osc) == len(stream_osc)
        assert len(batch_change) == len(stream_change)
        
        # Compare osc values where both are not NaN
        for i, (b, s) in enumerate(zip(batch_osc, stream_osc)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, atol=1e-9, 
                        msg=f"ACOSC osc streaming mismatch at index {i}")
        
        # Compare change values where both are not NaN
        for i, (b, s) in enumerate(zip(batch_change, stream_change)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, atol=1e-9,
                        msg=f"ACOSC change streaming mismatch at index {i}")
    
    def test_acosc_batch(self, test_data):
        """Test ACOSC batch processing"""
        high = test_data['high']
        low = test_data['low']
        
        result = ta_indicators.acosc_batch(high, low)
        
        assert 'osc' in result
        assert 'change' in result
        
        # Should have 1 row (no parameters)
        assert result['osc'].shape[0] == 1
        assert result['osc'].shape[1] == len(high)
        assert result['change'].shape[0] == 1
        assert result['change'].shape[1] == len(high)
        
        # Extract the single rows
        osc_row = result['osc'][0]
        change_row = result['change'][0]
        expected = EXPECTED_OUTPUTS['acosc']
        
        # Check last 5 values match
        assert_close(
            osc_row[-5:],
            expected['last_5_osc'],
            rtol=1e-1,  # ACOSC uses 1e-1 tolerance
            msg="ACOSC batch osc mismatch"
        )
        
        assert_close(
            change_row[-5:],
            expected['last_5_change'],
            rtol=1e-1,  # ACOSC uses 1e-1 tolerance
            msg="ACOSC batch change mismatch"
        )
    
    def test_acosc_kernel_parameter(self, test_data):
        """Test ACOSC with different kernel parameters"""
        high = test_data['high']
        low = test_data['low']
        
        # Test different kernels
        kernels = ['auto', 'scalar', 'avx2', 'avx512']
        for kernel in kernels:
            try:
                osc, change = ta_indicators.acosc(high, low, kernel=kernel)
                assert len(osc) == len(high)
                assert len(change) == len(high)
            except ValueError as e:
                # AVX kernels might not be available on all systems
                if "Unknown kernel" not in str(e) and "not available on this CPU" not in str(e) and "not compiled in this build" not in str(e):
                    raise


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
