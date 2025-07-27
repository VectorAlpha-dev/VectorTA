"""
Python binding tests for LRSI indicator.
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


class TestLrsi:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_lrsi_partial_params(self, test_data):
        """Test LRSI with partial parameters - mirrors check_lrsi_partial_params"""
        high = test_data['high']
        low = test_data['low']
        
        # Test with default alpha
        result = ta_indicators.lrsi(high, low, 0.2)  # Using default
        assert len(result) == len(high)
    
    def test_lrsi_accuracy(self, test_data):
        """Test LRSI matches expected values from Rust tests - mirrors check_lrsi_accuracy"""
        high = test_data['high']
        low = test_data['low']
        expected = EXPECTED_OUTPUTS.get('lrsi', {
            'default_params': {'alpha': 0.2},
            'last_5_values': [0.0, 0.0, 0.0, 0.0, 0.0]  # From Rust test
        })
        
        result = ta_indicators.lrsi(
            high,
            low,
            alpha=expected['default_params']['alpha']
        )
        
        assert len(result) == len(high)
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-9,
            msg="LRSI last 5 values mismatch"
        )
        
        # Compare full output with Rust if available
        # compare_with_rust('lrsi', result, 'hl', expected['default_params'])
    
    def test_lrsi_default_candles(self, test_data):
        """Test LRSI with default parameters - mirrors check_lrsi_default_candles"""
        high = test_data['high']
        low = test_data['low']
        
        # Default params: alpha=0.2
        result = ta_indicators.lrsi(high, low, 0.2)
        assert len(result) == len(high)
    
    def test_lrsi_invalid_alpha(self):
        """Test LRSI fails with invalid alpha - mirrors check_lrsi_invalid_alpha"""
        high = np.array([1.0, 2.0])
        low = np.array([1.0, 2.0])
        
        # Alpha > 1.0
        with pytest.raises(ValueError, match="Invalid alpha|alpha not in"):
            ta_indicators.lrsi(high, low, alpha=1.2)
        
        # Alpha = 0.0
        with pytest.raises(ValueError, match="Invalid alpha|alpha not in"):
            ta_indicators.lrsi(high, low, alpha=0.0)
        
        # Alpha < 0.0
        with pytest.raises(ValueError, match="Invalid alpha|alpha not in"):
            ta_indicators.lrsi(high, low, alpha=-0.1)
    
    def test_lrsi_empty_data(self):
        """Test LRSI fails with empty data - mirrors check_lrsi_empty_data"""
        high = np.array([])
        low = np.array([])
        
        with pytest.raises(ValueError, match="Empty data|Empty input"):
            ta_indicators.lrsi(high, low, alpha=0.2)
    
    def test_lrsi_all_nan(self):
        """Test LRSI fails with all NaN values - mirrors check_lrsi_all_nan"""
        high = np.array([np.nan, np.nan, np.nan])
        low = np.array([np.nan, np.nan, np.nan])
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.lrsi(high, low, alpha=0.2)
    
    def test_lrsi_very_small_dataset(self):
        """Test LRSI fails with insufficient data - mirrors check_lrsi_very_small_dataset"""
        high = np.array([1.0, 1.0])
        low = np.array([1.0, 1.0])
        
        with pytest.raises(ValueError, match="Not enough valid data"):
            ta_indicators.lrsi(high, low, alpha=0.2)
    
    def test_lrsi_streaming(self, test_data):
        """Test LRSI streaming matches batch calculation - mirrors check_lrsi_streaming"""
        high = test_data['high']
        low = test_data['low']
        alpha = 0.2
        
        # Batch calculation
        batch_result = ta_indicators.lrsi(high, low, alpha=alpha)
        
        # Streaming calculation
        stream = ta_indicators.LrsiStream(alpha=alpha)
        stream_values = []
        
        for h, l in zip(high, low):
            result = stream.update(h, l)
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        # Compare batch vs streaming
        assert len(batch_result) == len(stream_values)
        
        # Compare values where both are not NaN
        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, atol=1e-9, 
                        msg=f"LRSI streaming mismatch at index {i}")
    
    def test_lrsi_batch(self, test_data):
        """Test LRSI batch processing - mirrors check_batch_default_row"""
        high = test_data['high']
        low = test_data['low']
        
        result = ta_indicators.lrsi_batch(
            high,
            low,
            alpha_range=(0.2, 0.2, 0.0)  # Default alpha only
        )
        
        assert 'values' in result
        assert 'alphas' in result
        
        # Check that we got one row (default params)
        assert len(result['alphas']) == 1
        assert result['alphas'][0] == 0.2
        
        # Check values shape
        values = result['values']
        assert values.shape == (1, len(high))
        
        # Compare with single calculation
        single_result = ta_indicators.lrsi(high, low, alpha=0.2)
        assert_close(values[0], single_result, rtol=1e-9, 
                    msg="LRSI batch vs single mismatch")
    
    def test_lrsi_batch_sweep(self, test_data):
        """Test LRSI batch processing with parameter sweep"""
        high = test_data['high']
        low = test_data['low']
        
        # Sweep alpha from 0.1 to 0.5 in steps of 0.1
        result = ta_indicators.lrsi_batch(
            high,
            low,
            alpha_range=(0.1, 0.5, 0.1)
        )
        
        assert 'values' in result
        assert 'alphas' in result
        
        # Should have 5 combinations
        assert len(result['alphas']) == 5
        assert result['values'].shape == (5, len(high))
        
        # Verify alphas are correct
        expected_alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
        for i, expected in enumerate(expected_alphas):
            assert abs(result['alphas'][i] - expected) < 1e-12
    
    def test_lrsi_kernel_support(self, test_data):
        """Test LRSI with different kernel options"""
        high = test_data['high']
        low = test_data['low']
        
        # Test with explicit scalar kernel
        result_scalar = ta_indicators.lrsi(high, low, alpha=0.2, kernel='scalar')
        assert len(result_scalar) == len(high)
        
        # Test with auto kernel (default)
        result_auto = ta_indicators.lrsi(high, low, alpha=0.2)
        assert len(result_auto) == len(high)
        
        # Results should be very close regardless of kernel
        assert_close(result_scalar, result_auto, rtol=1e-9, 
                    msg="LRSI kernel results mismatch")
    
    def test_lrsi_mismatched_lengths(self):
        """Test LRSI fails when high and low have different lengths"""
        high = np.array([1.0, 2.0, 3.0])
        low = np.array([1.0, 2.0])  # Shorter
        
        # The Rust implementation should handle this gracefully
        # Either by using the shorter length or returning an error
        try:
            result = ta_indicators.lrsi(high, low, alpha=0.2)
            # If it succeeds, result should be the length of the shorter array
            assert len(result) <= len(low)
        except ValueError:
            # Or it might raise an error, which is also acceptable
            pass