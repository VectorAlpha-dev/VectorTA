"""
Python binding tests for Band-Pass indicator.
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


class TestBandPass:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_bandpass_partial_params(self, test_data):
        """Test Band-Pass with partial parameters - mirrors check_bandpass_partial_params"""
        close = test_data['close']
        
        # Test with default params
        result = ta_indicators.bandpass(close)  # Using defaults: period=20, bandwidth=0.3
        assert isinstance(result, dict)
        assert 'bp' in result
        assert 'bp_normalized' in result
        assert 'signal' in result
        assert 'trigger' in result
        assert len(result['bp']) == len(close)
        assert len(result['bp_normalized']) == len(close)
        assert len(result['signal']) == len(close)
        assert len(result['trigger']) == len(close)
    
    def test_bandpass_accuracy(self, test_data):
        """Test Band-Pass matches expected values from Rust tests - mirrors check_bandpass_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['bandpass']
        
        result = ta_indicators.bandpass(
            close,
            period=expected['default_params']['period'],
            bandwidth=expected['default_params']['bandwidth']
        )
        
        assert isinstance(result, dict)
        assert len(result['bp']) == len(close)
        
        # Check last 5 values match expected with 1e-1 tolerance (as in Rust tests)
        assert_close(
            result['bp'][-5:],
            expected['last_5_values']['bp'],
            rtol=0,
            atol=1e-1,
            msg="Band-Pass bp last 5 values mismatch"
        )
        
        assert_close(
            result['bp_normalized'][-5:],
            expected['last_5_values']['bp_normalized'],
            rtol=0,
            atol=1e-1,
            msg="Band-Pass bp_normalized last 5 values mismatch"
        )
        
        assert_close(
            result['signal'][-5:],
            expected['last_5_values']['signal'],
            rtol=0,
            atol=1e-1,
            msg="Band-Pass signal last 5 values mismatch"
        )
        
        assert_close(
            result['trigger'][-5:],
            expected['last_5_values']['trigger'],
            rtol=0,
            atol=1e-1,
            msg="Band-Pass trigger last 5 values mismatch"
        )
        
        # Also run full comparison with Rust for bp values only
        compare_with_rust('bandpass', result['bp'])
    
    def test_bandpass_default_params(self, test_data):
        """Test Band-Pass with default parameters"""
        close = test_data['close']
        
        result = ta_indicators.bandpass(close)  # Should use defaults: period=20, bandwidth=0.3
        assert isinstance(result, dict)
        assert len(result['bp']) == len(close)
        assert len(result['bp_normalized']) == len(close)
        assert len(result['signal']) == len(close)
        assert len(result['trigger']) == len(close)
    
    def test_bandpass_zero_period(self):
        """Test Band-Pass fails with zero period - mirrors check_bandpass_zero_period"""
        data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError):
            ta_indicators.bandpass(data, period=0, bandwidth=0.3)
    
    def test_bandpass_period_exceeds_length(self):
        """Test Band-Pass fails when period exceeds data length - mirrors check_bandpass_period_exceeds_length"""
        data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError):
            ta_indicators.bandpass(data, period=10, bandwidth=0.3)
    
    def test_bandpass_very_small_dataset(self):
        """Test Band-Pass fails with insufficient data - mirrors check_bandpass_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError):
            ta_indicators.bandpass(single_point, period=20, bandwidth=0.3)
    
    def test_bandpass_reinput(self, test_data):
        """Test Band-Pass applied twice (re-input) - mirrors check_bandpass_reinput"""
        close = test_data['close']
        
        # First pass with specific parameters
        first_result = ta_indicators.bandpass(close, period=20, bandwidth=0.3)
        assert len(first_result['bp']) == len(close)
        
        # Second pass with different parameters using first result bp as input
        second_result = ta_indicators.bandpass(first_result['bp'], period=30, bandwidth=0.5)
        assert len(second_result['bp']) == len(first_result['bp'])
    
    def test_bandpass_nan_handling(self, test_data):
        """Test Band-Pass handles NaN values correctly - mirrors check_bandpass_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.bandpass(close, period=20, bandwidth=0.3)
        assert len(result['bp']) == len(close)
        
        # After index 30, no NaN values should exist in any output array
        if len(result['bp']) > 30:
            assert not np.any(np.isnan(result['bp'][30:])), "Found unexpected NaN in bp after index 30"
            assert not np.any(np.isnan(result['bp_normalized'][30:])), "Found unexpected NaN in bp_normalized after index 30"
            assert not np.any(np.isnan(result['signal'][30:])), "Found unexpected NaN in signal after index 30"
            assert not np.any(np.isnan(result['trigger'][30:])), "Found unexpected NaN in trigger after index 30"
    
    def test_bandpass_kernel_selection(self, test_data):
        """Test Band-Pass with different kernel selections"""
        close = test_data['close']
        
        # Test with explicit kernel selection
        result_auto = ta_indicators.bandpass(close, kernel=None)  # Auto-detect
        result_scalar = ta_indicators.bandpass(close, kernel='scalar')
        
        assert len(result_auto['bp']) == len(close)
        assert len(result_scalar['bp']) == len(close)
        
        # Results should be very close regardless of kernel
        assert_close(result_auto['bp'], result_scalar['bp'], rtol=1e-10)
        assert_close(result_auto['bp_normalized'], result_scalar['bp_normalized'], rtol=1e-10)
        assert_close(result_auto['signal'], result_scalar['signal'], rtol=1e-10)
        assert_close(result_auto['trigger'], result_scalar['trigger'], rtol=1e-10)
    
    def test_bandpass_streaming(self, test_data):
        """Test Band-Pass streaming functionality"""
        close = test_data['close']
        
        # Create stream
        stream = ta_indicators.BandPassStream(period=20, bandwidth=0.3)
        
        # Process values one by one (only bp values are returned in streaming)
        stream_results = []
        for price in close:
            result = stream.update(price)
            stream_results.append(result)
        
        # Compare with batch calculation (bp values only)
        batch_results = ta_indicators.bandpass(close, period=20, bandwidth=0.3)
        
        # Results should match closely
        assert_close(stream_results, batch_results['bp'], rtol=1e-10)
    
    def test_bandpass_batch_single_parameter_set(self, test_data):
        """Test batch processing with single parameter combination"""
        close = test_data['close']
        
        # Single parameter set: period=20, bandwidth=0.3
        batch_result = ta_indicators.bandpass_batch(
            close,
            period_range=(20, 20, 0),
            bandwidth_range=(0.3, 0.3, 0.0)
        )
        
        # Should return a dict with values and parameters
        assert 'bp' in batch_result
        assert 'bp_normalized' in batch_result
        assert 'signal' in batch_result
        assert 'trigger' in batch_result
        assert 'periods' in batch_result
        assert 'bandwidths' in batch_result
        
        # Should have shape (1, len(close))
        assert batch_result['bp'].shape == (1, len(close))
        assert batch_result['bp_normalized'].shape == (1, len(close))
        assert batch_result['signal'].shape == (1, len(close))
        assert batch_result['trigger'].shape == (1, len(close))
        assert len(batch_result['periods']) == 1
        assert len(batch_result['bandwidths']) == 1
        
        # Should match single calculation
        single_result = ta_indicators.bandpass(close, period=20, bandwidth=0.3)
        assert_close(batch_result['bp'][0], single_result['bp'], rtol=1e-10)
        assert_close(batch_result['bp_normalized'][0], single_result['bp_normalized'], rtol=1e-10)
        assert_close(batch_result['signal'][0], single_result['signal'], rtol=1e-10)
        assert_close(batch_result['trigger'][0], single_result['trigger'], rtol=1e-10)
    
    def test_bandpass_batch_multiple_parameters(self, test_data):
        """Test batch processing with multiple parameter combinations"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        # Multiple parameter combinations
        batch_result = ta_indicators.bandpass_batch(
            close,
            period_range=(10, 30, 10),  # 10, 20, 30
            bandwidth_range=(0.2, 0.4, 0.1)  # 0.2, 0.3, 0.4
        )
        
        # Should have 3 * 3 = 9 combinations
        expected_combos = [
            (10, 0.2), (10, 0.3), (10, 0.4),
            (20, 0.2), (20, 0.3), (20, 0.4),
            (30, 0.2), (30, 0.3), (30, 0.4)
        ]
        
        assert batch_result['bp'].shape[0] == len(expected_combos)
        assert batch_result['bp'].shape[1] == len(close)
        assert len(batch_result['periods']) == len(expected_combos)
        assert len(batch_result['bandwidths']) == len(expected_combos)
        
        # Verify each row matches individual calculation
        for i, (period, bandwidth) in enumerate(expected_combos):
            single_result = ta_indicators.bandpass(close, period=period, bandwidth=bandwidth)
            assert_close(
                batch_result['bp'][i],
                single_result['bp'],
                rtol=1e-10,
                msg=f"Batch row {i} (period={period}, bandwidth={bandwidth}) mismatch"
            )
    
    def test_bandpass_invalid_bandwidth(self):
        """Test Band-Pass with invalid bandwidth values"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 10)
        
        # Bandwidth must be in [0, 1]
        with pytest.raises(ValueError):
            ta_indicators.bandpass(data, period=10, bandwidth=-0.1)
        
        with pytest.raises(ValueError):
            ta_indicators.bandpass(data, period=10, bandwidth=1.5)
        
        with pytest.raises(ValueError):
            ta_indicators.bandpass(data, period=10, bandwidth=np.nan)
    
    def test_bandpass_edge_cases(self):
        """Test Band-Pass with edge case inputs"""
        # Minimum valid period
        data = np.random.rand(50)
        result = ta_indicators.bandpass(data, period=2, bandwidth=0.3)
        assert len(result['bp']) == len(data)
        
        # Verify all outputs have correct length
        assert len(result['bp_normalized']) == len(data)
        assert len(result['signal']) == len(data)
        assert len(result['trigger']) == len(data)
        
        # All values should be finite
        assert np.all(np.isfinite(result['bp'][30:]))
        assert np.all(np.isfinite(result['bp_normalized'][30:]))
        assert np.all(np.isfinite(result['signal'][30:]))
        assert np.all(np.isfinite(result['trigger'][30:]))
    
    def test_bandpass_batch_default_row(self, test_data):
        """Test batch processing includes default parameter combination"""
        close = test_data['close']
        
        # Create a batch that includes the default parameters
        batch_result = ta_indicators.bandpass_batch(
            close,
            period_range=(15, 25, 5),  # 15, 20, 25
            bandwidth_range=(0.2, 0.4, 0.1)  # 0.2, 0.3, 0.4
        )
        
        # Find the row with default parameters (period=20, bandwidth=0.3)
        default_row = None
        for i, (period, bandwidth) in enumerate(zip(batch_result['periods'], batch_result['bandwidths'])):
            if period == 20 and abs(bandwidth - 0.3) < 1e-10:
                default_row = i
                break
        
        assert default_row is not None, "Default parameters not found in batch"
        
        # Verify against expected values
        expected = EXPECTED_OUTPUTS['bandpass']
        assert_close(
            batch_result['bp'][default_row][-5:],
            expected['last_5_values']['bp'],
            rtol=0,
            atol=1e-1,
            msg="Default row bp values mismatch"
        )