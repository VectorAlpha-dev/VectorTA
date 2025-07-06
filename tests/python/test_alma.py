"""
Python binding tests for ALMA indicator.
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


class TestAlma:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_alma_partial_params(self, test_data):
        """Test ALMA with partial parameters (None values) - mirrors check_alma_partial_params"""
        close = test_data['close']
        
        # Test with all default params (None)
        result = ta_indicators.alma(close, 9, 0.85, 6.0)  # Using defaults
        assert len(result) == len(close)
    
    def test_alma_accuracy(self, test_data):
        """Test ALMA matches expected values from Rust tests - mirrors check_alma_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['alma']
        
        result = ta_indicators.alma(
            close,
            period=expected['default_params']['period'],
            offset=expected['default_params']['offset'],
            sigma=expected['default_params']['sigma']
        )
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-8,
            msg="ALMA last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('alma', result, 'close', expected['default_params'])
    
    def test_alma_default_candles(self, test_data):
        """Test ALMA with default parameters - mirrors check_alma_default_candles"""
        close = test_data['close']
        
        # Default params: period=9, offset=0.85, sigma=6.0
        result = ta_indicators.alma(close, 9, 0.85, 6.0)
        assert len(result) == len(close)
    
    def test_alma_zero_period(self):
        """Test ALMA fails with zero period - mirrors check_alma_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.alma(input_data, period=0, offset=0.85, sigma=6.0)
    
    def test_alma_period_exceeds_length(self):
        """Test ALMA fails when period exceeds data length - mirrors check_alma_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.alma(data_small, period=10, offset=0.85, sigma=6.0)
    
    def test_alma_very_small_dataset(self):
        """Test ALMA fails with insufficient data - mirrors check_alma_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.alma(single_point, period=9, offset=0.85, sigma=6.0)
    
    def test_alma_empty_input(self):
        """Test ALMA fails with empty input - mirrors check_alma_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.alma(empty, period=9, offset=0.85, sigma=6.0)
    
    def test_alma_invalid_sigma(self):
        """Test ALMA fails with invalid sigma - mirrors check_alma_invalid_sigma"""
        data = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="Invalid sigma"):
            ta_indicators.alma(data, period=2, offset=0.85, sigma=0.0)
        
        with pytest.raises(ValueError, match="Invalid sigma"):
            ta_indicators.alma(data, period=2, offset=0.85, sigma=-1.0)
    
    def test_alma_invalid_offset(self):
        """Test ALMA fails with invalid offset - mirrors check_alma_invalid_offset"""
        data = np.array([1.0, 2.0, 3.0])
        
        # Test with NaN offset
        with pytest.raises(ValueError, match="Invalid offset"):
            ta_indicators.alma(data, period=2, offset=float('nan'), sigma=6.0)
        
        # Test with offset outside [0, 1]
        with pytest.raises(ValueError, match="Invalid offset"):
            ta_indicators.alma(data, period=2, offset=1.5, sigma=6.0)
        
        with pytest.raises(ValueError, match="Invalid offset"):
            ta_indicators.alma(data, period=2, offset=-0.1, sigma=6.0)
    
    def test_alma_reinput(self, test_data):
        """Test ALMA applied twice (re-input) - mirrors check_alma_reinput"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['alma']
        
        # First pass
        first_result = ta_indicators.alma(close, period=9, offset=0.85, sigma=6.0)
        assert len(first_result) == len(close)
        
        # Second pass - apply ALMA to ALMA output
        second_result = ta_indicators.alma(first_result, period=9, offset=0.85, sigma=6.0)
        assert len(second_result) == len(first_result)
        
        # Check last 5 values match expected
        assert_close(
            second_result[-5:],
            expected['reinput_last_5'],
            rtol=1e-8,
            msg="ALMA re-input last 5 values mismatch"
        )
    
    def test_alma_nan_handling(self, test_data):
        """Test ALMA handles NaN values correctly - mirrors check_alma_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.alma(close, period=9, offset=0.85, sigma=6.0)
        assert len(result) == len(close)
        
        # After warmup period (240), no NaN values should exist
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"
        
        # First period-1 values should be NaN
        assert np.all(np.isnan(result[:8])), "Expected NaN in warmup period"
    
    def test_alma_streaming(self, test_data):
        """Test ALMA streaming matches batch calculation - mirrors check_alma_streaming"""
        close = test_data['close']
        period = 9
        offset = 0.85
        sigma = 6.0
        
        # Batch calculation
        batch_result = ta_indicators.alma(close, period=period, offset=offset, sigma=sigma)
        
        # Streaming calculation
        stream = ta_indicators.AlmaStream(period=period, offset=offset, sigma=sigma)
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
                        msg=f"ALMA streaming mismatch at index {i}")
    
    def test_alma_batch(self, test_data):
        """Test ALMA batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        result = ta_indicators.alma_batch(
            close,
            period_range=(9, 9, 0),  # Default period only
            offset_range=(0.85, 0.85, 0.0),  # Default offset only
            sigma_range=(6.0, 6.0, 0.0)  # Default sigma only
        )
        
        assert 'values' in result
        assert 'periods' in result
        assert 'offsets' in result
        assert 'sigmas' in result
        
        # Should have 1 combination (default params)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        # Extract the single row
        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['alma']['last_5_values']
        
        # Check last 5 values match
        assert_close(
            default_row[-5:],
            expected,
            rtol=1e-8,
            msg="ALMA batch default row mismatch"
        )
    
    def test_alma_all_nan_input(self):
        """Test ALMA with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.alma(all_nan, period=9, offset=0.85, sigma=6.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])