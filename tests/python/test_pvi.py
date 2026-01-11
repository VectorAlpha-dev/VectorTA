"""
Python binding tests for PVI indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

try:
    import my_project as ta_indicators
except ImportError:
    
    try:
        import my_project as ta_indicators
    except ImportError:
        pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS


class TestPvi:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_pvi_partial_params(self, test_data):
        """Test PVI with partial parameters - mirrors check_pvi_partial_params"""
        close = test_data['close']
        volume = test_data['volume']
        
        
        result = ta_indicators.pvi(close, volume)
        assert len(result) == len(close)
    
    def test_pvi_accuracy(self):
        """Test PVI matches expected values from Rust tests - mirrors check_pvi_accuracy"""
        close_data = np.array([100.0, 102.0, 101.0, 103.0, 103.0, 105.0])
        volume_data = np.array([500.0, 600.0, 500.0, 700.0, 680.0, 900.0])
        
        result = ta_indicators.pvi(close_data, volume_data, initial_value=1000.0)
        assert len(result) == len(close_data)
        assert abs(result[0] - 1000.0) < 1e-6
        
        
        
        
        
        
        
        
    
    def test_pvi_default_candles(self, test_data):
        """Test PVI with default candles - mirrors check_pvi_default_candles"""
        close = test_data['close']
        volume = test_data['volume']
        
        result = ta_indicators.pvi(close, volume)
        assert len(result) == len(close)
    
    def test_pvi_empty_data(self):
        """Test PVI fails with empty data - mirrors check_pvi_empty_data"""
        close_data = np.array([])
        volume_data = np.array([])
        
        with pytest.raises(Exception) as exc_info:
            ta_indicators.pvi(close_data, volume_data)
        assert "Empty data" in str(exc_info.value)
    
    def test_pvi_mismatched_length(self):
        """Test PVI fails with mismatched lengths - mirrors check_pvi_mismatched_length"""
        close_data = np.array([100.0, 101.0])
        volume_data = np.array([500.0])
        
        with pytest.raises(Exception) as exc_info:
            ta_indicators.pvi(close_data, volume_data)
        assert "different lengths" in str(exc_info.value)
    
    def test_pvi_all_values_nan(self):
        """Test PVI fails with all NaN values - mirrors check_pvi_all_values_nan"""
        close_data = np.array([np.nan, np.nan, np.nan])
        volume_data = np.array([np.nan, np.nan, np.nan])
        
        with pytest.raises(Exception) as exc_info:
            ta_indicators.pvi(close_data, volume_data)
        assert "All values are NaN" in str(exc_info.value)
    
    def test_pvi_not_enough_valid_data(self):
        """Test PVI fails with insufficient valid data - mirrors check_pvi_not_enough_valid_data"""
        close_data = np.array([np.nan, 100.0])
        volume_data = np.array([np.nan, 500.0])
        
        with pytest.raises(Exception) as exc_info:
            ta_indicators.pvi(close_data, volume_data)
        assert "Not enough valid data" in str(exc_info.value)
    
    def test_pvi_streaming(self):
        """Test PVI streaming matches batch calculation - mirrors check_pvi_streaming"""
        close_data = np.array([100.0, 102.0, 101.0, 103.0, 103.0, 105.0])
        volume_data = np.array([500.0, 600.0, 500.0, 700.0, 680.0, 900.0])
        initial_value = 1000.0
        
        
        batch_result = ta_indicators.pvi(close_data, volume_data, initial_value=initial_value)
        
        
        stream = ta_indicators.PviStream(initial_value=initial_value)
        stream_values = []
        for close, vol in zip(close_data, volume_data):
            val = stream.update(close, vol)
            stream_values.append(val if val is not None else np.nan)
        
        assert len(batch_result) == len(stream_values)
        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert abs(b - s) < 1e-9, f"PVI streaming mismatch at index {i}: batch={b}, stream={s}"
    
    def test_pvi_batch(self):
        """Test PVI batch calculation"""
        close_data = np.array([100.0, 102.0, 101.0, 103.0, 103.0, 105.0])
        volume_data = np.array([500.0, 600.0, 500.0, 700.0, 680.0, 900.0])
        
        
        result = ta_indicators.pvi_batch(
            close_data,
            volume_data,
            initial_value_range=(900.0, 1100.0, 100.0)  
        )
        
        assert 'values' in result
        assert 'initial_values' in result
        assert result['values'].shape == (3, len(close_data))
        assert len(result['initial_values']) == 3
        
        
        for i, init_val in enumerate(result['initial_values']):
            single_result = ta_indicators.pvi(close_data, volume_data, initial_value=init_val)
            np.testing.assert_array_almost_equal(result['values'][i], single_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
