"""
Python binding tests for VWAP indicator.
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


class TestVwap:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_vwap_partial_params(self, test_data):
        """Test VWAP with default parameters - mirrors check_vwap_partial_params"""
        timestamps = test_data['timestamp']
        volumes = test_data['volume']
        prices = (test_data['high'] + test_data['low'] + test_data['close']) / 3.0  # hlc3
        
        # Test with default anchor (None)
        result = ta_indicators.vwap(timestamps, volumes, prices)
        assert len(result) == len(prices)
    
    def test_vwap_accuracy(self, test_data):
        """Test VWAP matches expected values from Rust tests - mirrors check_vwap_accuracy"""
        timestamps = test_data['timestamp']
        volumes = test_data['volume']
        prices = (test_data['high'] + test_data['low'] + test_data['close']) / 3.0  # hlc3
        expected = EXPECTED_OUTPUTS['vwap']
        
        result = ta_indicators.vwap(
            timestamps,
            volumes,
            prices,
            anchor="1D"  # Using uppercase D as in Rust test
        )
        
        assert len(result) == len(prices)
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected['anchor_1D'],
            rtol=1e-5,  # Using 1e-5 as in Rust test
            msg="VWAP last 5 values mismatch"
        )
    
    def test_vwap_anchor_parsing_error(self, test_data):
        """Test VWAP fails with invalid anchor - mirrors check_vwap_anchor_parsing_error"""
        timestamps = test_data['timestamp']
        volumes = test_data['volume']
        prices = (test_data['high'] + test_data['low'] + test_data['close']) / 3.0
        
        with pytest.raises(ValueError, match="Error parsing anchor"):
            ta_indicators.vwap(timestamps, volumes, prices, anchor="xyz")
    
    def test_vwap_mismatch_lengths(self):
        """Test VWAP fails when array lengths don't match"""
        timestamps = np.array([1000, 2000, 3000], dtype=np.int64)
        volumes = np.array([100.0, 200.0])  # Mismatched length
        prices = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Mismatch in length"):
            ta_indicators.vwap(timestamps, volumes, prices)
    
    def test_vwap_empty_data(self):
        """Test VWAP fails with empty input"""
        empty_ts = np.array([], dtype=np.int64)
        empty_vol = np.array([])
        empty_price = np.array([])
        
        with pytest.raises(ValueError, match="No data"):
            ta_indicators.vwap(empty_ts, empty_vol, empty_price)
    
    def test_vwap_streaming(self, test_data):
        """Test VWAP streaming functionality"""
        timestamps = test_data['timestamp']
        volumes = test_data['volume']
        prices = (test_data['high'] + test_data['low'] + test_data['close']) / 3.0
        
        # Batch calculation
        batch_result = ta_indicators.vwap(timestamps, volumes, prices, anchor="1d")
        
        # Streaming calculation
        stream = ta_indicators.VwapStream(anchor="1d")
        stream_values = []
        
        for i in range(len(timestamps)):
            result = stream.update(timestamps[i], prices[i], volumes[i])
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
            msg="VWAP streaming mismatch"
        )
    
    def test_vwap_batch(self, test_data):
        """Test VWAP batch processing - mirrors check_batch_anchor_grid"""
        timestamps = test_data['timestamp']
        volumes = test_data['volume']
        prices = (test_data['high'] + test_data['low'] + test_data['close']) / 3.0
        
        result = ta_indicators.vwap_batch(
            timestamps,
            volumes,
            prices,
            anchor_range=("1d", "3d", 1)
        )
        
        assert 'values' in result
        assert 'anchors' in result
        
        # Should have 3 combinations: 1d, 2d, 3d
        assert result['values'].shape[0] == 3
        assert result['values'].shape[1] == len(prices)
        assert list(result['anchors']) == ["1d", "2d", "3d"]
        
        # Check that 1d row matches single VWAP calculation
        single_vwap = ta_indicators.vwap(timestamps, volumes, prices, anchor="1d")
        assert_close(
            result['values'][0],  # First row is 1d
            single_vwap,
            rtol=1e-9,
            msg="VWAP batch 1d row mismatch"
        )
    
    def test_vwap_default_params(self, test_data):
        """Test VWAP with default parameters - mirrors check_vwap_with_default_params"""
        # Just verify defaults work
        timestamps = test_data['timestamp']
        volumes = test_data['volume']  
        prices = test_data['close']
        
        # Should use default anchor "1d"
        result = ta_indicators.vwap(timestamps, volumes, prices)
        assert len(result) == len(prices)
    
    def test_vwap_nan_handling(self, test_data):
        """Test VWAP handles finite values correctly - mirrors check_vwap_nan_handling"""
        timestamps = test_data['timestamp']
        volumes = test_data['volume']
        prices = (test_data['high'] + test_data['low'] + test_data['close']) / 3.0
        
        result = ta_indicators.vwap(timestamps, volumes, prices)
        assert len(result) == len(prices)
        
        # Check all non-NaN values are finite
        for val in result:
            if not np.isnan(val):
                assert np.isfinite(val), "Found non-finite value in VWAP output"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
