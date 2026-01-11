"""
Python binding tests for VPT (Volume Price Trend).
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


class TestVPT:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_vpt_basic_candles(self, test_data):
        """Test VPT with candle data - mirrors check_vpt_basic_candles"""
        close = test_data['close']
        volume = test_data['volume']
        
        result = ta_indicators.vpt(close, volume)
        assert len(result) == len(close)
    
    def test_vpt_basic_slices(self):
        """Test VPT with basic slice data - mirrors check_vpt_basic_slices"""
        price = np.array([1.0, 1.1, 1.05, 1.2, 1.3])
        volume = np.array([1000.0, 1100.0, 1200.0, 1300.0, 1400.0])
        
        result = ta_indicators.vpt(price, volume)
        assert len(result) == len(price)
        
        
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        
        assert not np.isnan(result[2])
    
    def test_vpt_accuracy_from_csv(self, test_data):
        """Test VPT matches expected values from Rust tests - mirrors check_vpt_accuracy_from_csv"""
        close = test_data['close']
        volume = test_data['volume']
        
        result = ta_indicators.vpt(close, volume)
        
        expected_last_five = [
            -18292.323972247592,
            -18292.510374716476,
            -18292.803266539282,
            -18292.62919783763,
            -18296.152568643138,
        ]
        
        assert len(result) >= 5
        
        
        assert_close(
            result[-5:],
            expected_last_five,
            rtol=0.0,
            atol=1e-9,
            msg="VPT last 5 values mismatch"
        )
    
    def test_vpt_not_enough_data(self):
        """Test VPT fails with insufficient data - mirrors check_vpt_not_enough_data"""
        price = np.array([100.0])
        volume = np.array([500.0])
        
        with pytest.raises(ValueError, match="Not enough valid data"):
            ta_indicators.vpt(price, volume)
    
    def test_vpt_empty_data(self):
        """Test VPT fails with empty data - mirrors check_vpt_empty_data"""
        price = np.array([])
        volume = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.vpt(price, volume)
    
    def test_vpt_all_nan(self):
        """Test VPT fails with all NaN values - mirrors check_vpt_all_nan"""
        price = np.array([np.nan, np.nan, np.nan])
        volume = np.array([np.nan, np.nan, np.nan])
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.vpt(price, volume)
    
    def test_vpt_mismatched_lengths(self):
        """Test VPT fails with mismatched input lengths"""
        price = np.array([1.0, 2.0, 3.0])
        volume = np.array([100.0, 200.0])  
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.vpt(price, volume)
    
    def test_vpt_stream(self):
        """Test VPT streaming calculation"""
        stream = ta_indicators.VptStream()
        
        
        result = stream.update(100.0, 1000.0)
        assert result is None
        
        
        result = stream.update(105.0, 1100.0)
        assert np.isnan(result)
        
        
        result = stream.update(103.0, 1200.0)
        assert not np.isnan(result)
    
    def test_vpt_batch(self, test_data):
        """Test VPT batch processing"""
        close = test_data['close'][:100]  
        volume = test_data['volume'][:100]
        
        result = ta_indicators.vpt_batch(close, volume)
        
        assert 'values' in result
        
        assert 'params' in result
        
        
        values_2d = result['values']
        assert values_2d.shape[0] == 1  
        assert values_2d.shape[1] == len(close)  
        
        
        single_result = ta_indicators.vpt(close, volume)
        assert_close(
            values_2d[0, :], 
            single_result,
            rtol=1e-10,
            msg="Batch vs single VPT mismatch"
        )
    
    def test_vpt_kernel_support(self, test_data):
        """Test VPT with different kernel options"""
        close = test_data['close'][:100]
        volume = test_data['volume'][:100]
        
        
        result_scalar = ta_indicators.vpt(close, volume, kernel='scalar')
        assert len(result_scalar) == len(close)
        
        
        result_auto = ta_indicators.vpt(close, volume)
        assert len(result_auto) == len(close)
        
        
        assert_close(
            result_scalar,
            result_auto,
            rtol=1e-10,
            msg="Kernel results mismatch"
        )
    
    def test_vpt_nan_handling(self, test_data):
        """Test VPT handles NaN values correctly"""
        close = test_data['close']
        volume = test_data['volume']
        
        
        close_with_nan = close.copy()
        volume_with_nan = volume.copy()
        close_with_nan[10] = np.nan
        volume_with_nan[20] = np.nan
        
        result = ta_indicators.vpt(close_with_nan, volume_with_nan)
        assert len(result) == len(close)
        
        
        assert np.isnan(result[0])
        
        
        assert np.isnan(result[11])  
        assert np.isnan(result[21])  
