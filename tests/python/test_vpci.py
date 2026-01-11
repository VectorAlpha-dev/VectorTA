"""
Python binding tests for VPCI indicator.
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
from rust_comparison import compare_with_rust


class TestVpci:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_vpci_partial_params(self, test_data):
        """Test VPCI with partial parameters - mirrors check_vpci_partial_params"""
        close = test_data['close']
        volume = test_data['volume']
        
        
        vpci, vpcis = ta_indicators.vpci(close, volume, 3, 25)
        assert len(vpci) == len(close)
        assert len(vpcis) == len(close)
    
    def test_vpci_accuracy(self, test_data):
        """Test VPCI matches expected values from Rust tests - mirrors check_vpci_accuracy"""
        close = test_data['close']
        volume = test_data['volume']
        expected = EXPECTED_OUTPUTS['vpci']
        
        vpci, vpcis = ta_indicators.vpci(
            close,
            volume,
            short_range=expected['default_params']['short_range'],
            long_range=expected['default_params']['long_range']
        )
        
        assert len(vpci) == len(close)
        assert len(vpcis) == len(close)
        
        
        assert_close(
            vpci[-5:], 
            expected['last_5_vpci'],
            rtol=5e-2,  
            msg="VPCI last 5 values mismatch"
        )
        
        assert_close(
            vpcis[-5:], 
            expected['last_5_vpcis'],
            rtol=5e-2,  
            msg="VPCIS last 5 values mismatch"
        )
        
        
        
        compare_with_rust('vpci', {'vpci': vpci, 'vpcis': vpcis}, 'close', expected['default_params'])
    
    def test_vpci_default_params(self, test_data):
        """Test VPCI with default parameters - mirrors check_vpci_default_candles"""
        close = test_data['close']
        volume = test_data['volume']
        
        
        vpci, vpcis = ta_indicators.vpci(close, volume, 5, 25)
        assert len(vpci) == len(close)
        assert len(vpcis) == len(close)
    
    def test_vpci_slice_input(self):
        """Test VPCI with slice input - mirrors check_vpci_slice_input"""
        close_data = np.array([10.0, 12.0, 14.0, 13.0, 15.0])
        volume_data = np.array([100.0, 200.0, 300.0, 250.0, 400.0])
        
        vpci, vpcis = ta_indicators.vpci(close_data, volume_data, 2, 3)
        assert len(vpci) == len(close_data)
        assert len(vpcis) == len(close_data)
    
    def test_vpci_zero_period(self):
        """Test VPCI fails with zero period"""
        close = np.array([10.0, 20.0, 30.0])
        volume = np.array([100.0, 200.0, 300.0])
        
        with pytest.raises(ValueError, match="Invalid range"):
            ta_indicators.vpci(close, volume, short_range=0, long_range=2)
        
        with pytest.raises(ValueError, match="Invalid range"):
            ta_indicators.vpci(close, volume, short_range=2, long_range=0)
    
    def test_vpci_period_exceeds_length(self):
        """Test VPCI fails when period exceeds data length"""
        close = np.array([10.0, 20.0, 30.0])
        volume = np.array([100.0, 200.0, 300.0])
        
        with pytest.raises(ValueError, match="Invalid range"):
            ta_indicators.vpci(close, volume, short_range=10, long_range=2)
        
        with pytest.raises(ValueError, match="Invalid range"):
            ta_indicators.vpci(close, volume, short_range=2, long_range=10)
    
    def test_vpci_mismatched_lengths(self):
        """Test VPCI fails when close and volume have different lengths"""
        close = np.array([10.0, 20.0, 30.0])
        volume = np.array([100.0, 200.0])
        
        with pytest.raises(ValueError, match="Close and volume arrays must have the same length"):
            ta_indicators.vpci(close, volume, short_range=2, long_range=3)
    
    def test_vpci_empty_input(self):
        """Test VPCI fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="All close or volume values are NaN"):
            ta_indicators.vpci(empty, empty, short_range=5, long_range=25)
    
    def test_vpci_all_nan(self):
        """Test VPCI fails with all NaN values"""
        nan_data = np.array([float('nan'), float('nan'), float('nan')])
        
        with pytest.raises(ValueError, match="All close or volume values are NaN"):
            ta_indicators.vpci(nan_data, nan_data, short_range=2, long_range=3)
    
    def test_vpci_nan_handling(self, test_data):
        """Test VPCI handles NaN values correctly"""
        close = test_data['close']
        volume = test_data['volume']
        
        vpci, vpcis = ta_indicators.vpci(close, volume, 5, 25)
        assert len(vpci) == len(close)
        assert len(vpcis) == len(close)
        
        
        assert all(np.isnan(vpci[:24]))
        assert all(np.isnan(vpcis[:24]))
        
        
        if len(vpci) > 30:
            assert not any(np.isnan(vpci[30:]))
            assert not any(np.isnan(vpcis[30:]))
    
    def test_vpci_batch_basic(self, test_data):
        """Test VPCI batch processing"""
        close = test_data['close']
        volume = test_data['volume']
        
        result = ta_indicators.vpci_batch(
            close,
            volume,
            short_range_tuple=(3, 7, 2),    
            long_range_tuple=(20, 30, 5)     
        )
        
        assert 'vpci' in result
        assert 'vpcis' in result
        assert 'short_ranges' in result
        assert 'long_ranges' in result
        
        
        vpci_values = np.array(result['vpci'])
        vpcis_values = np.array(result['vpcis'])
        short_ranges = np.array(result['short_ranges'])
        long_ranges = np.array(result['long_ranges'])
        
        expected_combos = 3 * 3  
        assert vpci_values.shape == (expected_combos, len(close))
        assert vpcis_values.shape == (expected_combos, len(close))
        assert len(short_ranges) == expected_combos
        assert len(long_ranges) == expected_combos
        
        
        assert list(short_ranges[:3]) == [3, 3, 3]  
        assert list(short_ranges[3:6]) == [5, 5, 5]  
        assert list(short_ranges[6:9]) == [7, 7, 7]  
        assert list(long_ranges[::3]) == [20, 20, 20]  
        assert list(long_ranges[1::3]) == [25, 25, 25]  
        assert list(long_ranges[2::3]) == [30, 30, 30]  
    
    def test_vpci_batch_single_param(self, test_data):
        """Test VPCI batch with single parameter combination"""
        close = test_data['close']
        volume = test_data['volume']
        
        result = ta_indicators.vpci_batch(
            close,
            volume,
            short_range_tuple=(5, 5, 1),
            long_range_tuple=(25, 25, 1)
        )
        
        vpci_values = np.array(result['vpci'])
        vpcis_values = np.array(result['vpcis'])
        
        
        assert vpci_values.shape == (1, len(close))
        assert vpcis_values.shape == (1, len(close))
        
        
        single_vpci, single_vpcis = ta_indicators.vpci(close, volume, 5, 25)
        assert_close(
            vpci_values[0],
            single_vpci,
            rtol=1e-10,
            msg="Batch VPCI should match single calculation"
        )
        assert_close(
            vpcis_values[0],
            single_vpcis,
            rtol=1e-10,
            msg="Batch VPCIS should match single calculation"
        )
    
    def test_vpci_with_kernel(self, test_data):
        """Test VPCI with different kernel options"""
        close = test_data['close']
        volume = test_data['volume']
        
        
        vpci_default, vpcis_default = ta_indicators.vpci(close, volume, 5, 25)
        
        
        vpci_scalar, vpcis_scalar = ta_indicators.vpci(close, volume, 5, 25, kernel='scalar')
        
        
        assert_close(
            vpci_default,
            vpci_scalar,
            rtol=1e-10,
            msg="Default and scalar kernels should produce same results"
        )
        assert_close(
            vpcis_default,
            vpcis_scalar,
            rtol=1e-10,
            msg="Default and scalar kernels should produce same results"
        )
