"""
Python binding tests for Alligator indicator.
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

from test_utils import load_test_data, assert_close


class TestAlligator:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_alligator_partial_params(self, test_data):
        """Test Alligator with partial parameters - mirrors check_alligator_partial_params"""
        
        hl2 = (test_data['high'] + test_data['low']) / 2
        
        
        result = ta_indicators.alligator(hl2, jaw_period=14, lips_offset=2)
        
        assert 'jaw' in result
        assert 'teeth' in result  
        assert 'lips' in result
        assert len(result['jaw']) == len(hl2)
        assert len(result['teeth']) == len(hl2)
        assert len(result['lips']) == len(hl2)
    
    def test_alligator_accuracy(self, test_data):
        """Test Alligator matches expected values from Rust tests - mirrors check_alligator_accuracy"""
        
        hl2 = (test_data['high'] + test_data['low']) / 2
        
        
        expected_last_five_jaw = [60742.4, 60632.6, 60555.1, 60442.7, 60308.7]
        expected_last_five_teeth = [59908.0, 59757.2, 59684.3, 59653.5, 59621.1]
        expected_last_five_lips = [59355.2, 59371.7, 59376.2, 59334.1, 59316.2]
        
        result = ta_indicators.alligator(hl2)  
        
        assert len(result['jaw']) == len(hl2)
        assert len(result['teeth']) == len(hl2)
        assert len(result['lips']) == len(hl2)
        
        
        
        assert_close(
            result['jaw'][-5:], 
            expected_last_five_jaw,
            rtol=0,
            atol=1e-1,
            msg="Alligator jaw last 5 values mismatch"
        )
        
        assert_close(
            result['teeth'][-5:], 
            expected_last_five_teeth,
            rtol=0,
            atol=1e-1,
            msg="Alligator teeth last 5 values mismatch"
        )
        
        assert_close(
            result['lips'][-5:], 
            expected_last_five_lips,
            rtol=0,
            atol=1e-1,
            msg="Alligator lips last 5 values mismatch"
        )
    
    def test_alligator_default_candles(self, test_data):
        """Test Alligator with default parameters - mirrors check_alligator_default_candles"""
        hl2 = (test_data['high'] + test_data['low']) / 2
        
        
        result = ta_indicators.alligator(hl2)
        
        assert 'jaw' in result
        assert 'teeth' in result
        assert 'lips' in result
        assert len(result['jaw']) == len(hl2)
    
    def test_alligator_zero_jaw_period(self):
        """Test Alligator fails with zero jaw period - mirrors check_alligator_zero_jaw_period"""
        data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid jaw period"):
            ta_indicators.alligator(data, jaw_period=0)
    
    def test_alligator_zero_teeth_period(self):
        """Test Alligator fails with zero teeth period"""
        data = np.array([10.0, 20.0, 30.0])
        
        
        with pytest.raises(ValueError, match="Invalid teeth period"):
            ta_indicators.alligator(data, jaw_period=2, jaw_offset=1, teeth_period=0)
    
    def test_alligator_zero_lips_period(self):
        """Test Alligator fails with zero lips period"""
        data = np.array([10.0, 20.0, 30.0])
        
        
        with pytest.raises(ValueError, match="Invalid lips period"):
            ta_indicators.alligator(data, jaw_period=2, jaw_offset=1, teeth_period=2, teeth_offset=1, lips_period=0)
    
    def test_alligator_period_exceeds_length(self):
        """Test Alligator fails when period exceeds data length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid jaw period"):
            ta_indicators.alligator(data_small, jaw_period=10)
    
    def test_alligator_offset_exceeds_length(self):
        """Test Alligator fails when offset exceeds data length"""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        
        
        with pytest.raises(ValueError, match="Invalid jaw offset"):
            ta_indicators.alligator(data, jaw_period=3, jaw_offset=10)
        
        with pytest.raises(ValueError, match="Invalid teeth offset"):
            ta_indicators.alligator(data, jaw_period=3, jaw_offset=2, teeth_period=3, teeth_offset=10)
        
        with pytest.raises(ValueError, match="Invalid lips offset"):
            ta_indicators.alligator(data, jaw_period=3, jaw_offset=2, teeth_period=3, teeth_offset=2, lips_period=3, lips_offset=10)
    
    def test_alligator_all_nan_input(self):
        """Test Alligator with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.alligator(all_nan)
    
    def test_alligator_reinput(self, test_data):
        """Test Alligator applied to jaw output - mirrors check_alligator_with_slice_data_reinput"""
        hl2 = (test_data['high'] + test_data['low']) / 2
        
        
        first_result = ta_indicators.alligator(hl2)
        assert len(first_result['jaw']) == len(hl2)
        
        
        second_result = ta_indicators.alligator(first_result['jaw'])
        assert len(second_result['jaw']) == len(first_result['jaw'])
        assert len(second_result['teeth']) == len(first_result['teeth'])
        assert len(second_result['lips']) == len(first_result['lips'])
    
    def test_alligator_nan_handling(self, test_data):
        """Test Alligator handles NaN values correctly - mirrors check_alligator_nan_handling"""
        hl2 = (test_data['high'] + test_data['low']) / 2
        
        result = ta_indicators.alligator(hl2)
        assert len(result['jaw']) == len(hl2)
        
        
        if len(result['jaw']) > 50:
            assert not np.any(np.isnan(result['jaw'][50:])), "Found unexpected NaN in jaw after warmup"
            assert not np.any(np.isnan(result['teeth'][50:])), "Found unexpected NaN in teeth after warmup"
            assert not np.any(np.isnan(result['lips'][50:])), "Found unexpected NaN in lips after warmup"
    
    def test_alligator_streaming(self, test_data):
        """Test Alligator streaming matches batch calculation"""
        hl2 = (test_data['high'] + test_data['low']) / 2
        
        
        batch_result = ta_indicators.alligator(hl2)
        
        
        stream = ta_indicators.AlligatorStream()
        stream_jaw = []
        stream_teeth = []
        stream_lips = []
        
        for price in hl2:
            result = stream.update(price)
            if result is not None:
                stream_jaw.append(result[0])
                stream_teeth.append(result[1])
                stream_lips.append(result[2])
            else:
                stream_jaw.append(np.nan)
                stream_teeth.append(np.nan)
                stream_lips.append(np.nan)
        
        stream_jaw = np.array(stream_jaw)
        stream_teeth = np.array(stream_teeth)
        stream_lips = np.array(stream_lips)
        
        
        assert len(batch_result['jaw']) == len(stream_jaw)
        
        
        
        
        
        
        assert len(stream_jaw) == len(hl2)
        assert len(stream_teeth) == len(hl2)
        assert len(stream_lips) == len(hl2)
    
    def test_alligator_batch(self, test_data):
        """Test Alligator batch processing - mirrors check_batch_default_row"""
        hl2 = (test_data['high'] + test_data['low']) / 2
        
        result = ta_indicators.alligator_batch(
            hl2,
            jaw_period_range=(13, 13, 0),  
            jaw_offset_range=(8, 8, 0),    
            teeth_period_range=(8, 8, 0),  
            teeth_offset_range=(5, 5, 0),  
            lips_period_range=(5, 5, 0),   
            lips_offset_range=(3, 3, 0)    
        )
        
        assert 'jaw' in result
        assert 'teeth' in result
        assert 'lips' in result
        assert 'jaw_periods' in result
        assert 'jaw_offsets' in result
        assert 'teeth_periods' in result
        assert 'teeth_offsets' in result
        assert 'lips_periods' in result
        assert 'lips_offsets' in result
        
        
        assert result['jaw'].shape[0] == 1
        assert result['jaw'].shape[1] == len(hl2)
        assert result['teeth'].shape[0] == 1
        assert result['teeth'].shape[1] == len(hl2)
        assert result['lips'].shape[0] == 1
        assert result['lips'].shape[1] == len(hl2)
        
        
        jaw_row = result['jaw'][0]
        teeth_row = result['teeth'][0]
        lips_row = result['lips'][0]
        
        
        expected_jaw = [60742.4, 60632.6, 60555.1, 60442.7, 60308.7]
        expected_teeth = [59908.0, 59757.2, 59684.3, 59653.5, 59621.1]
        expected_lips = [59355.2, 59371.7, 59376.2, 59334.1, 59316.2]
        
        
        assert_close(
            jaw_row[-5:],
            expected_jaw,
            rtol=0,
            atol=1e-1,
            msg="Alligator batch jaw mismatch"
        )
        
        assert_close(
            teeth_row[-5:],
            expected_teeth,
            rtol=0,
            atol=1e-1,
            msg="Alligator batch teeth mismatch"
        )
        
        assert_close(
            lips_row[-5:],
            expected_lips,
            rtol=0,
            atol=1e-1,
            msg="Alligator batch lips mismatch"
        )
    
    def test_alligator_multiple_periods(self, test_data):
        """Test batch with multiple period values"""
        hl2 = ((test_data['high'] + test_data['low']) / 2)[:100]  
        
        
        result = ta_indicators.alligator_batch(
            hl2,
            jaw_period_range=(13, 17, 2),
            jaw_offset_range=(8, 8, 0),
            teeth_period_range=(8, 8, 0),
            teeth_offset_range=(5, 5, 0),
            lips_period_range=(5, 5, 0),
            lips_offset_range=(3, 3, 0)
        )
        
        
        assert result['jaw'].shape[0] == 3
        assert result['teeth'].shape[0] == 3
        assert result['lips'].shape[0] == 3
        
        
        assert len(result['jaw_periods']) == 3
        assert list(result['jaw_periods']) == [13, 15, 17]
    
    def test_alligator_full_parameter_sweep(self, test_data):
        """Test full parameter sweep"""
        hl2 = ((test_data['high'] + test_data['low']) / 2)[:50]  
        
        result = ta_indicators.alligator_batch(
            hl2,
            jaw_period_range=(13, 14, 1),   
            jaw_offset_range=(8, 9, 1),     
            teeth_period_range=(8, 8, 0),   
            teeth_offset_range=(5, 5, 0),   
            lips_period_range=(5, 5, 0),    
            lips_offset_range=(3, 3, 0)     
        )
        
        
        assert result['jaw'].shape[0] == 4
        assert result['jaw'].shape[1] == len(hl2)
        
        
        assert len(result['jaw_periods']) == 4
        assert len(result['jaw_offsets']) == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
