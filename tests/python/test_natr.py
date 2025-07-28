"""
Python binding tests for NATR indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import the built module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'target/wheels'))

try:
    import my_project as ta_indicators
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS


class TestNatr:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_natr_accuracy(self, test_data):
        """Test NATR matches expected values from Rust tests"""
        period = EXPECTED_OUTPUTS['natr']['default_params']['period']
        
        # Run NATR with default parameters
        result = ta_indicators.natr(test_data['high'], test_data['low'], test_data['close'], period)
        
        # Check the length matches
        assert len(result) == len(test_data['close'])
        
        # Get the last 5 values
        result_last_5 = result[-5:]
        expected_last_5 = EXPECTED_OUTPUTS['natr']['last_5_values']
        
        # Check accuracy
        assert_close(result_last_5, expected_last_5, rtol=1e-8, 
                    msg="NATR last 5 values don't match expected")
    
    def test_natr_errors(self):
        """Test error handling"""
        # Test with zero period
        with pytest.raises(ValueError):
            ta_indicators.natr(np.array([10.0, 20.0]), np.array([5.0, 10.0]), np.array([7.0, 15.0]), 0)
        
        # Test with period exceeding data length
        with pytest.raises(ValueError):
            ta_indicators.natr(np.array([10.0, 20.0]), np.array([5.0, 10.0]), np.array([7.0, 15.0]), 10)
        
        # Test with all NaN values
        with pytest.raises(ValueError):
            ta_indicators.natr(np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), 2)
    
    def test_natr_streaming(self, test_data):
        """Test NATR streaming functionality"""
        period = 14
        
        # Get batch result
        batch_result = ta_indicators.natr(test_data['high'], test_data['low'], test_data['close'], period)
        
        # Test streaming
        stream = ta_indicators.NatrStream(period)
        stream_results = []
        
        for i in range(len(test_data['high'])):
            result = stream.update(test_data['high'][i], test_data['low'][i], test_data['close'][i])
            if result is not None:
                stream_results.append(result)
            else:
                stream_results.append(np.nan)
        
        # Convert to numpy array
        stream_results = np.array(stream_results)
        
        # Compare - they should be very close
        assert_close(batch_result, stream_results, rtol=1e-9, 
                    msg="NATR streaming results don't match batch results")
    
    def test_natr_batch(self, test_data):
        """Test NATR batch functionality"""
        # Test single period batch (equivalent to single calculation)
        batch_result = ta_indicators.natr_batch(
            test_data['high'], test_data['low'], test_data['close'], 
            (14, 14, 1)
        )
        
        # Verify structure
        assert 'values' in batch_result
        assert 'periods' in batch_result
        assert batch_result['values'].shape == (1, len(test_data['close']))
        assert len(batch_result['periods']) == 1
        assert batch_result['periods'][0] == 14
        
        # Compare with single calculation
        single_result = ta_indicators.natr(test_data['high'], test_data['low'], test_data['close'], 14)
        assert_close(batch_result['values'][0], single_result, rtol=1e-9,
                    msg="NATR batch doesn't match single calculation")
        
        # Test multiple periods
        batch_result_multi = ta_indicators.natr_batch(
            test_data['high'], test_data['low'], test_data['close'],
            (10, 20, 5)  # periods 10, 15, 20
        )
        
        assert batch_result_multi['values'].shape == (3, len(test_data['close']))
        assert len(batch_result_multi['periods']) == 3
        assert list(batch_result_multi['periods']) == [10, 15, 20]
        
        # Verify each row matches individual calculations
        for i, period in enumerate([10, 15, 20]):
            single = ta_indicators.natr(test_data['high'], test_data['low'], test_data['close'], period)
            assert_close(batch_result_multi['values'][i], single, rtol=1e-9,
                        msg=f"NATR batch row {i} (period {period}) doesn't match single calculation")
    
    def test_natr_with_nans(self, test_data):
        """Test NATR handles NaN values correctly"""
        # Create data with some NaN values
        high_with_nans = test_data['high'].copy()
        low_with_nans = test_data['low'].copy()
        close_with_nans = test_data['close'].copy()
        
        # Insert some NaNs
        high_with_nans[10:15] = np.nan
        low_with_nans[10:15] = np.nan
        close_with_nans[10:15] = np.nan
        
        # Should still compute where possible
        result = ta_indicators.natr(high_with_nans, low_with_nans, close_with_nans, 14)
        assert len(result) == len(test_data['close'])
        
        # Check that we have some valid values after the NaN region
        assert not np.all(np.isnan(result[20:]))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
