"""
Python binding tests for Ehlers PMA indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import my_project as rb
from test_utils import load_test_data
import os

class TestEhlersPma:
    @pytest.fixture(scope='class')
    def test_data(self):
        """Load test data for Ehlers PMA indicator."""
        return load_test_data()

    def test_ehlers_pma_accuracy(self, test_data):
        """Test Ehlers PMA matches expected values from Rust tests - mirrors check_ehlers_pma_accuracy"""
        
        close = test_data['close']
        predict, trigger = rb.ehlers_pma(close)
        
        
        expected_predict_last_five = [
            59161.97066327,
            59240.51785714,
            59260.29846939,
            59225.19005102,
            59192.78443878,
        ]
        expected_trigger_last_five = [
            59020.56403061,
            59141.96938776,
            59214.56709184,
            59232.46619898,
            59220.78227041,
        ]
        
        assert len(predict) == len(close)
        assert len(trigger) == len(close)
        
        
        np.testing.assert_allclose(
            predict[-5:],
            expected_predict_last_five,
            rtol=1e-6,
            err_msg="Ehlers PMA predict last 5 values mismatch"
        )
        
        np.testing.assert_allclose(
            trigger[-5:],
            expected_trigger_last_five,
            rtol=1e-6,
            err_msg="Ehlers PMA trigger last 5 values mismatch"
        )

    def test_ehlers_pma_default_candles(self, test_data):
        """Test Ehlers PMA with default parameters - mirrors check_ehlers_pma_default_candles"""
        close = test_data['close']
        
        
        predict, trigger = rb.ehlers_pma(close)
        assert len(predict) == len(close)
        assert len(trigger) == len(close)

    def test_ehlers_pma_empty_input(self):
        """Test Ehlers PMA fails with empty input - mirrors check_ehlers_pma_empty_input"""
        empty = np.array([])
        with pytest.raises(ValueError, match="Input data slice is empty"):
            rb.ehlers_pma(empty)

    def test_ehlers_pma_all_nan(self):
        """Test Ehlers PMA fails with all NaN values - mirrors check_ehlers_pma_all_nan"""
        all_nan = np.full(20, np.nan)
        with pytest.raises(ValueError, match="All values are NaN"):
            rb.ehlers_pma(all_nan)

    def test_ehlers_pma_insufficient_data(self):
        """Test Ehlers PMA fails with insufficient data - mirrors check_ehlers_pma_insufficient_data"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="Not enough valid data"):
            rb.ehlers_pma(data)
            
    def test_ehlers_pma_very_small_dataset(self):
        """Test Ehlers PMA fails with very small dataset - mirrors check_ehlers_pma_very_small_dataset"""
        single_point = np.array([42.0])
        with pytest.raises(ValueError, match="Not enough valid data"):
            rb.ehlers_pma(single_point)
            
        two_points = np.array([42.0, 43.0])
        with pytest.raises(ValueError, match="Not enough valid data"):
            rb.ehlers_pma(two_points)

    def test_ehlers_pma_nan_handling(self, test_data):
        """Test Ehlers PMA handles NaN values correctly - mirrors check_ehlers_pma_nan_handling"""
        close = test_data['close']
        
        predict, trigger = rb.ehlers_pma(close)
        assert len(predict) == len(close)
        assert len(trigger) == len(close)
        
        
        if len(predict) > 20:
            
            assert not np.any(np.isnan(predict[20:])), "Found unexpected NaN in predict after warmup"
            
            assert not np.any(np.isnan(trigger[20:])), "Found unexpected NaN in trigger after warmup"
        
        
        assert np.all(np.isnan(predict[:13])), "Expected NaN in predict warmup period (first 13)"
        
        assert np.all(np.isnan(trigger[:16])), "Expected NaN in trigger warmup period (first 16)"
    
    def test_ehlers_pma_streaming(self, test_data):
        """Test Ehlers PMA streaming matches batch calculation - mirrors check_ehlers_pma_streaming"""
        close = test_data['close'][:100]  
        
        
        batch_predict, batch_trigger = rb.ehlers_pma(close)
        
        
        stream = rb.EhlersPmaStream()
        stream_values = []
        
        for price in close:
            result = stream.update(price)
            stream_values.append(result)
        
        
        stream_predict = []
        stream_trigger = []
        for result in stream_values:
            if result is None:
                stream_predict.append(np.nan)
                stream_trigger.append(np.nan)
            else:
                predict, trigger = result
                stream_predict.append(predict)
                stream_trigger.append(trigger)
        
        stream_predict = np.array(stream_predict)
        stream_trigger = np.array(stream_trigger)
        
        
        
        for i in range(17, len(close)):  
            if not np.isnan(batch_predict[i]) and not np.isnan(stream_predict[i]):
                np.testing.assert_allclose(
                    batch_predict[i], 
                    stream_predict[i], 
                    rtol=1e-9, 
                    err_msg=f"Predict streaming mismatch at index {i}"
                )
            if not np.isnan(batch_trigger[i]) and not np.isnan(stream_trigger[i]):
                np.testing.assert_allclose(
                    batch_trigger[i], 
                    stream_trigger[i], 
                    rtol=1e-9,
                    err_msg=f"Trigger streaming mismatch at index {i}"
                )

    def test_ehlers_pma_batch(self, test_data):
        """Test Ehlers PMA batch processing - mirrors batch functionality"""
        close = test_data['close']
        
        
        result = rb.ehlers_pma_batch(close)
        
        assert 'values' in result
        assert 'lines' in result  
        
        
        assert result['values'].shape[0] == 2  
        assert result['values'].shape[1] == len(close)
        
        
        predict_row = result['values'][0]
        trigger_row = result['values'][1]
        
        
        expected_predict_last_five = [
            59161.97066327,
            59240.51785714,
            59260.29846939,
            59225.19005102,
            59192.78443878,
        ]
        expected_trigger_last_five = [
            59020.56403061,
            59141.96938776,
            59214.56709184,
            59232.46619898,
            59220.78227041,
        ]
        
        
        np.testing.assert_allclose(
            predict_row[-5:],
            expected_predict_last_five,
            rtol=1e-6,
            err_msg="Ehlers PMA batch predict mismatch"
        )
        
        np.testing.assert_allclose(
            trigger_row[-5:],
            expected_trigger_last_five,
            rtol=1e-6,
            err_msg="Ehlers PMA batch trigger mismatch"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
