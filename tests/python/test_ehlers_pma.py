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
        # Use close source to match Rust test (not hl2)
        close = test_data['close']
        predict, trigger = rb.ehlers_pma(close)
        
        # Reference values from Rust tests (using close source with TradingView parity)
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
        
        # The expected values are exactly the last 5 values of our output
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
        
        # Ehlers PMA has no parameters, just test it runs
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
        
        # After warmup period, no NaN values should exist
        if len(predict) > 20:
            # Check predict values after warmup (13 with 1-bar lag)
            assert not np.any(np.isnan(predict[20:])), "Found unexpected NaN in predict after warmup"
            # Check trigger values after warmup (16 with 1-bar lag)
            assert not np.any(np.isnan(trigger[20:])), "Found unexpected NaN in trigger after warmup"
        
        # First 12 values should be NaN for predict (Python binding warmup)
        assert np.all(np.isnan(predict[:12])), "Expected NaN in predict warmup period"
        
        # First 15 values should be NaN for trigger (Python binding warmup)
        assert np.all(np.isnan(trigger[:15])), "Expected NaN in trigger warmup period"
    
    def test_ehlers_pma_streaming(self, test_data):
        """Test Ehlers PMA streaming matches batch calculation - mirrors check_ehlers_pma_streaming"""
        close = test_data['close'][:100]  # Use first 100 values for speed
        
        # Batch calculation
        batch_predict, batch_trigger = rb.ehlers_pma(close)
        
        # Streaming calculation
        stream = rb.EhlersPmaStream()
        stream_values = []
        
        for price in close:
            result = stream.update(price)
            stream_values.append(result)
        
        # Convert stream results to arrays
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
        
        # Compare batch vs streaming after warmup
        # Note: Streaming has 1 additional lag due to implementation
        for i in range(17, len(close)):  # Start after warmup
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
        
        # Ehlers PMA batch with no parameters (since it has none to sweep)
        result = rb.ehlers_pma_batch(close)
        
        assert 'values' in result
        assert 'lines' in result  # Multi-output indicator
        
        # Should have 1 combination (no parameters to vary)
        assert result['values'].shape[0] == 2  # predict and trigger rows
        assert result['values'].shape[1] == len(close)
        
        # Extract predict and trigger rows
        predict_row = result['values'][0]
        trigger_row = result['values'][1]
        
        # Reference values from Rust tests
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
        
        # Check last 5 values match expected
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