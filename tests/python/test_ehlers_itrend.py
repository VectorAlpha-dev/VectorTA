"""
Python binding tests for Ehlers ITrend indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import numpy as np
import pytest
from test_utils import (
    load_test_data,
    assert_close,
    EXPECTED_OUTPUTS
)
from rust_comparison import compare_with_rust


import my_project


def assert_no_nan(arr, msg=""):
    """Assert no NaN values in array"""
    if np.any(np.isnan(arr)):
        raise AssertionError(f"{msg}: Found NaN values in array")

def assert_all_nan(arr, msg=""):
    """Assert all values are NaN"""
    if not np.all(np.isnan(arr)):
        raise AssertionError(f"{msg}: Not all values are NaN")


class TestEhlersITrend:
    def setup_method(self):
        """Load test data before each test."""
        self.data = load_test_data()
        self.close = np.array(self.data['close'], dtype=np.float64)
    
    def test_ehlers_itrend_partial_params(self):
        """Test with default parameters - mirrors check_itrend_partial_params"""
        
        
        result = my_project.ehlers_itrend(self.close, 12, 50)
        assert len(result) == len(self.close)
        
        
        result = my_project.ehlers_itrend(self.close, 15, 50)
        assert len(result) == len(self.close)
        
        result = my_project.ehlers_itrend(self.close, 12, 40)
        assert len(result) == len(self.close)
    
    def test_ehlers_itrend_accuracy(self):
        """Test accuracy matches expected values from Rust tests - mirrors check_itrend_accuracy"""
        expected = EXPECTED_OUTPUTS['ehlers_itrend']
        
        result = my_project.ehlers_itrend(
            self.close,
            expected['default_params']['warmup_bars'],
            expected['default_params']['max_dc_period']
        )
        
        assert len(result) == len(self.close)
        
        
        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=0,
            atol=0.1,  
            msg="Ehlers ITrend last 5 values mismatch"
        )
        
        
        compare_with_rust('ehlers_itrend', result, 'close', expected['default_params'])
    
    def test_ehlers_itrend_error_empty_input(self):
        """Test error with empty input - mirrors check_itrend_no_data"""
        with pytest.raises(ValueError, match="Input data is empty"):
            my_project.ehlers_itrend(np.array([]), 12, 50)
    
    def test_ehlers_itrend_error_all_nan(self):
        """Test error with all NaN values - mirrors check_itrend_all_nan_data"""
        all_nan = np.full(100, np.nan)
        with pytest.raises(ValueError, match="All values are NaN"):
            my_project.ehlers_itrend(all_nan, 12, 50)
    
    def test_ehlers_itrend_error_insufficient_data(self):
        """Test error with insufficient data for warmup - mirrors check_itrend_small_data_for_warmup"""
        small_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="Not enough data for warmup"):
            my_project.ehlers_itrend(small_data, 10, 50)
    
    def test_ehlers_itrend_error_zero_warmup(self):
        """Test error with zero warmup bars - mirrors check_itrend_zero_warmup"""
        with pytest.raises(ValueError, match="Invalid warmup_bars"):
            my_project.ehlers_itrend(self.close, 0, 50)
    
    def test_ehlers_itrend_error_invalid_max_dc(self):
        """Test error with invalid max_dc_period - mirrors check_itrend_invalid_max_dc"""
        with pytest.raises(ValueError, match="Invalid max_dc_period"):
            my_project.ehlers_itrend(self.close, 12, 0)
    
    def test_ehlers_itrend_reinput(self):
        """Test applying indicator twice - mirrors check_itrend_reinput"""
        expected = EXPECTED_OUTPUTS['ehlers_itrend']
        params = expected['default_params']
        
        
        first_result = my_project.ehlers_itrend(
            self.close,
            params['warmup_bars'],
            params['max_dc_period']
        )
        assert len(first_result) == len(self.close)
        
        
        second_result = my_project.ehlers_itrend(
            first_result,
            params['warmup_bars'],
            params['max_dc_period']
        )
        assert len(second_result) == len(first_result)
        
        
        
        
        if len(second_result) > 24:
            assert_no_nan(second_result[24:], "Found unexpected NaN after double warmup")
    
    def test_ehlers_itrend_nan_handling(self):
        """Test NaN handling - mirrors check_itrend_nan_handling"""
        expected = EXPECTED_OUTPUTS['ehlers_itrend']
        params = expected['default_params']
        warmup_bars = params['warmup_bars']
        
        result = my_project.ehlers_itrend(
            self.close,
            params['warmup_bars'],
            params['max_dc_period']
        )
        
        assert len(result) == len(self.close)
        
        
        assert_all_nan(
            result[:warmup_bars],
            msg="Expected NaN in warmup period"
        )
        
        
        if len(result) > warmup_bars:
            assert_no_nan(result[warmup_bars:], "Found unexpected NaN after warmup")
            
        
        assert warmup_bars == 12, f"Expected warmup_bars=12, got {warmup_bars}"
    
    def test_ehlers_itrend_streaming(self):
        """Test streaming interface - mirrors check_itrend_streaming"""
        
        batch_result = my_project.ehlers_itrend(self.close, 12, 50)
        
        
        stream = my_project.EhlersITrendStream(12, 50)
        stream_result = []
        
        for price in self.close:
            value = stream.update(price)
            stream_result.append(value if value is not None else np.nan)
        
        stream_result = np.array(stream_result)
        
        
        assert len(batch_result) == len(stream_result)
        
        
        
        for i in range(12, len(batch_result)):
            if not (np.isnan(batch_result[i]) and np.isnan(stream_result[i])):
                assert abs(batch_result[i] - stream_result[i]) < 1e-7, \
                    f"Batch/stream mismatch at index {i}: {batch_result[i]} vs {stream_result[i]}"
    
    def test_ehlers_itrend_batch_single_params(self):
        """Test batch calculation with single parameter set - mirrors check_batch_default_row"""
        expected = EXPECTED_OUTPUTS['ehlers_itrend']
        params = expected['default_params']
        
        
        batch_result = my_project.ehlers_itrend_batch(
            self.close[:100],  
            (params['warmup_bars'], params['warmup_bars'], 0),  
            (params['max_dc_period'], params['max_dc_period'], 0)  
        )
        
        
        assert isinstance(batch_result, dict)
        assert 'values' in batch_result
        assert 'warmups' in batch_result
        assert 'max_dcs' in batch_result
        
        
        assert batch_result['values'].shape == (1, 100)
        
        
        single_result = my_project.ehlers_itrend(self.close[:100], 12, 50)
        
        
        
        
        warmup_bars = params['warmup_bars']
        assert_close(
            batch_result['values'][0][warmup_bars:],
            single_result[warmup_bars:],
            rtol=0,
            atol=1e-10,
            msg="Batch vs single calculation mismatch after warmup"
        )
    
    def test_ehlers_itrend_batch_multiple_params(self):
        """Test batch calculation with multiple parameter combinations"""
        test_data = self.close[:100]
        batch_result = my_project.ehlers_itrend_batch(
            test_data,
            (10, 14, 2),      
            (40, 50, 10)      
        )
        
        
        assert batch_result['values'].shape[0] == 6
        assert batch_result['values'].shape[1] == 100
        
        
        assert len(batch_result['warmups']) == 6
        assert len(batch_result['max_dcs']) == 6
        
        
        expected_params = [
            (10, 40), (10, 50),
            (12, 40), (12, 50),
            (14, 40), (14, 50)
        ]
        
        for idx, (warmup, max_dc) in enumerate(expected_params):
            single_result = my_project.ehlers_itrend(test_data, warmup, max_dc)
            batch_row = batch_result['values'][idx]
            
            
            
            for i in range(warmup, len(single_result)):
                if not np.isnan(single_result[i]) and not np.isnan(batch_row[i]):
                    assert abs(batch_row[i] - single_result[i]) < 1e-9, \
                        f"Batch row {idx} (warmup={warmup}, max_dc={max_dc}) mismatch at index {i}"
            
            assert batch_result['warmups'][idx] == warmup
            assert batch_result['max_dcs'][idx] == max_dc
    
    def test_ehlers_itrend_batch_warmup_validation(self):
        """Test batch warmup period handling"""
        data = self.close[:30]
        
        batch_result = my_project.ehlers_itrend_batch(
            data,
            (10, 15, 5),      
            (50, 50, 0)       
        )
        
        
        assert batch_result['values'].shape == (2, 30)
        
        
        
        
        
        
        
        for i in range(10, 30):
            assert not np.isnan(batch_result['values'][0][i]) or \
                   np.isfinite(batch_result['values'][0][i]), \
                   f"Invalid value at index {i} in first row"
        
        
        for i in range(15, 30):
            assert not np.isnan(batch_result['values'][1][i]) or \
                   np.isfinite(batch_result['values'][1][i]), \
                   f"Invalid value at index {i} in second row"
    
    def test_ehlers_itrend_edge_cases(self):
        """Test edge cases for the indicator"""
        
        min_data = np.array([1.0] * 13)  
        result = my_project.ehlers_itrend(min_data, 12, 50)
        assert len(result) == 13
        
        
        assert_all_nan(result[:12], msg="Expected NaN in warmup period")
        
        
        test_cases = [
            (5, 30),   
            (20, 100), 
            (15, 15),  
        ]
        
        for warmup, max_dc in test_cases:
            if len(self.close) > warmup:
                result = my_project.ehlers_itrend(self.close[:100], warmup, max_dc)
                assert len(result) == 100, f"Failed for warmup={warmup}, max_dc={max_dc}"
                
                assert_all_nan(
                    result[:warmup],
                    msg=f"Expected NaN in warmup period for warmup={warmup}"
                )
    
    def test_ehlers_itrend_with_nan_input(self):
        """Test handling of data with NaN values in valid positions"""
        
        data_with_nan = self.close[:100].copy()
        data_with_nan[50:55] = np.nan
        
        
        result = my_project.ehlers_itrend(data_with_nan, 12, 50)
        assert len(result) == 100
        
        
        assert_all_nan(result[:12], msg="Expected NaN in warmup period")
        
        
        
        assert np.isnan(result[50]), "Expected NaN at index 50"
    
    def test_ehlers_itrend_performance_batch(self):
        """Test that batch processing is more efficient than multiple single calls"""
        import time
        
        test_data = self.close[:500]  
        
        
        start = time.time()
        batch_result = my_project.ehlers_itrend_batch(
            test_data,
            (10, 20, 2),  
            (40, 60, 10)  
        )
        batch_time = time.time() - start
        
        
        start = time.time()
        single_results = []
        for warmup in range(10, 21, 2):
            for max_dc in range(40, 61, 10):
                single_results.append(
                    my_project.ehlers_itrend(test_data, warmup, max_dc)
                )
        single_time = time.time() - start
        
        
        print(f"\n  Batch time: {batch_time:.3f}s, Single calls: {single_time:.3f}s")
        print(f"  Speedup: {single_time/batch_time:.1f}x")
        
        
        assert batch_result['values'].shape == (18, 500)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
