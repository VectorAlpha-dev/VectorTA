"""
Python binding tests for LINEARREG_SLOPE indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'target/wheels'))

try:
    import my_project as ta_indicators
except ImportError:
    
    try:
        import my_project as ta_indicators
    except ImportError:
        pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS


class TestLinearRegSlope:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_linearreg_slope_partial_params(self, test_data):
        """Test LINEARREG_SLOPE with default parameters - mirrors check_linearreg_slope_partial_params"""
        close = test_data['close']
        
        
        result = ta_indicators.linearreg_slope(close, 14)
        assert len(result) == len(close)
    
    def test_linearreg_slope_accuracy(self):
        """Test LINEARREG_SLOPE matches expected values from Rust tests - mirrors check_linearreg_slope_accuracy"""
        input_data = np.array([100.0, 98.0, 95.0, 90.0, 85.0, 80.0, 78.0, 77.0, 79.0, 81.0])
        
        result = ta_indicators.linearreg_slope(input_data, period=5)
        
        assert len(result) == len(input_data)
        
        
        for i in range(4):
            assert np.isnan(result[i]), f"Expected NaN at index {i}, got {result[i]}"
        
        
        expected_values = [-3.8, -4.6, -4.4, -3.3, -1.5, 0.3]
        
        for i, expected in enumerate(expected_values, start=4):
            assert_close(result[i], expected, rtol=1e-9, atol=1e-9,
                        msg=f"LINEARREG_SLOPE value mismatch at index {i}")
    
    def test_linearreg_slope_zero_period(self):
        """Test LINEARREG_SLOPE fails with zero period - mirrors check_linearreg_slope_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.linearreg_slope(input_data, period=0)
    
    def test_linearreg_slope_period_exceeds_length(self):
        """Test LINEARREG_SLOPE fails when period exceeds data length - mirrors check_linearreg_slope_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.linearreg_slope(data_small, period=10)
    
    def test_linearreg_slope_very_small_dataset(self):
        """Test LINEARREG_SLOPE fails with insufficient data - mirrors check_linearreg_slope_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.linearreg_slope(single_point, period=14)
    
    def test_linearreg_slope_empty_input(self):
        """Test LINEARREG_SLOPE fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data provided"):
            ta_indicators.linearreg_slope(empty, period=14)
    
    def test_linearreg_slope_nan_handling(self, test_data):
        """Test LINEARREG_SLOPE handles NaN values correctly - mirrors check_linearreg_slope_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.linearreg_slope(close, period=14)
        assert len(result) == len(close)
        
        
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"
        
        
        assert np.all(np.isnan(result[:13])), "Expected NaN in warmup period"
    
    def test_linearreg_slope_all_nan_input(self):
        """Test LINEARREG_SLOPE with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.linearreg_slope(all_nan, period=14)
    
    def test_linearreg_slope_streaming(self, test_data):
        """Test LINEARREG_SLOPE streaming matches batch calculation - mirrors streaming tests"""
        close = test_data['close'][:100]  
        period = 14
        
        
        batch_result = ta_indicators.linearreg_slope(close, period=period)
        
        
        stream = ta_indicators.LinearRegSlopeStream(period=period)
        stream_values = []
        
        for price in close:
            result = stream.update(price)
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        
        assert len(batch_result) == len(stream_values)
        
        
        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, atol=1e-9, 
                        msg=f"LINEARREG_SLOPE streaming mismatch at index {i}")
    
    def test_linearreg_slope_batch_single(self, test_data):
        """Test LINEARREG_SLOPE batch processing with single parameter set"""
        close = test_data['close'][:100]  
        
        result = ta_indicators.linearreg_slope_batch(
            close,
            period_range=(14, 14, 0)  
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        assert len(result['periods']) == 1
        assert result['periods'][0] == 14
        
        
        batch_row = result['values'][0]
        single_result = ta_indicators.linearreg_slope(close, period=14)
        
        
        for i in range(len(close)):
            if np.isnan(single_result[i]) and np.isnan(batch_row[i]):
                continue
            assert_close(batch_row[i], single_result[i], rtol=1e-9, atol=1e-9,
                        msg=f"Batch vs single mismatch at index {i}")
    
    def test_linearreg_slope_batch_multiple(self, test_data):
        """Test LINEARREG_SLOPE batch processing with multiple periods"""
        close = test_data['close'][:50]  
        
        result = ta_indicators.linearreg_slope_batch(
            close,
            period_range=(10, 20, 5)  
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        
        assert result['values'].shape[0] == 3
        assert result['values'].shape[1] == len(close)
        assert len(result['periods']) == 3
        
        
        expected_periods = [10, 15, 20]
        for i, period in enumerate(expected_periods):
            assert result['periods'][i] == period
            
            
            batch_row = result['values'][i]
            single_result = ta_indicators.linearreg_slope(close, period=period)
            
            for j in range(len(close)):
                if np.isnan(single_result[j]) and np.isnan(batch_row[j]):
                    continue
                assert_close(batch_row[j], single_result[j], rtol=1e-9, atol=1e-9,
                            msg=f"Period {period} mismatch at index {j}")
    
    def test_linearreg_slope_linear_data(self):
        """Test LINEARREG_SLOPE with perfectly linear data"""
        
        linear_data = np.array([2*i + 10 for i in range(20)], dtype=np.float64)
        
        result = ta_indicators.linearreg_slope(linear_data, period=14)
        
        
        for i in range(13, len(result)):
            assert_close(result[i], 2.0, rtol=1e-9, atol=1e-9,
                        msg=f"Expected slope=2.0 for linear data at index {i}")
    
    def test_linearreg_slope_constant_data(self):
        """Test LINEARREG_SLOPE with constant data"""
        
        constant_data = np.full(20, 100.0)
        
        result = ta_indicators.linearreg_slope(constant_data, period=10)
        
        
        for i in range(9, len(result)):
            assert_close(result[i], 0.0, rtol=1e-9, atol=1e-9,
                        msg=f"Expected slope=0.0 for constant data at index {i}")
    
    def test_linearreg_slope_batch_warmup_periods(self, test_data):
        """Test that batch processing correctly handles different warmup periods"""
        close = test_data['close'][:30]
        
        result = ta_indicators.linearreg_slope_batch(
            close,
            period_range=(5, 15, 5)  
        )
        
        
        periods = [5, 10, 15]
        for row_idx, period in enumerate(periods):
            row = result['values'][row_idx]
            
            
            for i in range(period - 1):
                assert np.isnan(row[i]), f"Expected NaN at index {i} for period {period}"
            
            
            for i in range(period - 1, len(row)):
                assert not np.isnan(row[i]), f"Unexpected NaN at index {i} for period {period}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
