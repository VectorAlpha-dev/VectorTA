"""
Python binding tests for NAMA indicator.
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


class TestNama:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_nama_partial_params(self, test_data):
        """Test NAMA with partial parameters (None values) - mirrors check_nama_partial_params"""
        close = test_data['close']
        
        
        result = ta_indicators.nama(close, 30)
        assert len(result) == len(close)
        
        
        warmup = 29  
        non_nan_count = np.sum(~np.isnan(result[warmup:]))
        assert non_nan_count > 0, "Should have non-NaN values after warmup"
    
    def test_nama_accuracy(self, test_data):
        """Test NAMA matches expected values from Rust tests - mirrors check_nama_accuracy"""
        
        close = test_data['close']
        
        
        result = ta_indicators.nama(close, period=30)
        
        assert len(result) == len(close)
        
        
        
        expected_last_five = np.array([
            59248.42400839,
            59226.18226649,
            59167.91952826,
            59163.80438196,
            59009.01273427
        ])
        
        
        last_5_actual = result[-5:]
        
        assert_close(last_5_actual, expected_last_five, rtol=0.0, atol=1e-6,
                    msg="NAMA last 5 values mismatch")
        
        
        
    
    def test_nama_default_candles(self, test_data):
        """Test NAMA with default parameters - mirrors check_nama_default_candles"""
        close = test_data['close']
        
        
        result = ta_indicators.nama(close, 30)
        assert len(result) == len(close)
        
        
        
        for i in range(29):
            assert np.isnan(result[i]), f"Expected NaN at index {i} during warmup"
        
        
        non_nan_count = np.sum(~np.isnan(result))
        assert non_nan_count > 0, "Should have some non-NaN values after warmup"
        
        
        assert non_nan_count == len(close) - 29, "All values after warmup should be non-NaN"
    
    def test_nama_zero_period(self):
        """Test NAMA fails with zero period - mirrors check_nama_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.nama(input_data, period=0)
    
    def test_nama_period_exceeds_length(self):
        """Test NAMA fails when period exceeds data length - mirrors check_nama_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.nama(data_small, period=10)
    
    def test_nama_very_small_dataset(self):
        """Test NAMA with very small dataset - mirrors check_nama_very_small_dataset"""
        single_point = np.array([42.0])
        
        
        result = ta_indicators.nama(single_point, period=1)
        assert len(result) == 1
        
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.nama(single_point, period=2)
    
    def test_nama_empty_input(self):
        """Test NAMA fails with empty input - mirrors check_nama_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty|Empty input"):
            ta_indicators.nama(empty, period=5)
    
    def test_nama_invalid_period(self):
        """Test NAMA fails with invalid period - mirrors check_nama_invalid_period"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.nama(data, period=0)
        
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.nama(data, period=10)
    
    def test_nama_nan_handling(self):
        """Test NAMA handles NaN values correctly - mirrors check_nama_nan_handling"""
        
        data_nan = np.array([np.nan, np.nan, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        
        result = ta_indicators.nama(data_nan, period=3)
        assert len(result) == len(data_nan)
        
        
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        
        
        
        
        for i in range(4):
            assert np.isnan(result[i]), f"Expected NaN at index {i} during warmup"
        
        
        for i in range(4, len(data_nan)):
            assert not np.isnan(result[i]), f"Expected non-NaN at index {i} after warmup"
    
    def test_nama_streaming(self, test_data):
        """Test NAMA streaming matches batch calculation - mirrors check_nama_streaming"""
        
        data = np.array([
            100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0,
            107.0, 109.0, 111.0, 110.0, 112.0, 114.0, 113.0, 115.0
        ])
        period = 5
        
        
        batch_result = ta_indicators.nama(data, period=period)
        
        
        stream = ta_indicators.NamaStream(period=period)
        stream_values = []
        
        for price in data:
            result = stream.update(price)
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        
        assert len(batch_result) == len(stream_values)
        
        
        warmup = 4  
        for i in range(warmup, len(data)):
            if np.isfinite(batch_result[i]) and np.isfinite(stream_values[i]):
                assert_close(batch_result[i], stream_values[i], rtol=1e-10, atol=1e-10,
                           msg=f"NAMA streaming mismatch at index {i}")
    
    def test_nama_batch(self, test_data):
        """Test NAMA batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        
        result = ta_indicators.nama_batch(
            close,
            period_range=(30, 30, 0)  
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        assert len(result['periods']) == 1
        assert result['periods'][0] == 30
        
        
        default_row = result['values'][0]
        
        
        for i in range(29):  
            assert np.isnan(default_row[i]), f"Expected NaN at index {i} during warmup"
        
        
        non_nan_count = np.sum(~np.isnan(default_row))
        assert non_nan_count > 0, "Should have non-NaN values after warmup"
    
    def test_nama_batch_sweep(self, test_data):
        """Test NAMA batch with parameter sweep"""
        close = test_data['close'][:100]  
        
        
        result = ta_indicators.nama_batch(
            close,
            period_range=(10, 30, 10)  
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        
        assert result['values'].shape[0] == 3
        assert result['values'].shape[1] == 100
        assert len(result['periods']) == 3
        
        
        expected_periods = [10, 20, 30]
        for i, period in enumerate(expected_periods):
            assert result['periods'][i] == period
            
            
            row = result['values'][i]
            warmup = period - 1  
            
            for j in range(warmup):
                assert np.isnan(row[j]), f"Expected NaN at index {j} for period {period}"
            
            
            non_nan_count = np.sum(~np.isnan(row))
            assert non_nan_count > 0, f"Should have non-NaN values for period {period}"
            
            
            if i > 0:
                prev_row = result['values'][i-1]
                
                valid_idx = ~(np.isnan(row) | np.isnan(prev_row))
                if np.any(valid_idx):
                    assert not np.allclose(row[valid_idx], prev_row[valid_idx], rtol=1e-10),\
                        f"Period {period} should produce different results than previous period"
    
    def test_nama_all_nan_input(self):
        """Test NAMA with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.nama(all_nan, period=30)
    
    def test_nama_with_candles(self, test_data):
        """Test NAMA with OHLC candle data"""
        
        
        close = test_data['close']
        high = test_data['high']
        low = test_data['low']
        
        
        result_close = ta_indicators.nama(close, period=30)
        
        
        
        
        assert len(result_close) == len(close)
        
        
        for i in range(29):  
            assert np.isnan(result_close[i]), f"Expected NaN at index {i} during warmup"
    
    def test_nama_edge_cases(self):
        """Test NAMA with various edge cases"""
        
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ta_indicators.nama(data, period=3)
        assert len(result) == 5
        
        
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert not np.isnan(result[2])
        
        
        constant = np.full(50, 100.0)
        result_const = ta_indicators.nama(constant, period=10)
        assert len(result_const) == 50
        
        
        
        for i in range(9, 50):  
            assert np.isfinite(result_const[i])
        
        
        alternating = np.array([100.0, 50.0] * 25)
        result_alt = ta_indicators.nama(alternating, period=5)
        assert len(result_alt) == 50
    
    def test_nama_infinity_handling(self):
        """Test NAMA handles infinity values correctly"""
        
        data_inf = np.array([100.0, 102.0, 101.0, np.inf, 105.0, 104.0, 106.0, 108.0])
        
        
        result = ta_indicators.nama(data_inf, period=3)
        assert len(result) == len(data_inf)
        
        
        non_nan_count = np.sum(~np.isnan(result))
        assert non_nan_count > 0, "Should have some non-NaN values despite infinity"
        
        
        data_neginf = np.array([100.0, 102.0, 101.0, -np.inf, 105.0, 104.0, 106.0, 108.0])
        result_neginf = ta_indicators.nama(data_neginf, period=3)
        assert len(result_neginf) == len(data_neginf)
    
    def test_nama_large_dataset(self):
        """Test NAMA with large dataset for performance"""
        
        large_data = np.random.randn(10000) * 10 + 100
        
        
        result = ta_indicators.nama(large_data, period=50)
        assert len(result) == len(large_data)
        
        
        for i in range(49):  
            assert np.isnan(result[i]), f"Expected NaN at index {i} during warmup"
        
        
        non_nan_count = np.sum(~np.isnan(result[49:]))
        assert non_nan_count == len(large_data) - 49, "All values after warmup should be non-NaN"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
