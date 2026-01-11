"""
Python binding tests for MEDIUM_AD indicator.
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


class TestMediumAd:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_medium_ad_partial_params(self, test_data):
        """Test MEDIUM_AD with partial parameters - mirrors check_medium_ad_partial_params"""
        close = test_data['close']
        
        
        result = ta_indicators.medium_ad(close, 5)
        assert len(result) == len(close)
    
    def test_medium_ad_accuracy(self, test_data):
        """Test MEDIUM_AD matches expected values from Rust tests - mirrors check_medium_ad_accuracy"""
        
        hl2 = (test_data['high'] + test_data['low']) / 2
        
        result = ta_indicators.medium_ad(hl2, period=5)
        
        assert len(result) == len(hl2)
        
        
        expected_last_five = [220.0, 78.5, 126.5, 48.0, 28.5]
        
        
        assert_close(
            result[-5:],
            expected_last_five,
            rtol=0,
            atol=1e-1,
            msg="MEDIUM_AD last 5 values mismatch"
        )
    
    def test_medium_ad_default_candles(self, test_data):
        """Test MEDIUM_AD with default parameters - mirrors check_medium_ad_default_candles"""
        close = test_data['close']
        
        
        result = ta_indicators.medium_ad(close, 5)
        assert len(result) == len(close)
    
    def test_medium_ad_zero_period(self):
        """Test MEDIUM_AD fails with zero period - mirrors check_medium_ad_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.medium_ad(input_data, period=0)
    
    def test_medium_ad_period_exceeds_length(self):
        """Test MEDIUM_AD fails when period exceeds data length - mirrors check_medium_ad_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.medium_ad(data_small, period=10)
    
    def test_medium_ad_very_small_dataset(self):
        """Test MEDIUM_AD fails with insufficient data - mirrors check_medium_ad_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.medium_ad(single_point, period=5)
    
    def test_medium_ad_empty_input(self):
        """Test MEDIUM_AD fails with empty input - mirrors check_medium_ad_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data provided|Empty input"):
            ta_indicators.medium_ad(empty, period=5)
    
    
    def test_medium_ad_nan_handling(self, test_data):
        """Test MEDIUM_AD handles NaN values correctly - mirrors check_medium_ad_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.medium_ad(close, period=5)
        assert len(result) == len(close)
        
        
        for i in range(4):
            assert np.isnan(result[i]), f"Expected NaN during warmup at index {i}, got {result[i]}"
        
        
        if len(result) > 60:
            for i in range(60, len(result)):
                if np.isnan(result[i]):
                    
                    assert np.isnan(close[i]) or any(np.isnan(close[max(0, i-4):i+1])), \
                        f"Found unexpected NaN at index {i}"
    
    def test_medium_ad_batch_single_period(self, test_data):
        """Test batch with single period value - mirrors batch tests"""
        close = test_data['close']
        
        
        batch_result = ta_indicators.medium_ad_batch(close, period_range=(5, 5, 0))
        
        
        assert 'values' in batch_result
        assert 'periods' in batch_result
        assert batch_result['values'].shape[0] == 1
        assert batch_result['values'].shape[1] == len(close)
        assert len(batch_result['periods']) == 1
        assert batch_result['periods'][0] == 5
        
        
        single_result = ta_indicators.medium_ad(close, 5)
        batch_values = batch_result['values'][0]
        
        assert_close(
            batch_values,
            single_result,
            rtol=1e-10,
            msg="Batch vs single mismatch"
        )
    
    def test_medium_ad_batch_multiple_periods(self, test_data):
        """Test batch with multiple period values - mirrors batch tests"""
        close = test_data['close'][:100]  
        
        
        batch_result = ta_indicators.medium_ad_batch(close, period_range=(5, 15, 5))
        
        
        assert batch_result['values'].shape == (3, 100)
        assert len(batch_result['periods']) == 3
        assert list(batch_result['periods']) == [5, 10, 15]
        
        
        periods = [5, 10, 15]
        for i, period in enumerate(periods):
            row_data = batch_result['values'][i]
            single_result = ta_indicators.medium_ad(close, period)
            
            assert_close(
                row_data,
                single_result,
                rtol=1e-10,
                msg=f"Period {period} batch mismatch"
            )
    
    def test_medium_ad_batch_edge_cases(self, test_data):
        """Test batch edge cases"""
        close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
        
        
        single_batch = ta_indicators.medium_ad_batch(close, period_range=(3, 3, 1))
        assert single_batch['values'].shape == (1, 10)
        assert single_batch['periods'][0] == 3
        
        
        large_batch = ta_indicators.medium_ad_batch(close, period_range=(3, 5, 10))
        
        assert large_batch['values'].shape == (1, 10)
        assert large_batch['periods'][0] == 3
        
        
        with pytest.raises(ValueError):
            ta_indicators.medium_ad_batch(np.array([]), period_range=(5, 5, 0))
    
    def test_medium_ad_streaming(self, test_data):
        """Test MEDIUM_AD streaming matches batch calculation - mirrors check_medium_ad_streaming"""
        close = test_data['close']
        period = 5
        
        
        batch_result = ta_indicators.medium_ad(close, period)
        
        
        stream = ta_indicators.MediumAdStream(period)
        stream_values = []
        
        for price in close:
            value = stream.update(price)
            stream_values.append(value if value is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        
        assert len(batch_result) == len(stream_values)
        
        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, atol=1e-9,
                        msg=f"MEDIUM_AD streaming mismatch at index {i}")
    
    def test_medium_ad_with_kernel_parameter(self, test_data):
        """Test MEDIUM_AD with explicit kernel parameter"""
        close = test_data['close']
        
        
        result_scalar = ta_indicators.medium_ad(close, 5, kernel='scalar')
        assert len(result_scalar) == len(close)
        
        
        result_auto = ta_indicators.medium_ad(close, 5, kernel=None)
        assert len(result_auto) == len(close)
        
        
        assert_close(
            result_scalar,
            result_auto,
            rtol=1e-10,
            msg="Kernel results should match"
        )
    
    def test_medium_ad_all_nan_input(self):
        """Test MEDIUM_AD with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.medium_ad(all_nan, period=5)
    
    def test_medium_ad_warmup_period(self, test_data):
        """Test MEDIUM_AD warmup period behavior"""
        close = test_data['close'][:20]  
        period = 5
        
        result = ta_indicators.medium_ad(close, period)
        
        
        for i in range(period - 1):
            assert np.isnan(result[i]), f"Expected NaN at index {i} during warmup"
        
        
        assert not np.isnan(result[period - 1]), f"Expected valid value at index {period - 1}"
