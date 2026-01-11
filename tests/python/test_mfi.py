"""
Python binding tests for MFI (Money Flow Index) indicator.
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


class TestMfi:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_mfi_partial_params(self, test_data):
        """Test MFI with default parameters - mirrors check_mfi_partial_params"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        typical_price = (high + low + close) / 3.0
        
        
        result = ta_indicators.mfi(typical_price, volume, period=14)
        assert len(result) == len(typical_price)
    
    def test_mfi_accuracy(self, test_data):
        """Test MFI matches expected values from Rust tests - mirrors check_mfi_accuracy"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        typical_price = (high + low + close) / 3.0
        expected = EXPECTED_OUTPUTS['mfi']
        
        result = ta_indicators.mfi(
            typical_price,
            volume,
            period=expected['default_params']['period']
        )
        
        assert len(result) == len(typical_price)
        
        
        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=0.0,
            atol=1e-1,
            msg="MFI last 5 values mismatch"
        )
        
        
        compare_with_rust('mfi', result, 'hlc3_volume', expected['default_params'])
    
    def test_mfi_default_candles(self, test_data):
        """Test MFI with default parameters - mirrors check_mfi_default_candles"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        typical_price = (high + low + close) / 3.0
        
        result = ta_indicators.mfi(typical_price, volume, period=14)
        assert len(result) == len(typical_price)
    
    def test_mfi_zero_period(self, test_data):
        """Test MFI fails with zero period - mirrors check_mfi_zero_period"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        volume = test_data['volume'][:100]
        typical_price = (high + low + close) / 3.0
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.mfi(typical_price, volume, period=0)
    
    def test_mfi_period_exceeds_length(self):
        """Test MFI fails when period exceeds data length - mirrors check_mfi_period_exceeds_length"""
        input_high = np.array([1.0, 2.0, 3.0])
        input_low = np.array([0.5, 1.5, 2.5])
        input_close = np.array([0.8, 1.8, 2.8])
        input_volume = np.array([100.0, 200.0, 300.0])
        typical_price = (input_high + input_low + input_close) / 3.0
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.mfi(typical_price, input_volume, period=10)
    
    def test_mfi_very_small_dataset(self):
        """Test MFI fails with insufficient data - mirrors check_mfi_very_small_dataset"""
        input_high = np.array([1.0])
        input_low = np.array([0.5])
        input_close = np.array([0.8])
        input_volume = np.array([100.0])
        typical_price = (input_high[0] + input_low[0] + input_close[0]) / 3.0
        typical_price = np.array([typical_price])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.mfi(typical_price, input_volume, period=14)
    
    def test_mfi_empty_input(self):
        """Test MFI fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.mfi(empty, empty, period=14)
    
    def test_mfi_mismatched_lengths(self):
        """Test MFI fails with mismatched array lengths"""
        typical_price = np.array([1.0, 2.0, 3.0])
        volume = np.array([100.0, 200.0])
        
        
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.mfi(typical_price, volume, period=2)
    
    def test_mfi_nan_handling(self, test_data):
        """Test MFI handles NaN values correctly"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        typical_price = (high + low + close) / 3.0
        
        result = ta_indicators.mfi(typical_price, volume, period=14)
        assert len(result) == len(typical_price)
        
        
        assert np.all(np.isnan(result[:13])), "Expected NaN in warmup period"
        
        
        if len(result) > 13:
            
            valid_after_warmup = result[13:]
            assert not np.all(np.isnan(valid_after_warmup)), "Expected valid values after warmup"
            
            
            valid_values = valid_after_warmup[~np.isnan(valid_after_warmup)]
            assert np.all(valid_values >= 0.0), "MFI values should be >= 0"
            assert np.all(valid_values <= 100.0), "MFI values should be <= 100"
    
    def test_mfi_with_nan_input(self):
        """Test MFI with NaN values in input"""
        
        n = 30
        typical_price = np.arange(n, dtype=float)
        typical_price[5] = np.nan  
        volume = np.ones(n) * 1000
        
        result = ta_indicators.mfi(typical_price, volume, period=14)
        assert len(result) == len(typical_price)
        
        
        
        
        period = 14
        assert np.all(np.isnan(result[:period-1])), "Expected NaNs during warmup"
        assert np.all(np.isnan(result[period-1:])), "Expected NaNs after warmup due to seed NaN"
    
    def test_mfi_all_nan_input(self):
        """Test MFI with all NaN values"""
        n = 20
        typical_price = np.full(n, np.nan)
        volume = np.full(n, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.mfi(typical_price, volume, period=14)
    
    def test_mfi_zero_volume(self):
        """Test MFI with zero volume - mirrors check_mfi_zero_volume"""
        
        n = 30
        typical_price = 100.0 + np.arange(n, dtype=float)
        volume = np.zeros(n)
        
        result = ta_indicators.mfi(typical_price, volume, period=14)
        
        
        valid_values = result[~np.isnan(result)]
        assert_close(valid_values, 0.0, rtol=1e-10, 
                    msg="MFI should be 0 with zero volume")
    
    def test_mfi_streaming(self, test_data):
        """Test MFI streaming matches batch calculation"""
        high = test_data['high'][:100]  
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        volume = test_data['volume'][:100]
        typical_price = (high + low + close) / 3.0
        
        
        batch_result = ta_indicators.mfi(typical_price, volume, period=14)
        
        
        stream = ta_indicators.MfiStream(period=14)
        stream_values = []
        
        for i in range(len(typical_price)):
            result = stream.update(typical_price[i], volume[i])
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        
        assert len(batch_result) == len(stream_values)
        
        
        
        assert np.all(np.isnan(stream_values[:14])), "Expected NaN during streaming warmup"
        
        
        
        for i in range(14, len(batch_result)):
            if np.isnan(batch_result[i]) and np.isnan(stream_values[i]):
                continue
            assert_close(batch_result[i], stream_values[i], rtol=1e-9, atol=1e-9,
                        msg=f"MFI streaming mismatch at index {i}")
    
    def test_mfi_batch(self, test_data):
        """Test MFI batch processing - mirrors check_batch_default_row"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        typical_price = (high + low + close) / 3.0
        
        result = ta_indicators.mfi_batch(
            typical_price,
            volume,
            period_range=(14, 14, 0)  
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(typical_price)
        
        
        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['mfi']['last_5_values']
        
        
        assert_close(
            default_row[-5:],
            expected,
            rtol=0.0,
            atol=1e-1,
            msg="MFI batch default row mismatch"
        )
    
    def test_mfi_batch_multiple_periods(self, test_data):
        """Test batch MFI with multiple periods"""
        high = test_data['high'][:100]  
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        volume = test_data['volume'][:100]
        typical_price = (high + low + close) / 3.0
        
        
        result = ta_indicators.mfi_batch(
            typical_price,
            volume,
            period_range=(10, 20, 5)
        )
        
        
        assert result['values'].shape == (3, 100)
        assert np.array_equal(result['periods'], [10, 15, 20])
        
        
        single_result = ta_indicators.mfi(typical_price, volume, period=10)
        assert_close(result['values'][0], single_result, rtol=1e-10,
                    msg="Batch first row should match single calculation")
        
        
        for i, period in enumerate([10, 15, 20]):
            row = result['values'][i]
            
            assert np.all(np.isnan(row[:period-1])), f"Expected NaN warmup for period={period}"
            
            if period < 100:
                assert not np.all(np.isnan(row[period-1:])), f"Expected valid values after warmup for period={period}"
    
    def test_mfi_kernel_selection(self, test_data):
        """Test MFI with different kernel options"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        volume = test_data['volume'][:100]
        typical_price = (high + low + close) / 3.0
        
        
        result_scalar = ta_indicators.mfi(typical_price, volume, period=14, kernel="scalar")
        result_auto = ta_indicators.mfi(typical_price, volume, period=14, kernel=None)
        
        
        assert_close(result_scalar, result_auto, rtol=1e-10,
                    msg="Kernel results should match")
    
    def test_mfi_different_periods(self, test_data):
        """Test MFI with various period values"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        volume = test_data['volume'][:100]
        typical_price = (high + low + close) / 3.0
        
        periods = [7, 14, 21]
        results = []
        
        for period in periods:
            result = ta_indicators.mfi(typical_price, volume, period=period)
            assert len(result) == len(typical_price)
            
            assert np.all(np.isnan(result[:period-1])), f"Expected NaN warmup for period={period}"
            results.append(result)
        
        
        for i in range(len(periods)-1):
            valid_indices = ~(np.isnan(results[i]) | np.isnan(results[i+1]))
            if np.any(valid_indices):
                assert not np.allclose(results[i][valid_indices], results[i+1][valid_indices]),\
                    f"Results should differ for periods {periods[i]} and {periods[i+1]}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
