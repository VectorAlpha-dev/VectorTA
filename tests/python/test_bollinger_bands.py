"""
Python binding tests for Bollinger Bands indicator.
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


class TestBollingerBands:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_bollinger_bands_partial_params(self, test_data):
        """Test Bollinger Bands with partial parameters - mirrors check_bb_partial_params"""
        close = test_data['close']
        
        
        upper, middle, lower = ta_indicators.bollinger_bands(
            close, 
            period=22,
            matype="sma"
        )
        
        assert len(upper) == len(close)
        assert len(middle) == len(close)
        assert len(lower) == len(close)
    
    def test_bollinger_bands_accuracy(self, test_data):
        """Test Bollinger Bands matches expected values from Rust tests - mirrors check_bb_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS.get('bollinger_bands', {})
        
        
        upper, middle, lower = ta_indicators.bollinger_bands(close)
        
        assert len(upper) == len(close)
        assert len(middle) == len(close) 
        assert len(lower) == len(close)
        
        
        expected_middle = [
            59403.199999999975,
            59423.24999999998,
            59370.49999999998,
            59371.39999999998,
            59351.299999999974,
        ]
        expected_lower = [
            58299.51497247008,
            58351.47038179873,
            58332.65135978715,
            58334.33194052157,
            58275.767369163135,
        ]
        expected_upper = [
            60506.88502752987,
            60495.029618201224,
            60408.348640212804,
            60408.468059478386,
            60426.83263083681,
        ]
        
        
        assert_close(
            upper[-5:], 
            expected_upper,
            rtol=1e-4,
            msg="Bollinger Bands upper band last 5 values mismatch"
        )
        assert_close(
            middle[-5:], 
            expected_middle,
            rtol=1e-4,
            msg="Bollinger Bands middle band last 5 values mismatch"
        )
        assert_close(
            lower[-5:], 
            expected_lower,
            rtol=1e-4,
            msg="Bollinger Bands lower band last 5 values mismatch"
        )
    
    def test_bollinger_bands_default_params(self, test_data):
        """Test Bollinger Bands with default parameters - mirrors check_bb_default_candles"""
        close = test_data['close']
        
        
        upper, middle, lower = ta_indicators.bollinger_bands(close)
        
        assert len(upper) == len(close)
        assert len(middle) == len(close)
        assert len(lower) == len(close)
    
    def test_bollinger_bands_zero_period(self):
        """Test Bollinger Bands fails with zero period - mirrors check_bb_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.bollinger_bands(input_data, period=0)
    
    def test_bollinger_bands_period_exceeds_length(self):
        """Test Bollinger Bands fails when period exceeds data length - mirrors check_bb_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.bollinger_bands(data_small, period=10)
    
    def test_bollinger_bands_very_small_dataset(self):
        """Test Bollinger Bands fails with insufficient data - mirrors check_bb_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.bollinger_bands(single_point)
    
    def test_bollinger_bands_empty_input(self):
        """Test Bollinger Bands fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data provided"):
            ta_indicators.bollinger_bands(empty)
    
    def test_bollinger_bands_reinput(self, test_data):
        """Test Bollinger Bands applied twice (re-input) - mirrors check_bb_reinput"""
        close = test_data['close']
        
        
        upper1, middle1, lower1 = ta_indicators.bollinger_bands(close, period=20)
        assert len(middle1) == len(close)
        
        
        upper2, middle2, lower2 = ta_indicators.bollinger_bands(middle1, period=10)
        assert len(middle2) == len(middle1)
    
    def test_bollinger_bands_nan_handling(self, test_data):
        """Test Bollinger Bands handles NaN values correctly - mirrors check_bb_nan_handling"""
        close = test_data['close']
        
        upper, middle, lower = ta_indicators.bollinger_bands(close, period=20)
        assert len(upper) == len(close)
        
        
        if len(upper) > 240:
            assert not np.any(np.isnan(upper[240:])), "Found unexpected NaN in upper band after warmup period"
            assert not np.any(np.isnan(middle[240:])), "Found unexpected NaN in middle band after warmup period"
            assert not np.any(np.isnan(lower[240:])), "Found unexpected NaN in lower band after warmup period"
        
        
        assert np.all(np.isnan(upper[:19])), "Expected NaN in upper band warmup period"
        assert np.all(np.isnan(middle[:19])), "Expected NaN in middle band warmup period"
        assert np.all(np.isnan(lower[:19])), "Expected NaN in lower band warmup period"
    
    def test_bollinger_bands_streaming(self, test_data):
        """Test Bollinger Bands streaming matches batch calculation - mirrors check_bb_streaming"""
        close = test_data['close']
        period = 20
        devup = 2.0
        devdn = 2.0
        
        
        batch_upper, batch_middle, batch_lower = ta_indicators.bollinger_bands(
            close, 
            period=period, 
            devup=devup, 
            devdn=devdn
        )
        
        
        stream = ta_indicators.BollingerBandsStream(period=period, devup=devup, devdn=devdn)
        stream_upper = []
        stream_middle = []
        stream_lower = []
        
        for price in close:
            result = stream.update(price)
            if result is not None:
                up, mid, low = result
                stream_upper.append(up)
                stream_middle.append(mid)
                stream_lower.append(low)
            else:
                stream_upper.append(np.nan)
                stream_middle.append(np.nan)
                stream_lower.append(np.nan)
        
        stream_upper = np.array(stream_upper)
        stream_middle = np.array(stream_middle)
        stream_lower = np.array(stream_lower)
        
        
        assert len(batch_upper) == len(stream_upper)
        
        
        for i in range(len(batch_upper)):
            if np.isnan(batch_upper[i]) and np.isnan(stream_upper[i]):
                continue
            assert_close(batch_upper[i], stream_upper[i], rtol=1e-6, atol=1e-6, 
                        msg=f"BB streaming upper mismatch at index {i}")
            assert_close(batch_middle[i], stream_middle[i], rtol=1e-6, atol=1e-6, 
                        msg=f"BB streaming middle mismatch at index {i}")
            assert_close(batch_lower[i], stream_lower[i], rtol=1e-6, atol=1e-6, 
                        msg=f"BB streaming lower mismatch at index {i}")
    
    def test_bollinger_bands_batch(self, test_data):
        """Test Bollinger Bands batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        result = ta_indicators.bollinger_bands_batch(
            close,
            period_range=(20, 20, 0),  
            devup_range=(2.0, 2.0, 0.0),  
            devdn_range=(2.0, 2.0, 0.0),  
            matype="sma",  
            devtype_range=(0, 0, 0)  
        )
        
        assert 'upper' in result
        assert 'middle' in result
        assert 'lower' in result
        assert 'periods' in result
        assert 'devups' in result
        assert 'devdns' in result
        assert 'matypes' in result
        assert 'devtypes' in result
        
        
        assert result['upper'].shape[0] == 1
        assert result['upper'].shape[1] == len(close)
        assert result['middle'].shape[0] == 1
        assert result['lower'].shape[0] == 1
        
        
        upper_row = result['upper'][0]
        middle_row = result['middle'][0]
        lower_row = result['lower'][0]
        
        
        expected_middle = [
            59403.199999999975,
            59423.24999999998,
            59370.49999999998,
            59371.39999999998,
            59351.299999999974,
        ]
        
        
        assert_close(
            middle_row[-5:],
            expected_middle,
            rtol=1e-4,
            msg="Bollinger Bands batch middle band mismatch"
        )
    
    def test_bollinger_bands_all_nan_input(self):
        """Test Bollinger Bands with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.bollinger_bands(all_nan)
    
    def test_bollinger_bands_different_matypes(self, test_data):
        """Test Bollinger Bands with different moving average types"""
        close = test_data['close'][:100]  
        
        
        upper_ema, middle_ema, lower_ema = ta_indicators.bollinger_bands(
            close, 
            period=20,
            matype="ema"
        )
        assert len(upper_ema) == len(close)
        
        
        upper_sma, middle_sma, lower_sma = ta_indicators.bollinger_bands(
            close, 
            period=20,
            matype="sma"
        )
        assert len(upper_sma) == len(close)
        
        
        assert not np.allclose(middle_ema[20:], middle_sma[20:], rtol=1e-8)
    
    def test_bollinger_bands_different_devtypes(self, test_data):
        """Test Bollinger Bands with different deviation types"""
        close = test_data['close'][:100]  
        
        
        upper0, middle0, lower0 = ta_indicators.bollinger_bands(
            close, 
            period=20,
            devtype=0
        )
        
        
        upper1, middle1, lower1 = ta_indicators.bollinger_bands(
            close, 
            period=20,
            devtype=1
        )
        
        
        upper2, middle2, lower2 = ta_indicators.bollinger_bands(
            close, 
            period=20,
            devtype=2
        )
        
        
        assert np.allclose(middle0[20:], middle1[20:], rtol=1e-8)
        assert np.allclose(middle0[20:], middle2[20:], rtol=1e-8)
        
        
        assert not np.allclose(upper0[20:], upper1[20:], rtol=1e-8)
        assert not np.allclose(upper0[20:], upper2[20:], rtol=1e-8)
    
    def test_bollinger_bands_batch_multiple_params(self, test_data):
        """Test Bollinger Bands batch with multiple parameter combinations"""
        close = test_data['close'][:500]  
        
        result = ta_indicators.bollinger_bands_batch(
            close,
            period_range=(10, 30, 10),  
            devup_range=(1.0, 3.0, 1.0),  
            devdn_range=(2.0, 2.0, 0.0),  
            matype="sma",
            devtype_range=(0, 0, 0)
        )
        
        
        assert result['upper'].shape[0] == 9
        assert result['middle'].shape[0] == 9
        assert result['lower'].shape[0] == 9
        
        
        assert len(result['periods']) == 9
        expected_periods = [10, 10, 10, 20, 20, 20, 30, 30, 30]
        assert np.allclose(result['periods'], expected_periods)
        
        
        expected_devups = [1.0, 2.0, 3.0] * 3
        assert np.allclose(result['devups'], expected_devups)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])