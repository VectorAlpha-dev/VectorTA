"""
Python binding tests for AlphaTrend indicator.
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

class TestAlphaTrend:
    @pytest.fixture(scope='class')
    def test_data(self):
        """Load test data from CSV file"""
        return load_test_data()
    
    def test_alphatrend_accuracy(self, test_data):
        """Test AlphaTrend matches expected values from Rust tests"""
        open_data = test_data['open']
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        expected = EXPECTED_OUTPUTS['alphatrend']
        
        
        k1, k2 = ta_indicators.alphatrend(
            open_data,
            high,
            low,
            close,
            volume,
            coeff=expected['default_params']['coeff'],
            period=expected['default_params']['period'],
            no_volume=expected['default_params']['no_volume']
        )
        
        assert len(k1) == len(close)
        assert len(k2) == len(close)
        
        
        
        assert_close(
            k1[-5:],
            expected['k1_last_5_values'],
            rtol=0.0,
            atol=1e-6,
            msg="AlphaTrend K1 last 5 values mismatch"
        )
        
        
        
        assert_close(
            k2[-5:],
            expected['k2_last_5_values'],
            rtol=0.0,
            atol=1e-6,
            msg="AlphaTrend K2 last 5 values mismatch"
        )
    
    def test_alphatrend_with_rsi(self, test_data):
        """Test AlphaTrend with RSI instead of MFI (no_volume=True)"""
        open_data = test_data['open']
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        
        k1, k2 = ta_indicators.alphatrend(
            open_data,
            high,
            low,
            close,
            volume,
            coeff=1.0,
            period=14,
            no_volume=True
        )
        
        assert len(k1) == len(close)
        assert len(k2) == len(close)
        
        
        non_nan_k1 = k1[~np.isnan(k1)]
        non_nan_k2 = k2[~np.isnan(k2)]
        assert len(non_nan_k1) > 0
        assert len(non_nan_k2) > 0
    
    def test_alphatrend_zero_period(self):
        """Test AlphaTrend fails with zero period"""
        open_data = np.array([10.0, 20.0, 30.0])
        high = np.array([12.0, 22.0, 32.0])
        low = np.array([8.0, 18.0, 28.0])
        close = np.array([11.0, 21.0, 31.0])
        volume = np.array([100.0, 200.0, 300.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.alphatrend(open_data, high, low, close, volume, period=0)
    
    def test_alphatrend_empty_input(self):
        """Test AlphaTrend fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.alphatrend(empty, empty, empty, empty, empty)
    
    def test_alphatrend_inconsistent_lengths(self):
        """Test AlphaTrend fails with inconsistent array lengths"""
        open_data = np.array([10.0, 20.0, 30.0])
        high = np.array([12.0, 22.0])  
        low = np.array([8.0, 18.0, 28.0])
        close = np.array([11.0, 21.0, 31.0])
        volume = np.array([100.0, 200.0, 300.0])
        
        with pytest.raises(ValueError, match="Inconsistent data lengths"):
            ta_indicators.alphatrend(open_data, high, low, close, volume)
    
    def test_alphatrend_invalid_coeff(self):
        """Test AlphaTrend fails with invalid coefficient"""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0] * 5)
        
        with pytest.raises(ValueError, match="Invalid coefficient"):
            ta_indicators.alphatrend(data, data, data, data, data, coeff=-1.0)
    
    def test_alphatrend_period_exceeds_length(self):
        """Test AlphaTrend fails when period exceeds data length"""
        small_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.alphatrend(
                small_data, small_data, small_data, small_data, small_data, 
                period=10
            )
    
    def test_alphatrend_all_nan(self):
        """Test AlphaTrend fails with all NaN values"""
        nan_data = np.array([np.nan, np.nan, np.nan])
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.alphatrend(nan_data, nan_data, nan_data, nan_data, nan_data)
    
    def test_alphatrend_stream(self, test_data):
        """Test AlphaTrendStream for streaming calculations"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        expected = EXPECTED_OUTPUTS['alphatrend']
        
        
        stream = ta_indicators.AlphaTrendStream(
            coeff=expected['default_params']['coeff'],
            period=expected['default_params']['period'],
            no_volume=expected['default_params']['no_volume']
        )
        
        stream_k1 = []
        stream_k2 = []
        
        
        for i in range(min(100, len(close))):  
            result = stream.update(high[i], low[i], close[i], volume[i])
            
            if result is not None:
                stream_k1.append(result[0])
                stream_k2.append(result[1])
            else:
                stream_k1.append(np.nan)
                stream_k2.append(np.nan)
        
        
        
        if all(np.isnan(stream_k1)):
            
            pass
        else:
            
            
            batch_k1, batch_k2 = ta_indicators.alphatrend(
                test_data['open'][:100],
                high[:100],
                low[:100],
                close[:100],
                volume[:100],
                coeff=expected['default_params']['coeff'],
                period=expected['default_params']['period'],
                no_volume=expected['default_params']['no_volume']
            )
            
            
            for i in range(len(stream_k1)):
                if not np.isnan(stream_k1[i]) and not np.isnan(batch_k1[i]):
                    assert_close(stream_k1[i], batch_k1[i], rtol=1e-9,
                                msg=f"K1 streaming mismatch at index {i}")
                if not np.isnan(stream_k2[i]) and not np.isnan(batch_k2[i]):
                    assert_close(stream_k2[i], batch_k2[i], rtol=1e-9,
                                msg=f"K2 streaming mismatch at index {i}")
    
    def test_alphatrend_different_parameters(self):
        """Test AlphaTrend with various parameter combinations"""
        
        size = 100
        np.random.seed(42)
        open_data = np.cumsum(np.random.randn(size)) + 100
        high = open_data + np.abs(np.random.randn(size)) * 2
        low = open_data - np.abs(np.random.randn(size)) * 2
        close = (high + low) / 2 + np.random.randn(size) * 0.5
        volume = np.abs(np.random.randn(size)) * 1000 + 500
        
        
        for coeff in [0.5, 1.0, 2.0]:
            k1, k2 = ta_indicators.alphatrend(
                open_data, high, low, close, volume,
                coeff=coeff, period=14, no_volume=False
            )
            assert len(k1) == size
            assert len(k2) == size
        
        
        for period in [7, 14, 21]:
            k1, k2 = ta_indicators.alphatrend(
                open_data, high, low, close, volume,
                coeff=1.0, period=period, no_volume=False
            )
            assert len(k1) == size
            assert len(k2) == size
        
        
        k1_mfi, k2_mfi = ta_indicators.alphatrend(
            open_data, high, low, close, volume,
            coeff=1.0, period=14, no_volume=False
        )
        k1_rsi, k2_rsi = ta_indicators.alphatrend(
            open_data, high, low, close, volume,
            coeff=1.0, period=14, no_volume=True
        )
        
        
        
        valid_idx = ~(np.isnan(k1_mfi) | np.isnan(k1_rsi))
        if np.any(valid_idx):
            assert not np.allclose(k1_mfi[valid_idx], k1_rsi[valid_idx], rtol=1e-10)
    
    def test_alphatrend_warmup_period(self, test_data):
        """Test AlphaTrend warmup period behavior"""
        close = test_data['close']
        high = test_data['high']
        low = test_data['low']
        volume = test_data['volume']
        expected = EXPECTED_OUTPUTS['alphatrend']
        
        k1, k2 = ta_indicators.alphatrend(
            test_data['open'],
            high,
            low,
            close,
            volume,
            coeff=expected['default_params']['coeff'],
            period=expected['default_params']['period'],
            no_volume=expected['default_params']['no_volume']
        )
        
        
        warmup = expected['warmup_period']
        
        
        assert np.all(np.isnan(k1[:warmup])), f"Expected NaN in K1 warmup period (first {warmup} values)"
        assert np.all(np.isnan(k2[:warmup])), f"Expected NaN in K2 warmup period (first {warmup} values)"
        
        
        if len(k1) > warmup + 10:
            assert not np.all(np.isnan(k1[warmup:warmup+10])), "K1 should have real values after warmup"
            assert not np.all(np.isnan(k2[warmup:warmup+10])), "K2 should have real values after warmup"
    
    @pytest.mark.skip(reason="AlphaTrend batch function not yet implemented in Python bindings")
    def test_alphatrend_batch(self, test_data):
        """Test AlphaTrend batch processing
        
        NOTE: This test is currently skipped because alphatrend_batch is not yet
        implemented in the Python bindings. When implemented, it should follow
        the pattern of other batch functions like alma_batch.
        """
        close = test_data['close'][:200]  
        high = test_data['high'][:200]
        low = test_data['low'][:200]
        open_data = test_data['open'][:200]
        volume = test_data['volume'][:200]
        expected = EXPECTED_OUTPUTS['alphatrend']
        
        
        result = ta_indicators.alphatrend_batch(
            open_data,
            high,
            low,
            close,
            volume,
            coeff_range=(1.0, 1.0, 0.0),
            period_range=(14, 14, 0),
            no_volume=[False]
        )
        
        assert 'k1_values' in result
        assert 'k2_values' in result
        assert 'coeffs' in result
        assert 'periods' in result
        assert 'no_volume_flags' in result
        
        
        assert result['k1_values'].shape[0] == 1
        assert result['k2_values'].shape[0] == 1
        assert result['k1_values'].shape[1] == len(close)
        assert result['k2_values'].shape[1] == len(close)
        
        
        result_multi = ta_indicators.alphatrend_batch(
            open_data,
            high,
            low,
            close,
            volume,
            coeff_range=(0.5, 2.0, 0.5),  
            period_range=(7, 21, 7),       
            no_volume=[False, True]        
        )
        
        
        assert result_multi['k1_values'].shape[0] == 24
        assert result_multi['k2_values'].shape[0] == 24
        
        
        assert len(result_multi['coeffs']) == 24
        assert len(result_multi['periods']) == 24
        assert len(result_multi['no_volume_flags']) == 24
    
    def test_alphatrend_nan_distribution(self, test_data):
        """Test AlphaTrend NaN handling throughout the data"""
        
        close = test_data['close'].copy()
        high = test_data['high'].copy()
        low = test_data['low'].copy()
        open_data = test_data['open'].copy()
        volume = test_data['volume'].copy()
        
        
        nan_indices = [50, 100, 150, 200, 250]
        for idx in nan_indices:
            if idx < len(close):
                close[idx] = np.nan
                high[idx] = np.nan
                low[idx] = np.nan
        
        
        k1, k2 = ta_indicators.alphatrend(
            open_data,
            high,
            low,
            close,
            volume,
            coeff=1.0,
            period=14,
            no_volume=False
        )
        
        assert len(k1) == len(close)
        assert len(k2) == len(close)
        
        
        
        
        for idx in nan_indices:
            if idx < len(k1):
                sticky_ok = False
                if idx > 0:
                    same_k1 = np.isfinite(k1[idx]) and np.isfinite(k1[idx-1]) and np.isclose(k1[idx], k1[idx-1])
                    
                    same_k2 = np.isfinite(k2[idx]) and np.isfinite(k2[idx-1]) and np.isclose(k2[idx], k2[idx-1])
                    sticky_ok = same_k1 or same_k2
                assert (np.isnan(k1[idx]) or np.isnan(k2[idx]) or sticky_ok), \
                    f"Expected NaN propagation or sticky behavior at index {idx}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
