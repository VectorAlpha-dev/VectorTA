"""
Python binding tests for Damiani Volatmeter indicator.
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


class TestDamianiVolatmeter:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_damiani_partial_params(self, test_data):
        """Test Damiani Volatmeter with partial parameters - mirrors check_damiani_partial_params"""
        close = test_data['close']
        
        
        vol, anti = ta_indicators.damiani(close, 13, 20, 40, 100, 1.4)
        assert len(vol) == len(close)
        assert len(anti) == len(close)
    
    def test_damiani_accuracy(self, test_data):
        """Test Damiani Volatmeter matches expected values from Rust tests - mirrors check_damiani_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['damiani_volatmeter']
        
        vol, anti = ta_indicators.damiani(
            close,
            vis_atr=expected['default_params']['vis_atr'],
            vis_std=expected['default_params']['vis_std'],
            sed_atr=expected['default_params']['sed_atr'],
            sed_std=expected['default_params']['sed_std'],
            threshold=expected['default_params']['threshold']
        )
        
        assert len(vol) == len(close)
        assert len(anti) == len(close)
        
        
        assert_close(
            vol[-5:], 
            expected['vol_last_5_values'],
            rtol=1e-2,  
            msg="Damiani Volatmeter vol last 5 values mismatch"
        )
        assert_close(
            anti[-5:], 
            expected['anti_last_5_values'],
            rtol=1e-2,  
            msg="Damiani Volatmeter anti last 5 values mismatch"
        )
        
        
        

    def test_damiani_accuracy_candles_stream(self, test_data):
        """Candles-based accuracy: stream across OHLC and compare last-5 to Rust references.

        Uses the feed-stream API (accepts H/L/C) to mirror Rust tests that use candles.
        """
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['damiani_volatmeter']

        stream = ta_indicators.DamianiVolatmeterFeedStream(
            vis_atr=expected['default_params']['vis_atr'],
            vis_std=expected['default_params']['vis_std'],
            sed_atr=expected['default_params']['sed_atr'],
            sed_std=expected['default_params']['sed_std'],
            threshold=expected['default_params']['threshold']
        )

        vols = []
        antis = []
        for h, l, c in zip(high, low, close):
            out = stream.update(h, l, c)
            if out is None:
                vols.append(np.nan)
                antis.append(np.nan)
            else:
                v, a = out
                vols.append(v)
                antis.append(a)

        vols = np.array(vols)
        antis = np.array(antis)

        
        assert_close(vols[-5:], expected['rust_vol_last_5_values'], rtol=1e-2,
                     msg="Damiani Volatmeter (candles) vol last 5 mismatch")
        assert_close(antis[-5:], expected['rust_anti_last_5_values'], rtol=1e-2,
                     msg="Damiani Volatmeter (candles) anti last 5 mismatch")
    
    def test_damiani_default_candles(self, test_data):
        """Test Damiani Volatmeter with default parameters - mirrors check_damiani_input_with_default_candles"""
        close = test_data['close']
        
        
        vol, anti = ta_indicators.damiani(close, 13, 20, 40, 100, 1.4)
        assert len(vol) == len(close)
        assert len(anti) == len(close)
    
    def test_damiani_zero_period(self):
        """Test Damiani Volatmeter fails with zero period - mirrors check_damiani_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0] * 25)  
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.damiani(input_data, vis_atr=0, vis_std=20, sed_atr=40, sed_std=100, threshold=1.4)
    
    def test_damiani_period_exceeds_length(self):
        """Test Damiani Volatmeter fails when period exceeds data length - mirrors check_damiani_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.damiani(data_small, vis_atr=99999, vis_std=20, sed_atr=40, sed_std=100, threshold=1.4)
    
    def test_damiani_very_small_dataset(self):
        """Test Damiani Volatmeter fails with insufficient data - mirrors check_damiani_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.damiani(single_point, vis_atr=9, vis_std=9, sed_atr=9, sed_std=9, threshold=1.4)
    
    def test_damiani_empty_input(self):
        """Test Damiani Volatmeter fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.damiani(empty, vis_atr=13, vis_std=20, sed_atr=40, sed_std=100, threshold=1.4)
    
    def test_damiani_all_nan_input(self):
        """Test Damiani Volatmeter fails with all NaN input"""
        all_nan = np.array([np.nan, np.nan, np.nan, np.nan, np.nan] * 30)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.damiani(all_nan, vis_atr=13, vis_std=20, sed_atr=40, sed_std=100, threshold=1.4)
    
    def test_damiani_nan_handling(self, test_data):
        """Test Damiani Volatmeter handles NaN values correctly"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['damiani_volatmeter']
        
        vol, anti = ta_indicators.damiani(
            close, 
            vis_atr=expected['default_params']['vis_atr'], 
            vis_std=expected['default_params']['vis_std'], 
            sed_atr=expected['default_params']['sed_atr'], 
            sed_std=expected['default_params']['sed_std'], 
            threshold=expected['default_params']['threshold']
        )
        assert len(vol) == len(close)
        assert len(anti) == len(close)
        
        
        warmup = expected['warmup_period']
        
        
        first_non_nan_vol = next((i for i, v in enumerate(vol) if not np.isnan(v)), -1)
        assert first_non_nan_vol == warmup - 1, f"Vol first non-NaN at index {first_non_nan_vol}, expected {warmup - 1}"
        
        
        first_non_nan_anti = next((i for i, v in enumerate(anti) if not np.isnan(v)), -1)
        assert first_non_nan_anti == warmup - 1, f"Anti first non-NaN at index {first_non_nan_anti}, expected {warmup - 1}"
        
        
        assert np.all(np.isnan(vol[:warmup-1])), f"Expected NaN in vol warmup period [0:{warmup-1})"
        assert np.all(np.isnan(anti[:warmup-1])), f"Expected NaN in anti warmup period [0:{warmup-1})"
        
        
        if len(vol) > warmup + 100:
            assert not np.any(np.isnan(vol[warmup+100:])), "Found unexpected NaN in vol after extended warmup"
            assert not np.any(np.isnan(anti[warmup+100:])), "Found unexpected NaN in anti after extended warmup"
    
    def test_damiani_streaming(self, test_data):
        """Test Damiani Volatmeter streaming matches batch calculation - mirrors check_damiani_streaming"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        
        stream = ta_indicators.DamianiVolatmeterStream(
            high, low, close,
            vis_atr=13, vis_std=20, sed_atr=40, sed_std=100, threshold=1.4
        )
        
        
        result = stream.update()
        
        
        assert result is None, "Stream update should return None on success"
        
        
        vol_expected, anti_expected = ta_indicators.damiani(close, 13, 20, 40, 100, 1.4)
        
        
        
        
    
    def test_damiani_batch(self, test_data):
        """Test Damiani Volatmeter batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        
        result = ta_indicators.damiani_batch(
            close,
            vis_atr_range=(13, 40, 1),  
            vis_std_range=(20, 40, 1),  
            sed_atr_range=(40, 40, 0),  
            sed_std_range=(100, 100, 0),  
            threshold_range=(1.4, 1.4, 0.0)  
        )
        
        assert 'vol' in result
        assert 'anti' in result
        assert 'vis_atr' in result
        assert 'vis_std' in result
        assert 'sed_atr' in result
        assert 'sed_std' in result
        assert 'threshold' in result
        
        
        vol_values = result['vol']
        anti_values = result['anti']
        assert vol_values.shape[1] == len(close)  
        assert anti_values.shape[1] == len(close)  
        
        
        
        
        
        
        
        
        expected_rows = 28 * 21  
        assert vol_values.shape[0] == expected_rows
        assert anti_values.shape[0] == expected_rows
    
    def test_damiani_invalid_parameters(self):
        """Test Damiani Volatmeter fails with invalid parameters - mirrors check_damiani_invalid_periods"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        
        with pytest.raises((ValueError, OverflowError)):
            ta_indicators.damiani(data, vis_atr=-1, vis_std=20, sed_atr=40, sed_std=100, threshold=1.4)
        
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.damiani(data, vis_atr=13, vis_std=0, sed_atr=40, sed_std=100, threshold=1.4)
        
        
        with pytest.raises((ValueError, OverflowError)):
            ta_indicators.damiani(data, vis_atr=13, vis_std=20, sed_atr=-5, sed_std=100, threshold=1.4)
        
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.damiani(data, vis_atr=13, vis_std=20, sed_atr=40, sed_std=0, threshold=1.4)
        
        
        
        vol, anti = ta_indicators.damiani(data, vis_atr=2, vis_std=2, sed_atr=2, sed_std=2, threshold=np.nan)
        assert np.all(np.isnan(anti[2:])), "Expected NaN anti values with NaN threshold"
        
        
        vol, anti = ta_indicators.damiani(data, vis_atr=2, vis_std=2, sed_atr=2, sed_std=2, threshold=-1.0)
        assert len(vol) == len(data)
    
    def test_damiani_warmup_verification(self, test_data):
        """Test Damiani Volatmeter warmup period calculation is correct"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['damiani_volatmeter']
        
        
        vol, anti = ta_indicators.damiani(
            close, 
            vis_atr=expected['default_params']['vis_atr'], 
            vis_std=expected['default_params']['vis_std'], 
            sed_atr=expected['default_params']['sed_atr'], 
            sed_std=expected['default_params']['sed_std'], 
            threshold=expected['default_params']['threshold']
        )
        
        
        calculated_warmup = max(13, 20, 40, 100, 3) + 1  
        assert calculated_warmup == expected['warmup_period']
        
        
        assert np.all(np.isnan(vol[:calculated_warmup-1]))
        assert np.all(np.isnan(anti[:calculated_warmup-1]))
        
        
        assert not np.isnan(vol[calculated_warmup-1])
        assert not np.isnan(anti[calculated_warmup-1])
        
        
        vol2, anti2 = ta_indicators.damiani(
            close, vis_atr=5, vis_std=10, sed_atr=15, sed_std=20, threshold=1.0
        )
        calculated_warmup2 = max(5, 10, 15, 20, 3) + 1  
        
        
        assert np.all(np.isnan(vol2[:calculated_warmup2-1]))
        assert not np.isnan(vol2[calculated_warmup2-1])
    
    def test_damiani_feed_stream(self, test_data):
        """Test Damiani Volatmeter feed-based streaming API"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        
        stream1 = ta_indicators.DamianiVolatmeterFeedStream(
            vis_atr=13, vis_std=20, sed_atr=40, sed_std=100, threshold=1.4
        )
        
        stream2 = ta_indicators.DamianiVolatmeterFeedStream(
            vis_atr=10, vis_std=15, sed_atr=30, sed_std=50, threshold=1.0
        )
        
        
        stream1_vol = []
        stream1_anti = []
        stream2_vol = []
        stream2_anti = []
        
        for i in range(min(200, len(close))):  
            
            result1 = stream1.update(high[i], low[i], close[i])
            if result1 is None:
                stream1_vol.append(np.nan)
                stream1_anti.append(np.nan)
            else:
                vol, anti = result1
                stream1_vol.append(vol)
                stream1_anti.append(anti)
            
            
            result2 = stream2.update(high[i], low[i], close[i])
            if result2 is None:
                stream2_vol.append(np.nan)
                stream2_anti.append(np.nan)
            else:
                vol, anti = result2
                stream2_vol.append(vol)
                stream2_anti.append(anti)
        
        stream1_vol = np.array(stream1_vol)
        stream1_anti = np.array(stream1_anti)
        stream2_vol = np.array(stream2_vol)
        stream2_anti = np.array(stream2_anti)
        
        
        assert len(stream1_vol) == 200
        assert len(stream1_anti) == 200
        assert len(stream2_vol) == 200
        assert len(stream2_anti) == 200
        
        
        
        
        first_valid1 = 99
        assert np.all(np.isnan(stream1_vol[:first_valid1])), "Stream1 vol should be NaN during warmup"
        assert np.all(np.isnan(stream1_anti[:first_valid1])), "Stream1 anti should be NaN during warmup"
        assert not np.isnan(stream1_vol[first_valid1]), f"Stream1 vol should not be NaN at index {first_valid1}"
        assert not np.isnan(stream1_anti[first_valid1]), f"Stream1 anti should not be NaN at index {first_valid1}"
        
        
        
        first_valid2 = 49
        assert np.all(np.isnan(stream2_vol[:first_valid2])), "Stream2 vol should be NaN during warmup"
        assert np.all(np.isnan(stream2_anti[:first_valid2])), "Stream2 anti should be NaN during warmup"
        assert not np.isnan(stream2_vol[first_valid2]), f"Stream2 vol should not be NaN at index {first_valid2}"
        assert not np.isnan(stream2_anti[first_valid2]), f"Stream2 anti should not be NaN at index {first_valid2}"
        
        
        
        assert abs(stream1_vol[150] - stream2_vol[150]) > 0.01, "Streams with different params should produce different vol"
        assert abs(stream1_anti[150] - stream2_anti[150]) > 0.01, "Streams with different params should produce different anti"
        
        
        stream_nan = ta_indicators.DamianiVolatmeterFeedStream(
            vis_atr=5, vis_std=10, sed_atr=15, sed_std=20, threshold=1.0
        )
        
        
        for i in range(30):
            stream_nan.update(high[i], low[i], close[i])
        
        
        result_nan = stream_nan.update(np.nan, np.nan, np.nan)
        
        assert result_nan is not None, "Should still compute after warmup even with NaN input"
    
    def test_damiani_batch_accuracy(self, test_data):
        """Test Damiani Volatmeter batch values match single calculations"""
        close = test_data['close'][:100]
        
        
        vol_single, anti_single = ta_indicators.damiani(close, 13, 20, 40, 100, 1.4)
        
        
        batch_result = ta_indicators.damiani_batch(
            close,
            vis_atr_range=(13, 13, 0),
            vis_std_range=(20, 20, 0),
            sed_atr_range=(40, 40, 0),
            sed_std_range=(100, 100, 0),
            threshold_range=(1.4, 1.4, 0.0)
        )
        
        assert batch_result['vol'].shape == (1, 100)
        assert batch_result['anti'].shape == (1, 100)
        
        
        vol_batch = batch_result['vol'][0]
        anti_batch = batch_result['anti'][0]
        
        
        vol_mask = ~np.isnan(vol_single) & ~np.isnan(vol_batch)
        anti_mask = ~np.isnan(anti_single) & ~np.isnan(anti_batch)
        
        if np.any(vol_mask):
            assert_close(vol_batch[vol_mask], vol_single[vol_mask], 1e-10, 1e-10)
        if np.any(anti_mask):
            assert_close(anti_batch[anti_mask], anti_single[anti_mask], 1e-10, 1e-10)
    
    def test_damiani_partial_nan_handling(self, test_data):
        """Test Damiani Volatmeter with partial NaN data"""
        close = test_data['close'][:200].copy()
        
        
        close[50:60] = np.nan  
        close[150] = np.nan    
        
        
        vol, anti = ta_indicators.damiani(close, 13, 20, 40, 100, 1.4)
        
        assert len(vol) == len(close)
        assert len(anti) == len(close)
        
        
        
        
        for i in range(50, 60):
            if i < len(vol):
                assert np.isnan(vol[i]) or np.isnan(anti[i]), f"Expected NaN at injected position {i}"
        
        
        
        nan_window = 100  
        affected_start = max(0, 150 - nan_window)
        affected_end = min(len(vol), 150 + nan_window)
        
        
        
        
    
    def test_damiani_different_thresholds(self, test_data):
        """Test Damiani Volatmeter with different threshold values"""
        close = test_data['close']
        
        
        thresholds = [0.5, 1.0, 1.4, 2.0]
        results = []
        
        for thresh in thresholds:
            vol, anti = ta_indicators.damiani(close, 13, 20, 40, 100, thresh)
            results.append((vol, anti))
        
        
        for i in range(len(thresholds) - 1):
            anti1 = results[i][1]
            anti2 = results[i + 1][1]
            
            valid_idx = ~(np.isnan(anti1) | np.isnan(anti2))
            if np.any(valid_idx):
                
                diff = anti2[valid_idx] - anti1[valid_idx]
                expected_diff = thresholds[i + 1] - thresholds[i]
                assert_close(diff, expected_diff, rtol=1e-10, atol=1e-10,
                           msg=f"Anti values should differ by threshold difference")
