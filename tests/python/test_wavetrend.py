"""
Python binding tests for WaveTrend indicator.
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

from test_utils import load_test_data, assert_close
from rust_comparison import compare_with_rust


class TestWavetrend:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_wavetrend_partial_params(self, test_data):
        """Test WaveTrend with partial parameters (None values) - mirrors check_wavetrend_partial_params"""
        hlc3 = (test_data['high'] + test_data['low'] + test_data['close']) / 3
        
        
        wt1, wt2, wt_diff = ta_indicators.wavetrend(hlc3, 9, 12, 3, 0.015)
        assert len(wt1) == len(hlc3)
        assert len(wt2) == len(hlc3)
        assert len(wt_diff) == len(hlc3)
    
    def test_wavetrend_accuracy(self, test_data):
        """Test WaveTrend matches expected values from Rust tests - mirrors check_wavetrend_accuracy"""
        hlc3 = (test_data['high'] + test_data['low'] + test_data['close']) / 3
        
        wt1, wt2, wt_diff = ta_indicators.wavetrend(
            hlc3,
            channel_length=9,
            average_length=12,
            ma_length=3,
            factor=0.015
        )
        
        assert len(wt1) == len(hlc3)
        assert len(wt2) == len(hlc3)
        assert len(wt_diff) == len(hlc3)
        
        
        expected_wt1 = [
            -29.02058232514538,
            -28.207769813591664,
            -31.991808642927193,
            -31.9218051759519,
            -44.956245952893866,
        ]
        expected_wt2 = [
            -30.651043230696555,
            -28.686329669808583,
            -29.740053593887932,
            -30.707127877490105,
            -36.2899532572575,
        ]
        
        
        assert_close(
            wt1[-5:], 
            expected_wt1,
            rtol=1e-6,
            msg="WaveTrend WT1 last 5 values mismatch"
        )
        
        assert_close(
            wt2[-5:], 
            expected_wt2,
            rtol=1e-6,
            msg="WaveTrend WT2 last 5 values mismatch"
        )
        
        
        expected_diff = [expected_wt2[i] - expected_wt1[i] for i in range(len(expected_wt1))]
        assert_close(
            wt_diff[-5:],
            expected_diff,
            rtol=1e-6,
            msg="WaveTrend WT_DIFF last 5 values mismatch"
        )
    
    def test_wavetrend_default_candles(self, test_data):
        """Test WaveTrend with default parameters - mirrors check_wavetrend_default_candles"""
        hlc3 = (test_data['high'] + test_data['low'] + test_data['close']) / 3
        
        
        wt1, wt2, wt_diff = ta_indicators.wavetrend(hlc3, 9, 12, 3, 0.015)
        assert len(wt1) == len(hlc3)
        assert len(wt2) == len(hlc3)
        assert len(wt_diff) == len(hlc3)
    
    def test_wavetrend_zero_channel(self):
        """Test WaveTrend fails with zero channel_length - mirrors check_wavetrend_zero_channel"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid channel_length"):
            ta_indicators.wavetrend(input_data, channel_length=0, average_length=12, ma_length=3, factor=0.015)
    
    def test_wavetrend_channel_exceeds_length(self):
        """Test WaveTrend fails when channel_length exceeds data length - mirrors check_wavetrend_channel_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid channel_length"):
            ta_indicators.wavetrend(data_small, channel_length=10, average_length=12, ma_length=3, factor=0.015)
    
    def test_wavetrend_very_small_dataset(self):
        """Test WaveTrend fails with insufficient data - mirrors check_wavetrend_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid|Not enough valid data"):
            ta_indicators.wavetrend(single_point, channel_length=9, average_length=12, ma_length=3, factor=0.015)
    
    def test_wavetrend_empty_input(self):
        """Test WaveTrend fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.wavetrend(empty, channel_length=9, average_length=12, ma_length=3, factor=0.015)
    
    def test_wavetrend_nan_handling(self, test_data):
        """Test WaveTrend handles NaN values correctly - mirrors check_wavetrend_nan_handling"""
        hlc3 = (test_data['high'] + test_data['low'] + test_data['close']) / 3
        
        wt1, wt2, wt_diff = ta_indicators.wavetrend(
            hlc3,
            channel_length=9,
            average_length=12,
            ma_length=3,
            factor=0.015
        )
        
        
        if len(wt1) > 240:
            non_nan_after_warmup = ~np.isnan(wt1[240:])
            assert np.all(non_nan_after_warmup), "Found unexpected NaN after warmup period"
    
    def test_wavetrend_all_nan_input(self):
        """Test WaveTrend with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.wavetrend(all_nan, channel_length=9, average_length=12, ma_length=3, factor=0.015)
    
    def test_wavetrend_kernel_parameter(self, test_data):
        """Test WaveTrend with kernel parameter"""
        hlc3 = (test_data['high'] + test_data['low'] + test_data['close']) / 3
        
        
        wt1_scalar, wt2_scalar, wt_diff_scalar = ta_indicators.wavetrend(
            hlc3, channel_length=9, average_length=12, ma_length=3, factor=0.015, kernel="scalar"
        )
        
        assert len(wt1_scalar) == len(hlc3)
        assert len(wt2_scalar) == len(hlc3)
        assert len(wt_diff_scalar) == len(hlc3)
        
        
        wt1_auto, wt2_auto, wt_diff_auto = ta_indicators.wavetrend(
            hlc3, channel_length=9, average_length=12, ma_length=3, factor=0.015, kernel="auto"
        )
        
        
        assert_close(wt1_scalar, wt1_auto, rtol=1e-10)
        assert_close(wt2_scalar, wt2_auto, rtol=1e-10)
        assert_close(wt_diff_scalar, wt_diff_auto, rtol=1e-10)
    
    def test_wavetrend_batch_single_param_set(self, test_data):
        """Test batch operation with single parameter combination"""
        hlc3 = (test_data['high'] + test_data['low'] + test_data['close']) / 3
        
        result = ta_indicators.wavetrend_batch(
            hlc3,
            channel_length_range=(9, 9, 0),
            average_length_range=(12, 12, 0),
            ma_length_range=(3, 3, 0),
            factor_range=(0.015, 0.015, 0.0)
        )
        
        
        assert result['wt1'].shape[0] == 1
        assert result['wt2'].shape[0] == 1
        assert result['wt_diff'].shape[0] == 1
        assert result['wt1'].shape[1] == len(hlc3)
        
        
        wt1_single, wt2_single, wt_diff_single = ta_indicators.wavetrend(
            hlc3, channel_length=9, average_length=12, ma_length=3, factor=0.015
        )
        
        assert_close(result['wt1'][0], wt1_single, rtol=1e-10)
        assert_close(result['wt2'][0], wt2_single, rtol=1e-10)
        assert_close(result['wt_diff'][0], wt_diff_single, rtol=1e-10)
    
    def test_wavetrend_batch_multiple_params(self, test_data):
        """Test batch operation with multiple parameter combinations"""
        hlc3 = ((test_data['high'] + test_data['low'] + test_data['close']) / 3)[:100]  
        
        result = ta_indicators.wavetrend_batch(
            hlc3,
            channel_length_range=(9, 11, 2),      
            average_length_range=(12, 13, 1),     
            ma_length_range=(3, 3, 0),            
            factor_range=(0.015, 0.020, 0.005)    
        )
        
        
        assert result['wt1'].shape[0] == 8
        assert result['wt2'].shape[0] == 8
        assert result['wt_diff'].shape[0] == 8
        assert result['wt1'].shape[1] == len(hlc3)
        
        
        assert len(result['channel_lengths']) == 8
        assert len(result['average_lengths']) == 8
        assert len(result['ma_lengths']) == 8
        assert len(result['factors']) == 8
    
    def test_wavetrend_streaming(self, test_data):
        """Test WaveTrend streaming functionality"""
        hlc3 = (test_data['high'] + test_data['low'] + test_data['close']) / 3
        
        
        stream = ta_indicators.WavetrendStream(
            channel_length=9,
            average_length=12,
            ma_length=3,
            factor=0.015
        )
        
        
        streaming_wt1 = []
        streaming_wt2 = []
        streaming_wt_diff = []
        
        for value in hlc3:
            result = stream.update(value)
            if result is not None:
                wt1, wt2, wt_diff = result
                streaming_wt1.append(wt1)
                streaming_wt2.append(wt2)
                streaming_wt_diff.append(wt_diff)
            else:
                streaming_wt1.append(np.nan)
                streaming_wt2.append(np.nan)
                streaming_wt_diff.append(np.nan)
        
        
        batch_wt1, batch_wt2, batch_wt_diff = ta_indicators.wavetrend(
            hlc3, channel_length=9, average_length=12, ma_length=3, factor=0.015
        )
        
        
        first_valid = 0
        for i in range(len(batch_wt1)):
            if not np.isnan(batch_wt1[i]):
                first_valid = i
                break
        
        
        for i in range(first_valid, len(hlc3)):
            if not np.isnan(batch_wt1[i]) and not np.isnan(streaming_wt1[i]):
                assert abs(streaming_wt1[i] - batch_wt1[i]) < 1e-9, f"Streaming vs batch mismatch at index {i}"
                assert abs(streaming_wt2[i] - batch_wt2[i]) < 1e-9, f"Streaming vs batch mismatch at index {i}"
                assert abs(streaming_wt_diff[i] - batch_wt_diff[i]) < 1e-9, f"Streaming vs batch mismatch at index {i}"