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
    # If not in virtual environment, try to import from installed location
    try:
        import my_project as ta_indicators
    except ImportError:
        pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close
from rust_comparison import compare_with_rust


class TestDamianiVolatmeter:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_damiani_partial_params(self, test_data):
        """Test Damiani Volatmeter with partial parameters - mirrors check_damiani_partial_params"""
        close = test_data['close']
        
        # Test with all default params
        vol, anti = ta_indicators.damiani_volatmeter(close, 13, 20, 40, 100, 1.4)
        assert len(vol) == len(close)
        assert len(anti) == len(close)
    
    def test_damiani_accuracy(self, test_data):
        """Test Damiani Volatmeter matches expected values from Rust tests - mirrors check_damiani_accuracy"""
        close = test_data['close']
        
        vol, anti = ta_indicators.damiani_volatmeter(
            close,
            vis_atr=13,
            vis_std=20,
            sed_atr=40,
            sed_std=100,
            threshold=1.4
        )
        
        assert len(vol) == len(close)
        assert len(anti) == len(close)
        
        # Expected values from Rust tests
        expected_vol = [
            0.9009485470514558,
            0.8333604467044887,
            0.815318380178986,
            0.8276892636184923,
            0.879447954127426,
        ]
        expected_anti = [
            1.1227721577887388,
            1.1250333024152703,
            1.1325501989919875,
            1.1403866079746106,
            1.1392919184055932,
        ]
        
        # Check last 5 values match expected
        assert_close(
            vol[-5:], 
            expected_vol,
            rtol=1e-2,  # Same tolerance as Rust tests
            msg="Damiani Volatmeter vol last 5 values mismatch"
        )
        assert_close(
            anti[-5:], 
            expected_anti,
            rtol=1e-2,  # Same tolerance as Rust tests
            msg="Damiani Volatmeter anti last 5 values mismatch"
        )
    
    def test_damiani_default_candles(self, test_data):
        """Test Damiani Volatmeter with default parameters - mirrors check_damiani_input_with_default_candles"""
        close = test_data['close']
        
        # Default params: vis_atr=13, vis_std=20, sed_atr=40, sed_std=100, threshold=1.4
        vol, anti = ta_indicators.damiani_volatmeter(close, 13, 20, 40, 100, 1.4)
        assert len(vol) == len(close)
        assert len(anti) == len(close)
    
    def test_damiani_zero_period(self):
        """Test Damiani Volatmeter fails with zero period - mirrors check_damiani_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0] * 25)  # Enough data
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.damiani_volatmeter(input_data, vis_atr=0, vis_std=20, sed_atr=40, sed_std=100, threshold=1.4)
    
    def test_damiani_period_exceeds_length(self):
        """Test Damiani Volatmeter fails when period exceeds data length - mirrors check_damiani_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.damiani_volatmeter(data_small, vis_atr=99999, vis_std=20, sed_atr=40, sed_std=100, threshold=1.4)
    
    def test_damiani_very_small_dataset(self):
        """Test Damiani Volatmeter fails with insufficient data - mirrors check_damiani_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.damiani_volatmeter(single_point, vis_atr=9, vis_std=9, sed_atr=9, sed_std=9, threshold=1.4)
    
    def test_damiani_empty_input(self):
        """Test Damiani Volatmeter fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.damiani_volatmeter(empty, vis_atr=13, vis_std=20, sed_atr=40, sed_std=100, threshold=1.4)
    
    def test_damiani_all_nan_input(self):
        """Test Damiani Volatmeter fails with all NaN input"""
        all_nan = np.array([np.nan, np.nan, np.nan, np.nan, np.nan] * 30)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.damiani_volatmeter(all_nan, vis_atr=13, vis_std=20, sed_atr=40, sed_std=100, threshold=1.4)
    
    def test_damiani_nan_handling(self, test_data):
        """Test Damiani Volatmeter handles NaN values correctly"""
        close = test_data['close']
        
        vol, anti = ta_indicators.damiani_volatmeter(close, vis_atr=13, vis_std=20, sed_atr=40, sed_std=100, threshold=1.4)
        assert len(vol) == len(close)
        assert len(anti) == len(close)
        
        # Check that vol values start appearing after warmup
        # Maximum warmup is max(vis_atr, vis_std, sed_atr, sed_std, 3) = 100
        first_non_nan_vol = next((i for i, v in enumerate(vol) if not np.isnan(v)), -1)
        assert first_non_nan_vol >= 100, f"Vol values appeared too early at index {first_non_nan_vol}"
        
        # Anti values need both stddev windows filled
        first_non_nan_anti = next((i for i, v in enumerate(anti) if not np.isnan(v)), -1)
        assert first_non_nan_anti >= 100, f"Anti values appeared too early at index {first_non_nan_anti}"
    
    def test_damiani_streaming(self, test_data):
        """Test Damiani Volatmeter streaming matches batch calculation - mirrors check_damiani_streaming"""
        # Get price data (using close as high/low/close for simplicity)
        close = test_data['close']
        high = test_data['high']
        low = test_data['low']
        
        vis_atr = 13
        vis_std = 20
        sed_atr = 40
        sed_std = 100
        threshold = 1.4
        
        # Batch calculation (using close only, as per the indicator's behavior)
        batch_vol, batch_anti = ta_indicators.damiani_volatmeter(
            close, vis_atr=vis_atr, vis_std=vis_std, sed_atr=sed_atr, sed_std=sed_std, threshold=threshold
        )
        
        # Streaming calculation
        stream = ta_indicators.DamianiVolatmeterStream(
            high=high.tolist(), low=low.tolist(), close=close.tolist(),
            vis_atr=vis_atr, vis_std=vis_std, sed_atr=sed_atr, sed_std=sed_std, threshold=threshold
        )
        stream_vol = []
        stream_anti = []
        
        for _ in range(len(close)):
            result = stream.update()
            if result is not None:
                vol_val, anti_val = result
                stream_vol.append(vol_val)
                stream_anti.append(anti_val)
            else:
                stream_vol.append(np.nan)
                stream_anti.append(np.nan)
        
        stream_vol = np.array(stream_vol)
        stream_anti = np.array(stream_anti)
        
        # Compare batch vs streaming
        assert len(batch_vol) == len(stream_vol)
        assert len(batch_anti) == len(stream_anti)
        
        # Compare vol values where both are not NaN
        for i, (b, s) in enumerate(zip(batch_vol, stream_vol)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-8, atol=1e-8, 
                        msg=f"Damiani Volatmeter vol streaming mismatch at index {i}")
        
        # Compare anti values where both are not NaN
        for i, (b, s) in enumerate(zip(batch_anti, stream_anti)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-8, atol=1e-8, 
                        msg=f"Damiani Volatmeter anti streaming mismatch at index {i}")
    
    def test_damiani_batch(self, test_data):
        """Test Damiani Volatmeter batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        # Test with single parameter set (default values)
        result = ta_indicators.damiani_volatmeter_batch(
            close,
            vis_atr_range=(13, 40, 1),  # Default sweep from Rust
            vis_std_range=(20, 40, 1),  # Default sweep from Rust 
            sed_atr_range=(40, 40, 0),  # Single value
            sed_std_range=(100, 100, 0),  # Single value
            threshold_range=(1.4, 1.4, 0.0)  # Single value
        )
        
        assert 'vol' in result
        assert 'anti' in result
        assert 'vis_atr' in result
        assert 'vis_std' in result
        assert 'sed_atr' in result
        assert 'sed_std' in result
        assert 'thresholds' in result
        
        # Check shapes
        vol_values = result['vol']
        anti_values = result['anti']
        assert vol_values.shape[1] == len(close)  # Columns = data length
        assert anti_values.shape[1] == len(close)  # Columns = data length
        
        # Check that we have the expected number of combinations
        # vis_atr: 13 to 40 step 1 = 28 values
        # vis_std: 20 to 40 step 1 = 21 values
        # sed_atr: 40 (single value)
        # sed_std: 100 (single value)
        # threshold: 1.4 (single value)
        # Total: 28 * 21 * 1 * 1 * 1 = 588 combinations
        expected_rows = 28 * 21  # 588
        assert vol_values.shape[0] == expected_rows
        assert anti_values.shape[0] == expected_rows
    
    def test_damiani_different_thresholds(self, test_data):
        """Test Damiani Volatmeter with different threshold values"""
        close = test_data['close']
        
        # Test with different thresholds
        thresholds = [0.5, 1.0, 1.4, 2.0]
        results = []
        
        for thresh in thresholds:
            vol, anti = ta_indicators.damiani_volatmeter(close, 13, 20, 40, 100, thresh)
            results.append((vol, anti))
        
        # Anti values should change with threshold (since anti = threshold - ratio)
        for i in range(len(thresholds) - 1):
            anti1 = results[i][1]
            anti2 = results[i + 1][1]
            # Find indices where both are not NaN
            valid_idx = ~(np.isnan(anti1) | np.isnan(anti2))
            if np.any(valid_idx):
                # The difference should be approximately the threshold difference
                diff = anti2[valid_idx] - anti1[valid_idx]
                expected_diff = thresholds[i + 1] - thresholds[i]
                assert_close(diff, expected_diff, rtol=1e-10, atol=1e-10,
                           msg=f"Anti values should differ by threshold difference")