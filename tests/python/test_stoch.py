"""
Python binding tests for Stoch indicator.
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


class TestStoch:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_stoch_partial_params(self, test_data):
        """Test Stoch with partial parameters (None values) - mirrors check_stoch_partial_params"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        k, d = ta_indicators.stoch(high, low, close)
        assert len(k) == len(close)
        assert len(d) == len(close)

    def test_stoch_accuracy(self, test_data):
        """Test Stoch matches expected values from Rust tests - mirrors check_stoch_accuracy"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        expected_k = [
            42.51122827572717,
            40.13864479593807,
            37.853934778363374,
            37.337021714266086,
            36.26053890551548,
        ]
        expected_d = [
            41.36561869426493,
            41.7691857059163,
            40.16793595000925,
            38.44320042952222,
            37.15049846604803,
        ]



        k, d = ta_indicators.stoch(
            high, low, close,
            fastk_period=14,
            slowk_period=3,
            slowk_ma_type="sma",
            slowd_period=3,
            slowd_ma_type="sma"
        )

        assert len(k) == len(close)
        assert len(d) == len(close)


        assert_close(
            k[-5:],
            expected_k,
            rtol=1e-6,
            msg="Stoch K last 5 values mismatch"
        )

        assert_close(
            d[-5:],
            expected_d,
            rtol=1e-6,
            msg="Stoch D last 5 values mismatch"
        )

    def test_stoch_default_candles(self, test_data):
        """Test Stoch with default parameters - mirrors check_stoch_default_candles"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']



        k, d = ta_indicators.stoch(high, low, close)
        assert len(k) == len(close)
        assert len(d) == len(close)

    def test_stoch_zero_period(self):
        """Test Stoch fails with zero period - mirrors check_stoch_zero_period"""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 9.5, 10.5])
        close = np.array([9.5, 10.6, 11.5])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.stoch(high, low, close, fastk_period=0)

    def test_stoch_period_exceeds_length(self):
        """Test Stoch fails when period exceeds data length - mirrors check_stoch_period_exceeds_length"""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 9.5, 10.5])
        close = np.array([9.5, 10.6, 11.5])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.stoch(high, low, close, fastk_period=10)

    def test_stoch_all_nan(self):
        """Test Stoch fails with all NaN values - mirrors check_stoch_all_nan"""

        nan_data = np.array([float('nan')] * 20)

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.stoch(nan_data, nan_data, nan_data)

    def test_stoch_empty_input(self):
        """Test Stoch fails with empty input"""
        empty = np.array([])

        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.stoch(empty, empty, empty)

    def test_stoch_mismatched_lengths(self):
        """Test Stoch fails with mismatched input lengths"""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 9.5])
        close = np.array([9.5, 10.6, 11.5])

        with pytest.raises(ValueError, match="Mismatched length"):
            ta_indicators.stoch(high, low, close)

    def test_stoch_nan_handling(self, test_data):
        """Test Stoch handles NaN values correctly"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']

        k, d = ta_indicators.stoch(high, low, close)
        assert len(k) == len(close)
        assert len(d) == len(close)



        warmup = 14 + 3 + 3 - 3
        if len(k) > warmup + 100:
            assert not np.any(np.isnan(k[warmup + 100:])), "Found unexpected NaN in K after warmup period"
            assert not np.any(np.isnan(d[warmup + 100:])), "Found unexpected NaN in D after warmup period"

    def test_stoch_streaming(self, test_data):
        """Test Stoch streaming matches batch calculation - mirrors check_stoch_streaming"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]


        batch_k, batch_d = ta_indicators.stoch(high, low, close)


        stream = ta_indicators.StochStream(
            fastk_period=14,
            slowk_period=3,
            slowk_ma_type="sma",
            slowd_period=3,
            slowd_ma_type="sma"
        )

        stream_k_values = []
        stream_d_values = []

        for h, l, c in zip(high, low, close):
            result = stream.update(h, l, c)
            if result is not None:
                k_val, d_val = result
                stream_k_values.append(k_val)
                stream_d_values.append(d_val)
            else:
                stream_k_values.append(np.nan)
                stream_d_values.append(np.nan)

        stream_k = np.array(stream_k_values)
        stream_d = np.array(stream_d_values)


        assert len(batch_k) == len(stream_k)
        assert len(batch_d) == len(stream_d)


        for i, (b, s) in enumerate(zip(batch_k, stream_k)):
            if np.isnan(b) and np.isnan(s):
                continue
            if not np.isnan(b) and not np.isnan(s):
                assert_close(b, s, rtol=1e-9, atol=1e-9,
                            msg=f"Stoch K streaming mismatch at index {i}")


        for i, (b, s) in enumerate(zip(batch_d, stream_d)):
            if np.isnan(b) and np.isnan(s):
                continue
            if not np.isnan(b) and not np.isnan(s):
                assert_close(b, s, rtol=1e-9, atol=1e-9,
                            msg=f"Stoch D streaming mismatch at index {i}")

    def test_stoch_batch(self, test_data):
        """Test Stoch batch processing - mirrors check_batch_default_row"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        result = ta_indicators.stoch_batch(
            high, low, close,
            fastk_range=(14, 14, 0),
            slowk_range=(3, 3, 0),
            slowk_ma_type="sma",
            slowd_range=(3, 3, 0),
            slowd_ma_type="sma"
        )

        assert 'k' in result
        assert 'd' in result
        assert 'fastk_periods' in result
        assert 'slowk_periods' in result
        assert 'slowk_types' in result
        assert 'slowd_periods' in result
        assert 'slowd_types' in result


        assert result['k'].shape[0] == 1
        assert result['d'].shape[0] == 1
        assert result['k'].shape[1] == len(close)
        assert result['d'].shape[1] == len(close)


        single_k, single_d = ta_indicators.stoch(high, low, close)

        assert_close(
            result['k'][0],
            single_k,
            rtol=1e-9,
            msg="Batch K values don't match single calculation"
        )

        assert_close(
            result['d'][0],
            single_d,
            rtol=1e-9,
            msg="Batch D values don't match single calculation"
        )

    def test_stoch_batch_multiple_params(self, test_data):
        """Test Stoch batch processing with multiple parameter sets"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]


        result = ta_indicators.stoch_batch(
            high, low, close,
            fastk_range=(10, 20, 5),
            slowk_range=(2, 4, 1),
            slowk_ma_type="sma",
            slowd_range=(2, 4, 1),
            slowd_ma_type="sma"
        )


        expected_combos = 3 * 3 * 3
        assert result['k'].shape[0] == expected_combos
        assert result['d'].shape[0] == expected_combos
        assert result['k'].shape[1] == len(close)
        assert result['d'].shape[1] == len(close)


        assert len(result['fastk_periods']) == expected_combos
        assert len(result['slowk_periods']) == expected_combos
        assert len(result['slowd_periods']) == expected_combos

    def test_stoch_kernel_selection(self, test_data):
        """Test Stoch with different kernel selections"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]


        k_scalar, d_scalar = ta_indicators.stoch(
            high, low, close,
            kernel="scalar"
        )


        k_auto, d_auto = ta_indicators.stoch(
            high, low, close,
            kernel=None
        )


        assert_close(k_scalar, k_auto, rtol=1e-9,
                    msg="K values differ between scalar and auto kernel")
        assert_close(d_scalar, d_auto, rtol=1e-9,
                    msg="D values differ between scalar and auto kernel")

    def test_stoch_different_ma_types(self, test_data):
        """Test Stoch with different MA types for smoothing"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]


        k_sma, d_sma = ta_indicators.stoch(
            high, low, close,
            slowk_ma_type="sma",
            slowd_ma_type="sma"
        )


        k_ema, d_ema = ta_indicators.stoch(
            high, low, close,
            slowk_ma_type="ema",
            slowd_ma_type="ema"
        )


        assert len(k_sma) == len(k_ema)
        assert len(d_sma) == len(d_ema)


        warmup = 20
        k_diff = np.abs(k_sma[warmup:] - k_ema[warmup:])
        d_diff = np.abs(d_sma[warmup:] - d_ema[warmup:])


        assert np.any(k_diff > 1e-6), "K values should differ between SMA and EMA"
        assert np.any(d_diff > 1e-6), "D values should differ between SMA and EMA"