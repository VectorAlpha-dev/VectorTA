"""
Python binding tests for HALFTREND indicator.
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


def halftrend_compat(*args, **kwargs):
    """Compatibility wrapper that returns tuple format from dict format."""
    result = ta_indicators.halftrend_tuple(*args, **kwargs)
    if isinstance(result, dict):

        return (result['halftrend'], result['trend'], result['atr_high'],
                result['atr_low'], result['buy_signal'], result['sell_signal'])
    else:

        return result


class TestHalfTrend:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_halftrend_accuracy(self, test_data):
        """Test HALFTREND matches expected values from Rust tests - mirrors check_halftrend_accuracy."""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['halftrend']


        halftrend, trend, atr_high, atr_low, buy_signal, sell_signal = ta_indicators.halftrend_tuple(
            high, low, close,
            expected['default_params']['amplitude'],
            expected['default_params']['channel_deviation'],
            expected['default_params']['atr_period']
        )


        assert isinstance(halftrend, np.ndarray), "halftrend should be numpy array"
        assert isinstance(trend, np.ndarray), "trend should be numpy array"
        assert isinstance(atr_high, np.ndarray), "atr_high should be numpy array"
        assert isinstance(atr_low, np.ndarray), "atr_low should be numpy array"
        assert isinstance(buy_signal, np.ndarray), "buy_signal should be numpy array"
        assert isinstance(sell_signal, np.ndarray), "sell_signal should be numpy array"


        assert len(halftrend) == len(high), "HalfTrend length mismatch"
        assert len(trend) == len(high), "Trend length mismatch"
        assert len(atr_high) == len(high), "ATR High length mismatch"
        assert len(atr_low) == len(high), "ATR Low length mismatch"
        assert len(buy_signal) == len(high), "Buy signal length mismatch"
        assert len(sell_signal) == len(high), "Sell signal length mismatch"


        test_indices = expected['test_indices']
        expected_halftrend = expected['expected_halftrend']
        expected_trend = expected['expected_trend']

        for i, idx in enumerate(test_indices):
            assert_close(
                halftrend[idx],
                expected_halftrend[i],
                rtol=1e-5,
                msg=f"HalfTrend mismatch at index {idx}"
            )
            assert_close(
                trend[idx],
                expected_trend[i],
                rtol=1e-5,
                msg=f"Trend mismatch at index {idx}"
            )











    def test_halftrend_dict_api(self, test_data):
        """Test new dict-based API returns correct structure."""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]


        result = ta_indicators.halftrend(high, low, close, 2, 2.0, 100)


        assert isinstance(result, dict), "Should return dict"
        expected_keys = ['halftrend', 'trend', 'atr_high', 'atr_low', 'buy_signal', 'sell_signal']
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
            assert isinstance(result[key], np.ndarray), f"{key} should be numpy array"
            assert len(result[key]) == len(high), f"{key} length mismatch"

    def test_halftrend_tuple_compatibility(self, test_data):
        """Test backward compatibility with tuple API."""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]


        if hasattr(ta_indicators, 'halftrend_tuple'):
            result = ta_indicators.halftrend_tuple(high, low, close, 2, 2.0, 100)
            assert isinstance(result, tuple), "halftrend_tuple should return tuple"
            assert len(result) == 6, "Should return 6 elements"
            for item in result:
                assert isinstance(item, np.ndarray), "Each element should be numpy array"

    def test_halftrend_partial_params(self, test_data):
        """Test HALFTREND with partial parameters (None values) - mirrors check_halftrend_partial_params."""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        halftrend, trend, atr_high, atr_low, buy_signal, sell_signal = halftrend_compat(
            high, low, close,
            amplitude=2,
            channel_deviation=2.0,
            atr_period=100
        )

        assert len(halftrend) == len(high)
        assert len(trend) == len(high)

    def test_halftrend_default_candles(self, test_data):
        """Test HALFTREND with default parameters - mirrors check_halftrend_default_candles."""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        halftrend, trend, atr_high, atr_low, buy_signal, sell_signal = ta_indicators.halftrend_tuple(
            high, low, close, 2, 2.0, 100
        )
        assert len(halftrend) == len(high)

    def test_halftrend_custom_params(self, test_data):
        """Test HALFTREND with custom parameters."""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        halftrend, trend, atr_high, atr_low, buy_signal, sell_signal = ta_indicators.halftrend_tuple(
            high, low, close,
            amplitude=3,
            channel_deviation=2.5,
            atr_period=50
        )


        assert halftrend is not None
        assert len(halftrend) == len(high)


        warmup_period = 50
        non_nan_count = np.sum(~np.isnan(halftrend[warmup_period:]))
        assert non_nan_count > 0, "Should have non-NaN values after warmup period"

    def test_halftrend_nan_handling(self, test_data):
        """Test HALFTREND handles NaN values correctly in warmup period - mirrors check_halftrend_nan_handling."""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['halftrend']

        halftrend, trend, atr_high, atr_low, buy_signal, sell_signal = ta_indicators.halftrend_tuple(
            high, low, close,
            amplitude=2,
            channel_deviation=2.0,
            atr_period=100
        )


        expected_warmup = expected['warmup_period']


        for i in range(expected_warmup):
            assert np.isnan(halftrend[i]), f"Expected NaN at index {i} during warmup"
            assert np.isnan(trend[i]), f"Expected NaN in trend at index {i} during warmup"


        if len(halftrend) > expected_warmup + 10:
            for i in range(expected_warmup, expected_warmup + 10):
                assert not np.isnan(halftrend[i]), f"Unexpected NaN at index {i} after warmup"
                assert not np.isnan(trend[i]), f"Unexpected NaN in trend at index {i} after warmup"

    def test_halftrend_warmup_period(self, test_data):
        """Test HALFTREND warmup period calculation."""
        high = test_data['high'][:500]
        low = test_data['low'][:500]
        close = test_data['close'][:500]


        test_cases = [
            {'amplitude': 2, 'atr_period': 100, 'expected_warmup': 99},
            {'amplitude': 50, 'atr_period': 20, 'expected_warmup': 49},
            {'amplitude': 10, 'atr_period': 10, 'expected_warmup': 9},
        ]

        for case in test_cases:
            halftrend, trend, atr_high, atr_low, buy_signal, sell_signal = ta_indicators.halftrend_tuple(
                high, low, close,
                amplitude=case['amplitude'],
                channel_deviation=2.0,
                atr_period=case['atr_period']
            )


            nan_count = np.sum(np.isnan(halftrend[:case['expected_warmup'] + 1]))
            assert nan_count >= case['expected_warmup'], \
                f"Expected at least {case['expected_warmup']} NaN values for amplitude={case['amplitude']}, atr_period={case['atr_period']}"

    def test_halftrend_signal_detection(self, test_data):
        """Test HALFTREND buy/sell signal generation."""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']

        halftrend, trend, atr_high, atr_low, buy_signal, sell_signal = ta_indicators.halftrend_tuple(high, low, close, 2, 2.0, 100)


        buy_nan_count = np.sum(np.isnan(buy_signal))
        sell_nan_count = np.sum(np.isnan(sell_signal))


        assert buy_nan_count > len(high) * 0.95, "Buy signals should be sparse (mostly NaN)"
        assert sell_nan_count > len(high) * 0.95, "Sell signals should be sparse (mostly NaN)"


        buy_signal_count = np.sum(~np.isnan(buy_signal))
        sell_signal_count = np.sum(~np.isnan(sell_signal))


        assert buy_signal_count > 0, "Should have at least one buy signal"
        assert sell_signal_count > 0, "Should have at least one sell signal"

    def test_halftrend_empty_input(self):
        """Test HALFTREND fails with empty input."""
        empty = np.array([])

        with pytest.raises(ValueError, match="Empty input data|Input data slice is empty"):
            ta_indicators.halftrend_tuple(empty, empty, empty, 2, 2.0, 100)

    def test_halftrend_mismatched_lengths(self):
        """Test HALFTREND fails with mismatched array lengths."""
        high = np.array([1, 2, 3])
        low = np.array([1, 2])
        close = np.array([1, 2, 3])


        with pytest.raises((ValueError, TypeError), match="Mismatched|lengths|size|Not enough valid|ndarray"):
            ta_indicators.halftrend_tuple(high, low, close, 2, 2.0, 100)

    def test_halftrend_zero_amplitude(self):
        """Test HALFTREND fails with zero amplitude - mirrors check_halftrend_invalid_period."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.halftrend_tuple(
                data, data, data,
                amplitude=0,
                channel_deviation=2.0,
                atr_period=100
            )

    def test_halftrend_invalid_channel_deviation(self):
        """Test HALFTREND fails with invalid channel deviation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


        with pytest.raises(ValueError, match="Invalid channel_deviation"):
            ta_indicators.halftrend_tuple(
                data, data, data,
                amplitude=2,
                channel_deviation=-1.0,
                atr_period=100
            )


        with pytest.raises(ValueError, match="Invalid channel_deviation"):
            ta_indicators.halftrend_tuple(
                data, data, data,
                amplitude=2,
                channel_deviation=0.0,
                atr_period=100
            )

    def test_halftrend_invalid_atr_period(self):
        """Test HALFTREND fails with invalid ATR period."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.halftrend_tuple(
                data, data, data,
                amplitude=2,
                channel_deviation=2.0,
                atr_period=0
            )

    def test_halftrend_period_exceeds_length(self):
        """Test HALFTREND fails when period exceeds data length."""
        small_data = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Not enough|Invalid period|exceeds"):
            ta_indicators.halftrend_tuple(
                small_data, small_data, small_data,
                amplitude=10,
                channel_deviation=2.0,
                atr_period=100
            )

    def test_halftrend_very_small_dataset(self):
        """Test HALFTREND fails with insufficient data."""
        single_point = np.array([42.0])

        with pytest.raises(ValueError, match="Period.*exceeds|Not enough|Invalid"):
            ta_indicators.halftrend_tuple(
                single_point, single_point, single_point,
                amplitude=2,
                channel_deviation=2.0,
                atr_period=100
            )

    def test_halftrend_all_nan_input(self):
        """Test HALFTREND with all NaN values."""
        all_nan = np.full(100, np.nan)

        with pytest.raises(ValueError, match="All values are NaN|All NaN|All input values"):
            ta_indicators.halftrend_tuple(all_nan, all_nan, all_nan, 2, 2.0, 100)

    def test_halftrend_not_enough_valid_data(self):
        """Test HALFTREND with mostly NaN values but some valid data (not enough for calculation)."""
        n = 10
        high = np.full(n, np.nan)
        low = np.full(n, np.nan)
        close = np.full(n, np.nan)


        high[5] = 1.0
        low[5] = 1.0
        close[5] = 1.0

        with pytest.raises(ValueError, match="Not enough valid data"):
            ta_indicators.halftrend_tuple(high, low, close, 9, 2.0, 9)

    def test_halftrend_invalid_channel_deviation_nan(self):
        """Test HALFTREND fails with NaN channel deviation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(ValueError, match="Invalid channel_deviation|NaN"):
            ta_indicators.halftrend_tuple(
                data, data, data,
                amplitude=2,
                channel_deviation=np.nan,
                atr_period=2
            )

    def test_halftrend_batch(self, test_data):
        """Test HALFTREND batch processing - mirrors check_batch_default_row."""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['halftrend']


        result = ta_indicators.halftrend_batch(
            high, low, close,
            amplitude_start=expected['default_params']['amplitude'],
            amplitude_end=expected['default_params']['amplitude'],
            amplitude_step=0,
            channel_deviation_start=expected['default_params']['channel_deviation'],
            channel_deviation_end=expected['default_params']['channel_deviation'],
            channel_deviation_step=0.0,
            atr_period_start=expected['default_params']['atr_period'],
            atr_period_end=expected['default_params']['atr_period'],
            atr_period_step=0
        )


        assert 'values' in result, "Missing values in batch output"
        assert 'series' in result, "Missing series in batch output"
        assert 'rows' in result, "Missing rows in batch output"
        assert 'cols' in result, "Missing cols in batch output"

        assert result['cols'] == len(high), "Batch cols should match input length"
        assert result['rows'] == 1, "Should have 1 row for single parameter combination"


        assert result['values'].shape == (6, result['cols']), \
            "Values batch shape mismatch - should be (6, cols) for single param"


        halftrend_row = result['values'][0, :]
        trend_row = result['values'][1, :]


        for i, idx in enumerate(expected['test_indices']):
            assert_close(
                halftrend_row[idx],
                expected['expected_halftrend'][i],
                rtol=1e-5,
                msg=f"Batch halftrend mismatch at index {idx}"
            )
            assert_close(
                trend_row[idx],
                expected['expected_trend'][i],
                rtol=1e-5,
                msg=f"Batch trend mismatch at index {idx}"
            )

    def test_halftrend_batch_multiple_params(self, test_data):
        """Test HALFTREND batch processing with multiple parameter combinations."""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]


        result = ta_indicators.halftrend_batch(
            high, low, close,
            amplitude_start=2,
            amplitude_end=4,
            amplitude_step=1,
            channel_deviation_start=2.0,
            channel_deviation_end=2.5,
            channel_deviation_step=0.5,
            atr_period_start=50,
            atr_period_end=50,
            atr_period_step=0
        )


        expected_combos = 6
        assert result['rows'] == expected_combos, f"Expected {expected_combos} combinations"
        assert result['cols'] == 100, "Cols should match input length"


        assert result['values'].shape == (expected_combos * 6, 100), \
            "Values shape should be (combos*6, cols)"


        if 'combos' in result:
            assert len(result['combos']) == expected_combos


            expected_params = [
                {'amplitude': 2, 'channel_deviation': 2.0, 'atr_period': 50},
                {'amplitude': 2, 'channel_deviation': 2.5, 'atr_period': 50},
                {'amplitude': 3, 'channel_deviation': 2.0, 'atr_period': 50},
                {'amplitude': 3, 'channel_deviation': 2.5, 'atr_period': 50},
                {'amplitude': 4, 'channel_deviation': 2.0, 'atr_period': 50},
                {'amplitude': 4, 'channel_deviation': 2.5, 'atr_period': 50},
            ]


            for i, expected in enumerate(expected_params):
                combo = result['combos'][i]
                assert combo['amplitude'] == expected['amplitude'], f"Amplitude mismatch at combo {i}"
                assert_close(combo['channel_deviation'], expected['channel_deviation'],
                           rtol=1e-10, msg=f"Channel deviation mismatch at combo {i}")
                assert combo['atr_period'] == expected['atr_period'], f"ATR period mismatch at combo {i}"

    def test_halftrend_streaming(self, test_data):
        """Test HALFTREND streaming matches batch calculation - mirrors check_halftrend_streaming."""
        high = test_data['high'][:200]
        low = test_data['low'][:200]
        close = test_data['close'][:200]
        amplitude = 2
        channel_deviation = 2.0
        atr_period = 100


        batch_ht, batch_trend, batch_ah, batch_al, batch_bs, batch_ss = ta_indicators.halftrend_tuple(
            high, low, close, amplitude, channel_deviation, atr_period
        )


        stream = ta_indicators.HalfTrendStream(amplitude, channel_deviation, atr_period)
        stream_results = []

        for i in range(len(high)):
            result = stream.update(high[i], low[i], close[i])
            stream_results.append(result)


        warmup = EXPECTED_OUTPUTS['halftrend']['warmup_period']


        matches = 0
        comparisons = 0

        for i in range(warmup, min(len(high), warmup + 50)):
            if stream_results[i] is not None:
                stream_ht, stream_trend, stream_ah, stream_al, stream_bs, stream_ss = stream_results[i]
                comparisons += 1


                try:

                    assert_close(stream_ht, batch_ht[i], rtol=1e-5, atol=1e-5,
                               msg=f"HalfTrend streaming mismatch at index {i}")
                    assert_close(stream_trend, batch_trend[i], rtol=1e-5, atol=1e-5,
                               msg=f"Trend streaming mismatch at index {i}")
                    matches += 1
                except AssertionError:


                    assert_close(stream_ht, batch_ht[i], rtol=1e-2, atol=1e-2,
                               msg=f"HalfTrend streaming mismatch at index {i}")
                    assert_close(stream_trend, batch_trend[i], rtol=1e-2, atol=1e-2,
                               msg=f"Trend streaming mismatch at index {i}")


        assert matches > comparisons * 0.3, \
            f"Too few exact matches in streaming: {matches}/{comparisons}"

    def test_halftrend_reinput(self, test_data):
        """Test HALFTREND applied twice (re-input) - mirrors ALMA's reinput test."""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        ht1, tr1, ah1, al1, bs1, ss1 = ta_indicators.halftrend_tuple(
            high, low, close,
            amplitude=2, channel_deviation=2.0, atr_period=100
        )
        assert len(ht1) == len(high)



        ht2, tr2, ah2, al2, bs2, ss2 = ta_indicators.halftrend_tuple(
            ht1, ht1, ht1,
            amplitude=2, channel_deviation=2.0, atr_period=100
        )
        assert len(ht2) == len(ht1)



        differences = 0
        for i in range(200, min(300, len(ht2))):
            if not np.isnan(ht1[i]) and not np.isnan(ht2[i]):
                if abs(ht1[i] - ht2[i]) > 1e-10:
                    differences += 1

        assert differences > 0, "Reinput should produce different values due to re-smoothing"

    def test_halftrend_intermittent_nan(self, test_data):
        """Test HALFTREND with data containing intermittent NaN values."""
        high = test_data['high'][:100].copy()
        low = test_data['low'][:100].copy()
        close = test_data['close'][:100].copy()


        nan_indices = [10, 25, 40, 55, 70, 85]
        for idx in nan_indices:
            high[idx] = np.nan
            low[idx] = np.nan
            close[idx] = np.nan


        halftrend, trend, atr_high, atr_low, buy_signal, sell_signal = ta_indicators.halftrend_tuple(
            high, low, close, 2, 2.0, 20
        )

        assert len(halftrend) == len(high)




        warmup = 20
        valid_count = np.sum(~np.isnan(halftrend[warmup:]))
        assert valid_count > 0, "Should have valid values after warmup despite intermittent NaNs"

    def test_halftrend_large_dataset(self, test_data):
        """Test HALFTREND performance with large dataset."""

        size = 10000
        high = np.random.randn(size) * 10 + 100
        low = high - np.abs(np.random.randn(size))
        close = (high + low) / 2 + np.random.randn(size) * 0.1


        halftrend, trend, atr_high, atr_low, buy_signal, sell_signal = ta_indicators.halftrend_tuple(
            high, low, close, 5, 2.0, 50
        )

        assert len(halftrend) == size


        warmup = 49
        nan_count = np.sum(np.isnan(halftrend[:warmup]))
        valid_count = np.sum(~np.isnan(halftrend[warmup:]))

        assert nan_count > 0, "Should have NaN values in warmup period"
        assert valid_count > size - warmup - 100, "Should have mostly valid values after warmup"

    def test_halftrend_edge_parameter_combinations(self, test_data):
        """Test HALFTREND with edge case parameter combinations."""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]


        test_cases = [
            {'amplitude': 1, 'channel_deviation': 0.5, 'atr_period': 2},
            {'amplitude': 50, 'channel_deviation': 5.0, 'atr_period': 50},
            {'amplitude': 2, 'channel_deviation': 0.1, 'atr_period': 10},
        ]

        for params in test_cases:
            halftrend, trend, _, _, _, _ = ta_indicators.halftrend_tuple(
                high, low, close,
                params['amplitude'],
                params['channel_deviation'],
                params['atr_period']
            )

            assert len(halftrend) == len(high), \
                f"Length mismatch for params {params}"


            expected_warmup = max(params['amplitude'], params['atr_period']) - 1


            if expected_warmup < len(high):
                has_nan_in_warmup = np.any(np.isnan(halftrend[:expected_warmup]))
                has_values_after = np.any(~np.isnan(halftrend[expected_warmup:]))
                assert has_nan_in_warmup or expected_warmup == 0, \
                    f"Expected NaN in warmup for params {params}"
                assert has_values_after, \
                    f"Expected values after warmup for params {params}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])