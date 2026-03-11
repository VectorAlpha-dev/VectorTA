import numpy as np
import pytest

try:
    import my_project as ta_indicators
except ImportError:
    try:
        import vector_ta as ta_indicators
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )

from test_utils import load_test_data, assert_close


def naive_normalized_volume_true_range(
    open_,
    high,
    low,
    close,
    volume,
    true_range_style="body",
    outlier_range=5.0,
    atr_length=14,
    volume_length=14,
):
    open_ = np.asarray(open_, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    volume = np.asarray(volume, dtype=float)
    length = close.shape[0]

    normalized_volume = np.full(length, np.nan)
    normalized_true_range = np.full(length, np.nan)
    baseline = np.full(length, np.nan)
    atr = np.full(length, np.nan)
    average_volume = np.full(length, np.nan)

    abs_sum = 0.0
    volume_sum = 0.0
    count = 0
    abs_variance_sum = 0.0
    abs_qualifying = 0
    abs_p_stdev = np.nan
    volume_variance_sum = 0.0
    volume_qualifying = 0
    volume_p_stdev = np.nan
    prev_close = np.nan
    have_prev_close = False

    atr_first = np.nan
    atr_ready = False
    atr_ring = np.zeros(atr_length, dtype=float)
    atr_sum = 0.0
    atr_head = 0

    volume_first = np.nan
    volume_ready = False
    volume_ring = np.zeros(volume_length, dtype=float)
    volume_sum_ma = 0.0
    volume_head = 0

    for idx in range(length):
        if true_range_style == "body":
            valid = np.isfinite(open_[idx]) and np.isfinite(close[idx]) and np.isfinite(volume[idx])
        elif true_range_style == "hl":
            valid = np.isfinite(high[idx]) and np.isfinite(low[idx]) and np.isfinite(volume[idx])
        else:
            valid = np.isfinite(close[idx]) and np.isfinite(volume[idx])

        if not valid:
            if np.isfinite(close[idx]):
                prev_close = close[idx]
                have_prev_close = True
            continue

        prev = prev_close if have_prev_close else close[idx]
        if true_range_style == "body":
            start, finish = open_[idx], close[idx]
        elif true_range_style == "hl":
            start, finish = low[idx], high[idx]
        else:
            start, finish = prev, close[idx]

        prev_close = close[idx]
        have_prev_close = True

        denom = min(start, finish)
        if not np.isfinite(denom) or denom <= 0.0:
            continue

        abs_percent = abs(finish - start) / denom
        if not np.isfinite(abs_percent):
            continue

        count += 1
        abs_sum += abs_percent
        volume_sum += volume[idx]
        avg_abs_percent = abs_sum / count
        avg_volume = volume_sum / count

        if abs_percent > avg_abs_percent:
            abs_variance_sum += (abs_percent - avg_abs_percent) ** 2
            abs_qualifying += 1
        if abs_qualifying >= 2:
            abs_p_stdev = np.sqrt(abs_variance_sum / (abs_qualifying - 1))

        if volume[idx] > avg_volume:
            volume_variance_sum += (volume[idx] - avg_volume) ** 2
            volume_qualifying += 1
        if volume_qualifying >= 2:
            volume_p_stdev = np.sqrt(volume_variance_sum / (volume_qualifying - 1))

        abs_percent_max = avg_abs_percent + abs_p_stdev * outlier_range if np.isfinite(abs_p_stdev) else np.nan
        normalized_avg_percent = avg_abs_percent / abs_percent_max if np.isfinite(abs_percent_max) and abs_percent_max > 0.0 else np.nan
        if (
            np.isfinite(normalized_avg_percent)
            and 0.0 < normalized_avg_percent < 1.0
            and np.isfinite(volume_p_stdev)
            and volume_p_stdev > 0.0
        ):
            scale_factor = avg_volume * (1.0 - normalized_avg_percent) / (normalized_avg_percent * volume_p_stdev)
        else:
            scale_factor = np.nan
        max_volume = avg_volume + volume_p_stdev * scale_factor if np.isfinite(scale_factor) and np.isfinite(volume_p_stdev) else np.nan

        norm_tr = abs_percent / abs_percent_max if np.isfinite(abs_percent_max) and abs_percent_max > 0.0 and abs_percent <= abs_percent_max else (
            abs_percent_max / abs_percent_max if np.isfinite(abs_percent_max) and abs_percent_max > 0.0 else np.nan
        )
        norm_vol = volume[idx] / max_volume if np.isfinite(max_volume) and max_volume > 0.0 and volume[idx] <= max_volume else (
            max_volume / max_volume if np.isfinite(max_volume) and max_volume > 0.0 else np.nan
        )
        norm_avg_vol = avg_volume / max_volume if np.isfinite(max_volume) and max_volume > 0.0 else np.nan

        norm_vol_pct = norm_vol * 100.0
        norm_tr_pct = norm_tr * 100.0
        baseline_pct = norm_avg_vol * 100.0

        if not atr_ready:
            if np.isfinite(norm_tr_pct):
                atr_first = norm_tr_pct
                atr_ring[:] = atr_first
                atr_sum = atr_first * atr_length
                atr_head = 1 % atr_length
                atr_ready = True
                atr_value = atr_first
            else:
                atr_value = np.nan
        else:
            source = norm_tr_pct if np.isfinite(norm_tr_pct) else atr_first
            atr_sum += source - atr_ring[atr_head]
            atr_ring[atr_head] = source
            atr_head = (atr_head + 1) % atr_length
            atr_value = atr_sum / atr_length

        if not volume_ready:
            if np.isfinite(norm_vol_pct):
                volume_first = norm_vol_pct
                volume_ring[:] = volume_first
                volume_sum_ma = volume_first * volume_length
                volume_head = 1 % volume_length
                volume_ready = True
                average_volume_value = volume_first
            else:
                average_volume_value = np.nan
        else:
            source = norm_vol_pct if np.isfinite(norm_vol_pct) else volume_first
            volume_sum_ma += source - volume_ring[volume_head]
            volume_ring[volume_head] = source
            volume_head = (volume_head + 1) % volume_length
            average_volume_value = volume_sum_ma / volume_length

        if all(
            np.isfinite(value)
            for value in (norm_vol_pct, norm_tr_pct, baseline_pct, atr_value, average_volume_value)
        ):
            normalized_volume[idx] = norm_vol_pct
            normalized_true_range[idx] = norm_tr_pct
            baseline[idx] = baseline_pct
            atr[idx] = atr_value
            average_volume[idx] = average_volume_value

    return normalized_volume, normalized_true_range, baseline, atr, average_volume


class TestNormalizedVolumeTrueRange:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_basic_output_shape(self, test_data):
        out = ta_indicators.normalized_volume_true_range(
            test_data["open"],
            test_data["high"],
            test_data["low"],
            test_data["close"],
            test_data["volume"],
        )
        assert len(out) == 5
        for series in out:
            assert len(series) == len(test_data["close"])
            assert np.any(np.isfinite(series[32:]))

    def test_matches_naive_reference(self):
        open_ = np.array([10.0, 10.5, 11.0, 11.2, 11.4, 11.1, 11.6, 11.9], dtype=float)
        high = np.array([10.8, 11.1, 11.5, 11.6, 11.8, 11.7, 12.2, 12.3], dtype=float)
        low = np.array([9.9, 10.2, 10.8, 11.0, 11.1, 10.9, 11.2, 11.6], dtype=float)
        close = np.array([10.6, 10.9, 11.2, 11.3, 11.2, 11.5, 12.0, 12.1], dtype=float)
        volume = np.array([1000.0, 1200.0, 1800.0, 1600.0, 2100.0, 1700.0, 2400.0, 2200.0], dtype=float)

        actual = ta_indicators.normalized_volume_true_range(
            open_, high, low, close, volume, true_range_style="delta", outlier_range=4.0, atr_length=3, volume_length=2
        )
        expected = naive_normalized_volume_true_range(
            open_, high, low, close, volume, true_range_style="delta", outlier_range=4.0, atr_length=3, volume_length=2
        )

        for lhs, rhs in zip(actual, expected):
            np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_stream_matches_single(self, test_data):
        expected = ta_indicators.normalized_volume_true_range(
            test_data["open"],
            test_data["high"],
            test_data["low"],
            test_data["close"],
            test_data["volume"],
            true_range_style="hl",
            outlier_range=6.0,
            atr_length=9,
            volume_length=6,
        )
        stream = ta_indicators.NormalizedVolumeTrueRangeStream(
            true_range_style="hl",
            outlier_range=6.0,
            atr_length=9,
            volume_length=6,
        )

        values = [np.full(len(test_data["close"]), np.nan) for _ in range(5)]
        for idx, item in enumerate(
            zip(
                test_data["open"],
                test_data["high"],
                test_data["low"],
                test_data["close"],
                test_data["volume"],
            )
        ):
            result = stream.update(*map(float, item))
            if result is not None:
                for series, value in zip(values, result):
                    series[idx] = value

        for lhs, rhs in zip(values, expected):
            np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12, equal_nan=True)

        stream.reset()
        assert stream.update(np.nan, np.nan, np.nan, np.nan, np.nan) is None

    def test_batch_matches_single(self, test_data):
        single = ta_indicators.normalized_volume_true_range(
            test_data["open"],
            test_data["high"],
            test_data["low"],
            test_data["close"],
            test_data["volume"],
            outlier_range=5.0,
            atr_length=14,
            volume_length=14,
        )
        batch = ta_indicators.normalized_volume_true_range_batch(
            test_data["open"],
            test_data["high"],
            test_data["low"],
            test_data["close"],
            test_data["volume"],
            outlier_range_range=(5.0, 5.0, 0.0),
            atr_length_range=(14, 14, 0),
            volume_length_range=(14, 14, 0),
            true_range_style="body",
        )

        assert batch["rows"] == 1
        assert batch["cols"] == len(test_data["close"])
        for key, expected in zip(
            ["normalized_volume", "normalized_true_range", "baseline", "atr", "average_volume"],
            single,
        ):
            assert batch[key].shape == (1, len(test_data["close"]))
            assert_close(batch[key][0], expected, rtol=1e-12, atol=1e-12)

    def test_errors(self):
        sample = np.arange(8, dtype=float) + 10.0
        with pytest.raises(ValueError, match="invalid outlier_range|Invalid outlier_range"):
            ta_indicators.normalized_volume_true_range(
                sample, sample, sample, sample, sample, outlier_range=0.25
            )
        with pytest.raises(ValueError, match="invalid atr_length|Invalid atr_length"):
            ta_indicators.normalized_volume_true_range(
                sample, sample, sample, sample, sample, atr_length=1
            )
        with pytest.raises(ValueError, match="invalid volume_length|Invalid volume_length"):
            ta_indicators.normalized_volume_true_range(
                sample, sample, sample, sample, sample, volume_length=1
            )
        with pytest.raises(ValueError, match="empty|Empty"):
            ta_indicators.normalized_volume_true_range(
                np.array([], dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=float),
            )
        with pytest.raises(ValueError, match="all values are NaN|All values are NaN"):
            nan = np.full(32, np.nan)
            ta_indicators.normalized_volume_true_range(nan, nan, nan, nan, nan)
        with pytest.raises(ValueError, match="inconsistent slice lengths|Inconsistent slice lengths"):
            ta_indicators.normalized_volume_true_range(
                np.arange(5, dtype=float),
                np.arange(4, dtype=float),
                np.arange(5, dtype=float),
                np.arange(5, dtype=float),
                np.arange(5, dtype=float),
            )
