import numpy as np
import pytest

from rust_comparison import compare_with_rust
from test_utils import load_test_data

try:
    import vector_ta as ta
except ImportError:
    try:
        import vector_ta as ta
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


def manual_range_filtered_trend_signals(
    high,
    low,
    close,
    kalman_alpha=0.01,
    kalman_beta=0.1,
    kalman_period=77,
    dev=1.2,
    supertrend_factor=0.7,
    supertrend_atr_period=7,
):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = close.shape[0]
    out = {
        "kalman": np.full(n, np.nan, dtype=np.float64),
        "supertrend": np.full(n, np.nan, dtype=np.float64),
        "upper_band": np.full(n, np.nan, dtype=np.float64),
        "lower_band": np.full(n, np.nan, dtype=np.float64),
        "trend": np.full(n, np.nan, dtype=np.float64),
        "kalman_trend": np.full(n, np.nan, dtype=np.float64),
        "state": np.full(n, np.nan, dtype=np.float64),
        "market_trending": np.full(n, np.nan, dtype=np.float64),
        "market_ranging": np.full(n, np.nan, dtype=np.float64),
        "short_term_bullish": np.full(n, np.nan, dtype=np.float64),
        "short_term_bearish": np.full(n, np.nan, dtype=np.float64),
        "long_term_bullish": np.full(n, np.nan, dtype=np.float64),
        "long_term_bearish": np.full(n, np.nan, dtype=np.float64),
    }

    wbuf = []
    wsum = 0.0
    wweighted = 0.0
    wdiv = WMA_DIVISOR = (200 * 201) / 2.0
    atr_sum = 0.0
    atr_count = 0
    atr_val = None
    atr_prev_close = np.nan
    k_val = np.nan
    k_cov = 1.0
    prev_close = np.nan
    prev_lower_band = np.nan
    prev_upper_band = np.nan
    prev_supertrend = np.nan
    prev_k = np.nan
    prev_atr_ready = False
    trend_state = 0.0
    prev_trend = None
    prev_ktrend = None
    prev_state = None

    for i in range(n):
        h = high[i]
        l = low[i]
        c = close[i]
        gain = k_cov / (k_cov + kalman_alpha * kalman_period)
        if np.isnan(k_val) and np.isfinite(prev_close):
            k_val = prev_close
        kalman = np.nan
        if np.isfinite(k_val):
            k_val = k_val + gain * (c - k_val)
            kalman = k_val
        k_cov = (1.0 - gain) * k_cov + kalman_beta / kalman_period
        prev_close = c

        tr = h - l if not np.isfinite(atr_prev_close) else max(h - l, abs(h - atr_prev_close), abs(l - atr_prev_close))
        atr_prev_close = c
        if atr_val is None:
            atr_count += 1
            atr_sum += tr
            if atr_count == supertrend_atr_period:
                atr_val = atr_sum / supertrend_atr_period
        else:
            atr_val = ((atr_val * (supertrend_atr_period - 1)) + tr) / supertrend_atr_period

        width = h - l
        if len(wbuf) < 200:
            wbuf.append(width)
            wsum += width
            wweighted += len(wbuf) * width
            vola = wweighted / wdiv if len(wbuf) == 200 else None
        else:
            oldest = wbuf.pop(0)
            old_sum = wsum
            wbuf.append(width)
            wweighted = wweighted - old_sum + 200.0 * width
            wsum = old_sum - oldest + width
            vola = wweighted / wdiv

        supertrend = None
        direction = None
        if np.isfinite(kalman) and atr_val is not None:
            upper = kalman + supertrend_factor * atr_val
            lower = kalman - supertrend_factor * atr_val
            pl = lower if np.isnan(prev_lower_band) else prev_lower_band
            pu = upper if np.isnan(prev_upper_band) else prev_upper_band
            pk = kalman if np.isnan(prev_k) else prev_k
            if not (lower > pl or pk < pl):
                lower = pl
            if not (upper < pu or pk > pu):
                upper = pu
            if not prev_atr_ready:
                direction = 1
            elif prev_supertrend == pu:
                direction = -1 if kalman > upper else 1
            else:
                direction = 1 if kalman < lower else -1
            supertrend = lower if direction == -1 else upper
            prev_lower_band = lower
            prev_upper_band = upper
            prev_supertrend = supertrend
            prev_k = kalman
            prev_atr_ready = True

        if vola is None or supertrend is None:
            continue

        upper_band = kalman + vola * dev
        lower_band = kalman - vola * dev
        if c > upper_band:
            trend_state = 1.0
        elif c < lower_band:
            trend_state = -1.0

        ktrend = 1.0 if direction < 0 else -1.0
        state = trend_state * ktrend
        out["kalman"][i] = kalman
        out["supertrend"][i] = supertrend
        out["upper_band"][i] = upper_band
        out["lower_band"][i] = lower_band
        out["trend"][i] = trend_state
        out["kalman_trend"][i] = ktrend
        out["state"][i] = state
        out["market_trending"][i] = 1.0 if prev_state is not None and state > 0 and prev_state <= 0 else 0.0
        out["market_ranging"][i] = 1.0 if prev_state is not None and state < 0 and prev_state >= 0 else 0.0
        out["short_term_bullish"][i] = 1.0 if prev_trend is not None and trend_state > 0 and prev_trend <= 0 else 0.0
        out["short_term_bearish"][i] = 1.0 if prev_trend is not None and trend_state < 0 and prev_trend >= 0 else 0.0
        out["long_term_bullish"][i] = 1.0 if prev_ktrend is not None and ktrend > 0 and prev_ktrend <= 0 else 0.0
        out["long_term_bearish"][i] = 1.0 if prev_ktrend is not None and ktrend < 0 and prev_ktrend >= 0 else 0.0
        prev_trend = trend_state
        prev_ktrend = ktrend
        prev_state = state

    return out


class TestRangeFilteredTrendSignals:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_manual_reference_matches_binding(self, test_data):
        high = np.asarray(test_data["high"][:360], dtype=np.float64)
        low = np.asarray(test_data["low"][:360], dtype=np.float64)
        close = np.asarray(test_data["close"][:360], dtype=np.float64)
        params = {
            "kalman_alpha": 0.01,
            "kalman_beta": 0.1,
            "kalman_period": 77,
            "dev": 1.2,
            "supertrend_factor": 0.7,
            "supertrend_atr_period": 7,
        }
        result = ta.range_filtered_trend_signals(high, low, close, **params)
        expected = manual_range_filtered_trend_signals(high, low, close, **params)
        for key, values in expected.items():
            np.testing.assert_allclose(
                np.asarray(result[key], dtype=np.float64),
                values,
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )

    def test_matches_rust_reference_generator(self, test_data):
        high = np.asarray(test_data["high"], dtype=np.float64)
        low = np.asarray(test_data["low"], dtype=np.float64)
        close = np.asarray(test_data["close"], dtype=np.float64)
        params = {
            "kalman_alpha": 0.01,
            "kalman_beta": 0.1,
            "kalman_period": 77,
            "dev": 1.2,
            "supertrend_factor": 0.7,
            "supertrend_atr_period": 7,
        }
        result = ta.range_filtered_trend_signals(high, low, close, **params)
        compare_with_rust("range_filtered_trend_signals", result, "ohlc", params, rtol=0.0, atol=1e-12)

    def test_stream_matches_batch(self, test_data):
        high = np.asarray(test_data["high"][:360], dtype=np.float64)
        low = np.asarray(test_data["low"][:360], dtype=np.float64)
        close = np.asarray(test_data["close"][:360], dtype=np.float64)
        params = {
            "kalman_alpha": 0.01,
            "kalman_beta": 0.1,
            "kalman_period": 77,
            "dev": 1.2,
            "supertrend_factor": 0.7,
            "supertrend_atr_period": 7,
        }
        batch = ta.range_filtered_trend_signals(high, low, close, **params)
        stream = ta.RangeFilteredTrendSignalsStream(**params)
        keys = list(batch.keys())
        collected = {key: [] for key in keys}
        for h, l, c in zip(high, low, close):
            point = stream.update(float(h), float(l), float(c))
            if point is None:
                for key in keys:
                    collected[key].append(np.nan)
            else:
                for idx, key in enumerate(keys):
                    collected[key].append(point[idx])
        for key in keys:
            np.testing.assert_allclose(
                np.asarray(collected[key], dtype=np.float64),
                np.asarray(batch[key], dtype=np.float64),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )

    def test_batch_metadata_and_first_row_match_single(self, test_data):
        high = np.asarray(test_data["high"][:360], dtype=np.float64)
        low = np.asarray(test_data["low"][:360], dtype=np.float64)
        close = np.asarray(test_data["close"][:360], dtype=np.float64)
        result = ta.range_filtered_trend_signals_batch(
            high,
            low,
            close,
            kalman_alpha_range=(0.01, 0.02, 0.01),
            kalman_beta_range=(0.1, 0.1, 0.0),
            kalman_period_range=(77, 77, 0),
            dev_range=(1.2, 1.2, 0.0),
            supertrend_factor_range=(0.7, 0.7, 0.0),
            supertrend_atr_period_range=(7, 7, 0),
        )
        assert result["rows"] == 2
        assert result["cols"] == close.shape[0]
        np.testing.assert_allclose(result["kalman_alphas"], np.array([0.01, 0.02], dtype=np.float64))
        single = ta.range_filtered_trend_signals(
            high,
            low,
            close,
            kalman_alpha=0.01,
            kalman_beta=0.1,
            kalman_period=77,
            dev=1.2,
            supertrend_factor=0.7,
            supertrend_atr_period=7,
        )
        for key in single.keys():
            np.testing.assert_allclose(
                np.asarray(result[key][0], dtype=np.float64),
                np.asarray(single[key], dtype=np.float64),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )

    def test_invalid_inputs_raise(self, test_data):
        high = np.asarray(test_data["high"][:256], dtype=np.float64)
        low = np.asarray(test_data["low"][:256], dtype=np.float64)
        close = np.asarray(test_data["close"][:256], dtype=np.float64)
        with pytest.raises(ValueError, match="invalid kalman_alpha|InvalidKalmanAlpha|invalid kalman_alpha"):
            ta.range_filtered_trend_signals(high, low, close, kalman_alpha=0.0)
        with pytest.raises(ValueError, match="invalid dev|InvalidDev|invalid dev"):
            ta.range_filtered_trend_signals(high, low, close, dev=-1.0)
