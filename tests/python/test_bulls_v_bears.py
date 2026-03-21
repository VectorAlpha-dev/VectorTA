import numpy as np
import pytest

from test_utils import load_test_data

try:
    import vector_ta as mp
except ImportError:
    try:
        import vector_ta as mp
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


def ema(close, period):
    out = np.full(close.shape[0], np.nan, dtype=np.float64)
    alpha = 2.0 / (period + 1.0)
    prev = np.nan
    for i, x in enumerate(close):
        if not np.isfinite(x):
            prev = np.nan
            continue
        prev = x if not np.isfinite(prev) else prev + alpha * (x - prev)
        out[i] = prev
    return out


def manual_reference(
    high,
    low,
    close,
    period=14,
    ma_type="ema",
    calculation_method="normalized",
    normalized_bars_back=120,
    raw_rolling_period=50,
    raw_threshold_percentile=95.0,
    threshold_level=80.0,
):
    assert ma_type == "ema"
    n = close.shape[0]
    ma = ema(close, period)
    bull = np.where(np.isfinite(ma) & np.isfinite(high) & np.isfinite(low), high - ma, np.nan)
    bear = np.where(np.isfinite(ma) & np.isfinite(high) & np.isfinite(low), ma - low, np.nan)
    value = np.full(n, np.nan, dtype=np.float64)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    bullish_signal = np.full(n, np.nan, dtype=np.float64)
    bearish_signal = np.full(n, np.nan, dtype=np.float64)
    zero_cross_up = np.full(n, np.nan, dtype=np.float64)
    zero_cross_down = np.full(n, np.nan, dtype=np.float64)

    if calculation_method == "normalized":
        for i in range(n):
            upper[i] = threshold_level
            lower[i] = -threshold_level
            left = max(0, i - normalized_bars_back + 1)
            bull_window = bull[left : i + 1]
            bear_window = bear[left : i + 1]
            if not (np.isfinite(bull[i]) and np.isfinite(bear[i])):
                continue
            bull_window = bull_window[np.isfinite(bull_window)]
            bear_window = bear_window[np.isfinite(bear_window)]
            if bull_window.size == 0 or bear_window.size == 0:
                continue
            bull_range = bull_window.max() - bull_window.min()
            bear_range = bear_window.max() - bear_window.min()
            if bull_range > 0 and bear_range > 0:
                norm_bull = ((bull[i] - bull_window.min()) / bull_range - 0.5) * 100.0
                norm_bear = ((bear[i] - bear_window.min()) / bear_range - 0.5) * 100.0
                value[i] = norm_bull - norm_bear
    else:
        value = np.where(np.isfinite(bull) & np.isfinite(bear), bull - bear, np.nan)
        for i in range(n):
            left = max(0, i - raw_rolling_period + 1)
            window = value[left : i + 1]
            window = window[np.isfinite(window)]
            if window.size == 0:
                continue
            lo = window.min()
            hi = window.max()
            rng = hi - lo
            upper[i] = lo + rng * (raw_threshold_percentile / 100.0)
            lower[i] = lo + rng * ((100.0 - raw_threshold_percentile) / 100.0)

    prev_total = np.nan
    for i in range(n):
        if np.isfinite(value[i]) and np.isfinite(upper[i]) and np.isfinite(lower[i]):
            bullish_signal[i] = 1.0 if value[i] > upper[i] else 0.0
            bearish_signal[i] = 1.0 if value[i] < lower[i] else 0.0
            zero_cross_up[i] = 1.0 if np.isfinite(prev_total) and value[i] > 0 and prev_total <= 0 else 0.0
            zero_cross_down[i] = 1.0 if np.isfinite(prev_total) and value[i] < 0 and prev_total >= 0 else 0.0
            prev_total = value[i]

    return {
        "value": value,
        "bull": bull,
        "bear": bear,
        "ma": ma,
        "upper": upper,
        "lower": lower,
        "bullish_signal": bullish_signal,
        "bearish_signal": bearish_signal,
        "zero_cross_up": zero_cross_up,
        "zero_cross_down": zero_cross_down,
    }


class TestBullsVBears:
    def test_manual_reference_matches_binding_normalized(self):
        data = load_test_data()
        high = np.asarray(data["high"][:192], dtype=np.float64)
        low = np.asarray(data["low"][:192], dtype=np.float64)
        close = np.asarray(data["close"][:192], dtype=np.float64)
        expected = manual_reference(high, low, close)
        result = mp.bulls_v_bears(high, low, close)
        for key, values in expected.items():
            np.testing.assert_allclose(result[key], values, rtol=0.0, atol=1e-12, equal_nan=True)

    def test_stream_matches_batch_raw(self):
        data = load_test_data()
        high = np.asarray(data["high"][:160], dtype=np.float64)
        low = np.asarray(data["low"][:160], dtype=np.float64)
        close = np.asarray(data["close"][:160], dtype=np.float64)
        batch = mp.bulls_v_bears(
            high,
            low,
            close,
            calculation_method="raw",
        )
        stream = mp.BullsVBearsStream(calculation_method="raw")
        streamed = np.array(
            [stream.update(float(h), float(l), float(c)) for h, l, c in zip(high, low, close)],
            dtype=np.float64,
        )
        keys = [
            "value",
            "bull",
            "bear",
            "ma",
            "upper",
            "lower",
            "bullish_signal",
            "bearish_signal",
            "zero_cross_up",
            "zero_cross_down",
        ]
        for idx, key in enumerate(keys):
            np.testing.assert_allclose(streamed[:, idx], batch[key], rtol=0.0, atol=1e-12, equal_nan=True)

    def test_batch_metadata_and_first_row_match_single(self):
        data = load_test_data()
        high = np.asarray(data["high"][:144], dtype=np.float64)
        low = np.asarray(data["low"][:144], dtype=np.float64)
        close = np.asarray(data["close"][:144], dtype=np.float64)
        result = mp.bulls_v_bears_batch(
            high,
            low,
            close,
            period_range=(14, 16, 2),
            calculation_method="raw",
        )
        assert result["rows"] == 2
        assert result["cols"] == close.shape[0]
        np.testing.assert_allclose(result["periods"], np.array([14, 16]), rtol=0.0, atol=1e-12)
        single = mp.bulls_v_bears(high, low, close, period=14, calculation_method="raw")
        for key in [
            "value",
            "bull",
            "bear",
            "ma",
            "upper",
            "lower",
            "bullish_signal",
            "bearish_signal",
            "zero_cross_up",
            "zero_cross_down",
        ]:
            np.testing.assert_allclose(result[key][0], single[key], rtol=0.0, atol=1e-12, equal_nan=True)

    def test_invalid_threshold_level(self):
        data = load_test_data()
        high = np.asarray(data["high"][:64], dtype=np.float64)
        low = np.asarray(data["low"][:64], dtype=np.float64)
        close = np.asarray(data["close"][:64], dtype=np.float64)
        with pytest.raises(ValueError, match="Invalid threshold_level"):
            mp.bulls_v_bears(high, low, close, threshold_level=120.0)
