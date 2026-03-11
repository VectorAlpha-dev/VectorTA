import numpy as np
import pytest

from test_utils import load_test_data

try:
    import vector_ta as mp
except ImportError:
    try:
        import my_project as mp
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


def manual_reference(open_, high, low, close, threshold_level=0.35):
    n = close.shape[0]
    value = np.full(n, np.nan, dtype=np.float64)
    ema = np.full(n, np.nan, dtype=np.float64)
    signal = np.full(n, np.nan, dtype=np.float64)
    alpha = 2.0 / 15.0
    prev_ema = np.nan
    prev_open = np.nan
    prev_high = np.nan
    prev_low = np.nan
    prev_close = np.nan
    has_prev = False

    for i in range(n):
        if not (
            np.isfinite(open_[i])
            and np.isfinite(high[i])
            and np.isfinite(low[i])
            and np.isfinite(close[i])
        ):
            continue

        ema[i] = close[i] if not np.isfinite(prev_ema) else prev_ema + alpha * (close[i] - prev_ema)
        if has_prev:
            spread = prev_high - prev_low
            value[i] = abs(prev_open - prev_close) / spread if spread != 0.0 else 0.0
        else:
            value[i] = 0.0

        if value[i] > threshold_level and close[i] > ema[i]:
            signal[i] = 2.0
        elif value[i] > threshold_level and close[i] < ema[i]:
            signal[i] = -2.0
        elif close[i] > ema[i]:
            signal[i] = 1.0
        elif close[i] < ema[i]:
            signal[i] = -1.0
        else:
            signal[i] = 0.0

        prev_open = open_[i]
        prev_high = high[i]
        prev_low = low[i]
        prev_close = close[i]
        prev_ema = ema[i]
        has_prev = True

    return {"value": value, "ema": ema, "signal": signal}


class TestDailyFactor:
    def test_manual_reference_matches_binding(self):
        data = load_test_data()
        open_ = np.asarray(data["open"][:192], dtype=np.float64)
        high = np.asarray(data["high"][:192], dtype=np.float64)
        low = np.asarray(data["low"][:192], dtype=np.float64)
        close = np.asarray(data["close"][:192], dtype=np.float64)
        expected = manual_reference(open_, high, low, close, 0.35)
        result = mp.daily_factor(open_, high, low, close, threshold_level=0.35)
        np.testing.assert_allclose(result["value"], expected["value"], rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(result["ema"], expected["ema"], rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(result["signal"], expected["signal"], rtol=0.0, atol=1e-12, equal_nan=True)

    def test_stream_matches_batch(self):
        data = load_test_data()
        open_ = np.asarray(data["open"][:160], dtype=np.float64)
        high = np.asarray(data["high"][:160], dtype=np.float64)
        low = np.asarray(data["low"][:160], dtype=np.float64)
        close = np.asarray(data["close"][:160], dtype=np.float64)
        batch = mp.daily_factor(open_, high, low, close, threshold_level=0.35)
        stream = mp.DailyFactorStream(0.35)
        streamed = np.array(
            [stream.update(float(o), float(h), float(l), float(c)) for o, h, l, c in zip(open_, high, low, close)],
            dtype=np.float64,
        )
        np.testing.assert_allclose(streamed[:, 0], batch["value"], rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(streamed[:, 1], batch["ema"], rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(streamed[:, 2], batch["signal"], rtol=0.0, atol=1e-12, equal_nan=True)

    def test_batch_metadata_and_first_row_match_single(self):
        data = load_test_data()
        open_ = np.asarray(data["open"][:144], dtype=np.float64)
        high = np.asarray(data["high"][:144], dtype=np.float64)
        low = np.asarray(data["low"][:144], dtype=np.float64)
        close = np.asarray(data["close"][:144], dtype=np.float64)
        result = mp.daily_factor_batch(
            open_,
            high,
            low,
            close,
            threshold_level_range=(0.35, 0.45, 0.10),
        )
        assert result["rows"] == 2
        assert result["cols"] == close.shape[0]
        np.testing.assert_allclose(result["threshold_levels"], np.array([0.35, 0.45]), rtol=0.0, atol=1e-12)
        single = mp.daily_factor(open_, high, low, close, threshold_level=0.35)
        np.testing.assert_allclose(result["value"][0], single["value"], rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(result["ema"][0], single["ema"], rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(result["signal"][0], single["signal"], rtol=0.0, atol=1e-12, equal_nan=True)

    def test_invalid_threshold_level(self):
        data = load_test_data()
        open_ = np.asarray(data["open"][:64], dtype=np.float64)
        high = np.asarray(data["high"][:64], dtype=np.float64)
        low = np.asarray(data["low"][:64], dtype=np.float64)
        close = np.asarray(data["close"][:64], dtype=np.float64)
        with pytest.raises(ValueError, match="Invalid threshold_level"):
            mp.daily_factor(open_, high, low, close, threshold_level=1.5)
