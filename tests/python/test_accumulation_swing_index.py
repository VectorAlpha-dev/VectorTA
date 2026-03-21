import numpy as np
import pytest

from test_utils import load_test_data

try:
    import vector_ta as mp
except ImportError:
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python' first",
        allow_module_level=True,
    )


def manual_accumulation_swing_index(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    daily_limit: float,
) -> np.ndarray:
    n = close.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)

    valid = np.flatnonzero(
        np.isfinite(open_) & np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
    )
    if valid.size == 0:
        return out

    first = int(valid[0])
    out[first] = 0.0
    accum = 0.0
    prev_open = float(open_[first])
    prev_close = float(close[first])

    for i in range(first + 1, n):
        o = float(open_[i])
        h = float(high[i])
        l = float(low[i])
        c = float(close[i])
        if (
            np.isfinite(o)
            and np.isfinite(h)
            and np.isfinite(l)
            and np.isfinite(c)
            and np.isfinite(prev_open)
            and np.isfinite(prev_close)
        ):
            abs_high_close = abs(h - prev_close)
            abs_low_close = abs(l - prev_close)
            abs_close_open = abs(prev_close - prev_open)
            k = abs_high_close if abs_high_close >= abs_low_close else abs_low_close
            bar_range = h - l
            if abs_high_close >= abs_low_close:
                if abs_high_close >= bar_range:
                    r = abs_high_close - 0.5 * abs_low_close + 0.25 * abs_close_open
                else:
                    r = bar_range + 0.25 * abs_close_open
            elif abs_low_close >= bar_range:
                r = abs_low_close - 0.5 * abs_high_close + 0.25 * abs_close_open
            else:
                r = bar_range + 0.25 * abs_close_open
            if r != 0.0:
                accum += (
                    50.0
                    * (
                        (
                            (c - prev_close)
                            + 0.5 * (c - o)
                            + 0.25 * (prev_close - prev_open)
                        )
                        / r
                    )
                    * k
                    / daily_limit
                )
        out[i] = accum
        prev_open = o
        prev_close = c

    return out


class TestAccumulationSwingIndex:
    def test_manual_reference_matches_binding(self):
        data = load_test_data()
        open_ = np.asarray(data["open"][:160], dtype=np.float64)
        high = np.asarray(data["high"][:160], dtype=np.float64)
        low = np.asarray(data["low"][:160], dtype=np.float64)
        close = np.asarray(data["close"][:160], dtype=np.float64)
        expected = manual_accumulation_swing_index(open_, high, low, close, 10000.0)
        result = mp.accumulation_swing_index(open_, high, low, close, daily_limit=10000.0)
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-12, equal_nan=True)

    def test_stream_matches_batch(self):
        data = load_test_data()
        open_ = np.asarray(data["open"][:128], dtype=np.float64)
        high = np.asarray(data["high"][:128], dtype=np.float64)
        low = np.asarray(data["low"][:128], dtype=np.float64)
        close = np.asarray(data["close"][:128], dtype=np.float64)
        batch = mp.accumulation_swing_index(open_, high, low, close, daily_limit=10000.0)
        stream = mp.AccumulationSwingIndexStream(10000.0)
        streamed = np.array(
            [stream.update(float(o), float(h), float(l), float(c)) for o, h, l, c in zip(open_, high, low, close)]
        )
        np.testing.assert_allclose(streamed, batch, rtol=0.0, atol=1e-12, equal_nan=True)

    def test_batch_metadata_and_first_row_match_single(self):
        data = load_test_data()
        open_ = np.asarray(data["open"][:128], dtype=np.float64)
        high = np.asarray(data["high"][:128], dtype=np.float64)
        low = np.asarray(data["low"][:128], dtype=np.float64)
        close = np.asarray(data["close"][:128], dtype=np.float64)
        result = mp.accumulation_swing_index_batch(
            open_, high, low, close, daily_limit_range=(10000.0, 12000.0, 2000.0)
        )
        assert result["rows"] == 2
        assert result["cols"] == open_.shape[0]
        np.testing.assert_allclose(result["daily_limits"], np.array([10000.0, 12000.0]))

        single = mp.accumulation_swing_index(open_, high, low, close, daily_limit=10000.0)
        np.testing.assert_allclose(result["values"][0], single, rtol=0.0, atol=1e-12, equal_nan=True)

    def test_kernel_parity(self):
        data = load_test_data()
        open_ = np.asarray(data["open"][:144], dtype=np.float64)
        high = np.asarray(data["high"][:144], dtype=np.float64)
        low = np.asarray(data["low"][:144], dtype=np.float64)
        close = np.asarray(data["close"][:144], dtype=np.float64)
        auto = mp.accumulation_swing_index(open_, high, low, close, daily_limit=10000.0)
        scalar = mp.accumulation_swing_index(
            open_, high, low, close, daily_limit=10000.0, kernel="scalar"
        )
        np.testing.assert_allclose(auto, scalar, rtol=0.0, atol=1e-12, equal_nan=True)

    def test_invalid_daily_limit(self):
        data = load_test_data()
        open_ = np.asarray(data["open"][:32], dtype=np.float64)
        high = np.asarray(data["high"][:32], dtype=np.float64)
        low = np.asarray(data["low"][:32], dtype=np.float64)
        close = np.asarray(data["close"][:32], dtype=np.float64)
        with pytest.raises(ValueError, match="Invalid daily_limit"):
            mp.accumulation_swing_index(open_, high, low, close, daily_limit=0.0)

    def test_length_mismatch(self):
        data = load_test_data()
        open_ = np.asarray(data["open"][:32], dtype=np.float64)
        high = np.asarray(data["high"][:31], dtype=np.float64)
        low = np.asarray(data["low"][:32], dtype=np.float64)
        close = np.asarray(data["close"][:32], dtype=np.float64)
        with pytest.raises(ValueError, match="Inconsistent slice lengths"):
            mp.accumulation_swing_index(open_, high, low, close, daily_limit=10000.0)
