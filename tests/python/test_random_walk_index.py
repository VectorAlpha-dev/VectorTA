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


def manual_random_walk_index(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int):
    n = close.shape[0]
    out_high = np.full(n, np.nan, dtype=np.float64)
    out_low = np.full(n, np.nan, dtype=np.float64)

    valid = np.flatnonzero(np.isfinite(high) & np.isfinite(low) & np.isfinite(close))
    if valid.size == 0:
        return out_high, out_low

    first = int(valid[0])
    warm = first + length - 1
    alpha = 1.0 / float(length)
    sqrt_length = np.sqrt(float(length))

    prev_close = close[first]
    sum_tr = high[first] - low[first]
    atr = np.nan

    if length == 1:
        atr = sum_tr
        denom = atr * sqrt_length
        if np.isfinite(denom) and denom != 0.0:
            out_high[first] = high[first] / denom
            out_low[first] = -low[first] / denom

    for i in range(first + 1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - prev_close),
            abs(low[i] - prev_close),
        )
        if i <= warm:
            sum_tr += tr
            if i == warm:
                atr = sum_tr / float(length)
        else:
            atr = atr + alpha * (tr - atr)

        if i >= warm:
            hist_low = low[i - length] if i >= length and np.isfinite(low[i - length]) else 0.0
            hist_high = high[i - length] if i >= length and np.isfinite(high[i - length]) else 0.0
            denom = atr * sqrt_length
            if np.isfinite(denom) and denom != 0.0:
                out_high[i] = (high[i] - hist_low) / denom
                out_low[i] = (hist_high - low[i]) / denom

        prev_close = close[i]

    return out_high, out_low


class TestRandomWalkIndex:
    def test_manual_reference_matches_binding(self):
        data = load_test_data()
        high = np.asarray(data["high"][:160], dtype=np.float64)
        low = np.asarray(data["low"][:160], dtype=np.float64)
        close = np.asarray(data["close"][:160], dtype=np.float64)
        expected_high, expected_low = manual_random_walk_index(high, low, close, 14)
        result = mp.random_walk_index(high, low, close, length=14)
        np.testing.assert_allclose(result["high"], expected_high, rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(result["low"], expected_low, rtol=0.0, atol=1e-12, equal_nan=True)

    def test_stream_matches_batch(self):
        data = load_test_data()
        high = np.asarray(data["high"][:128], dtype=np.float64)
        low = np.asarray(data["low"][:128], dtype=np.float64)
        close = np.asarray(data["close"][:128], dtype=np.float64)
        batch = mp.random_walk_index(high, low, close, length=14)
        stream = mp.RandomWalkIndexStream(14)
        streamed = np.array([stream.update(float(h), float(l), float(c)) for h, l, c in zip(high, low, close)])
        np.testing.assert_allclose(streamed[:, 0], batch["high"], rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(streamed[:, 1], batch["low"], rtol=0.0, atol=1e-12, equal_nan=True)

    def test_batch_metadata_and_first_row_match_single(self):
        data = load_test_data()
        high = np.asarray(data["high"][:128], dtype=np.float64)
        low = np.asarray(data["low"][:128], dtype=np.float64)
        close = np.asarray(data["close"][:128], dtype=np.float64)
        result = mp.random_walk_index_batch(high, low, close, length_range=(14, 16, 2))
        assert result["rows"] == 2
        assert result["cols"] == high.shape[0]
        np.testing.assert_array_equal(result["lengths"], np.array([14, 16], dtype=np.uint64))

        single = mp.random_walk_index(high, low, close, length=14)
        np.testing.assert_allclose(result["high"][0], single["high"], rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(result["low"][0], single["low"], rtol=0.0, atol=1e-12, equal_nan=True)

    def test_kernel_parity(self):
        data = load_test_data()
        high = np.asarray(data["high"][:144], dtype=np.float64)
        low = np.asarray(data["low"][:144], dtype=np.float64)
        close = np.asarray(data["close"][:144], dtype=np.float64)
        auto = mp.random_walk_index(high, low, close, length=14)
        scalar = mp.random_walk_index(high, low, close, length=14, kernel="scalar")
        np.testing.assert_allclose(auto["high"], scalar["high"], rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(auto["low"], scalar["low"], rtol=0.0, atol=1e-12, equal_nan=True)

    def test_invalid_length(self):
        data = load_test_data()
        high = np.asarray(data["high"][:32], dtype=np.float64)
        low = np.asarray(data["low"][:32], dtype=np.float64)
        close = np.asarray(data["close"][:32], dtype=np.float64)
        with pytest.raises(ValueError, match="Invalid length"):
            mp.random_walk_index(high, low, close, length=0)

    def test_length_mismatch(self):
        data = load_test_data()
        high = np.asarray(data["high"][:32], dtype=np.float64)
        low = np.asarray(data["low"][:31], dtype=np.float64)
        close = np.asarray(data["close"][:32], dtype=np.float64)
        with pytest.raises(ValueError, match="Inconsistent slice lengths"):
            mp.random_walk_index(high, low, close, length=14)
