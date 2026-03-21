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


def manual_ttf(high: np.ndarray, low: np.ndarray, length: int) -> np.ndarray:
    out = np.full(high.shape[0], np.nan, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(high) & np.isfinite(low))
    if valid.size == 0:
        return out

    first = int(valid[0])
    if high.shape[0] - first < length:
        return out

    warm = first + length - 1
    hh_series = np.full(high.shape[0], np.nan, dtype=np.float64)
    ll_series = np.full(low.shape[0], np.nan, dtype=np.float64)

    for i in range(warm, high.shape[0]):
        start = i + 1 - length
        hh = np.max(high[start : i + 1])
        ll = np.min(low[start : i + 1])
        hh_series[i] = hh
        ll_series[i] = ll
        hist_hh = hh_series[i - length] if i >= length and np.isfinite(hh_series[i - length]) else 0.0
        hist_ll = ll_series[i - length] if i >= length and np.isfinite(ll_series[i - length]) else 0.0
        buy_power = hh - hist_ll
        sell_power = hist_hh - ll
        denom = buy_power + sell_power
        if np.isfinite(denom) and denom != 0.0:
            out[i] = 200.0 * (buy_power - sell_power) / denom

    return out


class TestTrendTriggerFactor:
    def test_manual_reference_matches_binding(self):
        data = load_test_data()
        high = np.asarray(data["high"][:160], dtype=np.float64)
        low = np.asarray(data["low"][:160], dtype=np.float64)
        expected = manual_ttf(high, low, 15)
        result = mp.trend_trigger_factor(high, low, length=15)
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-12, equal_nan=True)

    def test_stream_matches_batch(self):
        data = load_test_data()
        high = np.asarray(data["high"][:128], dtype=np.float64)
        low = np.asarray(data["low"][:128], dtype=np.float64)
        batch = mp.trend_trigger_factor(high, low, length=15)
        stream = mp.TrendTriggerFactorStream(15)
        streamed = np.array([stream.update(float(h), float(l)) for h, l in zip(high, low)])
        np.testing.assert_allclose(streamed, batch, rtol=0.0, atol=1e-12, equal_nan=True)

    def test_batch_metadata_and_first_row_match_single(self):
        data = load_test_data()
        high = np.asarray(data["high"][:128], dtype=np.float64)
        low = np.asarray(data["low"][:128], dtype=np.float64)
        result = mp.trend_trigger_factor_batch(high, low, length_range=(15, 17, 2))
        assert result["rows"] == 2
        assert result["cols"] == high.shape[0]
        np.testing.assert_array_equal(result["lengths"], np.array([15, 17], dtype=np.uint64))

        single = mp.trend_trigger_factor(high, low, length=15)
        np.testing.assert_allclose(result["values"][0], single, rtol=0.0, atol=1e-12, equal_nan=True)

    def test_kernel_parity(self):
        data = load_test_data()
        high = np.asarray(data["high"][:144], dtype=np.float64)
        low = np.asarray(data["low"][:144], dtype=np.float64)
        auto = mp.trend_trigger_factor(high, low, length=15)
        scalar = mp.trend_trigger_factor(high, low, length=15, kernel="scalar")
        np.testing.assert_allclose(auto, scalar, rtol=0.0, atol=1e-12, equal_nan=True)

    def test_invalid_length(self):
        data = load_test_data()
        high = np.asarray(data["high"][:32], dtype=np.float64)
        low = np.asarray(data["low"][:32], dtype=np.float64)
        with pytest.raises(ValueError, match="Invalid length"):
            mp.trend_trigger_factor(high, low, length=0)

    def test_length_mismatch(self):
        data = load_test_data()
        high = np.asarray(data["high"][:32], dtype=np.float64)
        low = np.asarray(data["low"][:31], dtype=np.float64)
        with pytest.raises(ValueError, match="Inconsistent slice lengths"):
            mp.trend_trigger_factor(high, low, length=15)
