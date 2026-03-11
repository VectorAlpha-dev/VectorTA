import numpy as np
import pytest

from rust_comparison import compare_with_rust
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


def _build_weights(length: int, alpha: float, beta: float) -> np.ndarray:
    x = np.arange(length, dtype=np.float64) / float(length - 1)
    weights = np.sin((2.0 * np.pi) * np.power(x, alpha)) * (1.0 - np.power(x, beta))
    return weights / weights.sum()


def _weighted_filter(source: np.ndarray, first: int, length: int, weights: np.ndarray) -> np.ndarray:
    out = np.full(source.shape[0], np.nan, dtype=np.float64)
    start = first + length - 1
    for i in range(start, source.shape[0]):
        acc = 0.0
        for j in range(length):
            acc += source[i - j] * weights[j]
        out[i] = acc
    return out


def _pine_cross(prev_a: float, prev_b: float, curr_a: float, curr_b: float) -> bool:
    if not all(np.isfinite([prev_a, prev_b, curr_a, curr_b])):
        return False
    return (curr_a > curr_b and prev_a <= prev_b) or (curr_a < curr_b and prev_a >= prev_b)


def manual_adjustable_ma_alternating_extremities(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int,
    mult: float,
    alpha: float,
    beta: float,
):
    n = close.shape[0]
    out = {
        "ma": np.full(n, np.nan, dtype=np.float64),
        "upper": np.full(n, np.nan, dtype=np.float64),
        "lower": np.full(n, np.nan, dtype=np.float64),
        "extremity": np.full(n, np.nan, dtype=np.float64),
        "state": np.full(n, np.nan, dtype=np.float64),
        "changed": np.full(n, np.nan, dtype=np.float64),
        "smoothed_open": np.full(n, np.nan, dtype=np.float64),
        "smoothed_high": np.full(n, np.nan, dtype=np.float64),
        "smoothed_low": np.full(n, np.nan, dtype=np.float64),
        "smoothed_close": np.full(n, np.nan, dtype=np.float64),
    }

    valid = np.flatnonzero(np.isfinite(high) & np.isfinite(low) & np.isfinite(close))
    if valid.size == 0:
        return out
    first = int(valid[0])
    weights = _build_weights(length, alpha, beta)
    ma_warm = first + length - 1
    band_warm = first + (2 * length) - 2

    out["ma"] = _weighted_filter(close, first, length, weights)
    out["smoothed_high"] = _weighted_filter(high, first, length, weights)
    out["smoothed_low"] = _weighted_filter(low, first, length, weights)
    out["smoothed_close"] = out["ma"].copy()

    for i in range(ma_warm + 2, n):
        out["smoothed_open"][i] = 0.5 * (out["smoothed_close"][i - 1] + out["smoothed_close"][i - 2])

    rolling = 0.0
    for i in range(ma_warm, band_warm + 1):
        rolling += abs(close[i] - out["ma"][i])
    dev = (rolling / float(length)) * mult
    out["upper"][band_warm] = out["ma"][band_warm] + dev
    out["lower"][band_warm] = out["ma"][band_warm] - dev

    for i in range(band_warm + 1, n):
        rolling += abs(close[i] - out["ma"][i])
        rolling -= abs(close[i - length] - out["ma"][i - length])
        dev = (rolling / float(length)) * mult
        out["upper"][i] = out["ma"][i] + dev
        out["lower"][i] = out["ma"][i] - dev

    out["state"][band_warm] = 0.0
    out["changed"][band_warm] = 0.0
    out["extremity"][band_warm] = out["lower"][band_warm]

    for i in range(band_warm + 1, n):
        prev_state = out["state"][i - 1]
        cross_high = _pine_cross(high[i - 1], out["upper"][i - 1], high[i], out["upper"][i])
        cross_low = _pine_cross(low[i - 1], out["lower"][i - 1], low[i], out["lower"][i])
        next_state = 1.0 if cross_high else 0.0 if cross_low else prev_state
        out["state"][i] = next_state
        out["changed"][i] = 1.0 if abs(next_state - prev_state) > 0.0 else 0.0
        out["extremity"][i] = out["upper"][i] if next_state >= 0.5 else out["lower"][i]

    return out


class TestAdjustableMaAlternatingExtremities:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_manual_reference_matches_binding(self, test_data):
        n = 160
        high = np.asarray(test_data["high"][:n], dtype=np.float64)
        low = np.asarray(test_data["low"][:n], dtype=np.float64)
        close = np.asarray(test_data["close"][:n], dtype=np.float64)
        params = {"length": 12, "mult": 2.4, "alpha": 1.3, "beta": 0.7}

        result = mp.adjustable_ma_alternating_extremities(high, low, close, **params)
        expected = manual_adjustable_ma_alternating_extremities(high, low, close, **params)

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
        params = {"length": 50, "mult": 2.0, "alpha": 1.0, "beta": 0.5}

        result = mp.adjustable_ma_alternating_extremities(high, low, close, **params)
        compare_with_rust(
            "adjustable_ma_alternating_extremities",
            result,
            "ohlc",
            params,
            rtol=0.0,
            atol=1e-12,
        )

    def test_stream_matches_batch(self, test_data):
        n = 144
        high = np.asarray(test_data["high"][:n], dtype=np.float64)
        low = np.asarray(test_data["low"][:n], dtype=np.float64)
        close = np.asarray(test_data["close"][:n], dtype=np.float64)
        params = {"length": 10, "mult": 2.2, "alpha": 1.1, "beta": 0.6}

        batch = mp.adjustable_ma_alternating_extremities(high, low, close, **params)
        stream = mp.AdjustableMaAlternatingExtremitiesStream(**params)
        streamed = {key: [] for key in batch.keys()}

        keys = list(streamed.keys())
        for h, l, c in zip(high, low, close):
            value = stream.update(float(h), float(l), float(c))
            if value is None:
                for key in keys:
                    streamed[key].append(np.nan)
            else:
                for key, item in zip(keys, value):
                    streamed[key].append(item)

        for i in range(n):
            if np.isnan(streamed["upper"][i]):
                assert np.isnan(batch["upper"][i])
                assert np.isnan(batch["lower"][i])
                assert np.isnan(batch["extremity"][i])
                continue
            for key in keys:
                np.testing.assert_allclose(
                    float(streamed[key][i]),
                    float(batch[key][i]),
                    rtol=0.0,
                    atol=1e-12,
                    equal_nan=True,
                )

    def test_batch_metadata_and_first_row_match_single(self, test_data):
        n = 128
        high = np.asarray(test_data["high"][:n], dtype=np.float64)
        low = np.asarray(test_data["low"][:n], dtype=np.float64)
        close = np.asarray(test_data["close"][:n], dtype=np.float64)

        result = mp.adjustable_ma_alternating_extremities_batch(
            high,
            low,
            close,
            length_range=(10, 12, 2),
            mult_range=(2.0, 2.5, 0.5),
            alpha_range=(1.0, 1.0, 0.0),
            beta_range=(0.5, 0.5, 0.0),
        )

        assert result["rows"] == 4
        assert result["cols"] == n
        np.testing.assert_array_equal(result["lengths"], np.array([10, 10, 12, 12], dtype=np.uint64))
        np.testing.assert_allclose(result["mults"], np.array([2.0, 2.5, 2.0, 2.5]))
        np.testing.assert_allclose(result["alphas"], np.ones(4))
        np.testing.assert_allclose(result["betas"], np.full(4, 0.5))

        single = mp.adjustable_ma_alternating_extremities(
            high, low, close, length=10, mult=2.0, alpha=1.0, beta=0.5
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
        high = np.asarray(test_data["high"][:64], dtype=np.float64)
        low = np.asarray(test_data["low"][:64], dtype=np.float64)
        close = np.asarray(test_data["close"][:64], dtype=np.float64)

        with pytest.raises(ValueError, match="invalid length|Invalid length"):
            mp.adjustable_ma_alternating_extremities(high, low, close, length=1)
        with pytest.raises(ValueError, match="invalid mult|Invalid mult"):
            mp.adjustable_ma_alternating_extremities(high, low, close, mult=0.5)
        with pytest.raises(ValueError, match="invalid alpha|Invalid alpha"):
            mp.adjustable_ma_alternating_extremities(high, low, close, alpha=-1.0)
        with pytest.raises(ValueError, match="invalid beta|Invalid beta"):
            mp.adjustable_ma_alternating_extremities(high, low, close, beta=-0.1)
