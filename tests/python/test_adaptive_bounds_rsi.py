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


def manual_adaptive_bounds_rsi(data, rsi_length, alpha):
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[0]
    out = {
        "rsi": np.full(n, np.nan, dtype=np.float64),
        "lower_bound": np.full(n, np.nan, dtype=np.float64),
        "lower_mid": np.full(n, np.nan, dtype=np.float64),
        "mid": np.full(n, np.nan, dtype=np.float64),
        "upper_mid": np.full(n, np.nan, dtype=np.float64),
        "upper_bound": np.full(n, np.nan, dtype=np.float64),
        "regime": np.full(n, np.nan, dtype=np.float64),
        "regime_flip": np.full(n, np.nan, dtype=np.float64),
        "lower_signal": np.full(n, np.nan, dtype=np.float64),
        "upper_signal": np.full(n, np.nan, dtype=np.float64),
    }

    prev_price = np.nan
    seed_count = 0
    sum_gain = 0.0
    sum_loss = 0.0
    avg_gain = 0.0
    avg_loss = 0.0
    seeded = False
    inv_p = 1.0 / float(rsi_length)
    beta = 1.0 - inv_p

    c1, c2, c3, c4, c5 = 20.0, 40.0, 50.0, 60.0, 80.0
    prev_rsi = np.nan
    prev_c1 = np.nan
    prev_c5 = np.nan
    prev_regime = np.nan
    can_show_lower = True
    can_show_upper = True

    for i, value in enumerate(data):
        if not np.isfinite(value):
            prev_price = np.nan
            seed_count = 0
            sum_gain = 0.0
            sum_loss = 0.0
            avg_gain = 0.0
            avg_loss = 0.0
            seeded = False
            prev_rsi = np.nan
            prev_c1 = np.nan
            prev_c5 = np.nan
            prev_regime = np.nan
            continue

        if not np.isfinite(prev_price):
            prev_price = value
            continue

        delta = value - prev_price
        prev_price = value
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)

        if not seeded:
            sum_gain += gain
            sum_loss += loss
            seed_count += 1
            if seed_count != rsi_length:
                continue
            seeded = True
            avg_gain = sum_gain * inv_p
            avg_loss = sum_loss * inv_p
        else:
            avg_gain = avg_gain * beta + inv_p * gain
            avg_loss = avg_loss * beta + inv_p * loss

        denom = avg_gain + avg_loss
        rsi = 50.0 if denom == 0.0 else 100.0 * avg_gain / denom

        distances = [
            abs(rsi - c1),
            abs(rsi - c2),
            abs(rsi - c3),
            abs(rsi - c4),
            abs(rsi - c5),
        ]
        winner = int(np.argmin(distances))
        if winner == 0:
            c1 += (rsi - c1) * alpha
        elif winner == 1:
            c2 += (rsi - c2) * alpha
        elif winner == 2:
            c3 += (rsi - c3) * alpha
        elif winner == 3:
            c4 += (rsi - c4) * alpha
        else:
            c5 += (rsi - c5) * alpha

        regime = -2 if rsi <= c1 else -1 if rsi <= c2 else 0 if rsi <= c3 else 1 if rsi <= c4 else 2

        crossed_mid = np.isfinite(prev_rsi) and (
            (prev_rsi <= 50.0 and rsi > 50.0) or (prev_rsi >= 50.0 and rsi < 50.0)
        )
        if crossed_mid:
            can_show_lower = True
            can_show_upper = True

        lower_signal = (
            np.isfinite(prev_rsi)
            and np.isfinite(prev_c1)
            and prev_rsi >= prev_c1
            and rsi < c1
            and can_show_lower
        )
        upper_signal = (
            np.isfinite(prev_rsi)
            and np.isfinite(prev_c5)
            and prev_rsi <= prev_c5
            and rsi > c5
            and can_show_upper
        )

        if lower_signal:
            can_show_lower = False
        if upper_signal:
            can_show_upper = False

        regime_flip = np.isfinite(prev_regime) and prev_regime == 0 and regime != 0

        out["rsi"][i] = rsi
        out["lower_bound"][i] = c1
        out["lower_mid"][i] = c2
        out["mid"][i] = c3
        out["upper_mid"][i] = c4
        out["upper_bound"][i] = c5
        out["regime"][i] = float(regime)
        out["regime_flip"][i] = 1.0 if regime_flip else 0.0
        out["lower_signal"][i] = 1.0 if lower_signal else 0.0
        out["upper_signal"][i] = 1.0 if upper_signal else 0.0

        prev_rsi = rsi
        prev_c1 = c1
        prev_c5 = c5
        prev_regime = float(regime)

    return out


class TestAdaptiveBoundsRsi:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_manual_reference_matches_binding(self, test_data):
        close = np.asarray(test_data["close"][:180], dtype=np.float64)
        params = {"rsi_length": 14, "alpha": 0.1}
        result = ta.adaptive_bounds_rsi(close, **params)
        expected = manual_adaptive_bounds_rsi(close, **params)

        for key, values in expected.items():
            np.testing.assert_allclose(
                np.asarray(result[key], dtype=np.float64),
                values,
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )

    def test_matches_rust_reference_generator(self, test_data):
        close = np.asarray(test_data["close"], dtype=np.float64)
        params = {"rsi_length": 14, "alpha": 0.1}
        result = ta.adaptive_bounds_rsi(close, **params)
        compare_with_rust(
            "adaptive_bounds_rsi",
            result,
            "close",
            params,
            rtol=0.0,
            atol=1e-12,
        )

    def test_stream_matches_batch(self, test_data):
        close = np.asarray(test_data["close"][:144], dtype=np.float64)
        params = {"rsi_length": 14, "alpha": 0.1}
        batch = ta.adaptive_bounds_rsi(close, **params)
        stream = ta.AdaptiveBoundsRsiStream(**params)

        keys = [
            "rsi",
            "lower_bound",
            "lower_mid",
            "mid",
            "upper_mid",
            "upper_bound",
            "regime",
            "regime_flip",
            "lower_signal",
            "upper_signal",
        ]
        collected = {key: [] for key in keys}

        for value in close:
            point = stream.update(float(value))
            if point is None:
                for key in keys:
                    collected[key].append(np.nan)
            else:
                for index, key in enumerate(keys):
                    collected[key].append(point[index])

        for key in keys:
            np.testing.assert_allclose(
                np.asarray(collected[key], dtype=np.float64),
                np.asarray(batch[key], dtype=np.float64),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )

    def test_batch_metadata_and_first_row_match_single(self, test_data):
        close = np.asarray(test_data["close"][:128], dtype=np.float64)
        result = ta.adaptive_bounds_rsi_batch(
            close,
            rsi_length_range=(14, 16, 2),
            alpha_range=(0.1, 0.2, 0.1),
        )

        assert result["rows"] == 4
        assert result["cols"] == close.shape[0]
        np.testing.assert_array_equal(result["rsi_lengths"], np.array([14, 14, 16, 16], dtype=np.uint64))
        np.testing.assert_allclose(result["alphas"], np.array([0.1, 0.2, 0.1, 0.2], dtype=np.float64))

        single = ta.adaptive_bounds_rsi(close, rsi_length=14, alpha=0.1)
        for key in [
            "rsi",
            "lower_bound",
            "lower_mid",
            "mid",
            "upper_mid",
            "upper_bound",
            "regime",
            "regime_flip",
            "lower_signal",
            "upper_signal",
        ]:
            np.testing.assert_allclose(
                np.asarray(result[key][0], dtype=np.float64),
                np.asarray(single[key], dtype=np.float64),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )

    def test_invalid_inputs_raise(self, test_data):
        close = np.asarray(test_data["close"][:64], dtype=np.float64)

        with pytest.raises(ValueError, match="invalid rsi_length|Invalid rsi_length"):
            ta.adaptive_bounds_rsi(close, rsi_length=0)
        with pytest.raises(ValueError, match="invalid alpha|Invalid alpha"):
            ta.adaptive_bounds_rsi(close, alpha=0.0)
