import math
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


def _weights(size):
    center = (size - 1) * 0.5
    width = (size - 0.5) * (1024 ** (-0.5)) + 0.5
    values = np.array(
        [math.exp(-(((i - center) / width) ** 2) / 4.0) / math.sqrt(2.0 * math.pi) for i in range(size)],
        dtype=np.float64,
    )
    return values / values.sum()


def _estimate(values, style):
    data = np.asarray(values, dtype=np.float64)
    if style == "mean":
        return float(data.mean())
    data = np.sort(data)
    if style == "median":
        mid = data.shape[0] // 2
        if data.shape[0] % 2:
            return float(data[mid])
        return float((data[mid - 1] + data[mid]) * 0.5)
    return float(np.dot(data, _weights(data.shape[0])))


def manual_reference(
    data,
    length=25,
    offset=0,
    multiplier=2.0,
    slope_style="smooth_median",
    residual_style="smooth_median",
    deviation_style="mad",
    mad_style="smooth_median",
    include_prediction_in_deviation=False,
):
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[0]
    out = {
        "value": np.full(n, np.nan, dtype=np.float64),
        "upper": np.full(n, np.nan, dtype=np.float64),
        "lower": np.full(n, np.nan, dtype=np.float64),
        "slope": np.full(n, np.nan, dtype=np.float64),
        "intercept": np.full(n, np.nan, dtype=np.float64),
        "deviation": np.full(n, np.nan, dtype=np.float64),
    }
    warmup = length + offset - 1

    for idx in range(warmup, n):
        base = idx - offset
        start = base + 1 - length
        end = idx if include_prediction_in_deviation else base
        segment = data[start : end + 1]
        if not np.all(np.isfinite(segment)):
            continue

        slopes = []
        for i in range(length - 1):
            value_i = data[base - i]
            for j in range(i + 1, length):
                slopes.append((data[base - j] - value_i) / (j - i))
        beta_1 = _estimate(slopes, slope_style)

        residuals = [data[base - j] - beta_1 * j for j in range(length)]
        beta_0 = _estimate(residuals, residual_style)
        predicted = beta_0 - beta_1 * offset

        if deviation_style == "mad":
            point_start = -offset if include_prediction_in_deviation else 0
            errors = []
            for point in range(point_start, length):
                source_idx = idx - offset - point
                predicted_point = beta_0 + beta_1 * point
                errors.append(abs(data[source_idx] - predicted_point))
            dev = _estimate(errors, mad_style) * multiplier
        else:
            point_start = -offset if include_prediction_in_deviation else 0
            sq = 0.0
            count = 0
            for point in range(point_start, length):
                source_idx = idx - offset - point
                predicted_point = beta_0 + beta_1 * point
                diff = data[source_idx] - predicted_point
                sq += diff * diff
                count += 1
            dev = math.sqrt(sq / count) * multiplier

        out["value"][idx] = predicted
        out["upper"][idx] = predicted + dev
        out["lower"][idx] = predicted - dev
        out["slope"][idx] = beta_1
        out["intercept"][idx] = beta_0
        out["deviation"][idx] = dev

    return out


class TestSmoothTheilSen:
    def test_manual_reference_matches_binding(self):
        data = np.asarray(load_test_data()["close"][:192], dtype=np.float64)
        expected = manual_reference(
            data,
            length=21,
            offset=2,
            multiplier=1.75,
            slope_style="smooth_median",
            residual_style="median",
            deviation_style="mad",
            mad_style="smooth_median",
            include_prediction_in_deviation=True,
        )
        result = mp.smooth_theil_sen(
            data,
            length=21,
            offset=2,
            multiplier=1.75,
            slope_style="smooth_median",
            residual_style="median",
            deviation_style="mad",
            mad_style="smooth_median",
            include_prediction_in_deviation=True,
        )
        for key in expected:
            np.testing.assert_allclose(result[key], expected[key], rtol=0.0, atol=1e-10, equal_nan=True)

    def test_stream_matches_batch(self):
        data = np.asarray(load_test_data()["close"][:160], dtype=np.float64)
        batch = mp.smooth_theil_sen(
            data,
            length=21,
            offset=2,
            multiplier=1.5,
            slope_style="smooth_median",
            residual_style="median",
            deviation_style="mad",
            mad_style="smooth_median",
            include_prediction_in_deviation=True,
        )
        stream = mp.SmoothTheilSenStream(
            21, 2, 1.5, "smooth_median", "median", "mad", "smooth_median", True
        )
        streamed = np.array([stream.update(float(v)) for v in data], dtype=np.float64)
        np.testing.assert_allclose(streamed[:, 0], batch["value"], rtol=0.0, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(streamed[:, 1], batch["upper"], rtol=0.0, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(streamed[:, 2], batch["lower"], rtol=0.0, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(streamed[:, 3], batch["slope"], rtol=0.0, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(streamed[:, 4], batch["intercept"], rtol=0.0, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(streamed[:, 5], batch["deviation"], rtol=0.0, atol=1e-10, equal_nan=True)

    def test_batch_metadata_and_first_row_match_single(self):
        data = np.asarray(load_test_data()["close"][:144], dtype=np.float64)
        result = mp.smooth_theil_sen_batch(
            data,
            length_range=(21, 23, 2),
            offset_range=(1, 1, 0),
            multiplier_range=(1.5, 1.5, 0.0),
            slope_style="mean",
            residual_style="smooth_median",
            deviation_style="rmsd",
            mad_style="median",
            include_prediction_in_deviation=False,
        )
        single = mp.smooth_theil_sen(
            data,
            length=21,
            offset=1,
            multiplier=1.5,
            slope_style="mean",
            residual_style="smooth_median",
            deviation_style="rmsd",
            mad_style="median",
            include_prediction_in_deviation=False,
        )
        assert result["rows"] == 2
        assert result["cols"] == data.shape[0]
        np.testing.assert_allclose(result["lengths"], np.array([21, 23]), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(result["offsets"], np.array([1, 1]), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(result["multipliers"], np.array([1.5, 1.5]), rtol=0.0, atol=1e-12)
        for key in ["value", "upper", "lower", "slope", "intercept", "deviation"]:
            np.testing.assert_allclose(result[key][0], single[key], rtol=0.0, atol=1e-10, equal_nan=True)

    def test_invalid_style(self):
        data = np.asarray(load_test_data()["close"][:64], dtype=np.float64)
        with pytest.raises(ValueError, match="Invalid slope_style"):
            mp.smooth_theil_sen(data, slope_style="bad_style")
