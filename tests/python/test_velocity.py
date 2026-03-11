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


def hlcc4_series(test_data):
    return (test_data["high"] + test_data["low"] + 2.0 * test_data["close"]) * 0.25


def naive_velocity(data, length, smooth_length):
    data = np.asarray(data, dtype=float)
    out = np.full(data.shape[0], np.nan)
    finite = np.flatnonzero(np.isfinite(data))
    if finite.size == 0:
        return out

    first_valid = int(finite[0])
    raw = np.full(data.shape[0], np.nan)
    denom = smooth_length * (smooth_length + 1) / 2.0

    for idx in range(first_valid, data.shape[0]):
        value = data[idx]
        if not np.isfinite(value):
            continue
        acc = 0.0
        for lag in range(1, length + 1):
            hist = data[idx - lag] if idx >= lag and np.isfinite(data[idx - lag]) else 0.0
            acc += (value - hist) / lag
        raw[idx] = acc / length

    for idx in range(first_valid + smooth_length - 1, data.shape[0]):
        window = raw[idx - smooth_length + 1 : idx + 1]
        if np.all(np.isfinite(window)):
            out[idx] = np.dot(window, np.arange(1, smooth_length + 1, dtype=float)) / denom

    return out


class TestVelocity:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_basic_output_shape(self, test_data):
        data = hlcc4_series(test_data)
        result = ta_indicators.velocity(data)

        assert len(result) == len(data)
        assert np.all(np.isnan(result[:4]))
        assert np.any(np.isfinite(result[32:]))

    def test_matches_naive_reference(self):
        data = np.array([np.nan, np.nan, 10.0, 11.0, 12.0, 13.0, 12.0, 14.0], dtype=float)
        result = ta_indicators.velocity(data, length=3, smooth_length=2)
        expected = naive_velocity(data, 3, 2)
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_stream_matches_batch(self, test_data):
        data = hlcc4_series(test_data)
        batch = ta_indicators.velocity(data, length=21, smooth_length=5)
        stream = ta_indicators.VelocityStream(length=21, smooth_length=5)

        values = np.full(len(data), np.nan)
        for idx, value in enumerate(data):
            result = stream.update(float(value))
            if result is not None:
                values[idx] = result

        np.testing.assert_allclose(batch, values, rtol=1e-12, atol=1e-12, equal_nan=True)
        stream.reset()
        assert stream.update(float("nan")) is None

    def test_batch_matches_single(self, test_data):
        data = hlcc4_series(test_data)
        single = ta_indicators.velocity(data, length=21, smooth_length=5)
        batch = ta_indicators.velocity_batch(
            data,
            length_range=(21, 21, 0),
            smooth_length_range=(5, 5, 0),
        )

        assert batch["values"].shape == (1, len(data))
        assert list(batch["lengths"]) == [21]
        assert list(batch["smooth_lengths"]) == [5]
        assert batch["rows"] == 1
        assert batch["cols"] == len(data)
        assert_close(batch["values"][0], single, rtol=1e-12, atol=1e-12)

    def test_errors(self):
        with pytest.raises(ValueError, match="Invalid length|invalid length"):
            ta_indicators.velocity(np.arange(20, dtype=float), length=1, smooth_length=5)

        with pytest.raises(ValueError, match="Invalid smoothing length|invalid smoothing length"):
            ta_indicators.velocity(np.arange(20, dtype=float), length=21, smooth_length=0)

        with pytest.raises(ValueError, match="empty|Empty"):
            ta_indicators.velocity(np.array([], dtype=float))

        with pytest.raises(ValueError, match="all values are NaN|All values are NaN"):
            ta_indicators.velocity(np.full(32, np.nan))

        with pytest.raises(ValueError, match="not enough valid data|Not enough valid data"):
            ta_indicators.velocity(np.array([1.0, 2.0, 3.0], dtype=float), length=2, smooth_length=5)
