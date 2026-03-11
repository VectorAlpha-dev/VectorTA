import numpy as np
import pytest

from test_utils import load_test_data

try:
    import my_project as mp
except ImportError:
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python' first",
        allow_module_level=True,
    )


def naive_wave_smoother(data, period, phase):
    phase_rad = np.deg2rad(phase)
    weights = np.array(
        [
            np.sin(i * np.pi / period + phase_rad) * np.cos(i * np.pi / (2.0 * period))
            for i in range(period + 1)
        ],
        dtype=np.float64,
    )
    weight_sum = weights.sum()
    if not np.isfinite(weight_sum) or abs(weight_sum) <= np.finfo(np.float64).eps:
        return np.full(len(data), np.nan, dtype=np.float64)
    weights /= weight_sum

    src = np.full(len(data), np.nan, dtype=np.float64)
    prev_raw = np.nan
    first_nz = np.nan
    for i, value in enumerate(data):
        src[i] = 0.5 * (value + (prev_raw if np.isfinite(prev_raw) else 0.0)) if np.isfinite(value) else np.nan
        prev_raw = value
        if np.isnan(first_nz) and np.isfinite(src[i]):
            first_nz = src[i]

    out = np.full(len(data), np.nan, dtype=np.float64)
    if np.isnan(first_nz):
        return out

    first = int(np.flatnonzero(np.isfinite(data))[0])
    for i in range(first, len(data)):
        acc = 0.0
        for lag in range(period + 1):
            hist = src[i - lag] if i >= lag else first_nz
            acc += (hist if np.isfinite(hist) else first_nz) * weights[lag]
        out[i] = acc
    return out


class TestWaveSmoother:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        source = np.asarray(test_data["close"][:256], dtype=np.float64)
        result = mp.wave_smoother(source, 20, 70.0)

        assert len(result) == len(source)
        assert np.isfinite(result).any()

    def test_long_period_and_early_history_match_naive(self):
        source = np.asarray([10.0, 12.0, 14.0], dtype=np.float64)
        result = mp.wave_smoother(source, 20, 70.0)
        expected = naive_wave_smoother(source, 20, 70.0)
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_kernel_parity(self, test_data):
        source = np.asarray(test_data["close"][:220], dtype=np.float64)
        auto = mp.wave_smoother(source, 20, 70.0)
        scalar = mp.wave_smoother(source, 20, 70.0, kernel="scalar")
        np.testing.assert_allclose(auto, scalar, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_stream_matches_batch(self):
        source = np.asarray([np.nan, 10.0, 12.0, 14.0, np.nan, 16.0, 18.0, 20.0], dtype=np.float64)
        batch = mp.wave_smoother(source, 4, 70.0)

        stream = mp.WaveSmootherStream(4, 70.0)
        streamed = []
        for value in source:
            out = stream.update(float(value))
            streamed.append(np.nan if out is None else out)

        np.testing.assert_allclose(streamed, batch, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_batch_single_param_matches_single(self, test_data):
        source = np.asarray(test_data["close"][:220], dtype=np.float64)
        single = mp.wave_smoother(source, 20, 70.0)
        batch = mp.wave_smoother_batch(
            source,
            period_range=(20, 20, 0),
            phase_range=(70.0, 70.0, 0.0),
        )

        assert batch["rows"] == 1
        assert batch["cols"] == len(source)
        assert batch["values"].shape == (1, len(source))
        np.testing.assert_array_equal(batch["periods"], np.array([20], dtype=np.uint64))
        np.testing.assert_allclose(batch["phases"], np.array([70.0]), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            batch["values"][0], single, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_invalid_phase(self, test_data):
        source = np.asarray(test_data["close"][:64], dtype=np.float64)
        with pytest.raises(ValueError, match="Invalid phase"):
            mp.wave_smoother(source, 20, 120.0)
