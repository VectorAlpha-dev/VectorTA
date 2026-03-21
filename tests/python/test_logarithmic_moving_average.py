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


class TestLogarithmicMovingAverage:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        close = np.asarray(test_data["close"][:320], dtype=np.float64)
        lma, signal, position, momentum_confirmed = mp.logarithmic_moving_average(close)

        assert len(lma) == len(close)
        assert len(signal) == len(close)
        assert len(position) == len(close)
        assert len(momentum_confirmed) == len(close)
        assert np.isnan(lma[:99]).all()
        assert np.isfinite(signal[120:]).any()

    def test_kernel_parity(self, test_data):
        close = np.asarray(test_data["close"][:260], dtype=np.float64)

        auto = mp.logarithmic_moving_average(close, ma_type="ema")
        scalar = mp.logarithmic_moving_average(close, ma_type="ema", kernel="scalar")

        for a, b in zip(auto, scalar):
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_vwma_requires_volume(self, test_data):
        close = np.asarray(test_data["close"][:160], dtype=np.float64)

        with pytest.raises(ValueError, match="requires volume"):
            mp.logarithmic_moving_average(close, ma_type="vwma")

    def test_stream_matches_batch(self, test_data):
        close = np.asarray(test_data["close"][:240], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:240], dtype=np.float64)

        batch = mp.logarithmic_moving_average(close, ma_type="vwma", volume=volume)
        stream = mp.LogarithmicMovingAverageStream(ma_type="vwma")
        streamed_signal = []

        for value, vol in zip(close, volume):
            out = stream.update(float(value), float(vol))
            streamed_signal.append(np.nan if out is None else out[1])

        np.testing.assert_allclose(
            streamed_signal, batch[1], rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        close = np.asarray(test_data["close"][:220], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:220], dtype=np.float64)

        batch = mp.logarithmic_moving_average_batch(
            close,
            period_range=(100, 100, 0),
            steepness_range=(2.5, 2.5, 0.0),
            smooth_range=(10, 10, 0),
            momentum_weight_range=(1.2, 1.2, 0.0),
            long_threshold_range=(0.5, 0.5, 0.0),
            short_threshold_range=(-0.5, -0.5, 0.0),
            ma_type="vwma",
            volume=volume,
        )
        single = mp.logarithmic_moving_average(close, ma_type="vwma", volume=volume)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        np.testing.assert_allclose(
            batch["signal"][0], single[1], rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_metadata(self, test_data):
        close = np.asarray(test_data["close"][:180], dtype=np.float64)

        batch = mp.logarithmic_moving_average_batch(
            close,
            period_range=(90, 100, 10),
            steepness_range=(2.0, 2.5, 0.5),
            smooth_range=(8, 8, 0),
            momentum_weight_range=(1.2, 1.2, 0.0),
            long_threshold_range=(0.5, 0.5, 0.0),
            short_threshold_range=(-0.5, -0.5, 0.0),
            ma_type="ema",
        )

        assert batch["rows"] == 4
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(batch["periods"], np.array([90, 90, 100, 100], dtype=np.uint64))
        np.testing.assert_allclose(batch["steepnesses"], np.array([2.0, 2.5, 2.0, 2.5]))
