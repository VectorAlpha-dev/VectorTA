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


class TestEmaDeviationCorrectedT3:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        source = np.asarray(test_data["close"][:256], dtype=np.float64)
        corrected, t3 = mp.ema_deviation_corrected_t3(source, 10, 0.7, 0)

        assert len(corrected) == len(source)
        assert len(t3) == len(source)
        assert np.isfinite(corrected).any()
        assert np.isfinite(t3).any()

    def test_constant_input_corrected_matches_t3(self):
        source = np.full(64, 100.0, dtype=np.float64)
        corrected, t3 = mp.ema_deviation_corrected_t3(source, 10, 0.7, 0)
        np.testing.assert_allclose(corrected, t3, rtol=0.0, atol=1e-12, equal_nan=True)

    def test_kernel_parity(self, test_data):
        source = np.asarray(test_data["close"][:220], dtype=np.float64)
        corrected_auto, t3_auto = mp.ema_deviation_corrected_t3(source, 10, 0.7, 0)
        corrected_scalar, t3_scalar = mp.ema_deviation_corrected_t3(
            source, 10, 0.7, 0, kernel="scalar"
        )

        np.testing.assert_allclose(
            corrected_auto, corrected_scalar, rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(t3_auto, t3_scalar, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_stream_matches_batch(self, test_data):
        source = np.asarray(test_data["close"][:240], dtype=np.float64)
        corrected_batch, t3_batch = mp.ema_deviation_corrected_t3(source, 10, 0.7, 0)

        stream = mp.EmaDeviationCorrectedT3Stream(10, 0.7, 0)
        corrected_stream = []
        t3_stream = []
        for value in source:
            out = stream.update(float(value))
            if out is None:
                corrected_stream.append(np.nan)
                t3_stream.append(np.nan)
            else:
                corrected_stream.append(out[0])
                t3_stream.append(out[1])

        np.testing.assert_allclose(
            corrected_stream, corrected_batch, rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            t3_stream, t3_batch, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        source = np.asarray(test_data["close"][:220], dtype=np.float64)
        corrected_single, t3_single = mp.ema_deviation_corrected_t3(source, 10, 0.7, 0)
        batch = mp.ema_deviation_corrected_t3_batch(
            source,
            period_range=(10, 10, 0),
            hot_range=(0.7, 0.7, 0.0),
            t3_mode_range=(0, 0, 0),
        )

        assert batch["rows"] == 1
        assert batch["cols"] == len(source)
        assert batch["corrected"].shape == (1, len(source))
        assert batch["t3"].shape == (1, len(source))
        np.testing.assert_array_equal(batch["periods"], np.array([10], dtype=np.uint64))
        np.testing.assert_array_equal(batch["t3_modes"], np.array([0], dtype=np.uint64))
        np.testing.assert_allclose(batch["hots"], np.array([0.7]), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            batch["corrected"][0], corrected_single, rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            batch["t3"][0], t3_single, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_invalid_t3_mode(self, test_data):
        source = np.asarray(test_data["close"][:64], dtype=np.float64)
        with pytest.raises(ValueError, match="Invalid T3 mode"):
            mp.ema_deviation_corrected_t3(source, 10, 0.7, 2)
