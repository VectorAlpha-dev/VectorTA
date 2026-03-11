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


class TestCorrectedMovingAverage:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        source = np.asarray(test_data["close"][:256], dtype=np.float64)
        result = mp.corrected_moving_average(source, 35)

        assert len(result) == len(source)
        assert np.isnan(result[:34]).all()
        assert np.isfinite(result[34:]).any()

    def test_kernel_parity(self, test_data):
        source = np.asarray(test_data["close"][:220], dtype=np.float64)

        auto = mp.corrected_moving_average(source, 20)
        scalar = mp.corrected_moving_average(source, 20, kernel="scalar")

        np.testing.assert_allclose(auto, scalar, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_invalid_period(self, test_data):
        source = np.asarray(test_data["close"][:64], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid period"):
            mp.corrected_moving_average(source, 0)

    def test_stream_matches_batch(self, test_data):
        source = np.asarray(test_data["close"][:240], dtype=np.float64)
        batch = mp.corrected_moving_average(source, 17)

        stream = mp.CorrectedMovingAverageStream(17)
        streamed = []
        for value in source:
            out = stream.update(float(value))
            streamed.append(np.nan if out is None else out)

        np.testing.assert_allclose(streamed, batch, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_batch_single_param_matches_single(self, test_data):
        source = np.asarray(test_data["close"][:220], dtype=np.float64)
        batch = mp.corrected_moving_average_batch(source, period_range=(35, 35, 0))
        single = mp.corrected_moving_average(source, 35)

        assert batch["rows"] == 1
        assert batch["cols"] == len(source)
        assert batch["values"].shape == (1, len(source))
        np.testing.assert_array_equal(batch["periods"], np.array([35], dtype=np.uint64))
        np.testing.assert_allclose(
            batch["values"][0], single, rtol=1e-10, atol=1e-10, equal_nan=True
        )
