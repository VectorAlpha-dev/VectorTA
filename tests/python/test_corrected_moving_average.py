import numpy as np
import pytest

from test_utils import load_test_data

try:
    import vector_ta as mp
except ImportError:
    try:
        import vector_ta as mp
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


class TestCorrectedMovingAverage:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        result = mp.corrected_moving_average(data, 3)
        assert len(result) == len(data)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert abs(result[2] - 2.0) <= 1e-12

    def test_invalid_params(self, test_data):
        close = test_data["close"][:32]
        with pytest.raises(ValueError, match="Invalid period"):
            mp.corrected_moving_average(close, 0)

    def test_stream_matches_batch(self, test_data):
        close = test_data["close"][:256]
        batch = mp.corrected_moving_average(close, 17)
        stream = mp.CorrectedMovingAverageStream(17)
        values = []
        for value in close:
            out = stream.update(float(value))
            values.append(out if out is not None else np.nan)
        np.testing.assert_allclose(values, batch, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_batch_single_param_matches_single(self, test_data):
        close = test_data["close"][:256]
        batch = mp.corrected_moving_average_batch(close, period_range=(17, 17, 0))
        single = mp.corrected_moving_average(close, 17)
        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        np.testing.assert_allclose(
            batch["values"][0], single, rtol=1e-12, atol=1e-12, equal_nan=True
        )
