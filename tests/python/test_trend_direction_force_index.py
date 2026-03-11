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


class TestTrendDirectionForceIndex:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        close = np.asarray(test_data["close"][:256], dtype=np.float64)

        values = mp.trend_direction_force_index(close, 10)

        assert len(values) == len(close)
        assert np.isnan(values[0])
        assert np.isfinite(values[1:]).any()
        finite = values[np.isfinite(values)]
        assert np.all(np.abs(finite) <= 1.0 + 1e-12)

    def test_kernel_parity(self, test_data):
        close = np.asarray(test_data["close"][:256], dtype=np.float64)

        auto_values = mp.trend_direction_force_index(close, 10)
        scalar_values = mp.trend_direction_force_index(close, 10, kernel="scalar")

        np.testing.assert_allclose(
            auto_values, scalar_values, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_invalid_length(self, test_data):
        close = np.asarray(test_data["close"][:64], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid length"):
            mp.trend_direction_force_index(close, 0)

    def test_stream_matches_batch(self, test_data):
        close = np.asarray(test_data["close"][:192], dtype=np.float64)

        batch = mp.trend_direction_force_index(close, 10)
        stream = mp.TrendDirectionForceIndexStream(10)
        stream_values = []

        for value in close:
            out = stream.update(float(value))
            stream_values.append(np.nan if out is None else out)

        np.testing.assert_allclose(
            stream_values, batch, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        close = np.asarray(test_data["close"][:192], dtype=np.float64)

        result = mp.trend_direction_force_index_batch(close, length_range=(10, 10, 0))

        assert result["rows"] == 1
        assert result["cols"] == len(close)
        assert result["values"].shape == (1, len(close))
        np.testing.assert_array_equal(result["lengths"], np.array([10], dtype=np.uint64))

        single = mp.trend_direction_force_index(close, 10)
        np.testing.assert_allclose(
            result["values"][0], single, rtol=1e-10, atol=1e-10, equal_nan=True
        )
