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


class TestFractalDimensionIndex:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        close = test_data["close"]
        values = mp.fractal_dimension_index(close, length=30)

        assert len(values) == len(close)
        valid = np.flatnonzero(~np.isnan(values))
        assert valid.size > 0
        assert int(valid[0]) == 29
        assert np.all(np.isfinite(values[valid]))

    def test_kernel_parity(self, test_data):
        close = test_data["close"]
        auto = mp.fractal_dimension_index(close, 30)
        scalar = mp.fractal_dimension_index(close, 30, kernel="scalar")
        np.testing.assert_allclose(auto, scalar, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_invalid_params(self, test_data):
        close = test_data["close"]

        with pytest.raises(ValueError, match="Invalid length"):
            mp.fractal_dimension_index(close, length=1)

        with pytest.raises(ValueError, match="Invalid length"):
            mp.fractal_dimension_index(close[:10], length=30)

    def test_stream_matches_batch(self, test_data):
        close = test_data["close"]
        batch = mp.fractal_dimension_index(close, 24)

        stream = mp.FractalDimensionIndexStream(24)
        stream_values = []
        for value in close:
            out = stream.update(value)
            stream_values.append(np.nan if out is None else out)

        np.testing.assert_allclose(
            stream_values, batch, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        close = test_data["close"]
        batch = mp.fractal_dimension_index_batch(close, length_range=(30, 30, 0))
        single = mp.fractal_dimension_index(close, 30)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        assert batch["values"].shape == (1, len(close))
        np.testing.assert_allclose(
            batch["values"][0], single, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_metadata_multiple_ranges(self, test_data):
        close = test_data["close"][:256]
        batch = mp.fractal_dimension_index_batch(close, length_range=(20, 24, 2))

        assert batch["rows"] == 3
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(
            batch["lengths"], np.array([20, 22, 24], dtype=np.uint64)
        )
