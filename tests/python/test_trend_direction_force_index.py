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


class TestTrendDirectionForceIndex:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        close = test_data["close"][:256]
        values = mp.trend_direction_force_index(close, 10)

        assert len(values) == len(close)
        valid = np.flatnonzero(~np.isnan(values))
        assert valid.size > 0
        assert int(valid[0]) == 9

    def test_kernel_parity(self, test_data):
        close = test_data["close"]
        auto = mp.trend_direction_force_index(close, 14)
        scalar = mp.trend_direction_force_index(close, 14, kernel="scalar")
        np.testing.assert_allclose(auto, scalar, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_invalid_params(self, test_data):
        close = test_data["close"][:64]
        with pytest.raises(ValueError, match="Invalid length"):
            mp.trend_direction_force_index(close, 0)

    def test_stream_matches_batch(self, test_data):
        close = test_data["close"]
        batch = mp.trend_direction_force_index(close, 10)

        stream = mp.TrendDirectionForceIndexStream(10)
        streamed = []
        for value in close:
            out = stream.update(float(value))
            streamed.append(np.nan if out is None else out)

        np.testing.assert_allclose(streamed, batch, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_batch_single_param_matches_single(self, test_data):
        close = test_data["close"]
        batch = mp.trend_direction_force_index_batch(close, length_range=(12, 12, 0))
        single = mp.trend_direction_force_index(close, 12)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        assert batch["values"].shape == (1, len(close))
        np.testing.assert_allclose(
            batch["values"][0], single, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_metadata_multiple_lengths(self, test_data):
        close = test_data["close"][:256]
        batch = mp.trend_direction_force_index_batch(close, length_range=(8, 12, 2))

        assert batch["rows"] == 3
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(
            batch["lengths"], np.array([8, 10, 12], dtype=np.uint64)
        )
