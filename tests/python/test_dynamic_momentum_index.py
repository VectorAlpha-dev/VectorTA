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


class TestDynamicMomentumIndex:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        close = test_data["close"][:256]
        values = mp.dynamic_momentum_index(close, 14, 5, 10, 30, 5)

        assert len(values) == len(close)
        valid = np.flatnonzero(~np.isnan(values))
        assert valid.size > 0
        assert int(valid[0]) >= 13
        assert np.nanmin(values) >= 0.0
        assert np.nanmax(values) <= 100.0

    def test_kernel_parity(self, test_data):
        close = test_data["close"]
        auto = mp.dynamic_momentum_index(close, 14, 5, 10, 30, 5)
        scalar = mp.dynamic_momentum_index(close, 14, 5, 10, 30, 5, kernel="scalar")
        np.testing.assert_allclose(auto, scalar, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_invalid_params(self, test_data):
        close = test_data["close"][:32]

        with pytest.raises(ValueError, match="Invalid RSI period"):
            mp.dynamic_momentum_index(close, 0, 5, 10, 30, 5)

        with pytest.raises(ValueError, match="Invalid limits"):
            mp.dynamic_momentum_index(close, 14, 5, 10, 4, 5)

    def test_stream_matches_batch(self, test_data):
        close = test_data["close"]
        batch = mp.dynamic_momentum_index(close, 14, 5, 10, 30, 5)

        stream = mp.DynamicMomentumIndexStream(14, 5, 10, 30, 5)
        stream_values = []
        for value in close:
            out = stream.update(float(value))
            stream_values.append(np.nan if out is None else out)

        np.testing.assert_allclose(
            stream_values, batch, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        close = test_data["close"]
        batch = mp.dynamic_momentum_index_batch(close)
        single = mp.dynamic_momentum_index(close, 14, 5, 10, 30, 5)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        assert batch["values"].shape == (1, len(close))
        np.testing.assert_allclose(
            batch["values"][0], single, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_metadata_multiple_ranges(self, test_data):
        close = test_data["close"][:192]
        batch = mp.dynamic_momentum_index_batch(close, rsi_period_range=(10, 14, 2))

        assert batch["rows"] == 3
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(
            batch["rsi_periods"], np.array([10, 12, 14], dtype=np.uint64)
        )
        np.testing.assert_array_equal(
            batch["volatility_periods"], np.array([5, 5, 5], dtype=np.uint64)
        )
