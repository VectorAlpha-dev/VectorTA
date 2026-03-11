import numpy as np
import pytest

from test_utils import load_test_data

try:
    import vector_ta as mp
except ImportError:
    try:
        import my_project as mp
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


class TestVolumeWeightedRsi:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        close = test_data["close"][:256]
        volume = test_data["volume"][:256]
        values = mp.volume_weighted_rsi(close, volume, period=14)

        assert len(values) == len(close)
        valid = np.flatnonzero(~np.isnan(values))
        assert valid.size > 0
        assert int(valid[0]) == 13
        assert np.nanmin(values) >= 0.0
        assert np.nanmax(values) <= 100.0

    def test_kernel_parity(self, test_data):
        close = test_data["close"]
        volume = test_data["volume"]
        auto = mp.volume_weighted_rsi(close, volume, 14)
        scalar = mp.volume_weighted_rsi(close, volume, 14, kernel="scalar")
        np.testing.assert_allclose(auto, scalar, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_invalid_params(self, test_data):
        close = test_data["close"][:32]
        volume = test_data["volume"][:32]

        with pytest.raises(ValueError, match="Invalid period"):
            mp.volume_weighted_rsi(close, volume, period=0)

        with pytest.raises(ValueError, match="Input length mismatch|shape"):
            mp.volume_weighted_rsi(close, volume[:-1], period=14)

    def test_stream_matches_batch(self, test_data):
        close = test_data["close"]
        volume = test_data["volume"]
        batch = mp.volume_weighted_rsi(close, volume, 14)

        stream = mp.VolumeWeightedRsiStream(14)
        stream_values = []
        for c, v in zip(close, volume):
            out = stream.update(float(c), float(v))
            stream_values.append(np.nan if out is None else out)

        np.testing.assert_allclose(
            stream_values, batch, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        close = test_data["close"]
        volume = test_data["volume"]
        batch = mp.volume_weighted_rsi_batch(close, volume, period_range=(14, 14, 0))
        single = mp.volume_weighted_rsi(close, volume, 14)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        assert batch["values"].shape == (1, len(close))
        np.testing.assert_allclose(
            batch["values"][0], single, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_metadata_multiple_ranges(self, test_data):
        close = test_data["close"][:256]
        volume = test_data["volume"][:256]
        batch = mp.volume_weighted_rsi_batch(close, volume, period_range=(10, 14, 2))

        assert batch["rows"] == 3
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(
            batch["periods"], np.array([10, 12, 14], dtype=np.uint64)
        )
