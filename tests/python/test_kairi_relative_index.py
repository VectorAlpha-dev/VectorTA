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


class TestKairiRelativeIndex:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        source = test_data["close"][:256]
        volume = test_data["volume"][:256]
        values = mp.kairi_relative_index(source, volume, 50, "SMA")

        assert len(values) == len(source)
        valid = np.flatnonzero(~np.isnan(values))
        assert valid.size > 0
        assert int(valid[0]) == 49

    def test_kernel_parity(self, test_data):
        source = test_data["close"]
        volume = test_data["volume"]
        auto = mp.kairi_relative_index(source, volume, 20, "EMA")
        scalar = mp.kairi_relative_index(source, volume, 20, "EMA", kernel="scalar")
        np.testing.assert_allclose(auto, scalar, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_invalid_params(self, test_data):
        source = test_data["close"][:64]
        volume = test_data["volume"][:64]

        with pytest.raises(ValueError, match="Invalid length"):
            mp.kairi_relative_index(source, volume, 1, "SMA")

        with pytest.raises(ValueError, match="Invalid moving average type"):
            mp.kairi_relative_index(source, volume, 10, "BAD")

    def test_stream_matches_batch(self, test_data):
        source = test_data["close"]
        volume = test_data["volume"]
        batch = mp.kairi_relative_index(source, volume, 14, "TMA")

        stream = mp.KairiRelativeIndexStream(14, "TMA")
        streamed = []
        for s, v in zip(source, volume):
            out = stream.update(float(s), float(v))
            streamed.append(np.nan if out is None else out)

        np.testing.assert_allclose(streamed, batch, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_batch_single_param_matches_single(self, test_data):
        source = test_data["close"]
        volume = test_data["volume"]
        batch = mp.kairi_relative_index_batch(
            source,
            volume,
            length_range=(20, 20, 0),
            ma_type="HMA",
        )
        single = mp.kairi_relative_index(source, volume, 20, "HMA")

        assert batch["rows"] == 1
        assert batch["cols"] == len(source)
        assert batch["values"].shape == (1, len(source))
        np.testing.assert_allclose(
            batch["values"][0], single, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_metadata_multiple_lengths(self, test_data):
        source = test_data["close"][:256]
        volume = test_data["volume"][:256]
        batch = mp.kairi_relative_index_batch(
            source,
            volume,
            length_range=(10, 14, 2),
            ma_type="VWMA",
        )

        assert batch["rows"] == 3
        assert batch["cols"] == len(source)
        np.testing.assert_array_equal(
            batch["lengths"], np.array([10, 12, 14], dtype=np.uint64)
        )
