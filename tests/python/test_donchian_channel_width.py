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


class TestDonchianChannelWidth:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        values = mp.donchian_channel_width(high, low, 20)

        assert len(values) == len(high)
        valid = np.flatnonzero(~np.isnan(values))
        assert valid.size > 0
        assert int(valid[0]) == 19

    def test_kernel_parity(self, test_data):
        high = test_data["high"]
        low = test_data["low"]
        auto = mp.donchian_channel_width(high, low, 20)
        scalar = mp.donchian_channel_width(high, low, 20, kernel="scalar")
        np.testing.assert_allclose(auto, scalar, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_invalid_params(self, test_data):
        high = test_data["high"][:64]
        low = test_data["low"][:64]

        with pytest.raises(ValueError, match="Invalid period"):
            mp.donchian_channel_width(high, low, 0)

    def test_stream_matches_batch(self, test_data):
        high = test_data["high"]
        low = test_data["low"]
        batch = mp.donchian_channel_width(high, low, 20)

        stream = mp.DonchianChannelWidthStream(20)
        streamed = []
        for h, l in zip(high, low):
            out = stream.update(float(h), float(l))
            streamed.append(np.nan if out is None else out)

        np.testing.assert_allclose(streamed, batch, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_batch_single_param_matches_single(self, test_data):
        high = test_data["high"]
        low = test_data["low"]
        batch = mp.donchian_channel_width_batch(
            high,
            low,
            period_range=(20, 20, 0),
        )
        single = mp.donchian_channel_width(high, low, 20)

        assert batch["rows"] == 1
        assert batch["cols"] == len(high)
        assert batch["values"].shape == (1, len(high))
        np.testing.assert_allclose(
            batch["values"][0], single, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_metadata_multiple_periods(self, test_data):
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        batch = mp.donchian_channel_width_batch(
            high,
            low,
            period_range=(10, 14, 2),
        )

        assert batch["rows"] == 3
        assert batch["cols"] == len(high)
        np.testing.assert_array_equal(
            batch["periods"], np.array([10, 12, 14], dtype=np.uint64)
        )
