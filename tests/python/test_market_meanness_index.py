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


class TestMarketMeannessIndex:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        open_ = np.asarray(test_data["open"][:720], dtype=np.float64)
        close = np.asarray(test_data["close"][:720], dtype=np.float64)

        mmi, mmi_smoothed = mp.market_meanness_index(open_, close, 300, "Price")

        assert len(mmi) == len(close)
        assert len(mmi_smoothed) == len(close)
        assert np.isnan(mmi[:299]).all()
        assert np.isnan(mmi_smoothed[:598]).all()
        assert np.isfinite(mmi[299:]).any()
        assert np.isfinite(mmi_smoothed[598:]).any()

    def test_kernel_parity(self, test_data):
        open_ = np.asarray(test_data["open"][:720], dtype=np.float64)
        close = np.asarray(test_data["close"][:720], dtype=np.float64)

        auto_mmi, auto_smoothed = mp.market_meanness_index(
            open_, close, 120, "Change"
        )
        scalar_mmi, scalar_smoothed = mp.market_meanness_index(
            open_, close, 120, "Change", kernel="scalar"
        )

        np.testing.assert_allclose(
            auto_mmi, scalar_mmi, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            auto_smoothed, scalar_smoothed, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_invalid_source_mode(self, test_data):
        open_ = np.asarray(test_data["open"][:128], dtype=np.float64)
        close = np.asarray(test_data["close"][:128], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid source mode"):
            mp.market_meanness_index(open_, close, 20, "bogus")

    def test_stream_matches_batch(self, test_data):
        open_ = np.asarray(test_data["open"][:320], dtype=np.float64)
        close = np.asarray(test_data["close"][:320], dtype=np.float64)

        batch_mmi, batch_smoothed = mp.market_meanness_index(
            open_, close, 40, "Change"
        )
        stream = mp.MarketMeannessIndexStream(40, "Change")
        stream_mmi = []
        stream_smoothed = []

        for o, c in zip(open_, close):
            out = stream.update(float(o), float(c))
            if out is None:
                stream_mmi.append(np.nan)
                stream_smoothed.append(np.nan)
            else:
                a, b = out
                stream_mmi.append(a)
                stream_smoothed.append(b)

        np.testing.assert_allclose(
            stream_mmi, batch_mmi, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            stream_smoothed, batch_smoothed, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        open_ = np.asarray(test_data["open"][:320], dtype=np.float64)
        close = np.asarray(test_data["close"][:320], dtype=np.float64)

        result = mp.market_meanness_index_batch(
            open_,
            close,
            length_range=(40, 40, 0),
            source_mode="Price",
        )

        assert result["rows"] == 1
        assert result["cols"] == len(close)
        assert result["mmi"].shape == (1, len(close))
        assert result["mmi_smoothed"].shape == (1, len(close))
        np.testing.assert_array_equal(result["lengths"], np.array([40], dtype=np.uint64))
        assert result["source_modes"] == ["Price"]

        single_mmi, single_smoothed = mp.market_meanness_index(open_, close, 40, "Price")
        np.testing.assert_allclose(
            result["mmi"][0], single_mmi, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            result["mmi_smoothed"][0],
            single_smoothed,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )
