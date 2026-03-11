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


class TestHistoricalVolatilityRank:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        close = test_data["close"]

        hvr, hv = mp.historical_volatility_rank(
            close,
            hv_length=10,
            rank_length=20,
            annualization_days=365.0,
            bar_days=1.0,
        )

        assert len(hvr) == len(close)
        assert len(hv) == len(close)

        hv_valid = np.flatnonzero(~np.isnan(hv))
        hvr_valid = np.flatnonzero(~np.isnan(hvr))
        assert hv_valid.size > 0
        assert hvr_valid.size > 0
        assert int(hv_valid[0]) == 10
        assert int(hvr_valid[0]) == 29
        assert np.nanmin(hvr) >= 0.0
        assert np.nanmax(hvr) <= 100.0 + 1e-9

    def test_kernel_parity(self, test_data):
        close = test_data["close"]

        auto_hvr, auto_hv = mp.historical_volatility_rank(close, 10, 20, 365.0, 1.0)
        scalar_hvr, scalar_hv = mp.historical_volatility_rank(
            close, 10, 20, 365.0, 1.0, kernel="scalar"
        )

        np.testing.assert_allclose(auto_hvr, scalar_hvr, rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(auto_hv, scalar_hv, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_invalid_params(self, test_data):
        close = test_data["close"]

        with pytest.raises(ValueError, match="Invalid hv_length"):
            mp.historical_volatility_rank(close, hv_length=0, rank_length=20)

        with pytest.raises(ValueError, match="Invalid rank_length"):
            mp.historical_volatility_rank(close, hv_length=10, rank_length=0)

        with pytest.raises(ValueError, match="Invalid annualization_days"):
            mp.historical_volatility_rank(
                close, hv_length=10, rank_length=20, annualization_days=0.0
            )

        with pytest.raises(ValueError, match="Invalid bar_days"):
            mp.historical_volatility_rank(close, hv_length=10, rank_length=20, bar_days=0.0)

    def test_stream_matches_batch(self, test_data):
        close = test_data["close"]
        batch_hvr, batch_hv = mp.historical_volatility_rank(close, 10, 24, 365.0, 1.0)

        stream = mp.HistoricalVolatilityRankStream(10, 24, 365.0, 1.0)
        stream_hvr = []
        stream_hv = []

        for value in close:
            out = stream.update(value)
            if out is None:
                stream_hvr.append(np.nan)
                stream_hv.append(np.nan)
            else:
                hvr, hv = out
                stream_hvr.append(hvr)
                stream_hv.append(hv)

        np.testing.assert_allclose(
            stream_hvr, batch_hvr, rtol=1e-8, atol=1e-8, equal_nan=True
        )
        np.testing.assert_allclose(
            stream_hv, batch_hv, rtol=1e-8, atol=1e-8, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        close = test_data["close"]

        batch = mp.historical_volatility_rank_batch(
            close,
            hv_length_range=(10, 10, 0),
            rank_length_range=(20, 20, 0),
            annualization_days_range=(365.0, 365.0, 0.0),
            bar_days_range=(1.0, 1.0, 0.0),
        )
        single_hvr, single_hv = mp.historical_volatility_rank(close, 10, 20, 365.0, 1.0)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        assert batch["hvr"].shape == (1, len(close))
        assert batch["hv"].shape == (1, len(close))

        np.testing.assert_allclose(
            batch["hvr"][0], single_hvr, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            batch["hv"][0], single_hv, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_metadata_multiple_ranges(self, test_data):
        close = test_data["close"][:256]

        batch = mp.historical_volatility_rank_batch(
            close,
            hv_length_range=(10, 12, 2),
            rank_length_range=(20, 24, 4),
            annualization_days_range=(252.0, 252.0, 0.0),
            bar_days_range=(1.0, 1.0, 0.0),
        )

        assert batch["rows"] == 4
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(
            batch["hv_lengths"], np.array([10, 10, 12, 12], dtype=np.uint64)
        )
        np.testing.assert_array_equal(
            batch["rank_lengths"], np.array([20, 24, 20, 24], dtype=np.uint64)
        )
        np.testing.assert_allclose(batch["annualization_days"], np.full(4, 252.0))
        np.testing.assert_allclose(batch["bar_days"], np.full(4, 1.0))
