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


class TestRollingZScoreTrend:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        close = test_data["close"][:256]
        zscore, momentum = mp.rolling_z_score_trend(close, 20)

        assert len(zscore) == len(close)
        assert len(momentum) == len(close)
        z_valid = np.flatnonzero(~np.isnan(zscore))
        m_valid = np.flatnonzero(~np.isnan(momentum))
        assert z_valid.size > 0
        assert m_valid.size > 0
        assert int(z_valid[0]) == 19
        assert int(m_valid[0]) == 20

    def test_kernel_parity(self, test_data):
        close = test_data["close"]
        auto_z, auto_m = mp.rolling_z_score_trend(close, 20)
        scalar_z, scalar_m = mp.rolling_z_score_trend(close, 20, kernel="scalar")
        np.testing.assert_allclose(auto_z, scalar_z, rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(auto_m, scalar_m, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_invalid_params(self, test_data):
        close = test_data["close"][:64]

        with pytest.raises(ValueError, match="Invalid lookback_period"):
            mp.rolling_z_score_trend(close, 0)

    def test_stream_matches_batch(self, test_data):
        close = test_data["close"]
        batch_z, batch_m = mp.rolling_z_score_trend(close, 20)

        stream = mp.RollingZScoreTrendStream(20)
        z_stream = []
        m_stream = []
        for value in close:
            out = stream.update(float(value))
            if out is None:
                z_stream.append(np.nan)
                m_stream.append(np.nan)
            else:
                z, m = out
                z_stream.append(z)
                m_stream.append(m)

        np.testing.assert_allclose(
            z_stream, batch_z, rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            m_stream, batch_m, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        close = test_data["close"]
        batch = mp.rolling_z_score_trend_batch(
            close,
            lookback_period_range=(20, 20, 0),
        )
        single_z, single_m = mp.rolling_z_score_trend(close, 20)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        np.testing.assert_allclose(
            batch["zscore"][0], single_z, rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            batch["momentum"][0], single_m, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_metadata_multiple_ranges(self, test_data):
        close = test_data["close"][:256]
        batch = mp.rolling_z_score_trend_batch(
            close,
            lookback_period_range=(10, 20, 5),
        )

        assert batch["rows"] == 3
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(
            batch["lookback_periods"], np.array([10, 15, 20], dtype=np.uint64)
        )
