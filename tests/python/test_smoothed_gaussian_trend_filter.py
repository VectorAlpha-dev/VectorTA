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


class TestSmoothedGaussianTrendFilter:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        high = np.asarray(test_data["high"][:320], dtype=np.float64)
        low = np.asarray(test_data["low"][:320], dtype=np.float64)
        close = np.asarray(test_data["close"][:320], dtype=np.float64)

        filt, supertrend, trend, ranging = mp.smoothed_gaussian_trend_filter(
            high, low, close, 15, 3, 22, 7
        )

        assert len(filt) == len(close)
        assert len(supertrend) == len(close)
        assert len(trend) == len(close)
        assert len(ranging) == len(close)
        assert np.isfinite(filt).any()
        assert np.isfinite(supertrend).any()
        assert np.isfinite(trend).any()
        assert np.isfinite(ranging).any()

    def test_kernel_parity(self, test_data):
        high = np.asarray(test_data["high"][:256], dtype=np.float64)
        low = np.asarray(test_data["low"][:256], dtype=np.float64)
        close = np.asarray(test_data["close"][:256], dtype=np.float64)

        auto = mp.smoothed_gaussian_trend_filter(high, low, close, 18, 4, 20, 5)
        scalar = mp.smoothed_gaussian_trend_filter(
            high, low, close, 18, 4, 20, 5, kernel="scalar"
        )

        for lhs, rhs in zip(auto, scalar):
            np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_invalid_params(self, test_data):
        high = np.asarray(test_data["high"][:128], dtype=np.float64)
        low = np.asarray(test_data["low"][:128], dtype=np.float64)
        close = np.asarray(test_data["close"][:128], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid poles"):
            mp.smoothed_gaussian_trend_filter(high, low, close, 15, 5, 22, 7)

    def test_stream_matches_batch(self, test_data):
        high = np.asarray(test_data["high"][:240], dtype=np.float64)
        low = np.asarray(test_data["low"][:240], dtype=np.float64)
        close = np.asarray(test_data["close"][:240], dtype=np.float64)

        batch = mp.smoothed_gaussian_trend_filter(high, low, close, 15, 3, 22, 7)
        stream = mp.SmoothedGaussianTrendFilterStream(15, 3, 22, 7)

        filt = []
        supertrend = []
        trend = []
        ranging = []
        for h, l, c in zip(high, low, close):
            out = stream.update(float(h), float(l), float(c))
            if out is None:
                filt.append(np.nan)
                supertrend.append(np.nan)
                trend.append(np.nan)
                ranging.append(np.nan)
            else:
                f, s, t, r = out
                filt.append(f)
                supertrend.append(s)
                trend.append(t)
                ranging.append(r)

        np.testing.assert_allclose(filt, batch[0], rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(
            supertrend, batch[1], rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(trend, batch[2], rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(ranging, batch[3], rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_batch_single_param_matches_single(self, test_data):
        high = np.asarray(test_data["high"][:220], dtype=np.float64)
        low = np.asarray(test_data["low"][:220], dtype=np.float64)
        close = np.asarray(test_data["close"][:220], dtype=np.float64)

        result = mp.smoothed_gaussian_trend_filter_batch(
            high,
            low,
            close,
            gaussian_length_range=(15, 15, 0),
            poles_range=(3, 3, 0),
            smoothing_length_range=(22, 22, 0),
            linreg_offset_range=(7, 7, 0),
        )

        assert result["rows"] == 1
        assert result["cols"] == len(close)
        np.testing.assert_allclose(result["gaussian_lengths"], np.array([15]))
        np.testing.assert_allclose(result["poles"], np.array([3]))
        np.testing.assert_allclose(result["smoothing_lengths"], np.array([22]))
        np.testing.assert_allclose(result["linreg_offsets"], np.array([7]))

        single = mp.smoothed_gaussian_trend_filter(high, low, close, 15, 3, 22, 7)
        np.testing.assert_allclose(
            result["filter"][0], single[0], rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            result["supertrend"][0], single[1], rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            result["trend"][0], single[2], rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            result["ranging"][0], single[3], rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_metadata(self, test_data):
        high = np.asarray(test_data["high"][:180], dtype=np.float64)
        low = np.asarray(test_data["low"][:180], dtype=np.float64)
        close = np.asarray(test_data["close"][:180], dtype=np.float64)

        result = mp.smoothed_gaussian_trend_filter_batch(
            high,
            low,
            close,
            gaussian_length_range=(15, 17, 2),
            poles_range=(2, 4, 2),
            smoothing_length_range=(20, 22, 2),
            linreg_offset_range=(5, 7, 2),
        )

        assert result["rows"] == 16
        assert result["cols"] == len(close)
