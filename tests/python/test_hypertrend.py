import numpy as np
import pytest

from test_utils import load_test_data

try:
    import my_project as mp
except ImportError:
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python' first",
        allow_module_level=True,
    )


class TestHyperTrend:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        high = np.asarray(test_data["high"][:320], dtype=np.float64)
        low = np.asarray(test_data["low"][:320], dtype=np.float64)
        source = np.asarray(test_data["close"][:320], dtype=np.float64)

        upper, average, lower, trend, changed = mp.hypertrend(
            high, low, source, 5.0, 14.0, 80.0
        )

        assert len(upper) == len(source)
        assert len(average) == len(source)
        assert len(lower) == len(source)
        assert len(trend) == len(source)
        assert len(changed) == len(source)
        assert np.isfinite(average).any()
        assert np.all(np.isnan(upper) == np.isnan(lower))
        valid = ~np.isnan(upper)
        assert np.all(upper[valid] >= lower[valid])

    def test_kernel_parity(self, test_data):
        high = np.asarray(test_data["high"][:256], dtype=np.float64)
        low = np.asarray(test_data["low"][:256], dtype=np.float64)
        source = np.asarray(test_data["close"][:256], dtype=np.float64)

        auto = mp.hypertrend(high, low, source, 4.5, 10.0, 60.0)
        scalar = mp.hypertrend(
            high, low, source, 4.5, 10.0, 60.0, kernel="scalar"
        )

        for lhs, rhs in zip(auto, scalar):
            np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_invalid_params(self, test_data):
        high = np.asarray(test_data["high"][:128], dtype=np.float64)
        low = np.asarray(test_data["low"][:128], dtype=np.float64)
        source = np.asarray(test_data["close"][:128], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid factor"):
            mp.hypertrend(high, low, source, 0.0, 14.0, 80.0)

        with pytest.raises(ValueError, match="Invalid slope"):
            mp.hypertrend(high, low, source, 5.0, 0.0, 80.0)

        with pytest.raises(ValueError, match="Invalid width_percent"):
            mp.hypertrend(high, low, source, 5.0, 14.0, 120.0)

    def test_stream_matches_batch(self, test_data):
        high = np.asarray(test_data["high"][:240], dtype=np.float64)
        low = np.asarray(test_data["low"][:240], dtype=np.float64)
        source = np.asarray(test_data["close"][:240], dtype=np.float64)

        batch = mp.hypertrend(high, low, source, 4.0, 12.0, 55.0)
        stream = mp.HyperTrendStream(4.0, 12.0, 55.0)

        upper = []
        average = []
        lower = []
        trend = []
        changed = []
        for h, l, s in zip(high, low, source):
            out = stream.update(float(h), float(l), float(s))
            if out is None:
                upper.append(np.nan)
                average.append(np.nan)
                lower.append(np.nan)
                trend.append(np.nan)
                changed.append(np.nan)
            else:
                u, a, lo, t, c = out
                upper.append(u)
                average.append(a)
                lower.append(lo)
                trend.append(t)
                changed.append(c)

        np.testing.assert_allclose(upper, batch[0], rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(average, batch[1], rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(lower, batch[2], rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(trend, batch[3], rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(changed, batch[4], rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_batch_single_param_matches_single(self, test_data):
        high = np.asarray(test_data["high"][:240], dtype=np.float64)
        low = np.asarray(test_data["low"][:240], dtype=np.float64)
        source = np.asarray(test_data["close"][:240], dtype=np.float64)

        result = mp.hypertrend_batch(
            high,
            low,
            source,
            factor_range=(5.0, 5.0, 0.0),
            slope_range=(14.0, 14.0, 0.0),
            width_percent_range=(80.0, 80.0, 0.0),
        )

        assert result["rows"] == 1
        assert result["cols"] == len(source)
        assert result["upper"].shape == (1, len(source))
        assert result["average"].shape == (1, len(source))
        assert result["lower"].shape == (1, len(source))
        assert result["trend"].shape == (1, len(source))
        assert result["changed"].shape == (1, len(source))
        np.testing.assert_allclose(result["factors"], np.array([5.0]))
        np.testing.assert_allclose(result["slopes"], np.array([14.0]))
        np.testing.assert_allclose(result["width_percents"], np.array([80.0]))

        single = mp.hypertrend(high, low, source, 5.0, 14.0, 80.0)
        for idx, key in enumerate(["upper", "average", "lower", "trend", "changed"]):
            np.testing.assert_allclose(
                result[key][0], single[idx], rtol=1e-10, atol=1e-10, equal_nan=True
            )

    def test_batch_metadata(self, test_data):
        high = np.asarray(test_data["high"][:180], dtype=np.float64)
        low = np.asarray(test_data["low"][:180], dtype=np.float64)
        source = np.asarray(test_data["close"][:180], dtype=np.float64)

        result = mp.hypertrend_batch(
            high,
            low,
            source,
            factor_range=(4.0, 5.0, 1.0),
            slope_range=(10.0, 12.0, 2.0),
            width_percent_range=(60.0, 80.0, 20.0),
        )

        assert result["rows"] == 8
        assert result["cols"] == len(source)
        np.testing.assert_allclose(
            result["factors"], np.array([4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0])
        )
        np.testing.assert_allclose(
            result["slopes"], np.array([10.0, 10.0, 12.0, 12.0, 10.0, 10.0, 12.0, 12.0])
        )
        np.testing.assert_allclose(
            result["width_percents"], np.array([60.0, 80.0, 60.0, 80.0, 60.0, 80.0, 60.0, 80.0])
        )
