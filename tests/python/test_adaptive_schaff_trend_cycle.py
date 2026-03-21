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


class TestAdaptiveSchaffTrendCycle:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        high = np.asarray(test_data["high"][:320], dtype=np.float64)
        low = np.asarray(test_data["low"][:320], dtype=np.float64)
        close = np.asarray(test_data["close"][:320], dtype=np.float64)

        stc, histogram = mp.adaptive_schaff_trend_cycle(
            high, low, close, 55, 12, 0.45, 26, 50
        )

        assert len(stc) == len(close)
        assert len(histogram) == len(close)
        assert np.isfinite(stc).any()
        assert np.isfinite(histogram).any()

    def test_kernel_parity(self, test_data):
        high = np.asarray(test_data["high"][:256], dtype=np.float64)
        low = np.asarray(test_data["low"][:256], dtype=np.float64)
        close = np.asarray(test_data["close"][:256], dtype=np.float64)

        auto = mp.adaptive_schaff_trend_cycle(high, low, close, 40, 10, 0.38, 20, 42)
        scalar = mp.adaptive_schaff_trend_cycle(
            high, low, close, 40, 10, 0.38, 20, 42, kernel="scalar"
        )

        for lhs, rhs in zip(auto, scalar):
            np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_invalid_params(self, test_data):
        high = np.asarray(test_data["high"][:128], dtype=np.float64)
        low = np.asarray(test_data["low"][:128], dtype=np.float64)
        close = np.asarray(test_data["close"][:128], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid smoothing_factor"):
            mp.adaptive_schaff_trend_cycle(high, low, close, 55, 12, 0.0, 26, 50)

    def test_stream_matches_batch(self, test_data):
        high = np.asarray(test_data["high"][:240], dtype=np.float64)
        low = np.asarray(test_data["low"][:240], dtype=np.float64)
        close = np.asarray(test_data["close"][:240], dtype=np.float64)

        batch = mp.adaptive_schaff_trend_cycle(high, low, close, 34, 9, 0.5, 18, 40)
        stream = mp.AdaptiveSchaffTrendCycleStream(34, 9, 0.5, 18, 40)

        stc = []
        histogram = []
        for h, l, c in zip(high, low, close):
            out = stream.update(float(h), float(l), float(c))
            if out is None:
                stc.append(np.nan)
                histogram.append(np.nan)
            else:
                s, hist = out
                stc.append(s)
                histogram.append(hist)

        np.testing.assert_allclose(stc, batch[0], rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(histogram, batch[1], rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_batch_single_param_matches_single(self, test_data):
        high = np.asarray(test_data["high"][:220], dtype=np.float64)
        low = np.asarray(test_data["low"][:220], dtype=np.float64)
        close = np.asarray(test_data["close"][:220], dtype=np.float64)

        result = mp.adaptive_schaff_trend_cycle_batch(
            high,
            low,
            close,
            adaptive_length_range=(55, 55, 0),
            stc_length_range=(12, 12, 0),
            smoothing_factor_range=(0.45, 0.45, 0.0),
            fast_length_range=(26, 26, 0),
            slow_length_range=(50, 50, 0),
        )

        assert result["rows"] == 1
        assert result["cols"] == len(close)
        np.testing.assert_allclose(result["adaptive_lengths"], np.array([55]))
        np.testing.assert_allclose(result["stc_lengths"], np.array([12]))
        np.testing.assert_allclose(result["smoothing_factors"], np.array([0.45]))
        np.testing.assert_allclose(result["fast_lengths"], np.array([26]))
        np.testing.assert_allclose(result["slow_lengths"], np.array([50]))

        single = mp.adaptive_schaff_trend_cycle(high, low, close, 55, 12, 0.45, 26, 50)
        np.testing.assert_allclose(
            result["stc"][0], single[0], rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            result["histogram"][0], single[1], rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_metadata(self, test_data):
        high = np.asarray(test_data["high"][:180], dtype=np.float64)
        low = np.asarray(test_data["low"][:180], dtype=np.float64)
        close = np.asarray(test_data["close"][:180], dtype=np.float64)

        result = mp.adaptive_schaff_trend_cycle_batch(
            high,
            low,
            close,
            adaptive_length_range=(30, 32, 2),
            stc_length_range=(9, 11, 2),
            smoothing_factor_range=(0.4, 0.5, 0.1),
            fast_length_range=(18, 20, 2),
            slow_length_range=(40, 42, 2),
        )

        assert result["rows"] == 32
        assert result["cols"] == len(close)
