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


class TestStochasticAdaptiveD:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        high = np.asarray(test_data["high"][:320], dtype=np.float64)
        low = np.asarray(test_data["low"][:320], dtype=np.float64)
        close = np.asarray(test_data["close"][:320], dtype=np.float64)

        standard_d, adaptive_d, difference = mp.stochastic_adaptive_d(
            high, low, close, 20, 9, 20, 2.0
        )

        assert len(standard_d) == len(close)
        assert len(adaptive_d) == len(close)
        assert len(difference) == len(close)
        assert np.isfinite(standard_d).any()
        assert np.isfinite(adaptive_d).any()
        assert np.isfinite(difference).any()

    def test_kernel_parity(self, test_data):
        high = np.asarray(test_data["high"][:256], dtype=np.float64)
        low = np.asarray(test_data["low"][:256], dtype=np.float64)
        close = np.asarray(test_data["close"][:256], dtype=np.float64)

        auto = mp.stochastic_adaptive_d(high, low, close, 14, 5, 10, 1.6)
        scalar = mp.stochastic_adaptive_d(
            high, low, close, 14, 5, 10, 1.6, kernel="scalar"
        )

        for lhs, rhs in zip(auto, scalar):
            np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_invalid_params(self, test_data):
        high = np.asarray(test_data["high"][:128], dtype=np.float64)
        low = np.asarray(test_data["low"][:128], dtype=np.float64)
        close = np.asarray(test_data["close"][:128], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid attenuation"):
            mp.stochastic_adaptive_d(high, low, close, 20, 9, 20, 0.0)

    def test_stream_matches_batch(self, test_data):
        high = np.asarray(test_data["high"][:240], dtype=np.float64)
        low = np.asarray(test_data["low"][:240], dtype=np.float64)
        close = np.asarray(test_data["close"][:240], dtype=np.float64)

        batch = mp.stochastic_adaptive_d(high, low, close, 14, 5, 10, 1.6)
        stream = mp.StochasticAdaptiveDStream(14, 5, 10, 1.6)

        standard = []
        adaptive = []
        difference = []
        for h, l, c in zip(high, low, close):
            out = stream.update(float(h), float(l), float(c))
            if out is None:
                standard.append(np.nan)
                adaptive.append(np.nan)
                difference.append(np.nan)
            else:
                s, a, d = out
                standard.append(s)
                adaptive.append(a)
                difference.append(d)

        np.testing.assert_allclose(standard, batch[0], rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(adaptive, batch[1], rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(difference, batch[2], rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_batch_single_param_matches_single(self, test_data):
        high = np.asarray(test_data["high"][:220], dtype=np.float64)
        low = np.asarray(test_data["low"][:220], dtype=np.float64)
        close = np.asarray(test_data["close"][:220], dtype=np.float64)

        result = mp.stochastic_adaptive_d_batch(
            high,
            low,
            close,
            k_length_range=(20, 20, 0),
            d_smoothing_range=(9, 9, 0),
            pre_smooth_range=(20, 20, 0),
            attenuation_range=(2.0, 2.0, 0.0),
        )

        assert result["rows"] == 1
        assert result["cols"] == len(close)
        np.testing.assert_allclose(result["k_lengths"], np.array([20]))
        np.testing.assert_allclose(result["d_smoothings"], np.array([9]))
        np.testing.assert_allclose(result["pre_smooths"], np.array([20]))
        np.testing.assert_allclose(result["attenuations"], np.array([2.0]))

        single = mp.stochastic_adaptive_d(high, low, close, 20, 9, 20, 2.0)
        np.testing.assert_allclose(
            result["standard_d"][0], single[0], rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            result["adaptive_d"][0], single[1], rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            result["difference"][0], single[2], rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_metadata(self, test_data):
        high = np.asarray(test_data["high"][:180], dtype=np.float64)
        low = np.asarray(test_data["low"][:180], dtype=np.float64)
        close = np.asarray(test_data["close"][:180], dtype=np.float64)

        result = mp.stochastic_adaptive_d_batch(
            high,
            low,
            close,
            k_length_range=(14, 16, 2),
            d_smoothing_range=(5, 7, 2),
            pre_smooth_range=(10, 12, 2),
            attenuation_range=(1.5, 2.0, 0.5),
        )

        assert result["rows"] == 16
        assert result["cols"] == len(close)
        np.testing.assert_allclose(
            result["k_lengths"],
            np.array([14, 14, 14, 14, 14, 14, 14, 14, 16, 16, 16, 16, 16, 16, 16, 16]),
        )
