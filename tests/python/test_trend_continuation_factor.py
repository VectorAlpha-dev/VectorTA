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


class TestTrendContinuationFactor:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        close = np.asarray(test_data["close"][:256], dtype=np.float64)

        plus_tcf, minus_tcf = mp.trend_continuation_factor(close, 35)

        assert len(plus_tcf) == len(close)
        assert len(minus_tcf) == len(close)
        assert np.isnan(plus_tcf[:35]).all()
        assert np.isnan(minus_tcf[:35]).all()
        assert np.isfinite(plus_tcf[35:]).any()
        assert np.isfinite(minus_tcf[35:]).any()

    def test_kernel_parity(self, test_data):
        close = np.asarray(test_data["close"][:256], dtype=np.float64)

        auto_plus, auto_minus = mp.trend_continuation_factor(close, 20)
        scalar_plus, scalar_minus = mp.trend_continuation_factor(
            close, 20, kernel="scalar"
        )

        np.testing.assert_allclose(
            auto_plus, scalar_plus, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            auto_minus, scalar_minus, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_invalid_length(self, test_data):
        close = np.asarray(test_data["close"][:64], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid length"):
            mp.trend_continuation_factor(close, 0)

    def test_stream_matches_batch(self, test_data):
        close = np.asarray(test_data["close"][:192], dtype=np.float64)

        batch_plus, batch_minus = mp.trend_continuation_factor(close, 18)
        stream = mp.TrendContinuationFactorStream(18)
        stream_plus = []
        stream_minus = []

        for value in close:
            out = stream.update(float(value))
            if out is None:
                stream_plus.append(np.nan)
                stream_minus.append(np.nan)
            else:
                p, m = out
                stream_plus.append(p)
                stream_minus.append(m)

        np.testing.assert_allclose(
            stream_plus, batch_plus, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            stream_minus, batch_minus, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        close = np.asarray(test_data["close"][:192], dtype=np.float64)

        result = mp.trend_continuation_factor_batch(close, length_range=(12, 12, 0))

        assert result["rows"] == 1
        assert result["cols"] == len(close)
        assert result["plus_tcf"].shape == (1, len(close))
        assert result["minus_tcf"].shape == (1, len(close))
        np.testing.assert_array_equal(result["lengths"], np.array([12], dtype=np.uint64))

        single_plus, single_minus = mp.trend_continuation_factor(close, 12)
        np.testing.assert_allclose(
            result["plus_tcf"][0], single_plus, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            result["minus_tcf"][0], single_minus, rtol=1e-10, atol=1e-10, equal_nan=True
        )
