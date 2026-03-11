"""
Python binding tests for Historical Volatility Percentile.
"""
import pytest
import numpy as np

from test_utils import load_test_data

try:
    import my_project as mp
except ImportError:
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python' first",
        allow_module_level=True,
    )


class TestHistoricalVolatilityPercentile:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        close = np.asarray(test_data["close"][:256], dtype=np.float64)

        hvp, hvp_sma = mp.historical_volatility_percentile(close, 5, 10)

        assert len(hvp) == len(close)
        assert len(hvp_sma) == len(close)
        assert np.isnan(hvp[:13]).all()
        assert np.isnan(hvp_sma[:17]).all()
        assert np.isfinite(hvp[13:]).any()
        assert np.isfinite(hvp_sma[17:]).any()

    def test_kernel_parity(self, test_data):
        close = np.asarray(test_data["close"][:256], dtype=np.float64)

        auto_hvp, auto_hvp_sma = mp.historical_volatility_percentile(close, 5, 10)
        scalar_hvp, scalar_hvp_sma = mp.historical_volatility_percentile(
            close, 5, 10, kernel="scalar"
        )

        np.testing.assert_allclose(auto_hvp, scalar_hvp, rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(
            auto_hvp_sma, scalar_hvp_sma, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_invalid_length(self, test_data):
        close = np.asarray(test_data["close"][:64], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid length"):
            mp.historical_volatility_percentile(close, 1, 10)

    def test_stream_matches_batch(self, test_data):
        close = np.asarray(test_data["close"][:256], dtype=np.float64)

        batch_hvp, batch_hvp_sma = mp.historical_volatility_percentile(close, 5, 10)
        stream = mp.HistoricalVolatilityPercentileStream(5, 10)
        stream_hvp = []
        stream_hvp_sma = []

        for value in close:
            out = stream.update(float(value))
            if out is None:
                stream_hvp.append(np.nan)
                stream_hvp_sma.append(np.nan)
            else:
                v0, v1 = out
                stream_hvp.append(v0)
                stream_hvp_sma.append(v1)

        np.testing.assert_allclose(
            stream_hvp, batch_hvp, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            stream_hvp_sma, batch_hvp_sma, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        close = np.asarray(test_data["close"][:192], dtype=np.float64)

        result = mp.historical_volatility_percentile_batch(
            close,
            length_range=(5, 5, 0),
            annual_length_range=(10, 10, 0),
        )

        assert result["rows"] == 1
        assert result["cols"] == len(close)
        assert result["hvp"].shape == (1, len(close))
        assert result["hvp_sma"].shape == (1, len(close))
        np.testing.assert_array_equal(result["lengths"], np.array([5], dtype=np.uint64))
        np.testing.assert_array_equal(result["annual_lengths"], np.array([10], dtype=np.uint64))

        single_hvp, single_hvp_sma = mp.historical_volatility_percentile(close, 5, 10)
        np.testing.assert_allclose(
            result["hvp"][0], single_hvp, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            result["hvp_sma"][0], single_hvp_sma, rtol=1e-10, atol=1e-10, equal_nan=True
        )
