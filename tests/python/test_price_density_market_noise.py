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


class TestPriceDensityMarketNoise:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        high = np.asarray(test_data["high"][:720], dtype=np.float64)
        low = np.asarray(test_data["low"][:720], dtype=np.float64)
        close = np.asarray(test_data["close"][:720], dtype=np.float64)

        price_density, price_density_percent = mp.price_density_market_noise(
            high, low, close, 14, 200
        )

        assert len(price_density) == len(close)
        assert len(price_density_percent) == len(close)
        assert np.isnan(price_density[:13]).all()
        assert np.isnan(price_density_percent[:212]).all()
        assert np.isfinite(price_density[13:]).any()
        assert np.isfinite(price_density_percent[212:]).any()

    def test_kernel_parity(self, test_data):
        high = np.asarray(test_data["high"][:512], dtype=np.float64)
        low = np.asarray(test_data["low"][:512], dtype=np.float64)
        close = np.asarray(test_data["close"][:512], dtype=np.float64)

        auto_pd, auto_percent = mp.price_density_market_noise(high, low, close, 18, 60)
        scalar_pd, scalar_percent = mp.price_density_market_noise(
            high, low, close, 18, 60, kernel="scalar"
        )

        np.testing.assert_allclose(
            auto_pd, scalar_pd, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            auto_percent, scalar_percent, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_invalid_eval_period(self, test_data):
        high = np.asarray(test_data["high"][:128], dtype=np.float64)
        low = np.asarray(test_data["low"][:128], dtype=np.float64)
        close = np.asarray(test_data["close"][:128], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid eval period"):
            mp.price_density_market_noise(high, low, close, 14, 0)

    def test_stream_matches_batch(self, test_data):
        high = np.asarray(test_data["high"][:320], dtype=np.float64)
        low = np.asarray(test_data["low"][:320], dtype=np.float64)
        close = np.asarray(test_data["close"][:320], dtype=np.float64)

        batch_pd, batch_percent = mp.price_density_market_noise(high, low, close, 12, 40)
        stream = mp.PriceDensityMarketNoiseStream(12, 40)
        stream_pd = []
        stream_percent = []

        for h, l, c in zip(high, low, close):
            out = stream.update(float(h), float(l), float(c))
            if out is None:
                stream_pd.append(np.nan)
                stream_percent.append(np.nan)
            else:
                a, b = out
                stream_pd.append(a)
                stream_percent.append(b)

        np.testing.assert_allclose(
            stream_pd, batch_pd, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            stream_percent, batch_percent, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        high = np.asarray(test_data["high"][:320], dtype=np.float64)
        low = np.asarray(test_data["low"][:320], dtype=np.float64)
        close = np.asarray(test_data["close"][:320], dtype=np.float64)

        result = mp.price_density_market_noise_batch(
            high,
            low,
            close,
            length_range=(12, 12, 0),
            eval_period_range=(40, 40, 0),
        )

        assert result["rows"] == 1
        assert result["cols"] == len(close)
        assert result["price_density"].shape == (1, len(close))
        assert result["price_density_percent"].shape == (1, len(close))
        np.testing.assert_array_equal(result["lengths"], np.array([12], dtype=np.uint64))
        np.testing.assert_array_equal(
            result["eval_periods"], np.array([40], dtype=np.uint64)
        )

        single_pd, single_percent = mp.price_density_market_noise(high, low, close, 12, 40)
        np.testing.assert_allclose(
            result["price_density"][0],
            single_pd,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            result["price_density_percent"][0],
            single_percent,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )
