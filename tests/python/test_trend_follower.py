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


class TestTrendFollower:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        high = np.asarray(test_data["high"][:256], dtype=np.float64)
        low = np.asarray(test_data["low"][:256], dtype=np.float64)
        close = np.asarray(test_data["close"][:256], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:256], dtype=np.float64)

        values = mp.trend_follower(high, low, close, volume)

        assert len(values) == len(close)
        assert np.isfinite(values).any()

    def test_kernel_parity(self, test_data):
        high = np.asarray(test_data["high"][:256], dtype=np.float64)
        low = np.asarray(test_data["low"][:256], dtype=np.float64)
        close = np.asarray(test_data["close"][:256], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:256], dtype=np.float64)

        auto_values = mp.trend_follower(high, low, close, volume)
        scalar_values = mp.trend_follower(
            high, low, close, volume, kernel="scalar"
        )

        np.testing.assert_allclose(
            auto_values, scalar_values, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_stream_matches_batch(self, test_data):
        high = np.asarray(test_data["high"][:192], dtype=np.float64)
        low = np.asarray(test_data["low"][:192], dtype=np.float64)
        close = np.asarray(test_data["close"][:192], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:192], dtype=np.float64)

        batch = mp.trend_follower(high, low, close, volume)
        stream = mp.TrendFollowerStream()
        stream_values = []

        for h, l, c, v in zip(high, low, close, volume):
            out = stream.update(float(h), float(l), float(c), float(v))
            stream_values.append(np.nan if out is None else out)

        np.testing.assert_allclose(
            stream_values, batch, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        high = np.asarray(test_data["high"][:192], dtype=np.float64)
        low = np.asarray(test_data["low"][:192], dtype=np.float64)
        close = np.asarray(test_data["close"][:192], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:192], dtype=np.float64)

        result = mp.trend_follower_batch(
            high,
            low,
            close,
            volume,
            trend_period_range=(20, 20, 0),
            ma_period_range=(20, 20, 0),
            channel_rate_percent_range=(1.0, 1.0, 0.0),
            linear_regression_period_range=(5, 5, 0),
            matype="ema",
            use_linear_regression=True,
        )

        assert result["rows"] == 1
        assert result["cols"] == len(close)
        assert result["values"].shape == (1, len(close))

        single = mp.trend_follower(high, low, close, volume)
        np.testing.assert_allclose(
            result["values"][0], single, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_vwma_depends_on_volume(self, test_data):
        high = np.asarray(test_data["high"][:160], dtype=np.float64)
        low = np.asarray(test_data["low"][:160], dtype=np.float64)
        close = np.asarray(test_data["close"][:160], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:160], dtype=np.float64)
        reversed_volume = volume[::-1].copy()

        a = mp.trend_follower(
            high,
            low,
            close,
            volume,
            matype="vwma",
            ma_period=12,
            use_linear_regression=False,
        )
        b = mp.trend_follower(
            high,
            low,
            close,
            reversed_volume,
            matype="vwma",
            ma_period=12,
            use_linear_regression=False,
        )

        assert np.any(
            np.abs(np.nan_to_num(a, nan=0.0) - np.nan_to_num(b, nan=0.0)) > 1e-9
        )

    def test_invalid_matype(self, test_data):
        high = np.asarray(test_data["high"][:64], dtype=np.float64)
        low = np.asarray(test_data["low"][:64], dtype=np.float64)
        close = np.asarray(test_data["close"][:64], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:64], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid MA type"):
            mp.trend_follower(high, low, close, volume, matype="hma")
