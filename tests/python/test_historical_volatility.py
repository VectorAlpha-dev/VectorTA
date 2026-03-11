import numpy as np
import pytest

from test_utils import load_test_data

try:
    import my_project as mp
except ImportError:
    try:
        import vector_ta as mp
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


class TestHistoricalVolatility:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        close = test_data["close"]
        values = mp.historical_volatility(close, lookback=20, annualization_days=250.0)

        assert len(values) == len(close)
        valid = np.flatnonzero(~np.isnan(values))
        assert valid.size > 0
        first_valid = int(valid[0])
        assert first_valid >= 20

        tail_start = min(first_valid + 32, len(values))
        assert not np.any(np.isnan(values[tail_start:]))

    def test_kernel_parity(self, test_data):
        close = test_data["close"]

        auto = mp.historical_volatility(close, lookback=30, annualization_days=252.0)
        scalar = mp.historical_volatility(
            close,
            lookback=30,
            annualization_days=252.0,
            kernel="scalar",
        )

        np.testing.assert_allclose(auto, scalar, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_invalid_lookback(self, test_data):
        close = test_data["close"]

        with pytest.raises(ValueError, match="Invalid lookback"):
            mp.historical_volatility(close, lookback=0, annualization_days=250.0)

    def test_invalid_annualization_days(self, test_data):
        close = test_data["close"]

        with pytest.raises(ValueError, match="Invalid annualization_days"):
            mp.historical_volatility(close, lookback=20, annualization_days=0.0)

    def test_streaming_matches_batch(self, test_data):
        close = test_data["close"]
        batch = mp.historical_volatility(close, lookback=20, annualization_days=250.0)

        stream = mp.HistoricalVolatilityStream(20, 250.0)
        streamed = []
        for value in close:
            out = stream.update(value)
            streamed.append(np.nan if out is None else out)

        np.testing.assert_allclose(
            streamed, batch, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        close = test_data["close"]
        result = mp.historical_volatility_batch(
            close,
            lookback_range=(20, 20, 0),
            annualization_days_range=(250.0, 250.0, 0.0),
        )

        assert result["rows"] == 1
        assert result["cols"] == len(close)
        assert result["values"].shape == (1, len(close))
        np.testing.assert_array_equal(result["lookbacks"], np.array([20], dtype=np.uint64))
        np.testing.assert_allclose(
            result["annualization_days"], np.array([250.0]), rtol=0, atol=0
        )

        single = mp.historical_volatility(close, lookback=20, annualization_days=250.0)
        np.testing.assert_allclose(
            result["values"][0], single, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_multiple_metadata(self, test_data):
        close = test_data["close"][:256]
        result = mp.historical_volatility_batch(
            close,
            lookback_range=(10, 14, 2),
            annualization_days_range=(250.0, 252.0, 2.0),
        )

        assert result["rows"] == 6
        assert result["cols"] == len(close)
        assert result["values"].shape == (6, len(close))
        np.testing.assert_array_equal(
            result["lookbacks"], np.array([10, 10, 12, 12, 14, 14], dtype=np.uint64)
        )
        np.testing.assert_allclose(
            result["annualization_days"],
            np.array([250.0, 252.0, 250.0, 252.0, 250.0, 252.0]),
            rtol=0,
            atol=0,
        )

    def test_invalid_window_recovers(self, test_data):
        close = test_data["close"][:80].copy()
        close[30] = np.nan

        values = mp.historical_volatility(close, lookback=10, annualization_days=250.0)
        assert np.isnan(values[30])
        assert np.isnan(values[39])
        assert np.isnan(values[40])
        assert np.isfinite(values[41])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
