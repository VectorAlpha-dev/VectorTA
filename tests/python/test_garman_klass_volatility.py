"""
Python binding tests for Garman-Klass volatility.
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


class TestGarmanKlassVolatility:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]

        values = mp.garman_klass_volatility(open_, high, low, close, lookback=14)

        assert len(values) == len(close)

        valid = np.flatnonzero(~np.isnan(values))
        assert valid.size > 0
        first_valid = int(valid[0])
        assert first_valid >= 13

        tail_start = min(first_valid + 32, len(values))
        assert not np.any(np.isnan(values[tail_start:]))

    def test_kernel_parity(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]

        auto = mp.garman_klass_volatility(open_, high, low, close, lookback=20)
        scalar = mp.garman_klass_volatility(
            open_, high, low, close, lookback=20, kernel="scalar"
        )

        np.testing.assert_allclose(auto, scalar, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_length_mismatch(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]

        with pytest.raises(ValueError, match="length mismatch"):
            mp.garman_klass_volatility(open_[:-1], high, low, close, lookback=14)

    def test_invalid_lookback(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]

        with pytest.raises(ValueError, match="Invalid lookback"):
            mp.garman_klass_volatility(open_, high, low, close, lookback=0)

    def test_streaming_matches_batch(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]

        batch = mp.garman_klass_volatility(open_, high, low, close, lookback=20)

        stream = mp.GarmanKlassVolatilityStream(20)
        streamed = []

        for o, h, l, c in zip(open_, high, low, close):
            out = stream.update(o, h, l, c)
            streamed.append(np.nan if out is None else out)

        np.testing.assert_allclose(
            streamed, batch, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]

        result = mp.garman_klass_volatility_batch(
            open_,
            high,
            low,
            close,
            lookback_range=(20, 20, 0),
        )

        assert "values" in result
        assert "lookbacks" in result
        assert result["rows"] == 1
        assert result["cols"] == len(close)
        assert result["values"].shape == (1, len(close))

        single = mp.garman_klass_volatility(open_, high, low, close, lookback=20)

        np.testing.assert_allclose(
            result["values"][0], single, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_array_equal(result["lookbacks"], np.array([20], dtype=np.uint64))

    def test_batch_multiple_lookbacks_metadata(self, test_data):
        open_ = test_data["open"][:256]
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        close = test_data["close"][:256]

        result = mp.garman_klass_volatility_batch(
            open_,
            high,
            low,
            close,
            lookback_range=(10, 14, 2),
        )

        assert result["rows"] == 3
        assert result["cols"] == len(close)
        assert result["values"].shape == (3, len(close))
        np.testing.assert_array_equal(
            result["lookbacks"], np.array([10, 12, 14], dtype=np.uint64)
        )

        single = mp.garman_klass_volatility(open_, high, low, close, lookback=10)
        np.testing.assert_allclose(
            result["values"][0], single, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_invalid_bar_window_recovers(self, test_data):
        open_ = test_data["open"][:80].copy()
        high = test_data["high"][:80].copy()
        low = test_data["low"][:80].copy()
        close = test_data["close"][:80].copy()
        open_[30] = np.nan
        high[30] = np.nan
        low[30] = np.nan
        close[30] = np.nan

        values = mp.garman_klass_volatility(open_, high, low, close, lookback=10)
        assert np.isnan(values[30])
        assert np.isnan(values[39])
        assert np.isfinite(values[40])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
