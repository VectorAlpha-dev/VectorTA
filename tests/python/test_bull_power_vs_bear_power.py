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


class TestBullPowerVsBearPower:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]
        values = mp.bull_power_vs_bear_power(open_, high, low, close, period=5)

        assert len(values) == len(close)
        valid = np.flatnonzero(~np.isnan(values))
        assert valid.size > 0
        assert int(valid[0]) >= 4
        tail_start = min(int(valid[0]) + 32, len(values))
        assert not np.any(np.isnan(values[tail_start:]))

    def test_kernel_parity(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]

        auto = mp.bull_power_vs_bear_power(open_, high, low, close, period=9)
        scalar = mp.bull_power_vs_bear_power(
            open_, high, low, close, period=9, kernel="scalar"
        )

        np.testing.assert_allclose(auto, scalar, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_invalid_period(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]

        with pytest.raises(ValueError, match="Invalid period"):
            mp.bull_power_vs_bear_power(open_, high, low, close, period=0)

    def test_length_mismatch(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]

        with pytest.raises(ValueError, match="length mismatch"):
            mp.bull_power_vs_bear_power(open_[:-1], high, low, close, period=5)

    def test_streaming_matches_batch(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]
        batch = mp.bull_power_vs_bear_power(open_, high, low, close, period=7)

        stream = mp.BullPowerVsBearPowerStream(7)
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
        result = mp.bull_power_vs_bear_power_batch(
            open_,
            high,
            low,
            close,
            period_range=(5, 5, 0),
        )

        assert result["rows"] == 1
        assert result["cols"] == len(close)
        assert result["values"].shape == (1, len(close))
        np.testing.assert_array_equal(result["periods"], np.array([5], dtype=np.uint64))

        single = mp.bull_power_vs_bear_power(open_, high, low, close, period=5)
        np.testing.assert_allclose(
            result["values"][0], single, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_multiple_metadata(self, test_data):
        open_ = test_data["open"][:256]
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        close = test_data["close"][:256]
        result = mp.bull_power_vs_bear_power_batch(
            open_,
            high,
            low,
            close,
            period_range=(5, 9, 2),
        )

        assert result["rows"] == 3
        assert result["cols"] == len(close)
        assert result["values"].shape == (3, len(close))
        np.testing.assert_array_equal(
            result["periods"], np.array([5, 7, 9], dtype=np.uint64)
        )

    def test_invalid_window_recovers(self, test_data):
        open_ = test_data["open"][:96].copy()
        high = test_data["high"][:96].copy()
        low = test_data["low"][:96].copy()
        close = test_data["close"][:96].copy()
        close[30] = np.nan

        values = mp.bull_power_vs_bear_power(open_, high, low, close, period=5)
        assert np.isnan(values[30])
        assert np.isnan(values[34])
        assert np.isfinite(values[35])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
