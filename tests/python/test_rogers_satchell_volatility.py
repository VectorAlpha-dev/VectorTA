"""
Python binding tests for Rogers-Satchell volatility.
"""
import pytest
import numpy as np

from test_utils import load_test_data

try:
    import vector_ta as mp
except ImportError:
    try:
        import vector_ta as mp
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


class TestRogersSatchellVolatility:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]

        rs, signal = mp.rogers_satchell_volatility(
            open_,
            high,
            low,
            close,
            lookback=14,
            signal_length=6,
        )

        assert len(rs) == len(close)
        assert len(signal) == len(close)

        rs_valid = np.flatnonzero(~np.isnan(rs))
        signal_valid = np.flatnonzero(~np.isnan(signal))
        assert rs_valid.size > 0
        assert signal_valid.size > 0
        assert int(rs_valid[0]) >= 13
        assert int(signal_valid[0]) >= 18

        tail_start = min(int(signal_valid[0]) + 32, len(signal))
        assert not np.any(np.isnan(rs[tail_start:]))
        assert not np.any(np.isnan(signal[tail_start:]))

    def test_kernel_parity(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]

        rs_auto, signal_auto = mp.rogers_satchell_volatility(
            open_,
            high,
            low,
            close,
            lookback=20,
            signal_length=7,
        )
        rs_scalar, signal_scalar = mp.rogers_satchell_volatility(
            open_,
            high,
            low,
            close,
            lookback=20,
            signal_length=7,
            kernel="scalar",
        )

        np.testing.assert_allclose(
            rs_auto, rs_scalar, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            signal_auto, signal_scalar, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_length_mismatch(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]

        with pytest.raises(ValueError, match="length mismatch"):
            mp.rogers_satchell_volatility(open_[:-1], high, low, close, lookback=14)

    def test_invalid_params(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]

        with pytest.raises(ValueError, match="Invalid lookback"):
            mp.rogers_satchell_volatility(
                open_, high, low, close, lookback=0, signal_length=6
            )

        with pytest.raises(ValueError, match="Invalid signal length"):
            mp.rogers_satchell_volatility(
                open_, high, low, close, lookback=14, signal_length=0
            )

    def test_streaming_matches_batch(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]

        batch_rs, batch_signal = mp.rogers_satchell_volatility(
            open_,
            high,
            low,
            close,
            lookback=20,
            signal_length=5,
        )

        stream = mp.RogersSatchellVolatilityStream(20, 5)
        stream_rs = []
        stream_signal = []

        for o, h, l, c in zip(open_, high, low, close):
            out = stream.update(o, h, l, c)
            if out is None:
                stream_rs.append(np.nan)
                stream_signal.append(np.nan)
            else:
                rs, signal = out
                stream_rs.append(rs)
                stream_signal.append(signal)

        np.testing.assert_allclose(
            stream_rs, batch_rs, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            stream_signal, batch_signal, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]

        result = mp.rogers_satchell_volatility_batch(
            open_,
            high,
            low,
            close,
            lookback_range=(20, 20, 0),
            signal_length_range=(6, 6, 0),
        )

        assert "rs" in result
        assert "signal" in result
        assert "lookbacks" in result
        assert "signal_lengths" in result
        assert result["rows"] == 1
        assert result["cols"] == len(close)
        assert result["rs"].shape == (1, len(close))
        assert result["signal"].shape == (1, len(close))

        single_rs, single_signal = mp.rogers_satchell_volatility(
            open_,
            high,
            low,
            close,
            lookback=20,
            signal_length=6,
        )

        np.testing.assert_allclose(
            result["rs"][0], single_rs, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            result["signal"][0], single_signal, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_array_equal(result["lookbacks"], np.array([20], dtype=np.uint64))
        np.testing.assert_array_equal(
            result["signal_lengths"], np.array([6], dtype=np.uint64)
        )

    def test_batch_metadata_multiple_ranges(self, test_data):
        open_ = test_data["open"][:256]
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        close = test_data["close"][:256]

        result = mp.rogers_satchell_volatility_batch(
            open_,
            high,
            low,
            close,
            lookback_range=(10, 14, 2),
            signal_length_range=(4, 6, 2),
        )

        assert result["rows"] == 6
        assert result["cols"] == len(close)
        assert result["rs"].shape == (6, len(close))
        assert result["signal"].shape == (6, len(close))
        np.testing.assert_array_equal(
            result["lookbacks"], np.array([10, 10, 12, 12, 14, 14], dtype=np.uint64)
        )
        np.testing.assert_array_equal(
            result["signal_lengths"], np.array([4, 6, 4, 6, 4, 6], dtype=np.uint64)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
