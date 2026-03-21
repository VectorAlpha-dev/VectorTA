import numpy as np
import pytest

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


class TestCandleStrengthOscillator:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        open_ = test_data["open"][:256]
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        close = test_data["close"][:256]

        strength, highs, lows, mid, long_signal, short_signal = mp.candle_strength_oscillator(
            open_, high, low, close
        )

        assert len(strength) == len(close)
        assert len(highs) == len(close)
        assert len(lows) == len(close)
        assert len(mid) == len(close)
        assert len(long_signal) == len(close)
        assert len(short_signal) == len(close)

        strength_valid = np.flatnonzero(~np.isnan(strength))
        highs_valid = np.flatnonzero(~np.isnan(highs))
        lows_valid = np.flatnonzero(~np.isnan(lows))
        mid_valid = np.flatnonzero(~np.isnan(mid))
        assert strength_valid.size > 0
        assert highs_valid.size > 0
        assert lows_valid.size > 0
        assert mid_valid.size > 0
        assert int(strength_valid[0]) == 55
        assert int(highs_valid[0]) == 104
        assert int(lows_valid[0]) == 104
        assert int(mid_valid[0]) == 104

    def test_invalid_params(self, test_data):
        open_ = test_data["open"][:128]
        high = test_data["high"][:128]
        low = test_data["low"][:128]
        close = test_data["close"][:128]

        with pytest.raises(ValueError, match="Invalid period"):
            mp.candle_strength_oscillator(open_, high, low, close, period=0)

        with pytest.raises(ValueError, match="Invalid atr_length"):
            mp.candle_strength_oscillator(
                open_, high, low, close, atr_enabled=True, atr_length=0
            )

        with pytest.raises(ValueError, match="Invalid mode"):
            mp.candle_strength_oscillator(open_, high, low, close, mode="bad")

    def test_stream_matches_batch(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]
        batch = mp.candle_strength_oscillator(
            open_, high, low, close, period=24, atr_enabled=True, atr_length=10
        )

        stream = mp.CandleStrengthOscillatorStream(24, True, 10, "bollinger")
        strength = []
        highs = []
        lows = []
        mid = []
        long_signal = []
        short_signal = []
        for o, h, l, c in zip(open_, high, low, close):
            out = stream.update(float(o), float(h), float(l), float(c))
            if out is None:
                strength.append(np.nan)
                highs.append(np.nan)
                lows.append(np.nan)
                mid.append(np.nan)
                long_signal.append(0.0)
                short_signal.append(0.0)
            else:
                s, hi, lo, md, lng, sht = out
                strength.append(s)
                highs.append(hi)
                lows.append(lo)
                mid.append(md)
                long_signal.append(lng)
                short_signal.append(sht)

        np.testing.assert_allclose(strength, batch[0], rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(highs, batch[1], rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(lows, batch[2], rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(mid, batch[3], rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(long_signal, batch[4], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(short_signal, batch[5], rtol=1e-12, atol=1e-12)

    def test_batch_single_param_matches_single(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]
        batch = mp.candle_strength_oscillator_batch(
            open_,
            high,
            low,
            close,
            period_range=(30, 30, 0),
            atr_enabled=True,
            atr_length_range=(14, 14, 0),
            mode="donchian",
        )
        single = mp.candle_strength_oscillator(
            open_, high, low, close, period=30, atr_enabled=True, atr_length=14, mode="donchian"
        )

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        np.testing.assert_allclose(batch["strength"][0], single[0], rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(batch["highs"][0], single[1], rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(batch["lows"][0], single[2], rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(batch["mid"][0], single[3], rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(batch["long_signal"][0], single[4], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(batch["short_signal"][0], single[5], rtol=1e-12, atol=1e-12)

    def test_batch_metadata_multiple_ranges(self, test_data):
        open_ = test_data["open"][:256]
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        close = test_data["close"][:256]
        batch = mp.candle_strength_oscillator_batch(
            open_,
            high,
            low,
            close,
            period_range=(30, 34, 2),
            atr_enabled=True,
            atr_length_range=(14, 14, 0),
            mode="bollinger",
        )

        assert batch["rows"] == 3
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(
            batch["periods"], np.array([30, 32, 34], dtype=np.uint64)
        )
        np.testing.assert_array_equal(
            batch["atr_lengths"], np.array([14, 14, 14], dtype=np.uint64)
        )
        np.testing.assert_array_equal(
            batch["atr_enableds"], np.array([True, True, True], dtype=np.bool_)
        )
        assert batch["modes"] == ["bollinger", "bollinger", "bollinger"]
