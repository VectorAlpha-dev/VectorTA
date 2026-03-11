import numpy as np
import pytest

from test_utils import load_test_data

try:
    import vector_ta as mp
except ImportError:
    try:
        import my_project as mp
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


class TestPossibleRsi:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        close = np.asarray(test_data["close"][:384], dtype=np.float64)
        value, buy_level, sell_level, middle_level, state, long_signal, short_signal = (
            mp.possible_rsi(close)
        )

        assert len(value) == len(close)
        assert len(buy_level) == len(close)
        assert len(sell_level) == len(close)
        assert len(middle_level) == len(close)
        assert len(state) == len(close)
        assert len(long_signal) == len(close)
        assert len(short_signal) == len(close)
        assert np.isfinite(value).any()
        assert np.isfinite(value[-1])
        assert np.isfinite(buy_level[-1])
        assert np.isfinite(sell_level[-1])
        assert np.isfinite(middle_level[-1])
        assert np.isfinite(state[-1])

    def test_invalid_params(self, test_data):
        close = np.asarray(test_data["close"][:96], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid period"):
            mp.possible_rsi(close, period=0)

        with pytest.raises(ValueError, match="Invalid RSI mode"):
            mp.possible_rsi(close, rsi_mode="bad_mode")

    def test_stream_matches_batch(self, test_data):
        close = np.asarray(test_data["close"][:360], dtype=np.float64)
        batch = mp.possible_rsi(
            close,
            period=28,
            rsi_mode="cutler",
            norm_period=90,
            normalization_mode="softmax",
            normalization_length=12,
            nonlag_period=11,
            dynamic_zone_period=18,
            signal_type="levels_crossover",
            run_highpass=True,
            highpass_period=13,
        )

        stream = mp.PossibleRsiStream(
            period=28,
            rsi_mode="cutler",
            norm_period=90,
            normalization_mode="softmax",
            normalization_length=12,
            nonlag_period=11,
            dynamic_zone_period=18,
            signal_type="levels_crossover",
            run_highpass=True,
            highpass_period=13,
        )

        value = []
        buy_level = []
        sell_level = []
        middle_level = []
        state = []
        long_signal = []
        short_signal = []
        for sample in close:
            out = stream.update(float(sample))
            if out is None:
                value.append(np.nan)
                buy_level.append(np.nan)
                sell_level.append(np.nan)
                middle_level.append(np.nan)
                state.append(np.nan)
                long_signal.append(0.0)
                short_signal.append(0.0)
            else:
                v, b, s, m, st, lng, sht = out
                value.append(v)
                buy_level.append(b)
                sell_level.append(s)
                middle_level.append(m)
                state.append(st)
                long_signal.append(lng)
                short_signal.append(sht)

        start = int(np.flatnonzero(~np.isnan(state))[0])
        np.testing.assert_allclose(
            value[start:], batch[0][start:], rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            buy_level[start:], batch[1][start:], rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            sell_level[start:], batch[2][start:], rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            middle_level[start:], batch[3][start:], rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            state[start:], batch[4][start:], rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        close = np.asarray(test_data["close"][:320], dtype=np.float64)
        batch = mp.possible_rsi_batch(
            close,
            period_range=(32, 32, 0),
            norm_period_range=(100, 100, 0),
            normalization_length_range=(15, 15, 0),
            nonlag_period_range=(15, 15, 0),
            dynamic_zone_period_range=(20, 20, 0),
        )
        single = mp.possible_rsi(close)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        np.testing.assert_allclose(
            batch["value"][0], single[0], rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            batch["state"][0], single[4], rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_metadata_multiple_ranges(self, test_data):
        close = np.asarray(test_data["close"][:300], dtype=np.float64)
        batch = mp.possible_rsi_batch(
            close,
            period_range=(30, 32, 1),
            nonlag_period_range=(14, 16, 1),
        )

        assert batch["rows"] == 9
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(
            batch["periods"],
            np.array([30, 30, 30, 31, 31, 31, 32, 32, 32], dtype=np.uint64),
        )
        np.testing.assert_array_equal(
            batch["nonlag_periods"],
            np.array([14, 15, 16, 14, 15, 16, 14, 15, 16], dtype=np.uint64),
        )
        assert batch["rsi_modes"] == ["regular"] * 9
