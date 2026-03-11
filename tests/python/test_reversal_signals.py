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


class TestReversalSignals:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        open_ = test_data["open"][:256]
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        close = test_data["close"][:256]
        volume = test_data["volume"][:256]

        buy_signal, sell_signal, stepped_ma, state = mp.reversal_signals(
            open_, high, low, close, volume, 12, 3, True, 50, "EMA", 33
        )

        assert len(buy_signal) == len(close)
        assert len(sell_signal) == len(close)
        assert len(stepped_ma) == len(close)
        assert len(state) == len(close)
        assert np.isfinite(stepped_ma).any()
        assert np.isfinite(state).any()

    def test_invalid_params(self, test_data):
        open_ = test_data["open"][:64]
        high = test_data["high"][:64]
        low = test_data["low"][:64]
        close = test_data["close"][:64]
        volume = test_data["volume"][:64]

        with pytest.raises(ValueError, match="Invalid lookback_period"):
            mp.reversal_signals(open_, high, low, close, volume, 0, 3, True, 50, "EMA", 33)

        with pytest.raises(ValueError, match="Invalid trend_ma_period"):
            mp.reversal_signals(open_, high, low, close, volume, 12, 3, True, 0, "EMA", 33)

        with pytest.raises(ValueError, match="Invalid trend_ma_type"):
            mp.reversal_signals(
                open_, high, low, close, volume, 12, 3, True, 50, "BAD", 33
            )

    def test_batch_single_param_matches_single(self, test_data):
        open_ = test_data["open"][:220]
        high = test_data["high"][:220]
        low = test_data["low"][:220]
        close = test_data["close"][:220]
        volume = test_data["volume"][:220]

        batch = mp.reversal_signals_batch(
            open_,
            high,
            low,
            close,
            volume,
            lookback_period_range=(12, 12, 0),
            confirmation_period_range=(3, 3, 0),
            trend_ma_period_range=(50, 50, 0),
            ma_step_period_range=(33, 33, 0),
            use_volume_confirmation=True,
            trend_ma_type="EMA",
        )
        single = mp.reversal_signals(open_, high, low, close, volume, 12, 3, True, 50, "EMA", 33)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(batch["lookback_periods"], np.array([12], dtype=np.uint64))
        np.testing.assert_array_equal(
            batch["confirmation_periods"], np.array([3], dtype=np.uint64)
        )
        np.testing.assert_array_equal(batch["trend_ma_periods"], np.array([50], dtype=np.uint64))
        np.testing.assert_array_equal(batch["ma_step_periods"], np.array([33], dtype=np.uint64))

        for key, expected in zip(("buy_signal", "sell_signal", "stepped_ma", "state"), single):
            np.testing.assert_allclose(
                batch[key][0], expected, rtol=1e-12, atol=1e-12, equal_nan=True
            )

    def test_stream_matches_single(self, test_data):
        open_ = test_data["open"][:220]
        high = test_data["high"][:220]
        low = test_data["low"][:220]
        close = test_data["close"][:220]
        volume = test_data["volume"][:220]

        batch = mp.reversal_signals(open_, high, low, close, volume, 12, 3, True, 50, "VWMA", 33)
        stream = mp.ReversalSignalsStream(12, 3, True, 50, "VWMA", 33)

        streamed = [[], [], [], []]
        for o, h, l, c, v in zip(open_, high, low, close, volume):
            point = stream.update(float(o), float(h), float(l), float(c), float(v))
            if point is None:
                for arr in streamed:
                    arr.append(np.nan)
            else:
                for arr, value in zip(streamed, point):
                    arr.append(value)

        for expected, got in zip(batch, streamed):
            np.testing.assert_allclose(expected, got, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_batch_multi_param_contract(self, test_data):
        open_ = test_data["open"][:180]
        high = test_data["high"][:180]
        low = test_data["low"][:180]
        close = test_data["close"][:180]
        volume = test_data["volume"][:180]

        batch = mp.reversal_signals_batch(
            open_,
            high,
            low,
            close,
            volume,
            lookback_period_range=(10, 12, 2),
            confirmation_period_range=(2, 3, 1),
            trend_ma_period_range=(30, 30, 0),
            ma_step_period_range=(20, 20, 0),
            use_volume_confirmation=False,
            trend_ma_type="SMA",
        )

        assert batch["rows"] == 4
        assert batch["cols"] == len(close)
        assert batch["buy_signal"].shape == (4, len(close))
        assert batch["stepped_ma"].shape == (4, len(close))
