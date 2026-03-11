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


class TestFvgPositioningAverage:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        open_ = test_data["open"][:256]
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        close = test_data["close"][:256]

        bull_average, bear_average, bull_mid, bear_mid = mp.fvg_positioning_average(
            open_, high, low, close, 30, "Bar Count", 0.25
        )

        assert len(bull_average) == len(close)
        assert len(bear_average) == len(close)
        assert len(bull_mid) == len(close)
        assert len(bear_mid) == len(close)
        assert np.isfinite(bull_average).any() or np.isfinite(bear_average).any()

    def test_invalid_params(self, test_data):
        open_ = test_data["open"][:64]
        high = test_data["high"][:64]
        low = test_data["low"][:64]
        close = test_data["close"][:64]

        with pytest.raises(ValueError, match="Invalid lookback"):
            mp.fvg_positioning_average(open_, high, low, close, 0, "Bar Count", 0.25)

        with pytest.raises(ValueError, match="Invalid lookback_type"):
            mp.fvg_positioning_average(open_, high, low, close, 30, "bad", 0.25)

        with pytest.raises(ValueError, match="Invalid atr_multiplier"):
            mp.fvg_positioning_average(open_, high, low, close, 30, "Bar Count", -1.0)

    def test_lookback_mode_behavior(self, test_data):
        open_ = test_data["open"][:320]
        high = test_data["high"][:320]
        low = test_data["low"][:320]
        close = test_data["close"][:320]

        bull_bar, bear_bar, _, _ = mp.fvg_positioning_average(
            open_, high, low, close, 30, "Bar Count", 0.25
        )
        bull_count, bear_count, _, _ = mp.fvg_positioning_average(
            open_, high, low, close, 3, "FVG Count", 0.25
        )

        assert not np.array_equal(bull_bar, bull_count)
        assert len(bear_bar) == len(bear_count)

    def test_batch_single_param_matches_single(self, test_data):
        open_ = test_data["open"][:256]
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        close = test_data["close"][:256]

        batch = mp.fvg_positioning_average_batch(
            open_,
            high,
            low,
            close,
            lookback_range=(30, 30, 0),
            atr_multiplier_range=(0.25, 0.25, 0.0),
            lookback_type="Bar Count",
        )
        single = mp.fvg_positioning_average(open_, high, low, close, 30, "Bar Count", 0.25)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(batch["lookbacks"], np.array([30], dtype=np.uint64))
        assert batch["lookback_types"] == ["Bar Count"]
        np.testing.assert_allclose(
            batch["atr_multipliers"], np.array([0.25]), rtol=0.0, atol=0.0
        )

        for key, expected in zip(
            ("bull_average", "bear_average", "bull_mid", "bear_mid"), single
        ):
            np.testing.assert_allclose(
                batch[key][0], expected, rtol=1e-12, atol=1e-12, equal_nan=True
            )

    def test_stream_matches_batch(self, test_data):
        open_ = test_data["open"][:220]
        high = test_data["high"][:220]
        low = test_data["low"][:220]
        close = test_data["close"][:220]

        batch = mp.fvg_positioning_average(open_, high, low, close, 30, "Bar Count", 0.25)
        stream = mp.FvgPositioningAverageStream(30, "Bar Count", 0.25)

        streamed = [[], [], [], []]
        for o, h, l, c in zip(open_, high, low, close):
            point = stream.update(float(o), float(h), float(l), float(c))
            if point is None:
                for arr in streamed:
                    arr.append(np.nan)
            else:
                for arr, value in zip(streamed, point):
                    arr.append(value)

        for expected, got in zip(batch, streamed):
            np.testing.assert_allclose(expected, got, rtol=1e-12, atol=1e-12, equal_nan=True)
