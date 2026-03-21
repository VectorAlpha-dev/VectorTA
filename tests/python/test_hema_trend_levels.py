import numpy as np
import pytest

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


def sample_ohlc(length=256):
    idx = np.arange(length, dtype=np.float64)
    open_ = 100.0 + idx * 0.05 + np.sin(idx * 0.09) * 1.3
    close = open_ + np.cos(idx * 0.07) * 1.1
    high = np.maximum(open_, close) + 0.65 + np.abs(np.sin(idx * 0.03)) * 0.25
    low = np.minimum(open_, close) - 0.65 - np.abs(np.cos(idx * 0.05)) * 0.25
    return open_, high, low, close


class TestHemaTrendLevels:
    def test_output_contract(self):
        open_, high, low, close = sample_ohlc()
        outputs = mp.hema_trend_levels(open_, high, low, close, fast_length=20, slow_length=40)
        keys = [
            "fast_hema",
            "slow_hema",
            "trend_direction",
            "bar_state",
            "bullish_crossover",
            "bearish_crossunder",
            "box_offset",
            "bull_box_top",
            "bull_box_bottom",
            "bear_box_top",
            "bear_box_bottom",
            "bullish_test",
            "bearish_test",
            "bullish_test_level",
            "bearish_test_level",
        ]

        assert set(outputs.keys()) == set(keys)
        for key in keys:
            series = outputs[key]
            assert len(series) == len(close)

        fast_hema = outputs["fast_hema"]
        slow_hema = outputs["slow_hema"]
        trend_direction = outputs["trend_direction"]
        bar_state = outputs["bar_state"]
        bullish_crossover = outputs["bullish_crossover"]
        bearish_crossunder = outputs["bearish_crossunder"]
        bull_box_top = outputs["bull_box_top"]
        bear_box_top = outputs["bear_box_top"]
        bullish_test = outputs["bullish_test"]
        bearish_test = outputs["bearish_test"]
        bullish_test_level = outputs["bullish_test_level"]
        bearish_test_level = outputs["bearish_test_level"]
        assert np.isfinite(fast_hema).any()
        assert np.isfinite(slow_hema).any()
        assert np.isfinite(bull_box_top).any() or np.isfinite(bear_box_top).any()

        for series in (trend_direction, bar_state):
            valid = series[np.isfinite(series)]
            assert np.all((valid == -1.0) | (valid == 0.0) | (valid == 1.0))

        for signal in (bullish_crossover, bearish_crossunder, bullish_test, bearish_test):
            valid = signal[np.isfinite(signal)]
            assert np.all((valid == 0.0) | (valid == 1.0))

        assert np.all(np.isnan(bullish_test_level) | (bullish_test_level >= 0.0))
        assert np.all(np.isnan(bearish_test_level) | (bearish_test_level >= 0.0))

    def test_kernel_parity(self):
        open_, high, low, close = sample_ohlc(192)
        auto = mp.hema_trend_levels(open_, high, low, close, 20, 40)
        scalar = mp.hema_trend_levels(open_, high, low, close, 20, 40, kernel="scalar")
        assert auto.keys() == scalar.keys()
        for key in auto:
            np.testing.assert_allclose(
                auto[key], scalar[key], rtol=1e-12, atol=1e-12, equal_nan=True
            )

    def test_invalid_inputs(self):
        open_, high, low, close = sample_ohlc(32)

        with pytest.raises(ValueError, match="Inconsistent slice lengths"):
            mp.hema_trend_levels(open_[:-1], high, low, close, 20, 40)

        all_nan = np.full(64, np.nan, dtype=np.float64)
        with pytest.raises(ValueError, match="All values are NaN"):
            mp.hema_trend_levels(all_nan, all_nan, all_nan, all_nan, 20, 40)

        with pytest.raises(ValueError, match="Invalid fast_length"):
            mp.hema_trend_levels(open_, high, low, close, 0, 40)

    def test_stream_matches_batch_with_reset(self):
        open_, high, low, close = sample_ohlc(220)
        open_[90] = np.nan
        high[90] = np.nan
        low[90] = np.nan
        close[90] = np.nan

        batch = mp.hema_trend_levels(open_, high, low, close, 20, 40)
        stream = mp.HemaTrendLevelsStream(fast_length=20, slow_length=40)
        keys = [
            "fast_hema",
            "slow_hema",
            "trend_direction",
            "bar_state",
            "bullish_crossover",
            "bearish_crossunder",
            "box_offset",
            "bull_box_top",
            "bull_box_bottom",
            "bear_box_top",
            "bear_box_bottom",
            "bullish_test",
            "bearish_test",
            "bullish_test_level",
            "bearish_test_level",
        ]

        streamed = {key: [] for key in keys}
        for values in zip(open_, high, low, close):
            out = stream.update(*map(float, values))
            if out is None:
                for key in keys:
                    streamed[key].append(np.nan)
            else:
                for key in keys:
                    streamed[key].append(out[key])

        assert stream.warmup_period == 0
        for key in keys:
            np.testing.assert_allclose(
                batch[key],
                streamed[key],
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )

    def test_batch_single_row_matches_single(self):
        open_, high, low, close = sample_ohlc(128)
        batch = mp.hema_trend_levels_batch(open_, high, low, close)
        single = mp.hema_trend_levels(open_, high, low, close, 20, 40)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        assert batch["fast_lengths"][0] == 20
        assert batch["slow_lengths"][0] == 40

        keys = [
            "fast_hema",
            "slow_hema",
            "trend_direction",
            "bar_state",
            "bullish_crossover",
            "bearish_crossunder",
            "box_offset",
            "bull_box_top",
            "bull_box_bottom",
            "bear_box_top",
            "bear_box_bottom",
            "bullish_test",
            "bearish_test",
            "bullish_test_level",
            "bearish_test_level",
        ]
        for key in keys:
            np.testing.assert_allclose(
                batch[key][0], single[key], rtol=1e-12, atol=1e-12, equal_nan=True
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
